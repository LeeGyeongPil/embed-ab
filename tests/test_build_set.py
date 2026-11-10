import json

import pytest

from embed_ab.chunking import chunk_folder, chunk_text
from embed_ab.cli import main
from embed_ab.evalset import EvalSetError, load_evalset
from embed_ab.synth import OllamaQuestionGen, build_evalset, copy_ratio, make_generator


@pytest.fixture
def docs(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "a.md").write_text(
        "Backups run every night and keep seven copies of your notes.\n\n"
        "Restoring a backup overwrites all current notes in the vault.\n",
        encoding="utf-8",
    )
    (folder / "b.txt").write_text("동기화는 계정으로 로그인한 뒤 설정에서 켭니다.\n\n충돌 사본은 원본 옆에 표시됩니다.\n", encoding="utf-8")
    return folder


def test_chunk_ids_and_packing(docs):
    chunks = chunk_folder(docs, chunk_chars=70)
    assert [c.id for c in chunks] == ["a.md#0000", "a.md#0001", "b.txt#0000"]  # b.txt paragraphs fit in one chunk
    assert len(chunk_folder(docs, chunk_chars=600)) == 2  # paragraphs packed together


def test_chunk_overlap_and_validation():
    pieces = chunk_text("one two three four five six seven eight nine ten", chunk_chars=20, overlap=8)
    assert len(pieces) > 1
    with pytest.raises(EvalSetError, match="--overlap"):
        chunk_text("x", chunk_chars=10, overlap=10)


def test_empty_folder_error(tmp_path):
    (tmp_path / "e").mkdir()
    with pytest.raises(EvalSetError, match="No .md"):
        chunk_folder(tmp_path / "e")


def test_build_set_is_byte_identical_and_beir_compatible(docs, tmp_path, capsys):
    out1, out2 = tmp_path / "o1", tmp_path / "o2"
    for out in (out1, out2):
        assert main(["build-set", str(docs), "--chunk-chars", "70", "--out", str(out)]) == 0
    for name in ("corpus.jsonl", "queries.jsonl", "qrels.tsv"):
        assert (out1 / name).read_bytes() == (out2 / name).read_bytes()
    assert (out1 / "qrels.tsv").read_text(encoding="utf-8").splitlines()[0] == "query-id\tcorpus-id\tscore"
    es = load_evalset(out1)
    assert len(es.corpus) == 3 and len(es.queries) == 3
    assert json.loads((out1 / "corpus.jsonl").read_text(encoding="utf-8").splitlines()[0])["_id"] == "a.md#0000"


def test_copy_ratio_warning_and_filter(docs, capsys):
    chunks = chunk_folder(docs, chunk_chars=70)
    result = build_evalset(chunks, make_generator("mock"))
    assert result.warnings and "lexical-overlap bias" in result.warnings[0]
    ratios = [copy_ratio(result.evalset.queries[q], result.evalset.corpus[next(iter(r))]) for q, r in result.evalset.qrels.items()]
    cut = sorted(ratios)[0] + 1e-9  # keep only the least-copied question
    filtered = build_evalset(chunks, make_generator("mock"), max_copy=cut)
    assert 0 < len(filtered.evalset.queries) < len(result.evalset.queries)
    assert filtered.dropped == len(result.evalset.queries) - len(filtered.evalset.queries)


def test_copy_ratio_values():
    assert copy_ratio("a b c d", "a b x y") == 0.5
    assert copy_ratio("", "anything") == 0.0


def test_filter_everything_is_an_error(docs):
    with pytest.raises(EvalSetError, match="No questions were produced"):
        build_evalset(chunk_folder(docs, chunk_chars=70), make_generator("mock"), max_copy=0.0)


def test_ollama_generator_parses_lines_with_fake_transport(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "127.0.0.1:1")
    seen = []

    def transport(url, payload, headers):
        seen.append((url, payload))
        return {"response": "1. How do backups work?\n- Where is the folder?\n\n"}

    gen = OllamaQuestionGen("qwen-test", transport)
    from embed_ab.chunking import Chunk

    assert gen.questions(Chunk("c#0", "text"), 2) == ["How do backups work?", "Where is the folder?"]
    assert seen[0][0] == "http://127.0.0.1:1/api/generate"
    assert seen[0][1]["stream"] is False and "Do NOT copy" in seen[0][1]["prompt"]


def test_unknown_llm_spec_exit_code(docs, tmp_path):
    assert main(["build-set", str(docs), "--llm", "gpt", "--out", str(tmp_path / "o")]) == 2


class _FixedGen:
    """Fake generator: question 0 copies exactly 3 of 5 words (0.6), question 1 copies 1 of 5."""

    name = "fixed"

    def questions(self, chunk, n):
        return ["a b c x y", "a p q r s"][:n]


def test_max_copy_boundary_drops_ratio_equal_to_threshold():
    chunks = [chunk for chunk in [type("C", (), {"id": "d#0", "text": "a b c d e"})()]]
    assert copy_ratio("a b c x y", chunks[0].text) == 0.6
    result = build_evalset(chunks, _FixedGen(), per_chunk=2, max_copy=0.6)
    # the 0.6 question is dropped (>= matches the warning rule); only the 0.2 one stays
    assert list(result.evalset.queries.values()) == ["a p q r s"]
    assert result.dropped == 1
    assert result.warnings == []
