"""BEIR-style evaluation sets: ``corpus.jsonl``, ``queries.jsonl``, ``qrels.tsv``."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


class EvalSetError(ValueError):
    """The evaluation set is missing, malformed, or inconsistent."""


@dataclass
class EvalSet:
    """Documents, queries and graded relevance judgements."""

    corpus: dict[str, str]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]] = field(default_factory=dict)

    def subset_queries(self, max_queries: int | None) -> EvalSet:
        """Keep the first ``max_queries`` queries (file order); the corpus is unchanged."""
        if max_queries is None or max_queries >= len(self.queries):
            return self
        if max_queries < 1:
            raise EvalSetError("max_queries must be at least 1.")
        ids = list(self.queries)[:max_queries]
        return EvalSet(
            corpus=self.corpus,
            queries={q: self.queries[q] for q in ids},
            qrels={q: self.qrels[q] for q in ids},
        )


def _read_jsonl(path: Path, what: str) -> list[tuple[int, dict]]:
    if not path.is_file():
        raise EvalSetError(
            f"{what} file not found: {path}. An evaluation set folder needs corpus.jsonl, "
            "queries.jsonl and qrels.tsv; create one with `embed-ab build-set <docs> --out <dir>`."
        )
    rows: list[tuple[int, dict]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvalSetError(f"{path.name} line {lineno}: invalid JSON ({exc.msg}).") from exc
        if not isinstance(obj, dict) or "_id" not in obj or "text" not in obj:
            raise EvalSetError(f'{path.name} line {lineno}: expected an object with "_id" and "text".')
        rows.append((lineno, obj))
    return rows


def _read_id_text(path: Path, what: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for lineno, obj in _read_jsonl(path, what):
        key = str(obj["_id"])
        if key in out:
            raise EvalSetError(f"{path.name} line {lineno}: duplicate _id {key!r}.")
        text = str(obj["text"])
        title = str(obj.get("title") or "").strip()
        out[key] = f"{title}\n{text}" if title else text
    if not out:
        raise EvalSetError(f"{path.name} is empty; add at least one entry.")
    return out


def load_evalset(path: str | Path) -> EvalSet:
    """Load and validate an evaluation set folder."""
    folder = Path(path)
    if not folder.is_dir():
        raise EvalSetError(f"Evaluation set folder not found: {folder}")
    corpus = _read_id_text(folder / "corpus.jsonl", "corpus")
    queries = _read_id_text(folder / "queries.jsonl", "queries")

    qrels_path = folder / "qrels.tsv"
    if not qrels_path.is_file():
        raise EvalSetError(f"qrels file not found: {qrels_path}. Expected tab-separated query-id, corpus-id, score.")
    qrels: dict[str, dict[str, int]] = {}
    for lineno, line in enumerate(qrels_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if lineno == 1 and parts[0].strip() == "query-id":
            continue
        if len(parts) != 3:
            raise EvalSetError(f"qrels.tsv line {lineno}: expected 3 tab-separated columns, got {len(parts)}.")
        qid, did, raw = (p.strip() for p in parts)
        try:
            score = int(raw)
        except ValueError as exc:
            raise EvalSetError(f"qrels.tsv line {lineno}: score {raw!r} is not an integer.") from exc
        if qid not in queries:
            raise EvalSetError(f"qrels.tsv line {lineno}: query id {qid!r} is not in queries.jsonl.")
        if did not in corpus:
            raise EvalSetError(f"qrels.tsv line {lineno}: corpus id {did!r} is not in corpus.jsonl.")
        qrels.setdefault(qid, {})[did] = score

    unjudged = [q for q in queries if not any(s > 0 for s in qrels.get(q, {}).values())]
    if unjudged:
        shown = ", ".join(unjudged[:5])
        raise EvalSetError(
            f"{len(unjudged)} query(ies) have no relevant document in qrels.tsv (e.g. {shown}). "
            "Add a row with score >= 1 for each, or remove those queries from queries.jsonl."
        )
    return EvalSet(corpus=corpus, queries=queries, qrels=qrels)


def write_evalset(evalset: EvalSet, folder: str | Path) -> None:
    """Write the three BEIR files deterministically (same input, same bytes)."""
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)

    def dump(mapping: dict[str, str]) -> str:
        return "".join(
            json.dumps({"_id": k, "text": v}, ensure_ascii=False, sort_keys=True) + "\n" for k, v in mapping.items()
        )

    (out / "corpus.jsonl").write_text(dump(evalset.corpus), encoding="utf-8")
    (out / "queries.jsonl").write_text(dump(evalset.queries), encoding="utf-8")
    lines = ["query-id\tcorpus-id\tscore"]
    for qid in evalset.queries:
        for did, score in sorted(evalset.qrels.get(qid, {}).items()):
            lines.append(f"{qid}\t{did}\t{score}")
    (out / "qrels.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
