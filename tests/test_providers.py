import numpy as np
import pytest

from embed_ab import providers
from embed_ab.providers import MockEmbedder, ProviderError, make_embedder, parse_spec


def test_parse_spec():
    assert parse_spec("st:sentence-transformers/all-MiniLM-L6-v2") == ("st", "sentence-transformers/all-MiniLM-L6-v2")
    assert parse_spec("ollama:bge-m3") == ("ollama", "bge-m3")
    assert parse_spec("openai:text-embedding-3-small") == ("openai", "text-embedding-3-small")
    assert parse_spec("mock:word") == ("mock", "word")
    for bad in ("bge-m3", "ollama:", "foo:bar", ""):
        with pytest.raises(ProviderError, match="scheme:model|Unknown provider"):
            parse_spec(bad)


@pytest.mark.parametrize("model", ["word", "char3"])
def test_mock_is_deterministic_and_normalised(model):
    texts = ["백업은 설정에서", "plain english text", ""]
    a = MockEmbedder(model).embed_documents(texts)
    b = MockEmbedder(model).embed_documents(texts)
    assert np.array_equal(a, b)
    assert a.dtype == np.float32 and a.shape == (3, 512)
    assert np.linalg.norm(a[0]) == pytest.approx(1.0, abs=1e-5)
    assert not a[2].any()  # empty text -> zero vector, no NaN


def test_mock_word_vs_char3_on_korean_particles():
    q = "동기화를"
    d = "동기화는"
    w = MockEmbedder("word")
    c = MockEmbedder("char3")
    sim_w = float(w.embed_queries([q])[0] @ w.embed_documents([d])[0])
    sim_c = float(c.embed_queries([q])[0] @ c.embed_documents([d])[0])
    assert sim_w == 0.0 and sim_c > 0.0


def test_mock_unknown_model():
    with pytest.raises(ProviderError, match="mock:word or mock:char3"):
        make_embedder("mock:bogus")


class FakeST:
    """Stands in for SentenceTransformer: records which encode method and prompt were used."""

    max_seq_length = 256

    def __init__(self):
        self.calls = []

    def encode_query(self, texts, **kw):
        self.calls.append(("query", kw.get("prompt")))
        return np.array([[3.0, 4.0]] * len(texts))

    def encode_document(self, texts, **kw):
        self.calls.append(("document", kw.get("prompt")))
        return np.array([[0.0, 2.0]] * len(texts))

    def get_sentence_embedding_dimension(self):
        return 2


def test_st_uses_query_and_document_encoders_and_normalises(monkeypatch):
    fake = FakeST()
    monkeypatch.setattr(providers, "_load_st_model", lambda name: fake)
    emb = make_embedder("st:fake/model", query_prompt="query: ", doc_prompt="passage: ")
    q = emb.embed_queries(["a", "b"])
    d = emb.embed_documents(["c"])
    assert fake.calls == [("query", "query: "), ("document", "passage: ")]
    assert np.allclose(np.linalg.norm(q, axis=1), 1.0) and np.allclose(np.linalg.norm(d, axis=1), 1.0)
    assert q[0].tolist() == pytest.approx([0.6, 0.8])
    assert emb.info()["max_seq_length"] == 256 and emb.info()["query_prompt"] == "query: "


def test_st_missing_dependency_message(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name.startswith("sentence_transformers"):
            raise ImportError("no module")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ProviderError, match=r"pip install 'embed-ab\[st\]'"):
        make_embedder("st:whatever")


def test_ollama_request_body(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9999")
    seen = []

    def transport(url, payload, headers):
        seen.append((url, payload, headers))
        return {"embeddings": [[1.0, 0.0] for _ in payload["input"]]}

    emb = make_embedder("ollama:bge-m3", query_prompt="Q: ", transport=transport)
    out = emb.embed_queries(["x", "y"])
    assert seen == [("http://127.0.0.1:9999/api/embed", {"model": "bge-m3", "input": ["Q: x", "Q: y"]}, {})]
    assert out.shape == (2, 2)


def test_ollama_down_gives_fix_hint():
    def transport(url, payload, headers):
        raise ProviderError("Cannot reach http://localhost:11434/api/embed: refused")

    with pytest.raises(ProviderError, match="ollama serve"):
        make_embedder("ollama:bge-m3", transport=transport).embed_documents(["x"])


def test_openai_request_body_and_order(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8080/v1/")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    seen = []

    def transport(url, payload, headers):
        seen.append((url, payload, headers))
        return {"data": [{"index": 1, "embedding": [0.0, 1.0]}, {"index": 0, "embedding": [1.0, 0.0]}]}

    out = make_embedder("openai:m", transport=transport).embed_documents(["a", "b"])
    assert seen[0][0] == "http://127.0.0.1:8080/v1/embeddings"
    assert seen[0][1] == {"model": "m", "input": ["a", "b"]}
    assert seen[0][2] == {"Authorization": "Bearer test-key-not-real"}
    assert out.tolist() == [[1.0, 0.0], [0.0, 1.0]]  # re-ordered by index


def test_openai_without_key_sends_no_auth_header(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    seen = []

    def transport(url, payload, headers):
        seen.append(headers)
        return {"data": [{"index": 0, "embedding": [1.0, 0.0]}]}

    make_embedder("openai:m", transport=transport).embed_documents(["a"])
    assert seen == [{}]


def test_batches_are_split():
    sizes = []

    def transport(url, payload, headers):
        sizes.append(len(payload["input"]))
        return {"embeddings": [[1.0, 0.0]] * len(payload["input"])}

    make_embedder("ollama:m", transport=transport).embed_documents([str(i) for i in range(70)])
    assert sizes == [32, 32, 6]
