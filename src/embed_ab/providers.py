"""Embedding providers: ``mock:``, ``st:``, ``ollama:``, ``openai:``.

Every provider turns texts into L2-normalised float32 vectors, with separate entry points for
queries and documents (asymmetric models such as E5 or BGE need different prompts).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import Any, Protocol

import numpy as np

HTTP_TIMEOUT_SECONDS = 600  # shared local servers can be slow; slowness is not an error
MOCK_DIM = 512
HF_HINT = (
    "If the download fails on macOS or behind a proxy, try "
    "SSL_CERT_FILE=/etc/ssl/cert.pem HF_HUB_DISABLE_XET=1; once the model is cached, "
    "HF_HUB_OFFLINE=1 runs offline."
)

Transport = Callable[[str, dict[str, Any], dict[str, str]], dict[str, Any]]


class ProviderError(RuntimeError):
    """An embedding provider could not be created or called."""


class Embedder(Protocol):
    spec: str

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray: ...
    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...
    def info(self) -> dict[str, Any]: ...


def normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows (zero rows stay zero) and return float32."""
    m = np.asarray(matrix, dtype=np.float64)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


def parse_spec(spec: str) -> tuple[str, str]:
    """Split ``scheme:model`` and validate the scheme."""
    scheme, sep, model = spec.partition(":")
    if not sep or not model:
        raise ProviderError(
            f"Invalid model spec {spec!r}. Use scheme:model, e.g. st:sentence-transformers/all-MiniLM-L6-v2, "
            "ollama:bge-m3, openai:text-embedding-3-small, mock:word."
        )
    if scheme not in {"mock", "st", "ollama", "openai"}:
        raise ProviderError(f"Unknown provider {scheme!r} in {spec!r}. Choose one of: mock, st, ollama, openai.")
    return scheme, model


def default_transport(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    """POST JSON with urllib and return the decoded JSON body."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SECONDS) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:300]
        raise ProviderError(f"HTTP {exc.code} from {url}: {body}") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise ProviderError(f"Cannot reach {url}: {exc}") from exc


class _PromptMixin:
    query_prompt: str
    doc_prompt: str

    def _q(self, texts: Sequence[str]) -> list[str]:
        return [self.query_prompt + t for t in texts]

    def _d(self, texts: Sequence[str]) -> list[str]:
        return [self.doc_prompt + t for t in texts]


class MockEmbedder(_PromptMixin):
    """Deterministic hashed bag-of-n-gram vectors (practice mode, not a real model).

    ``mock:word`` hashes whitespace-separated words, so inflected Korean words ("백업을" vs
    "백업은") count as different tokens. ``mock:char3`` hashes character 3-grams and is robust to that.
    """

    def __init__(self, model: str, query_prompt: str = "", doc_prompt: str = ""):
        if model not in {"word", "char3"}:
            raise ProviderError(f"Unknown mock model {model!r}; use mock:word or mock:char3.")
        self.spec = f"mock:{model}"
        self.model = model
        self.query_prompt = query_prompt
        self.doc_prompt = doc_prompt

    def _features(self, text: str) -> list[str]:
        text = text.lower()
        if self.model == "word":
            return re.findall(r"\w+", text)
        compact = " ".join(re.findall(r"\w+", text))
        return [compact[i : i + 3] for i in range(max(len(compact) - 2, 0))]

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), MOCK_DIM), dtype=np.float64)
        for row, text in enumerate(texts):
            for feat in self._features(text):
                h = int.from_bytes(hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest(), "big")
                out[row, h % MOCK_DIM] += 1.0 if (h >> 32) & 1 else -1.0
        return normalize(out)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(self._q(texts))

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(self._d(texts))

    def info(self) -> dict[str, Any]:
        return {"dim": MOCK_DIM, "mock": True, "revision": "n/a", "query_prompt": self.query_prompt, "doc_prompt": self.doc_prompt}


def _load_st_model(name: str) -> Any:
    """Load a sentence-transformers model (module-level so tests can replace it)."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ProviderError(
            "The st: provider needs sentence-transformers. Install it with: pip install 'embed-ab[st]' "
            "(or: uv tool install 'embed-ab[st]')."
        ) from exc
    try:
        return SentenceTransformer(name, device="cpu")
    except Exception as exc:  # noqa: BLE001 - surface any loader failure with a hint
        raise ProviderError(f"Could not load model {name!r}: {exc}. {HF_HINT}") from exc


class SentenceTransformerEmbedder:
    """Local sentence-transformers model using ``encode_query`` / ``encode_document``."""

    def __init__(self, model: str, query_prompt: str = "", doc_prompt: str = "", loaded: Any = None):
        self.spec = f"st:{model}"
        self.model_name = model
        self.query_prompt = query_prompt
        self.doc_prompt = doc_prompt
        self._model = loaded if loaded is not None else _load_st_model(model)

    def _encode(self, method: str, texts: Sequence[str], prompt: str) -> np.ndarray:
        kwargs: dict[str, Any] = {"convert_to_numpy": True, "show_progress_bar": False}
        if prompt:
            kwargs["prompt"] = prompt
        return normalize(getattr(self._model, method)(list(texts), **kwargs))

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode("encode_query", texts, self.query_prompt)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._encode("encode_document", texts, self.doc_prompt)

    def info(self) -> dict[str, Any]:
        revision = "unknown"
        try:
            revision = str(self._model[0].auto_model.config._commit_hash or "unknown")
        except Exception:  # noqa: BLE001 - revision is best-effort metadata
            pass
        return {
            "dim": getattr(self._model, "get_sentence_embedding_dimension", lambda: None)(),
            "revision": revision,
            "max_seq_length": getattr(self._model, "max_seq_length", None),
            "similarity": getattr(self._model, "similarity_fn_name", None),
            "query_prompt": self.query_prompt,
            "doc_prompt": self.doc_prompt,
        }


class _HttpEmbedder(_PromptMixin):
    batch_size = 32

    def __init__(self, model: str, query_prompt: str, doc_prompt: str, transport: Transport | None):
        self.model = model
        self.query_prompt = query_prompt
        self.doc_prompt = doc_prompt
        self.transport = transport or default_transport
        self._dim: int | None = None

    def _batch(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def _embed(self, texts: list[str]) -> np.ndarray:
        rows: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            rows.extend(self._batch(texts[i : i + self.batch_size]))
        if not rows:
            return np.zeros((0, self._dim or 1), dtype=np.float32)
        out = normalize(np.array(rows))
        self._dim = out.shape[1]
        return out

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(self._q(texts))

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed(self._d(texts))

    def info(self) -> dict[str, Any]:
        return {"dim": self._dim, "revision": "n/a", "query_prompt": self.query_prompt, "doc_prompt": self.doc_prompt}


class OllamaEmbedder(_HttpEmbedder):
    """Ollama ``/api/embed``. Host from ``OLLAMA_HOST`` (default http://localhost:11434)."""

    def __init__(self, model: str, query_prompt: str = "", doc_prompt: str = "", transport: Transport | None = None):
        super().__init__(model, query_prompt, doc_prompt, transport)
        self.spec = f"ollama:{model}"

    def _batch(self, texts: list[str]) -> list[list[float]]:
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        if not host.startswith("http"):
            host = "http://" + host
        try:
            body = self.transport(f"{host}/api/embed", {"model": self.model, "input": texts}, {})
        except ProviderError as exc:
            raise ProviderError(
                f"{exc}. Is Ollama running (`ollama serve`) and the model pulled (`ollama pull {self.model}`)?"
            ) from exc
        if "embeddings" not in body:
            raise ProviderError(f"Unexpected Ollama response (no 'embeddings'): {str(body)[:200]}")
        return body["embeddings"]


class OpenAIEmbedder(_HttpEmbedder):
    """OpenAI-compatible ``/v1/embeddings``. ``OPENAI_BASE_URL`` and ``OPENAI_API_KEY`` are optional."""

    def __init__(self, model: str, query_prompt: str = "", doc_prompt: str = "", transport: Transport | None = None):
        super().__init__(model, query_prompt, doc_prompt, transport)
        self.spec = f"openai:{model}"
        self.batch_size = 64

    def _batch(self, texts: list[str]) -> list[list[float]]:
        base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
        key = os.environ.get("OPENAI_API_KEY", "")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        body = self.transport(f"{base}/embeddings", {"model": self.model, "input": texts}, headers)
        data = body.get("data")
        if not isinstance(data, list):
            raise ProviderError(f"Unexpected response from {base}/embeddings (no 'data'): {str(body)[:200]}")
        return [item["embedding"] for item in sorted(data, key=lambda d: d.get("index", 0))]


def make_embedder(
    spec: str,
    *,
    query_prompt: str = "",
    doc_prompt: str = "",
    transport: Transport | None = None,
) -> Embedder:
    """Create an embedder from a ``scheme:model`` spec."""
    scheme, model = parse_spec(spec)
    if scheme == "mock":
        return MockEmbedder(model, query_prompt, doc_prompt)
    if scheme == "st":
        return SentenceTransformerEmbedder(model, query_prompt, doc_prompt)
    if scheme == "ollama":
        return OllamaEmbedder(model, query_prompt, doc_prompt, transport)
    return OpenAIEmbedder(model, query_prompt, doc_prompt, transport)
