"""Synthetic question generation for documents that come without queries."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from embed_ab.chunking import Chunk
from embed_ab.evalset import EvalSet, EvalSetError
from embed_ab.i18n import t
from embed_ab.providers import ProviderError, Transport, default_transport

OLLAMA_PROMPT = (
    "Write {n} short question(s) that the passage below answers. Do NOT copy phrases from the passage: "
    "paraphrase, as a user who has not read it would ask. Use the passage's language. "
    "Output one question per line, nothing else.\n\nPassage:\n{chunk}\n"
)


def _words(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def copy_ratio(query: str, chunk: str) -> float:
    """Share of the query's distinct words that also occur in the chunk (0..1)."""
    q = _words(query)
    return len(q & _words(chunk)) / len(q) if q else 0.0


class MockQuestionGen:
    """Deterministic generator that cuts a chunk's first sentence into a question.

    It intentionally copies the chunk's wording, which reproduces the lexical-overlap bias of
    careless synthetic questions (BM25 looks better than it is).
    """

    name = "mock"

    def questions(self, chunk: Chunk, n: int) -> list[str]:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?。])\s+", chunk.text) if s.strip()]
        out: list[str] = []
        for sentence in sentences[:n] or [chunk.text]:
            words = sentence.rstrip(".!?。 ").split()[:14]
            hangul = re.search(r"[가-힣]", sentence) is not None
            body = " ".join(words)
            out.append(f"{body} - 이 내용을 설명해 주세요" if hangul else f"{body}?")
        return out


class OllamaQuestionGen:
    """Ask a local Ollama model for paraphrased questions (``/api/generate``)."""

    def __init__(self, model: str, transport: Transport | None = None):
        self.name = f"ollama:{model}"
        self.model = model
        self.transport = transport or default_transport

    def questions(self, chunk: Chunk, n: int) -> list[str]:
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
        if not host.startswith("http"):
            host = "http://" + host
        payload = {
            "model": self.model,
            "prompt": OLLAMA_PROMPT.format(n=n, chunk=chunk.text),
            "stream": False,
            "options": {"temperature": 0},
        }
        try:
            body = self.transport(f"{host}/api/generate", payload, {})
        except ProviderError as exc:
            raise ProviderError(t("err.ollama_gen_down", detail=exc, model=self.model)) from exc
        lines = [re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", ln).strip() for ln in str(body.get("response", "")).splitlines()]
        return [ln for ln in lines if ln][:n]


def make_generator(spec: str, transport: Transport | None = None) -> MockQuestionGen | OllamaQuestionGen:
    if spec == "mock":
        return MockQuestionGen()
    if spec.startswith("ollama:") and len(spec) > len("ollama:"):
        return OllamaQuestionGen(spec.split(":", 1)[1], transport)
    raise EvalSetError(t("err.llm_unknown", spec=spec))


@dataclass
class BuildResult:
    evalset: EvalSet
    warnings: list[str]
    dropped: int


def build_evalset(
    chunks: Sequence[Chunk],
    generator: MockQuestionGen | OllamaQuestionGen,
    *,
    per_chunk: int = 1,
    max_copy: float | None = None,
    max_queries: int | None = None,
    warn_threshold: float = 0.6,
    progress: Callable[[int, int], None] | None = None,
) -> BuildResult:
    """Generate one qrel-labelled question set; chunks are the ground truth for their questions."""
    if per_chunk < 1:
        raise EvalSetError(t("err.per_chunk"))
    corpus = {c.id: c.text for c in chunks}
    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    ratios: list[float] = []
    dropped = 0
    for done, chunk in enumerate(chunks, start=1):
        for j, question in enumerate(generator.questions(chunk, per_chunk)):
            ratio = copy_ratio(question, chunk.text)
            if max_copy is not None and ratio >= max_copy:
                dropped += 1
                continue
            qid = f"q{len(queries) + 1:04d}"
            queries[qid] = question
            qrels[qid] = {chunk.id: 1}
            ratios.append(ratio)
        if progress:
            progress(done, len(chunks))
        if max_queries is not None and len(queries) >= max_queries:
            break
    if not queries:
        part = t("err.no_questions_dropped", dropped=dropped, max_copy=max_copy) if dropped else ""
        raise EvalSetError(t("err.no_questions", dropped_part=part))
    warnings: list[str] = []
    high = sum(r >= warn_threshold for r in ratios)
    if high:
        warnings.append(t("warn.build_copy", high=high, total=len(ratios), threshold=warn_threshold))
    return BuildResult(EvalSet(corpus, queries, qrels), warnings, dropped)


__all__ = ["BuildResult", "MockQuestionGen", "OllamaQuestionGen", "build_evalset", "copy_ratio", "make_generator"]
