"""The comparison pipeline: index, search, metrics, statistics, overlap, warnings."""

from __future__ import annotations

import platform
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import metadata
from typing import Any

import numpy as np

from embed_ab import metrics as M
from embed_ab.bm25 import BM25, TOKENIZERS
from embed_ab.cache import EmbeddingCache
from embed_ab.checks import COPY_THRESHOLD, run_checks
from embed_ab.evalset import EvalSet, EvalSetError
from embed_ab.overlap import jaccard_at_k, rbo_ext
from embed_ab.paths import cache_subdir
from embed_ab.providers import Embedder, make_embedder
from embed_ab.search import dense_search, top_k
from embed_ab.stats import paired_bootstrap_ci
from embed_ab.synth import copy_ratio

PRIMARY = "ndcg"
RBO_P = 0.9


@dataclass
class CompareResult:
    """Everything a report needs; ``data`` is plain JSON-serialisable data."""

    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.data

    def diff(self, metric: str) -> dict[str, Any]:
        """Candidate-minus-baseline summary for ``metric`` such as ``ndcg@10``."""
        return self.data["comparison"]["metrics"][metric]


def parse_metric_name(name: str, k: int) -> str:
    """Validate ``ndcg@10``-style names against the run's ``k``."""
    m = re.fullmatch(r"(recall|ndcg|mrr|hit)@(\d+)", name.strip().lower())
    if not m:
        raise EvalSetError(
            f"Unknown metric {name!r}. Use one of recall@K, ndcg@K, mrr@K, hit@K (for example ndcg@{k})."
        )
    if int(m.group(2)) != k:
        raise EvalSetError(f"Metric {name!r} does not match -k {k}. Run with -k {m.group(2)} or gate on {m.group(1)}@{k}.")
    return f"{m.group(1)}@{k}"


def _version(pkg: str) -> str | None:
    try:
        return metadata.version(pkg)
    except metadata.PackageNotFoundError:
        return None


def _index_embedder(
    emb: Embedder, evalset: EvalSet, k: int, use_cache: bool, log: Callable[[str], None]
) -> dict[str, Any]:
    doc_ids = list(evalset.corpus)
    q_ids = list(evalset.queries)
    prompts = f"q={emb.info().get('query_prompt', '')}|d={emb.info().get('doc_prompt', '')}"
    cache = EmbeddingCache(cache_subdir("embeddings"), f"{emb.spec}|{prompts}") if use_cache else None

    def run(kind: str, texts: list[str], fn: Callable[..., np.ndarray]) -> tuple[np.ndarray, int, int]:
        if cache is None:
            return fn(texts), 0, len(texts)
        return cache.embed(kind, texts, fn)

    log(f"  indexing {len(doc_ids)} chunks with {emb.spec} ...")
    t0 = time.perf_counter()
    doc_vecs, h1, m1 = run("d", [evalset.corpus[d] for d in doc_ids], emb.embed_documents)
    index_s = time.perf_counter() - t0
    q_vecs, h2, m2 = run("q", [evalset.queries[q] for q in q_ids], emb.embed_queries)
    t1 = time.perf_counter()
    ranked = dense_search(q_vecs, doc_vecs, doc_ids, k)
    search_s = time.perf_counter() - t1
    info = emb.info()
    return {
        "spec": emb.spec, "kind": "embedding", "ranked": ranked, "info": info,
        "index_seconds": round(index_s, 4), "search_seconds": round(search_s, 4),
        "cache_hits": h1 + h2, "cache_misses": m1 + m2,
        "vector_bytes": int(doc_vecs.nbytes),
    }


def compare(
    evalset: EvalSet,
    a: str,
    b: str,
    *,
    k: int = 10,
    bm25: str | None = "word",
    seed: int = 0,
    n_boot: int = 2000,
    use_cache: bool = True,
    a_prompts: tuple[str, str] = ("", ""),
    b_prompts: tuple[str, str] = ("", ""),
    embedders: tuple[Embedder, Embedder] | None = None,
    log: Callable[[str], None] = lambda _msg: None,
) -> CompareResult:
    """Compare embedding model ``a`` (baseline) with ``b`` (candidate) on ``evalset``.

    ``bm25`` is ``"word"``, ``"char2"`` or ``None``. ``embedders`` injects ready-made embedders (tests).
    """
    if k < 1:
        raise EvalSetError("-k must be at least 1.")
    if bm25 is not None and bm25 not in TOKENIZERS:
        raise EvalSetError(f"Unknown BM25 tokenizer {bm25!r}; choose one of {', '.join(TOKENIZERS)}.")
    if a == b and embedders is None:
        raise EvalSetError("-a and -b are the same model; pick two different models to compare.")
    ea, eb = embedders or (
        make_embedder(a, query_prompt=a_prompts[0], doc_prompt=a_prompts[1]),
        make_embedder(b, query_prompt=b_prompts[0], doc_prompt=b_prompts[1]),
    )

    q_ids = list(evalset.queries)
    doc_ids = list(evalset.corpus)
    qrels = [evalset.qrels[q] for q in q_ids]
    systems: dict[str, dict[str, Any]] = {"A": _index_embedder(ea, evalset, k, use_cache, log),
                                          "B": _index_embedder(eb, evalset, k, use_cache, log)}
    if bm25:
        log(f"  indexing BM25 ({bm25}) ...")
        t0 = time.perf_counter()
        index = BM25([evalset.corpus[d] for d in doc_ids], bm25)
        index_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        scores = np.stack([index.scores(evalset.queries[q]) for q in q_ids])
        systems["BM25"] = {
            "spec": f"bm25:{bm25}", "kind": "lexical", "ranked": top_k(scores, doc_ids, k), "info": {},
            "index_seconds": round(index_s, 4), "search_seconds": round(time.perf_counter() - t1, 4),
            "cache_hits": 0, "cache_misses": 0, "vector_bytes": 0,
        }

    per_query: dict[str, dict[str, np.ndarray]] = {
        name: M.all_metrics(s["ranked"], qrels, k) for name, s in systems.items()
    }
    primary = f"{PRIMARY}@{k}"
    comparison: dict[str, Any] = {"baseline": "A", "candidate": "B", "metrics": {}}
    for metric in per_query["A"]:
        diff = per_query["B"][metric] - per_query["A"][metric]
        mean, lo, hi = paired_bootstrap_ci(diff, n_boot=n_boot, seed=seed)
        comparison["metrics"][metric] = {
            "mean_diff": mean, "ci_lo": lo, "ci_hi": hi,
            "wins": int((diff > 1e-12).sum()), "losses": int((diff < -1e-12).sum()),
            "ties": int((np.abs(diff) <= 1e-12).sum()),
        }

    jac = np.array([jaccard_at_k(x, y, k) for x, y in zip(systems["A"]["ranked"], systems["B"]["ranked"])])
    rbo = np.array([rbo_ext(x, y, RBO_P) for x, y in zip(systems["A"]["ranked"], systems["B"]["ranked"])])

    rows = []
    for i, qid in enumerate(q_ids):
        row: dict[str, Any] = {
            "id": qid, "text": evalset.queries[qid],
            "relevant": sorted(d for d, g in qrels[i].items() if g > 0),
            "diff_primary": float(per_query["B"][primary][i] - per_query["A"][primary][i]),
            "jaccard": float(jac[i]), "rbo": float(rbo[i]),
        }
        for name, s in systems.items():
            row[name] = {"top": s["ranked"][i], **{m: float(v[i]) for m, v in per_query[name].items()}}
        rows.append(row)

    ratios = [
        max(copy_ratio(evalset.queries[q], evalset.corpus[d]) for d, g in evalset.qrels[q].items() if g > 0)
        for q in q_ids
    ]
    copy_share = float(np.mean([r >= COPY_THRESHOLD for r in ratios]))
    pd = comparison["metrics"][primary]
    seq = {systems[s]["spec"]: systems[s]["info"].get("max_seq_length") for s in ("A", "B")}
    warnings = run_checks(
        n_queries=len(q_ids),
        mean_a=float(per_query["A"][primary].mean()), mean_b=float(per_query["B"][primary].mean()),
        tie_share=pd["ties"] / len(q_ids), ci=(pd["ci_lo"], pd["ci_hi"]),
        copy_share=copy_share, max_chunk_chars=max(len(t) for t in evalset.corpus.values()),
        max_seq_lengths=seq, metric=primary,
    )

    shown = {d for r in rows for s in systems for d in r[s]["top"][:3]} | {d for r in rows for d in r["relevant"]}
    out_systems = {}
    for name, s in systems.items():
        out_systems[name] = {key: s[key] for key in
                             ("spec", "kind", "info", "index_seconds", "search_seconds", "cache_hits", "cache_misses", "vector_bytes")}
        out_systems[name]["metrics"] = {m: float(v.mean()) for m, v in per_query[name].items()}
    mock = any(systems[s]["info"].get("mock") for s in ("A", "B"))
    result = {
        "tool": {
            "name": "embed-ab", "version": _version("embed-ab"), "python": platform.python_version(),
            "numpy": _version("numpy"), "sentence_transformers": _version("sentence-transformers"),
        },
        "mock_notice": "Practice mock mode: mock embedders are hashed n-gram vectors, not real models. "
        "The numbers illustrate the method and say nothing about real model quality." if mock else None,
        "params": {"k": k, "seed": seed, "bootstrap": n_boot, "primary_metric": primary, "rbo_p": RBO_P,
                   "bm25": bm25, "copy_threshold": COPY_THRESHOLD},
        "evalset": {"documents": len(doc_ids), "queries": len(q_ids),
                    "max_chunk_chars": max(len(t) for t in evalset.corpus.values())},
        "systems": out_systems,
        "comparison": comparison,
        "overlap": {"jaccard_mean": float(jac.mean()), "rbo_mean": float(rbo.mean()),
                    "jaccard": [float(x) for x in jac], "rbo": [float(x) for x in rbo]},
        "queries": rows,
        "docs": {d: evalset.corpus[d][:160] for d in sorted(shown)},
        "warnings": warnings,
    }
    return CompareResult(result)
