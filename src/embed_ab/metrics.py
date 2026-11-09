"""Ranking metrics. Each function returns one value per query (a numpy array)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

Ranked = Sequence[Sequence[str]]
Qrels = Sequence[dict[str, int]]  # one {doc_id: grade} per query, aligned with Ranked

METRIC_NAMES = ("recall", "ndcg", "mrr", "hit")


def _relevant(judged: dict[str, int]) -> set[str]:
    return {d for d, g in judged.items() if g > 0}


def recall_at_k(ranked: Ranked, qrels: Qrels, k: int) -> np.ndarray:
    """Share of relevant documents found in the top k."""
    out = []
    for docs, judged in zip(ranked, qrels, strict=True):
        rel = _relevant(judged)
        out.append(len(rel & set(docs[:k])) / len(rel) if rel else 0.0)
    return np.array(out)


def hit_at_k(ranked: Ranked, qrels: Qrels, k: int) -> np.ndarray:
    """1.0 if any relevant document is in the top k."""
    return np.array(
        [1.0 if _relevant(j) & set(d[:k]) else 0.0 for d, j in zip(ranked, qrels, strict=True)]
    )


def mrr_at_k(ranked: Ranked, qrels: Qrels, k: int) -> np.ndarray:
    """Reciprocal rank of the first relevant document within the top k."""
    out = []
    for docs, judged in zip(ranked, qrels, strict=True):
        rel = _relevant(judged)
        out.append(next((1.0 / (i + 1) for i, d in enumerate(docs[:k]) if d in rel), 0.0))
    return np.array(out)


def ndcg_at_k(ranked: Ranked, qrels: Qrels, k: int) -> np.ndarray:
    """nDCG with linear gain (grade) and log2(rank + 1) discount."""
    out = []
    for docs, judged in zip(ranked, qrels, strict=True):
        dcg = sum(max(judged.get(d, 0), 0) / math.log2(i + 2) for i, d in enumerate(docs[:k]))
        ideal = sorted((g for g in judged.values() if g > 0), reverse=True)[:k]
        idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
        out.append(dcg / idcg if idcg > 0 else 0.0)
    return np.array(out)


_FUNCS = {"recall": recall_at_k, "ndcg": ndcg_at_k, "mrr": mrr_at_k, "hit": hit_at_k}


def all_metrics(ranked: Ranked, qrels: Qrels, k: int) -> dict[str, np.ndarray]:
    """Per-query arrays keyed ``recall@k``, ``ndcg@k``, ``mrr@k``, ``hit@k``."""
    return {f"{name}@{k}": fn(ranked, qrels, k) for name, fn in _FUNCS.items()}
