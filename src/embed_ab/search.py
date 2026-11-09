"""Exhaustive top-k search."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def top_k(scores: np.ndarray, ids: Sequence[str], k: int) -> list[list[str]]:
    """Rank each score row; ties keep corpus order (stable), so results are deterministic."""
    k = min(k, len(ids))
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    return [[ids[j] for j in row] for row in order]


def dense_search(query_vecs: np.ndarray, doc_vecs: np.ndarray, ids: Sequence[str], k: int) -> list[list[str]]:
    """Inner product of normalised vectors (= cosine similarity)."""
    return top_k(query_vecs @ doc_vecs.T, ids, k)
