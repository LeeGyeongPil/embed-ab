"""Paired bootstrap confidence interval."""

from __future__ import annotations

import numpy as np


def paired_bootstrap_ci(
    diff: np.ndarray, n_boot: int = 2000, seed: int = 0, confidence: float = 0.95
) -> tuple[float, float, float]:
    """Return ``(mean, lo, hi)`` of per-query differences (candidate minus baseline).

    Queries are resampled with replacement ``n_boot`` times; ``lo``/``hi`` are the percentile bounds.
    A seeded generator makes the result reproducible.
    """
    diff = np.asarray(diff, dtype=np.float64)
    if diff.size == 0:
        raise ValueError("Need at least one query to compute a confidence interval.")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    batch = 200
    for start in range(0, n_boot, batch):
        size = min(batch, n_boot - start)
        idx = rng.integers(0, diff.size, size=(size, diff.size))
        means[start : start + size] = diff[idx].mean(axis=1)
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(means, [alpha, 1 - alpha])
    return float(diff.mean()), float(lo), float(hi)
