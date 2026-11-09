"""Agreement between two ranked lists, without any relevance judgements."""

from __future__ import annotations

from collections.abc import Sequence


def jaccard_at_k(a: Sequence[str], b: Sequence[str], k: int) -> float:
    """|A ∩ B| / |A ∪ B| over the top k (order ignored)."""
    sa, sb = set(a[:k]), set(b[:k])
    union = sa | sb
    return len(sa & sb) / len(union) if union else 1.0


def rbo_ext(a: Sequence[str], b: Sequence[str], p: float = 0.9) -> float:
    """Extrapolated rank-biased overlap (Webber et al. 2010, eq. 32) for equal-depth lists.

    Top ranks weigh more (smaller ``p`` = more top-heavy). Unlike truncated RBO it is 1.0 for
    identical lists and 0.0 for disjoint lists regardless of depth.
    """
    if not 0 < p < 1:
        raise ValueError("p must be between 0 and 1 (exclusive).")
    depth = min(len(a), len(b))
    if depth == 0:
        return 1.0
    seen_a: set[str] = set()
    seen_b: set[str] = set()
    agree_sum = 0.0
    x_d = 0
    for d in range(1, depth + 1):
        seen_a.add(a[d - 1])
        seen_b.add(b[d - 1])
        x_d = len(seen_a & seen_b)
        agree_sum += (x_d / d) * p**d
    return (x_d / depth) * p**depth + ((1 - p) / p) * agree_sum
