"""Warnings that tell you when a comparison cannot be trusted."""

from __future__ import annotations

from typing import Any

MIN_QUERIES = 30
CEILING_MEAN = 0.95
CEILING_TIE_SHARE = 0.8
COPY_THRESHOLD = 0.6
COPY_SHARE_WARN = 0.3
CHARS_PER_TOKEN = 3  # rough lower bound used only to flag likely truncation


def run_checks(
    *,
    n_queries: int,
    mean_a: float,
    mean_b: float,
    tie_share: float,
    ci: tuple[float, float],
    copy_share: float,
    max_chunk_chars: int,
    max_seq_lengths: dict[str, int | None] | None = None,
    metric: str = "ndcg",
) -> list[dict[str, Any]]:
    """Return warnings as ``{"code", "message"}`` dicts (empty list = nothing suspicious)."""
    out: list[dict[str, Any]] = []
    if n_queries < MIN_QUERIES:
        out.append({
            "code": "few_queries",
            "message": f"Only {n_queries} queries (< {MIN_QUERIES}): differences are easy to over-read. "
            "Add more queries, ideally some hard ones.",
        })
    if (mean_a >= CEILING_MEAN and mean_b >= CEILING_MEAN) or tie_share >= CEILING_TIE_SHARE:
        out.append({
            "code": "ceiling",
            "message": f"Score ceiling: both systems are near perfect or {tie_share:.0%} of queries tie on {metric}. "
            "This evaluation set is too easy to tell the models apart.",
        })
    lo, hi = ci
    if lo <= 0 <= hi:
        out.append({
            "code": "not_significant",
            "message": f"The 95% confidence interval [{lo:+.3f}, {hi:+.3f}] of the {metric} difference includes 0: "
            "the difference may be noise.",
        })
    if copy_share >= COPY_SHARE_WARN:
        out.append({
            "code": "copied_queries",
            "message": f"{copy_share:.0%} of queries reuse >= {COPY_THRESHOLD:.0%} of their answer chunk's words. "
            "Lexical overlap favours BM25 and word-level models; mix in real user questions.",
        })
    for spec, limit in (max_seq_lengths or {}).items():
        if limit and max_chunk_chars > limit * CHARS_PER_TOKEN:
            out.append({
                "code": "chunk_truncation",
                "message": f"{spec} reads at most {limit} tokens (~{limit * CHARS_PER_TOKEN} characters) but the "
                f"longest chunk has {max_chunk_chars} characters: the tail is ignored. Use smaller chunks.",
            })
    return out
