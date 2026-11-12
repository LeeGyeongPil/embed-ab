"""Warnings that tell you when a comparison cannot be trusted."""

from __future__ import annotations

from typing import Any

from embed_ab.i18n import t

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
    """Return warnings as ``{"code", "message", "params"}`` dicts (empty list = nothing suspicious).

    ``message`` is always English (it is stored in ``result.json``); ``params`` holds the numbers
    so the text report and the HTML report can re-render the same warning in another language
    with ``t("warn.<code>", **params)``. Callers drop ``params`` before writing JSON.
    """
    out: list[dict[str, Any]] = []

    def add(code: str, **params: Any) -> None:
        out.append({"code": code, "message": t(f"warn.{code}", "en", **params), "params": params})

    if n_queries < MIN_QUERIES:
        add("few_queries", n_queries=n_queries, min_queries=MIN_QUERIES)
    if (mean_a >= CEILING_MEAN and mean_b >= CEILING_MEAN) or tie_share >= CEILING_TIE_SHARE:
        add("ceiling", tie_share=tie_share, metric=metric)
    lo, hi = ci
    if lo <= 0 <= hi:
        add("not_significant", lo=lo, hi=hi, metric=metric)
    if copy_share >= COPY_SHARE_WARN:
        add("copied_queries", copy_share=copy_share, threshold=COPY_THRESHOLD)
    for spec, limit in (max_seq_lengths or {}).items():
        if limit and max_chunk_chars > limit * CHARS_PER_TOKEN:
            add("chunk_truncation", spec=spec, limit=limit, approx=limit * CHARS_PER_TOKEN, max_chars=max_chunk_chars)
    return out


def localize_warning(warning: dict[str, Any], lang: str | None = None) -> str:
    """Render a warning from :func:`run_checks` in ``lang`` (default: the current language)."""
    return t(f"warn.{warning['code']}", lang, **warning["params"])
