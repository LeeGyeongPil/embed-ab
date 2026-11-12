"""Write ``result.json`` and a single-file static ``report.html`` (no server, no external resources)."""

from __future__ import annotations

import html
import json
from pathlib import Path
from string import Template
from typing import Any

from embed_ab.i18n import get_lang, pad, set_lang, t

TOP_ROWS = 8

CSS = """
body{font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;background:#f6f7f9;color:#1c2330}
main{max-width:1000px;margin:0 auto;padding:24px}
h1{font-size:22px;margin:0 0 4px} h2{font-size:17px;margin:28px 0 8px;border-bottom:1px solid #d8dde5;padding-bottom:4px}
.sub{color:#5b6677;margin:0 0 12px}
.notice{background:#fff4d6;border:1px solid #e8c766;border-radius:6px;padding:10px 12px;margin:12px 0}
.warn{background:#fde8e8;border:1px solid #e4a1a1;border-radius:6px;padding:8px 12px;margin:6px 0}
table{border-collapse:collapse;width:100%;background:#fff;border:1px solid #d8dde5}
th,td{padding:6px 10px;border-bottom:1px solid #e6e9ee;text-align:left;vertical-align:top;font-size:14px}
th{background:#eef1f5} td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
.good{color:#0b7a3b;font-weight:600}.bad{color:#b3261e;font-weight:600}
.hit{background:#dff3e4;border-radius:3px;padding:0 3px}.miss{color:#5b6677}
code{background:#eef1f5;padding:1px 4px;border-radius:3px;font-size:13px}
.k{font-size:13px;color:#5b6677} details{margin-top:10px} pre{background:#fff;border:1px solid #d8dde5;padding:10px;overflow:auto}
"""

PAGE = Template("""<!doctype html>
<html lang="$lang"><head><meta charset="utf-8"><title>$title</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>$css</style></head>
<body><main>
<h1>$heading</h1>
<p class="sub">$sub</p>
$notice
<h2>$h_warnings</h2>
$warnings
<h2>$h_summary</h2>
$summary
<h2>$h_ci</h2>
<p class="k">$note_ci</p>
$ci_svg
<h2>$h_queries</h2>
<p class="k">$note_queries</p>
$queries
<h2>$h_overlap</h2>
<p class="k">$note_overlap</p>
$overlap_svg
<h2>$h_repro</h2>
$repro
<script type="application/json" id="embed-ab-data">$data</script>
</main></body></html>
""")



def _e(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _summary(result: dict[str, Any]) -> str:
    systems = result["systems"]
    metrics = list(next(iter(systems.values()))["metrics"])
    head = "".join(f'<th class="n">{_e(m)}</th>' for m in metrics)
    rows = []
    for name, s in systems.items():
        cells = "".join(f'<td class="n">{_fmt(s["metrics"][m])}</td>' for m in metrics)
        dim = s["info"].get("dim") or "-"
        rows.append(
            f"<tr><td><b>{_e(name)}</b> <code>{_e(s['spec'])}</code></td>{cells}"
            f'<td class="n">{s["index_seconds"]:.2f}s</td><td class="n">{_e(dim)}</td>'
            f'<td class="n">{s["vector_bytes"] / 1024:.0f} KiB</td></tr>'
        )
    return (f'<table><tr><th>{t("html.th_system")}</th>{head}<th class="n">{t("html.th_index")}</th>'
            f'<th class="n">{t("html.th_dim")}</th><th class="n">{t("html.th_vectors")}</th></tr>{"".join(rows)}</table>')


def _ci_svg(result: dict[str, Any]) -> str:
    items = list(result["comparison"]["metrics"].items())
    lim = max(0.05, max(max(abs(v["ci_lo"]), abs(v["ci_hi"])) for _, v in items) * 1.15)
    left, width, row_h = 130, 640, 34
    zero = left + width / 2
    scale = (width / 2) / lim
    parts = [f'<svg viewBox="0 0 {left + width + 20} {row_h * len(items) + 30}" width="100%" role="img" '
             f'aria-label="{_e(t("html.ci_label"))}">',
             f'<line x1="{zero}" y1="4" x2="{zero}" y2="{row_h * len(items) + 6}" stroke="#555" stroke-dasharray="4 3"/>']
    for i, (name, v) in enumerate(items):
        y = 20 + i * row_h
        colour = "#0b7a3b" if v["ci_lo"] > 0 else "#b3261e" if v["ci_hi"] < 0 else "#6b7686"
        x1, x2, xm = zero + v["ci_lo"] * scale, zero + v["ci_hi"] * scale, zero + v["mean_diff"] * scale
        parts.append(
            f'<text x="4" y="{y + 4}" font-size="13">{_e(name)}</text>'
            f'<line x1="{x1:.1f}" y1="{y}" x2="{x2:.1f}" y2="{y}" stroke="{colour}" stroke-width="4"/>'
            f'<circle cx="{xm:.1f}" cy="{y}" r="5" fill="{colour}"/>'
            f'<text x="{left + width + 18}" y="{y + 4}" font-size="12" text-anchor="end" fill="#333">'
            f'{v["mean_diff"]:+.3f} [{v["ci_lo"]:+.3f}, {v["ci_hi"]:+.3f}]</text>'
        )
    parts.append(f'<text x="{zero:.1f}" y="{row_h * len(items) + 24}" font-size="11" text-anchor="middle">0</text></svg>')
    return "".join(parts)


def _top_cell(top: list[str], relevant: set[str]) -> str:
    items = []
    for d in top[:3]:
        cls = "hit" if d in relevant else "miss"
        items.append(f'<div class="{cls}"><code>{_e(d)}</code></div>')
    return "".join(items) or "-"


def _queries(result: dict[str, Any], primary: str) -> str:
    rows = sorted(result["queries"], key=lambda r: r["diff_primary"])
    worst = [r for r in rows if r["diff_primary"] < 0][:TOP_ROWS]
    best = [r for r in reversed(rows) if r["diff_primary"] > 0][:TOP_ROWS]
    if not worst and not best:
        return f'<p>{t("html.all_tie")}</p>'

    def block(title: str, picked: list[dict[str, Any]], cls: str) -> str:
        if not picked:
            return ""
        body = []
        for r in picked:
            rel = set(r["relevant"])
            body.append(
                f'<tr><td>{_e(r["text"])}<div class="k">{_e(r["id"])}</div></td>'
                f'<td class="n {cls}">{r["diff_primary"]:+.3f}</td>'
                f'<td>{_top_cell(r["A"]["top"], rel)}</td><td>{_top_cell(r["B"]["top"], rel)}</td></tr>'
            )
        return (f"<h3>{_e(title)}</h3><table><tr><th>{t('html.th_query')}</th><th class=\"n\">&Delta; {_e(primary)}</th>"
                f"<th>{t('html.th_a_top')}</th><th>{t('html.th_b_top')}</th></tr>{''.join(body)}</table>")

    wins = result["comparison"]["metrics"][primary]
    head = "<p>" + t("html.wins", wins=wins["wins"], losses=wins["losses"], ties=wins["ties"], primary=_e(primary)) + "</p>"
    return head + block(t("html.worst"), worst, "bad") + block(t("html.best"), best, "good")


def _histogram(values: list[float], bins: int = 10) -> str:
    counts = [0] * bins
    for v in values:
        counts[min(int(v * bins), bins - 1)] += 1
    peak = max(counts) or 1
    w, h = 60, 120
    parts = [f'<svg viewBox="0 0 {bins * w + 10} {h + 40}" width="100%" role="img" aria-label="{_e(t("html.hist_label"))}">']
    for i, c in enumerate(counts):
        bh = c / peak * h
        parts.append(
            f'<rect x="{5 + i * w}" y="{h - bh + 5:.1f}" width="{w - 6}" height="{bh:.1f}" fill="#4a78c2"/>'
            f'<text x="{5 + i * w + (w - 6) / 2}" y="{h + 20}" font-size="11" text-anchor="middle">{i / bins:.1f}</text>'
            f'<text x="{5 + i * w + (w - 6) / 2}" y="{h - bh + 1:.1f}" font-size="11" text-anchor="middle">{c or ""}</text>'
        )
    parts.append(f'<text x="{bins * w / 2}" y="{h + 36}" font-size="11" text-anchor="middle">{_e(t("html.hist_axis"))}</text></svg>')
    return "".join(parts)


def _warning_texts(result: dict[str, Any], warning_params: list[dict[str, Any]] | None) -> list[str]:
    """Warning texts in the current language; without ``warning_params`` the stored English text is used."""
    out = []
    for i, w in enumerate(result["warnings"]):
        if warning_params is not None and i < len(warning_params):
            out.append(t(f"warn.{w['code']}", **warning_params[i]))
        else:
            out.append(w["message"])
    return out


def render_html(
    result: dict[str, Any], lang: str | None = None, warning_params: list[dict[str, Any]] | None = None
) -> str:
    """Render the report in ``lang`` (default: current language); every dynamic value is HTML-escaped.

    ``warning_params`` (``CompareResult.warning_params``) lets warnings follow the language; without
    it they appear as stored in ``result``.
    """
    previous = set_lang(lang) if lang else None
    try:
        return _render_html(result, warning_params)
    finally:
        if previous is not None:
            set_lang(previous)


def _render_html(result: dict[str, Any], warning_params: list[dict[str, Any]] | None) -> str:
    p = result["params"]
    primary = p["primary_metric"]
    sysm = result["systems"]
    notice = f'<div class="notice">{_e(t("mock.notice"))}</div>' if result["mock_notice"] else ""
    texts = _warning_texts(result, warning_params)
    warnings = "".join(f'<div class="warn"><b>{_e(w["code"])}</b>: {_e(text)}</div>'
                       for w, text in zip(result["warnings"], texts)) or f'<p>{t("html.no_warnings")}</p>'
    repro_rows = []
    for name, s in sysm.items():
        info = {kk: vv for kk, vv in s["info"].items() if vv not in (None, "")}
        cache = t("html.cache", hits=s["cache_hits"], misses=s["cache_misses"])
        repro_rows.append(f"<tr><td>{_e(name)}</td><td><code>{_e(s['spec'])}</code></td><td>{_e(json.dumps(info, ensure_ascii=False))}</td>"
                          f"<td>{_e(cache)}</td></tr>")
    tool = result["tool"]
    versions = t("html.versions", version=_e(tool["version"]), python=_e(tool["python"]), numpy=_e(tool["numpy"]),
                 st=_e(tool["sentence_transformers"]), seed=p["seed"], bootstrap=p["bootstrap"], bm25=_e(p["bm25"]))
    repro = (f"<table><tr><th>{t('html.th_system')}</th><th>{t('html.th_spec')}</th><th>{t('html.th_info')}</th>"
             f"<th>{t('html.th_cache')}</th></tr>{''.join(repro_rows)}</table>"
             f'<p class="k">{versions}</p>')
    ov = result["overlap"]
    data = json.dumps(result, ensure_ascii=False).replace("</", "<\\/")
    a, b = _e(sysm["A"]["spec"]), _e(sysm["B"]["spec"])
    n_queries, n_docs = result["evalset"]["queries"], result["evalset"]["documents"]
    return PAGE.substitute(
        css=CSS, lang=get_lang(), title=t("html.title", a=a, b=b), heading=t("html.heading", a=a, b=b),
        sub=t("html.sub", n_queries=n_queries, n_docs=n_docs, k=p["k"], primary=_e(primary)),
        notice=notice, h_warnings=t("html.h_warnings"), warnings=warnings,
        h_summary=t("html.h_summary"), summary=_summary(result),
        h_ci=t("html.h_ci"), note_ci=t("html.note_ci"), ci_svg=_ci_svg(result),
        h_queries=t("html.h_queries"), note_queries=t("html.note_queries", primary=_e(primary)),
        queries=_queries(result, primary),
        h_overlap=t("html.h_overlap"),
        note_overlap=t("html.note_overlap", k=p["k"], jac_mean=_fmt(ov["jaccard_mean"]), rbo_p=p["rbo_p"],
                       rbo_mean=_fmt(ov["rbo_mean"])),
        overlap_svg=_histogram(ov["jaccard"]), h_repro=t("html.h_repro"), repro=repro, data=data,
    )


def write_report(
    result: dict[str, Any], html_path: Path | None, json_path: Path | None,
    lang: str | None = None, warning_params: list[dict[str, Any]] | None = None,
) -> None:
    """Write the HTML and/or JSON report, creating parent folders.

    The JSON file is identical in every language; only the HTML follows ``lang``.
    """
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if html_path is not None:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(render_html(result, lang, warning_params), encoding="utf-8")


def render_text(
    result: dict[str, Any], lang: str | None = None, warning_params: list[dict[str, Any]] | None = None
) -> str:
    """Compact terminal summary in ``lang`` (default: current language)."""
    previous = set_lang(lang) if lang else None
    try:
        return _render_text(result, warning_params)
    finally:
        if previous is not None:
            set_lang(previous)


def _render_text(result: dict[str, Any], warning_params: list[dict[str, Any]] | None) -> str:
    k = result["params"]["k"]
    names = [f"recall@{k}", f"ndcg@{k}", f"mrr@{k}", f"hit@{k}"]
    lines = []
    if result["mock_notice"]:
        lines += [t("text.mock_banner"), ""]
    lines.append(t("text.summary", queries=result["evalset"]["queries"], docs=result["evalset"]["documents"], k=k))
    for w, text in zip(result["warnings"], _warning_texts(result, warning_params)):
        lines.append(t("text.warning", code=w["code"], message=text))
    lines.append("")
    lines.append(pad(t("text.system"), 26) + "".join(f"{n:>12}" for n in names))
    for name, s in result["systems"].items():
        lines.append(f"{(name + ' ' + s['spec'])[:25]:<26}" + "".join(f"{s['metrics'][n]:>12.3f}" for n in names))
    lines.append("")
    lines.append(t("text.diff"))
    for n in names:
        d = result["comparison"]["metrics"][n]
        lines.append(f"  {n:<10} {d['mean_diff']:+.3f}  [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]  {d['wins']}/{d['losses']}/{d['ties']}")
    ov = result["overlap"]
    lines.append(t("text.overlap", k=k, jaccard=ov["jaccard_mean"], rbo=ov["rbo_mean"]))
    return "\n".join(lines)
