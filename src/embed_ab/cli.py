"""Command line interface: ``embed-ab demo | build-set | compare``.

Exit codes: 0 ok, 1 regression gate failed, 2 input or configuration error.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import time
from pathlib import Path

from embed_ab import __version__
from embed_ab.chunking import chunk_folder
from embed_ab.compare import compare, parse_metric_name
from embed_ab.evalset import EvalSetError, load_evalset, write_evalset
from embed_ab.paths import cache_subdir
from embed_ab.providers import ProviderError
from embed_ab.report import render_text, write_report
from embed_ab.synth import build_evalset, make_generator

DEMO_DIR = Path(__file__).parent / "data" / "demo"
EXIT_OK, EXIT_REGRESSION, EXIT_INPUT = 0, 1, 2


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _fraction(text: str) -> float:
    value = float(text)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return value


def _parse_gate(raw: str, k: int) -> tuple[str, float]:
    metric, sep, delta = raw.partition(":")
    if not sep:
        raise EvalSetError(f"--fail-if-worse expects METRIC@K:DELTA such as ndcg@{k}:0.02, got {raw!r}.")
    try:
        threshold = float(delta)
    except ValueError as exc:
        raise EvalSetError(f"--fail-if-worse threshold {delta!r} is not a number (example: ndcg@{k}:0.02).") from exc
    if threshold < 0:
        raise EvalSetError("--fail-if-worse threshold must be >= 0.")
    return parse_metric_name(metric, k), threshold


def _report_paths(args: argparse.Namespace, label: str) -> tuple[Path, Path]:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    default_dir = cache_subdir("reports") / f"{re.sub(r'[^A-Za-z0-9._-]+', '_', label)[:60]}-{stamp}"
    out = Path(args.out) if getattr(args, "out", None) else default_dir
    html = Path(args.report) if getattr(args, "report", None) else out / "report.html"
    js = Path(args.json) if getattr(args, "json", None) else out / "result.json"
    return html, js


def _run_compare(args: argparse.Namespace, evalset, a: str, b: str, gate: tuple[str, float] | None, bm25: str | None) -> int:
    result = compare(
        evalset.subset_queries(args.max_queries), a, b, k=args.k, bm25=bm25, seed=args.seed,
        n_boot=args.bootstrap, use_cache=not args.no_cache,
        a_prompts=(getattr(args, "a_query_prompt", ""), getattr(args, "a_doc_prompt", "")),
        b_prompts=(getattr(args, "b_query_prompt", ""), getattr(args, "b_doc_prompt", "")),
        log=_log,
    )
    data = result.to_dict()
    html_path, json_path = _report_paths(args, f"{a}__vs__{b}")
    write_report(data, html_path, json_path)
    print(render_text(data))
    print(f"\nreport: {html_path}\njson:   {json_path}")
    if gate:
        metric, threshold = gate
        d = result.diff(metric)
        failed = d["mean_diff"] < -threshold and d["ci_hi"] < 0
        verdict = "FAIL" if failed else "PASS"
        print(f"gate {metric}: B-A = {d['mean_diff']:+.3f} (95% CI [{d['ci_lo']:+.3f}, {d['ci_hi']:+.3f}]), "
              f"fails when worse than -{threshold} with the whole interval below 0 -> {verdict}")
        return EXIT_REGRESSION if failed else EXIT_OK
    return EXIT_OK


def cmd_demo(args: argparse.Namespace) -> int:
    if args.export:
        dest = Path(args.export)
        if dest.exists() and any(dest.iterdir()):
            raise EvalSetError(f"{dest} is not empty; choose a new folder for --export.")
        shutil.copytree(DEMO_DIR, dest, dirs_exist_ok=True)
        print(f"Demo documents and evaluation set copied to {dest} (docs/ holds the source documents).")
        return EXIT_OK
    print("embed-ab demo: practice mock mode (hashed n-gram embedders; not real model quality).")
    evalset = load_evalset(DEMO_DIR)
    return _run_compare(args, evalset, "mock:char3", "mock:word", None, "word")


def cmd_build_set(args: argparse.Namespace) -> int:
    chunks = chunk_folder(args.docs, args.chunk_chars, args.overlap)
    if args.max_chunks:
        chunks = chunks[: args.max_chunks]
    gen = make_generator(args.llm)
    result = build_evalset(chunks, gen, per_chunk=args.per_chunk, max_copy=args.max_copy,
                           max_queries=args.max_queries,
                           progress=lambda i, n: _log(f"  generating questions {i}/{n}") if i % 10 == 0 or i == n else None)
    out = Path(args.out) if args.out else cache_subdir("evalsets") / f"evalset-{time.strftime('%Y%m%d-%H%M%S')}"
    write_evalset(result.evalset, out)
    es = result.evalset
    print(f"Wrote {len(es.corpus)} chunks, {len(es.queries)} queries to {out}"
          + (f" ({result.dropped} dropped by --max-copy)" if result.dropped else ""))
    for w in result.warnings:
        print(f"WARNING {w}")
    if args.llm == "mock":
        print("NOTE: --llm mock copies chunk wording on purpose (practice mode). Use --llm ollama:<model> for real questions.")
    return EXIT_OK


def cmd_compare(args: argparse.Namespace) -> int:
    gate = _parse_gate(args.fail_if_worse, args.k) if args.fail_if_worse else None  # validate before the slow part
    evalset = load_evalset(args.evalset)
    return _run_compare(args, evalset, args.a, args.b, gate, args.bm25)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="embed-ab", description="A/B-test embedding models on your own documents.")
    p.add_argument("--version", action="version", version=f"embed-ab {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("-k", type=int, default=10, help="cut-off for metrics and overlap (default 10)")
        sp.add_argument("--seed", type=int, default=0, help="bootstrap seed (default 0)")
        sp.add_argument("--bootstrap", type=int, default=2000, help="bootstrap resamples (default 2000)")
        sp.add_argument("--max-queries", type=int, default=None, help="use only the first N queries")
        sp.add_argument("--no-cache", action="store_true", help="do not read or write the embedding cache")
        sp.add_argument("--out", help="folder for report.html and result.json (default: cache dir)")

    d = sub.add_parser("demo", help="offline demo with mock embedders and the built-in evaluation set")
    common(d)
    d.add_argument("--export", metavar="DIR", help="copy the demo documents and evaluation set to DIR and exit")
    d.set_defaults(func=cmd_demo)

    b = sub.add_parser("build-set", help="documents folder -> corpus.jsonl, queries.jsonl, qrels.tsv")
    b.add_argument("docs", help="folder with .md/.txt/.jsonl files")
    b.add_argument("--out", help="output folder (default: cache dir)")
    b.add_argument("--llm", default="mock", help="question generator: mock or ollama:<model> (default mock)")
    b.add_argument("--per-chunk", type=int, default=1, help="questions per chunk (default 1)")
    b.add_argument("--chunk-chars", type=int, default=600, help="target chunk size in characters (default 600)")
    b.add_argument("--overlap", type=int, default=0, help="characters carried over between chunks (default 0)")
    b.add_argument("--max-copy", type=_fraction, default=None, help="drop questions copying at least this share of chunk words (0-1)")
    b.add_argument("--max-queries", type=int, default=None, help="stop after N questions")
    b.add_argument("--max-chunks", type=int, default=None, help="use only the first N chunks")
    b.set_defaults(func=cmd_build_set)

    c = sub.add_parser("compare", help="compare model A (baseline) with model B (candidate)")
    c.add_argument("evalset", help="folder with corpus.jsonl, queries.jsonl, qrels.tsv")
    c.add_argument("-a", required=True, help="baseline, e.g. st:sentence-transformers/all-MiniLM-L6-v2")
    c.add_argument("-b", required=True, help="candidate, e.g. ollama:bge-m3 or openai:text-embedding-3-small")
    c.add_argument("--bm25", nargs="?", const="word", default=None, choices=["word", "char2"],
                   help="add a BM25 baseline (tokenizer word or char2; default word)")
    c.add_argument("--report", help="path of report.html (default: cache dir)")
    c.add_argument("--json", help="path of result.json (default: cache dir)")
    c.add_argument("--fail-if-worse", metavar="METRIC@K:DELTA",
                   help="exit 1 if B is worse than A by more than DELTA and the 95%% interval is below 0")
    for side in ("a", "b"):
        c.add_argument(f"--{side}-query-prompt", default="", help=f"text prepended to queries for model {side.upper()}")
        c.add_argument(f"--{side}-doc-prompt", default="", help=f"text prepended to documents for model {side.upper()}")
    common(c)
    c.set_defaults(func=cmd_compare)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (EvalSetError, ProviderError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_INPUT
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
