"""Command line interface: ``embed-ab demo | build-set | compare``.

Exit codes: 0 ok, 1 regression gate failed, 2 input or configuration error.

Messages are English or Korean: ``--lang en|ko`` > ``EMBED_AB_LANG`` > ``LC_ALL`` > ``LC_MESSAGES`` > ``LANG``.
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
from embed_ab.i18n import LANGS, resolve_lang, set_lang, t
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
        raise argparse.ArgumentTypeError(t("err.fraction"))
    return value


def _parse_gate(raw: str, k: int) -> tuple[str, float]:
    metric, sep, delta = raw.partition(":")
    if not sep:
        raise EvalSetError(t("err.gate_format", k=k, raw=raw))
    try:
        threshold = float(delta)
    except ValueError as exc:
        raise EvalSetError(t("err.gate_number", delta=delta, k=k)) from exc
    if threshold < 0:
        raise EvalSetError(t("err.gate_negative"))
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
    write_report(data, html_path, json_path, warning_params=result.warning_params)
    print(render_text(data, warning_params=result.warning_params))
    print("\n" + t("cli.paths", html=html_path, json=json_path))
    if gate:
        metric, threshold = gate
        d = result.diff(metric)
        failed = d["mean_diff"] < -threshold and d["ci_hi"] < 0
        verdict = t("cli.fail") if failed else t("cli.pass")
        print(t("cli.gate", metric=metric, mean=d["mean_diff"], lo=d["ci_lo"], hi=d["ci_hi"],
                threshold=threshold, verdict=verdict))
        return EXIT_REGRESSION if failed else EXIT_OK
    return EXIT_OK


def cmd_demo(args: argparse.Namespace) -> int:
    if args.export:
        dest = Path(args.export)
        if dest.exists() and any(dest.iterdir()):
            raise EvalSetError(t("err.export_not_empty", dest=dest))
        shutil.copytree(DEMO_DIR, dest, dirs_exist_ok=True)
        print(t("cli.demo_exported", dest=dest))
        return EXIT_OK
    print(t("cli.demo_banner"))
    evalset = load_evalset(DEMO_DIR)
    return _run_compare(args, evalset, "mock:char3", "mock:word", None, "word")


def cmd_build_set(args: argparse.Namespace) -> int:
    chunks = chunk_folder(args.docs, args.chunk_chars, args.overlap)
    if args.max_chunks:
        chunks = chunks[: args.max_chunks]
    gen = make_generator(args.llm)
    result = build_evalset(chunks, gen, per_chunk=args.per_chunk, max_copy=args.max_copy,
                           max_queries=args.max_queries,
                           progress=lambda i, n: _log(t("cli.generating", i=i, n=n)) if i % 10 == 0 or i == n else None)
    out = Path(args.out) if args.out else cache_subdir("evalsets") / f"evalset-{time.strftime('%Y%m%d-%H%M%S')}"
    write_evalset(result.evalset, out)
    es = result.evalset
    print(t("cli.wrote", chunks=len(es.corpus), queries=len(es.queries), out=out)
          + (t("cli.dropped", dropped=result.dropped) if result.dropped else ""))
    for w in result.warnings:
        print(t("cli.build_warning", text=w))
    if args.llm == "mock":
        print(t("cli.mock_note"))
    return EXIT_OK


def cmd_compare(args: argparse.Namespace) -> int:
    gate = _parse_gate(args.fail_if_worse, args.k) if args.fail_if_worse else None  # validate before the slow part
    evalset = load_evalset(args.evalset)
    return _run_compare(args, evalset, args.a, args.b, gate, args.bm25)


# --- argparse with localized chrome ------------------------------------------------------------


class _Formatter(argparse.HelpFormatter):
    """Help formatter whose 'usage:' and section titles follow the selected language."""

    def add_usage(self, usage, actions, groups, prefix=None):  # noqa: ANN001
        super().add_usage(usage, actions, groups, prefix if prefix is not None else t("help.usage"))

    def start_section(self, heading):  # noqa: ANN001
        titles = {"options": "help.options", "optional arguments": "help.options", "positional arguments": "help.positional"}
        super().start_section(t(titles[heading]) if heading in titles else heading)


def _localize_argparse_message(message: str) -> str:
    """Translate argparse's built-in error texts; unknown texts stay as they are."""
    patterns = (
        (r"the following arguments are required: (?P<names>.+)", "help.required"),
        (r"argument (?P<name>.+?): invalid choice: (?P<value>.+?) \(choose from (?P<choices>.+)\)", "help.invalid_choice"),
        (r"argument (?P<name>.+?): expected one argument", "help.expected_one"),
        (r"unrecognized arguments: (?P<args>.+)", "help.unrecognized"),
        (r"argument (?P<name>.+?): invalid (?P<kind>\S+) value: (?P<value>.+)", "help.invalid_value"),
        (r"argument (?P<name>.+?): not allowed with argument (?P<other>.+)", "help.not_allowed"),
    )
    wrapped = {"help.invalid_choice", "help.invalid_value", "help.expected_one", "help.not_allowed"}
    for pattern, key in patterns:
        m = re.fullmatch(pattern, message, re.S)
        if not m:
            continue
        params = m.groupdict()
        if key in wrapped:
            # these texts describe one option: keep the "argument <name>:" lead-in
            name = params.pop("name")
            return t("help.arg_prefix", name=name, message=t(key, name=name, **params))
        return t(key, **params)
    m = re.fullmatch(r"argument (?P<name>.+?): (?P<message>.+)", message, re.S)
    if m:
        return t("help.arg_prefix", name=m["name"], message=m["message"])
    return message


class _Parser(argparse.ArgumentParser):
    """ArgumentParser that localizes -h text and error messages (exit code stays 2)."""

    def __init__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        kwargs.setdefault("formatter_class", _Formatter)
        kwargs["add_help"] = False
        super().__init__(*args, **kwargs)
        self.add_argument("-h", "--help", action="help", default=argparse.SUPPRESS, help=t("help.show_help"))

    def error(self, message: str):  # noqa: ANN201
        self.print_usage(sys.stderr)
        self.exit(EXIT_INPUT, f"{self.prog}: {t('help.error_prefix')}{_localize_argparse_message(message)}\n")


def _h(key: str, **params: str) -> str:
    """Help text for argparse (a literal % must be doubled there)."""
    return t(key, **params).replace("%", "%%")


def build_parser() -> argparse.ArgumentParser:
    """Build the parser in the current language (call ``set_lang`` first)."""
    p = _Parser(prog="embed-ab", description=t("cli.description"))
    p.add_argument("--version", action="version", version=f"embed-ab {__version__}", help=t("help.show_version"))
    p.add_argument("--lang", choices=LANGS, default=None, help=_h("opt.lang"))
    sub = p.add_subparsers(dest="command", required=True)

    def lang_option(sp: argparse.ArgumentParser) -> None:
        # also accepted after the subcommand; SUPPRESS keeps a value given before it
        sp.add_argument("--lang", choices=LANGS, default=argparse.SUPPRESS, help=_h("opt.lang"))

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("-k", type=int, default=10, help=_h("opt.k"))
        sp.add_argument("--seed", type=int, default=0, help=_h("opt.seed"))
        sp.add_argument("--bootstrap", type=int, default=2000, help=_h("opt.bootstrap"))
        sp.add_argument("--max-queries", type=int, default=None, help=_h("opt.max_queries"))
        sp.add_argument("--no-cache", action="store_true", help=_h("opt.no_cache"))
        sp.add_argument("--out", help=_h("opt.out"))
        lang_option(sp)

    d = sub.add_parser("demo", help=_h("cmd.demo"))
    common(d)
    d.add_argument("--export", metavar="DIR", help=_h("opt.export"))
    d.set_defaults(func=cmd_demo)

    b = sub.add_parser("build-set", help=_h("cmd.build_set"))
    b.add_argument("docs", help=_h("opt.docs"))
    b.add_argument("--out", help=_h("opt.build_out"))
    b.add_argument("--llm", default="mock", help=_h("opt.llm"))
    b.add_argument("--per-chunk", type=int, default=1, help=_h("opt.per_chunk"))
    b.add_argument("--chunk-chars", type=int, default=600, help=_h("opt.chunk_chars"))
    b.add_argument("--overlap", type=int, default=0, help=_h("opt.overlap"))
    b.add_argument("--max-copy", type=_fraction, default=None, help=_h("opt.max_copy"))
    b.add_argument("--max-queries", type=int, default=None, help=_h("opt.build_max_queries"))
    b.add_argument("--max-chunks", type=int, default=None, help=_h("opt.max_chunks"))
    lang_option(b)
    b.set_defaults(func=cmd_build_set)

    c = sub.add_parser("compare", help=_h("cmd.compare"))
    c.add_argument("evalset", help=_h("opt.evalset"))
    c.add_argument("-a", required=True, help=_h("opt.a"))
    c.add_argument("-b", required=True, help=_h("opt.b"))
    c.add_argument("--bm25", nargs="?", const="word", default=None, choices=["word", "char2"], help=_h("opt.bm25"))
    c.add_argument("--report", help=_h("opt.report"))
    c.add_argument("--json", help=_h("opt.json"))
    c.add_argument("--fail-if-worse", metavar="METRIC@K:DELTA", help=_h("opt.fail_if_worse"))
    for side in ("a", "b"):
        c.add_argument(f"--{side}-query-prompt", default="", help=_h("opt.query_prompt", side=side.upper()))
        c.add_argument(f"--{side}-doc-prompt", default="", help=_h("opt.doc_prompt", side=side.upper()))
    common(c)
    c.set_defaults(func=cmd_compare)
    return p


def _scan_lang(argv: list[str]) -> str | None:
    """Find ``--lang X`` / ``--lang=X`` before parsing, so help and errors already use that language."""
    found: str | None = None
    for i, token in enumerate(argv):
        if token == "--lang" and i + 1 < len(argv):
            found = argv[i + 1]
        elif token.startswith("--lang="):
            found = token.split("=", 1)[1]
    return found


def main(argv: list[str] | None = None) -> int:
    argv_list = list(sys.argv[1:] if argv is None else argv)
    previous = set_lang(resolve_lang(_scan_lang(argv_list)))
    try:
        args = build_parser().parse_args(argv_list)
        try:
            return args.func(args)
        except (EvalSetError, ProviderError) as exc:
            print(t("cli.error", detail=exc), file=sys.stderr)
            return EXIT_INPUT
        except KeyboardInterrupt:
            print(t("cli.interrupted"), file=sys.stderr)
            return 130
    finally:
        set_lang(previous)
