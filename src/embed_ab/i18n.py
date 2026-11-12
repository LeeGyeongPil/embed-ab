"""English and Korean messages for everything a person reads (CLI, errors, reports).

Machine-readable output (``result.json``) never goes through this module, except that the
warning ``message`` strings stored there are always the English ones.

Language order: ``--lang`` > ``EMBED_AB_LANG`` > ``LC_ALL`` > ``LC_MESSAGES`` > ``LANG``.
A value starting with ``ko`` selects Korean; anything else (or nothing set) selects English.
Library calls use English until :func:`set_lang` is called (the CLI does that in ``main``).
"""

from __future__ import annotations

import json
import os
import socket
import unicodedata
import urllib.error
from collections.abc import Mapping
from typing import Any

LANGS = ("en", "ko")
ENV_VARS = ("EMBED_AB_LANG", "LC_ALL", "LC_MESSAGES", "LANG")

_current = "en"


def resolve_lang(cli_value: str | None = None, environ: Mapping[str, str] | None = None) -> str:
    """Pick ``"en"`` or ``"ko"``; an empty variable counts as unset (POSIX convention)."""
    env = os.environ if environ is None else environ
    candidates = [cli_value, *(env.get(name) for name in ENV_VARS)]
    for value in candidates:
        if value is None or not value.strip():
            continue
        return "ko" if value.strip().lower().startswith("ko") else "en"
    return "en"


def set_lang(lang: str) -> str:
    """Set the language used by :func:`t` and return the previous one."""
    global _current
    if lang not in LANGS:
        raise ValueError(f"Unsupported language {lang!r}; choose one of {', '.join(LANGS)}.")
    previous, _current = _current, lang
    return previous


def get_lang() -> str:
    """Return the language :func:`t` uses when none is passed."""
    return _current


def t(key: str, /, lang: str | None = None, **params: Any) -> str:
    """Render message ``key`` in ``lang`` (default: the current language)."""
    table = MESSAGES.get(lang or _current, MESSAGES["en"])
    template = table.get(key) or MESSAGES["en"][key]
    return template.format(**params) if params else template


def display_width(text: str) -> int:
    """Terminal columns of ``text`` (Hangul and other wide characters take two)."""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def pad(text: str, width: int) -> str:
    """Left-align ``text`` in ``width`` terminal columns."""
    return text + " " * max(width - display_width(text), 0)


# --- details of wrapped Python errors -------------------------------------------------------

_JSON_REASONS = (  # (English prefix produced by the json module, message key)
    ("Expecting value", "json.expecting_value"),
    ("Expecting property name enclosed in double quotes", "json.expecting_name"),
    ("Expecting ':' delimiter", "json.expecting_colon"),
    ("Expecting ',' delimiter", "json.expecting_comma"),
    ("Extra data", "json.extra_data"),
    ("Unterminated string", "json.unterminated"),
    ("Invalid control character", "json.control_char"),
    ("Invalid \\escape", "json.bad_escape"),
    ("Invalid \\uXXXX escape", "json.bad_escape"),
)


def describe_json_error(exc: json.JSONDecodeError) -> str:
    """Explain a JSON syntax error in the current language (English keeps the stdlib wording)."""
    if _current == "en":
        return exc.msg
    for prefix, key in _JSON_REASONS:
        if exc.msg.startswith(prefix):
            return t(key, col=exc.colno)
    return t("json.other", col=exc.colno)


def describe_os_error(exc: BaseException) -> str:
    """Explain an OS or network error in the current language, without its English text."""
    if isinstance(exc, urllib.error.URLError) and not isinstance(exc, urllib.error.HTTPError):
        reason = exc.reason
        if isinstance(reason, BaseException):
            return describe_os_error(reason)
        return t("os.network_text", text=str(reason))
    if isinstance(exc, FileNotFoundError):
        return t("os.not_found")
    if isinstance(exc, PermissionError):
        return t("os.permission")
    if isinstance(exc, IsADirectoryError):
        return t("os.is_dir")
    if isinstance(exc, NotADirectoryError):
        return t("os.not_dir")
    if isinstance(exc, socket.gaierror):
        return t("os.dns")
    if isinstance(exc, (TimeoutError, socket.timeout)):
        return t("os.timeout")
    if isinstance(exc, ConnectionRefusedError):
        return t("os.refused")
    if isinstance(exc, ConnectionError):
        return t("os.connection")
    if isinstance(exc, UnicodeDecodeError):
        return t("os.not_utf8", pos=exc.start)
    errno = getattr(exc, "errno", None)
    return t("os.other", kind=type(exc).__name__, errno=errno if errno is not None else "?")


# --- message tables ---------------------------------------------------------------------------

_EN: dict[str, str] = {
    # argparse chrome
    "help.usage": "usage: ",
    "help.options": "options",
    "help.positional": "positional arguments",
    "help.show_help": "show this help message and exit",
    "help.show_version": "show program's version number and exit",
    "help.error_prefix": "error: ",
    "help.required": "the following arguments are required: {names}",
    "help.invalid_choice": "invalid choice: {value} (choose from {choices})",
    "help.expected_one": "expected one argument",
    "help.unrecognized": "unrecognized arguments: {args}",
    "help.invalid_value": "invalid {kind} value: {value}",
    "help.not_allowed": "not allowed with argument {other}",
    "help.arg_prefix": "argument {name}: {message}",
    # CLI help text
    "cli.description": "A/B-test embedding models on your own documents.",
    "opt.lang": "language of messages and reports: en or ko (default: EMBED_AB_LANG, then LC_ALL, LC_MESSAGES, LANG; otherwise en)",
    "opt.k": "cut-off for metrics and overlap (default 10)",
    "opt.seed": "bootstrap seed (default 0)",
    "opt.bootstrap": "bootstrap resamples (default 2000)",
    "opt.max_queries": "use only the first N queries",
    "opt.no_cache": "do not read or write the embedding cache",
    "opt.out": "folder for report.html and result.json (default: cache dir)",
    "cmd.demo": "offline demo with mock embedders and the built-in evaluation set",
    "opt.export": "copy the demo documents and evaluation set to DIR and exit",
    "cmd.build_set": "documents folder -> corpus.jsonl, queries.jsonl, qrels.tsv",
    "opt.docs": "folder with .md/.txt/.jsonl files",
    "opt.build_out": "output folder (default: cache dir)",
    "opt.llm": "question generator: mock or ollama:<model> (default mock)",
    "opt.per_chunk": "questions per chunk (default 1)",
    "opt.chunk_chars": "target chunk size in characters (default 600)",
    "opt.overlap": "characters carried over between chunks (default 0)",
    "opt.max_copy": "drop questions copying at least this share of chunk words (0-1)",
    "opt.build_max_queries": "stop after N questions",
    "opt.max_chunks": "use only the first N chunks",
    "cmd.compare": "compare model A (baseline) with model B (candidate)",
    "opt.evalset": "folder with corpus.jsonl, queries.jsonl, qrels.tsv",
    "opt.a": "baseline, e.g. st:sentence-transformers/all-MiniLM-L6-v2",
    "opt.b": "candidate, e.g. ollama:bge-m3 or openai:text-embedding-3-small",
    "opt.bm25": "add a BM25 baseline (tokenizer word or char2; default word)",
    "opt.report": "path of report.html (default: cache dir)",
    "opt.json": "path of result.json (default: cache dir)",
    "opt.fail_if_worse": "exit 1 if B is worse than A by more than DELTA and the 95% interval is below 0",
    "opt.query_prompt": "text prepended to queries for model {side}",
    "opt.doc_prompt": "text prepended to documents for model {side}",
    # CLI runtime output
    "cli.interrupted": "interrupted",
    "cli.error": "error: {detail}",
    "cli.demo_banner": "embed-ab demo: practice mock mode (hashed n-gram embedders; not real model quality).",
    "cli.demo_exported": "Demo documents and evaluation set copied to {dest} (docs/ holds the source documents).",
    "cli.paths": "report: {html}\njson:   {json}",
    "cli.gate": "gate {metric}: B-A = {mean:+.3f} (95% CI [{lo:+.3f}, {hi:+.3f}]), "
    "fails when worse than -{threshold} with the whole interval below 0 -> {verdict}",
    "cli.pass": "PASS",
    "cli.fail": "FAIL",
    "cli.generating": "  generating questions {i}/{n}",
    "cli.wrote": "Wrote {chunks} chunks, {queries} queries to {out}",
    "cli.dropped": " ({dropped} dropped by --max-copy)",
    "cli.build_warning": "WARNING {text}",
    "cli.mock_note": "NOTE: --llm mock copies chunk wording on purpose (practice mode). Use --llm ollama:<model> for real questions.",
    "log.indexing": "  indexing {n} chunks with {spec} ...",
    "log.indexing_bm25": "  indexing BM25 ({tokenizer}) ...",
    # errors
    "err.fraction": "must be between 0 and 1",
    "err.gate_format": "--fail-if-worse expects METRIC@K:DELTA such as ndcg@{k}:0.02, got {raw!r}.",
    "err.gate_number": "--fail-if-worse threshold {delta!r} is not a number (example: ndcg@{k}:0.02).",
    "err.gate_negative": "--fail-if-worse threshold must be >= 0.",
    "err.export_not_empty": "{dest} is not empty; choose a new folder for --export.",
    "err.metric_unknown": "Unknown metric {name!r}. Use one of recall@K, ndcg@K, mrr@K, hit@K (for example ndcg@{k}).",
    "err.metric_k": "Metric {name!r} does not match -k {k}. Run with -k {metric_k} or gate on {family}@{k}.",
    "err.k_min": "-k must be at least 1.",
    "err.bm25_unknown": "Unknown BM25 tokenizer {name!r}; choose one of {choices}.",
    "err.same_model": "-a and -b are the same model; pick two different models to compare.",
    "err.max_queries_min": "max_queries must be at least 1.",
    "what.corpus": "corpus",
    "what.queries": "queries",
    "err.file_missing": "{what} file not found: {path}. An evaluation set folder needs corpus.jsonl, "
    "queries.jsonl and qrels.tsv; create one with `embed-ab build-set <docs> --out <dir>`.",
    "err.read_failed": "Cannot read {path}: {detail}.",
    "err.json_invalid": "{name} line {lineno}: invalid JSON ({detail}).",
    "err.row_shape": '{name} line {lineno}: expected an object with "_id" and "text".',
    "err.duplicate_id": "{name} line {lineno}: duplicate _id {key!r}.",
    "err.empty_file": "{name} is empty; add at least one entry.",
    "err.folder_missing": "Evaluation set folder not found: {folder}",
    "err.qrels_missing": "qrels file not found: {path}. Expected tab-separated query-id, corpus-id, score.",
    "err.qrels_columns": "qrels.tsv line {lineno}: expected 3 tab-separated columns, got {count}.",
    "err.qrels_score": "qrels.tsv line {lineno}: score {raw!r} is not an integer.",
    "err.qrels_query": "qrels.tsv line {lineno}: query id {qid!r} is not in queries.jsonl.",
    "err.qrels_corpus": "qrels.tsv line {lineno}: corpus id {did!r} is not in corpus.jsonl.",
    "err.unjudged": "{count} query(ies) have no relevant document in qrels.tsv (e.g. {shown}). "
    "Add a row with score >= 1 for each, or remove those queries from queries.jsonl.",
    "err.spec_invalid": "Invalid model spec {spec!r}. Use scheme:model, e.g. st:sentence-transformers/all-MiniLM-L6-v2, "
    "ollama:bge-m3, openai:text-embedding-3-small, mock:word.",
    "err.provider_unknown": "Unknown provider {scheme!r} in {spec!r}. Choose one of: mock, st, ollama, openai.",
    "err.http": "HTTP {code} from {url}: {body}",
    "err.unreachable": "Cannot reach {url}: {detail}",
    "err.mock_unknown": "Unknown mock model {model!r}; use mock:word or mock:char3.",
    "err.st_missing": "The st: provider needs sentence-transformers. Install it with: pip install 'embed-ab[st]' "
    "(or: uv tool install 'embed-ab[st]').",
    "err.st_load": "Could not load model {name!r}: {detail}. {hint}",
    "hint.hf": "If the download fails on macOS or behind a proxy, try "
    "SSL_CERT_FILE=/etc/ssl/cert.pem HF_HUB_DISABLE_XET=1; once the model is cached, "
    "HF_HUB_OFFLINE=1 runs offline.",
    "err.ollama_down": "{detail}. Is Ollama running (`ollama serve`) and the model pulled (`ollama pull {model}`)?",
    "err.ollama_response": "Unexpected Ollama response (no 'embeddings'): {body}",
    "err.openai_response": "Unexpected response from {base}/embeddings (no 'data'): {body}",
    "err.ollama_gen_down": "{detail}. Is Ollama running and `ollama pull {model}` done?",
    "err.llm_unknown": "Unknown --llm {spec!r}. Use 'mock' or 'ollama:<model>'.",
    "err.per_chunk": "--per-chunk must be at least 1.",
    "err.no_questions": "No questions were produced{dropped_part}. Check the --llm setting or raise --max-copy.",
    "err.no_questions_dropped": " (all {dropped} were at or above --max-copy {max_copy})",
    "err.chunk_chars": "--chunk-chars must be at least 1.",
    "err.overlap": "--overlap must be >= 0 and smaller than --chunk-chars.",
    "err.docs_missing": "Documents folder not found: {root}",
    "err.doc_row": '{rel} line {lineno}: expected {{"id", "text"}} JSON.',
    "err.no_docs": "No .md, .txt or .jsonl documents with text found under {root}.",
    "warn.build_copy": "{high}/{total} questions reuse >= {threshold:.0%} of their chunk's words "
    "(lexical-overlap bias favours BM25). Use --max-copy to filter, or add real questions.",
    # JSON / OS error details (shown in Korean only; English keeps the library's own text)
    "json.expecting_value": "a value is expected (column {col})",
    "json.expecting_name": "a property name in double quotes is expected (column {col})",
    "json.expecting_colon": "a ':' is expected (column {col})",
    "json.expecting_comma": "a ',' is expected (column {col})",
    "json.extra_data": "extra text after the value (column {col})",
    "json.unterminated": "a string is not closed (column {col})",
    "json.control_char": "an unescaped control character (column {col})",
    "json.bad_escape": "an invalid escape sequence (column {col})",
    "json.other": "syntax error (column {col})",
    "os.not_found": "the file or folder does not exist",
    "os.permission": "permission denied",
    "os.is_dir": "it is a folder, not a file",
    "os.not_dir": "a part of the path is not a folder",
    "os.dns": "the host name could not be resolved",
    "os.timeout": "the connection timed out",
    "os.refused": "the connection was refused",
    "os.connection": "the connection failed",
    "os.not_utf8": "the file is not valid UTF-8 text (byte {pos})",
    "os.other": "OS error {errno} ({kind})",
    "os.network_text": "network error: {text}",
    # warnings (also stored in result.json, always in English)
    "warn.few_queries": "Only {n_queries} queries (< {min_queries}): differences are easy to over-read. "
    "Add more queries, ideally some hard ones.",
    "warn.ceiling": "Score ceiling: both systems are near perfect or {tie_share:.0%} of queries tie on {metric}. "
    "This evaluation set is too easy to tell the models apart.",
    "warn.not_significant": "The 95% confidence interval [{lo:+.3f}, {hi:+.3f}] of the {metric} difference includes 0: "
    "the difference may be noise.",
    "warn.copied_queries": "{copy_share:.0%} of queries reuse >= {threshold:.0%} of their answer chunk's words. "
    "Lexical overlap favours BM25 and word-level models; mix in real user questions.",
    "warn.chunk_truncation": "{spec} reads at most {limit} tokens (~{approx} characters) but the "
    "longest chunk has {max_chars} characters: the tail is ignored. Use smaller chunks.",
    "mock.notice": "Practice mock mode: mock embedders are hashed n-gram vectors, not real models. "
    "The numbers illustrate the method and say nothing about real model quality.",
    # terminal report
    "text.mock_banner": "*** PRACTICE MOCK MODE (mock embedders: hashed n-gram vectors, not real models) ***",
    "text.summary": "{queries} queries, {docs} chunks, k={k}",
    "text.warning": "WARNING [{code}] {message}",
    "text.system": "system",
    "text.diff": "B - A (95% bootstrap interval, wins/losses/ties)",
    "text.overlap": "top-{k} overlap A vs B: Jaccard {jaccard:.3f}, RBO_ext {rbo:.3f}",
    # HTML report (keys starting with html. may contain trusted markup)
    "html.title": "embed-ab report: {a} vs {b}",
    "html.heading": "embed-ab: {a} vs {b}",
    "html.sub": "{n_queries} queries, {n_docs} chunks, k = {k}, primary metric {primary}. Static file: works offline.",
    "html.h_warnings": "Warnings",
    "html.h_summary": "Summary",
    "html.h_ci": "Difference B &minus; A (95% bootstrap interval)",
    "html.note_ci": "A bar that crosses the 0 line means the difference may be noise. Positive = B is better.",
    "html.h_queries": "Per-query wins and losses",
    "html.note_queries": 'Top 3 results per model; <span class="hit">highlighted</span> ids are relevant. '
    "Sorted by {primary} change (B &minus; A).",
    "html.h_overlap": "Top-k overlap between A and B",
    "html.note_overlap": "Mean Jaccard@{k} = {jac_mean}, mean RBO_ext (p={rbo_p}) = {rbo_mean}. "
    "Low overlap means re-indexing changes what users see, even when metrics look alike.",
    "html.h_repro": "Reproduction",
    "html.no_warnings": "No warnings.",
    "html.th_system": "System",
    "html.th_index": "index",
    "html.th_dim": "dim",
    "html.th_vectors": "vectors",
    "html.ci_label": "Confidence intervals of metric differences",
    "html.th_query": "Query",
    "html.th_a_top": "A top 3",
    "html.th_b_top": "B top 3",
    "html.all_tie": "Every query ties between A and B.",
    "html.wins": "B wins on <b>{wins}</b> queries, loses on <b>{losses}</b>, ties on <b>{ties}</b> ({primary}).",
    "html.worst": "Biggest regressions (B worse)",
    "html.best": "Biggest improvements (B better)",
    "html.hist_label": "Jaccard overlap histogram",
    "html.hist_axis": "Jaccard@k per query (queries per bin)",
    "html.th_spec": "Spec",
    "html.th_info": "Model info (revision, prompts, dim)",
    "html.th_cache": "Embedding cache",
    "html.cache": "{hits} hits / {misses} misses",
    "html.versions": "embed-ab {version}, Python {python}, numpy {numpy}, sentence-transformers {st}; "
    "seed {seed}, bootstrap {bootstrap}, BM25 tokenizer {bm25}.",
}

_KO: dict[str, str] = {
    "help.usage": "사용법: ",
    "help.options": "옵션",
    "help.positional": "위치 인자",
    "help.show_help": "도움말을 표시하고 종료합니다",
    "help.show_version": "버전을 표시하고 종료합니다",
    "help.error_prefix": "오류: ",
    "help.required": "다음 인자가 필요합니다: {names}",
    "help.invalid_choice": "잘못된 선택: {value} ({choices} 중에서 고르세요)",
    "help.expected_one": "값이 하나 필요합니다",
    "help.unrecognized": "알 수 없는 인자: {args}",
    "help.invalid_value": "잘못된 {kind} 값: {value}",
    "help.not_allowed": "{other} 인자와 함께 쓸 수 없습니다",
    "help.arg_prefix": "인자 {name}: {message}",
    "cli.description": "내 문서로 임베딩 모델을 A/B 테스트합니다.",
    "opt.lang": "메시지와 보고서의 언어: en 또는 ko (기본: EMBED_AB_LANG, 다음 LC_ALL, LC_MESSAGES, LANG, 없으면 en)",
    "opt.k": "지표와 겹침 계산에 쓰는 상위 개수 (기본 10)",
    "opt.seed": "부트스트랩 시드 (기본 0)",
    "opt.bootstrap": "부트스트랩 재표집 횟수 (기본 2000)",
    "opt.max_queries": "앞에서부터 N개 질문만 사용",
    "opt.no_cache": "임베딩 캐시를 읽지도 쓰지도 않음",
    "opt.out": "report.html 과 result.json 을 저장할 폴더 (기본: 캐시 폴더)",
    "cmd.demo": "모의(mock) 임베더와 내장 평가 세트로 하는 오프라인 데모",
    "opt.export": "데모 문서와 평가 세트를 DIR 로 복사하고 종료",
    "cmd.build_set": "문서 폴더 -> corpus.jsonl, queries.jsonl, qrels.tsv",
    "opt.docs": ".md/.txt/.jsonl 파일이 든 폴더",
    "opt.build_out": "출력 폴더 (기본: 캐시 폴더)",
    "opt.llm": "질문 생성기: mock 또는 ollama:<model> (기본 mock)",
    "opt.per_chunk": "청크당 질문 수 (기본 1)",
    "opt.chunk_chars": "청크 목표 크기(글자 수) (기본 600)",
    "opt.overlap": "청크 사이에 이어 붙일 글자 수 (기본 0)",
    "opt.max_copy": "청크 단어를 이 비율 이상 베낀 질문을 버림 (0-1)",
    "opt.build_max_queries": "질문 N개에서 멈춤",
    "opt.max_chunks": "앞에서부터 N개 청크만 사용",
    "cmd.compare": "모델 A(기준)와 모델 B(후보)를 비교",
    "opt.evalset": "corpus.jsonl, queries.jsonl, qrels.tsv 가 든 폴더",
    "opt.a": "기준 모델, 예: st:sentence-transformers/all-MiniLM-L6-v2",
    "opt.b": "후보 모델, 예: ollama:bge-m3 또는 openai:text-embedding-3-small",
    "opt.bm25": "BM25 기준선 추가 (토크나이저 word 또는 char2, 기본 word)",
    "opt.report": "report.html 경로 (기본: 캐시 폴더)",
    "opt.json": "result.json 경로 (기본: 캐시 폴더)",
    "opt.fail_if_worse": "B 가 A 보다 DELTA 넘게 나쁘고 95% 구간이 0 아래면 종료 코드 1",
    "opt.query_prompt": "모델 {side} 의 질문 앞에 붙일 텍스트",
    "opt.doc_prompt": "모델 {side} 의 문서 앞에 붙일 텍스트",
    "cli.interrupted": "중단됨",
    "cli.error": "오류: {detail}",
    "cli.demo_banner": "embed-ab 데모: 연습용 모의(mock) 모드입니다 (해시 n-gram 임베더이며 실제 모델 품질이 아닙니다).",
    "cli.demo_exported": "데모 문서와 평가 세트를 {dest} 에 복사했습니다 (docs/ 에 원문 문서가 있습니다).",
    "cli.paths": "보고서: {html}\nJSON:   {json}",
    "cli.gate": "게이트 {metric}: B-A = {mean:+.3f} (95% 신뢰구간 [{lo:+.3f}, {hi:+.3f}]), "
    "-{threshold} 보다 나쁘고 구간 전체가 0 아래이면 실패 -> {verdict}",
    "cli.pass": "통과",
    "cli.fail": "실패",
    "cli.generating": "  질문 생성 중 {i}/{n}",
    "cli.wrote": "청크 {chunks}개, 질문 {queries}개를 {out} 에 썼습니다",
    "cli.dropped": " (--max-copy 로 {dropped}개 제외)",
    "cli.build_warning": "경고 {text}",
    "cli.mock_note": "참고: --llm mock 은 일부러 청크 문장을 그대로 베낍니다(연습 모드). 실제 질문은 --llm ollama:<model> 을 쓰세요.",
    "log.indexing": "  {spec} 로 청크 {n}개 색인 중 ...",
    "log.indexing_bm25": "  BM25 ({tokenizer}) 색인 중 ...",
    "err.fraction": "0 이상 1 이하여야 합니다",
    "err.gate_format": "--fail-if-worse 는 ndcg@{k}:0.02 같은 METRIC@K:DELTA 형식이어야 합니다 (입력: {raw!r}).",
    "err.gate_number": "--fail-if-worse 임계값 {delta!r} 은(는) 숫자가 아닙니다 (예: ndcg@{k}:0.02).",
    "err.gate_negative": "--fail-if-worse 임계값은 0 이상이어야 합니다.",
    "err.export_not_empty": "{dest} 가 비어 있지 않습니다. --export 에는 새 폴더를 지정하세요.",
    "err.metric_unknown": "알 수 없는 지표 {name!r}. recall@K, ndcg@K, mrr@K, hit@K 중 하나를 쓰세요 (예: ndcg@{k}).",
    "err.metric_k": "지표 {name!r} 이(가) -k {k} 와 맞지 않습니다. -k {metric_k} 로 실행하거나 {family}@{k} 로 게이트를 거세요.",
    "err.k_min": "-k 는 1 이상이어야 합니다.",
    "err.bm25_unknown": "알 수 없는 BM25 토크나이저 {name!r}. {choices} 중에서 고르세요.",
    "err.same_model": "-a 와 -b 가 같은 모델입니다. 서로 다른 두 모델을 지정하세요.",
    "err.max_queries_min": "max_queries 는 1 이상이어야 합니다.",
    "what.corpus": "corpus",
    "what.queries": "queries",
    "err.file_missing": "{what} 파일을 찾을 수 없습니다: {path}. 평가 세트 폴더에는 corpus.jsonl, "
    "queries.jsonl, qrels.tsv 가 필요합니다. `embed-ab build-set <docs> --out <dir>` 로 만들 수 있습니다.",
    "err.read_failed": "{path} 을(를) 읽을 수 없습니다: {detail}.",
    "err.json_invalid": "{name} {lineno}번째 줄: JSON 형식이 올바르지 않습니다 ({detail}).",
    "err.row_shape": '{name} {lineno}번째 줄: "_id" 와 "text" 가 있는 객체여야 합니다.',
    "err.duplicate_id": "{name} {lineno}번째 줄: _id {key!r} 가 중복됩니다.",
    "err.empty_file": "{name} 이(가) 비어 있습니다. 항목을 하나 이상 넣으세요.",
    "err.folder_missing": "평가 세트 폴더를 찾을 수 없습니다: {folder}",
    "err.qrels_missing": "qrels 파일을 찾을 수 없습니다: {path}. 탭으로 구분한 query-id, corpus-id, score 형식이어야 합니다.",
    "err.qrels_columns": "qrels.tsv {lineno}번째 줄: 탭으로 구분한 3개 열이 필요한데 {count}개입니다.",
    "err.qrels_score": "qrels.tsv {lineno}번째 줄: 점수 {raw!r} 이(가) 정수가 아닙니다.",
    "err.qrels_query": "qrels.tsv {lineno}번째 줄: 질문 id {qid!r} 가 queries.jsonl 에 없습니다.",
    "err.qrels_corpus": "qrels.tsv {lineno}번째 줄: 문서 id {did!r} 가 corpus.jsonl 에 없습니다.",
    "err.unjudged": "질문 {count}개에 qrels.tsv 의 정답 문서가 없습니다 (예: {shown}). "
    "각 질문에 점수 1 이상인 행을 추가하거나 queries.jsonl 에서 해당 질문을 지우세요.",
    "err.spec_invalid": "모델 지정 {spec!r} 이(가) 올바르지 않습니다. scheme:model 형식으로 쓰세요. 예: "
    "st:sentence-transformers/all-MiniLM-L6-v2, ollama:bge-m3, openai:text-embedding-3-small, mock:word.",
    "err.provider_unknown": "{spec!r} 의 제공자 {scheme!r} 를 알 수 없습니다. mock, st, ollama, openai 중에서 고르세요.",
    "err.http": "{url} 에서 HTTP {code} 응답: {body}",
    "err.unreachable": "{url} 에 연결할 수 없습니다: {detail}",
    "err.mock_unknown": "알 수 없는 mock 모델 {model!r}. mock:word 또는 mock:char3 를 쓰세요.",
    "err.st_missing": "st: 제공자에는 sentence-transformers 가 필요합니다. 다음으로 설치하세요: pip install 'embed-ab[st]' "
    "(또는: uv tool install 'embed-ab[st]').",
    "err.st_load": "모델 {name!r} 을(를) 불러오지 못했습니다: {detail}. {hint}",
    "hint.hf": "macOS 이거나 프록시 뒤에서 내려받기가 실패하면 "
    "SSL_CERT_FILE=/etc/ssl/cert.pem HF_HUB_DISABLE_XET=1 을 시도해 보세요. 모델이 캐시된 뒤에는 "
    "HF_HUB_OFFLINE=1 로 오프라인 실행할 수 있습니다.",
    "err.ollama_down": "{detail}. Ollama 가 실행 중인지(`ollama serve`), 모델을 받았는지(`ollama pull {model}`) 확인하세요.",
    "err.ollama_response": "예상 밖의 Ollama 응답입니다 ('embeddings' 없음): {body}",
    "err.openai_response": "{base}/embeddings 의 응답이 예상과 다릅니다 ('data' 없음): {body}",
    "err.ollama_gen_down": "{detail}. Ollama 가 실행 중인지, `ollama pull {model}` 을 했는지 확인하세요.",
    "err.llm_unknown": "알 수 없는 --llm {spec!r}. 'mock' 또는 'ollama:<model>' 을 쓰세요.",
    "err.per_chunk": "--per-chunk 는 1 이상이어야 합니다.",
    "err.no_questions": "만들어진 질문이 없습니다{dropped_part}. --llm 설정을 확인하거나 --max-copy 를 올리세요.",
    "err.no_questions_dropped": " (전체 {dropped}개가 --max-copy {max_copy} 이상이었습니다)",
    "err.chunk_chars": "--chunk-chars 는 1 이상이어야 합니다.",
    "err.overlap": "--overlap 은 0 이상이고 --chunk-chars 보다 작아야 합니다.",
    "err.docs_missing": "문서 폴더를 찾을 수 없습니다: {root}",
    "err.doc_row": '{rel} {lineno}번째 줄: {{"id", "text"}} 형태의 JSON 이어야 합니다.',
    "err.no_docs": "{root} 아래에서 텍스트가 든 .md, .txt, .jsonl 문서를 찾지 못했습니다.",
    "warn.build_copy": "질문 {high}/{total}개가 청크 단어의 {threshold:.0%} 이상을 재사용합니다 "
    "(어휘 겹침 편향이 BM25 에 유리). --max-copy 로 거르거나 실제 질문을 추가하세요.",
    "json.expecting_value": "값이 와야 합니다 ({col}열)",
    "json.expecting_name": "큰따옴표로 감싼 속성 이름이 와야 합니다 ({col}열)",
    "json.expecting_colon": "':' 가 와야 합니다 ({col}열)",
    "json.expecting_comma": "',' 가 와야 합니다 ({col}열)",
    "json.extra_data": "값 뒤에 불필요한 내용이 있습니다 ({col}열)",
    "json.unterminated": "문자열이 닫히지 않았습니다 ({col}열)",
    "json.control_char": "이스케이프되지 않은 제어 문자가 있습니다 ({col}열)",
    "json.bad_escape": "잘못된 이스케이프 문자열입니다 ({col}열)",
    "json.other": "구문 오류 ({col}열)",
    "os.not_found": "파일이나 폴더가 없습니다",
    "os.permission": "접근 권한이 없습니다",
    "os.is_dir": "파일이 아니라 폴더입니다",
    "os.not_dir": "경로의 일부가 폴더가 아닙니다",
    "os.dns": "호스트 이름을 찾을 수 없습니다",
    "os.timeout": "연결 시간이 초과되었습니다",
    "os.refused": "연결이 거부되었습니다",
    "os.connection": "연결에 실패했습니다",
    "os.not_utf8": "UTF-8 텍스트가 아닙니다 ({pos}번째 바이트)",
    "os.other": "운영체제 오류 {errno} ({kind})",
    "os.network_text": "네트워크 오류: {text}",
    "warn.few_queries": "질문이 {n_queries}개뿐입니다 (< {min_queries}). 차이를 과대 해석하기 쉽습니다. "
    "질문을 더 추가하세요. 어려운 질문이 섞여 있으면 더 좋습니다.",
    "warn.ceiling": "점수 천장: 두 시스템 모두 거의 만점이거나 질문의 {tie_share:.0%} 가 {metric} 에서 동점입니다. "
    "이 평가 세트는 너무 쉬워서 모델을 구별할 수 없습니다.",
    "warn.not_significant": "{metric} 차이의 95% 신뢰구간 [{lo:+.3f}, {hi:+.3f}] 이 0 을 포함합니다. "
    "차이가 노이즈일 수 있습니다.",
    "warn.copied_queries": "질문의 {copy_share:.0%} 가 정답 청크 단어의 {threshold:.0%} 이상을 재사용합니다. "
    "어휘 겹침은 BM25 와 단어 단위 모델에 유리합니다. 실제 사용자 질문을 섞으세요.",
    "warn.chunk_truncation": "{spec} 는 최대 {limit} 토큰(약 {approx}자)까지만 읽지만 "
    "가장 긴 청크는 {max_chars}자입니다. 뒷부분은 무시됩니다. 청크를 더 작게 나누세요.",
    "mock.notice": "연습용 모의(mock) 모드: mock 임베더는 해시 n-gram 벡터이며 실제 모델이 아닙니다. "
    "숫자는 방법을 보여 줄 뿐 실제 모델 품질을 말해 주지 않습니다.",
    "text.mock_banner": "*** 연습용 모의(MOCK) 모드 (mock 임베더: 해시 n-gram 벡터, 실제 모델 아님) ***",
    "text.summary": "질문 {queries}개, 청크 {docs}개, k={k}",
    "text.warning": "경고 [{code}] {message}",
    "text.system": "시스템",
    "text.diff": "B - A (95% 부트스트랩 구간, 승/패/무)",
    "text.overlap": "상위 {k} 겹침 A 대 B: Jaccard {jaccard:.3f}, RBO_ext {rbo:.3f}",
    "html.title": "embed-ab 보고서: {a} 대 {b}",
    "html.heading": "embed-ab: {a} 대 {b}",
    "html.sub": "질문 {n_queries}개, 청크 {n_docs}개, k = {k}, 주 지표 {primary}. 정적 파일이라 오프라인에서 열립니다.",
    "html.h_warnings": "경고",
    "html.h_summary": "요약",
    "html.h_ci": "B &minus; A 차이 (95% 부트스트랩 구간)",
    "html.note_ci": "막대가 0 선을 가로지르면 차이가 노이즈일 수 있습니다. 양수 = B 가 더 좋음.",
    "html.h_queries": "질문별 승패",
    "html.note_queries": '모델별 상위 3개 결과이며 <span class="hit">강조</span>된 id 가 정답입니다. '
    "{primary} 변화(B &minus; A) 순으로 정렬했습니다.",
    "html.h_overlap": "A 와 B 의 상위 k 겹침",
    "html.note_overlap": "평균 Jaccard@{k} = {jac_mean}, 평균 RBO_ext (p={rbo_p}) = {rbo_mean}. "
    "지표가 비슷해도 겹침이 낮으면 재색인 후 사용자가 보는 결과가 달라집니다.",
    "html.h_repro": "재현 정보",
    "html.no_warnings": "경고 없음.",
    "html.th_system": "시스템",
    "html.th_index": "색인",
    "html.th_dim": "차원",
    "html.th_vectors": "벡터",
    "html.ci_label": "지표 차이의 신뢰구간",
    "html.th_query": "질문",
    "html.th_a_top": "A 상위 3",
    "html.th_b_top": "B 상위 3",
    "html.all_tie": "모든 질문이 A 와 B 에서 동점입니다.",
    "html.wins": "B 가 <b>{wins}</b>개 질문에서 이기고 <b>{losses}</b>개에서 지고 <b>{ties}</b>개에서 비깁니다 ({primary}).",
    "html.worst": "가장 나빠진 질문 (B 가 더 나쁨)",
    "html.best": "가장 좋아진 질문 (B 가 더 좋음)",
    "html.hist_label": "Jaccard 겹침 히스토그램",
    "html.hist_axis": "질문별 Jaccard@k (구간별 질문 수)",
    "html.th_spec": "지정",
    "html.th_info": "모델 정보 (리비전, 프롬프트, 차원)",
    "html.th_cache": "임베딩 캐시",
    "html.cache": "적중 {hits} / 미적중 {misses}",
    "html.versions": "embed-ab {version}, Python {python}, numpy {numpy}, sentence-transformers {st}; "
    "시드 {seed}, 부트스트랩 {bootstrap}, BM25 토크나이저 {bm25}.",
}

MESSAGES: dict[str, dict[str, str]] = {"en": _EN, "ko": _KO}

__all__ = [
    "LANGS", "MESSAGES", "describe_json_error", "describe_os_error", "display_width", "get_lang",
    "pad", "resolve_lang", "set_lang", "t",
]
