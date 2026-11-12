# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] - 2026-11-12

### Added
- English and Korean output. `--lang en|ko` (before or after the subcommand) or the `EMBED_AB_LANG` environment variable selects the language of help, progress lines, tables, errors, the text summary and `report.html`; without them `LC_ALL`, `LC_MESSAGES` and `LANG` decide (a value starting with `ko` means Korean, everything else English).
- Errors that wrap OS or JSON problems (missing file, permission denied, invalid UTF-8, JSON syntax) now explain the cause in the selected language. Unreadable evaluation set and document files are reported as input errors (exit code 2) instead of a traceback.
- `<html lang>` of `report.html` follows the selected language.

### Changed
- `result.json` is unchanged and language independent: keys, numbers, and the English warning `code` and `message` strings are identical in every language, and so are the exit codes (0, 1, 2).
- Human-readable text such as the `FAIL` / `PASS` word of the gate line is translated when Korean is selected (for example by `LANG=ko_KR.UTF-8`); scripts should read the exit code instead of grepping the text. Set `EMBED_AB_LANG=en` to keep the English text.

## [0.1.1] - 2026-11-10

### Fixed
- `--max-copy` now drops questions at or above the threshold (`>=`), matching the `copied_queries` warning rule, so `--max-copy 0.6` no longer leaves questions that still trigger the warning.

## [0.1.0] - 2026-11-10

### Added
- `embed-ab demo`: offline demo comparing two mock embedders and a BM25 baseline on a built-in Korean/English evaluation set.
- `embed-ab build-set`: documents folder to BEIR-style `corpus.jsonl` / `queries.jsonl` / `qrels.tsv`, with mock or Ollama question generation and copy-ratio filtering.
- `embed-ab compare`: A/B comparison with Recall@k, nDCG@k, MRR@k, Hit@k, paired bootstrap confidence intervals, per-query wins/losses, Jaccard@k and RBO_ext overlap, warnings, JSON and single-file static HTML reports.
- Providers: `mock:`, `st:` (sentence-transformers, optional extra), `ollama:`, `openai:` (OpenAI-compatible).
- `--fail-if-worse METRIC@K:DELTA` CI gate with exit codes 0 / 1 / 2.
- On-disk embedding cache under `~/.cache/embed-ab` (or `EMBED_AB_CACHE_DIR`).
