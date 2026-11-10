# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
