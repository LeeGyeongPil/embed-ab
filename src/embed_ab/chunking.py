"""Turn a folder of documents into chunks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from embed_ab.evalset import EvalSetError

TEXT_SUFFIXES = {".md", ".txt"}


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str


def _split_long(paragraph: str, limit: int) -> list[str]:
    """Split an over-long paragraph on sentence ends, then on hard windows."""
    pieces = [s.strip() for s in re.split(r"(?<=[.!?。])\s+", paragraph) if s.strip()]
    out: list[str] = []
    for piece in pieces:
        while len(piece) > limit:
            out.append(piece[:limit])
            piece = piece[limit:]
        out.append(piece)
    return out


def chunk_text(text: str, chunk_chars: int = 600, overlap: int = 0) -> list[str]:
    """Pack paragraphs greedily into chunks of about ``chunk_chars`` characters."""
    if chunk_chars < 1:
        raise EvalSetError("--chunk-chars must be at least 1.")
    if overlap < 0 or overlap >= chunk_chars:
        raise EvalSetError("--overlap must be >= 0 and smaller than --chunk-chars.")
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if not para:
            continue
        units.extend(_split_long(para, chunk_chars) if len(para) > chunk_chars else [para])
    chunks: list[str] = []
    current = ""
    for unit in units:
        if current and len(current) + 1 + len(unit) > chunk_chars:
            chunks.append(current)
            current = unit
        else:
            current = f"{current} {unit}".strip()
    if current:
        chunks.append(current)
    if overlap:
        for i in range(len(chunks) - 1, 0, -1):
            tail = chunks[i - 1][-overlap:]
            tail = tail[tail.find(" ") + 1 :] if " " in tail else tail
            chunks[i] = f"{tail} {chunks[i]}"
    return chunks


def chunk_folder(folder: str | Path, chunk_chars: int = 600, overlap: int = 0) -> list[Chunk]:
    """Chunk every ``.md``/``.txt``/``.jsonl`` file under ``folder`` (sorted, deterministic).

    Chunk ids are ``relative/path.md#0003``. ``.jsonl`` files hold ``{"id", "text"}`` rows.
    """
    root = Path(folder)
    if not root.is_dir():
        raise EvalSetError(f"Documents folder not found: {root}")
    chunks: list[Chunk] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        rel = path.relative_to(root).as_posix()
        if path.suffix in TEXT_SUFFIXES:
            for i, piece in enumerate(chunk_text(path.read_text(encoding="utf-8"), chunk_chars, overlap)):
                chunks.append(Chunk(f"{rel}#{i:04d}", piece))
        elif path.suffix == ".jsonl":
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    chunks.append(Chunk(str(obj["id"]), str(obj["text"])))
                except (json.JSONDecodeError, KeyError, TypeError) as exc:
                    raise EvalSetError(f'{rel} line {lineno}: expected {{"id", "text"}} JSON.') from exc
    if not chunks:
        raise EvalSetError(f"No .md, .txt or .jsonl documents with text found under {root}.")
    return chunks
