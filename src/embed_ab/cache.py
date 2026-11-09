"""On-disk embedding cache keyed by (provider, model, prompts, text hash)."""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

MAX_ENTRIES = 100_000  # per model file; oldest entries are dropped beyond this


def _text_key(kind: str, text: str) -> str:
    return kind + ":" + hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


class EmbeddingCache:
    """One ``.npz`` file per (spec, prompts) holding text-hash keys and float32 vectors."""

    def __init__(self, directory: Path, namespace: str):
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", namespace)[:60]
        digest = hashlib.blake2b(namespace.encode("utf-8"), digest_size=5).hexdigest()
        self.path = directory / f"{slug}-{digest}.npz"
        self._vecs: dict[str, np.ndarray] = {}
        if self.path.is_file():
            try:
                with np.load(self.path, allow_pickle=False) as data:
                    for key, vec in zip(data["keys"].tolist(), data["vecs"], strict=True):
                        self._vecs[key] = vec
            except (OSError, ValueError, KeyError):
                self._vecs = {}  # corrupt cache: start over rather than fail the run

    def embed(
        self, kind: str, texts: Sequence[str], fn: Callable[[Sequence[str]], np.ndarray]
    ) -> tuple[np.ndarray, int, int]:
        """Return (matrix, hits, misses); only missing texts are sent to ``fn``."""
        keys = [_text_key(kind, t) for t in texts]
        missing = [i for i, k in enumerate(keys) if k not in self._vecs]
        if missing:
            fresh = fn([texts[i] for i in missing])
            for i, vec in zip(missing, fresh, strict=True):
                self._vecs[keys[i]] = np.asarray(vec, dtype=np.float32)
            self._save()
        if not texts:
            return np.zeros((0, 1), dtype=np.float32), 0, 0
        matrix = np.stack([self._vecs[k] for k in keys])
        return matrix, len(texts) - len(missing), len(missing)

    def _save(self) -> None:
        items = list(self._vecs.items())[-MAX_ENTRIES:]
        self._vecs = dict(items)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.stem + ".tmp.npz")
        np.savez(tmp, keys=np.array([k for k, _ in items]), vecs=np.stack([v for _, v in items]))
        os.replace(tmp, self.path)
