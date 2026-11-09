"""Where embed-ab writes generated files.

Everything lands in one cache directory so the working tree and the repo stay clean:
``EMBED_AB_CACHE_DIR`` > ``$XDG_CACHE_HOME/embed-ab`` > ``~/.cache/embed-ab``.
"""

from __future__ import annotations

import os
from pathlib import Path


def cache_root() -> Path:
    """Return the cache root (not created)."""
    explicit = os.environ.get("EMBED_AB_CACHE_DIR")
    if explicit:
        return Path(explicit).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "embed-ab"


def cache_subdir(name: str) -> Path:
    """Return (and create) a sub-directory such as ``embeddings`` or ``reports``."""
    path = cache_root() / name
    path.mkdir(parents=True, exist_ok=True)
    return path
