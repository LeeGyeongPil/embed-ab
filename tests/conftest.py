import socket
from pathlib import Path

import pytest

from embed_ab import cli

DEMO = Path(cli.DEMO_DIR)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Every test writes its cache and reports under tmp_path, never the repo or the real home."""
    monkeypatch.setenv("EMBED_AB_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path / "cache"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Tests must run offline: any outgoing socket connection fails loudly."""

    def blocked(*_args, **_kwargs):
        raise RuntimeError("network access is blocked in tests")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture
def demo_dir() -> Path:
    return DEMO
