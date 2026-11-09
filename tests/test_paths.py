import os
from pathlib import Path

from embed_ab import compare, load_evalset
from embed_ab.cli import main
from embed_ab.paths import cache_root


def snapshot(folder: Path):
    return sorted(str(p.relative_to(folder)) for p in folder.rglob("*"))


def test_cache_root_precedence(monkeypatch, tmp_path):
    monkeypatch.delenv("EMBED_AB_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert cache_root() == tmp_path / "xdg" / "embed-ab"
    monkeypatch.setenv("EMBED_AB_CACHE_DIR", str(tmp_path / "explicit"))
    assert cache_root() == tmp_path / "explicit"
    monkeypatch.delenv("EMBED_AB_CACHE_DIR")
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert cache_root() == Path.home() / ".cache" / "embed-ab"


def test_artifacts_only_in_xdg_cache_and_cwd_untouched(monkeypatch, tmp_path, capsys):
    cwd = tmp_path / "work"
    cwd.mkdir()
    (cwd / "keep.txt").write_text("x", encoding="utf-8")
    monkeypatch.chdir(cwd)
    monkeypatch.delenv("EMBED_AB_CACHE_DIR")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    before = snapshot(cwd)
    assert main(["demo"]) == 0
    assert snapshot(cwd) == before
    base = tmp_path / "xdg" / "embed-ab"
    assert list(base.glob("embeddings/*.npz")) and list(base.glob("reports/*/report.html")) and list(base.glob("reports/*/result.json"))
    assert str(base) in capsys.readouterr().out


def test_second_run_hits_the_cache(demo_dir):
    es = load_evalset(demo_dir)
    first = compare(es, "mock:char3", "mock:word").to_dict()
    second = compare(es, "mock:char3", "mock:word").to_dict()
    assert first["systems"]["A"]["cache_hits"] == 0 and first["systems"]["A"]["cache_misses"] > 0
    assert second["systems"]["A"]["cache_hits"] > 0 and second["systems"]["A"]["cache_misses"] == 0
    assert first["comparison"] == second["comparison"]  # cache does not change results


def test_no_cache_flag(demo_dir, isolated_cache):
    compare(load_evalset(demo_dir), "mock:char3", "mock:word", use_cache=False)
    assert not (isolated_cache / "embeddings").exists()


def test_corrupt_cache_is_ignored(demo_dir, isolated_cache):
    es = load_evalset(demo_dir)
    compare(es, "mock:char3", "mock:word")
    for f in (isolated_cache / "embeddings").glob("*.npz"):
        f.write_bytes(b"garbage")
    assert compare(es, "mock:char3", "mock:word").to_dict()["systems"]["A"]["cache_misses"] > 0


def test_tests_run_offline_guard():
    import socket

    import pytest

    with pytest.raises(RuntimeError, match="network access is blocked"):
        socket.create_connection(("127.0.0.1", 9))
    assert os.environ["EMBED_AB_CACHE_DIR"]
