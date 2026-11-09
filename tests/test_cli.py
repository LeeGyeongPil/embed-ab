import json
import subprocess
import sys
import time

from embed_ab.cli import main

GATE = "ndcg@10:0.02"


def run(demo_dir, a, b, *extra, out):
    return main(["compare", str(demo_dir), "-a", a, "-b", b, "--out", str(out), *extra])


def test_gate_fails_when_b_is_worse(demo_dir, tmp_path, capsys):
    assert run(demo_dir, "mock:char3", "mock:word", "--fail-if-worse", GATE, out=tmp_path / "r") == 1
    assert "FAIL" in capsys.readouterr().out


def test_gate_passes_when_roles_are_swapped(demo_dir, tmp_path):
    assert run(demo_dir, "mock:word", "mock:char3", "--fail-if-worse", GATE, out=tmp_path / "r") == 0


def test_without_gate_exit_zero(demo_dir, tmp_path):
    assert run(demo_dir, "mock:char3", "mock:word", out=tmp_path / "r") == 0
    result = json.loads((tmp_path / "r" / "result.json").read_text(encoding="utf-8"))
    assert result["systems"]["A"]["spec"] == "mock:char3"


def test_input_errors_exit_two(demo_dir, tmp_path, capsys):
    assert run(tmp_path / "missing", "mock:word", "mock:char3", out=tmp_path / "r") == 2
    assert "not found" in capsys.readouterr().err
    assert run(demo_dir, "mock:word", "mock:char3", "--fail-if-worse", "bogus@10:0.1", out=tmp_path / "r") == 2
    assert run(demo_dir, "mock:word", "mock:char3", "--fail-if-worse", "ndcg@5:0.1", out=tmp_path / "r") == 2
    assert run(demo_dir, "mock:word", "mock:char3", "--fail-if-worse", "ndcg@10", out=tmp_path / "r") == 2
    assert run(demo_dir, "nope", "mock:char3", out=tmp_path / "r") == 2
    assert run(demo_dir, "mock:word", "mock:word", out=tmp_path / "r") == 2


def test_demo_prints_summary_and_mock_notice(tmp_path, capsys):
    start = time.time()
    assert main(["demo", "--out", str(tmp_path / "d")]) == 0
    out = capsys.readouterr().out
    assert time.time() - start < 60
    for needle in ("mock:word", "mock:char3", "bm25", "ndcg@10", "PRACTICE MOCK MODE", "report.html"):
        assert needle in out
    assert (tmp_path / "d" / "report.html").is_file()


def test_demo_export(tmp_path):
    dest = tmp_path / "exported"
    assert main(["demo", "--export", str(dest)]) == 0
    assert (dest / "corpus.jsonl").is_file() and (dest / "docs" / "backup.ko.md").is_file()
    assert main(["demo", "--export", str(dest)]) == 2  # refuses to overwrite


def test_few_queries_triggers_warning(demo_dir, tmp_path, capsys):
    assert run(demo_dir, "mock:char3", "mock:word", "--max-queries", "6", out=tmp_path / "r") == 0
    assert "few_queries" in capsys.readouterr().out


def test_python_dash_m_and_version():
    out = subprocess.run([sys.executable, "-m", "embed_ab", "--version"], capture_output=True, text=True, check=True)
    assert out.stdout.startswith("embed-ab ")
