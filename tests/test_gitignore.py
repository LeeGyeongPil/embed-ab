import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("git") is None or not (ROOT / ".git").exists(), reason="needs a git checkout")
def test_gitignore_does_not_hide_src_or_tests():
    out = subprocess.run(
        ["git", "ls-files", "--others", "--ignored", "--exclude-standard", "--", "src", "tests"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    hidden = [ln for ln in out.splitlines() if "__pycache__" not in ln and not ln.endswith(".pyc")]
    assert hidden == []
