import json
import re

from embed_ab import compare, load_evalset
from embed_ab.report import render_html, write_report


def make_result(demo_dir):
    return compare(load_evalset(demo_dir), "mock:char3", "mock:word", bm25="word", n_boot=300).to_dict()


def test_html_is_a_single_static_file(demo_dir, tmp_path):
    result = make_result(demo_dir)
    html = render_html(result)
    assert not re.search(r"https?://", html)
    assert "<script src" not in html and "<link" not in html and "@import" not in html
    for section in ("Warnings", "Summary", "Difference B", "Per-query wins and losses", "Top-k overlap", "Reproduction"):
        assert section in html
    assert "Practice mock mode" in html and "<svg" in html
    notice = re.search(r'<div class="notice">(.*?)</div>', html, re.S).group(1)
    assert notice.count("Practice mock mode") == 1
    # data is embedded and parses back
    payload = re.search(r'<script type="application/json" id="embed-ab-data">(.*?)</script>', html, re.S).group(1)
    assert json.loads(payload.replace("<\\/", "</"))["params"]["k"] == 10
    path = tmp_path / "x" / "report.html"
    write_report(result, path, tmp_path / "x" / "result.json")
    assert path.read_text(encoding="utf-8") == html
    assert path.resolve().as_uri().startswith("file://")


def test_html_escapes_query_text(demo_dir):
    result = make_result(demo_dir)
    result["queries"][0]["text"] = "<script>alert(1)</script>"
    result["queries"][0]["diff_primary"] = -0.9
    html = render_html(result)
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html


def test_json_contains_repro_info(demo_dir):
    result = make_result(demo_dir)
    assert result["tool"]["numpy"] and result["params"]["seed"] == 0 and result["systems"]["A"]["info"]["dim"] == 512
