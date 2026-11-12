"""English / Korean output: message tables, language priority, localized errors and reports."""

import json
import re
import string

import pytest

from embed_ab import i18n
from embed_ab.cli import main
from embed_ab.i18n import MESSAGES, describe_json_error, resolve_lang, t

VOLATILE = {"index_seconds", "search_seconds", "python", "numpy", "sentence_transformers"}


def fields(template: str) -> set[str]:
    return {name for _, name, _, _ in string.Formatter().parse(template) if name}


def strip_volatile(obj):
    if isinstance(obj, dict):
        return {k: strip_volatile(v) for k, v in obj.items() if k not in VOLATILE}
    if isinstance(obj, list):
        return [strip_volatile(v) for v in obj]
    return obj


def run_demo(tmp_path, name, *args):
    out = tmp_path / name
    assert main([*args, "demo", "--no-cache", "--out", str(out)]) == 0
    return out


# --- (d) message tables ---------------------------------------------------------------------


def test_english_and_korean_have_the_same_keys():
    assert set(MESSAGES["en"]) == set(MESSAGES["ko"])
    assert set(MESSAGES) == set(i18n.LANGS) == {"en", "ko"}


def test_placeholders_match_between_languages():
    for key, english in MESSAGES["en"].items():
        assert fields(english) == fields(MESSAGES["ko"][key]), key


def test_no_message_is_empty_and_korean_is_really_korean():
    for lang, table in MESSAGES.items():
        assert all(text.strip() for text in table.values()), lang
    assert re.search("[가-힣]", t("cli.pass", "ko")) and not re.search("[가-힣]", t("cli.pass", "en"))


def test_every_message_renders_with_its_own_placeholders():
    for lang, table in MESSAGES.items():
        for key, template in table.items():
            t(key, lang, **{name: 1 for name in fields(template)})  # must not raise


# --- language priority ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("cli_value", "env", "expected"),
    [
        (None, {}, "en"),
        ("ko", {"EMBED_AB_LANG": "en"}, "ko"),
        ("en", {"EMBED_AB_LANG": "ko"}, "en"),
        (None, {"EMBED_AB_LANG": "ko", "LC_ALL": "en_US.UTF-8"}, "ko"),
        (None, {"EMBED_AB_LANG": "en", "LC_ALL": "ko_KR.UTF-8"}, "en"),
        (None, {"LC_ALL": "en_US.UTF-8", "LC_MESSAGES": "ko_KR.UTF-8", "LANG": "ko_KR.UTF-8"}, "en"),
        (None, {"LC_MESSAGES": "ko_KR.UTF-8", "LANG": "en_US.UTF-8"}, "ko"),
        (None, {"LANG": "ko_KR.UTF-8"}, "ko"),
        (None, {"LANG": "ko"}, "ko"),
        (None, {"LANG": "KO_kr"}, "ko"),
        (None, {"LANG": "fr_FR.UTF-8"}, "en"),
        (None, {"LANG": "C"}, "en"),
        (None, {"LANG": ""}, "en"),
        (None, {"EMBED_AB_LANG": "klingon", "LANG": "ko_KR.UTF-8"}, "en"),
    ],
)
def test_language_priority(cli_value, env, expected):
    assert resolve_lang(cli_value, env) == expected


def test_empty_variable_counts_as_unset():
    assert resolve_lang(None, {"EMBED_AB_LANG": "", "LC_ALL": "", "LANG": "ko_KR.UTF-8"}) == "ko"


def test_set_lang_rejects_unknown_language():
    with pytest.raises(ValueError, match="Unsupported language"):
        i18n.set_lang("fr")


def test_environment_decides_when_no_flag_is_given(monkeypatch, capsys):
    monkeypatch.setenv("EMBED_AB_LANG", "ko")
    assert main(["compare", "/no/such/folder", "-a", "mock:word", "-b", "mock:char3"]) == 2
    assert "오류:" in capsys.readouterr().err
    monkeypatch.delenv("EMBED_AB_LANG")
    monkeypatch.setenv("LC_ALL", "C")
    monkeypatch.setenv("LANG", "ko_KR.UTF-8")  # LC_ALL wins and is not Korean
    assert main(["compare", "/no/such/folder", "-a", "mock:word", "-b", "mock:char3"]) == 2
    assert "error: Evaluation set folder not found" in capsys.readouterr().err


def test_flag_beats_environment_and_language_is_restored(monkeypatch, capsys):
    monkeypatch.setenv("EMBED_AB_LANG", "ko")
    assert main(["--lang", "en", "compare", "/no/such/folder", "-a", "mock:word", "-b", "mock:char3"]) == 2
    assert capsys.readouterr().err.startswith("error: ")
    assert i18n.get_lang() == "en"  # main() restores the previous language after a run
    assert main(["--lang", "ko", "compare", "/no/such/folder", "-a", "mock:word", "-b", "mock:char3"]) == 2
    assert i18n.get_lang() == "en"


def test_lang_flag_works_before_and_after_the_subcommand(tmp_path, capsys):
    main(["--lang", "ko", "demo", "--no-cache", "--out", str(tmp_path / "a")])
    first = capsys.readouterr().out
    main(["demo", "--no-cache", "--out", str(tmp_path / "b"), "--lang", "ko"])
    second = capsys.readouterr().out
    assert "연습용 모의(mock) 모드" in first and "연습용 모의(mock) 모드" in second


# --- (b) demo and reports -----------------------------------------------------------------------


def test_default_output_is_english(tmp_path, capsys):
    run_demo(tmp_path, "en")
    out = capsys.readouterr().out
    assert "PRACTICE MOCK MODE" in out and "report: " in out and "queries," in out
    assert not re.search("[가-힣]", out)


def test_korean_demo_prints_korean_text_and_writes_korean_html(tmp_path, capsys):
    out_dir = run_demo(tmp_path, "ko", "--lang", "ko")
    captured = capsys.readouterr()
    assert "연습용 모의(MOCK) 모드" in captured.out and "보고서: " in captured.out and "시스템" in captured.out
    assert "색인 중" in captured.err
    html = (out_dir / "report.html").read_text(encoding="utf-8")
    assert '<html lang="ko">' in html
    for heading in ("<h2>경고</h2>", "<h2>요약</h2>", "<h2>질문별 승패</h2>", "<h2>재현 정보</h2>"):
        assert heading in html
    assert "ndcg@10" in html  # metric names are not translated


def test_html_lang_attribute_follows_language(tmp_path):
    en = run_demo(tmp_path, "en")
    assert '<html lang="en">' in (en / "report.html").read_text(encoding="utf-8")
    assert "<h2>Warnings</h2>" in (en / "report.html").read_text(encoding="utf-8")


def test_text_rendering_pads_wide_characters():
    assert i18n.display_width("시스템") == 6
    assert i18n.display_width(i18n.pad("시스템", 10)) == 10


# --- (c) machine output stays the same -----------------------------------------------------------


def test_result_json_is_identical_in_both_languages(tmp_path):
    en = json.loads((run_demo(tmp_path, "en", "--lang", "en") / "result.json").read_text(encoding="utf-8"))
    ko = json.loads((run_demo(tmp_path, "ko", "--lang", "ko") / "result.json").read_text(encoding="utf-8"))
    assert strip_volatile(en) == strip_volatile(ko)
    assert en["mock_notice"].startswith("Practice mock mode")


def test_warnings_in_json_stay_english_but_text_follows_language(demo_dir, tmp_path, capsys):
    results = {}
    for lang in ("en", "ko"):
        out = tmp_path / lang
        args = ["--lang", lang, "compare", str(demo_dir), "-a", "mock:char3", "-b", "mock:word",
                "--max-queries", "6", "--no-cache", "--out", str(out)]
        assert main(args) == 0
        results[lang] = (json.loads((out / "result.json").read_text(encoding="utf-8")), capsys.readouterr().out,
                         (out / "report.html").read_text(encoding="utf-8"))
    en_json, en_out, en_html = results["en"]
    ko_json, ko_out, ko_html = results["ko"]
    assert en_json["warnings"] == ko_json["warnings"]
    assert {w["code"] for w in ko_json["warnings"]} >= {"few_queries"}
    assert all(set(w) == {"code", "message"} for w in ko_json["warnings"])
    assert "Only 6 queries" in en_out and "Only 6 queries" in en_html
    assert "질문이 6개뿐입니다" in ko_out and "질문이 6개뿐입니다" in ko_html
    assert "[few_queries]" in ko_out  # the machine-readable code is not translated
    assert strip_volatile(en_json) == strip_volatile(ko_json)


def test_gate_line_and_exit_code_do_not_depend_on_language(demo_dir, tmp_path, capsys):
    for lang, word in (("en", "FAIL"), ("ko", "실패")):
        args = ["--lang", lang, "compare", str(demo_dir), "-a", "mock:char3", "-b", "mock:word",
                "--fail-if-worse", "ndcg@10:0.02", "--out", str(tmp_path / lang)]
        assert main(args) == 1
        assert word in capsys.readouterr().out


# --- localized errors (exit code stays 2) -----------------------------------------------------------


def make_set(folder, queries='{"_id": "q1", "text": "질문"}\n', corpus='{"_id": "d1", "text": "문서"}\n'):
    folder.mkdir()
    (folder / "corpus.jsonl").write_bytes(corpus if isinstance(corpus, bytes) else corpus.encode("utf-8"))
    (folder / "queries.jsonl").write_text(queries, encoding="utf-8")
    (folder / "qrels.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\n", encoding="utf-8")
    return folder


def compare_args(folder, *extra):
    return ["compare", str(folder), "-a", "mock:word", "-b", "mock:char3", *extra]


def test_missing_evalset_folder_in_korean(tmp_path, capsys):
    assert main(["--lang", "ko", *compare_args(tmp_path / "missing")]) == 2
    err = capsys.readouterr().err
    assert err.startswith("오류: 평가 세트 폴더를 찾을 수 없습니다")
    assert "not found" not in err


def test_broken_json_line_in_korean(tmp_path, capsys):
    folder = make_set(tmp_path / "set", queries='{"_id": "q1", "text": "질문"}\n{broken\n')
    assert main(["--lang", "ko", *compare_args(folder)]) == 2
    err = capsys.readouterr().err
    assert "queries.jsonl 2번째 줄" in err and "큰따옴표로 감싼 속성 이름" in err
    assert "Expecting" not in err
    assert main(["--lang", "en", *compare_args(folder)]) == 2
    assert "queries.jsonl line 2: invalid JSON (Expecting property name" in capsys.readouterr().err


def test_unreadable_text_is_explained_in_korean(tmp_path, capsys):
    folder = make_set(tmp_path / "set", corpus=b'{"_id": "d1", "text": "\xff\xfe"}\n')
    assert main(["--lang", "ko", *compare_args(folder)]) == 2
    err = capsys.readouterr().err
    assert "을(를) 읽을 수 없습니다" in err and "UTF-8" in err and "codec" not in err
    assert main(["--lang", "en", *compare_args(folder)]) == 2
    assert "is not valid UTF-8 text" in capsys.readouterr().err


def test_os_errors_are_described_without_python_text():
    i18n.set_lang("ko")
    try:
        assert i18n.describe_os_error(FileNotFoundError(2, "No such file or directory")) == "파일이나 폴더가 없습니다"
        assert i18n.describe_os_error(PermissionError(13, "Permission denied")) == "접근 권한이 없습니다"
        assert "운영체제 오류 5" in i18n.describe_os_error(OSError(5, "Input/output error"))
        assert describe_json_error(json.JSONDecodeError("Extra data", "{} x", 3)) == "값 뒤에 불필요한 내용이 있습니다 (4열)"
    finally:
        i18n.set_lang("en")
    assert i18n.describe_os_error(PermissionError(13, "Permission denied")) == "permission denied"
    assert describe_json_error(json.JSONDecodeError("Extra data", "{} x", 3)) == "Extra data"


def test_provider_error_in_korean(demo_dir, capsys):
    assert main(["--lang", "ko", "compare", str(demo_dir), "-a", "nope", "-b", "mock:word"]) == 2
    assert "오류: 모델 지정 'nope'" in capsys.readouterr().err


def test_build_set_messages_in_korean(tmp_path, capsys):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("백업은 매일 새벽에 실행됩니다. 복구 절차는 문서에 있습니다.", encoding="utf-8")
    assert main(["--lang", "ko", "build-set", str(docs), "--out", str(tmp_path / "set")]) == 0
    out = capsys.readouterr().out
    assert "청크 1개, 질문 1개" in out and "참고: --llm mock" in out
    assert main(["--lang", "ko", "build-set", str(tmp_path / "none")]) == 2
    assert "문서 폴더를 찾을 수 없습니다" in capsys.readouterr().err


def test_argparse_errors_are_localized_and_exit_two(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--lang", "ko", "build-set", "x", "--max-copy", "7"])
    assert exc.value.code == 2
    assert "오류: 인자 --max-copy: 0 이상 1 이하여야 합니다" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        main(["--lang", "ko", "compare", "folder"])
    assert exc.value.code == 2
    assert "다음 인자가 필요합니다: -a, -b" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        main(["compare", "folder", "-a", "x", "-b", "y", "--bm25", "zzz"])
    assert exc.value.code == 2
    assert "error: argument --bm25: invalid choice" in capsys.readouterr().err
    with pytest.raises(SystemExit) as exc:
        main(["--lang", "fr", "demo"])  # only en and ko exist
    assert exc.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_help_is_available_in_both_languages(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--lang", "ko", "--help"])
    assert exc.value.code == 0
    ko = capsys.readouterr().out
    assert "사용법:" in ko and "옵션:" in ko and "위치 인자:" in ko and "내 문서로 임베딩 모델을 A/B 테스트합니다." in ko
    with pytest.raises(SystemExit):
        main(["compare", "--lang=ko", "--help"])
    assert "corpus.jsonl, queries.jsonl, qrels.tsv 가 든 폴더" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        main(["--lang", "en", "compare", "--help"])
    en = capsys.readouterr().out
    assert "usage:" in en and "options:" in en and "95% interval" in en


def test_version_output_is_not_translated(capsys):
    with pytest.raises(SystemExit):
        main(["--lang", "ko", "--version"])
    assert capsys.readouterr().out.startswith("embed-ab 0.")
