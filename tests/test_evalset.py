import pytest

from embed_ab.evalset import EvalSetError, load_evalset


def make(tmp_path, corpus='{"_id":"d1","text":"x"}\n', queries='{"_id":"q1","text":"y"}\n', qrels="query-id\tcorpus-id\tscore\nq1\td1\t1\n"):
    (tmp_path / "corpus.jsonl").write_text(corpus, encoding="utf-8")
    (tmp_path / "queries.jsonl").write_text(queries, encoding="utf-8")
    (tmp_path / "qrels.tsv").write_text(qrels, encoding="utf-8")
    return tmp_path


def test_valid(tmp_path):
    es = load_evalset(make(tmp_path))
    assert es.qrels == {"q1": {"d1": 1}}


def test_title_is_prepended(tmp_path):
    es = load_evalset(make(tmp_path, corpus='{"_id":"d1","title":"T","text":"x"}\n'))
    assert es.corpus["d1"] == "T\nx"


@pytest.mark.parametrize(
    ("kw", "message"),
    [
        ({"qrels": "q1\tnope\t1\n"}, "corpus id 'nope'"),
        ({"qrels": "nope\td1\t1\n"}, "query id 'nope'"),
        ({"qrels": "q1\td1\n"}, "3 tab-separated"),
        ({"qrels": "q1\td1\tx\n"}, "not an integer"),
        ({"qrels": "q1\td1\t0\n"}, "no relevant document"),
        ({"corpus": "not json\n"}, "invalid JSON"),
        ({"corpus": '{"id":"d1"}\n'}, '"_id" and "text"'),
        ({"corpus": '{"_id":"d1","text":"a"}\n{"_id":"d1","text":"b"}\n'}, "duplicate"),
        ({"corpus": ""}, "is empty"),
    ],
)
def test_invalid_inputs_explain_the_fix(tmp_path, kw, message):
    with pytest.raises(EvalSetError, match=message):
        load_evalset(make(tmp_path, **kw))


def test_missing_files(tmp_path):
    with pytest.raises(EvalSetError, match="embed-ab build-set"):
        load_evalset(tmp_path)
    with pytest.raises(EvalSetError, match="folder not found"):
        load_evalset(tmp_path / "nope")


def test_subset_queries(demo_dir):
    es = load_evalset(demo_dir)
    assert len(es.subset_queries(5).queries) == 5
    assert es.subset_queries(None) is es
    with pytest.raises(EvalSetError):
        es.subset_queries(0)
