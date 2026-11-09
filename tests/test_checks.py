from embed_ab.checks import run_checks

BASE = dict(n_queries=100, mean_a=0.6, mean_b=0.55, tie_share=0.3, ci=(-0.1, -0.01), copy_share=0.0, max_chunk_chars=500)


def codes(**over):
    return [w["code"] for w in run_checks(**{**BASE, **over})]


def test_clean_run_has_no_warnings():
    assert codes() == []


def test_few_queries_only_below_threshold():
    assert codes(n_queries=29) == ["few_queries"]
    assert codes(n_queries=30) == []


def test_ceiling_by_mean_or_ties():
    assert codes(mean_a=0.96, mean_b=0.97) == ["ceiling"]
    assert codes(mean_a=0.96, mean_b=0.7) == []  # only one system at the ceiling
    assert codes(tie_share=0.8) == ["ceiling"]
    assert codes(tie_share=0.79) == []


def test_ci_containing_zero():
    assert codes(ci=(-0.05, 0.02)) == ["not_significant"]
    assert codes(ci=(0.01, 0.05)) == []
    assert codes(ci=(0.0, 0.0)) == ["not_significant"]


def test_copied_queries_and_truncation():
    assert codes(copy_share=0.3) == ["copied_queries"]
    assert codes(copy_share=0.29) == []
    assert codes(max_seq_lengths={"st:x": 128}, max_chunk_chars=500) == ["chunk_truncation"]
    assert codes(max_seq_lengths={"st:x": 512}, max_chunk_chars=500) == []
    assert codes(max_seq_lengths={"mock:word": None}) == []
