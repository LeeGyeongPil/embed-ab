import math

import numpy as np
import pytest

from embed_ab.metrics import all_metrics, hit_at_k, mrr_at_k, ndcg_at_k, recall_at_k

# Hand-computed cases. q1: relevant {a:1, c:1}; ranking [b, a, c].  q2: graded {x:3, y:1}; ranking [y, z, x].
RANKED = [["b", "a", "c"], ["y", "z", "x"]]
QRELS = [{"a": 1, "c": 1}, {"x": 3, "y": 1}]


def test_recall():
    assert recall_at_k(RANKED, QRELS, 2).tolist() == pytest.approx([0.5, 0.5])
    assert recall_at_k(RANKED, QRELS, 3).tolist() == pytest.approx([1.0, 1.0])
    assert recall_at_k(RANKED, QRELS, 1).tolist() == pytest.approx([0.0, 0.5])


def test_hit():
    assert hit_at_k(RANKED, QRELS, 1).tolist() == [0.0, 1.0]
    assert hit_at_k(RANKED, QRELS, 2).tolist() == [1.0, 1.0]


def test_mrr():
    assert mrr_at_k(RANKED, QRELS, 3).tolist() == pytest.approx([0.5, 1.0])
    assert mrr_at_k([["b", "c", "d"]], [{"a": 1}], 3).tolist() == [0.0]
    assert mrr_at_k([["b", "a"]], [{"a": 1}], 1).tolist() == [0.0]  # relevant doc below the cut-off


def test_ndcg_binary():
    # q1: dcg = 1/log2(3) + 1/log2(4); ideal = 1 + 1/log2(3)
    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 + 1 / math.log2(3)
    assert ndcg_at_k(RANKED[:1], QRELS[:1], 3)[0] == pytest.approx(dcg / idcg, abs=1e-9)


def test_ndcg_graded():
    # q2: dcg = 1/log2(2) + 3/log2(4); ideal = 3/log2(2) + 1/log2(3)
    dcg = 1.0 + 3 / 2
    idcg = 3.0 + 1 / math.log2(3)
    assert ndcg_at_k(RANKED[1:], QRELS[1:], 3)[0] == pytest.approx(dcg / idcg, abs=1e-9)


def test_ndcg_perfect_and_empty():
    assert ndcg_at_k([["a", "c"]], [{"a": 1, "c": 1}], 2)[0] == pytest.approx(1.0, abs=1e-9)
    assert ndcg_at_k([["a"]], [{"zzz": 1}], 1)[0] == 0.0


def test_all_metrics_names_and_shape():
    out = all_metrics(RANKED, QRELS, 3)
    assert set(out) == {"recall@3", "ndcg@3", "mrr@3", "hit@3"}
    assert all(isinstance(v, np.ndarray) and v.shape == (2,) for v in out.values())
