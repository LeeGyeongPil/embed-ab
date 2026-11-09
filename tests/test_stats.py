import numpy as np
import pytest

from embed_ab.overlap import jaccard_at_k, rbo_ext
from embed_ab.stats import paired_bootstrap_ci


def test_bootstrap_is_reproducible():
    diff = np.array([0.1, -0.2, 0.3, 0.0, 0.25, -0.05])
    assert paired_bootstrap_ci(diff, 500, seed=3) == paired_bootstrap_ci(diff, 500, seed=3)
    assert paired_bootstrap_ci(diff, 500, seed=3) != paired_bootstrap_ci(diff, 500, seed=4)


def test_bootstrap_identical_runs_give_zero_interval():
    assert paired_bootstrap_ci(np.zeros(20)) == (0.0, 0.0, 0.0)


def test_bootstrap_clear_difference_excludes_zero():
    mean, lo, hi = paired_bootstrap_ci(np.full(30, 0.2) + np.linspace(-0.02, 0.02, 30))
    assert lo > 0 and hi > lo and mean == pytest.approx(0.2, abs=1e-9)


def test_bootstrap_rejects_empty():
    with pytest.raises(ValueError, match="at least one query"):
        paired_bootstrap_ci(np.array([]))


def test_jaccard():
    assert jaccard_at_k(["a", "b", "c"], ["a", "b", "c"], 3) == 1.0
    assert jaccard_at_k(["a", "b"], ["c", "d"], 2) == 0.0
    assert jaccard_at_k(["a", "b", "c"], ["c", "d", "a"], 3) == pytest.approx(2 / 4)
    assert jaccard_at_k(["a", "b", "c"], ["b", "a", "x"], 2) == 1.0  # order ignored inside k


def test_rbo_boundaries_hold_for_any_depth():
    for depth in (3, 10):
        same = [str(i) for i in range(depth)]
        other = [f"x{i}" for i in range(depth)]
        assert rbo_ext(same, same, 0.9) == pytest.approx(1.0, abs=1e-12)
        assert rbo_ext(same, other, 0.9) == 0.0


def test_rbo_hand_computed_swap():
    # [a,b] vs [b,a], p=0.5: X1=0, X2=2 -> (2/2)*0.25 + ((1-0.5)/0.5) * (0*0.5 + (2/2)*0.25) = 0.5
    assert rbo_ext(["a", "b"], ["b", "a"], 0.5) == pytest.approx(0.5, abs=1e-12)


def test_rbo_top_weighted():
    base = ["a", "b", "c", "d", "e"]
    top_swap = ["x", "b", "c", "d", "e"]
    bottom_swap = ["a", "b", "c", "d", "x"]
    assert rbo_ext(base, top_swap, 0.8) < rbo_ext(base, bottom_swap, 0.8)


def test_rbo_invalid_p():
    with pytest.raises(ValueError):
        rbo_ext(["a"], ["a"], 1.0)
