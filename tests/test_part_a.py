import numpy as np
import pandas as pd

from phinn_sw import part_a as A


def test_holm_matches_textbook():
    p = pd.Series({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005})
    adj = A.holm(p)
    assert np.allclose(adj[["d", "a", "c", "b"]], [0.02, 0.03, 0.06, 0.06])


def test_weighted_median():
    assert A._wmedian(np.array([1.0, 100.0, 2.0]), np.array([12, 1, 10])) == 1.0
    assert A._wmedian(np.array([5.0, -1e8]), np.array([100, 1])) == 5.0


def _toy(n=60, seed=1, signal=True):
    r = np.random.default_rng(seed)
    df = pd.DataFrame({c: r.normal(size=n) for c in A.DST + A.BETTI + A.COV})
    df["y"] = 0.5 * df["dst_min_nT"] + (1.5 * df["hours_below_100"] if signal else 0) + r.normal(0, 0.3, n)
    return df


def test_loso_finds_a_real_nested_effect_and_not_a_fake_one():
    pc = {"ridge_penalty": 1.0, "bootstrap_resamples": 200}
    yes = A.compare(_toy(), "y", "M0", "M1", "ols", pc, np.random.default_rng(0))
    no = A.compare(_toy(signal=False), "y", "M0", "M1", "ols", pc, np.random.default_rng(0))
    assert yes["p"] < 0.001 and yes["rel_reduction"] > 0.5
    assert no["rel_reduction"] < 0.05
