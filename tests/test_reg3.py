"""Registration 3 dry run: the frozen code on Part A's (already seen) intense storms and on synthetic data.
Nothing here reads a moderate-storm outcome."""
import numpy as np
import pandas as pd
import pytest

from phinn_sw import part_a as A
from phinn_sw import reg3 as R


@pytest.mark.skipif(not __import__("os").path.exists("data/derived/custody_proxies_daily.parquet"), reason="needs data")
def test_like_for_like_reproduces_part_a_exploratory_table():
    daily = pd.read_parquet("data/derived/custody_proxies_daily.parquet")
    s = A.daily_series(daily, "inactive", 600)
    st = pd.read_parquet("data/derived/storms.parquet")
    st = st[st["in_study"] & (st["anchor"].fillna("") == "")].sort_values("t_min")
    ctl = pd.read_parquet("data/derived/controls.parquet")
    y = R.like_for_like(s, st, ctl, A.PRIMARY)
    ref = pd.read_csv("data/derived/part_a/explore_like_for_like.csv")
    m = y.merge(ref, on="storm_id", suffixes=("", "_ref")).dropna(subset=["P2"])
    assert len(m) >= 105
    for p in A.PRIMARY:
        assert np.allclose(m[p], m[p + "_ref"], equal_nan=True)


def _synthetic(shift_mod, n=100, seed=0):
    r = np.random.default_rng(seed)
    mod = pd.DataFrame({"P1": r.normal(-0.3, 0.9, n), "P2": r.normal(shift_mod, 0.5, n), "P3": r.normal(0.0, 0.6, n)})
    intense = pd.DataFrame({"P1": r.normal(-0.2, 0.9, 107), "P2": r.normal(0.55, 0.55, 107), "P3": r.normal(0.55, 0.7, 107)})
    return mod, intense


def test_bets_won_when_effects_exist():
    mod, intense = _synthetic(0.2)
    v = R.bet_verdicts(R.primary_tests(mod, intense))
    assert v == {"B1": "won", "B2": "won", "B3": "won"}


def test_b1_lost_without_a_rise():
    mod, intense = _synthetic(0.0)
    v = R.bet_verdicts(R.primary_tests(mod, intense))
    assert v["B1"] == "lost"


def test_gatekeeping_blocks_stage_two_when_b2_fails():
    r = pd.DataFrame({"bet": ["B1", "B2", "B2", "B3"], "p": [0.001, 0.2, 0.001, 0.001]})
    g = R.gatekeep(r)
    assert not g["test_passed"].any()
