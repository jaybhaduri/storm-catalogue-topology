"""Registration 3 confirmatory analysis (frozen before any Registration 3 outcome is computed).

Primary sample: isolated moderate storms 2000-2025 (data/derived/reg3/moderate_storms.csv) with at
least 3 quiet-time controls. Population: inactive objects below 600 km (as Part A).

Outcome (like for like): log( max of the daily proxy over storm days 0..+10
                              / mean over the storm's controls of the max over control days 0..+10 ).

B1  P2 rises in moderate storms: one-sided Wilcoxon signed-rank, median log ratio > 0.
B2  Dose-response across storm classes: moderate-storm log ratios are lower than Part A's intense-storm
    log ratios (data/derived/part_a/explore_like_for_like.csv, frozen), for P2 and for P3:
    one-sided Mann-Whitney U, two tests.
B3  Staleness does not rise materially: P1 median log ratio below log(1.10): one-sided Wilcoxon
    signed-rank on (log ratio - log 1.10), alternative "less".
Error control (gatekeeping, family-wise alpha 0.05): stage 1 tests B2-P2 then B2-P3 in fixed order, each
at 0.05; only if both pass, stage 2 tests B1 and B3 with Holm at 0.05. A test not reached is not passed.
A bet is won when every one of its tests passes.

B4 (secondary confirmatory, forward): intense storms (Part A definition) with t0 from 1 January 2026
to 30 June 2027: P2 and P3 log ratios > 0, one-sided Wilcoxon, Holm across the two. Tested only if at
least 6 storms qualify; otherwise descriptive.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from . import part_a as A

MARGIN_P1 = float(np.log(1.10))
PEAK_DAYS = 10
MIN_CONTROLS = 3
ALPHA = 0.05
N_BOOT = 2000


def like_for_like(series: pd.DataFrame, storms: pd.DataFrame, controls: pd.DataFrame, proxies) -> pd.DataFrame:
    rows = []
    for _, s in storms.iterrows():
        d0 = s["t_min"].normalize()
        pk = series.loc[d0:d0 + pd.Timedelta(days=PEAK_DAYS), proxies].max()
        cs = controls[controls["storm_id"] == s["storm_id"]]
        r = {"storm_id": s["storm_id"], "t_min": s["t_min"], "dst_min_nT": s["dst_min_nT"],
             "dst_version_at_min": s.get("dst_version_at_min"), "n_controls": len(cs)}
        if len(cs) < MIN_CONTROLS:
            rows.append({**r, **{p: np.nan for p in proxies}})
            continue
        cp = pd.concat([series.loc[c["control_start"]:c["control_start"] + pd.Timedelta(days=PEAK_DAYS), proxies].max()
                        for _, c in cs.iterrows()], axis=1).mean(axis=1)
        rows.append({**r, **{p: float(np.log(pk[p] / cp[p])) if (pk[p] > 0 and cp[p] > 0) else np.nan for p in proxies}})
    return pd.DataFrame(rows)


def _boot_median(x, rng):
    b = [np.median(rng.choice(x, len(x))) for _ in range(N_BOOT)]
    return np.percentile(b, [2.5, 97.5])


def primary_tests(mod: pd.DataFrame, intense: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    x = mod["P2"].dropna().to_numpy()
    lo, hi = _boot_median(x, rng)
    rows.append({"bet": "B1", "test": "P2 moderate > 0", "n": len(x), "median_log_ratio": np.median(x),
                 "median_ratio": np.exp(np.median(x)), "ci_lo": np.exp(lo), "ci_hi": np.exp(hi),
                 "p": stats.wilcoxon(x, alternative="greater").pvalue})
    for p in ["P2", "P3"]:
        x, y = mod[p].dropna().to_numpy(), intense[p].dropna().to_numpy()
        diffs = [np.median(rng.choice(x, len(x))) - np.median(rng.choice(y, len(y))) for _ in range(N_BOOT)]
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        rows.append({"bet": "B2", "test": f"{p} moderate < intense", "n": len(x), "n_reference": len(y),
                     "median_log_ratio": np.median(x), "median_ratio": np.exp(np.median(x)),
                     "reference_median_ratio": np.exp(np.median(y)),
                     "diff_ci_lo": lo, "diff_ci_hi": hi, "p": stats.mannwhitneyu(x, y, alternative="less").pvalue})
    x = mod["P1"].dropna().to_numpy()
    lo, hi = _boot_median(x, rng)
    rows.append({"bet": "B3", "test": "P1 moderate < log 1.10", "n": len(x), "median_log_ratio": np.median(x),
                 "median_ratio": np.exp(np.median(x)), "ci_lo": np.exp(lo), "ci_hi": np.exp(hi),
                 "p": stats.wilcoxon(x - MARGIN_P1, alternative="less").pvalue})
    r = pd.DataFrame(rows)
    return gatekeep(r)


def gatekeep(r: pd.DataFrame) -> pd.DataFrame:
    r = r.copy()
    r["p_adjusted"], r["test_passed"] = np.nan, False
    gate = True
    for i in r.index[r["bet"] == "B2"]:                       # stage 1: fixed sequence
        r.loc[i, "p_adjusted"] = r.loc[i, "p"]
        gate = gate and r.loc[i, "p"] < ALPHA
        r.loc[i, "test_passed"] = gate
    s2 = r.index[r["bet"].isin(["B1", "B3"])]
    if gate:                                                   # stage 2: Holm over B1 and B3
        r.loc[s2, "p_adjusted"] = A.holm(r.loc[s2, "p"]).to_numpy()
        r.loc[s2, "test_passed"] = r.loc[s2, "p_adjusted"] < ALPHA
    return r


def bet_verdicts(r: pd.DataFrame) -> dict:
    return {b: ("won" if g["test_passed"].all() else "lost") for b, g in r.groupby("bet")}


def forward_tests(fwd: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p in ["P2", "P3"]:
        x = fwd[p].dropna().to_numpy()
        rows.append({"bet": "B4", "test": f"{p} intense 2026-27 > 0", "n": len(x),
                     "median_ratio": float(np.exp(np.median(x))) if len(x) else np.nan,
                     "p": stats.wilcoxon(x, alternative="greater").pvalue if len(x) >= 6 else np.nan})
    r = pd.DataFrame(rows)
    r["p_holm"] = A.holm(r["p"]).to_numpy() if r["p"].notna().all() else np.nan
    r["test_passed"] = r["p_holm"] < ALPHA
    return r


def run(out="data/derived/reg3", population="inactive", max_alt=600, mod_file="moderate_storms.csv",
        ctl_file="moderate_controls.csv", label="primary"):
    daily = pd.read_parquet("data/derived/custody_proxies_daily.parquet")
    series = A.daily_series(daily, population, max_alt)
    mod = pd.read_csv(f"{out}/{mod_file}", parse_dates=["t_min"])
    ctl = pd.read_csv(f"{out}/{ctl_file}", parse_dates=["control_start", "control_end"])
    y = like_for_like(series, mod, ctl, A.PRIMARY)
    y.to_csv(f"{out}/{label}_outcomes.csv", index=False, lineterminator="\n")
    intense = pd.read_csv("data/derived/part_a/explore_like_for_like.csv")
    r = primary_tests(y, intense)
    r.to_csv(f"{out}/{label}_tests.csv", index=False, lineterminator="\n")
    return y, r, bet_verdicts(r)


def part_a_control_overlap(mod: pd.DataFrame, part_a_controls: pd.DataFrame) -> pd.Series:
    """True where the storm's days 0..+10 overlap any Part A control window (partial prior exposure)."""
    c = part_a_controls.drop_duplicates("control_start")
    def ov(t):
        d0 = t.normalize()
        return bool(((c["control_start"] <= d0 + pd.Timedelta(days=PEAK_DAYS)) & (c["control_end"] >= d0)).any())
    return mod["t_min"].apply(ov)


def robustness(out="data/derived/reg3"):
    """Secondary checks (registered; reported whichever way they fall, not part of the bet verdicts)."""
    y = pd.read_csv(f"{out}/primary_outcomes.csv", parse_dates=["t_min"])
    intense = pd.read_csv("data/derived/part_a/explore_like_for_like.csv")
    pac = pd.read_csv("data/derived/controls.csv", parse_dates=["control_start", "control_end"])
    checks = {
        "exclude_part_a_control_overlap": y[~part_a_control_overlap(y, pac)],
        "final_dst_only": y[y["dst_version_at_min"] == "final"],
        "five_controls_only": y[y["n_controls"] == 5],
    }
    rows = []
    for k, d in checks.items():
        r = primary_tests(d.reset_index(drop=True), intense)
        r.insert(0, "check", k)
        rows.append(r)
    # B1 with P3 in place of P2; other populations
    x = y["P3"].dropna().to_numpy()
    rows.append(pd.DataFrame([{"check": "B1_with_P3", "bet": "B1", "n": len(x), "median_ratio": float(np.exp(np.median(x))),
                               "p": stats.wilcoxon(x, alternative="greater").pvalue}]))
    for label, pop, alt in [("all_inactive_leo", "inactive", None), ("active", "active", None)]:
        _, r, _ = run(out, pop, alt, label=label)
        r.insert(0, "check", label)
        rows.append(r)
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(f"{out}/robustness.csv", index=False, lineterminator="\n")
    return res
