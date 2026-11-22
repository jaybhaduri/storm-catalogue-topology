"""Figure data and supplementary checks for the manuscript (exploratory unless stated).

1. Superposed-epoch curves: daily P1-P3 (primary population) from day -3 to +10, divided by the storm's
   own control mean, median and interquartile range across storms, intense and moderate separately.
2. Placebo test: random storm-free dates (no storm with minimum below -50 nT within 13 days), controls by
   the same rule, the same like-for-like ratio. Under no storm effect it should centre near 1.
3. Dose-response across all storms: like-for-like log ratio against |Dst minimum|.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from . import part_a as A
from . import reg3 as R
from .snapshot import read_snapshot
from .storms import detect_storms, select_controls

OUT = "data/derived/paper"


def epoch(series, storms, controls, label):
    rows = []
    for _, s in storms.iterrows():
        cs = controls[controls["storm_id"] == s["storm_id"]]
        if len(cs) < 3:
            continue
        base = pd.concat([series.loc[c["control_start"]:c["control_end"], ["P1", "P2", "P3"]] for _, c in cs.iterrows()]).mean()
        d0 = s["t_min"].normalize()
        for k in range(-3, 11):
            v = series.loc[d0 + pd.Timedelta(days=k), ["P1", "P2", "P3"]] / base
            rows.append({"class": label, "storm_id": s["storm_id"], "day": k, **v.to_dict()})
    e = pd.DataFrame(rows)
    return e.groupby(["class", "day"])[["P1", "P2", "P3"]].quantile([0.25, 0.5, 0.75]).unstack().reset_index()


def main(n_placebo=200, seed=0):
    cfg = yaml.safe_load(open("config.yaml"))
    daily = pd.read_parquet("data/derived/custody_proxies_daily.parquet")
    s = A.daily_series(daily, "inactive", 600)
    st = pd.read_parquet("data/derived/storms.parquet")
    intense = st[st["in_study"] & (st["anchor"].fillna("") == "")]
    ictl = pd.read_parquet("data/derived/controls.parquet")
    mod = pd.read_csv("data/derived/reg3/moderate_storms.csv", parse_dates=["t_min"])
    mctl = pd.read_csv("data/derived/reg3/moderate_controls.csv", parse_dates=["control_start", "control_end"])
    ep = pd.concat([epoch(s, intense, ictl, "intense"), epoch(s, mod, mctl, "moderate")])
    ep.columns = ["_".join(str(x) for x in c if x != "") for c in ep.columns]
    ep.to_csv(f"{OUT}/superposed_epoch.csv", index=False, lineterminator="\n")

    # placebo
    c = copy.deepcopy(cfg); c["storms"]["threshold_nT"], c["storms"]["anchors"] = -50, []
    dst, kp, omni = (read_snapshot(n, "data/snapshots") for n in ("kyoto_dst", "gfz_kp", "omni2_hourly"))
    allst = detect_storms(dst, c)
    tmins = allst["t_min"].to_numpy()
    rng = np.random.default_rng(seed)
    days = pd.date_range("2000-01-05", "2025-12-15", freq="D")
    cand = [d for d in rng.permutation(days) if np.all(np.abs((tmins - np.datetime64(d)) / np.timedelta64(1, "D")) > 13)]
    pl = pd.DataFrame({"t_min": pd.to_datetime(cand[:n_placebo]) + pd.Timedelta(hours=12)})
    pl["storm_id"] = ["P" + t.strftime("%Y%m%d") for t in pl["t_min"]]
    pl["window_start"], pl["window_end"] = pl["t_min"] - pd.Timedelta(days=3), pl["t_min"] + pd.Timedelta(days=10)
    pl["dst_min_nT"], pl["in_study"], pl["dst_version_at_min"] = np.nan, True, None
    blk = pd.concat([allst.assign(in_study=False), pl], ignore_index=True)
    pctl = select_controls(blk, kp, omni, c)
    py = R.like_for_like(s, pl, pctl, ["P1", "P2", "P3"])
    py.to_csv(f"{OUT}/placebo_outcomes.csv", index=False, lineterminator="\n")
    rows = []
    for p in ["P1", "P2", "P3"]:
        x = py[p].dropna().to_numpy()
        b = [np.median(rng.choice(x, len(x))) for _ in range(2000)]
        rows.append({"proxy": p, "n": len(x), "median_ratio": float(np.exp(np.median(x))),
                     "ci_lo": float(np.exp(np.percentile(b, 2.5))), "ci_hi": float(np.exp(np.percentile(b, 97.5))),
                     "p_two_sided": float(stats.wilcoxon(x).pvalue)})
    plr = pd.DataFrame(rows); plr.to_csv(f"{OUT}/placebo_summary.csv", index=False, lineterminator="\n")
    print(plr.round(3).to_string(index=False))

    # dose-response across all storms
    li = pd.read_csv("data/derived/part_a/explore_like_for_like.csv").assign(**{"class": "intense"})
    lm = pd.read_csv("data/derived/reg3/primary_outcomes.csv").assign(**{"class": "moderate"})
    dr = pd.concat([li[["storm_id", "class", "dst_min_nT", "P1", "P2", "P3"]], lm[["storm_id", "class", "dst_min_nT", "P1", "P2", "P3"]]])
    dr.to_csv(f"{OUT}/dose_response.csv", index=False, lineterminator="\n")
    for p in ["P1", "P2", "P3"]:
        d = dr.dropna(subset=[p])
        r = stats.spearmanr(-d["dst_min_nT"], d[p])
        print(f"[dose] {p}: n={len(d)} Spearman(|Dst|, log ratio) = {r[0]:.3f}, p = {r[1]:.2g}")
    # absolute levels for context
    for lab, ss, cc in [("intense", intense, ictl), ("moderate", mod, mctl)]:
        pk, bs = [], []
        for _, r_ in ss.iterrows():
            cs = cc[cc["storm_id"] == r_["storm_id"]]
            if len(cs) < 3: continue
            d0 = r_["t_min"].normalize()
            pk.append(s.loc[d0:d0 + pd.Timedelta(days=10), ["P1", "P2", "P3"]].max())
            bs.append(pd.concat([s.loc[c_["control_start"]:c_["control_end"], ["P1", "P2", "P3"]] for _, c_ in cs.iterrows()]).mean())
        print(f"[levels] {lab}: median peak {pd.DataFrame(pk).median().round(4).to_dict()}  median control mean {pd.DataFrame(bs).median().round(4).to_dict()}")


if __name__ == "__main__":
    main()
