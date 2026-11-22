"""Part A exploratory analyses (labelled exploratory; not part of Registration 1's confirmatory tests).

1. Storm-locked signal: storm peak (days 0..+10) against the same 11-day peak statistic taken at the
   start of each control window (like-for-like), per storm; Wilcoxon signed-rank on log ratios.
2. Does minimum Dst alone predict the outcome? LOSO M0-without-Dst (covariates only) against M0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from scipy import stats

from . import part_a as A


def main(out_dir="data/derived/part_a"):
    cfg = yaml.safe_load(open("config.yaml"))
    pc = cfg["part_a"]
    daily = pd.read_parquet("data/derived/custody_proxies_daily.parquet")
    storms = pd.read_parquet("data/derived/storms.parquet")
    storms = storms[storms["in_study"] & ~storms["anchor"].fillna("").isin(pc["exclude_from_models"])].sort_values("t_min")
    ctl = pd.read_parquet("data/derived/controls.parquet")
    s = A.daily_series(daily, pc["population"], pc["max_altitude_km"])
    rows = []
    for _, st in storms.iterrows():
        d0 = st["t_min"].normalize()
        pk = s.loc[d0:d0 + pd.Timedelta(days=10), A.PRIMARY].max()
        cs = ctl[ctl["storm_id"] == st["storm_id"]]
        cp = pd.concat([s.loc[c["control_start"]:c["control_start"] + pd.Timedelta(days=10), A.PRIMARY].max()
                        for _, c in cs.iterrows()], axis=1).mean(axis=1)
        rows.append({"storm_id": st["storm_id"], "severe": st["severe"], "dst_min_nT": st["dst_min_nT"],
                     **{p: np.log(pk[p] / cp[p]) if pk[p] > 0 and cp[p] > 0 else np.nan for p in A.PRIMARY}})
    t = pd.DataFrame(rows)
    t.to_csv(f"{out_dir}/explore_like_for_like.csv", index=False, lineterminator="\n")
    out = []
    for p in A.PRIMARY:
        for lab, sub in [("all", t), ("severe", t[t["severe"]])]:
            x = sub[p].dropna()
            out.append({"proxy": p, "storms": lab, "n": len(x), "median_log_ratio": x.median(),
                        "median_ratio": float(np.exp(x.median())), "share_above_1": float((x > 0).mean()),
                        "p_greater": float(stats.wilcoxon(x, alternative="greater").pvalue),
                        "spearman_vs_dst_min": float(stats.spearmanr(sub["dst_min_nT"], sub[p], nan_policy="omit")[0])})
    r = pd.DataFrame(out)
    # 2. minimum Dst against covariates only
    data = pd.read_csv(f"{out_dir}/storm_table.csv", parse_dates=["t_min"])
    data = data[~data["anchor"].fillna("").isin(pc["exclude_from_models"])].reset_index(drop=True)
    A.MODEL_COLS["C"] = A.COV
    rng = np.random.default_rng(0)
    dst = []
    for p in A.PRIMARY:
        res = A.compare(data.dropna(subset=[p]).reset_index(drop=True), p, "C", "M0", "ols", pc, rng)
        dst.append({"proxy": p, **{k: res[k] for k in ["n", "rel_reduction", "ci_lo", "ci_hi", "p"]}})
    d = pd.DataFrame(dst)
    r.to_csv(f"{out_dir}/explore_storm_locked.csv", index=False, lineterminator="\n")
    d.to_csv(f"{out_dir}/explore_dst_min_alone.csv", index=False, lineterminator="\n")
    pd.set_option("display.width", 200)
    print(r.round(3).to_string(index=False))
    print(d.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
