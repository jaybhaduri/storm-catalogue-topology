"""Exploratory ML benchmark (after both registrations; labelled exploratory in the manuscript).

Does the null for topological features survive nonlinear learners and richer topological inputs?
Target: like-for-like log ratio (storm 11-day peak / mean control 11-day peak) for P1-P3 and P4.
Feature sets (all leakage-free: scaling and PCA fitted inside each training fold):
  I   intensity: minimum Dst + 5 covariates
  D   I + duration (hours below -100 nT, integrated Dst)
  T5  D + 5 Betti summaries
  TC  D + first 10 principal components of the full mean Betti-0 and Betti-1 curves (122 values)
Learners: ridge (penalty 1), random forest (300 trees, min leaf 5), gradient boosting (300 stumps-depth-2,
learning rate 0.05, subsample 0.8). Validation: leave-one-storm-out. Samples: 107 intense storms; 206 pooled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

from . import part_a as A
from .snapshot import read_snapshot
from .topology import storm_features

OUT = "data/derived/paper"
CURVE = [f"c{j}" for j in range(122)]


def features(storms, dst, controls, omni, daily, tc):
    X = A.predictors(storms, dst, controls, omni, daily, tc)
    s = dst.set_index("time")["dst_nT"].asfreq("h")
    rows = []
    for _, st in storms.iterrows():
        x = s[st["window_start"]:st["window_end"]].interpolate(limit_direction="both").to_numpy()
        f = storm_features(x, tc)
        v = np.r_[f["b0_curves"].mean(axis=0), f["b1_curves"].mean(axis=0)]
        rows.append({"storm_id": st["storm_id"], **{c: float(v[j]) for j, c in enumerate(CURVE)}})
    return X.merge(pd.DataFrame(rows), on="storm_id")


SETS = {"I": ["dst_min_nT"] + A.COV, "D": A.DST + A.COV}


def design(tr, te, fs):
    base = SETS["D"] if fs in ("T5", "TC") else SETS[fs]
    mu, sd = tr[base].mean(), tr[base].std(ddof=0).replace(0, 1)
    a, b = ((tr[base] - mu) / sd).to_numpy(), ((te[base] - mu) / sd).to_numpy()
    extra = A.BETTI if fs == "T5" else (CURVE if fs == "TC" else [])
    if extra:
        m2, s2 = tr[extra].mean(), tr[extra].std(ddof=0).replace(0, 1)
        ea, eb = ((tr[extra] - m2) / s2).to_numpy(), ((te[extra] - m2) / s2).to_numpy()
        if fs == "TC":
            p = PCA(n_components=10).fit(ea)
            ea, eb = p.transform(ea), p.transform(eb)
        a, b = np.c_[a, ea], np.c_[b, eb]
    return a, b


def learner(k, seed):
    if k == "ridge":
        return Ridge(alpha=1.0)
    if k == "forest":
        return RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=seed, n_jobs=-1)
    return GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=seed)


def loso(df, y, fs, k):
    e = np.empty(len(df))
    for i in range(len(df)):
        tr, te = df.drop(index=i), df.loc[[i]]
        a, b = design(tr, te, fs)
        e[i] = (learner(k, i).fit(a, tr[y].to_numpy()).predict(b)[0] - te[y].iloc[0]) ** 2
    return e


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    tc = cfg["topology"]
    dst, omni = read_snapshot("kyoto_dst", "data/snapshots"), read_snapshot("omni2_hourly", "data/snapshots")
    daily = pd.read_parquet("data/derived/custody_proxies_daily.parquet")
    st = pd.read_parquet("data/derived/storms.parquet")
    intense = st[st["in_study"] & (st["anchor"].fillna("") == "")].sort_values("t_min")
    mod = pd.read_csv("data/derived/reg3/moderate_storms.csv", parse_dates=["t_min", "window_start", "window_end"])
    ictl = pd.read_parquet("data/derived/controls.parquet")
    mctl = pd.read_csv("data/derived/reg3/moderate_controls.csv", parse_dates=["control_start", "control_end"])
    fi = features(intense, dst, ictl, omni, daily, tc).assign(cls=1)
    fm = features(mod, dst, mctl, omni, daily, tc).assign(cls=0)
    yi = pd.read_csv("data/derived/part_a/explore_like_for_like.csv")[["storm_id"] + A.PRIMARY]
    ym = pd.read_csv("data/derived/reg3/primary_outcomes.csv")[["storm_id"] + A.PRIMARY]
    samples = {"intense": fi.merge(yi, on="storm_id"), "pooled": pd.concat([fi.merge(yi, on="storm_id"), fm.merge(ym, on="storm_id")])}
    rng = np.random.default_rng(0)
    rows = []
    for sname, df0 in samples.items():
        for y in ["P2", "P3", "P1", "P4"]:
            df = df0.dropna(subset=[y] + SETS["D"] + A.BETTI + CURVE).reset_index(drop=True)
            for k in ["ridge", "forest", "boosting"]:
                e = {fs: loso(df, y, fs, k) for fs in ["I", "D", "T5", "TC"]}
                for fs, ref in [("D", "I"), ("T5", "D"), ("TC", "D")]:
                    n = len(df)
                    rr = 1 - e[fs].mean() / e[ref].mean()
                    bs = [1 - e[fs][b].mean() / e[ref][b].mean() for b in (rng.integers(0, n, n) for _ in range(2000))]
                    rows.append({"sample": sname, "target": y, "learner": k, "added": fs, "vs": ref, "n": n,
                                 "rel_reduction": rr, "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5),
                                 "share_better": float((e[fs] < e[ref]).mean()), "mse_ref": e[ref].mean()})
                print(sname, y, k, flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(f"{OUT}/ml_benchmark.csv", index=False, lineterminator="\n")
    pd.set_option("display.width", 200)
    print(r.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
