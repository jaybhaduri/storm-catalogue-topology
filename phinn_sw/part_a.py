"""Part A (Registration 1, DOI 10.5281/zenodo.24421887): H1 and H2.

H1  M1 (M0 + hours below -100 nT + integrated Dst) beats M0 (minimum Dst + covariates)
H2  M2 (M1 + first two PCs of five Betti features) beats M1
Outcome per storm and proxy: log(peak daily value over days 0..+10 / mean over the storm's controls).
LOSO cross-validation over the intense storms; one-sided Wilcoxon signed-rank on paired squared errors;
Holm across the four primary proxies; 2,000-resample bootstrap of the relative MSE reduction.
Every setting comes from config.yaml `part_a` (frozen at commit f87999f).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge

from .topology import storm_features

PRIMARY = ["P1", "P2", "P3", "P4"]
COUNT_SHARE = {   # proxy name -> (numerator column, denominator column)
    "P1": ("p1_stale_over_24h", "n_tracked"),
    "P2": ("p2_objects_z_over_3", "n_pairs"),
    "P3": ("p3_objects_over_quiet_q99", "n_tracked"),
    "p1_stale_over_48h": ("p1_stale_over_48h", "n_tracked"),
    "p1_stale_over_72h": ("p1_stale_over_72h", "n_tracked"),
    "p3_objects_over_5km": ("p3_objects_over_5km", "n_tracked"),
    "p3_objects_over_10km": ("p3_objects_over_10km", "n_tracked"),
    "p3_objects_over_25km": ("p3_objects_over_25km", "n_tracked"),
}
DST = ["dst_min_nT", "hours_below_100", "integrated_dst_nTh"]
BETTI = ["beta0_track_max", "beta0_track_mean", "beta1_track_max", "b0_area", "beta1_total"]
COV = ["f107_window_mean", "log_catalogue_size", "publication_cadence_30d", "sin_doy", "cos_doy"]


# ------------------------------------------------------------------ daily proxy series

def _wmedian(v, w):
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return float(v[o][np.searchsorted(c, c[-1] / 2)]) if c[-1] > 0 else np.nan


def daily_series(daily: pd.DataFrame, population: str | None, max_alt_km: float | None) -> pd.DataFrame:
    """One row per day: every proxy for the chosen population and altitude range, bands combined."""
    d = daily
    if population is not None:
        d = d[d["population"] == population]
    if max_alt_km is not None:
        d = d[d["band_km"] + 50 <= max_alt_km]          # bands are 50 km wide, labelled by lower edge
    cnt = [c for c in d.columns if c.startswith(("p1_", "p2_objects", "p3_", "p4_reentries", "n_", "m1_"))]
    g = d.groupby("day")
    out = g[cnt].sum()
    # P4: decay rate = minus the pair-weighted median of the band medians of da/dt (positive = faster
    # decay). A weighted mean was tried first: re-entry outliers in the 100-250 km bands (band medians
    # up to 1e8 km/day) made the control mean negative for 49 of 108 storms. Decided 21 Nov 2026,
    # after registration; reported as a deviation.
    out["P4"] = -d.dropna(subset=["p4_da_median_km_per_day"]).groupby("day").apply(
        lambda g: _wmedian(g["p4_da_median_km_per_day"].to_numpy(), g["n_pairs"].to_numpy()))
    for k, (a, b) in COUNT_SHARE.items():
        out[k] = out[a] / out[b].replace(0, np.nan)
    out["p4_reentries"] = out["p4_reentries"].astype(float)
    return out.asfreq("D")


def outcomes(series: pd.DataFrame, storms: pd.DataFrame, controls: pd.DataFrame, proxies, peak_days) -> pd.DataFrame:
    """log(peak ratio) per storm and proxy; NaN where the peak or the control mean is not positive."""
    rows = []
    for _, s in storms.iterrows():
        d0 = s["t_min"].normalize()
        pk = series.loc[d0 + pd.Timedelta(days=peak_days[0]):d0 + pd.Timedelta(days=peak_days[1])]
        cs = controls[controls["storm_id"] == s["storm_id"]]
        ctl = pd.concat([series.loc[c["control_start"]:c["control_end"]] for _, c in cs.iterrows()]) if len(cs) else None
        r = {"storm_id": s["storm_id"], "n_controls": len(cs)}
        for p in proxies:
            peak = pk[p].max()
            base = ctl[p].mean() if ctl is not None else np.nan
            r[p] = float(np.log(peak / base)) if (peak > 0 and base > 0) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ storm predictors

def predictors(storms, dst, controls, omni, all_daily, tc) -> pd.DataFrame:
    s = dst.set_index("time")["dst_nT"].asfreq("h")
    f = omni.set_index("time")["f107_sfu"].resample("D").mean()
    cat = all_daily.groupby("day")[["n_tracked", "n_pairs"]].sum().asfreq("D")
    rows = []
    for _, st in storms.iterrows():
        x = s[st["window_start"]:st["window_end"]].interpolate(limit_direction="both").to_numpy()
        bf = storm_features(x, tc)
        d0 = st["t_min"].normalize()
        prev = cat.loc[d0 - pd.Timedelta(days=30):d0 - pd.Timedelta(days=1)]
        doy = st["t_min"].dayofyear
        cs = controls[controls["storm_id"] == st["storm_id"]]
        f107 = cs["storm_f107"].iloc[0] if len(cs) else f.loc[st["window_start"].normalize():st["window_end"].normalize()].mean()
        rows.append({"storm_id": st["storm_id"], **{k: st[k] for k in DST},
                     **{k: bf[k] for k in BETTI},
                     "f107_window_mean": float(f107),
                     "log_catalogue_size": float(np.log(cat.loc[d0 - pd.Timedelta(days=1), "n_tracked"])),
                     "publication_cadence_30d": float(prev["n_pairs"].sum() / prev["n_tracked"].sum()),
                     "sin_doy": np.sin(2 * np.pi * doy / 365.25), "cos_doy": np.cos(2 * np.pi * doy / 365.25)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ models

MODEL_COLS = {"M0": ["dst_min_nT"] + COV, "M1": DST + COV, "M2": DST + COV}


def _design(train: pd.DataFrame, test: pd.DataFrame, model: str):
    cols = MODEL_COLS[model]
    mu, sd = train[cols].mean(), train[cols].std(ddof=0).replace(0, 1)
    Xtr, Xte = ((train[cols] - mu) / sd).to_numpy(), ((test[cols] - mu) / sd).to_numpy()
    if model == "M2":                                   # PCA fitted inside the training fold only
        bm, bs = train[BETTI].mean(), train[BETTI].std(ddof=0).replace(0, 1)
        pca = PCA(n_components=2).fit(((train[BETTI] - bm) / bs).to_numpy())
        Xtr = np.c_[Xtr, pca.transform(((train[BETTI] - bm) / bs).to_numpy())]
        Xte = np.c_[Xte, pca.transform(((test[BETTI] - bm) / bs).to_numpy())]
    return Xtr, Xte


def _estimator(kind: str, ridge_penalty: float):
    return {"ols": LinearRegression(), "ridge": Ridge(alpha=ridge_penalty),
            "huber": HuberRegressor(max_iter=1000)}[kind]


def loso_errors(df: pd.DataFrame, y: str, model: str, kind: str, ridge_penalty: float) -> np.ndarray:
    err = np.empty(len(df))
    idx = np.arange(len(df))
    for i in idx:
        tr, te = df.iloc[idx != i], df.iloc[[i]]
        Xtr, Xte = _design(tr, te, model)
        m = _estimator(kind, ridge_penalty).fit(Xtr, tr[y].to_numpy())
        err[i] = (m.predict(Xte)[0] - te[y].iloc[0]) ** 2
    return err


def max_vif(df: pd.DataFrame, model: str) -> float:
    X, _ = _design(df, df.iloc[:1], model)
    v = []
    for j in range(X.shape[1]):
        o = np.delete(X, j, axis=1)
        r2 = LinearRegression().fit(o, X[:, j]).score(o, X[:, j])
        v.append(1 / max(1 - r2, 1e-12))
    return float(max(v))


def compare(df, y, small, big, kind, pc, rng) -> dict:
    e0 = loso_errors(df, y, small, kind, pc["ridge_penalty"])
    e1 = loso_errors(df, y, big, kind, pc["ridge_penalty"])
    d = e0 - e1
    p = float(stats.wilcoxon(d, alternative="greater").pvalue) if np.any(d != 0) else 1.0
    rr = 1 - e1.mean() / e0.mean()
    n = len(d)
    boots = np.array([1 - e1[b].mean() / e0[b].mean() for b in (rng.integers(0, n, n) for _ in range(pc["bootstrap_resamples"]))])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": n, "mse_small": e0.mean(), "mse_big": e1.mean(), "rel_reduction": rr,
            "ci_lo": lo, "ci_hi": hi, "p": p}


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values().index
    m, adj, run = len(p), {}, 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i]))
        adj[i] = run
    return pd.Series(adj).reindex(p.index)


def test_hypotheses(data: pd.DataFrame, proxies, pc: dict, kind: str = "ols", seed: int = 0, vif_switch: bool = True):
    """Rows per hypothesis x proxy. With vif_switch, a comparison whose larger model has VIF > threshold
    is refitted as ridge (primary) and the OLS fit is kept alongside."""
    rng = np.random.default_rng(seed)
    rows = []
    for hyp, small, big in [("H1", "M0", "M1"), ("H2", "M1", "M2")]:
        for p in proxies:
            df = data.dropna(subset=[p] + DST + BETTI + COV).reset_index(drop=True)
            r = {"hypothesis": hyp, "proxy": p, "fit": kind}
            if len(df) < pc["min_storms_per_proxy"]:
                rows.append({**r, "n": len(df), "descriptive_only": True})
                continue
            vif = max_vif(df, big)
            k = "ridge" if (vif_switch and kind == "ols" and vif > pc["vif_ridge_threshold"]) else kind
            res = compare(df, p, small, big, k, pc, rng)
            row = {**r, "fit": k, "max_vif": vif, "descriptive_only": False, **res}
            if k == "ridge":
                o = compare(df, p, small, big, "ols", pc, np.random.default_rng(seed))
                row.update({"ols_rel_reduction": o["rel_reduction"], "ols_p": o["p"]})
            rows.append(row)
    out = pd.DataFrame(rows)
    for c in ["p", "ci_lo", "rel_reduction"]:
        if c not in out:
            out[c] = np.nan
    out["p_holm"] = np.nan
    for hyp in ["H1", "H2"]:
        m = (out["hypothesis"] == hyp) & ~out["descriptive_only"]
        if m.any():
            out.loc[m, "p_holm"] = holm(out.loc[m, "p"]).to_numpy()
    out["supported"] = (out["p_holm"] < pc["alpha"]) & (out["ci_lo"] > 0)
    out["above_min_effect"] = out["rel_reduction"] >= pc["min_effect"]
    return out


def verdict(res: pd.DataFrame, pc: dict) -> dict:
    v = {}
    for hyp in ["H1", "H2"]:
        k = int(res[(res["hypothesis"] == hyp) & res["proxy"].isin(PRIMARY)]["supported"].sum())
        v[hyp] = "supported" if k >= pc["supported_if_proxies"] else ("partially supported" if k >= 1 else "not supported")
        v[hyp + "_proxies"] = k
    return v


# ------------------------------------------------------------------ the full run

SECONDARY_SHARES = ["p1_stale_over_48h", "p1_stale_over_72h", "p3_objects_over_5km",
                    "p3_objects_over_10km", "p3_objects_over_25km", "p4_reentries"]


def run(cfg: dict, snap_dir: str = "data/snapshots", out_dir: str = "data/derived/part_a"):
    import copy
    import os
    from .snapshot import read_snapshot
    from .storms import select_controls
    pc, tc = cfg["part_a"], cfg["topology"]
    os.makedirs(out_dir, exist_ok=True)
    dst, omni, kp = (read_snapshot(n, snap_dir) for n in ("kyoto_dst", "omni2_hourly", "gfz_kp"))
    storms_all = pd.read_parquet("data/derived/storms.parquet")
    controls = pd.read_parquet("data/derived/controls.parquet")
    daily = pd.read_parquet("data/derived/custody_proxies_daily.parquet")
    study = storms_all[storms_all["in_study"]].sort_values("t_min").reset_index(drop=True)
    model_storms = study[~study["anchor"].fillna("").isin(pc["exclude_from_models"])].reset_index(drop=True)

    X = predictors(study, dst, controls, omni, daily, tc)
    allp = PRIMARY + SECONDARY_SHARES
    series = {"primary": daily_series(daily, pc["population"], pc["max_altitude_km"]),
              "all_inactive_leo": daily_series(daily, "inactive", None),
              "active": daily_series(daily, "active", None)}
    Y = {k: outcomes(v, study, controls, allp, pc["peak_window_days"]) for k, v in series.items()}
    meta = study[["storm_id", "t_min", "severe", "dst_version_at_min", "anchor"]]
    base = meta.merge(X, on="storm_id").merge(Y["primary"], on="storm_id")
    base.to_csv(f"{out_dir}/storm_table.csv", index=False, lineterminator="\n")
    data = base[base["storm_id"].isin(model_storms["storm_id"])].reset_index(drop=True)

    results = []
    def add(label, res):
        res.insert(0, "analysis", label)
        results.append(res)
        v = verdict(res, pc)
        print(f"[part_a] {label:28s} n={int(res['n'].max()):3d}  H1 {v['H1']} ({v['H1_proxies']}/4)  H2 {v['H2']} ({v['H2_proxies']}/4)")
        return v

    verdicts = {"primary": add("primary", test_hypotheses(data, PRIMARY, pc))}
    # robustness 1-4 (conclusions are claimed only where they hold across these)
    verdicts["huber"] = add("huber", test_hypotheses(data, PRIMARY, pc, kind="huber", vif_switch=False))
    verdicts["exclude_severe"] = add("exclude_severe", test_hypotheses(data[~data["severe"]].reset_index(drop=True), PRIMARY, pc))
    d2000 = data["t_min"].dt.strftime("%Y-%m-%d").isin(["2000-04-06", "2000-04-07", "2000-07-15", "2000-07-16"])
    verdicts["exclude_2000"] = add("exclude_2000_04_07_and_07_16", test_hypotheses(data[~d2000].reset_index(drop=True), PRIMARY, pc))
    fin = (data["dst_version_at_min"] == "final") & (data["t_min"].dt.year <= 2020)
    verdicts["final_dst_only"] = add("final_dst_only", test_hypotheses(data[fin].reset_index(drop=True), PRIMARY, pc))
    # 5 secondary proxies
    add("secondary_proxies", test_hypotheses(data, SECONDARY_SHARES, pc))
    # 6 other populations
    for k in ["all_inactive_leo", "active"]:
        dk = meta.merge(X, on="storm_id").merge(Y[k], on="storm_id")
        dk = dk[dk["storm_id"].isin(model_storms["storm_id"])].reset_index(drop=True)
        add(k, test_hypotheses(dk, PRIMARY, pc))
    # 7 the originally planned control rule, storms with at least three controls under it
    c2 = copy.deepcopy(cfg)
    c2["controls"].update({"kp_max": 3.0, "f107_tolerance_sfu": 20, "season_tolerance_d": 45,
                           "year_tolerance": 1, "allow_reuse": False})
    orig = select_controls(storms_all, kp, omni, c2)
    have = orig.groupby("storm_id").size() if len(orig) else pd.Series(dtype=int)
    keep = have[have >= 3].index
    print(f"[part_a] original control rule: {len(keep)} storms with >= 3 controls")
    if len(keep):
        yo = outcomes(series["primary"], study[study["storm_id"].isin(keep)], orig, PRIMARY, pc["peak_window_days"])
        do = meta.merge(X, on="storm_id").merge(yo, on="storm_id")
        add("original_control_rule", test_hypotheses(do[do["storm_id"].isin(model_storms["storm_id"])].reset_index(drop=True), PRIMARY, pc))

    allres = pd.concat(results, ignore_index=True)
    allres.to_csv(f"{out_dir}/h1_h2_results.csv", index=False, lineterminator="\n")
    claim = {h: all(verdicts[k][h] == verdicts["primary"][h] for k in ["huber", "exclude_severe", "exclude_2000", "final_dst_only"])
             for h in ["H1", "H2"]}
    print(f"[part_a] primary verdict holds across robustness checks 1-4: {claim}")
    return allres, verdicts, claim
