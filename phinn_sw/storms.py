"""Storm catalogue, quiet-time controls and the development / held-out split (WP1)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def detect_storms(dst: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """A storm is a run of hours with Dst below the threshold; runs separated by fewer than
    merge_gap_h hours are one storm (multi-step storms). Returns one row per storm, 1957 onward."""
    sc = cfg["storms"]
    d = dst.set_index("time")["dst_nT"].asfreq("h")
    below = (d < sc["threshold_nT"]).to_numpy()
    idx = np.flatnonzero(below)
    if len(idx) == 0:
        return pd.DataFrame()
    breaks = np.flatnonzero(np.diff(idx) > sc["merge_gap_h"])
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    vals, times = d.to_numpy(), d.index
    ver = dst.set_index("time")["version"].reindex(d.index)
    rows = []
    for a, b in zip(starts, ends):
        seg = vals[a:b + 1]
        k = a + int(np.nanargmin(seg))
        tmin = times[k]
        w0, w1 = tmin - pd.Timedelta(days=sc["window_before_d"]), tmin + pd.Timedelta(days=sc["window_after_d"])
        win = d[w0:w1]
        rows.append({
            "storm_id": tmin.strftime("S%Y%m%d%H"),
            "t_min": tmin,
            "dst_min_nT": float(vals[k]),
            "first_below": times[a], "last_below": times[b],
            "hours_below_100": int(np.sum(vals[a:b + 1] < -100)),
            "integrated_dst_nTh": float(np.nansum(np.minimum(win.to_numpy(), 0.0))),
            "window_start": w0, "window_end": w1,
            "dst_version_at_min": ver.iloc[k],
            "window_complete": bool(win.notna().all() and len(win) == (w1 - w0) // pd.Timedelta(hours=1) + 1),
        })
    df = pd.DataFrame(rows)
    df["anchor"] = ""
    for an in sc.get("anchors", []):                 # named events below the threshold (e.g. Feb 2022)
        a0, a1 = pd.Timestamp(str(an["search_start"])), pd.Timestamp(str(an["search_end"])) + pd.Timedelta(days=1)
        seg = d[a0:a1 - pd.Timedelta(hours=1)]
        tmin = seg.idxmin()
        hit = (df["t_min"] - tmin).abs() <= pd.Timedelta(days=1)
        if hit.any():
            df.loc[hit, "anchor"] = an["name"]
            continue
        w0, w1 = tmin - pd.Timedelta(days=sc["window_before_d"]), tmin + pd.Timedelta(days=sc["window_after_d"])
        win = d[w0:w1]
        k = d.index.get_loc(tmin)
        df = pd.concat([df, pd.DataFrame([{
            "storm_id": tmin.strftime("A%Y%m%d%H"), "t_min": tmin, "dst_min_nT": float(seg.min()),
            "first_below": pd.NaT, "last_below": pd.NaT, "hours_below_100": int((win < -100).sum()),
            "integrated_dst_nTh": float(np.nansum(np.minimum(win.to_numpy(), 0.0))),
            "window_start": w0, "window_end": w1, "dst_version_at_min": ver.iloc[k],
            "window_complete": bool(win.notna().all()), "anchor": an["name"]}])], ignore_index=True)
    df = df.sort_values("t_min").reset_index(drop=True)
    df["severe"] = df["dst_min_nT"] < sc["severe_nT"]
    df["overlaps_previous"] = df["window_start"] < df["window_end"].shift(1)
    s0, s1 = pd.Timestamp(str(sc["study_start"])), pd.Timestamp(str(sc["study_end"])) + pd.Timedelta(days=1)
    df["in_study"] = (df["t_min"] >= s0) & (df["t_min"] < s1)
    sp = cfg["split"]
    yr = df["t_min"].dt.year
    df["split"] = np.where(~df["in_study"], "envelope_only",
                  np.where(yr.between(*sp["development"]), "development",
                  np.where(yr.between(*sp["held_out"]), "held_out", "envelope_only")))
    return df


def _daily(kp: pd.DataFrame, f107: pd.DataFrame):
    kp_max = kp.set_index("time")["Kp"].resample("D").max()
    f = f107.set_index("time")["f107_sfu"].resample("D").mean()
    return kp_max, f


def _circ_doy_diff(a, b):
    d = np.abs(a - b) % 365.25
    return np.minimum(d, 365.25 - d)


def select_controls(storms: pd.DataFrame, kp: pd.DataFrame, omni: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Five quiet 14-day windows per study storm: Kp <= 3 throughout, mean F10.7 within 20 sfu of
    the storm window's, window centre within 45 days of the storm's day of year, year within +/-1,
    and no overlap with any storm window or with another storm's controls. Closest F10.7 first."""
    cc = cfg["controls"]
    L = int(cc["length_d"])
    kp_max, f = _daily(kp, omni)
    days = kp_max.index
    quiet_day = (kp_max <= cc["kp_max"]).to_numpy()
    quiet_run = pd.Series(quiet_day.astype(int), index=days).rolling(L).sum().to_numpy() == L   # window ending here
    f_mean = f.reindex(days).rolling(L, min_periods=L - 2).mean().to_numpy()

    blocked = np.zeros(len(days), bool)
    for _, s in storms.iterrows():
        blocked |= (days >= s["window_start"].normalize()) & (days <= s["window_end"].normalize())
    blocked_run = pd.Series(blocked.astype(int), index=days).rolling(L).sum().to_numpy() > 0

    ends = np.flatnonzero(quiet_run & ~blocked_run & np.isfinite(f_mean))
    cand = pd.DataFrame({"end": days[ends], "f107": f_mean[ends]})
    cand["start"] = cand["end"] - pd.Timedelta(days=L - 1)
    centre = cand["start"] + pd.Timedelta(days=L / 2)
    cand["doy"], cand["year"] = centre.dt.dayofyear, centre.dt.year

    used = np.zeros(len(days), bool)
    day_pos = pd.Series(np.arange(len(days)), index=days)
    out = []
    study = storms[storms["in_study"]].sort_values("t_min")
    for _, s in study.iterrows():
        sw = f.loc[s["window_start"].normalize():s["window_end"].normalize()].mean()
        ok = cand[(np.abs(cand["f107"] - sw) <= cc["f107_tolerance_sfu"])
                  & (_circ_doy_diff(cand["doy"], s["t_min"].dayofyear) <= cc["season_tolerance_d"])
                  & (np.abs(cand["year"] - s["t_min"].year) <= cc["year_tolerance"])].copy()
        ok["dist"] = np.abs(ok["f107"] - sw)
        picked = 0
        mine = np.zeros(len(days), bool)          # this storm's own controls never overlap each other
        for _, c in ok.sort_values(["dist", "start"]).iterrows():
            a, b = day_pos[c["start"]], day_pos[c["end"]]
            if mine[a:b + 1].any() or (not cc.get("allow_reuse", False) and used[a:b + 1].any()):
                continue
            mine[a:b + 1] = True
            used[a:b + 1] = True
            out.append({"storm_id": s["storm_id"], "control_start": c["start"], "control_end": c["end"],
                        "control_f107": round(float(c["f107"]), 1), "storm_f107": round(float(sw), 1)})
            picked += 1
            if picked == cc["per_storm"]:
                break
        if picked < cc["per_storm"]:
            print(f"[controls] {s['storm_id']}: only {picked} of {cc['per_storm']} controls found")
    return pd.DataFrame(out)
