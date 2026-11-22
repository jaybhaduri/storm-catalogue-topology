"""WP2 custody proxies from public GP element sets (paper section 3.4).

Raw Space-Track records stay on the local disk. This module writes:
- pair-level working files (one row per consecutive element-set pair) to a local work folder, and
- small, aggregated tables to data/derived/ for the repository:
    custody_proxies_daily.parquet  one row per day x 50 km band x population
    custody_gates.parquet          quiet-time 99th-percentile correction gate per band

Proxies:
  P1 staleness   S_h: share of tracked objects whose newest element set is older than h hours (00 UT)
  P2 correction  position change at re-acquisition: the old set propagated with SGP4 to the new
                 epoch versus the new set, in radial / in-track / cross-track; z-score of |in-track|
                 against the object's own quiet-time distribution over the previous 30 days
  P3 gate        objects whose correction exceeds 5, 10, 25 km, or the band's quiet 99th percentile
  P4 decay       change in semi-major axis per day and in B*; re-entries from decay records
  M1 (interim)   orbit-raise events among active payloads per day (the drag-only M1 needs WP7 density)
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from sgp4.api import WGS72, Satrec

RE_KM = 6378.137
TWO_PI = 2 * np.pi
EPOCH0 = pd.Timestamp("1949-12-31")


# ------------------------------------------------------------------ loading

def load_sets(gp_dir: str, start: pd.Timestamp, end: pd.Timestamp, columns=None) -> pd.DataFrame:
    """Element sets with EPOCH in [start, end), deduplicated and sorted by object and epoch."""
    files = []
    for f in sorted(glob.glob(os.path.join(gp_dir, "gp_*.parquet"))):
        d = pd.Timestamp(os.path.basename(f)[3:11])
        if d + pd.Timedelta(days=11) >= start and d <= end:
            files.append(f)
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)
    df["EPOCH"] = pd.to_datetime(df["EPOCH"], format="mixed")
    df = df[(df["EPOCH"] >= start) & (df["EPOCH"] < end)]
    df = df.sort_values(["NORAD_CAT_ID", "EPOCH", "GP_ID"]).drop_duplicates(["NORAD_CAT_ID", "EPOCH"], keep="last")
    df["alt_km"] = df["SEMIMAJOR_AXIS"] - RE_KM
    df = df[(df["alt_km"] > 100) & (df["alt_km"] < 2000)]
    df["band_km"] = (np.floor(df["alt_km"] / 50) * 50).astype("int16")
    return df.reset_index(drop=True)


# ------------------------------------------------------------------ SGP4 pairs (P2)

def pair_corrections(df: pd.DataFrame, max_gap_days: float = 10.0) -> pd.DataFrame:
    """For each consecutive pair of sets of one object: propagate the older set to the newer epoch
    and compare with the newer set at its own epoch, in the newer set's RIC frame (km)."""
    nid = df["NORAD_CAT_ID"].to_numpy()
    same = nid[1:] == nid[:-1]
    gap = (df["EPOCH"].to_numpy()[1:] - df["EPOCH"].to_numpy()[:-1]) / np.timedelta64(1, "m")
    ok = np.flatnonzero(same & (gap > 0) & (gap <= max_gap_days * 1440))
    ep = ((df["EPOCH"] - EPOCH0) / pd.Timedelta(days=1)).to_numpy()
    bs, ec = df["BSTAR"].to_numpy(float), df["ECCENTRICITY"].to_numpy(float)
    ap, inc = np.radians(df["ARG_OF_PERICENTER"].to_numpy(float)), np.radians(df["INCLINATION"].to_numpy(float))
    ma, raan = np.radians(df["MEAN_ANOMALY"].to_numpy(float)), np.radians(df["RA_OF_ASC_NODE"].to_numpy(float))
    no = df["MEAN_MOTION"].to_numpy(float) * TWO_PI / 1440.0

    def _satrec(j):
        s = Satrec()
        s.sgp4init(WGS72, "i", int(nid[j]) % 100000, ep[j], bs[j], 0.0, 0.0, ec[j], ap[j], inc[j], ma[j], no[j], raan[j])
        return s

    out = np.full((len(ok), 4), np.nan)
    err = np.zeros(len(ok), dtype=np.int8)
    for k, i in enumerate(ok):
        try:
            old, new = _satrec(i), _satrec(i + 1)
            e1, r1, _ = old.sgp4_tsince(gap[i])
            e2, r2, v2 = new.sgp4_tsince(0.0)
        except Exception:
            err[k] = 9
            continue
        if e1 or e2:
            err[k] = e1 or e2
            continue
        r1, r2, v2 = np.asarray(r1), np.asarray(r2), np.asarray(v2)
        R = r2 / np.linalg.norm(r2)
        C = np.cross(r2, v2)
        C /= np.linalg.norm(C)
        I = np.cross(C, R)
        d = r1 - r2
        out[k] = (d @ R, d @ I, d @ C, np.linalg.norm(d))
    a, b = df.iloc[ok], df.iloc[ok + 1]
    return pd.DataFrame({
        "NORAD_CAT_ID": b["NORAD_CAT_ID"].to_numpy(), "epoch": b["EPOCH"].to_numpy(),
        "gap_h": gap[ok] / 60.0, "band_km": b["band_km"].to_numpy(), "object_type": b["OBJECT_TYPE"].to_numpy(),
        "dR_km": out[:, 0], "dI_km": out[:, 1], "dC_km": out[:, 2], "dpos_km": out[:, 3], "sgp4_error": err,
        "da_km_per_day": (b["SEMIMAJOR_AXIS"].to_numpy() - a["SEMIMAJOR_AXIS"].to_numpy()) / (gap[ok] / 1440.0),
        "da_km": b["SEMIMAJOR_AXIS"].to_numpy() - a["SEMIMAJOR_AXIS"].to_numpy(),
        "dbstar": b["BSTAR"].to_numpy() - a["BSTAR"].to_numpy(),
    })


# ------------------------------------------------------------------ populations

def label_population(pairs: pd.DataFrame, raise_km: float, active_window_d: int) -> pd.Series:
    """inactive: debris, rocket bodies, and payloads with no orbit raise in the previous year;
    active: payloads with a raise (a step up of more than raise_km) in the previous year; other: unknown."""
    pt = pairs["object_type"].fillna("UNKNOWN").str.upper()
    raised = (pt == "PAYLOAD") & (pairs["da_km"] > raise_km)
    last_raise = pairs["epoch"].where(raised)
    last_raise = last_raise.groupby(pairs["NORAD_CAT_ID"]).ffill()
    active = (pt == "PAYLOAD") & last_raise.notna() & ((pairs["epoch"] - last_raise) <= pd.Timedelta(days=active_window_d))
    pop = np.where(pt.isin(["DEBRIS", "ROCKET BODY"]) | ((pt == "PAYLOAD") & ~active), "inactive",
                   np.where(active, "active", "other"))
    return pd.Series(pop, index=pairs.index, name="population"), raised


# ------------------------------------------------------------------ z-scores against own quiet history

def add_zscores(pairs: pd.DataFrame, quiet_days: pd.DatetimeIndex, window: str = "30D", floor_km: float = 0.05):
    p = pairs.sort_values(["NORAD_CAT_ID", "epoch"]).copy()
    v = p["dI_km"].abs()
    p["absI_quiet"] = v.where(p["epoch"].dt.normalize().isin(quiet_days))
    g = p.set_index("epoch").groupby("NORAD_CAT_ID")["absI_quiet"]
    med = g.rolling(window, closed="left", min_periods=5).median().reset_index(level=0, drop=True)
    q75 = g.rolling(window, closed="left", min_periods=5).quantile(0.75).reset_index(level=0, drop=True)
    q25 = g.rolling(window, closed="left", min_periods=5).quantile(0.25).reset_index(level=0, drop=True)
    p["base_med"] = med.to_numpy()
    scale = np.maximum((q75 - q25).to_numpy() / 1.349, floor_km)
    p["z_I"] = (v.to_numpy() - p["base_med"].to_numpy()) / scale
    return p.drop(columns="absI_quiet")


# ------------------------------------------------------------------ staleness (P1)

def staleness_daily(df: pd.DataFrame, days: pd.DatetimeIndex, decay: pd.DataFrame, max_age_d: int = 30) -> pd.DataFrame:
    """At 00 UT each day: age of each object's newest set; tracked objects are those with age <= 30 d
    and not decayed. Returns counts per day, band and object type."""
    sets = df[["NORAD_CAT_ID", "EPOCH", "band_km", "OBJECT_TYPE"]].sort_values("EPOCH")
    grid = pd.DataFrame({"day": np.repeat(days.values, 1)}).sort_values("day")
    res = []
    for nid, s in sets.groupby("NORAD_CAT_ID", sort=False):
        e = s["EPOCH"].to_numpy()
        idx = np.searchsorted(e, days.values, side="right") - 1
        have = idx >= 0
        if not have.any():
            continue
        age_h = np.full(len(days), np.nan)
        age_h[have] = (days.values[have] - e[idx[have]]) / np.timedelta64(1, "h")
        keep = have & (age_h <= max_age_d * 24)
        if not keep.any():
            continue
        res.append(pd.DataFrame({"day": days.values[keep], "NORAD_CAT_ID": nid,
                                 "band_km": s["band_km"].to_numpy()[idx[keep]],
                                 "object_type": s["OBJECT_TYPE"].to_numpy()[idx[keep]], "age_h": age_h[keep]}))
    if not res:
        return pd.DataFrame()
    st = pd.concat(res, ignore_index=True)
    if len(decay):
        dd = decay.drop_duplicates("NORAD_CAT_ID").set_index("NORAD_CAT_ID")["DECAY_EPOCH"]
        dec = pd.to_datetime(st["NORAD_CAT_ID"].map(dd), errors="coerce")
        st = st[~(dec <= st["day"])]
    return st


# ------------------------------------------------------------------ driver

HIST_BINS = np.logspace(-3, 4, 701)          # 1 m to 10,000 km, for the quiet-time gate quantiles


def _populations(df: pd.DataFrame, raises: pd.DataFrame, window_d: int) -> pd.DataFrame:
    """Adds `population` to rows with day, NORAD_CAT_ID and object_type, from the raise history."""
    df = df.assign(day=df["day"].astype("datetime64[ns]")).sort_values("day").reset_index(drop=True)
    t = df["object_type"].fillna("UNKNOWN").astype(str).str.upper().to_numpy()
    active = np.zeros(len(df), bool)
    if len(raises):
        r = raises.rename(columns={"epoch": "day"})
        r = r.assign(day=r["day"].astype("datetime64[ns]")).sort_values("day")
        r = r.assign(last_raise=r["day"], NORAD_CAT_ID=r["NORAD_CAT_ID"].astype(df["NORAD_CAT_ID"].dtype))
        m = pd.merge_asof(df[["day", "NORAD_CAT_ID"]], r[["day", "NORAD_CAT_ID", "last_raise"]],
                          on="day", by="NORAD_CAT_ID", direction="backward")
        active = (t == "PAYLOAD") & m["last_raise"].notna().to_numpy() & \
                 ((m["day"] - m["last_raise"]) <= pd.Timedelta(days=window_d)).to_numpy()
    df["population"] = np.where(np.isin(t, ["DEBRIS", "ROCKET BODY"]) | ((t == "PAYLOAD") & ~active), "inactive",
                                np.where(active, "active", "other"))
    return df


def run(cfg: dict, st_dir: str, work_dir: str, out_dir: str = "data/derived", years=None):
    pc = cfg["proxies"]
    gp_dir = os.path.join(st_dir, "gp_history")
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    kp = pd.read_parquet(os.path.join(cfg["snapshot"]["dir"], "gfz_kp.parquet"))
    kp_day = kp.set_index("time")["Kp"].resample("D").max()
    quiet_days = kp_day.index[kp_day <= pc["quiet_kp_max"]]
    dpath = os.path.join(st_dir, "decay.parquet")
    decay = pd.read_parquet(dpath) if os.path.exists(dpath) else pd.DataFrame()
    if len(decay):
        if "PRECEDENCE" in decay:
            decay = decay[decay["PRECEDENCE"].astype(str).isin(["1", "2"])]
        decay = decay.assign(DECAY_EPOCH=pd.to_datetime(decay["DECAY_EPOCH"], errors="coerce", format="mixed"))
    start, end = pd.Timestamp(str(cfg["sources"]["spacetrack"]["start"])), pd.Timestamp(str(cfg["snapshot"]["end"]))
    years = list(years or range(start.year, end.year + 1))
    raises_all = pd.DataFrame({"NORAD_CAT_ID": pd.Series(dtype="int64"), "epoch": pd.Series(dtype="datetime64[ns]")})
    hist, daily = {}, []
    key = ["day", "band_km", "population"]
    for y in years:
        y0, y1 = pd.Timestamp(f"{y}-01-01"), min(pd.Timestamp(f"{y + 1}-01-01"), end + pd.Timedelta(days=1))
        # quarter by quarter, each with a 45-day lead-in for pairs and 30-day z baselines (bounded memory)
        chunks, n_sets = [], 0
        for q0 in pd.date_range(y0, y1, freq="QS")[:4]:
            q1 = min(q0 + pd.offsets.QuarterBegin(1, startingMonth=1), y1)
            if q0 >= y1:
                break
            qs = load_sets(gp_dir, q0 - pd.Timedelta(days=45), q1)
            if qs.empty:
                continue
            n_sets += int(((qs["EPOCH"] >= q0) & (qs["EPOCH"] < q1)).sum())
            qp = add_zscores(pair_corrections(qs, pc["max_pair_gap_days"]), quiet_days)
            chunks.append(qp[(qp["epoch"] >= q0) & (qp["epoch"] < q1)])
            del qs, qp
        if not chunks:
            print(f"[proxies] {y}: no element sets on disk, skipped")
            continue
        pairs = pd.concat(chunks, ignore_index=True)
        del chunks
        pairs["raise_event"] = (pairs["object_type"].fillna("").astype(str).str.upper() == "PAYLOAD") & \
                               (pairs["da_km"] > pc["raise_km"])
        raises_all = pd.concat([raises_all, pairs.loc[pairs["raise_event"], ["NORAD_CAT_ID", "epoch"]]]).drop_duplicates()
        pairs = pairs.assign(day=pairs["epoch"].dt.normalize())
        pairs = _populations(pairs, raises_all, pc["active_window_d"])
        sets = load_sets(gp_dir, y0 - pd.Timedelta(days=31), y1,
                         columns=["NORAD_CAT_ID", "EPOCH", "GP_ID", "OBJECT_TYPE", "SEMIMAJOR_AXIS"])
        pairs.to_parquet(os.path.join(work_dir, f"pairs_{y}.parquet"), index=False, compression="zstd")

        q = pairs[pairs["day"].isin(quiet_days) & pairs["dpos_km"].notna()]
        for b, v in q.groupby("band_km")["dpos_km"]:
            hist[b] = hist.get(b, 0) + np.histogram(v.clip(1e-3, 1e4), HIST_BINS)[0]

        pairs["absI"] = pairs["dI_km"].abs()
        g = pairs.groupby(key)
        agg = pd.DataFrame({
            "n_pairs": g.size(),
            "p2_absI_median_km": g["absI"].median(),
            "p2_absI_p90_km": g["absI"].quantile(0.9),
            "p2_dpos_median_km": g["dpos_km"].median(),
            "p2_z_median": g["z_I"].median(),
            "p4_da_median_km_per_day": g["da_km_per_day"].median(),
            "p4_dbstar_median": g["dbstar"].median(),
            "m1_raise_events": g["raise_event"].sum(),
        })
        for gate in pc["gates_km"]:
            agg[f"p3_objects_over_{gate}km"] = pairs[pairs["dpos_km"] > gate].groupby(key)["NORAD_CAT_ID"].nunique()
        agg["p2_objects_z_over_3"] = pairs[pairs["z_I"] > 3].groupby(key)["NORAD_CAT_ID"].nunique()

        days = pd.date_range(y0, y1 - pd.Timedelta(days=1), freq="D")
        st = staleness_daily(sets, days, decay, pc["max_age_d"])
        if len(st):
            st = _populations(st, raises_all, pc["active_window_d"])
            for h in pc["stale_hours"]:
                st[f"p1_stale_over_{h}h"] = st["age_h"] > h
            gs = st.groupby(key)
            agg = agg.join(gs[[f"p1_stale_over_{h}h" for h in pc["stale_hours"]]].sum()
                           .join(gs.size().rename("n_tracked")), how="outer")
        if len(decay):
            dy = decay[(decay["DECAY_EPOCH"] >= y0) & (decay["DECAY_EPOCH"] < y1)]
            last = sets.sort_values("EPOCH").drop_duplicates("NORAD_CAT_ID", keep="last").set_index("NORAD_CAT_ID")
            dy = dy[dy["NORAD_CAT_ID"].isin(last.index)].drop_duplicates("NORAD_CAT_ID")
            if len(dy):
                re = pd.DataFrame({"day": dy["DECAY_EPOCH"].dt.normalize().to_numpy(),
                                   "NORAD_CAT_ID": dy["NORAD_CAT_ID"].to_numpy(),
                                   "band_km": last.loc[dy["NORAD_CAT_ID"], "band_km"].to_numpy(),
                                   "object_type": last.loc[dy["NORAD_CAT_ID"], "OBJECT_TYPE"].to_numpy()})
                re = _populations(re, raises_all, pc["active_window_d"])
                agg = agg.join(re.groupby(key).size().rename("p4_reentries"), how="outer")
        daily.append(agg.reset_index())
        print(f"[proxies] {y}: {n_sets:,} element sets, {len(pairs):,} pairs, "
              f"{int((pairs['sgp4_error'] > 0).sum()):,} SGP4 errors")

    gates = pd.DataFrame([{"band_km": int(b), "n_quiet_pairs": int(h.sum()),
                           "quiet_q99_km": float(HIST_BINS[1:][np.searchsorted(np.cumsum(h) / h.sum(), pc["quiet_quantile"])])}
                          for b, h in sorted(hist.items()) if h.sum() > 0])
    out = pd.concat(daily, ignore_index=True) if daily else pd.DataFrame(columns=key)
    if len(gates) and len(out):
        qcol = []
        for y in years:
            pth = os.path.join(work_dir, f"pairs_{y}.parquet")
            if os.path.exists(pth):
                p = pd.read_parquet(pth, columns=key + ["NORAD_CAT_ID", "dpos_km"]).merge(
                    gates[["band_km", "quiet_q99_km"]], on="band_km", how="left")
                qcol.append(p[p["dpos_km"] > p["quiet_q99_km"]].groupby(key)["NORAD_CAT_ID"].nunique()
                            .rename("p3_objects_over_quiet_q99").reset_index())
        if qcol:
            out = out.merge(pd.concat(qcol), on=key, how="left")
    count_cols = [c for c in out.columns if c.startswith(("n_", "p1_", "p3_", "p4_reentries", "m1_", "p2_objects"))]
    out[count_cols] = out[count_cols].fillna(0).astype("int32")
    out = out.sort_values(key).reset_index(drop=True)
    out.to_parquet(os.path.join(out_dir, "custody_proxies_daily.parquet"), index=False, compression="zstd")
    gates.to_parquet(os.path.join(out_dir, "custody_gates.parquet"), index=False)
    gates.to_csv(os.path.join(out_dir, "custody_gates.csv"), index=False, lineterminator="\n")
    print(f"[proxies] wrote {len(out):,} daily rows and {len(gates)} band gates")
    return out, gates
