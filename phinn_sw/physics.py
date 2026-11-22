"""WP5: the physics gate (paper section 3.7).

Gate 1  plausible drivers: generalised Pareto tails of V, |B|, southward Bz and n from OMNI (1995-),
        declustered; bound = 1-in-N-year return level at its upper 95% bootstrap limit.
Gate 2  magnetospheric response: Dst from the O'Brien-McPherron (2000) model, calibrated per storm.
        (Temerin-Li 2002, the second model, is a follow-up.)
Gate 3  the 1957-present envelope: storm descriptor vectors and an in-record / beyond-record label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import least_squares

# ------------------------------------------------------------------ Gate 1: extreme-value bounds

def decluster(x: pd.Series, u: float, run_h: int = 48) -> np.ndarray:
    """Cluster maxima of exceedances over u; clusters end after run_h hours below u."""
    v = x.to_numpy()
    above = np.flatnonzero(v > u)
    if len(above) == 0:
        return np.array([])
    breaks = np.flatnonzero(np.diff(above) > run_h)
    starts, ends = np.r_[0, breaks + 1], np.r_[breaks, len(above) - 1]
    return np.array([v[above[a:b + 1]].max() for a, b in zip(starts, ends)])


def gpd_return_level(peaks: np.ndarray, u: float, years: float, rate_per_year: float):
    xi, _, sigma = stats.genpareto.fit(peaks - u, floc=0)
    m = years * rate_per_year
    return u + (sigma / xi) * (m ** xi - 1) if abs(xi) > 1e-6 else u + sigma * np.log(m), xi, sigma


def driver_bounds(omni: pd.DataFrame, gc: dict, seed: int = 0) -> pd.DataFrame:
    """Return levels (point and upper 95% bootstrap limit) for each driver and each return period."""
    o = omni[omni["time"] >= pd.Timestamp(str(gc["omni_start"]))].set_index("time")
    drivers = {"V_kms": o["V_kms"], "B_nT": o["B_nT"], "Bs_nT": (-o["Bz_gsm_nT"]).clip(lower=0), "n_cc": o["n_cc"]}
    yrs = (o.index.max() - o.index.min()) / pd.Timedelta(days=365.25)
    rng = np.random.default_rng(seed)
    rows = []
    for name, x in drivers.items():
        x = x.dropna()
        u = float(x.quantile(gc["threshold_quantile"]))
        peaks = decluster(x, u, gc["decluster_h"])
        rate = len(peaks) / yrs
        for N in gc["return_years"]:
            rl, xi, sigma = gpd_return_level(peaks, u, N, rate)
            boots = []
            for _ in range(int(gc["n_boot"])):
                bp = rng.choice(peaks, size=len(peaks), replace=True)
                try:
                    boots.append(gpd_return_level(bp, u, N, rate)[0])
                except Exception:
                    continue
            rows.append({"driver": name, "return_years": N, "threshold": u, "n_clusters": len(peaks),
                         "shape_xi": xi, "scale_sigma": sigma, "return_level": rl,
                         "upper95": float(np.nanpercentile(boots, 97.5)), "observed_max": float(x.max())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ Gate 2: O'Brien-McPherron Dst

OBM = {"a": -4.4, "Ec": 0.49, "b": 7.26, "c": 11.0}   # nT/h per mV/m, mV/m, nT nPa^-1/2, nT


def obm_dst(V, Bz, Pdyn, dst0: float, a_scale: float = 1.0, tau_scale: float = 1.0, dt_h: float = 1.0):
    """Hourly Dst from solar-wind drivers (O'Brien and McPherron, 2000).
    dDst*/dt = Q - Dst*/tau, Q = a (V Bs - Ec) for V Bs > Ec, tau = 2.40 exp(9.74 / (4.69 + V Bs)) h,
    Dst* = Dst - b sqrt(Pdyn) + c. V in km/s, Bz in nT, Pdyn in nPa; V Bs in mV/m."""
    V, Bz, P = (np.asarray(v, float) for v in (V, Bz, Pdyn))
    VBs = V * np.clip(-Bz, 0, None) * 1e-3
    Q = np.where(VBs > OBM["Ec"], a_scale * OBM["a"] * (VBs - OBM["Ec"]), 0.0)
    tau = tau_scale * 2.40 * np.exp(9.74 / (4.69 + VBs))
    star = np.empty(len(V))
    s = dst0 - OBM["b"] * np.sqrt(max(P[0], 0)) + OBM["c"]
    for i in range(len(V)):
        s = s + dt_h * (Q[i] - s / tau[i])
        star[i] = s
    return star + OBM["b"] * np.sqrt(np.clip(P, 0, None)) - OBM["c"]


def calibrate_obm(V, Bz, Pdyn, dst_obs):
    """Per-storm calibration (paper 3.7): scale the injection and decay terms to fit observed Dst."""
    ok = np.isfinite(dst_obs)
    def res(p):
        return (obm_dst(V, Bz, Pdyn, dst_obs[0], p[0], p[1]) - dst_obs)[ok]
    fit = least_squares(res, x0=[1.0, 1.0], bounds=([0.3, 0.3], [3.0, 3.0]))
    model = obm_dst(V, Bz, Pdyn, dst_obs[0], *fit.x)
    rmse = float(np.sqrt(np.mean((model - dst_obs)[ok] ** 2)))
    r = float(np.corrcoef(model[ok], dst_obs[ok])[0, 1])
    return {"a_scale": float(fit.x[0]), "tau_scale": float(fit.x[1]), "rmse_nT": rmse, "r": r}, model


def storm_drivers(omni: pd.DataFrame, t0, t1) -> pd.DataFrame:
    w = omni.set_index("time").loc[t0:t1, ["V_kms", "Bz_gsm_nT", "Pdyn_nPa", "n_cc"]]
    return w.interpolate(limit=6, limit_direction="both")


# ------------------------------------------------------------------ Gate 3: the 1957-present envelope

def descriptors(x: np.ndarray, i_min: int) -> dict:
    """Storm descriptors from an hourly Dst window with its minimum at index i_min."""
    dmin = float(x[i_min])
    j = i_min
    while j > 0 and x[j - 1] < -50:               # main phase: from the last time above -50 nT
        j -= 1
    main_h = max(i_min - j, 1)
    k = i_min
    while k < len(x) - 1 and x[k] < dmin / np.e:  # recovery e-folding time
        k += 1
    return {"dst_min_nT": dmin, "fall_rate_nT_per_h": (x[j] - dmin) / main_h, "main_phase_h": main_h,
            "hours_below_100": int(np.sum(x < -100)), "recovery_efold_h": k - i_min,
            "integrated_dst_nTh": float(np.sum(np.minimum(x, 0)))}


def storm_descriptors(dst: pd.DataFrame, storms: pd.DataFrame) -> pd.DataFrame:
    s = dst.set_index("time")["dst_nT"].asfreq("h")
    rows = []
    for _, st in storms.iterrows():
        w = s[st["window_start"]:st["window_end"]]
        x = w.interpolate(limit_direction="both").to_numpy()
        if np.isnan(x).all():
            continue
        i_min = int(w.index.get_loc(st["t_min"]))       # this storm's own minimum, not a neighbour's
        rows.append({"storm_id": st["storm_id"], "t_min": st["t_min"], **descriptors(x, i_min)})
    return pd.DataFrame(rows)


DESC_COLS = ["fall_rate_nT_per_h", "main_phase_h", "hours_below_100", "recovery_efold_h", "integrated_dst_nTh"]


def envelope_label(desc_obs: pd.DataFrame, cand: dict, ec: dict) -> str:
    """in-record if the candidate's descriptors sit inside the support of observed storms of
    comparable intensity (kernel density above a floor); beyond-record otherwise; Carrington-class
    for minima in the reconstructed Carrington range."""
    m = cand["dst_min_nT"]
    if ec["carrington_range"][0] >= m >= ec["carrington_range"][1] or m < ec["carrington_range"][1]:
        return "carrington_class"
    if m < ec["record_min_nT"]:
        return "beyond_record"
    near = desc_obs[np.abs(desc_obs["dst_min_nT"] - m) <= ec["intensity_tolerance"] * abs(m)]
    if len(near) < 8:                              # too few comparable storms for a 5-D density
        return "beyond_record"
    z = lambda d: np.log1p(np.abs(np.asarray(d[DESC_COLS], float)))
    X = z(near).T
    kde = stats.gaussian_kde(X)
    # leave-one-out densities of the observed storms: a storm's own kernel must not set its floor
    obs_d = np.array([stats.gaussian_kde(np.delete(X, i, axis=1))(X[:, i:i + 1])[0] for i in range(X.shape[1])])
    floor = np.quantile(obs_d, ec["density_floor_quantile"])
    return "in_record" if kde(z(pd.DataFrame([cand]))[0][:, None])[0] >= floor else "beyond_record"
