"""WP3: topological fingerprints of Dst (paper section 3.5) and the two artefact controls.

Storm side: hourly Dst is delay-embedded (d = 3, tau = 3 h); Vietoris-Rips persistence is computed in
24-hour windows stepped every 3 hours; Betti-0 and Betti-1 curves are read on an absolute filtration
grid of 0-60 nT (edge length, i.e. diameter), so storm severity stays inside the fingerprint. The
tracking series is beta0 at 15 nT.

Controls (H2):
- shuffle test: permuting the hours inside a storm window leaves every level statistic unchanged; a
  change in the Betti curves shows they encode the order of events.
- IAAFT surrogates keep the spectrum and the amplitude distribution but destroy nonlinear structure;
  Betti features must separate real storms from their surrogates.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from ripser import ripser


def delay_embed(x: np.ndarray, d: int, tau: int) -> np.ndarray:
    n = len(x) - (d - 1) * tau
    return np.stack([x[j * tau: j * tau + n] for j in range(d)], axis=1) if n > 0 else np.empty((0, d))


def betti_curves(points: np.ndarray, grid: np.ndarray):
    """Betti-0 and Betti-1 counts at each filtration value (edge length) on the grid."""
    dg = ripser(points, maxdim=1)["dgms"]
    out = []
    for k in (0, 1):
        b, dd = dg[k][:, 0], dg[k][:, 1]
        out.append(((b[:, None] <= grid[None, :]) & (dd[:, None] > grid[None, :])).sum(axis=0))
    return out[0], out[1]


def _grid(tc):
    return np.arange(tc["grid_min_nT"], tc["grid_max_nT"] + tc["grid_step_nT"] / 2, tc["grid_step_nT"], dtype=float)


def window_features(series: np.ndarray, ends: np.ndarray, tc: dict):
    """Betti curves for 24-hour windows ending at each index in `ends` (hourly series)."""
    W, d, tau = int(tc["window_h"]), int(tc["d"]), int(tc["tau_h"])
    grid = _grid(tc)
    b0 = np.full((len(ends), len(grid)), -1, dtype=np.int16)
    b1 = np.full((len(ends), len(grid)), -1, dtype=np.int16)
    for i, e in enumerate(ends):
        seg = series[e - W + 1: e + 1]
        if e - W + 1 < 0 or np.isnan(seg).any():
            continue
        b0[i], b1[i] = betti_curves(delay_embed(seg, d, tau), grid)
    return b0, b1, grid


def dst_record(dst: pd.DataFrame, tc: dict) -> pd.DataFrame:
    """Betti fingerprints of Dst for the whole record, one row per window end (every step_h hours)."""
    s = dst.set_index("time")["dst_nT"].asfreq("h")
    x = s.to_numpy(dtype=float)
    ends = np.arange(int(tc["window_h"]) - 1, len(x), int(tc["step_h"]))
    b0, b1, grid = window_features(x, ends, tc)
    ok = b0[:, 0] >= 0
    k = int(np.argmin(np.abs(grid - tc["track_scale_nT"])))
    df = pd.DataFrame({"time": s.index[ends][ok], "beta0_track": b0[ok, k], "beta1_track": b1[ok, k]})
    cols = {f"b0_{int(g):02d}": b0[ok, j] for j, g in enumerate(grid)}
    cols.update({f"b1_{int(g):02d}": b1[ok, j] for j, g in enumerate(grid)})
    return pd.concat([df, pd.DataFrame(cols)], axis=1)


def storm_features(x: np.ndarray, tc: dict) -> dict:
    """Betti summary of one storm window (hourly Dst array): curves over all its 3-hourly windows."""
    ends = np.arange(int(tc["window_h"]) - 1, len(x), int(tc["step_h"]))
    b0, b1, grid = window_features(x, ends, tc)
    ok = b0[:, 0] >= 0
    b0, b1 = b0[ok].astype(float), b1[ok].astype(float)
    k = int(np.argmin(np.abs(grid - tc["track_scale_nT"])))
    return {"b0_curves": b0, "b1_curves": b1,
            "beta0_track_max": float(b0[:, k].max()), "beta0_track_mean": float(b0[:, k].mean()),
            "beta1_track_max": float(b1[:, k].max()), "beta1_total": float(b1.sum() / len(b1)),
            "b0_area": float(b0.sum() / len(b0))}


def level_stats(x: np.ndarray) -> dict:
    return {"min": float(np.nanmin(x)), "mean": float(np.nanmean(x)), "std": float(np.nanstd(x)),
            "hours_below_100": int(np.sum(x < -100)), "integrated": float(np.nansum(np.minimum(x, 0)))}


def iaaft(x: np.ndarray, rng: np.random.Generator, n_iter: int = 200) -> np.ndarray:
    """Iterative amplitude-adjusted Fourier transform surrogate (Schreiber and Schmitz, 1996)."""
    amp = np.abs(np.fft.rfft(x))
    sorted_x = np.sort(x)
    y = rng.permutation(x)
    for _ in range(n_iter):
        ph = np.angle(np.fft.rfft(y))
        y = np.fft.irfft(amp * np.exp(1j * ph), n=len(x))
        y = sorted_x[np.argsort(np.argsort(y))]
    return y


def curve_distance(a: dict, b: dict) -> float:
    """Mean absolute difference of the Betti-0 and Betti-1 curves across all windows (counts)."""
    n = min(len(a["b0_curves"]), len(b["b0_curves"]))
    return float(np.abs(a["b0_curves"][:n] - b["b0_curves"][:n]).mean()
                 + np.abs(a["b1_curves"][:n] - b["b1_curves"][:n]).mean())


def storm_controls(dst: pd.DataFrame, storms: pd.DataFrame, tc: dict, seed: int = 0) -> pd.DataFrame:
    """Shuffle test and IAAFT surrogate test for every study storm."""
    s = dst.set_index("time")["dst_nT"].asfreq("h")
    rng = np.random.default_rng(seed)
    rows = []
    for _, st in storms[storms["in_study"]].iterrows():
        x = s[st["window_start"]:st["window_end"]].to_numpy(dtype=float)
        if np.isnan(x).any():
            x = pd.Series(x).interpolate(limit_direction="both").to_numpy()
        real = storm_features(x, tc)
        xs = rng.permutation(x)
        shuf = storm_features(xs, tc)
        lv_r, lv_s = level_stats(x), level_stats(xs)
        stat = lambda f: f["b0_area"] + f["beta1_total"]          # one scalar Betti statistic per storm
        sur = [storm_features(iaaft(x, rng), tc) for _ in range(int(tc["n_surrogates"]))]
        sur_stat = np.array([stat(f) for f in sur])
        sur_dist = np.array([curve_distance(real, f) for f in sur])
        r = stat(real)
        rank = int((sur_stat < r).sum())                          # 0..n: position of the real storm
        n = len(sur_stat)
        rows.append({
            "storm_id": st["storm_id"], "dst_min_nT": st["dst_min_nT"], "split": st["split"],
            "level_stats_unchanged_by_shuffle": all(np.isclose(lv_r[k], lv_s[k]) for k in lv_r),
            "shuffle_curve_distance": curve_distance(real, shuf),
            "real_betti_stat": r, "surrogate_stat_mean": float(sur_stat.mean()),
            "surrogate_stat_sd": float(sur_stat.std()),
            "surrogate_rank": rank, "n_surrogates": n,
            "p_two_sided": float(min(1.0, 2 * min(rank + 1, n - rank + 1) / (n + 1))),
            "mean_curve_distance_to_surrogates": float(sur_dist.mean()),
            "beta0_track_max": real["beta0_track_max"], "beta1_track_max": real["beta1_track_max"],
        })
    return pd.DataFrame(rows)
