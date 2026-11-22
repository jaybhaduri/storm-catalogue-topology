"""Registration 3 design work on Part A's intense storms (already seen; exploratory, disclosed).

Picks the stretch shape bet. Outcome: like-for-like log ratio (Part A exploratory table).
Variants, each added to Part A's M1 (Dst summaries + covariates), ridge (penalty 1) on standardised
predictors, leave-one-storm-out:
  L   heating level: integrated Hp30 and integrated DTC over t0-1 d .. t0+3 d
  D   Dst Betti features over the focused window t0-1 d .. t0+3 d
  H   Hp30 Betti features over the focused window
  LH  level + Hp30 Betti
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from . import part_a as A
from .snapshot import read_snapshot
from .topology import storm_features

HP_TC = {"d": 3, "tau_h": 3, "window_h": 24, "step_h": 3, "grid_min_nT": 0.0, "grid_max_nT": 6.0,
         "grid_step_nT": 0.1, "track_scale_nT": 1.0}


def focused_features(storms: pd.DataFrame, dst, hp, dtc, tc) -> pd.DataFrame:
    sd = dst.set_index("time")["dst_nT"].asfreq("h")
    sh = hp.set_index("time")["Hp30"].resample("h").max()
    sc = dtc.set_index("time")["DTC_K"].resample("h").mean()
    rows = []
    for _, s in storms.iterrows():
        a, b = s["t_min"] - pd.Timedelta(days=1), s["t_min"] + pd.Timedelta(days=3)
        xd = sd[a:b].interpolate(limit_direction="both").to_numpy()
        xh = sh[a:b].interpolate(limit_direction="both").to_numpy()
        xc = sc[a:b].interpolate(limit_direction="both").to_numpy()
        fd, fh = storm_features(xd, tc), storm_features(xh, HP_TC)
        rows.append({"storm_id": s["storm_id"],
                     "hp30_integral": float(np.nansum(xh)), "dtc_integral": float(np.nansum(xc - np.nanmin(xc))),
                     **{f"dst_{k}": fd[k] for k in A.BETTI}, **{f"hp_{k}": fh[k] for k in A.BETTI}})
    return pd.DataFrame(rows)


def loso_mse(df, y, cols, pen=1.0):
    from sklearn.linear_model import Ridge
    e = np.empty(len(df))
    for i in range(len(df)):
        tr = df.drop(index=i)
        mu, sd = tr[cols].mean(), tr[cols].std(ddof=0).replace(0, 1)
        m = Ridge(alpha=pen).fit((tr[cols] - mu) / sd, tr[y])
        e[i] = (m.predict(((df.loc[[i], cols] - mu) / sd))[0] - df.loc[i, y]) ** 2
    return e


def main():
    cfg = yaml.safe_load(open("config.yaml"))
    dst, hp, dtc = (read_snapshot(n, "data/snapshots") for n in ("kyoto_dst", "gfz_hp30", "set_dtc"))
    tab = pd.read_csv("data/derived/part_a/storm_table.csv", parse_dates=["t_min"])
    tab = tab[tab["anchor"].isna()].drop(columns=A.PRIMARY)
    lfl = pd.read_csv("data/derived/part_a/explore_like_for_like.csv")[["storm_id"] + A.PRIMARY]
    f = focused_features(tab, dst, hp, dtc, cfg["topology"])
    df = tab.merge(lfl, on="storm_id").merge(f, on="storm_id").reset_index(drop=True)
    base = A.DST + A.COV
    sets = {"M1": base, "L": base + ["hp30_integral", "dtc_integral"],
            "D": base + [f"dst_{k}" for k in A.BETTI], "H": base + [f"hp_{k}" for k in A.BETTI],
            "LH": base + ["hp30_integral", "dtc_integral"] + [f"hp_{k}" for k in A.BETTI],
            "LD": base + ["hp30_integral", "dtc_integral"] + [f"dst_{k}" for k in A.BETTI]}
    rows = []
    for y in ["P2", "P3"]:
        e = {k: loso_mse(df, y, v) for k, v in sets.items()}
        for k in sets:
            rows.append({"outcome": y, "model": k, "mse": e[k].mean(),
                         "vs_M1": 1 - e[k].mean() / e["M1"].mean(), "vs_L": 1 - e[k].mean() / e["L"].mean(),
                         "share_storms_better_than_L": float((e[k] < e["L"]).mean())})
    r = pd.DataFrame(rows)
    r.to_csv("data/derived/reg3/dev_shape_variants.csv", index=False, lineterminator="\n")
    print(r.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
