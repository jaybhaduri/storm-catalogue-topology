"""Registration 3 sample: isolated moderate storms 2000-2025 and their quiet-time controls.

Built from Dst, Kp and F10.7 only; no custody outcome is read here.
Moderate storm: a run of hours with Dst below -50 nT (runs < 24 h apart merged) whose minimum stays
above -100 nT. Isolated: no other storm (minimum below -50 nT) has its minimum within 10 days before or
10 days after this one's, so neither its peak window (days 0..+10) nor its pre-storm days overlap any
storm studied in Part A. The February 2022 Starlink anchor (seen in Part A) is excluded.
Controls: rule K (config.yaml `controls`), blocking every storm window (minimum below -50 nT).
"""
from __future__ import annotations

import copy

import pandas as pd
import yaml

from .snapshot import read_snapshot
from .storms import detect_storms, select_controls

ISOLATION_D = 10


def build(cfg, snap="data/snapshots", out="data/derived/reg3"):
    c = copy.deepcopy(cfg)
    c["storms"]["threshold_nT"], c["storms"]["anchors"] = -50, []
    dst, kp, omni = (read_snapshot(n, snap) for n in ("kyoto_dst", "gfz_kp", "omni2_hourly"))
    s = detect_storms(dst, c)
    study = s[s["in_study"]]
    t = study["t_min"]
    mod = study[study["dst_min_nT"] > -100].copy()
    iso = mod.apply(lambda r: ((t >= r.t_min - pd.Timedelta(days=ISOLATION_D)) &
                               (t <= r.t_min + pd.Timedelta(days=ISOLATION_D))).sum() == 1, axis=1)
    feb22 = mod["t_min"].between("2022-02-01", "2022-02-06")
    mod = mod[iso & ~feb22].copy()
    mod["split"] = "moderate_2000_2025"
    blk = s.copy()
    blk["in_study"] = blk["storm_id"].isin(mod["storm_id"])
    ctl = select_controls(blk, kp, omni, c)
    mod.to_csv(f"{out}/moderate_storms.csv", index=False, lineterminator="\n")
    ctl.to_csv(f"{out}/moderate_controls.csv", index=False, lineterminator="\n")
    n = ctl.groupby("storm_id").size().reindex(mod["storm_id"]).fillna(0)
    print(f"[reg3] {len(mod)} isolated moderate storms; controls: {int((n == 5).sum())} with 5, "
          f"{int(((n >= 3) & (n < 5)).sum())} with 3-4, {int((n < 3).sum())} with fewer than 3")
    return mod, ctl


if __name__ == "__main__":
    build(yaml.safe_load(open("config.yaml")))
