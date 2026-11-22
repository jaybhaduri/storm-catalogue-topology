"""Command line for the storm and catalogue-custody analysis.

    python -m phinn_sw pull [--only kyoto_dst omni2_hourly gfz_kp gfz_hp30 set]
    python -m phinn_sw storms
    python -m phinn_sw spacetrack --out data/spacetrack     # needs SPACETRACK_USER / SPACETRACK_PASSWORD
"""
from __future__ import annotations

import argparse
import os

import yaml

from . import sources
from .snapshot import read_snapshot, write_snapshot
from .storms import detect_storms, select_controls

CITE = {
    "kyoto_dst": {"source": "WDC for Geomagnetism, Kyoto (HAPI hour_dst, best available)",
                  "citation": "World Data Center for Geomagnetism, Kyoto, Nose et al. (2015), doi:10.17593/14515-74000",
                  "note": "version column: final, provisional or realtime, per hour"},
    "omni2_hourly": {"source": "NASA SPDF OMNI2 hourly (low_res_omni)",
                     "citation": "King and Papitashvili (2005), JGR 110, A02104"},
    "gfz_kp": {"source": "GFZ Potsdam Kp web service", "license": "CC BY 4.0",
               "citation": "Matzka et al. (2021), doi:10.5880/Kp.0001"},
    "gfz_hp30": {"source": "GFZ Potsdam Hp30 web service", "license": "CC BY 4.0",
                 "citation": "Yamazaki et al. (2022), doi:10.5880/Hpo.0002"},
    "set_solfsmy": {"source": "Space Environment Technologies SOLFSMY.TXT (JB2008 solar indices)"},
    "set_dtc": {"source": "Space Environment Technologies DTCFILE.TXT (JB2008 hourly DTC)"},
}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="phinn_sw")
    ap.add_argument("--config", default="config.yaml")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pull")
    p.add_argument("--only", nargs="*")
    sub.add_parser("storms")
    p = sub.add_parser("spacetrack")
    p.add_argument("--out", default="data/spacetrack")
    sub.add_parser("topology", help="WP3 Dst Betti fingerprints, shuffle test and IAAFT surrogates")
    sub.add_parser("physics", help="WP5 gate 1 bounds, gate 2 OBM calibration, gate 3 envelope")
    sub.add_parser("reg3", help="Registration 3 confirmatory tests (run only after Registration 3 is filed)")
    sub.add_parser("part_a", help="Registration 1 analysis: H1 and H2 with robustness checks")
    p = sub.add_parser("proxies", help="WP2 custody proxies from downloaded Space-Track data")
    p.add_argument("--spacetrack", default="data/spacetrack", help="folder the spacetrack command wrote")
    p.add_argument("--work", default="data/work", help="local folder for pair-level files (not committed)")
    p.add_argument("--years", nargs="*", type=int)
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(open(a.config))
    snap = cfg["snapshot"]["dir"]

    if a.cmd == "pull":
        want = set(a.only or ["kyoto_dst", "omni2_hourly", "gfz_kp", "gfz_hp30", "set"])
        if "kyoto_dst" in want:
            write_snapshot(sources.pull_kyoto_dst(cfg), "kyoto_dst", snap, CITE["kyoto_dst"])
        if "omni2_hourly" in want:
            write_snapshot(sources.pull_omni2_hourly(cfg), "omni2_hourly", snap, CITE["omni2_hourly"])
        if "gfz_kp" in want:
            write_snapshot(sources.pull_gfz(cfg, "Kp", cfg["sources"]["gfz"]["kp_start"]), "gfz_kp", snap, CITE["gfz_kp"])
        if "gfz_hp30" in want:
            write_snapshot(sources.pull_gfz(cfg, "Hp30", cfg["sources"]["gfz"]["hp30_start"]), "gfz_hp30", snap, CITE["gfz_hp30"])
        if "set" in want:
            sol, dtc = sources.pull_set(cfg)
            write_snapshot(sol, "set_solfsmy", snap, CITE["set_solfsmy"])
            write_snapshot(dtc, "set_dtc", snap, CITE["set_dtc"])

    elif a.cmd == "storms":
        dst, kp, omni = read_snapshot("kyoto_dst", snap), read_snapshot("gfz_kp", snap), read_snapshot("omni2_hourly", snap)
        storms = detect_storms(dst, cfg)
        controls = select_controls(storms, kp, omni, cfg)
        os.makedirs("data/derived", exist_ok=True)
        storms.to_parquet("data/derived/storms.parquet", index=False)
        storms.to_csv("data/derived/storms.csv", index=False, lineterminator="\n")
        controls.to_parquet("data/derived/controls.parquet", index=False)
        controls.to_csv("data/derived/controls.csv", index=False, lineterminator="\n")
        st = storms[storms["in_study"]]
        print(f"[storms] {len(storms)} storms since 1957; {len(st)} in 2000-2025 "
              f"({int(st['severe'].sum())} severe); development {int((st['split'] == 'development').sum())}, "
              f"held out {int((st['split'] == 'held_out').sum())}; {len(controls)} controls")

    elif a.cmd == "spacetrack":
        from .spacetrack import pull_gp_history
        pull_gp_history(cfg, a.out)

    elif a.cmd == "topology":
        import pandas as pd
        from .topology import dst_record, storm_controls
        dst = read_snapshot("kyoto_dst", snap)
        rec = dst_record(dst, cfg["topology"])
        os.makedirs("data/derived", exist_ok=True)
        rec.to_parquet("data/derived/dst_betti.parquet", index=False, compression="zstd")
        print(f"[topology] Dst fingerprints: {len(rec):,} windows, {rec['time'].min()} to {rec['time'].max()}")
        storms = pd.read_parquet("data/derived/storms.parquet")
        ctl = storm_controls(dst, storms, cfg["topology"])
        ctl.to_csv("data/derived/topology_controls.csv", index=False, lineterminator="\n")
        sig = (ctl["p_two_sided"] <= 0.1).mean()
        print(f"[topology] {len(ctl)} storms: shuffle keeps level statistics in {int(ctl['level_stats_unchanged_by_shuffle'].sum())}, "
              f"changes Betti curves in {int((ctl['shuffle_curve_distance'] > 0).sum())}; "
              f"real storm outside its IAAFT surrogates (p <= 0.10) in {sig:.0%}")

    elif a.cmd == "physics":
        import numpy as np
        import pandas as pd
        from . import physics as P
        pc = cfg["physics"]
        dst, omni = read_snapshot("kyoto_dst", snap), read_snapshot("omni2_hourly", snap)
        storms = pd.read_parquet("data/derived/storms.parquet")
        os.makedirs("data/derived", exist_ok=True)
        b = P.driver_bounds(omni, pc["gate1"])
        b.to_csv("data/derived/gate1_driver_bounds.csv", index=False, lineterminator="\n")
        print(b[b["return_years"] == 100][["driver", "threshold", "n_clusters", "shape_xi", "return_level", "upper95", "observed_max"]].round(2).to_string(index=False))
        desc = P.storm_descriptors(dst, storms)
        loo = [P.envelope_label(desc.drop(index=i), r.to_dict(), pc["envelope"]) for i, r in desc.iterrows()]
        desc["label_leave_one_out"] = loo
        desc.to_csv("data/derived/storm_descriptors.csv", index=False, lineterminator="\n")
        print(f"[physics] gate 3: {len(desc)} observed storms; leave-one-out labels {pd.Series(loo).value_counts().to_dict()}")
        rows = []
        for _, st in storms[storms["in_study"]].iterrows():
            w = P.storm_drivers(omni, st["window_start"], st["window_end"])
            d_obs = dst.set_index("time")["dst_nT"].reindex(w.index).to_numpy(dtype=float)
            if w[["V_kms", "Bz_gsm_nT", "Pdyn_nPa"]].isna().any().any() or np.isnan(d_obs).any():
                rows.append({"storm_id": st["storm_id"], "drivers_complete": False})
                continue
            fit, _ = P.calibrate_obm(w["V_kms"], w["Bz_gsm_nT"], w["Pdyn_nPa"], d_obs)
            rows.append({"storm_id": st["storm_id"], "drivers_complete": True, "dst_min_nT": st["dst_min_nT"], **fit})
        cal = pd.DataFrame(rows)
        cal.to_csv("data/derived/gate2_obm_calibration.csv", index=False, lineterminator="\n")
        ok = cal[cal["drivers_complete"]]
        print(f"[physics] gate 2: OBM calibrated on {len(ok)} of {len(cal)} study storms; median r {ok['r'].median():.2f}, "
              f"median RMSE {ok['rmse_nT'].median():.1f} nT")

    elif a.cmd == "reg3":
        from .reg3 import run as run_reg3
        y, r, v = run_reg3()
        print(r.round(4).to_string(index=False))
        print(f"[reg3] verdicts: {v}")
        from .reg3 import robustness
        print(robustness().round(4).to_string(index=False))

    elif a.cmd == "part_a":
        from .part_a import run as run_part_a
        run_part_a(cfg, snap)

    elif a.cmd == "proxies":
        from .proxies import run
        run(cfg, a.spacetrack, a.work, years=a.years)


if __name__ == "__main__":
    main()
