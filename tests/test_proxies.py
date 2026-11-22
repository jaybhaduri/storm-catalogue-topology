"""WP2 custody proxies on synthetic element sets (no Space-Track data needed)."""
import numpy as np
import pandas as pd
import yaml

from phinn_sw import proxies

CFG = yaml.safe_load(open("config.yaml"))
MU = 398600.4418


def _sets(nid, otype, start, n, step_h, a0_km, dadt_km_per_day=-0.05, raise_at=None, jitter_km=0.0, seed=0):
    rng = np.random.default_rng(seed)
    rows, a, ma = [], a0_km, 0.0
    for k in range(n):
        t = pd.Timestamp(start) + pd.Timedelta(hours=step_h * k)
        if k:
            a += dadt_km_per_day * step_h / 24 + rng.normal(0, jitter_km)
            if raise_at is not None and k == raise_at:
                a += 2.0
        n_rev_day = np.sqrt(MU / a ** 3) * 86400 / (2 * np.pi)
        if k:
            ma = (ma + 360.0 * n_rev_day * step_h / 24) % 360
        rows.append({"GP_ID": nid * 10000 + k, "NORAD_CAT_ID": nid, "OBJECT_TYPE": otype, "EPOCH": t,
                     "MEAN_MOTION": n_rev_day, "ECCENTRICITY": 0.0005, "INCLINATION": 51.6, "RA_OF_ASC_NODE": 100.0,
                     "ARG_OF_PERICENTER": 90.0, "MEAN_ANOMALY": ma, "BSTAR": 1e-4, "MEAN_MOTION_DOT": 0.0,
                     "MEAN_MOTION_DDOT": 0.0, "SEMIMAJOR_AXIS": a, "PERIAPSIS": a - 6378.137, "APOAPSIS": a - 6378.137,
                     "CLASSIFICATION_TYPE": "U", "ELEMENT_SET_NO": 999, "REV_AT_EPOCH": 1, "CREATION_DATE": t})
    return pd.DataFrame(rows)


def test_identical_orbit_gives_small_correction():
    df = _sets(1, "DEBRIS", "2000-01-01", 2, 1 / 60, 6778.0, dadt_km_per_day=0.0)   # one minute apart
    df["alt_km"], df["band_km"] = df["SEMIMAJOR_AXIS"] - 6378.137, 400
    p = proxies.pair_corrections(df)
    assert len(p) == 1 and p["sgp4_error"].iloc[0] == 0
    assert p["dpos_km"].iloc[0] < 1.0


def test_pipeline_runs_end_to_end(tmp_path):
    st = tmp_path / "st"
    (st / "gp_history").mkdir(parents=True)
    df = pd.concat([_sets(1, "DEBRIS", "2000-01-01", 120, 12, 6778.0, jitter_km=0.01, seed=1),
                    _sets(2, "PAYLOAD", "2000-01-01", 120, 12, 6900.0, raise_at=40, jitter_km=0.01, seed=2),
                    _sets(3, "ROCKET BODY", "2000-01-01", 60, 24, 7000.0, jitter_km=0.01, seed=3)])
    for w0 in pd.date_range("2000-01-01", "2000-03-01", freq="10D"):
        part = df[(df["EPOCH"] >= w0) & (df["EPOCH"] < w0 + pd.Timedelta(days=10))]
        part.assign(EPOCH=part["EPOCH"].dt.strftime("%Y-%m-%dT%H:%M:%S.%f")).to_parquet(
            st / "gp_history" / f"gp_{w0:%Y%m%d}.parquet", index=False)
    out, gates = proxies.run(CFG, str(st), str(tmp_path / "work"), str(tmp_path / "derived"), years=[2000])
    assert len(out) > 0 and set(out["population"]) <= {"inactive", "active", "other"}
    assert "active" in set(out["population"])                       # the raised payload becomes active
    assert out["m1_raise_events"].sum() == 1
    assert (out["n_tracked"] >= 0).all() and out["p1_stale_over_24h"].max() >= 0
    assert (tmp_path / "derived" / "custody_proxies_daily.parquet").exists()
