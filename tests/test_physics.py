"""WP5 physics gate: model equations and gate logic."""
import numpy as np
import pandas as pd

from phinn_sw import physics as P


def test_obm_quiet_wind_stays_near_zero_and_southward_bz_drives_a_storm():
    n = 48
    V, P_ = np.full(n, 400.0), np.full(n, 2.0)
    quiet = P.obm_dst(V, np.full(n, 2.0), P_, dst0=0.0)
    assert np.all(np.abs(quiet) < 15)                        # northward Bz: no injection
    bz = np.r_[np.full(6, 2.0), np.full(12, -20.0), np.full(30, 2.0)]
    storm = P.obm_dst(np.full(n, 600.0), bz, P_, dst0=0.0)
    assert storm.min() < -100 and storm[-1] > storm.min()   # main phase, then recovery


def test_obm_injection_uses_published_constants():
    # V Bs = 600 km/s * 20 nT = 12 mV/m; Q = -4.4 * (12 - 0.49) = -50.6 nT/h
    d = P.obm_dst([600.0, 600.0], [-20.0, -20.0], [0.0, 0.0], dst0=-11.0)    # Dst* starts at 0
    assert abs((d[0] + 11.0) - (-4.4 * (12 - 0.49))) < 1e-9


def test_calibration_recovers_scaled_model():
    rng = np.random.default_rng(0)
    bz = np.r_[np.zeros(24), -15 + rng.normal(0, 2, 24), np.zeros(72)]
    V, Pd = np.full(120, 550.0), np.full(120, 3.0)
    truth = P.obm_dst(V, bz, Pd, 0.0, a_scale=1.3, tau_scale=0.8)
    fit, _ = P.calibrate_obm(V, bz, Pd, truth)
    assert abs(fit["a_scale"] - 1.3) < 0.05 and abs(fit["tau_scale"] - 0.8) < 0.05 and fit["r"] > 0.99


def test_envelope_labels():
    rng = np.random.default_rng(1)
    obs = pd.DataFrame({"dst_min_nT": rng.uniform(-300, -200, 40), "fall_rate_nT_per_h": rng.uniform(10, 40, 40),
                        "main_phase_h": rng.uniform(4, 12, 40), "hours_below_100": rng.uniform(10, 40, 40),
                        "recovery_efold_h": rng.uniform(10, 30, 40), "integrated_dst_nTh": rng.uniform(-20000, -8000, 40)})
    ec = {"record_min_nT": -589, "carrington_range": [-850, -1150], "intensity_tolerance": 0.3, "density_floor_quantile": 0.01}
    typical = obs.median().to_dict()
    assert P.envelope_label(obs, typical, ec) == "in_record"
    assert P.envelope_label(obs, {**typical, "dst_min_nT": -700}, ec) == "beyond_record"
    assert P.envelope_label(obs, {**typical, "dst_min_nT": -900}, ec) == "carrington_class"


def test_decluster_counts_separate_events():
    x = pd.Series(np.r_[np.zeros(10), [5, 6, 5], np.zeros(60), [7], np.zeros(5)])
    assert list(P.decluster(x, 1.0, run_h=48)) == [6, 7]
