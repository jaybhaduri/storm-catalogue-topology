"""WP1 acceptance tests. Offline tests use the committed snapshots; `-m network` re-pulls sources."""
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from phinn_sw import sources
from phinn_sw.snapshot import content_hash, read_snapshot
from phinn_sw.storms import detect_storms

CFG = yaml.safe_load(open("config.yaml"))
MAN = json.load(open("data/snapshots/manifest.json"))


@pytest.fixture(scope="module")
def storms():
    return detect_storms(read_snapshot("kyoto_dst"), CFG)


def test_committed_snapshots_match_manifest():
    for name, entry in MAN.items():
        assert content_hash(read_snapshot(name)) == entry["content_sha256"], name


def test_finds_published_storm_minima(storms):
    def at(day):
        hit = storms[storms["t_min"].dt.strftime("%Y-%m-%d") == day]
        assert len(hit) == 1, day
        return hit.iloc[0]
    assert at("2003-11-20")["dst_min_nT"] == -422                  # final Dst
    assert at("1989-03-14")["dst_min_nT"] == -589                  # record minimum
    may = at("2024-05-11")
    assert may["dst_min_nT"] == -406 and may["dst_version_at_min"] == "provisional"   # decision 10: frozen snapshot value


def test_february_2022_anchor_included(storms):
    a = storms[storms["anchor"] == "February 2022 Starlink loss"]
    assert len(a) == 1 and -80 < a.iloc[0]["dst_min_nT"] < -50


def test_split_and_windows(storms):
    s = storms[storms["in_study"]]
    assert set(s["split"]) == {"development", "held_out"}
    assert (s.loc[s["split"] == "development", "t_min"].dt.year <= 2002).all()
    sev = s[s["severe"]]
    assert (sev["split"] == "development").sum() == 5 and (sev["split"] == "held_out").sum() == 5
    assert ((s["window_end"] - s["window_start"]) == pd.Timedelta(days=13)).all()


def test_parsers_on_known_lines():
    line = ("2024   1  0 2596 51 52  60  36   5.3   5.1  -7.3 322.9   4.0  -3.0  -0.6  -2.8  -1.3   0.1   1.5"
            "   0.3   0.4   1.5   31114.   7.4  306.  -2.3   2.8 0.042  1.35    4691.   0.4    2.   0.3   1.3"
            " 0.007   0.40   1.75   7.9  7  55     0   20 999999.99 99999.99 99999.99 99999.99 99999.99"
            " 99999.99  0   3 131.2   0.3    -9    11  5.0")
    o = sources.parse_omni2(line).iloc[0]
    assert o["time"] == pd.Timestamp("2024-01-01 00:00") and o["Bz_gsm_nT"] == np.float32(-1.3)
    assert o["V_kms"] == 306 and o["Kp"] == np.float32(0.7) and o["f107_sfu"] == np.float32(131.2)
    k = sources.parse_kyoto_hapi_csv("2024-05-11T02:29:30Z,-406,10\n2024-05-11T03:29:30Z,99999,10\n")
    assert k["time"].iloc[0] == pd.Timestamp("2024-05-11 02:00") and np.isnan(k["dst_nT"].iloc[1])
    assert k["version"].iloc[0] == "provisional"


@pytest.mark.network
def test_repull_reproduces_hashes():
    """Re-pulling a source must give the same content hash as the committed snapshot."""
    sol, dtc = sources.pull_set(CFG)
    assert content_hash(sol) == MAN["set_solfsmy"]["content_sha256"]
    assert content_hash(dtc) == MAN["set_dtc"]["content_sha256"]
    assert content_hash(sources.pull_kyoto_dst(CFG)) == MAN["kyoto_dst"]["content_sha256"]
