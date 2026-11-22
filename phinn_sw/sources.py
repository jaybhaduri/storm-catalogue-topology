"""Downloaders and parsers for the public datasets. Each returns a tidy DataFrame with a UTC `time`
(start of the hour or interval) or `date` column; snapshot.py writes it into the repository."""
from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd
import requests

UA = {"User-Agent": "storm-catalogue-research/0.1"}


def _get(url: str, params=None, tries: int = 4, timeout: int = 120) -> requests.Response:
    for i in range(tries):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


# ------------------------------------------------------------------ WDC Kyoto Dst (HAPI)

def parse_kyoto_hapi_csv(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), header=None, names=["center", "dst_nT", "version_code"])
    t = pd.to_datetime(df["center"], utc=True)
    out = pd.DataFrame({
        "time": t.dt.floor("h").dt.tz_convert(None),              # hour start; values are hourly means
        "dst_nT": df["dst_nT"].where(df["dst_nT"] != 99999).astype("float32"),
        "version_code": df["version_code"].astype("int16"),
    })
    out["version"] = out["version_code"].floordiv(10).map({0: "realtime", 1: "provisional", 2: "final"})
    return out


def pull_kyoto_dst(cfg: dict) -> pd.DataFrame:
    s = cfg["sources"]["kyoto_dst"]
    start, end = pd.Timestamp(str(s["start"])), pd.Timestamp(str(cfg["snapshot"]["end"])) + pd.Timedelta(days=1)
    parts = []
    for y0 in range(start.year, end.year + 1, 10):
        a, b = max(start, pd.Timestamp(f"{y0}-01-01")), min(end, pd.Timestamp(f"{y0 + 10}-01-01"))
        if a >= b:
            continue
        half = pd.Timedelta(minutes=29, seconds=30)       # HAPI stamps are bin centres, HH:29:30
        r = _get(f"{s['hapi']}/data", {"id": s["dataset"], "time.min": (a + half).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                       "time.max": (b + half).strftime("%Y-%m-%dT%H:%M:%SZ"), "format": "csv"})
        parts.append(parse_kyoto_hapi_csv(r.text))
    df = pd.concat(parts).drop_duplicates("time").sort_values("time").reset_index(drop=True)
    return df[(df["time"] >= start) & (df["time"] < end)].reset_index(drop=True)


# ------------------------------------------------------------------ NASA OMNI2 hourly

# word number (1-based) -> (column name, fill value); see omniweb.gsfc.nasa.gov/html/ow_data.html
OMNI2_COLS = {
    9: ("B_nT", 999.9), 13: ("Bx_gse_nT", 999.9), 16: ("By_gsm_nT", 999.9), 17: ("Bz_gsm_nT", 999.9),
    23: ("T_K", 9999999.0), 24: ("n_cc", 999.9), 25: ("V_kms", 9999.0), 29: ("Pdyn_nPa", 99.99),
    36: ("Ey_mVm", 999.99), 39: ("Kp_x10", 99), 41: ("dst_omni_nT", 99999), 50: ("ap", 999),
    51: ("f107_sfu", 999.9),
}


def parse_omni2(text: str) -> pd.DataFrame:
    a = np.loadtxt(io.StringIO(text))
    if a.ndim == 1:
        a = a[None]
    year, doy, hour = a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2].astype(int)
    t = (pd.to_datetime(year.astype(str), format="%Y") + pd.to_timedelta(doy - 1, unit="D")
         + pd.to_timedelta(hour, unit="h"))
    out = {"time": t}
    for word, (name, fill) in OMNI2_COLS.items():
        col = a[:, word - 1]
        out[name] = np.where(np.isclose(col, fill), np.nan, col).astype("float32")
    df = pd.DataFrame(out)
    df["Kp"] = (df.pop("Kp_x10") / 10.0).astype("float32")
    return df


def pull_omni2_hourly(cfg: dict) -> pd.DataFrame:
    s = cfg["sources"]["omni2_hourly"]
    end = pd.Timestamp(str(cfg["snapshot"]["end"]))
    parts = [parse_omni2(_get(s["url"].format(year=y)).text) for y in range(int(s["start_year"]), end.year + 1)]
    df = pd.concat(parts).sort_values("time").reset_index(drop=True)
    return df[df["time"] < end + pd.Timedelta(days=1)].reset_index(drop=True)


# ------------------------------------------------------------------ GFZ Kp and Hp30

def pull_gfz(cfg: dict, index: str, start) -> pd.DataFrame:
    s = cfg["sources"]["gfz"]
    end = pd.Timestamp(str(cfg["snapshot"]["end"])) + pd.Timedelta(days=1)
    parts = []
    for y in range(pd.Timestamp(str(start)).year, end.year):
        r = _get(s["url"], {"start": f"{y}-01-01T00:00:00Z", "end": f"{y}-12-31T23:59:59Z", "index": index}).json()
        parts.append(pd.DataFrame({"time": pd.to_datetime(r["datetime"], utc=True).tz_convert(None),
                                   index: np.asarray(r[index], dtype="float32"),
                                   "status": r.get("status", ["unstated"] * len(r[index]))}))   # Hp30 has no status
    return pd.concat(parts).drop_duplicates("time").sort_values("time").reset_index(drop=True)


# ------------------------------------------------------------------ SET JB2008 indices

def parse_solfsmy(text: str) -> pd.DataFrame:
    rows = [ln.split() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    df = pd.DataFrame(rows, columns=["year", "doy", "jd", "F10", "F81c", "S10", "S81c", "M10", "M81c",
                                     "Y10", "Y81c", "Ssrc"])
    date = pd.to_datetime(df["year"], format="%Y") + pd.to_timedelta(df["doy"].astype(int) - 1, unit="D")
    out = df[["F10", "F81c", "S10", "S81c", "M10", "M81c", "Y10", "Y81c"]].astype("float32")
    out.insert(0, "date", date)
    out["Ssrc"] = df["Ssrc"]
    return out


def parse_dtcfile(text: str) -> pd.DataFrame:
    recs = []
    for ln in text.splitlines():
        p = ln.split()
        if len(p) < 27 or p[0] != "DTC":
            continue
        day = pd.Timestamp(f"{p[1]}-01-01") + pd.Timedelta(days=int(p[2]) - 1)
        recs += [(day + pd.Timedelta(hours=h), float(v)) for h, v in enumerate(p[3:27])]
    return pd.DataFrame(recs, columns=["time", "DTC_K"]).astype({"DTC_K": "float32"})


def pull_set(cfg: dict):
    s = cfg["sources"]["set_jb2008"]
    end = pd.Timestamp(str(cfg["snapshot"]["end"])) + pd.Timedelta(days=1)
    sol = parse_solfsmy(_get(s["solfsmy"]).text)
    dtc = parse_dtcfile(_get(s["dtcfile"]).text)
    return sol[sol["date"] < end].reset_index(drop=True), dtc[dtc["time"] < end].reset_index(drop=True)
