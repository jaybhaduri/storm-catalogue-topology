"""Space-Track.org client for LEO GP history, SATCAT and decay records.

Credentials: SPACETRACK_USER / SPACETRACK_PASSWORD from the environment, or typed at a prompt
(the password is read with getpass and never echoed, logged or stored).

Space-Track limits the API to under 30 requests a minute and 300 an hour and asks that history be
downloaded once and kept locally. This client stays under both limits, downloads in fixed windows,
skips windows already on disk (so it can be stopped and restarted), and keeps only the fields the
analysis needs. Records stay on this computer: the Space-Track User Agreement restricts passing
them on, so only derived proxies go into the repository.
"""
from __future__ import annotations

import datetime as dt
import getpass
import io
import os
import time
from collections import deque

import pandas as pd
import requests

FIELDS = ["GP_ID", "NORAD_CAT_ID", "OBJECT_TYPE", "EPOCH", "MEAN_MOTION", "ECCENTRICITY", "INCLINATION",
          "RA_OF_ASC_NODE", "ARG_OF_PERICENTER", "MEAN_ANOMALY", "BSTAR", "MEAN_MOTION_DOT", "MEAN_MOTION_DDOT",
          "SEMIMAJOR_AXIS", "PERIAPSIS", "APOAPSIS", "CLASSIFICATION_TYPE", "ELEMENT_SET_NO", "REV_AT_EPOCH",
          "CREATION_DATE"]


class RateLimiter:
    def __init__(self, per_minute=25, per_hour=280):
        self.limits = [(per_minute, 60.0), (per_hour, 3600.0)]
        self.calls = deque()

    def wait(self):
        while True:
            now = time.monotonic()
            while self.calls and now - self.calls[0] > 3600:
                self.calls.popleft()
            waits = [span - (now - self.calls[-n]) for n, span in self.limits if len(self.calls) >= n]
            w = max([0.0] + waits)
            if w <= 0:
                self.calls.append(now)
                return
            time.sleep(min(w, 60))


class SpaceTrack:
    def __init__(self, base="https://www.space-track.org"):
        self.base, self.rl = base, RateLimiter()
        self.user = os.environ.get("SPACETRACK_USER") or input("Space-Track username (email): ").strip()
        self._pw = os.environ.get("SPACETRACK_PASSWORD") or getpass.getpass("Space-Track password: ")
        self.login()

    def login(self):
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "storm-catalogue-research/0.1"
        self.rl.wait()
        r = self.s.post(f"{self.base}/ajaxauth/login", data={"identity": self.user, "password": self._pw}, timeout=60)
        r.raise_for_status()
        if "Failed" in r.text or "login" in r.url.lower() and r.text.strip().startswith("<"):
            raise RuntimeError("Space-Track login failed: check the username and password")
        print("[spacetrack] signed in")

    def query(self, path: str, tries: int = 4) -> pd.DataFrame:
        for i in range(tries):
            self.rl.wait()
            try:
                r = self.s.get(f"{self.base}/basicspacedata/query/{path}/format/csv", timeout=900)
                if r.status_code in (401, 403):
                    self.login()
                    continue
                r.raise_for_status()
                txt = r.text
                if not txt.strip():
                    return pd.DataFrame(columns=FIELDS)
                return pd.read_csv(io.StringIO(txt), low_memory=False)
            except (requests.RequestException, pd.errors.ParserError) as e:
                if i == tries - 1:
                    raise
                print(f"[spacetrack] retry {i + 1} after {type(e).__name__}")
                time.sleep(30 * (i + 1))
        raise RuntimeError("unreachable")


def _windows(start: pd.Timestamp, end: pd.Timestamp, days: int):
    a = start
    while a <= end:
        b = min(a + pd.Timedelta(days=days), end + pd.Timedelta(days=1))
        yield a, b
        a = b


ROW_CAP = 500_000          # Space-Track returns at most this many rows per request


def fetch_window(st: "SpaceTrack", a: pd.Timestamp, b: pd.Timestamp, mm: float, pred: str) -> pd.DataFrame:
    """One window; if Space-Track's row cap is hit, split the window in two and fetch each half.
    EPOCH ranges are inclusive at both ends, so halves are deduplicated on GP_ID."""
    df = st.query(f"class/gp_history/EPOCH/{a:%Y-%m-%d %H:%M:%S}--{b:%Y-%m-%d %H:%M:%S}/MEAN_MOTION/>{mm}"
                  f"/orderby/NORAD_CAT_ID,EPOCH/predicates/{pred}")
    if len(df) != ROW_CAP:
        return df
    mid = a + (b - a) / 2
    print(f"[spacetrack] {a:%Y-%m-%d %H:%M} to {b:%Y-%m-%d %H:%M} hit the {ROW_CAP:,}-row cap; splitting")
    parts = [fetch_window(st, a, mid, mm, pred), fetch_window(st, mid, b, mm, pred)]
    return pd.concat(parts, ignore_index=True).drop_duplicates("GP_ID").sort_values(["NORAD_CAT_ID", "EPOCH"])


def pull_gp_history(cfg: dict, out_dir: str):
    s = cfg["sources"]["spacetrack"]
    start, end = pd.Timestamp(str(s["start"])), pd.Timestamp(str(cfg["snapshot"]["end"]))
    days = int(s.get("window_days", 10))
    mm = float(s["leo_min_mean_motion"])
    gp_dir = os.path.join(out_dir, "gp_history")
    os.makedirs(gp_dir, exist_ok=True)
    log = open(os.path.join(out_dir, "download_log.txt"), "a")
    wins = list(_windows(start, end, days))
    import pyarrow.parquet as pq
    def _needs(a):
        p = os.path.join(gp_dir, f"gp_{a:%Y%m%d}.parquet")
        # missing, or truncated: exactly ROW_CAP rows (a repaired, split window holds more than the cap)
        return not os.path.exists(p) or pq.ParquetFile(p).metadata.num_rows == ROW_CAP
    todo = [(a, b) for a, b in wins if _needs(a)]
    print(f"[spacetrack] {len(wins)} windows of {days} days; {len(wins) - len(todo)} already on disk, {len(todo)} to fetch. "
          f"Space-Track's rate limit makes this take at least {len(todo) / 280:.1f} hours.")
    st = SpaceTrack(s["base"])
    for path, cls in (("satcat.parquet", "satcat"), ("decay.parquet", "decay")):
        p = os.path.join(out_dir, path)
        if not os.path.exists(p):
            df = st.query(f"class/{cls}/orderby/NORAD_CAT_ID")
            df.to_parquet(p, index=False, compression="zstd")
            print(f"[spacetrack] {cls}: {len(df):,} rows")
    pred = ",".join(FIELDS)
    t0 = time.time()
    for i, (a, b) in enumerate(todo, 1):
        df = fetch_window(st, a, b, mm, pred)
        df.to_parquet(os.path.join(gp_dir, f"gp_{a:%Y%m%d}.parquet"), index=False, compression="zstd")
        rate = (time.time() - t0) / i
        msg = (f"{dt.datetime.now():%Y-%m-%d %H:%M} window {a:%Y-%m-%d}: {len(df):,} element sets "
               f"({i}/{len(todo)}, about {rate * (len(todo) - i) / 3600:.1f} h left)")
        print("[spacetrack] " + msg)
        log.write(msg + "\n")
        log.flush()
    print("[spacetrack] done")
