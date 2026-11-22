"""Write datasets as Parquet snapshots and record them in a SHA-256 manifest.

Two hashes per dataset:
- file_sha256: the Parquet bytes, as committed.
- content_sha256: a canonical CSV rendering of the table, independent of the Parquet writer
  version. Re-pulling a source and getting the same content_sha256 proves the data did not change.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os

import pandas as pd


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_hash(df: pd.DataFrame) -> str:
    """Hash of the table's values: columns in order, rows in order, fixed float formatting."""
    csv = df.to_csv(index=False, lineterminator="\n", float_format="%.6g", date_format="%Y-%m-%dT%H:%M:%SZ")
    return sha256_bytes(csv.encode())


def write_snapshot(df: pd.DataFrame, name: str, snap_dir: str, meta: dict) -> dict:
    os.makedirs(snap_dir, exist_ok=True)
    path = os.path.join(snap_dir, f"{name}.parquet")
    df.to_parquet(path, index=False, compression="zstd")
    tcol = next((c for c in ("time", "date") if c in df.columns), None)
    entry = {
        "name": name,
        "file": os.path.basename(path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "first": str(df[tcol].min()) if tcol else None,
        "last": str(df[tcol].max()) if tcol else None,
        "file_sha256": sha256_file(path),
        "content_sha256": content_hash(df),
        "retrieved_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        **meta,
    }
    man_path = os.path.join(snap_dir, "manifest.json")
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else {}
    manifest[name] = entry
    with open(man_path, "w") as f:
        json.dump(dict(sorted(manifest.items())), f, indent=2)
        f.write("\n")
    print(f"[snapshot] {name}: {entry['rows']:,} rows, {entry['first']} to {entry['last']}, "
          f"{os.path.getsize(path) / 1e6:.1f} MB, content {entry['content_sha256'][:12]}")
    return entry


def read_snapshot(name: str, snap_dir: str = "data/snapshots") -> pd.DataFrame:
    return pd.read_parquet(os.path.join(snap_dir, f"{name}.parquet"))
