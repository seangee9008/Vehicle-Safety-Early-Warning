"""Save a copy of every table the demo reads, so it keeps working after BigQuery's free tables expire.

    python run.py snapshot             # app/snapshot/*.parquet + manifest.json ("saved copy")
    python run.py snapshot --final     # the same, marked final: the app shows it from now on
    python -m src.snapshot --days-left # print days until the earliest BigQuery table expires

Reads BigQuery only (never local data). Refuses empty tables and anything with trajectory
coordinates, checks that the app loads every tab from the new copy (AppTest), and only then
replaces app/snapshot/. The manifest holds no secrets: dates, row counts and the source name.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import data as D
from src import bq_reader

OUT = ROOT / "app" / "snapshot"
COORDINATE_COLUMNS = {"past_path", "true_future", "baseline_path", "model_paths", "lane_polylines",
                      "model_confidences", "x", "y"}
TABS = ["Overview", "Recall watchlist", "AI agent", "Model results", "Trajectory", "Data and limitations"]


class SnapshotError(RuntimeError):
    pass


def _parquet_safe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if str(df[col].dtype) in ("dbdate", "dbtime"):
            df[col] = pd.to_datetime(df[col].astype(str), errors="coerce")
    return df


def export(dest: Path) -> dict[str, int]:
    counts = {}
    for name in D.SOURCES:
        df = D._bq_read(name)
        if df.empty:
            raise SnapshotError(f"{name}: no rows in BigQuery")
        bad = COORDINATE_COLUMNS & set(df.columns)
        if bad:
            raise SnapshotError(f"{name}: has coordinate columns {sorted(bad)}; never snapshot those")
        _parquet_safe(df).to_parquet(dest / f"{name}.parquet", index=False)
        counts[name] = len(df)
    return counts


def validate(dest: Path, final: bool) -> None:
    """Load every tab of the app from the new copy; raise on any exception."""
    from streamlit.testing.v1 import AppTest

    saved = {k: os.environ.get(k) for k in ("VSE_SNAPSHOT_DIR", "VSE_SNAPSHOT_REMOTE", "FORCE_MODE")}
    os.environ.update(VSE_SNAPSHOT_DIR=str(dest), VSE_SNAPSHOT_REMOTE="0",
                      FORCE_MODE="final" if final else "snapshot")
    try:
        at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=180)
        at.run()
        for tab in TABS:
            at.session_state["main_tab"] = tab
            at.run()
            if at.exception:
                raise SnapshotError(f"app fails on {tab}: {at.exception[0].message[:300]}")
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def days_left() -> float:
    expiry = bq_reader.dataset_expiry()
    if expiry is None:
        raise SnapshotError("no table in the dataset has an expiry (or the dataset is empty)")
    return (expiry - datetime.now(UTC)).total_seconds() / 86400


def snapshot(final: bool = False, out: Path = OUT) -> dict:
    D.use_streamlit_secrets()
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp)
        counts = export(dest)
        expiry = bq_reader.dataset_expiry()
        log = pd.read_parquet(dest / "publish_log.parquet")
        manifest = {
            "final": final,
            "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "last_updated": D.last_published(log),
            "bq_expires_at": expiry.isoformat() if expiry else None,
            "source": f"bigquery {bq_reader.PROJECT}.{bq_reader.DATASET}",
            "tables": counts,
        }
        (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        validate(dest, final)
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(dest, out)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--final", action="store_true", help="mark the copy final (the app uses it for good)")
    parser.add_argument("--days-left", action="store_true", help="print days until BigQuery tables expire")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    try:
        if args.days_left:
            print(f"{days_left():.2f}")
            return 0
        manifest = snapshot(final=args.final, out=args.out)
    except Exception as error:  # noqa: BLE001 - one clear line for the log, non-zero exit
        print(f"snapshot failed: {str(error).splitlines()[0]}", file=sys.stderr)
        return 3
    size = sum(f.stat().st_size for f in args.out.iterdir()) / 1e6
    print(f"wrote {args.out} ({len(manifest['tables'])} tables, {size:.2f} MB, final={manifest['final']})")
    for name, rows in manifest["tables"].items():
        print(f"  {name:<24}{rows:>6,} rows")
    print(f"BigQuery expires {manifest['bq_expires_at']}; last published {manifest['last_updated']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
