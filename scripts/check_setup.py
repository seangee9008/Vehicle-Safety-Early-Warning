"""Check that your environment and data are ready. Safe to run any time.

Usage:
    python scripts/check_setup.py
"""
import importlib
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OK, MISS = "  [ok]  ", "  [--]  "


def check(label: str, passed: bool, hint: str = "") -> bool:
    print(f"{OK if passed else MISS}{label}" + ("" if passed or not hint else f"   -> {hint}"))
    return passed


def main() -> None:
    print("Python")
    check(f"Python {sys.version.split()[0]}", sys.version_info >= (3, 10), "use Python 3.11")
    check("running inside .venv", ".venv" in sys.prefix, "run: source .venv/bin/activate")

    print("\nPackages")
    for mod in ["pandas", "pyarrow", "duckdb", "requests", "dbt", "google.cloud.bigquery",
                "google.protobuf", "grpc_tools", "numpy", "sklearn", "lightgbm", "shap", "torch",
                "sentence_transformers", "anthropic", "streamlit", "dotenv"]:
        try:
            m = importlib.import_module(mod)
            check(f"{mod} {getattr(m, '__version__', '')}".strip(), True)
        except Exception:
            check(mod, False, "run: pip install -r requirements.txt")

    print("\nCommand-line tools")
    for tool, hint in [("git", "install Git"), ("gcloud", "install the Google Cloud CLI"),
                       ("gsutil", "comes with the Google Cloud CLI (needed for Waymo)"),
                       ("dbt", "pip install -r requirements.txt")]:
        check(tool, shutil.which(tool) is not None, hint)

    print("\nSecrets (.env)")
    env = ROOT / ".env"
    check(".env exists", env.exists(), "cp .env.example .env")
    if env.exists():
        keys = dict(line.split("=", 1) for line in env.read_text().splitlines() if "=" in line and not line.startswith("#"))
        for k in ["ANTHROPIC_API_KEY", "GCP_PROJECT_ID"]:
            check(f"{k} set", bool(keys.get(k, "").strip()), "fill it in when your layer needs it")

    print("\nNHTSA data")
    raw = ROOT / "data" / "raw" / "nhtsa"
    for t in ["complaints", "recalls", "investigations"]:
        n = len(list((raw / t).glob("*.txt"))) if (raw / t).exists() else 0
        check(f"raw {t}: {n} file(s)", n > 0, "python scripts/download_nhtsa.py")
    db = ROOT / "data" / "nhtsa.duckdb"
    if check("data/nhtsa.duckdb", db.exists(), "python scripts/load_nhtsa.py"):
        import duckdb
        con = duckdb.connect(str(db), read_only=True)
        for (t,) in con.execute("select table_name from information_schema.tables where table_schema='raw' order by 1").fetchall():
            n = con.execute(f"select count(*) from raw.{t}").fetchone()[0]
            check(f"raw.{t}: {n:,} rows", n > 0)

    print("\nWaymo data")
    check("compiled Waymo protos (src/waymo_pb)", (ROOT / "src" / "waymo_pb" / "waymo_open_dataset").exists(),
          "python scripts/build_waymo_protos.py")
    shards = list((ROOT / "data" / "raw" / "waymo").rglob("*tfrecord*"))
    check(f"raw shards: {len(shards)}", bool(shards), "bash scripts/download_waymo.sh")
    pq = list((ROOT / "data" / "processed" / "waymo").glob("*_tracks.parquet"))
    check(f"parquet track files: {len(pq)}", bool(pq), "python scripts/waymo_to_parquet.py")
    print()


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
