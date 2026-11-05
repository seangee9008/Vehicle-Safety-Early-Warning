"""Load the raw NHTSA .txt files into Parquet + a local DuckDB database (and optionally BigQuery).

Every column is loaded as text exactly as NHTSA publishes it; cleaning and typing is up to you
(dbt models in dbt/models/, or pandas in a notebook).

Usage:
    python scripts/load_nhtsa.py
    python scripts/load_nhtsa.py --bigquery my-gcp-project.nhtsa

Output:
    data/processed/nhtsa/{complaints,recalls,investigations}.parquet
    data/nhtsa.duckdb  ->  tables raw.complaints, raw.recalls, raw.investigations
"""
import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.nhtsa_schemas import DROP, TABLES  # noqa: E402

RAW = ROOT / "data" / "raw" / "nhtsa"
OUT = ROOT / "data" / "processed" / "nhtsa"
DB = ROOT / "data" / "nhtsa.duckdb"


def read_tab(path: Path, names: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, dtype=str, quoting=3, encoding="latin-1",
                     on_bad_lines="warn", keep_default_na=False)
    if df.shape[1] == len(names):
        df.columns = names
    else:
        print(f"  WARNING: {path.name} has {df.shape[1]} columns but the schema lists {len(names)}. "
              f"Using col_1..col_{df.shape[1]}; fix src/nhtsa_schemas.py using the data dictionary.")
        df.columns = [f"col_{i + 1}" for i in range(df.shape[1])]
    return df


def load(table: str, names: list[str]) -> pd.DataFrame | None:
    files = sorted((RAW / table).glob("*.txt"))
    if not files:
        print(f"{table}: no .txt files in {RAW / table} (run download_nhtsa.py first)")
        return None
    df = pd.concat([read_tab(f, names) for f in files], ignore_index=True)
    df = df.drop(columns=[c for c in DROP.get(table, []) if c in df.columns]).drop_duplicates()
    print(f"{table}: {len(df):,} rows x {df.shape[1]} columns from {len(files)} file(s)")
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--bigquery", metavar="PROJECT.DATASET", help="also load into BigQuery")
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB))
    con.execute("create schema if not exists raw")
    for table, names in TABLES.items():
        df = load(table, names)
        if df is None:
            continue
        pq = OUT / f"{table}.parquet"
        df.to_parquet(pq, index=False)
        con.execute(f"create or replace table raw.{table} as select * from read_parquet('{pq.as_posix()}')")
        if args.bigquery:
            from google.cloud import bigquery
            dest = f"{args.bigquery}.raw_{table}"
            bigquery.Client().load_table_from_dataframe(
                df, dest, job_config=bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
            ).result()
            print(f"  -> BigQuery {dest}")
    con.close()
    print(f"\nDone. DuckDB database: {DB}")


if __name__ == "__main__":
    main()
