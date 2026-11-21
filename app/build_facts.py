"""Write app/facts.json (dataset sizes) and app/agent_snapshot.json (agent details), which the
demo shows but BigQuery doesn't hold.

    python -m app.build_facts

Counted from the local data (DuckDB read-only, the model's test predictions, the Waymo Parquet
files) so Streamlit Cloud, which has no local data, shows the same numbers. Rerun after the data
changes; the file records when it was made.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "nhtsa.duckdb"
PROCESSED = ROOT / "data" / "processed"
WAYMO = PROCESSED / "waymo"
OUT = Path(__file__).resolve().parent / "facts.json"


def _one(con, sql: str):
    return con.sql(sql).fetchall()[0]


def build() -> dict:
    with duckdb.connect(str(DB), read_only=True) as con:
        records, complaints, c_first, c_last = _one(
            con, "SELECT count(*), count(DISTINCT odi_no), min(ldate), max(ldate) FROM stg.complaints")
        r_records, campaigns, r_first, r_last = _one(
            con, "SELECT count(*), count(DISTINCT camp_no), min(recall_date), max(recall_date) FROM stg.recalls")
        (investigations,) = _one(con, "SELECT count(DISTINCT ACTION_NUMBER) FROM raw.investigations")
        # per year, complete calendar years only (the last year in the data is still running)
        per_year_c = con.sql(f"""
            SELECT avg(n) FROM (SELECT year(ldate) y, count(DISTINCT odi_no) n FROM stg.complaints
                                WHERE year(ldate) < year(DATE '{c_last}') GROUP BY 1)""").fetchall()[0][0]
        per_year_r = con.sql(f"""
            SELECT avg(n) FROM (SELECT year(recall_date) y, count(DISTINCT camp_no) n FROM stg.recalls
                                WHERE year(recall_date) < year(DATE '{r_last}') GROUP BY 1)""").fetchall()[0][0]
    preds = (PROCESSED / "predictions_stg.parquet").as_posix()
    per_month, test_first, test_last = _one(duckdb.connect(), f"""
        SELECT avg(n), min(m), max(m) FROM (
            SELECT month AS m, count(*) AS n FROM read_parquet('{preds}') WHERE split = 'test' GROUP BY 1)""")
    scenes = {}
    for split in ("training", "validation"):
        files = sorted(WAYMO.glob(f"{split}_*_tracks.parquet"))
        paths = ", ".join(f"'{f.as_posix()}'" for f in files)
        scenes[split] = int(_one(duckdb.connect(), f"SELECT count(DISTINCT scenario_id) FROM read_parquet([{paths}])")[0])
    return {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "complaints": {"complaints": int(complaints), "records": int(records),
                       "first": str(c_first), "last": str(c_last), "per_year": round(float(per_year_c))},
        "recalls": {"campaigns": int(campaigns), "records": int(r_records),
                    "first": str(r_first), "last": str(r_last), "per_year": round(float(per_year_r))},
        "investigations": int(investigations),
        "model": {"car_parts_ranked_per_month": round(float(per_month)),
                  "test_first_month": str(test_first)[:10], "test_last_month": str(test_last)[:10]},
        "waymo": {"train_scenes": scenes["training"], "test_scenes": scenes["validation"]},
    }


def main() -> None:
    facts = build()
    OUT.write_text(json.dumps(facts, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(json.dumps(facts, indent=2))

    import pandas as pd

    from app import agent_data
    from src import bq_publish

    briefs = pd.DataFrame(bq_publish.build_agent_briefs())
    citations = pd.DataFrame(bq_publish.build_agent_citations())
    path = agent_data.write_snapshot(briefs, citations)
    print(f"wrote {path.relative_to(ROOT)} ({len(briefs)} briefs)")


if __name__ == "__main__":
    main()
