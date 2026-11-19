"""One place to read the four hand-off tables, real or placeholder.

Each getter returns a ``Source`` whose ``sql`` is a SELECT with the contract columns
(contracts/contracts.yml). It uses the teammate's real table when it exists and passes its
contract; otherwise a placeholder (a SQL view over the raw NHTSA tables, or the sample
watchlist CSV), and it logs which one it picked.

    python -m src.sources          # real vs placeholder + row counts per table
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import duckdb

from src.contracts import check_contract, load_contracts

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.environ.get("NHTSA_DUCKDB_PATH", ROOT / "data" / "nhtsa.duckdb"))
SAMPLE_WATCHLIST = ROOT / "src" / "agent" / "fixtures" / "watchlist_sample.csv"

log = logging.getLogger(__name__)


def normalize_model_sql(expr: str) -> str:
    """Same model normalization as Angelina's standardize_model (F-150 -> F150)."""
    return rf"regexp_replace(upper(trim(coalesce(CAST({expr} AS VARCHAR), ''))), '[\s\-_./]+', '', 'g')"


def _top_component_sql(expr: str) -> str:
    return f"upper(trim(split_part(coalesce({expr}, ''), ':', 1)))"


PLACEHOLDER_COMPLAINTS = f"""
SELECT CMPLID AS cmpl_id,
       ODINO AS odi_no,
       upper(trim(MAKETXT)) AS make,
       {normalize_model_sql('MODELTXT')} AS model,
       TRY_CAST(YEARTXT AS INTEGER) AS model_year,
       {_top_component_sql('COMPDESC')} AS component,
       TRY_STRPTIME(LDATE, '%Y%m%d')::DATE AS ldate,
       CASE WHEN upper(trim(CRASH)) = 'Y' THEN 1 ELSE 0 END AS is_crash,
       CASE WHEN upper(trim(FIRE)) = 'Y' THEN 1 ELSE 0 END AS is_fire,
       coalesce(TRY_CAST(INJURED AS INTEGER), 0) AS injured_count,
       coalesce(TRY_CAST(DEATHS AS INTEGER), 0) AS deaths_count,
       TRY_CAST(MILES AS DOUBLE) AS miles,
       CDESCR AS narrative
FROM raw.complaints
WHERE coalesce(upper(trim(CMPL_TYPE)), '') NOT IN ('RC', 'EWR')
  AND TRY_STRPTIME(LDATE, '%Y%m%d') IS NOT NULL
"""

PLACEHOLDER_RECALLS = f"""
SELECT upper(trim(CAMPNO)) AS camp_no,
       upper(trim(MAKETXT)) AS make,
       {normalize_model_sql('MODELTXT')} AS model,
       TRY_CAST(YEARTXT AS INTEGER) AS model_year,
       {_top_component_sql('COMPNAME')} AS component,
       TRY_STRPTIME(RCDATE, '%Y%m%d')::DATE AS recall_date,
       DESC_DEFECT AS defect_summary,
       CONSEQUENCE_DEFECT AS consequence_summary
FROM raw.recalls
WHERE TRY_STRPTIME(RCDATE, '%Y%m%d') IS NOT NULL
"""

# Contract columns only, from the placeholder complaints. Sparse (months with complaints),
# counts are distinct complaints, and label is unknown (NULL).
PLACEHOLDER_MONTHLY_FEATURES = f"""
WITH m AS (
    SELECT make, model, model_year, component,
           date_trunc('month', ldate)::DATE AS month,
           count(DISTINCT odi_no) AS n,
           count(DISTINCT odi_no) FILTER (
               WHERE is_crash = 1 OR is_fire = 1 OR injured_count > 0 OR deaths_count > 0
           ) AS n_severe
    FROM ({PLACEHOLDER_COMPLAINTS})
    WHERE model_year IS NOT NULL
    GROUP BY ALL
)
SELECT make, model, model_year, component, month,
       n::DOUBLE AS cmpl_1m,
       sum(n) OVER w3 ::DOUBLE AS cmpl_3m,
       sum(n) OVER w6 ::DOUBLE AS cmpl_6m,
       sum(n_severe) OVER w3 ::DOUBLE AS severe_3m,
       (year(month) - model_year)::INTEGER AS vehicle_age,
       NULL::BOOLEAN AS label
FROM m
WINDOW
    w3 AS (PARTITION BY make, model, model_year, component ORDER BY month
           RANGE BETWEEN INTERVAL 2 MONTH PRECEDING AND CURRENT ROW),
    w6 AS (PARTITION BY make, model, model_year, component ORDER BY month
           RANGE BETWEEN INTERVAL 5 MONTH PRECEDING AND CURRENT ROW)
"""


def placeholder_watchlist_sql(path: Path = SAMPLE_WATCHLIST) -> str:
    csv = str(path).replace("'", "''")
    return f"""
SELECT CAST(month AS DATE) AS month, CAST(make AS VARCHAR) AS make,
       CAST(model AS VARCHAR) AS model, CAST(model_year AS INTEGER) AS model_year,
       CAST(component AS VARCHAR) AS component, CAST(score AS DOUBLE) AS score,
       CAST(rank AS INTEGER) AS rank, CAST(top_reasons AS VARCHAR) AS top_reasons
FROM read_csv('{csv}', header = true, all_varchar = true)
"""


PLACEHOLDERS = {
    "complaints": (lambda: PLACEHOLDER_COMPLAINTS, "SQL view over raw.complaints"),
    "recalls": (lambda: PLACEHOLDER_RECALLS, "SQL view over raw.recalls"),
    "monthly_features": (lambda: PLACEHOLDER_MONTHLY_FEATURES, "SQL view over raw.complaints"),
    "watchlist": (placeholder_watchlist_sql, "sample CSV src/agent/fixtures/watchlist_sample.csv"),
}


@dataclass(frozen=True)
class Source:
    name: str
    kind: str  # "real" or "placeholder"
    relation: str  # real table name, or what the placeholder is
    sql: str  # SELECT with (at least) the contract columns
    problem: str = ""  # why the real table was not used

    @property
    def is_real(self) -> bool:
        return self.kind == "real"


def connect(read_only: bool = True) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(DB_PATH), read_only=read_only)


def resolve(name: str, con: duckdb.DuckDBPyConnection | None = None) -> Source:
    contracts = load_contracts()
    if name not in contracts:
        raise KeyError(f"unknown contract {name!r}; expected one of {list(contracts)}")
    table = contracts[name]["table"]
    owns = con is None
    con = con or connect()
    try:
        result = check_contract(con, name, contracts=contracts)
    finally:
        if owns:
            con.close()
    if result.ok:
        source = Source(name, "real", table, f"SELECT * FROM {table}")
        log.info("%s: using real table %s", name, table)
        return source
    placeholder_sql, description = PLACEHOLDERS[name]
    problem = result.describe()
    source = Source(name, "placeholder", description, placeholder_sql(), problem)
    if result.exists:
        log.warning("%s: %s; using placeholder (%s)", name, problem, description)
    else:
        log.info("%s: %s not built yet; using placeholder (%s)", name, table, description)
    return source


def get_complaints(con: duckdb.DuckDBPyConnection | None = None) -> Source:
    return resolve("complaints", con)


def get_recalls(con: duckdb.DuckDBPyConnection | None = None) -> Source:
    return resolve("recalls", con)


def get_monthly_features(con: duckdb.DuckDBPyConnection | None = None) -> Source:
    return resolve("monthly_features", con)


def get_watchlist(con: duckdb.DuckDBPyConnection | None = None) -> Source:
    return resolve("watchlist", con)


def status() -> list[dict[str, object]]:
    """Real or placeholder, relation and row count for each contract table."""
    rows = []
    with connect() as con:
        for name in load_contracts():
            source = resolve(name, con)
            try:
                count = con.sql(f"SELECT count(*) FROM ({source.sql})").fetchall()[0][0]
            except duckdb.Error as error:  # e.g. raw tables not loaded yet
                count = None
                source = Source(name, source.kind, source.relation, source.sql, str(error))
            rows.append({
                "table": name,
                "kind": source.kind,
                "relation": source.relation,
                "rows": count,
                "note": source.problem if source.kind == "placeholder" else "",
            })
    return rows


def main() -> None:
    for row in status():
        rows = "?" if row["rows"] is None else f"{row['rows']:,}"
        note = f"  ({row['note']})" if row["note"] else ""
        print(f"{row['table']:<17} {row['kind']:<12} {rows:>11} rows  {row['relation']}{note}")


if __name__ == "__main__":
    main()
