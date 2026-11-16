"""Complaint and recall inputs for the recall model, from raw tables or the cleaned stg tables.

Both versions return the same columns, named like Angelina's stg.complaints / stg.recalls
(built by scripts/nhtsa_data_table_cleaning.py), so nothing downstream knows which one it got.
Pick one with --source raw|stg.
"""
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "nhtsa.duckdb"
OUT = ROOT / "data" / "processed"

COMPLAINT_COLUMNS = ["cmpl_id", "odi_no", "make", "model", "model_year", "component", "ldate",
                     "is_crash", "is_fire", "injured_count", "deaths_count", "miles", "narrative"]
RECALL_COLUMNS = ["camp_no", "make", "model", "model_year", "component", "recall_date", "defect_summary"]


def component_sql(col):
    # Temporary part-name map for the known mismatches, raw path only.
    # Shared names are the newer complaint names, so the agent's LIKE match on raw complaints finds them.
    part = f"upper(trim(split_part({col}, ':', 1)))"
    return f"""case
        when {part} like 'FUEL%' then 'FUEL/PROPULSION SYSTEM'
        when {part} like 'ELECTRONIC STABILITY CONTROL%' then 'ELECTRONIC STABILITY CONTROL (ESC)'
        when {part} = 'ENGINE AND ENGINE COOLING' then 'ENGINE'
        when {part} like 'SERVICE BRAKES%' then 'SERVICE BRAKES'
        when {part} = 'VISIBILITY/WIPER' then 'VISIBILITY'
        else {part} end"""


RAW_COMPLAINTS = f"""
select CMPLID as cmpl_id,
       ODINO as odi_no,
       upper(trim(MAKETXT)) as make,
       upper(trim(MODELTXT)) as model,
       try_cast(YEARTXT as int) as model_year,
       {component_sql('COMPDESC')} as component,
       try_strptime(LDATE, '%Y%m%d')::date as ldate,
       (CRASH = 'Y')::int as is_crash,
       (FIRE = 'Y')::int as is_fire,
       coalesce(try_cast(INJURED as int), 0) as injured_count,
       coalesce(try_cast(DEATHS as int), 0) as deaths_count,
       try_cast(MILES as double) as miles,
       CDESCR as narrative
from raw.complaints
where PROD_TYPE = 'V' and YEARTXT <> '9999'
  and coalesce(CMPL_TYPE, '') not in ('RC', 'EWR')
"""

RAW_RECALLS = f"""
select CAMPNO as camp_no,
       upper(trim(MAKETXT)) as make,
       upper(trim(MODELTXT)) as model,
       try_cast(YEARTXT as int) as model_year,
       {component_sql('COMPNAME')} as component,
       try_strptime(RCDATE, '%Y%m%d')::date as recall_date,
       DESC_DEFECT as defect_summary
from raw.recalls
where RCLTYPECD = 'V' and YEARTXT <> '9999'
"""

# casts keep the types the same as the raw queries even if the stg types change
STG_COMPLAINTS = """
select cmpl_id::varchar as cmpl_id, odi_no::varchar as odi_no, make, model,
       try_cast(model_year as int) as model_year, component,
       try_cast(ldate as date) as ldate,
       try_cast(is_crash as int) as is_crash, try_cast(is_fire as int) as is_fire,
       coalesce(try_cast(injured_count as int), 0) as injured_count,
       coalesce(try_cast(deaths_count as int), 0) as deaths_count,
       try_cast(miles as double) as miles, narrative
from stg.complaints
"""
STG_RECALLS = """
select camp_no::varchar as camp_no, make, model, try_cast(model_year as int) as model_year,
       component, try_cast(recall_date as date) as recall_date, defect_summary
from stg.recalls
"""

SOURCES = {
    "raw": (RAW_COMPLAINTS, RAW_RECALLS),
    "stg": (STG_COMPLAINTS, STG_RECALLS),
}


def get_sql(source):
    if source not in SOURCES:
        raise ValueError(f"source must be one of {list(SOURCES)}, got {source!r}")
    return SOURCES[source]


def connect():
    return duckdb.connect(str(DB), read_only=True)


def check_columns(con, source):
    # fail loudly if a stg table is missing a column the model reads
    if source == "raw":
        return
    for table, cols in [("stg.complaints", COMPLAINT_COLUMNS), ("stg.recalls", RECALL_COLUMNS)]:
        have = {r[0] for r in con.sql(f"describe {table}").fetchall()}
        missing = [c for c in cols if c not in have]
        if missing:
            raise SystemExit(f"{table} is missing columns {missing}; run scripts/nhtsa_data_table_cleaning.py "
                             f"or check the column names with the data lane")


def add_source_arg(parser):
    parser.add_argument("--source", choices=list(SOURCES), default="raw")
