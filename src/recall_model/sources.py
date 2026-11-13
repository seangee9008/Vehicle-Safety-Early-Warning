"""Complaint and recall inputs for the recall model, from raw tables or the dbt stg_ tables.

Both versions return the stg_complaints / stg_recalls contract columns from AGENTS.md, so
nothing downstream knows which one it got. Pick one with --source raw|dbt.
"""
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "nhtsa.duckdb"
OUT = ROOT / "data" / "processed"

COMPLAINT_COLUMNS = ["complaint_id", "odino", "make", "model", "model_year", "component", "received_date",
                     "is_crash", "is_fire", "injured", "deaths", "miles", "narrative"]
RECALL_COLUMNS = ["campaign_id", "make", "model", "model_year", "component", "recall_date", "defect_summary"]


def component_sql(col):
    # Temporary part-name map for the known mismatches. The dbt map replaces it.
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
select CMPLID as complaint_id,
       ODINO as odino,
       upper(trim(MAKETXT)) as make,
       upper(trim(MODELTXT)) as model,
       try_cast(YEARTXT as int) as model_year,
       {component_sql('COMPDESC')} as component,
       try_strptime(LDATE, '%Y%m%d')::date as received_date,
       CRASH = 'Y' as is_crash,
       FIRE = 'Y' as is_fire,
       coalesce(try_cast(INJURED as int), 0) as injured,
       coalesce(try_cast(DEATHS as int), 0) as deaths,
       try_cast(MILES as int) as miles,
       CDESCR as narrative
from raw.complaints
where PROD_TYPE = 'V' and YEARTXT <> '9999'
  and coalesce(CMPL_TYPE, '') not in ('RC', 'EWR')
"""

RAW_RECALLS = f"""
select CAMPNO as campaign_id,
       upper(trim(MAKETXT)) as make,
       upper(trim(MODELTXT)) as model,
       try_cast(YEARTXT as int) as model_year,
       {component_sql('COMPNAME')} as component,
       try_strptime(RCDATE, '%Y%m%d')::date as recall_date,
       DESC_DEFECT as defect_summary
from raw.recalls
where RCLTYPECD = 'V' and YEARTXT <> '9999'
"""

# the contract doesn't fix types, so cast them to what the raw queries give
DBT_COMPLAINTS = """
select complaint_id::varchar as complaint_id, odino::varchar as odino, make, model,
       try_cast(model_year as int) as model_year, component,
       try_cast(received_date as date) as received_date,
       try_cast(is_crash as boolean) as is_crash, try_cast(is_fire as boolean) as is_fire,
       coalesce(try_cast(injured as int), 0) as injured, coalesce(try_cast(deaths as int), 0) as deaths,
       try_cast(miles as int) as miles, narrative
from main.stg_complaints
"""
DBT_RECALLS = """
select campaign_id::varchar as campaign_id, make, model, try_cast(model_year as int) as model_year,
       component, try_cast(recall_date as date) as recall_date, defect_summary
from main.stg_recalls
"""

SOURCES = {
    "raw": (RAW_COMPLAINTS, RAW_RECALLS),
    "dbt": (DBT_COMPLAINTS, DBT_RECALLS),
}


def get_sql(source):
    if source not in SOURCES:
        raise ValueError(f"source must be one of {list(SOURCES)}, got {source!r}")
    return SOURCES[source]


def connect():
    return duckdb.connect(str(DB), read_only=True)


def add_source_arg(parser):
    parser.add_argument("--source", choices=list(SOURCES), default="raw")
