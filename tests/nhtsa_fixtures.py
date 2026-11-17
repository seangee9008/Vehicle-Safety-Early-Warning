"""Tiny DuckDB databases shaped like the real raw and stg NHTSA tables, for offline tests."""

from __future__ import annotations

from pathlib import Path

import duckdb

RAW_COMPLAINT_COLUMNS = (
    "CMPLID", "ODINO", "MAKETXT", "MODELTXT", "YEARTXT", "COMPDESC", "LDATE", "CDESCR",
    "CRASH", "FIRE", "INJURED", "DEATHS", "MILES", "CMPL_TYPE",
)
RAW_RECALL_COLUMNS = (
    "CAMPNO", "MAKETXT", "MODELTXT", "YEARTXT", "COMPNAME", "RCDATE", "DESC_DEFECT",
    "CONSEQUENCE_DEFECT",
)
# Angelina's stg tables (scripts/nhtsa_data_table_cleaning.py), contract columns + a few extras.
STG_COMPLAINTS_DDL = """
CREATE TABLE stg.complaints (
    cmpl_id VARCHAR, odi_no VARCHAR, make VARCHAR, model VARCHAR, model_raw VARCHAR,
    model_year INTEGER, component VARCHAR, component_raw VARCHAR, ldate DATE,
    complaint_month DATE, is_crash INTEGER, is_fire INTEGER, injured_count INTEGER,
    deaths_count INTEGER, miles DOUBLE, narrative VARCHAR, cmpl_type VARCHAR
)
"""
STG_RECALLS_DDL = """
CREATE TABLE stg.recalls (
    record_id VARCHAR, camp_no VARCHAR, make VARCHAR, model VARCHAR, model_year INTEGER,
    component VARCHAR, recall_date DATE, defect_summary VARCHAR, consequence_summary VARCHAR
)
"""


def raw_complaint(odino, ldate, *, make="FORD", model="FUSION", year="2010",
                  comp="AIR BAGS:FRONTAL", text="Airbag warning light.", crash="N",
                  fire="N", injured="0", deaths="0", miles="50000", cmpl_type="IVOQ"):
    return (f"c{odino}", odino, make, model, year, comp, ldate, text, crash, fire,
            injured, deaths, miles, cmpl_type)


def raw_recall(campno, rcdate, *, make="FORD", model="FUSION", year="2010",
               comp="AIR BAGS", defect="Inflator defect.", consequence="May not deploy."):
    return (campno, make, model, year, comp, rcdate, defect, consequence)


def make_db(path: Path, complaints=(), recalls=(), *, stg_complaints=None, stg_recalls=None) -> Path:
    """raw.complaints / raw.recalls always; stg tables only when rows are given."""
    with duckdb.connect(str(path)) as con:
        con.execute("CREATE SCHEMA raw")
        con.execute(
            "CREATE TABLE raw.complaints ("
            + ", ".join(f"{c} VARCHAR" for c in RAW_COMPLAINT_COLUMNS) + ")"
        )
        con.execute(
            "CREATE TABLE raw.recalls (" + ", ".join(f"{c} VARCHAR" for c in RAW_RECALL_COLUMNS) + ")"
        )
        if complaints:
            marks = ", ".join("?" for _ in RAW_COMPLAINT_COLUMNS)
            con.executemany(f"INSERT INTO raw.complaints VALUES ({marks})", list(complaints))
        if recalls:
            marks = ", ".join("?" for _ in RAW_RECALL_COLUMNS)
            con.executemany(f"INSERT INTO raw.recalls VALUES ({marks})", list(recalls))
        if stg_complaints is not None or stg_recalls is not None:
            con.execute("CREATE SCHEMA stg")
        if stg_complaints is not None:
            con.execute(STG_COMPLAINTS_DDL)
        if stg_complaints:
            con.executemany(
                "INSERT INTO stg.complaints VALUES (" + ", ".join("?" * 17) + ")",
                list(stg_complaints),
            )
        if stg_recalls is not None:
            con.execute(STG_RECALLS_DDL)
        if stg_recalls:
            con.executemany(
                "INSERT INTO stg.recalls VALUES (" + ", ".join("?" * 9) + ")", list(stg_recalls)
            )
    return path


def stg_complaint(odino, ldate, *, make="FORD", model="F150", model_raw="F-150", year=2013,
                  comp="POWER TRAIN", comp_raw="POWER TRAIN:AUTOMATIC TRANSMISSION",
                  text="Transmission slipped.", crash=0, fire=0, injured=0, deaths=0):
    return (f"c{odino}", odino, make, model, model_raw, year, comp, comp_raw, ldate,
            ldate[:8] + "01", crash, fire, injured, deaths, 60000.0, text, "IVOQ")
