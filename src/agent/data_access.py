"""Contract 1 access for NHTSA complaints and historical recalls."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = Path(os.environ.get("NHTSA_DUCKDB_PATH", ROOT / "data" / "nhtsa.duckdb"))

CONTRACT_1_COLUMNS = (
    "complaint_id",
    "odino",
    "make",
    "model",
    "model_year",
    "component",
    "received_date",
    "description",
    "crash",
    "fire",
    "injuries",
    "deaths",
)
_PII_PATTERNS = (
    (re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE), "[REDACTED EMAIL]"),
    (re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE), "[REDACTED URL]"),
    (re.compile(r"\b[A-HJ-NPR-Z0-9]{17}\b", re.IGNORECASE), "[REDACTED VIN]"),
    (
        re.compile(
            r"(?<!\w)(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})[\s.-]\d{3}[\s.-]\d{4}(?!\w)"
        ),
        "[REDACTED PHONE]",
    ),
    (
        re.compile(
            r"\b\d{1,6}\s+(?:[\w.'-]+\s+){0,5}"
            r"(?:street|st|avenue|ave|road|rd|boulevard|blvd|drive|dr|"
            r"lane|ln|court|ct|highway|hwy)\b",
            re.IGNORECASE,
        ),
        "[REDACTED ADDRESS]",
    ),
)


def _before_date(value: str | date | datetime) -> str:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return date.fromisoformat(str(value)[:10]).isoformat()


def _limit(value: int) -> int:
    limit = int(value)
    if limit < 1:
        raise ValueError("limit must be a positive integer")
    return limit


def _raw_complaints_sql() -> str:
    return """
        SELECT
            NULLIF(TRIM(CMPLID), '') AS complaint_id,
            NULLIF(TRIM(ODINO), '') AS odino,
            NULLIF(UPPER(TRIM(MAKETXT)), '') AS make,
            NULLIF(UPPER(TRIM(MODELTXT)), '') AS model,
            TRY_CAST(NULLIF(TRIM(YEARTXT), '') AS INTEGER) AS model_year,
            NULLIF(UPPER(TRIM(COMPDESC)), '') AS component,
            TRY_STRPTIME(NULLIF(TRIM(LDATE), ''), '%Y%m%d')::DATE AS received_date,
            NULLIF(TRIM(CDESCR), '') AS description,
            CASE UPPER(TRIM(CRASH))
                WHEN 'Y' THEN TRUE WHEN 'N' THEN FALSE ELSE NULL
            END AS crash,
            CASE UPPER(TRIM(FIRE))
                WHEN 'Y' THEN TRUE WHEN 'N' THEN FALSE ELSE NULL
            END AS fire,
            COALESCE(TRY_CAST(NULLIF(TRIM(INJURED), '') AS INTEGER), 0) AS injuries,
            COALESCE(TRY_CAST(NULLIF(TRIM(DEATHS), '') AS INTEGER), 0) AS deaths
        FROM raw.complaints
    """


def _stg_complaints_sql() -> str:
    return """
        SELECT
            NULLIF(TRIM(CAST(complaint_id AS VARCHAR)), '') AS complaint_id,
            NULLIF(TRIM(CAST(odino AS VARCHAR)), '') AS odino,
            NULLIF(UPPER(TRIM(CAST(make AS VARCHAR))), '') AS make,
            NULLIF(UPPER(TRIM(CAST(model AS VARCHAR))), '') AS model,
            TRY_CAST(model_year AS INTEGER) AS model_year,
            NULLIF(UPPER(TRIM(CAST(component AS VARCHAR))), '') AS component,
            TRY_CAST(received_date AS DATE) AS received_date,
            NULLIF(TRIM(CAST(description AS VARCHAR)), '') AS description,
            CASE LOWER(TRIM(CAST(crash AS VARCHAR)))
                WHEN 'true' THEN TRUE WHEN 't' THEN TRUE WHEN 'y' THEN TRUE WHEN '1' THEN TRUE
                WHEN 'false' THEN FALSE WHEN 'f' THEN FALSE WHEN 'n' THEN FALSE WHEN '0' THEN FALSE
                ELSE NULL
            END AS crash,
            CASE LOWER(TRIM(CAST(fire AS VARCHAR)))
                WHEN 'true' THEN TRUE WHEN 't' THEN TRUE WHEN 'y' THEN TRUE WHEN '1' THEN TRUE
                WHEN 'false' THEN FALSE WHEN 'f' THEN FALSE WHEN 'n' THEN FALSE WHEN '0' THEN FALSE
                ELSE NULL
            END AS fire,
            COALESCE(TRY_CAST(injuries AS INTEGER), 0) AS injuries,
            COALESCE(TRY_CAST(deaths AS INTEGER), 0) AS deaths
        FROM main.stg_complaints
    """


def _complaint_source_sql() -> str:
    source = os.environ.get("AGENT_SOURCE", "raw").strip().lower()
    if source == "raw":
        return _raw_complaints_sql()
    if source == "stg":
        return _stg_complaints_sql()
    raise ValueError(f"AGENT_SOURCE must be 'raw' or 'stg', got {source!r}")


def _fetch_dicts(sql: str, params: Sequence[Any]) -> list[dict[str, Any]]:
    with duckdb.connect(str(DB_PATH), read_only=True) as connection:
        cursor = connection.execute(sql, list(params))
        names = [column[0] for column in cursor.description]
        rows = cursor.fetchall()
    return [
        {
            name: value.isoformat() if isinstance(value, (date, datetime)) else value
            for name, value in zip(names, row)
        }
        for row in rows
    ]


def _redact_description(value: str | None) -> str | None:
    if value is None:
        return None
    for pattern, replacement in _PII_PATTERNS:
        value = pattern.sub(replacement, value)
    return value


def get_complaints(
    make: str,
    model: str,
    model_year: int | str,
    component: str,
    before_date: str | date | datetime,
    limit: int = 40,
) -> list[dict[str, Any]]:
    """Return cleaned Contract 1 complaint rows received strictly before a date."""
    root_component = component.split(":", maxsplit=1)[0].strip()
    sql = f"""
        WITH complaints AS ({_complaint_source_sql()})
        SELECT {", ".join(CONTRACT_1_COLUMNS)}
        FROM complaints
        WHERE received_date IS NOT NULL
          AND received_date < CAST(? AS DATE)
          AND make = UPPER(TRIM(?))
          AND model = UPPER(TRIM(?))
          AND model_year = TRY_CAST(? AS INTEGER)
          AND (
              UPPER(COALESCE(component, '')) LIKE '%' || UPPER(TRIM(?)) || '%'
              OR UPPER(COALESCE(component, '')) LIKE '%' || UPPER(TRIM(?)) || '%'
          )
        ORDER BY received_date DESC, complaint_id
        LIMIT ?
    """
    rows = _fetch_dicts(
        sql,
        [
            _before_date(before_date),
            make,
            model,
            str(model_year),
            component,
            root_component,
            _limit(limit),
        ],
    )
    for row in rows:
        row["description"] = _redact_description(row["description"])
    return rows


def find_similar_recalls(
    component: str,
    keywords: Sequence[str] | None,
    before_date: str | date | datetime,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Find recalls with matching component/keywords announced before a date."""
    terms = [term.strip() for term in (keywords or ()) if term and term.strip()]
    terms.extend(part.strip() for part in component.split(":") if part.strip())
    terms = list(dict.fromkeys(terms))
    if not terms:
        raise ValueError("component or keywords must contain a search term")
    where_terms = " OR ".join(
        "UPPER(search_text) LIKE '%' || UPPER(?) || '%'" for _ in terms
    )
    sql = f"""
        WITH recalls AS (
            SELECT
                NULLIF(TRIM(CAMPNO), '') AS campaign_id,
                NULLIF(UPPER(TRIM(MAKETXT)), '') AS make,
                NULLIF(UPPER(TRIM(MODELTXT)), '') AS model,
                TRY_CAST(NULLIF(TRIM(YEARTXT), '') AS INTEGER) AS model_year,
                NULLIF(UPPER(TRIM(COMPNAME)), '') AS component,
                TRY_STRPTIME(NULLIF(TRIM(RCDATE), ''), '%Y%m%d')::DATE AS recall_date,
                COALESCE(COMPNAME, '') || ' ' || COALESCE(DESC_DEFECT, '') || ' '
                    || COALESCE(CONSEQUENCE_DEFECT, '') AS search_text
            FROM raw.recalls
        )
        SELECT campaign_id, make, model, model_year, component, recall_date
        FROM recalls
        WHERE recall_date IS NOT NULL
          AND recall_date < CAST(? AS DATE)
          AND ({where_terms})
        ORDER BY recall_date DESC, campaign_id
        LIMIT ?
    """
    return _fetch_dicts(
        sql,
        [_before_date(before_date), *terms, _limit(limit)],
    )


def severity_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Compute severity totals from complaint rows, deduplicated by ODINO."""
    by_odino: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        key = str(row.get("odino") or row.get("ODINO") or f"row-{index}")
        existing = by_odino.setdefault(
            key, {"crash": False, "fire": False, "injuries": 0, "deaths": 0}
        )
        existing["crash"] = existing["crash"] or row.get("crash") in (True, "Y", "y", "1", 1)
        existing["fire"] = existing["fire"] or row.get("fire") in (True, "Y", "y", "1", 1)
        for field in ("injuries", "deaths"):
            existing[field] = max(existing[field], int(row.get(field) or 0))
    return {
        "crashes": sum(item["crash"] for item in by_odino.values()),
        "fires": sum(item["fire"] for item in by_odino.values()),
        "injuries": sum(item["injuries"] for item in by_odino.values()),
        "deaths": sum(item["deaths"] for item in by_odino.values()),
    }
