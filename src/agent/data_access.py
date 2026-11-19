"""Agent access to complaints and recalls through src/sources.py (contract column names).

Complaints and recalls come from Angelina's stg tables when they exist and pass their
contracts, otherwise from placeholder views over the raw tables. Column names follow
contracts/contracts.yml (odi_no, ldate, narrative, is_crash, ...).

Evidence cutoff: a watchlist row for month M uses complaints received by the END of M,
so the exclusive cutoff is the first day of the next month (``evidence_cutoff``).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from functools import lru_cache
from typing import Any

from src import sources
from src.contracts import load_contracts

ROOT = sources.ROOT
COMPLAINT_COLUMNS = tuple(load_contracts()["complaints"]["columns"])
RECALL_OUTPUT_COLUMNS = ("camp_no", "make", "model", "model_year", "component", "recall_date")
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


def _as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _before_date(value: str | date | datetime) -> str:
    return _as_date(value).isoformat()


def evidence_cutoff(month: str | date | datetime) -> str:
    """Exclusive cutoff for a watchlist month: the first day of the following month."""
    first = _as_date(month).replace(day=1)
    if first.month == 12:
        return first.replace(year=first.year + 1, month=1).isoformat()
    return first.replace(month=first.month + 1).isoformat()


def _limit(value: int) -> int:
    limit = int(value)
    if limit < 1:
        raise ValueError("limit must be a positive integer")
    return limit


@lru_cache(maxsize=16)
def _source_sql(name: str, db_path: str) -> str:
    del db_path  # part of the cache key only
    return sources.resolve(name).sql


def _sql(name: str) -> str:
    return _source_sql(name, str(sources.DB_PATH))


def _fetch_dicts(sql: str, params: Sequence[Any]) -> list[dict[str, Any]]:
    with sources.connect() as connection:
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


# Matches the watchlist's car + part to complaint rows, for real and placeholder tables:
# model is compared after Angelina's normalization (F-150 == F150), and a component matches
# exactly, by its top-level part, or as a prefix ("ENGINE" -> "ENGINE AND ENGINE COOLING").
_VEHICLE_FILTER = f"""
    ldate IS NOT NULL
    AND ldate < CAST(? AS DATE)
    AND make = upper(trim(?))
    AND {sources.normalize_model_sql('model')} = {sources.normalize_model_sql('?')}
    AND model_year = TRY_CAST(? AS INTEGER)
    AND (component = upper(trim(?)) OR component = upper(trim(?))
         OR component LIKE upper(trim(?)) || ' %')
"""


def _vehicle_params(make, model, model_year, component, before_date) -> list[Any]:
    root = component.split(":", maxsplit=1)[0].strip()
    return [_before_date(before_date), make, model, str(model_year), component, root, root]


def get_complaints(
    make: str,
    model: str,
    model_year: int | str,
    component: str,
    before_date: str | date | datetime,
    limit: int = 40,
) -> list[dict[str, Any]]:
    """Most recent complaint rows for the car + part received strictly before a date."""
    sql = f"""
        WITH complaints AS ({_sql("complaints")})
        SELECT {", ".join(COMPLAINT_COLUMNS)}
        FROM complaints
        WHERE {_VEHICLE_FILTER}
        ORDER BY ldate DESC, cmpl_id
        LIMIT ?
    """
    rows = _fetch_dicts(
        sql, [*_vehicle_params(make, model, model_year, component, before_date), _limit(limit)]
    )
    for row in rows:
        row["narrative"] = _redact_description(row["narrative"])
    return rows


def eligible_complaints(
    make: str,
    model: str,
    model_year: int | str,
    component: str,
    before_date: str | date | datetime,
) -> list[dict[str, Any]]:
    """Every (odi_no, ldate) for the car + part before a date; used to validate citations."""
    sql = f"""
        WITH complaints AS ({_sql("complaints")})
        SELECT DISTINCT odi_no, ldate
        FROM complaints
        WHERE {_VEHICLE_FILTER}
    """
    return _fetch_dicts(sql, _vehicle_params(make, model, model_year, component, before_date))


_COMPONENT_MATCH = """
    (component = upper(trim(?)) OR component = upper(trim(?))
     OR component LIKE upper(trim(?)) || ' %')
"""
SAME_MAKE = "same_make"
ALL_MAKES = "all_makes"


def find_similar_recalls(
    make: str,
    component: str,
    keywords: Sequence[str] | None,
    before_date: str | date | datetime,
    limit: int = 5,
) -> dict[str, Any]:
    """Earlier recalls for the same part: same make first, all makes only if none.

    Returns ``{"scope": "same_make" | "all_makes", "make": ..., "recalls": [...]}``. Within a
    scope, recalls whose text mentions a keyword come first, then the most recent.
    """
    root = component.split(":", maxsplit=1)[0].strip()
    terms = [term.strip() for term in (keywords or ()) if term and term.strip()]
    keyword_hit = (
        " OR ".join("upper(search_text) LIKE '%' || upper(?) || '%'" for _ in terms)
        if terms else "FALSE"
    )
    columns = ", ".join(RECALL_OUTPUT_COLUMNS)

    def search(same_make: bool) -> list[dict[str, Any]]:
        make_filter = "AND make = upper(trim(?))" if same_make else ""
        sql = f"""
            WITH recalls AS (
                SELECT *, coalesce(defect_summary, '') || ' '
                          || coalesce(consequence_summary, '') AS search_text
                FROM ({_sql("recalls")})
            ),
            matched AS (
                SELECT {columns}, left(any_value(defect_summary), 300) AS defect_summary,
                       bool_or({keyword_hit}) AS keyword_match
                FROM recalls
                WHERE recall_date IS NOT NULL
                  AND recall_date < CAST(? AS DATE)
                  AND {_COMPONENT_MATCH}
                  {make_filter}
                GROUP BY ALL
            )
            SELECT {columns}, defect_summary, keyword_match
            FROM matched
            ORDER BY keyword_match DESC, recall_date DESC, camp_no, model, model_year
            LIMIT ?
        """
        params: list[Any] = [*terms, _before_date(before_date), component, root, root]
        if same_make:
            params.append(make)
        params.append(_limit(limit))
        return _fetch_dicts(sql, params)

    same = search(same_make=True)
    if same:
        return {"scope": SAME_MAKE, "make": make.upper(), "recalls": same}
    return {"scope": ALL_MAKES, "make": make.upper(), "recalls": search(same_make=False)}


def _flag(value: Any) -> bool:
    return value in (True, 1, "1", "Y", "y", "true", "True")


def severity_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Compute severity totals from complaint rows, deduplicated by odi_no."""
    by_odino: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        key = str(row.get("odi_no") or f"row-{index}")
        existing = by_odino.setdefault(
            key, {"crash": False, "fire": False, "injuries": 0, "deaths": 0}
        )
        existing["crash"] = existing["crash"] or _flag(row.get("is_crash"))
        existing["fire"] = existing["fire"] or _flag(row.get("is_fire"))
        existing["injuries"] = max(existing["injuries"], int(row.get("injured_count") or 0))
        existing["deaths"] = max(existing["deaths"], int(row.get("deaths_count") or 0))
    return {
        "crashes": sum(item["crash"] for item in by_odino.values()),
        "fires": sum(item["fire"] for item in by_odino.values()),
        "injuries": sum(item["injuries"] for item in by_odino.values()),
        "deaths": sum(item["deaths"] for item in by_odino.values()),
    }
