"""Details for the AI agent tab that aren't in BigQuery: for each brief, how many complaints were on
file before the flag, complaints per month for the 12 months up to the flag, and the date and a
short excerpt of every cited complaint.

Read live from the local DuckDB (read-only, with the agent's own filters) when it's there;
otherwise from app/agent_snapshot.json, which `python -m app.build_facts` writes from the same
queries so Streamlit Cloud shows the same thing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

SNAPSHOT = Path(__file__).resolve().parent / "agent_snapshot.json"
EXCERPT_CHARS = 170


def _excerpt(text: str | None) -> str:
    from src.agent.data_access import _redact_description

    clean = " ".join(str(_redact_description(text) or "").split())
    if len(clean) <= EXCERPT_CHARS:
        return clean
    return clean[:EXCERPT_CHARS].rsplit(" ", 1)[0] + "…"


def build_live(briefs: pd.DataFrame, citations: pd.DataFrame | None) -> dict[str, dict]:
    """Query the local DuckDB for every brief (raises if the data isn't there)."""
    import duckdb

    from src import sources
    from src.agent import data_access

    out: dict[str, dict] = {}
    with sources.connect() as con:
        complaints_sql = sources.get_complaints(con).sql
        features_sql = sources.get_monthly_features(con).sql
        for _, b in briefs.iterrows():
            month = pd.Timestamp(b["month"])
            cutoff = data_access.evidence_cutoff(month.date())
            on_file = data_access.eligible_complaints(b["make"], b["model"], int(b["model_year"]),
                                                      b["component"], cutoff)
            monthly = con.execute(
                f"SELECT month, cmpl_1m FROM ({features_sql}) WHERE make = ? AND model = ? AND model_year = ? "
                "AND component = ? AND month BETWEEN ? AND ? ORDER BY month",
                [b["make"], b["model"], int(b["model_year"]), b["component"],
                 (month - pd.DateOffset(months=11)).date(), month.date()]).fetchall()
            ids = [] if citations is None else list(citations.loc[citations["brief_id"] == b["brief_id"], "odino"])
            cited = {}
            if ids:
                marks = ", ".join("?" for _ in ids)
                rows = con.execute(
                    f"SELECT odi_no, min(ldate), any_value(narrative) FROM ({complaints_sql}) "
                    f"WHERE odi_no IN ({marks}) GROUP BY odi_no", ids).fetchall()
                cited = {str(o): {"date": str(d)[:10], "excerpt": _excerpt(n)} for o, d, n in rows}
            out[b["brief_id"]] = {
                "complaints_on_file": len(on_file),
                "monthly": [{"month": str(m)[:10], "complaints": float(n or 0)} for m, n in monthly],
                "cited": cited,
            }
    if not out and len(briefs):
        raise duckdb.Error("no agent details found")
    return out


def load(briefs: pd.DataFrame, citations: pd.DataFrame | None, live: bool = True) -> tuple[dict, str]:
    """(details by brief_id, "local" | "snapshot" | "missing")."""
    if live:
        try:
            return build_live(briefs, citations), "local"
        except Exception as error:  # noqa: BLE001 - no local DuckDB (e.g. Streamlit Cloud): use the snapshot
            logging.getLogger(__name__).info("agent details from the snapshot: %s", error)
    if SNAPSHOT.exists():
        return json.loads(SNAPSHOT.read_text(encoding="utf-8")).get("briefs", {}), "snapshot"
    return {}, "missing"


def write_snapshot(briefs: pd.DataFrame, citations: pd.DataFrame | None) -> Path:
    from datetime import UTC, datetime

    data = {"generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "briefs": build_live(briefs, citations)}
    SNAPSHOT.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return SNAPSHOT
