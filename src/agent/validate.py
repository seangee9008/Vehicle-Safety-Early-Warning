"""Validate ODINO citations and evidence bullets in generated briefs."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

CITATION_RE = re.compile(r"\[ODINO\s+(\d+)\]", re.IGNORECASE)
BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+")


def _as_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def _evidence_bullets(brief: str) -> list[str]:
    lines = brief.splitlines()
    evidence_lines: list[str] = []
    in_evidence = False
    saw_evidence_heading = False
    for line in lines:
        if line.lstrip().startswith("#"):
            heading = line.lstrip("# ").strip().lower()
            if heading == "evidence":
                in_evidence = True
                saw_evidence_heading = True
                continue
            if in_evidence:
                break
        if in_evidence:
            evidence_lines.append(line)
    if not saw_evidence_heading:
        evidence_lines = lines
    return [line for line in evidence_lines if BULLET_RE.match(line)]


def validate_brief(
    brief: str,
    complaints: Sequence[Mapping[str, Any]],
    before_date: str | date | datetime,
) -> dict[str, Any]:
    """Count citations and reject IDs not present strictly before the watchlist date."""
    cutoff = _as_date(before_date)
    if cutoff is None:
        raise ValueError(f"Invalid watchlist date: {before_date!r}")

    eligible_ids = {
        str(row.get("odino") or row.get("ODINO"))
        for row in complaints
        if row.get("odino") or row.get("ODINO")
        if (received := _as_date(row.get("received_date") or row.get("LDATE"))) is not None
        and received < cutoff
    }
    citations = CITATION_RE.findall(brief)
    valid_ids = [citation for citation in citations if citation in eligible_ids]
    evidence_bullets = _evidence_bullets(brief)
    uncited_bullets = [line.strip() for line in evidence_bullets if not CITATION_RE.search(line)]
    invalid_ids = [citation for citation in citations if citation not in eligible_ids]
    return {
        "citations_total": len(citations),
        "citations_valid": len(valid_ids),
        "citations_invalid": len(invalid_ids),
        "evidence_bullets": len(evidence_bullets),
        "bullets_without_citations": len(uncited_bullets),
        "invalid_odinos": ";".join(invalid_ids),
        "uncited_bullets": " | ".join(uncited_bullets),
    }
