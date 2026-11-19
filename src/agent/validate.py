"""Validate ODINO citations and evidence bullets in generated briefs."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import date, datetime
from typing import Any

# One bracket pair may hold one or more IDs separated by commas or semicolons, with
# the ODINO prefix optional after the first: [ODINO 111], [ODINO 111, 222],
# [ODINO 111; ODINO 222]. Adjacent pairs like [ODINO 111][ODINO 222] match separately.
CITATION_RE = re.compile(
    r"\[\s*ODINO\s+(\d+(?:\s*[,;]\s*(?:ODINO\s+)?\d+)*)\s*\]",
    re.IGNORECASE,
)
_ID_RE = re.compile(r"\d+")
BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+")


def extract_citations(text: str) -> list[str]:
    """Return every cited ODINO ID in order, expanding grouped citations."""
    return [
        odino
        for group in CITATION_RE.findall(text)
        for odino in _ID_RE.findall(group)
    ]


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
        str(odino)
        for row in complaints
        if (odino := row.get("odi_no") or row.get("odino") or row.get("ODINO"))
        if (received := _as_date(row.get("ldate") or row.get("received_date") or row.get("LDATE")))
        is not None
        and received < cutoff
    }
    citations = extract_citations(brief)
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


MIN_EVIDENCE_BULLETS = 3
MAX_EVIDENCE_BULLETS = 6


def validation_errors(validation: Mapping[str, Any]) -> list[str]:
    """Exact, human-readable reasons a brief fails; empty when it passes."""
    errors = []
    if validation["citations_invalid"]:
        errors.append(
            f"{validation['citations_invalid']} citation(s) are not complaints for this vehicle "
            f"and component received before the cutoff: "
            + ", ".join(f"ODINO {odino}" for odino in validation["invalid_odinos"].split(";") if odino)
            + ". Cite only ODINO IDs returned by get_complaints, copied digit for digit."
        )
    if validation["bullets_without_citations"]:
        errors.append(
            f"{validation['bullets_without_citations']} Evidence bullet(s) have no [ODINO ...] "
            f"citation: {validation['uncited_bullets']}"
        )
    bullets = validation["evidence_bullets"]
    if not MIN_EVIDENCE_BULLETS <= bullets <= MAX_EVIDENCE_BULLETS:
        errors.append(
            f"The Evidence section has {bullets} bullets; it must have "
            f"{MIN_EVIDENCE_BULLETS}-{MAX_EVIDENCE_BULLETS}. Merge or drop bullets as needed."
        )
    return errors
