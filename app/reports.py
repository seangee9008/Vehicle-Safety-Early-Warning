"""Results that live only in the committed reports/ files (shipped with the repo, so they also work
on Streamlit Cloud). Each reader returns None when its file or table isn't there."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def _csv(name: str) -> pd.DataFrame | None:
    path = REPORTS / name
    return pd.read_csv(path) if path.exists() else None


def overlap(top: int = 20) -> pd.Series | None:
    """Who caught which recalls at a list size, and the median lead time of each group."""
    df = _csv("recall_overlap_stg.csv")
    if df is None or (df["top"] == top).sum() == 0:
        return None
    return df[df["top"] == top].iloc[0]


def miss_reasons() -> pd.DataFrame | None:
    return _csv("recall_miss_reasons_stg.csv")


def _md_table(text: str, start_marker: str, marker_is_header: bool = False) -> list[list[str]]:
    """Rows (cells) of the first Markdown table at/after start_marker, header and rule excluded."""
    start = text.find(start_marker)
    if start < 0:
        return []
    rows = []
    for line in text[start:].splitlines()[1:]:
        if line.startswith("|"):
            cells = [c.strip().strip("*").strip() for c in line.strip().strip("|").split("|")]
            if not all(set(c) <= set("-: ") for c in cells):
                rows.append(cells)
        elif rows:
            break
    return rows if marker_is_header else rows[1:]  # drop the header row


def _pct(cell: str) -> list[float]:
    return [float(x) / 100 for x in re.findall(r"([\d.]+)%", cell)]


def ablation() -> pd.DataFrame | None:
    """Vehicle age only vs complaints only vs both (the committed recall write-up, section 7)."""
    path = REPORTS / "recall_results_stg.md"
    if not path.exists():
        return None
    rows = _md_table(path.read_text(encoding="utf-8"), "## 7. Vehicle age vs. complaints")
    out = []
    for cells in rows:
        if len(cells) < 5:
            continue
        p = _pct(cells[1])
        out.append({"model": cells[0], "hit_rate_20": p[0] if p else None,
                    "hit_rate_100": p[1] if len(p) > 1 else None,
                    "pr_auc": float(cells[4]) if re.fullmatch(r"[\d.]+", cells[4]) else None})
    return pd.DataFrame(out) if out else None


def label_audit_misses() -> pd.DataFrame | None:
    """How many of the model's misses each looser label match would turn into hits."""
    path = REPORTS / "label_audit.md"
    if not path.exists():
        return None
    rows = _md_table(path.read_text(encoding="utf-8"), "| Looser match |", marker_is_header=True)
    out = [{"match": c[0], "misses": int(c[1]), "share": _pct(c[2])[0]}
           for c in rows if len(c) >= 3 and c[1].isdigit()]
    return pd.DataFrame(out) if out else None
