"""Generate and validate investigation briefs from a Contract 3 watchlist CSV."""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google.genai.errors import APIError

from src.agent.brief import DEFAULT_MODEL, create_brief
from src.agent.data_access import ROOT

WATCHLIST_COLUMNS = (
    "month",
    "make",
    "model",
    "model_year",
    "component",
    "score",
    "rank",
    "top_reasons",
)
VALIDATION_COLUMNS = (
    "make",
    "model",
    "model_year",
    "component",
    "month",
    "model_used",
    "citations_total",
    "citations_valid",
    "citations_invalid",
    "evidence_bullets",
    "bullets_without_citations",
    "invalid_odinos",
    "uncited_bullets",
)
BRIEFS_DIR = ROOT / "reports" / "briefs"

load_dotenv(ROOT / ".env")


def _read_watchlist(path: Path, top: int) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or tuple(reader.fieldnames) != WATCHLIST_COLUMNS:
            raise ValueError(
                f"Watchlist must have exactly these Contract 3 columns in order: "
                f"{', '.join(WATCHLIST_COLUMNS)}"
            )
        return [dict(row) for row in list(reader)[:top]]


def _slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _brief_path(item: dict[str, str]) -> Path:
    filename = "_".join(
        _slug(item[field]) for field in ("make", "model", "model_year", "component")
    )
    return BRIEFS_DIR / f"{filename}.md"


def run(watchlist_path: Path, top: int) -> int:
    if top < 1:
        raise ValueError("--top must be a positive integer")
    items = _read_watchlist(watchlist_path, top)
    if not items:
        raise ValueError(f"No watchlist rows found in {watchlist_path}")
    BRIEFS_DIR.mkdir(parents=True, exist_ok=True)
    validation_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for item in items:
        label = f"{item['make']} {item['model']} {item['model_year']} {item['component']}"
        try:
            brief, validation = create_brief(item)
            path = _brief_path(item)
            path.write_text(brief, encoding="utf-8")
            result = {
                **{field: item[field] for field in ("make", "model", "model_year", "component", "month")},
                "model_used": os.environ.get("GEMINI_MODEL", DEFAULT_MODEL),
                **validation,
            }
            validation_rows.append(result)
            print(
                f"{label}: saved {path.relative_to(ROOT)} "
                f"(citations {validation['citations_valid']}/{validation['citations_total']} valid, "
                f"{validation['bullets_without_citations']} uncited evidence bullets)"
            )
            if (
                validation["citations_invalid"]
                or validation["bullets_without_citations"]
                or not 3 <= validation["evidence_bullets"] <= 6
            ):
                failures.append(
                    f"{label}: brief failed citation/evidence checks "
                    f"(invalid={validation['citations_invalid']}, "
                    f"uncited={validation['bullets_without_citations']}, "
                    f"evidence_bullets={validation['evidence_bullets']})"
                )
        except (APIError, OSError, RuntimeError, ValueError) as error:
            failures.append(f"{label}: {error}")
            print(f"ERROR {label}: {error}", file=sys.stderr)

    validation_path = BRIEFS_DIR / "validation.csv"
    if validation_rows:
        with validation_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=VALIDATION_COLUMNS)
            writer.writeheader()
            writer.writerows(validation_rows)
        print(f"Validation results: {validation_path.relative_to(ROOT)}")
    if failures:
        raise RuntimeError(f"{len(failures)} brief(s) failed:\n" + "\n".join(failures))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--watchlist",
        type=Path,
        default=Path("src/agent/fixtures/watchlist_sample.csv"),
    )
    parser.add_argument("--top", type=int, default=5)
    args = parser.parse_args()
    try:
        raise SystemExit(run(args.watchlist, args.top))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
