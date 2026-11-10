"""Generate and validate investigation briefs from a Contract 3 watchlist CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google.genai.errors import APIError

from src.agent import data_access
from src.agent.brief import COMPLAINT_LIMIT, create_brief
from src.agent.data_access import ROOT
from src.agent.validate import validate_brief

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
    "source",
    "citations_total",
    "citations_valid",
    "citations_invalid",
    "evidence_bullets",
    "bullets_without_citations",
    "invalid_odinos",
    "uncited_bullets",
)
BRIEFS_DIR = ROOT / "reports" / "briefs"
VALIDATION_PATH = BRIEFS_DIR / "validation.csv"
UNKNOWN_MODEL = "unknown"

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


def _display(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _meta_path(brief_path: Path) -> Path:
    return brief_path.with_suffix(".meta.json")


def _previous_models() -> dict[tuple[str, ...], str]:
    """model_used from an earlier validation.csv, for briefs written before metadata files."""
    if not VALIDATION_PATH.exists():
        return {}
    with VALIDATION_PATH.open(newline="", encoding="utf-8") as file:
        return {
            tuple(row.get(field, "") for field in ("make", "model", "model_year", "component")):
            row.get("model_used") or UNKNOWN_MODEL
            for row in csv.DictReader(file)
        }


def _cached_model(path: Path, item: dict[str, str], previous: dict[tuple[str, ...], str]) -> str:
    meta = _meta_path(path)
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8")).get("model_used") or UNKNOWN_MODEL
    key = tuple(item[field] for field in ("make", "model", "model_year", "component"))
    return previous.get(key, UNKNOWN_MODEL)


def _validate_cached(path: Path, item: dict[str, str]) -> dict[str, Any]:
    complaints = data_access.get_complaints(
        item["make"],
        item["model"],
        int(item["model_year"]),
        item["component"],
        item["month"],
        limit=COMPLAINT_LIMIT,
    )
    return validate_brief(path.read_text(encoding="utf-8"), complaints, item["month"])


def run(watchlist_path: Path, top: int, *, force: bool = False) -> int:
    if top < 1:
        raise ValueError("--top must be a positive integer")
    items = _read_watchlist(watchlist_path, top)
    if not items:
        raise ValueError(f"No watchlist rows found in {watchlist_path}")
    BRIEFS_DIR.mkdir(parents=True, exist_ok=True)
    previous_models = _previous_models()
    validation_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for item in items:
        label = f"{item['make']} {item['model']} {item['model_year']} {item['component']}"
        path = _brief_path(item)
        try:
            if path.exists() and not force:
                validation = {
                    **_validate_cached(path, item),
                    "model_used": _cached_model(path, item, previous_models),
                }
                source = "cached"
            else:
                brief, validation = create_brief(item)
                path.write_text(brief, encoding="utf-8")
                _meta_path(path).write_text(
                    json.dumps({"model_used": validation["model_used"]}, indent=2) + "\n",
                    encoding="utf-8",
                )
                source = "generated"
            result = {
                **{field: item[field] for field in ("make", "model", "model_year", "component", "month")},
                "source": source,
                **validation,
            }
            validation_rows.append(result)
            print(
                f"{label}: {source} {_display(path)} by {validation['model_used']} "
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

    if validation_rows:
        with VALIDATION_PATH.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=VALIDATION_COLUMNS)
            writer.writeheader()
            writer.writerows(validation_rows)
        print(f"Validation results: {_display(VALIDATION_PATH)}")
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate briefs that already exist (default: reuse and revalidate them).",
    )
    args = parser.parse_args()
    try:
        raise SystemExit(run(args.watchlist, args.top, force=args.force))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
