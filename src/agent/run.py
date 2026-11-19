"""Generate and validate investigation briefs from a Contract 3 watchlist CSV."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google.genai.errors import APIError

from src import sources
from src.agent import data_access
from src.agent.brief import (
    RequestBudgetExceeded,
    create_brief,
    requests_used,
    set_request_budget,
)
from src.agent.data_access import ROOT
from src.agent.validate import validate_brief, validation_errors

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
    "rank",
    "make",
    "model",
    "model_year",
    "component",
    "month",
    "model_used",
    "source",
    "self_correction",
    "recall_search",
    "citations_total",
    "citations_valid",
    "citations_invalid",
    "evidence_bullets",
    "bullets_without_citations",
    "invalid_odinos",
    "uncited_bullets",
    "first_try_errors",
)
BRIEFS_DIR = ROOT / "reports" / "briefs"
UNKNOWN_MODEL = "unknown"

load_dotenv(ROOT / ".env")


def _watchlist_from_source(
    month: str | None,
) -> tuple[list[dict[str, str]], str, Path]:
    """Rows from Justine's main.watchlist (one month) or, if not built, the sample CSV.

    Returns (rows sorted by month and rank, description, output directory).
    """
    source = sources.get_watchlist()
    columns = ", ".join(WATCHLIST_COLUMNS)
    with sources.connect() as con:
        if source.is_real:
            month_sql = "(SELECT max(month) FROM wl)" if month in (None, "latest") else "CAST(? AS DATE)"
            params = [] if month in (None, "latest") else [month]
            rows = con.execute(
                f"WITH wl AS ({source.sql}) SELECT {columns} FROM wl "
                f"WHERE month = {month_sql} ORDER BY rank",
                params,
            ).fetchall()
        else:
            rows = con.execute(
                f"WITH wl AS ({source.sql}) SELECT {columns} FROM wl ORDER BY month, rank"
            ).fetchall()
    items = [
        {
            column: value.isoformat() if hasattr(value, "isoformat") else str(value)
            for column, value in zip(WATCHLIST_COLUMNS, row)
        }
        for row in rows
    ]
    if source.is_real:
        chosen = items[0]["month"][:7] if items else str(month)
        return items, f"{source.relation} ({chosen})", BRIEFS_DIR / f"watchlist_{chosen}"
    if month not in (None, "latest"):
        items = [item for item in items if item["month"][:10] == str(month)[:10]]
    return items, source.relation, BRIEFS_DIR


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


def _brief_path(item: dict[str, str], out_dir: Path) -> Path:
    filename = "_".join(
        _slug(item[field]) for field in ("make", "model", "model_year", "component")
    )
    return out_dir / f"{filename}.md"


def _display(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def _meta_path(brief_path: Path) -> Path:
    return brief_path.with_suffix(".meta.json")


def _previous_models(validation_path: Path) -> dict[tuple[str, ...], str]:
    """model_used from an earlier validation.csv, for briefs written before metadata files."""
    if not validation_path.exists():
        return {}
    with validation_path.open(newline="", encoding="utf-8") as file:
        return {
            tuple(row.get(field, "") for field in ("make", "model", "model_year", "component")):
            row.get("model_used") or UNKNOWN_MODEL
            for row in csv.DictReader(file)
        }


def _cached_meta(path: Path, item: dict[str, str], previous: dict[tuple[str, ...], str]) -> dict[str, str]:
    """What we know about how a cached brief was written (model, self-correction, recall search)."""
    meta_path = _meta_path(path)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    key = tuple(item[field] for field in ("make", "model", "model_year", "component"))
    return {
        "model_used": meta.get("model_used") or previous.get(key, UNKNOWN_MODEL),
        # Briefs from agent v0 had no retry; one that passes now passed on its first try.
        "self_correction": meta.get("self_correction", "passed_first_try"),
        "first_try_errors": meta.get("first_try_errors", ""),
        "recall_search": meta.get("recall_search", "v0 (any make)"),
    }


def _validate_cached(path: Path, item: dict[str, str]) -> dict[str, Any]:
    cutoff = data_access.evidence_cutoff(item["month"])
    eligible = data_access.eligible_complaints(
        item["make"], item["model"], int(item["model_year"]), item["component"], cutoff
    )
    return validate_brief(path.read_text(encoding="utf-8"), eligible, cutoff)


def run(
    watchlist_path: Path | None,
    top: int,
    *,
    force: bool = False,
    month: str | None = None,
    max_requests: int | None = None,
) -> int:
    """Write and validate briefs. With no watchlist_path, read the watchlist via src.sources."""
    if top < 1:
        raise ValueError("--top must be a positive integer")
    if watchlist_path is None:
        items, described, out_dir = _watchlist_from_source(month)
        items = items[:top]
    else:
        items, described, out_dir = _read_watchlist(watchlist_path, top), str(watchlist_path), BRIEFS_DIR
    if not items:
        raise ValueError(f"No watchlist rows found in {described}")
    print(f"Watchlist: {described}, {len(items)} row(s) -> {_display(out_dir)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    validation_path = out_dir / "validation.csv"
    previous_models = _previous_models(validation_path)
    set_request_budget(max_requests)
    not_run: list[str] = []
    validation_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for item in items:
        label = f"{item['make']} {item['model']} {item['model_year']} {item['component']}"
        path = _brief_path(item, out_dir)
        if not_run:
            not_run.append(label)
            continue
        try:
            validation = None
            source = "generated"
            if path.exists() and not force:
                cached = _validate_cached(path, item)
                if validation_errors(cached):
                    # A cached brief that fails the current validator is regenerated, not edited.
                    print(f"{label}: cached brief fails the validator "
                          f"({'; '.join(validation_errors(cached))}); regenerating")
                    source = "regenerated"
                else:
                    validation = {**cached, **_cached_meta(path, item, previous_models)}
                    source = "cached"
            if validation is None:
                brief, validation = create_brief(item)
                path.write_text(brief, encoding="utf-8")
                meta = {key: validation[key] for key in
                        ("model_used", "self_correction", "first_try_errors", "recall_search")}
                _meta_path(path).write_text(
                    json.dumps({**meta, "agent_version": "v1"}, indent=2) + "\n", encoding="utf-8"
                )
            result = {
                "rank": item.get("rank", ""),
                **{field: item[field] for field in ("make", "model", "model_year", "component", "month")},
                "source": source,
                **validation,
            }
            validation_rows.append(result)
            print(
                f"{label}: {source} {_display(path)} by {validation['model_used']} "
                f"(citations {validation['citations_valid']}/{validation['citations_total']} valid, "
                f"{validation['bullets_without_citations']} uncited, "
                f"{validation['evidence_bullets']} evidence bullets, {validation['self_correction']})"
            )
            if errors := validation_errors(validation):
                failures.append(f"{label}: " + "; ".join(errors))
        except RequestBudgetExceeded as error:
            print(f"STOP {label}: {error}; remaining briefs left for the next run", file=sys.stderr)
            not_run.append(label)
        except (APIError, OSError, RuntimeError, ValueError) as error:
            failures.append(f"{label}: {error}")
            print(f"ERROR {label}: {error}", file=sys.stderr)

    print(f"Gemini requests used: {requests_used()}")
    if not_run:
        print(f"Not run ({len(not_run)}), the cache picks them up next time: " + "; ".join(not_run))
    if validation_rows:
        with validation_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=VALIDATION_COLUMNS)
            writer.writeheader()
            writer.writerows(validation_rows)
        print(f"Validation results: {_display(validation_path)}")
    if failures:
        raise RuntimeError(f"{len(failures)} brief(s) failed:\n" + "\n".join(failures))
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--watchlist",
        type=Path,
        default=None,
        help="Contract watchlist CSV. Default: main.watchlist if built, else the sample CSV.",
    )
    parser.add_argument(
        "--month",
        default=None,
        help="Watchlist month (YYYY-MM-01) or 'latest'. Real watchlist default: latest.",
    )
    parser.add_argument("--top", type=int, default=5)
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Stop after this many Gemini requests (retries count); unfinished briefs wait for the next run.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate briefs that already exist (default: reuse and revalidate them).",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per Gemini request otherwise
    try:
        raise SystemExit(run(
                args.watchlist, args.top, force=args.force, month=args.month,
                max_requests=args.max_requests,
            ))
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
