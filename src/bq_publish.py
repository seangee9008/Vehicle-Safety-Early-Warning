"""Publish the project's small final results to the BigQuery sandbox (one shared copy).

    python scripts/publish_bigquery.py              # dry run: print the plan, upload nothing
    python scripts/publish_bigquery.py --publish    # upload, create the views, verify
    python run.py publish [--publish]               # same

Sandbox rules this follows: no DML (every table is a load job with WRITE_TRUNCATE; views are
CREATE OR REPLACE VIEW); storage is limited and not refunded on delete, so every publish is
size-checked and counted in a committed ledger; tables expire after 60 days, so publish again
before then. Billing is never needed.

Only the tables in ``TABLES`` are published. Raw/stg complaints and recalls, monthly_features
and Waymo tracks are never published (``FORBIDDEN``).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "waymo-project-513802"
DATASET = "vehicle_safety"
LOCATION = "US"
REPORTS = ROOT / "reports"
BRIEFS_DIR = REPORTS / "briefs"
LEDGER = REPORTS / "bq_publish_ledger.csv"
WAYMO_DEMO = ROOT / "data" / "processed" / "waymo" / "demo_examples.json"

MB = 1_000_000
MAX_TABLE_BYTES = 25 * MB
MAX_TABLE_ROWS = 200_000
MAX_PUBLISH_BYTES = 100 * MB
MAX_LIFETIME_BYTES = 2_000 * MB

# raw/stg layers and the big source tables, under any of these names
FORBIDDEN = re.compile(
    r"^(raw|stg)_|^(complaints?|recalls?|monthly_features|investigations|tracks?)$|_tracks?$"
)
LEDGER_COLUMNS = ["published_at", "git_commit", "table_name", "rows", "bytes", "cumulative_bytes"]


class PublishRefused(RuntimeError):
    """A guard stopped the publish; nothing was uploaded."""


@dataclass(frozen=True)
class Column:
    name: str
    type: str  # BigQuery standard SQL type
    description: str
    mode: str = "NULLABLE"


@dataclass(frozen=True)
class Table:
    name: str
    description: str
    columns: tuple[Column, ...]
    build: Callable[[], list[dict[str, Any]]]


# ---------------------------------------------------------------------------------------------
# builders: each returns JSON-ready rows with exactly the table's columns


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                               capture_output=True, text=True, check=True).stdout.strip()
        return out + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _clean(value: Any) -> Any:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if hasattr(value, "item"):  # numpy scalar
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    return value


def build_watchlist() -> list[dict[str, Any]]:
    from src import sources

    with sources.connect() as con:
        source = sources.get_watchlist(con)
        if not source.is_real:
            raise PublishRefused(f"main.watchlist is not built or fails its contract: {source.problem}")
        df = con.sql(
            "SELECT month, make, model, model_year, component, score, rank, top_reasons, source "
            "FROM main.watchlist ORDER BY month, rank"
        ).df()
    df["month"] = pd.to_datetime(df["month"]).dt.date.astype(str)
    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict("records")]


RECALL_METRIC_FILES = {
    # file -> (fixed fields, column holding top_k or a fixed top_k)
    "recall_metrics_stg.csv": ("split/parts/scorer", 20),
    "recall_topk_stg.csv": ("top/scorer", None),
    "recall_ablation_stg.csv": ("top/model", None),  # published only if Justine commits it
}


def build_recall_metrics() -> list[dict[str, Any]]:
    rows = []
    for name in RECALL_METRIC_FILES:
        path = REPORTS / name
        if not path.exists():
            continue
        df = pd.read_csv(path)
        for record in df.to_dict("records"):
            scorer = record.pop("scorer", None) or record.pop("model", None)
            top_k = int(record.pop("top", 20))
            split = record.pop("split", "test")
            parts = record.pop("parts", "all")
            record.pop("source", None)
            for metric, value in record.items():
                if isinstance(value, (int, float)) and not pd.isna(value):
                    rows.append({"source_file": name, "split": split, "parts": parts,
                                 "scorer": str(scorer), "top_k": top_k, "metric": metric,
                                 "value": float(value)})
    return rows


def build_recall_audit() -> list[dict[str, Any]]:
    rows = []
    specs = [("label_audit", "label_audit_stg.csv", "label", None),
             ("investigations", "investigations_experiment_stg.csv", "model", "split")]
    for experiment, name, variant_col, split_col in specs:
        path = REPORTS / name
        if not path.exists():
            continue
        for record in pd.read_csv(path).to_dict("records"):
            variant = record.pop(variant_col)
            split = record.pop(split_col) if split_col else "test"
            for metric, value in record.items():
                if isinstance(value, (int, float)) and not pd.isna(value):
                    rows.append({"experiment": experiment, "source_file": name, "variant": variant,
                                 "split": split, "metric": metric, "value": float(value)})
    return rows


def _brief_dirs() -> list[Path]:
    # only briefs for Justine's real watchlist (reports/briefs/watchlist_YYYY-MM/), not samples
    return sorted(p for p in BRIEFS_DIR.glob("watchlist_*") if (p / "validation.csv").exists())


def _brief_file(directory: Path, row: dict[str, str]) -> Path:
    stem = "_".join(re.sub(r"[^a-z0-9]+", "_", str(row[k]).lower()).strip("_")
                    for k in ("make", "model", "model_year", "component"))
    return directory / f"{stem}.md"


def _briefs() -> list[tuple[dict[str, Any], str, list[str], set[str]]]:
    out = []
    for directory in _brief_dirs():
        with (directory / "validation.csv").open(encoding="utf-8") as file:
            for row in csv.DictReader(file):
                path = _brief_file(directory, row)
                if not path.exists():
                    continue
                month = row["month"][:10]
                brief_id = f"{month[:7]}-r{int(row['rank']):02d}"
                text = path.read_text(encoding="utf-8")
                meta_path = path.with_suffix(".meta.json")
                meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
                generated = datetime.fromtimestamp(path.stat().st_mtime, UTC)
                outcome = meta.get("self_correction") or row.get("self_correction") or "passed_first_try"
                record = {
                    "brief_id": brief_id, "month": month, "rank": int(row["rank"]),
                    "make": row["make"], "model": row["model"], "model_year": int(row["model_year"]),
                    "component": row["component"], "brief_markdown": text,
                    "model_used": meta.get("model_used") or row.get("model_used"),
                    "citations_valid": int(row["citations_valid"]),
                    "citations_total": int(row["citations_total"]),
                    "uncited_bullets": int(row["bullets_without_citations"]),
                    "evidence_bullets": int(row["evidence_bullets"]),
                    "passed_first_try": outcome == "passed_first_try",
                    "passed_after_retry": outcome == "passed_after_retry",
                    "still_failing": outcome == "still_failing",
                    "similar_recall_search_scope": meta.get("recall_search") or row.get("recall_search"),
                    "agent_version": meta.get("agent_version", "v0"),
                    "generated_at": generated.isoformat(),
                }
                from src.agent.validate import extract_citations

                cited = list(dict.fromkeys(extract_citations(text)))
                invalid = set(re.findall(r"\d+", row.get("invalid_odinos") or ""))
                out.append((record, brief_id, cited, invalid))
    return out


def build_agent_briefs() -> list[dict[str, Any]]:
    return [record for record, *_ in _briefs()]


def build_agent_citations() -> list[dict[str, Any]]:
    return [{"brief_id": brief_id, "odino": odino, "valid": odino not in invalid}
            for _, brief_id, cited, invalid in _briefs() for odino in cited]


PREDICTORS = {"constant_velocity": "cv", "model_best_guess": "best", "model_best_of_6": "min"}


def build_waymo_results() -> list[dict[str, Any]]:
    df = pd.read_csv(REPORTS / "waymo_model_vs_baseline.csv")
    rows = []
    for record in df.to_dict("records"):
        for predictor, prefix in PREDICTORS.items():
            vs_cv = None if prefix == "cv" else record.get(f"{prefix}_fde_vs_cv_pct")
            rows.append({k: _clean(v) for k, v in {
                "object_type": record["object_type"], "horizon_s": int(record["horizon_s"]),
                "n_agents": int(record["n_agents"]), "predictor": predictor,
                "ade_m": record[f"{prefix}_ade_m"], "fde_m": record[f"{prefix}_fde_m"],
                "fde_per_m_mean": record.get(f"{prefix}_fde_per_m_mean"),
                "fde_per_m_median": record.get(f"{prefix}_fde_per_m_median"),
                "fde_improvement_vs_cv_pct": vs_cv,
            }.items()})
    return rows


WAYMO_JSON_FIELDS = ("past_path", "true_future", "baseline_path", "model_paths",
                     "model_confidences", "lane_polylines")


def build_waymo_demo_examples() -> list[dict[str, Any]]:
    if not WAYMO_DEMO.exists():
        raise PublishRefused(f"{WAYMO_DEMO.relative_to(ROOT)} missing: run "
                             "python -m src.trajectory.demo_examples")
    return json.loads(WAYMO_DEMO.read_text(encoding="utf-8"))


# small final tables the public demo needs (so it never reads local data)

FACTS = ROOT / "app" / "facts.json"


def _flatten(prefix: str, value: Any, out: list[dict[str, Any]]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else k, v, out)
    else:
        out.append({"key": prefix, "value": str(value)})


def build_dataset_facts() -> list[dict[str, Any]]:
    """app/facts.json (python -m app.build_facts) flattened, plus the agent's complaint limit."""
    if not FACTS.exists():
        raise PublishRefused("app/facts.json missing: run python -m app.build_facts")
    rows: list[dict[str, Any]] = []
    _flatten("", json.loads(FACTS.read_text(encoding="utf-8")), rows)
    from src.agent.brief import COMPLAINT_LIMIT

    rows.append({"key": "agent.read_limit", "value": str(COMPLAINT_LIMIT)})
    return rows


def _reports():
    from app import reports

    return reports


def build_recall_ablation() -> list[dict[str, Any]]:
    df = _reports().ablation()
    if df is None:
        raise PublishRefused("reports/recall_results_stg.md has no age vs complaints table")
    return [{"model": r["model"], "hit_rate_20": _clean(r["hit_rate_20"]), "hit_rate_100": _clean(r["hit_rate_100"]),
             "pr_auc": _clean(r["pr_auc"])} for r in df.to_dict("records")]


OVERLAP_INT = ("top", "caught_by_both", "only_baseline", "only_lgbm", "both_lgbm_earlier", "both_same_month",
               "both_lgbm_later")
OVERLAP_FLOAT = ("both_median_weeks_baseline", "both_median_weeks_lgbm", "only_baseline_median_weeks",
                 "only_lgbm_median_weeks")


def build_recall_overlap() -> list[dict[str, Any]]:
    df = pd.read_csv(REPORTS / "recall_overlap_stg.csv")
    return [{**{c: int(r[c]) for c in OVERLAP_INT}, **{c: _clean(float(r[c])) for c in OVERLAP_FLOAT}}
            for r in df.to_dict("records")]


def build_recall_miss_reasons() -> list[dict[str, Any]]:
    df = pd.read_csv(REPORTS / "recall_miss_reasons_stg.csv")
    return [{"reason": r["reason"], "recalls": int(r["recalls"]), "share": float(r["share"])}
            for r in df.to_dict("records")]


def build_label_audit_misses() -> list[dict[str, Any]]:
    df = _reports().label_audit_misses()
    if df is None:
        raise PublishRefused("reports/label_audit.md has no misses table")
    return [{"match": r["match"], "misses": int(r["misses"]), "share": float(r["share"])} for r in df.to_dict("records")]


_AGENT_DETAILS: dict[str, Any] = {}


def _agent_details() -> dict[str, Any]:
    """Complaints on file, monthly counts and cited excerpts for each brief (local DuckDB, read-only)."""
    if not _AGENT_DETAILS:
        from app import agent_data

        briefs = pd.DataFrame(build_agent_briefs())
        citations = pd.DataFrame(build_agent_citations())
        _AGENT_DETAILS.update(agent_data.build_live(briefs, citations))
    return _AGENT_DETAILS


def build_agent_complaints() -> list[dict[str, Any]]:
    return [{"brief_id": b, "odino": o, "received_date": c["date"], "excerpt": c["excerpt"]}
            for b, d in _agent_details().items() for o, c in d["cited"].items()]


def build_agent_monthly() -> list[dict[str, Any]]:
    return [{"brief_id": b, "month": m["month"], "complaints": float(m["complaints"])}
            for b, d in _agent_details().items() for m in d["monthly"]]


def build_agent_brief_stats() -> list[dict[str, Any]]:
    return [{"brief_id": b, "complaints_on_file": int(d["complaints_on_file"])} for b, d in _agent_details().items()]


# ---------------------------------------------------------------------------------------------
# table catalog: explicit schemas with descriptions

def _c(name: str, type_: str, description: str, mode: str = "NULLABLE") -> Column:
    return Column(name, type_, description, mode)


WATCHLIST_COLUMNS = (
    _c("month", "DATE", "Watchlist month M (first day); uses complaints received by the end of M", "REQUIRED"),
    _c("make", "STRING", "Standardized make (Angelina's stg tables)", "REQUIRED"),
    _c("model", "STRING", "Standardized model, punctuation removed (F-150 -> F150)", "REQUIRED"),
    _c("model_year", "INT64", "Model year", "REQUIRED"),
    _c("component", "STRING", "Standardized top-level part, e.g. POWER TRAIN", "REQUIRED"),
    _c("score", "FLOAT64", "LightGBM probability of a recall within 6 months"),
    _c("rank", "INT64", "Rank within the month (1 = highest score)", "REQUIRED"),
    _c("top_reasons", "STRING", "Top SHAP reasons, '; '-separated, plain English"),
    _c("source", "STRING", "Which complaint tables built it (stg)"),
)

TABLES: dict[str, Table] = {t.name: t for t in [
    Table("watchlist",
          "Justine's recall watchlist: top 20 car + parts per test month (2022-01 to 2024-12), "
          "from main.watchlist.", WATCHLIST_COLUMNS, build_watchlist),
    Table("recall_metrics",
          "Recall model results (Justine's committed CSVs, stg source), long format: one row per "
          "file, split, scorer, top_k and metric. Test = 2022-24, validation = 2020-21.",
          (_c("source_file", "STRING", "CSV in reports/ the row came from", "REQUIRED"),
           _c("split", "STRING", "val (2020-21) or test (2022-24)", "REQUIRED"),
           _c("parts", "STRING", "all, or 'no air bags'", "REQUIRED"),
           _c("scorer", "STRING", "lgbm (model), baseline (3-month complaint count), or an ablation", "REQUIRED"),
           _c("top_k", "INT64", "Watchlist size per month the metric is computed at", "REQUIRED"),
           _c("metric", "STRING", "e.g. precision_at_20, pr_auc, median_weeks_warning", "REQUIRED"),
           _c("value", "FLOAT64", "Metric value (fractions, not percentages)", "REQUIRED")),
          build_recall_metrics),
    Table("recall_audit",
          "Sean's read-only checks of the recall model: label audit (looser label matches) and the "
          "open-NHTSA-investigations feature experiment. Long format.",
          (_c("experiment", "STRING", "label_audit or investigations", "REQUIRED"),
           _c("source_file", "STRING", "CSV in reports/ the row came from", "REQUIRED"),
           _c("variant", "STRING", "Label definition or model variant", "REQUIRED"),
           _c("split", "STRING", "val or test", "REQUIRED"),
           _c("metric", "STRING", "Metric name", "REQUIRED"),
           _c("value", "FLOAT64", "Metric value", "REQUIRED")),
          build_recall_audit),
    Table("agent_briefs",
          "Gemini investigation briefs for watchlist rows, AI-generated and not hand-edited, with "
          "their citation validation results. One row per brief.",
          (_c("brief_id", "STRING", "YYYY-MM-rNN: watchlist month and rank", "REQUIRED"),
           _c("month", "DATE", "Watchlist month", "REQUIRED"),
           _c("rank", "INT64", "Watchlist rank in that month", "REQUIRED"),
           _c("make", "STRING", "Make", "REQUIRED"),
           _c("model", "STRING", "Model", "REQUIRED"),
           _c("model_year", "INT64", "Model year", "REQUIRED"),
           _c("component", "STRING", "Part", "REQUIRED"),
           _c("brief_markdown", "STRING", "The brief, Markdown, citations as [ODINO n]", "REQUIRED"),
           _c("model_used", "STRING", "Gemini model that wrote it"),
           _c("citations_valid", "INT64", "Cited ODINOs that exist for this car + part before the cutoff"),
           _c("citations_total", "INT64", "Cited ODINOs"),
           _c("uncited_bullets", "INT64", "Evidence bullets without a citation"),
           _c("evidence_bullets", "INT64", "Evidence bullets (3-6 required)"),
           _c("passed_first_try", "BOOL", "Passed the validator on the first try"),
           _c("passed_after_retry", "BOOL", "Passed after one self-correction retry"),
           _c("still_failing", "BOOL", "Still failing after the retry"),
           _c("similar_recall_search_scope", "STRING", "same_make, all_makes, or 'v0 (any make)'"),
           _c("agent_version", "STRING", "Agent version that wrote it (v0 / v1)"),
           _c("generated_at", "TIMESTAMP", "When the brief file was written")),
          build_agent_briefs),
    Table("agent_citations",
          "One row per complaint cited in a brief (distinct per brief).",
          (_c("brief_id", "STRING", "agent_briefs.brief_id", "REQUIRED"),
           _c("odino", "STRING", "NHTSA complaint number (ODINO); look up on nhtsa.gov", "REQUIRED"),
           _c("valid", "BOOL", "True if the validator found it for this car + part before the cutoff", "REQUIRED")),
          build_agent_citations),
    Table("waymo_results",
          "Waymo Open Motion test shard (validation_00000): MLP (K=6) vs constant velocity, per agent "
          "type and horizon. Long format: one row per predictor.",
          (_c("object_type", "STRING", "all, vehicle, pedestrian, cyclist", "REQUIRED"),
           _c("horizon_s", "INT64", "Prediction horizon in seconds (3, 5, 8)", "REQUIRED"),
           _c("n_agents", "INT64", "Agents scored", "REQUIRED"),
           _c("predictor", "STRING", "constant_velocity, model_best_guess (top path), model_best_of_6 (minADE/minFDE)", "REQUIRED"),
           _c("ade_m", "FLOAT64", "Average displacement error, metres"),
           _c("fde_m", "FLOAT64", "Final displacement error, metres"),
           _c("fde_per_m_mean", "FLOAT64", "8 s only: mean of FDE / metres travelled (agents moving >= 1 m)"),
           _c("fde_per_m_median", "FLOAT64", "8 s only: median of FDE / metres travelled"),
           _c("fde_improvement_vs_cv_pct", "FLOAT64", "100 * (1 - FDE / constant-velocity FDE); null for CV")),
          build_waymo_results),
    Table("dataset_facts",
          "Dataset sizes the demo shows (complaints, recalls, scenes, test window), as key/value text.",
          (_c("key", "STRING", "Dotted name, e.g. complaints.per_year", "REQUIRED"),
           _c("value", "STRING", "Value as text (numbers and dates)", "REQUIRED")),
          build_dataset_facts),
    Table("recall_ablation",
          "Model trained on vehicle age only, complaints only, and both, vs the baseline (test 2022-24).",
          (_c("model", "STRING", "Which inputs the model used", "REQUIRED"),
           _c("hit_rate_20", "FLOAT64", "Precision with 20 flags a month"),
           _c("hit_rate_100", "FLOAT64", "Precision with 100 flags a month"),
           _c("pr_auc", "FLOAT64", "PR-AUC")),
          build_recall_ablation),
    Table("recall_overlap",
          "Which recalls the model and the baseline each caught, and their median weeks of warning.",
          tuple(_c(c, "INT64", c.replace("_", " "), "REQUIRED") for c in OVERLAP_INT)
          + tuple(_c(c, "FLOAT64", c.replace("_", " ") + " (weeks)") for c in OVERLAP_FLOAT),
          build_recall_overlap),
    Table("recall_miss_reasons",
          "Why missed first recalls were missed (test 2022-24).",
          (_c("reason", "STRING", "Group of missed recalls", "REQUIRED"),
           _c("recalls", "INT64", "Missed recalls in the group", "REQUIRED"),
           _c("share", "FLOAT64", "Share of all missed recalls", "REQUIRED")),
          build_recall_miss_reasons),
    Table("label_audit_misses",
          "How many of the model's misses each looser label match would count as hits.",
          (_c("match", "STRING", "Looser match", "REQUIRED"),
           _c("misses", "INT64", "Misses it matches", "REQUIRED"),
           _c("share", "FLOAT64", "Share of misses", "REQUIRED")),
          build_label_audit_misses),
    Table("agent_complaints",
          "Each complaint cited in a brief: received date and a short excerpt (personal details removed).",
          (_c("brief_id", "STRING", "agent_briefs.brief_id", "REQUIRED"),
           _c("odino", "STRING", "NHTSA complaint number", "REQUIRED"),
           _c("received_date", "DATE", "When NHTSA received it"),
           _c("excerpt", "STRING", "First ~170 characters, redacted")),
          build_agent_complaints),
    Table("agent_monthly",
          "Complaints per month for each briefed car + part, 12 months up to the flag (months with any).",
          (_c("brief_id", "STRING", "agent_briefs.brief_id", "REQUIRED"),
           _c("month", "DATE", "Month", "REQUIRED"),
           _c("complaints", "FLOAT64", "Complaints that month", "REQUIRED")),
          build_agent_monthly),
    Table("agent_brief_stats",
          "Complaints on file for each briefed car + part before the flag (what the agent could read).",
          (_c("brief_id", "STRING", "agent_briefs.brief_id", "REQUIRED"),
           _c("complaints_on_file", "INT64", "Distinct complaints before the evidence cutoff", "REQUIRED")),
          build_agent_brief_stats),
]}

PUBLISH_LOG = Table(
    "publish_log",
    "One row per table per publish (mirror of reports/bq_publish_ledger.csv, replaced each "
    "publish). bytes = estimated upload size; cumulative_bytes counts every publish ever.",
    (_c("published_at", "TIMESTAMP", "When the publish ran", "REQUIRED"),
     _c("git_commit", "STRING", "Commit the publish ran from (-dirty = uncommitted changes)"),
     _c("table_name", "STRING", "Table published", "REQUIRED"),
     _c("rows", "INT64", "Rows loaded", "REQUIRED"),
     _c("bytes", "INT64", "Estimated bytes (newline-delimited JSON size)", "REQUIRED"),
     _c("cumulative_bytes", "INT64", "Bytes published ever, including this row", "REQUIRED")),
    list,
)

VIEWS = {
    "v_watchlist_with_briefs": (
        "Every watchlist row with its brief, if one was written (has_brief).",
        """
SELECT w.*, b.brief_id, b.brief_markdown, b.model_used, b.citations_valid, b.citations_total,
       b.uncited_bullets, b.evidence_bullets, b.passed_first_try, b.passed_after_retry,
       b.still_failing, b.similar_recall_search_scope,
       b.brief_id IS NOT NULL AS has_brief
FROM `{ds}.watchlist` AS w
LEFT JOIN `{ds}.agent_briefs` AS b ON w.month = b.month AND w.rank = b.rank
"""),
    "v_headline_metrics": (
        ("One row of headline numbers for the demo overview (test 2022-24 for the recall model, "
         "validation_00000 for Waymo)."),
        """
WITH m AS (
  SELECT scorer, metric, value FROM `{ds}.recall_metrics`
  WHERE source_file = 'recall_metrics_stg.csv' AND split = 'test' AND parts = 'all'
),
k AS (
  SELECT scorer, top_k, metric, value FROM `{ds}.recall_metrics`
  WHERE source_file = 'recall_topk_stg.csv'
),
w AS (
  SELECT predictor, fde_m, fde_improvement_vs_cv_pct FROM `{ds}.waymo_results`
  WHERE object_type = 'all' AND horizon_s = 8
)
SELECT
  (SELECT value FROM m WHERE scorer = 'lgbm' AND metric = 'precision_at_20') AS model_precision_at_20,
  (SELECT value FROM m WHERE scorer = 'baseline' AND metric = 'precision_at_20') AS baseline_precision_at_20,
  (SELECT value FROM m WHERE scorer = 'lgbm' AND metric = 'pr_auc') AS model_pr_auc,
  (SELECT value FROM m WHERE scorer = 'lgbm' AND metric = 'median_weeks_warning') AS model_median_weeks_warning,
  (SELECT value FROM k WHERE scorer = 'lgbm' AND top_k = 20 AND metric = 'recalls_caught') AS model_recalls_caught_top20,
  (SELECT value FROM k WHERE scorer = 'lgbm' AND top_k = 100 AND metric = 'recalls_caught') AS model_recalls_caught_top100,
  (SELECT COUNT(DISTINCT month) FROM `{ds}.watchlist`) AS watchlist_months,
  (SELECT COUNT(*) FROM `{ds}.agent_briefs`) AS briefs,
  (SELECT COUNTIF(NOT still_failing AND uncited_bullets = 0 AND citations_valid = citations_total)
     FROM `{ds}.agent_briefs`) AS briefs_fully_valid,
  (SELECT SAFE_DIVIDE(SUM(citations_valid), SUM(citations_total)) FROM `{ds}.agent_briefs`) AS citation_valid_rate,
  (SELECT fde_m FROM w WHERE predictor = 'constant_velocity') AS waymo_cv_fde_8s_m,
  (SELECT fde_m FROM w WHERE predictor = 'model_best_guess') AS waymo_model_fde_8s_m,
  (SELECT fde_m FROM w WHERE predictor = 'model_best_of_6') AS waymo_model_min_fde_8s_m,
  (SELECT fde_improvement_vs_cv_pct FROM w WHERE predictor = 'model_best_guess') AS waymo_best_guess_vs_cv_pct,
  (SELECT MAX(published_at) FROM `{ds}.publish_log`) AS last_published_at
"""),
}


# ---------------------------------------------------------------------------------------------
# plan + guards

@dataclass
class Planned:
    table: Table
    rows: list[dict[str, Any]]

    @property
    def n_rows(self) -> int:
        return len(self.rows)

    @property
    def bytes(self) -> int:
        return estimate_bytes(self.rows)


def estimate_bytes(rows: list[dict[str, Any]]) -> int:
    """Newline-delimited JSON size: what the load job uploads; above BigQuery's stored size."""
    return sum(len(json.dumps(row, default=str).encode("utf-8")) + 1 for row in rows)


def check_allowed(name: str) -> None:
    if name not in TABLES and name != PUBLISH_LOG.name:
        raise PublishRefused(f"{name}: not in the publish list {sorted(TABLES)}")
    if FORBIDDEN.search(name):
        raise PublishRefused(f"{name}: raw/stg complaints, recalls, features and tracks are never published")


def read_ledger(path: Path = LEDGER) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=LEDGER_COLUMNS)
    return pd.read_csv(path)


def lifetime_bytes(ledger: pd.DataFrame) -> int:
    return int(ledger["bytes"].sum()) if len(ledger) else 0


def plan(names: list[str] | None = None) -> list[Planned]:
    names = list(TABLES) if names is None else names
    planned = []
    for name in names:
        check_allowed(name)
        table = TABLES[name]
        rows = table.build()
        columns = [c.name for c in table.columns]
        for row in rows:
            if list(row) != columns:
                raise PublishRefused(f"{name}: row columns {list(row)} != schema {columns}")
        planned.append(Planned(table, rows))
    return planned


def check_guards(planned: list[Planned], ledger: pd.DataFrame) -> None:
    problems = []
    for p in planned:
        if p.n_rows > MAX_TABLE_ROWS:
            problems.append(f"{p.table.name}: {p.n_rows:,} rows > {MAX_TABLE_ROWS:,}")
        if p.bytes > MAX_TABLE_BYTES:
            problems.append(f"{p.table.name}: {p.bytes / MB:.1f} MB > {MAX_TABLE_BYTES / MB:.0f} MB")
        if p.n_rows == 0:
            problems.append(f"{p.table.name}: no rows")
    total = sum(p.bytes for p in planned)
    if total > MAX_PUBLISH_BYTES:
        problems.append(f"this publish {total / MB:.1f} MB > {MAX_PUBLISH_BYTES / MB:.0f} MB")
    if lifetime_bytes(ledger) + total > MAX_LIFETIME_BYTES:
        problems.append(f"lifetime {(lifetime_bytes(ledger) + total) / MB:.1f} MB would pass "
                        f"{MAX_LIFETIME_BYTES / MB:.0f} MB (reports/bq_publish_ledger.csv)")
    if problems:
        raise PublishRefused("refused, nothing uploaded:\n  " + "\n  ".join(problems))


def format_plan(planned: list[Planned], ledger: pd.DataFrame) -> str:
    lines = [f"{'table':<22}{'rows':>9}{'est. MB':>10}", "-" * 41]
    for p in planned:
        lines.append(f"{p.table.name:<22}{p.n_rows:>9,}{p.bytes / MB:>10.3f}")
    total = sum(p.bytes for p in planned)
    lines += ["-" * 41,
              f"{'total':<22}{sum(p.n_rows for p in planned):>9,}{total / MB:>10.3f}",
              (f"ledger before: {lifetime_bytes(ledger) / MB:.3f} MB; after: "
               f"{(lifetime_bytes(ledger) + total) / MB:.3f} MB of {MAX_LIFETIME_BYTES / MB:.0f} MB"),
              f"views (no storage): {', '.join(VIEWS)}",
              f"plus {PUBLISH_LOG.name} (ledger mirror)"]
    missing = [n for n in RECALL_METRIC_FILES if not (REPORTS / n).exists()]
    if missing:
        lines.append(f"not found, skipped: {', '.join('reports/' + n for n in missing)}")
    return "\n".join(lines)


def ledger_rows(planned: list[Planned], ledger: pd.DataFrame, published_at: str,
                commit: str) -> pd.DataFrame:
    rows, running = [], lifetime_bytes(ledger)
    for p in planned:
        running += p.bytes
        rows.append({"published_at": published_at, "git_commit": commit,
                     "table_name": p.table.name, "rows": p.n_rows, "bytes": p.bytes,
                     "cumulative_bytes": running})
    return pd.concat([ledger, pd.DataFrame(rows, columns=LEDGER_COLUMNS)], ignore_index=True)


# ---------------------------------------------------------------------------------------------
# BigQuery

def client():
    try:
        from google.auth.exceptions import DefaultCredentialsError
        from google.cloud import bigquery
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("pip install google-cloud-bigquery") from error
    try:
        return bigquery.Client(project=PROJECT, location=LOCATION)
    except DefaultCredentialsError as error:
        raise RuntimeError("Not authenticated to Google Cloud. Run: "
                           "gcloud auth application-default login") from error


def _schema(table: Table):
    from google.cloud import bigquery

    return [bigquery.SchemaField(c.name, c.type, mode=c.mode, description=c.description)
            for c in table.columns]


def load_table(bq, table: Table, rows: list[dict[str, Any]]) -> None:
    """WRITE_TRUNCATE load job (not DML), so it works in the sandbox."""
    from google.cloud import bigquery

    check_allowed(table.name)
    config = bigquery.LoadJobConfig(
        schema=_schema(table),
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
        destination_table_description=table.description,
    )
    bq.load_table_from_json(rows, f"{PROJECT}.{DATASET}.{table.name}", job_config=config).result()


def create_views(bq) -> None:
    ds = f"{PROJECT}.{DATASET}"
    for name, (description, sql) in VIEWS.items():
        desc = description.replace('"', "'")
        bq.query(f'CREATE OR REPLACE VIEW `{ds}.{name}` OPTIONS(description="{desc}") AS'
                 + sql.format(ds=ds)).result()


def publish(planned: list[Planned], bq, ledger_path: Path = LEDGER) -> pd.DataFrame:
    ledger = read_ledger(ledger_path)
    check_guards(planned, ledger)
    published_at = datetime.now(UTC).isoformat(timespec="seconds")
    from google.cloud import bigquery

    dataset = bigquery.Dataset(f"{PROJECT}.{DATASET}")
    dataset.location = LOCATION
    bq.create_dataset(dataset, exists_ok=True)
    for p in planned:
        load_table(bq, p.table, p.rows)
        print(f"  loaded {p.table.name}: {p.n_rows:,} rows", flush=True)
    new_ledger = ledger_rows(planned, ledger, published_at, _git_commit())
    log_rows = [{k: _clean(v) for k, v in r.items()} for r in new_ledger.to_dict("records")]
    log = Planned(PUBLISH_LOG, log_rows)
    check_guards([log], new_ledger)
    load_table(bq, PUBLISH_LOG, log_rows)
    new_ledger = ledger_rows([log], new_ledger, published_at, new_ledger["git_commit"].iloc[-1])
    new_ledger.to_csv(ledger_path, index=False)
    print(f"  loaded {PUBLISH_LOG.name}: {len(log_rows):,} rows", flush=True)
    create_views(bq)
    print(f"  views: {', '.join(VIEWS)}", flush=True)
    return new_ledger


def _first(job) -> Any:
    return next(iter(job.result()))


def verify(bq, planned: list[Planned]) -> list[str]:
    ds = f"{PROJECT}.{DATASET}"
    lines, ok = [], True
    for p in planned:
        n = _first(bq.query(f"SELECT COUNT(*) AS n FROM `{ds}.{p.table.name}`"))["n"]
        match = n == p.n_rows
        ok &= match
        lines.append(f"{p.table.name:<22} local {p.n_rows:>6,}  bigquery {n:>6,}  {'OK' if match else 'MISMATCH'}")
    v = _first(bq.query(f"SELECT COUNT(*) AS n, COUNTIF(has_brief) AS with_brief, "
                        f"MAX(month) AS latest FROM `{ds}.v_watchlist_with_briefs`"))
    lines.append(f"v_watchlist_with_briefs: {v['n']} rows, {v['with_brief']} with a brief, latest {v['latest']}")
    h = dict(_first(bq.query(f"SELECT * FROM `{ds}.v_headline_metrics`")).items())
    lines.append("v_headline_metrics: " + ", ".join(
        f"{k}={round(v, 4) if isinstance(v, float) else v}" for k, v in h.items()))
    if not ok:
        raise RuntimeError("row counts differ:\n" + "\n".join(lines))
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish final results to BigQuery (dry run by default).")
    parser.add_argument("--publish", action="store_true", help="actually upload (default: dry run)")
    parser.add_argument("--verify-only", action="store_true", help="only compare row counts + query views")
    args = parser.parse_args(argv)
    planned = plan()
    ledger = read_ledger()
    print(f"BigQuery {PROJECT}.{DATASET} ({LOCATION})")
    print(format_plan(planned, ledger))
    if args.verify_only:
        print("\n".join(verify(client(), planned)))
        return 0
    check_guards(planned, ledger)
    if not args.publish:
        print("\nDry run: nothing uploaded. Add --publish to upload.")
        return 0
    bq = client()
    new_ledger = publish(planned, bq)
    print(f"published; ledger total {lifetime_bytes(new_ledger) / MB:.3f} MB")
    print("\n".join(verify(bq, planned)))
    return 0
