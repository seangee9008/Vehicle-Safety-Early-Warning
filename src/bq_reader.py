"""Read the published results from BigQuery (read-only). For the demo team.

    from src.bq_reader import read_table, read_view, list_tables

    watchlist = read_table("watchlist")                  # pandas DataFrame
    with_briefs = read_view("v_watchlist_with_briefs")
    headline = read_view("v_headline_metrics").iloc[0]
    excerpts = read_table("agent_complaints")            # cited complaints: date + short excerpt

Needs Google credentials once per laptop: ``gcloud auth application-default login``, and the
project owner (Sean) must have given your Google account BigQuery Data Viewer + Job User on
waymo-project-513802. Reading is free in the sandbox (1 TB of queries a month).
Only SELECT queries are run; nothing here can change a table.
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

PROJECT = "waymo-project-513802"
DATASET = "vehicle_safety"
LOCATION = "US"
TABLES = ("watchlist", "recall_metrics", "recall_audit", "agent_briefs", "agent_citations",
          "waymo_results", "publish_log", "dataset_facts", "recall_ablation", "recall_overlap",
          "recall_miss_reasons", "label_audit_misses", "agent_complaints", "agent_monthly",
          "agent_brief_stats")
VIEWS = ("v_watchlist_with_briefs", "v_headline_metrics")
JSON_COLUMNS = ("past_path", "true_future", "baseline_path", "model_paths", "model_confidences",
                "lane_polylines")

_AUTH_HELP = (
    "Not authenticated to BigQuery. Run `gcloud auth application-default login` with the Google "
    f"account that has access to {PROJECT} (ask Sean to grant BigQuery Data Viewer + Job User)."
)

_client = None


def get_client():
    global _client
    if _client is None:
        from google.auth.exceptions import DefaultCredentialsError
        from google.cloud import bigquery

        try:
            _client = bigquery.Client(project=PROJECT, location=LOCATION)
        except DefaultCredentialsError as error:
            raise RuntimeError(_AUTH_HELP) from error
    return _client


def _read(name: str, allowed: tuple[str, ...], kind: str, limit: int | None,
          timeout: float | None = None) -> pd.DataFrame:
    if name not in allowed:
        raise ValueError(f"unknown {kind} {name!r}; expected one of {list(allowed)}")
    sql = f"SELECT * FROM `{PROJECT}.{DATASET}.{name}`" + (f" LIMIT {int(limit)}" if limit else "")
    from google.api_core.exceptions import Forbidden, NotFound
    from google.auth.exceptions import RefreshError

    try:
        # small tables: the REST endpoint is enough (no BigQuery Storage API client needed)
        rows = get_client().query(sql, timeout=timeout).result(timeout=timeout)
        df = rows.to_dataframe(create_bqstorage_client=False)
    except RefreshError as error:
        raise RuntimeError(_AUTH_HELP) from error
    except Forbidden as error:
        raise RuntimeError(f"{_AUTH_HELP}\n{error}") from error
    except NotFound as error:
        raise RuntimeError(f"{name} not found: has it been published (python run.py publish "
                           "--publish)? Sandbox tables expire after 60 days.") from error
    for column in JSON_COLUMNS:
        if column in df.columns:
            df[column] = df[column].map(lambda v: json.loads(v) if isinstance(v, str) else v)
    return df


def read_table(name: str, limit: int | None = None, timeout: float | None = None) -> pd.DataFrame:
    """``timeout`` (seconds) bounds the request, e.g. so a demo can fall back to local data."""
    return _read(name, TABLES, "table", limit, timeout)


def read_view(name: str, limit: int | None = None, timeout: float | None = None) -> pd.DataFrame:
    return _read(name, VIEWS, "view", limit, timeout)


def list_tables() -> list[str]:
    return list(TABLES + VIEWS)


def dataset_expiry(timeout: float | None = 20) -> datetime | None:
    """Earliest expiration time of any table in the dataset (sandbox tables expire; views don't),
    read from BigQuery's table metadata. None if no table has one."""
    from google.api_core.exceptions import Forbidden, NotFound
    from google.auth.exceptions import RefreshError

    try:
        items = list(get_client().list_tables(f"{PROJECT}.{DATASET}", timeout=timeout))
    except (RefreshError, Forbidden) as error:
        raise RuntimeError(_AUTH_HELP) from error
    except NotFound as error:
        raise RuntimeError(f"dataset {DATASET} not found (expired or never published)") from error
    expiries = [t.expires for t in items if getattr(t, "expires", None)]
    return min(expiries) if expiries else None
