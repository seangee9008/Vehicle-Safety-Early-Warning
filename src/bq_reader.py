"""Read the published results from BigQuery (read-only). For the demo team.

    from src.bq_reader import read_table, read_view, list_tables

    watchlist = read_table("watchlist")                  # pandas DataFrame
    with_briefs = read_view("v_watchlist_with_briefs")
    headline = read_view("v_headline_metrics").iloc[0]
    examples = read_table("waymo_demo_examples")         # JSON columns come back as Python lists

Needs Google credentials once per laptop: ``gcloud auth application-default login``, and the
project owner (Sean) must have given your Google account BigQuery Data Viewer + Job User on
waymo-project-513802. Reading is free in the sandbox (1 TB of queries a month).
Only SELECT queries are run; nothing here can change a table.
"""

from __future__ import annotations

import json

import pandas as pd

PROJECT = "waymo-project-513802"
DATASET = "vehicle_safety"
LOCATION = "US"
TABLES = ("watchlist", "recall_metrics", "recall_audit", "agent_briefs", "agent_citations",
          "waymo_results", "waymo_demo_examples", "publish_log")
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


def _read(name: str, allowed: tuple[str, ...], kind: str, limit: int | None) -> pd.DataFrame:
    if name not in allowed:
        raise ValueError(f"unknown {kind} {name!r}; expected one of {list(allowed)}")
    sql = f"SELECT * FROM `{PROJECT}.{DATASET}.{name}`" + (f" LIMIT {int(limit)}" if limit else "")
    from google.api_core.exceptions import Forbidden, NotFound
    from google.auth.exceptions import RefreshError

    try:
        # small tables: the REST endpoint is enough (no BigQuery Storage API client needed)
        df = get_client().query(sql).to_dataframe(create_bqstorage_client=False)
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


def read_table(name: str, limit: int | None = None) -> pd.DataFrame:
    return _read(name, TABLES, "table", limit)


def read_view(name: str, limit: int | None = None) -> pd.DataFrame:
    return _read(name, VIEWS, "view", limit)


def list_tables() -> list[str]:
    return list(TABLES + VIEWS)
