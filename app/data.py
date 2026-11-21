"""Data for the demo: BigQuery first (src/bq_reader.py), local files as a fallback.

Every table is loaded once (in parallel) and cached. If BigQuery is not reachable or not
authenticated, or ``VSE_DATA=local`` is set (e.g. no wifi on stage), each table is built from
the local files instead: DuckDB (read-only), reports/ and reports/briefs/, the Waymo demo
examples JSON. A table that is in neither place comes back as None and its tab says so.

No Gemini calls anywhere: the briefs were written and validated beforehand.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
BQ_TIMEOUT_S = 20

# name -> "view" or "table" in BigQuery
SOURCES = {
    "v_headline_metrics": "view",
    "v_watchlist_with_briefs": "view",
    "recall_metrics": "table",
    "recall_audit": "table",
    "agent_citations": "table",
    "waymo_results": "table",
    "waymo_demo_examples": "table",
}

BRIEF_COLUMNS = ["brief_id", "brief_markdown", "model_used", "citations_valid", "citations_total",
                 "uncited_bullets", "evidence_bullets", "passed_first_try", "passed_after_retry",
                 "still_failing", "similar_recall_search_scope"]


# ---------------------------------------------------------------------------------------------
# local fallbacks: the same builders the BigQuery publish uses, so the columns match

def _frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _local_watchlist_with_briefs() -> pd.DataFrame:
    from src import bq_publish

    watch = _frame(bq_publish.build_watchlist())
    try:
        briefs = _frame(bq_publish.build_agent_briefs())
    except Exception:  # noqa: BLE001 - no local briefs: the watchlist alone still works
        briefs = pd.DataFrame(columns=["month", "rank", *BRIEF_COLUMNS])
    briefs = briefs.reindex(columns=["month", "rank", *BRIEF_COLUMNS])
    merged = watch.merge(briefs, on=["month", "rank"], how="left")
    merged["has_brief"] = merged["brief_id"].notna()
    return merged


def _local(builder_name: str) -> Callable[[], pd.DataFrame]:
    def build() -> pd.DataFrame:
        from src import bq_publish

        return _frame(getattr(bq_publish, builder_name)())
    return build


def _local_waymo_examples() -> pd.DataFrame:
    """The demo examples JSON; generated from the local Waymo test data if it isn't there yet."""
    import subprocess
    import sys

    from src import bq_publish

    if not bq_publish.WAYMO_DEMO.exists():
        subprocess.run([sys.executable, "-m", "src.trajectory.demo_examples"], cwd=ROOT,
                       check=True, capture_output=True, timeout=300)
    return _frame(bq_publish.build_waymo_demo_examples())


LOCAL_LOADERS: dict[str, Callable[[], pd.DataFrame]] = {
    "v_watchlist_with_briefs": _local_watchlist_with_briefs,
    "recall_metrics": _local("build_recall_metrics"),
    "recall_audit": _local("build_recall_audit"),
    "agent_citations": _local("build_agent_citations"),
    "waymo_results": _local("build_waymo_results"),
    "waymo_demo_examples": lambda: _local_waymo_examples(),
    # v_headline_metrics is computed from the other tables (headline() below)
}


def use_streamlit_secrets() -> bool:
    """On Streamlit Cloud: log in to BigQuery with the read-only service account in the app's
    Secrets ([gcp_service_account]). Locally (no secrets) gcloud's login is used as usual."""
    try:
        import streamlit as st

        info = dict(st.secrets["gcp_service_account"])
    except Exception:  # noqa: BLE001 - no secrets file / no such key: nothing to do
        return False
    from google.cloud import bigquery
    from google.oauth2 import service_account

    from src import bq_reader

    credentials = service_account.Credentials.from_service_account_info(info)
    bq_reader._client = bigquery.Client(project=bq_reader.PROJECT, location=bq_reader.LOCATION,
                                        credentials=credentials)
    return True


def _bq_read(name: str) -> pd.DataFrame:
    from src import bq_reader

    reader = bq_reader.read_view if SOURCES[name] == "view" else bq_reader.read_table
    return reader(name, timeout=BQ_TIMEOUT_S)


# ---------------------------------------------------------------------------------------------

@dataclass
class Data:
    tables: dict[str, pd.DataFrame | None]
    source: dict[str, str]  # name -> "bigquery" | "local" | "missing"
    problems: dict[str, str] = field(default_factory=dict)
    bigquery: str = "ok"  # ok | off (VSE_DATA=local) | auth (not logged in) | unreachable

    @property
    def notice(self) -> str | None:
        """One plain sentence when local data is used; raw errors stay in self.problems."""
        if not self.any_local:
            return None
        return {
            "off": "Using local data: BigQuery is switched off (VSE_DATA=local).",
            "auth": "Using local data: run `gcloud auth application-default login` to read from BigQuery.",
        }.get(self.bigquery, "Using local data: BigQuery isn't reachable right now.")

    def get(self, name: str) -> pd.DataFrame | None:
        return self.tables.get(name)

    @property
    def any_local(self) -> bool:
        return any(s == "local" for s in self.source.values())

    @property
    def all_bigquery(self) -> bool:
        return all(s == "bigquery" for s in self.source.values())


def _load_one(name: str, use_bq: bool) -> tuple[pd.DataFrame | None, str, str]:
    problem = ""
    if use_bq:
        try:
            return _bq_read(name), "bigquery", ""
        except Exception as error:  # noqa: BLE001 - any failure means: try the local copy
            problem = f"BigQuery: {str(error).splitlines()[0][:200]}"
    loader = LOCAL_LOADERS.get(name)
    if loader is not None:
        try:
            return loader(), "local", problem
        except Exception as error:  # noqa: BLE001
            problem = (problem + "; " if problem else "") + f"local: {str(error).splitlines()[0][:200]}"
    return None, "missing", problem


def _normalize(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Same dtypes whichever source the table came from (BigQuery dates vs local strings)."""
    if df is None:
        return None
    df = df.copy()
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"].astype(str))
    if "has_brief" in df.columns:
        df["has_brief"] = df["has_brief"].fillna(False).astype(bool)
    return df


AUTH_HINTS = ("not authenticated", "credentials", "refresherror", "invalid_grant", "reauth")


def _bigquery_status(problems: dict[str, str], use_bq: bool) -> str:
    if not use_bq and not problems:
        return "off"
    text = " ".join(problems.values()).lower()
    if not text:
        return "ok"
    return "auth" if any(hint in text for hint in AUTH_HINTS) else "unreachable"


def load_all() -> Data:
    """Load every table; BigQuery unless VSE_DATA=local or the first query fails."""
    names = list(SOURCES)
    use_bq = os.environ.get("VSE_DATA", "").lower() != "local"
    wanted_bq = use_bq
    if use_bq:
        use_streamlit_secrets()
    tables, source, problems = {}, {}, {}
    if use_bq:  # one probe first, so an offline laptop doesn't wait for every table to time out
        df, src, problem = _load_one("v_headline_metrics", use_bq=True)
        use_bq = src == "bigquery"
        tables["v_headline_metrics"], source["v_headline_metrics"] = df, src
        if problem:
            problems["v_headline_metrics"] = problem
        names.remove("v_headline_metrics")
    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        results = dict(zip(names, pool.map(lambda n: _load_one(n, use_bq), names), strict=True))
    for name, (df, src, problem) in results.items():
        tables[name], source[name] = _normalize(df), src
        if problem:
            problems[name] = problem
    if tables.get("v_headline_metrics") is None:
        computed = headline_from_tables(tables)
        tables["v_headline_metrics"] = computed
        source["v_headline_metrics"] = "local" if computed is not None else "missing"
    status = "off" if not wanted_bq else _bigquery_status(
        {k: v for k, v in problems.items() if v.startswith("BigQuery")}, True)
    return Data(tables, source, problems, status)


# ---------------------------------------------------------------------------------------------
# derived numbers (all from the tables, nothing hardcoded)

def metric(recall_metrics: pd.DataFrame | None, *, scorer: str, metric: str,
           source_file: str = "recall_metrics_stg.csv", top_k: int | None = None,
           split: str = "test", parts: str = "all") -> float | None:
    if recall_metrics is None or recall_metrics.empty:
        return None
    m = recall_metrics[(recall_metrics["source_file"] == source_file)
                       & (recall_metrics["scorer"] == scorer) & (recall_metrics["metric"] == metric)
                       & (recall_metrics["split"] == split) & (recall_metrics["parts"] == parts)]
    if top_k is not None:
        m = m[m["top_k"] == top_k]
    return float(m["value"].iloc[0]) if len(m) else None


def waymo_fde(waymo_results: pd.DataFrame | None, predictor: str, object_type: str = "all",
              horizon_s: int = 8) -> float | None:
    if waymo_results is None or waymo_results.empty:
        return None
    w = waymo_results[(waymo_results["predictor"] == predictor)
                      & (waymo_results["object_type"] == object_type)
                      & (waymo_results["horizon_s"] == horizon_s)]
    return float(w["fde_m"].iloc[0]) if len(w) else None


def headline_from_tables(tables: dict[str, pd.DataFrame | None]) -> pd.DataFrame | None:
    """Same columns as the BigQuery view v_headline_metrics, computed from the other tables."""
    rm, wr = tables.get("recall_metrics"), tables.get("waymo_results")
    wb = tables.get("v_watchlist_with_briefs")
    if rm is None and wr is None and wb is None:
        return None
    briefs = wb[wb["has_brief"]] if wb is not None else pd.DataFrame()
    cv, best = waymo_fde(wr, "constant_velocity"), waymo_fde(wr, "model_best_guess")
    row = {
        "model_precision_at_20": metric(rm, scorer="lgbm", metric="precision_at_20"),
        "baseline_precision_at_20": metric(rm, scorer="baseline", metric="precision_at_20"),
        "model_pr_auc": metric(rm, scorer="lgbm", metric="pr_auc"),
        "model_median_weeks_warning": metric(rm, scorer="lgbm", metric="median_weeks_warning"),
        "model_recalls_caught_top20": metric(rm, scorer="lgbm", metric="recalls_caught",
                                             source_file="recall_topk_stg.csv", top_k=20),
        "model_recalls_caught_top100": metric(rm, scorer="lgbm", metric="recalls_caught",
                                              source_file="recall_topk_stg.csv", top_k=100),
        "watchlist_months": wb["month"].nunique() if wb is not None else None,
        "briefs": len(briefs),
        "briefs_fully_valid": int(((~briefs["still_failing"].astype(bool))
                                   & (briefs["uncited_bullets"] == 0)
                                   & (briefs["citations_valid"] == briefs["citations_total"])).sum())
        if len(briefs) else 0,
        "citation_valid_rate": (briefs["citations_valid"].sum() / briefs["citations_total"].sum())
        if len(briefs) and briefs["citations_total"].sum() else None,
        "waymo_cv_fde_8s_m": cv,
        "waymo_model_fde_8s_m": best,
        "waymo_model_min_fde_8s_m": waymo_fde(wr, "model_best_of_6"),
        "waymo_best_guess_vs_cv_pct": 100 * (1 - best / cv) if cv and best else None,
        "last_published_at": None,
    }
    return pd.DataFrame([row])


def month_label(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m")
