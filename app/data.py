"""Data for the demo, one code path for every place it runs.

Where the tables come from, in order:
  1. FINAL  - a snapshot whose manifest says "final": true (BigQuery's free tables have expired).
  2. LIVE   - BigQuery (service account from Streamlit secrets, or gcloud's login locally).
  3. SAVED  - if BigQuery fails for any reason, the latest snapshot.
  4. LOCAL  - DuckDB / files on a dev machine that has data/ (or VSE_DATA=local).

Snapshots live in the public repo's `snapshot` branch (read over HTTPS, so a final copy goes live
without a merge) and, as a fallback, in app/snapshot/ bundled with the app.

    FORCE_MODE=live|snapshot|final   preview a mode locally
    VSE_DATA=local                   skip BigQuery and snapshots, use local data (dev)
    VSE_SNAPSHOT_REMOTE=0            don't fetch the snapshot branch (tests, offline)

Trajectory coordinates are never published: the interactive trajectory view only exists on a dev
machine with data/; everywhere else the app shows pre-rendered images (app/static/trajectory/).
"""

from __future__ import annotations

import io
import json
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
APP = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
BQ_TIMEOUT_S = 20
REPO = "seangee9008/Vehicle-Safety-Early-Warning"
REMOTE_SNAPSHOT = f"https://raw.githubusercontent.com/{REPO}/snapshot/app/snapshot/"
STATIC_TRAJECTORY = APP / "static" / "trajectory"

# name -> "view" or "table" in BigQuery; every one of these is also in a snapshot
SOURCES = {
    "v_headline_metrics": "view",
    "v_watchlist_with_briefs": "view",
    "recall_metrics": "table",
    "recall_audit": "table",
    "agent_citations": "table",
    "waymo_results": "table",
    "publish_log": "table",
    "dataset_facts": "table",
    "recall_ablation": "table",
    "recall_overlap": "table",
    "recall_miss_reasons": "table",
    "label_audit_misses": "table",
    "agent_complaints": "table",
    "agent_monthly": "table",
    "agent_brief_stats": "table",
}

BRIEF_COLUMNS = ["brief_id", "brief_markdown", "model_used", "citations_valid", "citations_total",
                 "uncited_bullets", "evidence_bullets", "passed_first_try", "passed_after_retry",
                 "still_failing", "similar_recall_search_scope"]


def snapshot_dir() -> Path:
    return Path(os.environ.get("VSE_SNAPSHOT_DIR", APP / "snapshot"))


# ---------------------------------------------------------------------------------------------
# local (dev machine) loaders: the same builders the BigQuery publish uses, so the columns match

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


def _local_publish_log() -> pd.DataFrame:
    from src import bq_publish

    return bq_publish.read_ledger()


LOCAL_LOADERS: dict[str, Callable[[], pd.DataFrame]] = {
    "v_watchlist_with_briefs": _local_watchlist_with_briefs,
    "recall_metrics": _local("build_recall_metrics"),
    "recall_audit": _local("build_recall_audit"),
    "agent_citations": _local("build_agent_citations"),
    "waymo_results": _local("build_waymo_results"),
    "publish_log": _local_publish_log,
    "dataset_facts": _local("build_dataset_facts"),
    "recall_ablation": _local("build_recall_ablation"),
    "recall_overlap": _local("build_recall_overlap"),
    "recall_miss_reasons": _local("build_recall_miss_reasons"),
    "label_audit_misses": _local("build_label_audit_misses"),
    "agent_complaints": _local("build_agent_complaints"),
    "agent_monthly": _local("build_agent_monthly"),
    "agent_brief_stats": _local("build_agent_brief_stats"),
    # v_headline_metrics is computed from the other tables (headline_from_tables)
}


def load_local_examples() -> pd.DataFrame | None:
    """Waymo demo examples with coordinates: only on a dev machine with data/, never published."""
    if os.environ.get("FORCE_MODE"):
        return None  # previewing a public mode: show what the public app shows
    from src import bq_publish

    path = bq_publish.WAYMO_DEMO
    if not path.exists():
        return None
    return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))


def static_examples() -> pd.DataFrame | None:
    """Pre-rendered trajectory examples (descriptions + errors, no coordinates) and their images."""
    path = STATIC_TRAJECTORY / "examples.csv"
    return pd.read_csv(path) if path.exists() else None


# ---------------------------------------------------------------------------------------------
# BigQuery

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


def _bq_expiry() -> datetime | None:
    from src import bq_reader

    return bq_reader.dataset_expiry(timeout=BQ_TIMEOUT_S)


# ---------------------------------------------------------------------------------------------
# snapshots

@dataclass
class Snapshot:
    manifest: dict
    origin: str  # "remote" (snapshot branch) or "bundled" (app/snapshot)
    read: Callable[[str], pd.DataFrame]


def _remote_snapshot() -> Snapshot | None:
    if os.environ.get("VSE_SNAPSHOT_REMOTE", "1") == "0":
        return None
    try:
        import requests

        response = requests.get(REMOTE_SNAPSHOT + "manifest.json", timeout=5)
        if response.status_code != 200:
            return None
        manifest = response.json()
    except Exception:  # noqa: BLE001 - offline, branch missing, bad JSON: use the bundled copy
        return None

    def read(name: str) -> pd.DataFrame:
        r = requests.get(REMOTE_SNAPSHOT + f"{name}.parquet", timeout=15)
        r.raise_for_status()
        return pd.read_parquet(io.BytesIO(r.content))
    return Snapshot(manifest, "remote", read)


def _bundled_snapshot() -> Snapshot | None:
    folder = snapshot_dir()
    manifest_path = folder / "manifest.json"
    if not manifest_path.exists():
        return None
    return Snapshot(json.loads(manifest_path.read_text(encoding="utf-8")), "bundled",
                    lambda name: pd.read_parquet(folder / f"{name}.parquet"))


def find_snapshot() -> Snapshot | None:
    """The snapshot branch first (lets a final copy go live without a merge), else the bundled one.
    A remote copy only wins if it's final or newer than the bundled one."""
    remote, bundled = _remote_snapshot(), _bundled_snapshot()
    if remote and bundled and not remote.manifest.get("final") and \
            str(remote.manifest.get("frozen_at", "")) < str(bundled.manifest.get("frozen_at", "")):
        return bundled
    return remote or bundled


# ---------------------------------------------------------------------------------------------

@dataclass
class Data:
    tables: dict[str, pd.DataFrame | None]
    source: dict[str, str]  # name -> "bigquery" | "snapshot" | "local" | "missing"
    problems: dict[str, str] = field(default_factory=dict)
    bigquery: str = "ok"  # ok | off | auth (not logged in) | unreachable
    mode: str = "live"  # live | saved | final | local | missing
    manifest: dict = field(default_factory=dict)
    expires_at: str | None = None  # earliest BigQuery table expiry (ISO, UTC)

    @property
    def notice(self) -> str | None:
        """One plain sentence when local data is used; raw errors stay in self.problems."""
        if self.mode != "local":
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


def _normalize(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Same dtypes whichever source the table came from (BigQuery dates vs local strings)."""
    if df is None:
        return None
    df = df.copy()
    for col in ("month", "received_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col].astype(str), errors="coerce")
    if "has_brief" in df.columns:
        df["has_brief"] = df["has_brief"].fillna(False).astype(bool)
    return df


def _load_each(names: list[str], reader: Callable[[str], pd.DataFrame], label: str,
               fallback: Callable[[str], tuple] | None = None) -> dict[str, tuple]:
    def one(name: str) -> tuple:
        try:
            return reader(name), label, ""
        except Exception as error:  # noqa: BLE001 - recorded; the table falls back or shows a notice
            problem = f"{label}: {str(error).splitlines()[0][:200]}"
            if fallback is not None:
                df, src, extra = fallback(name)
                return df, src, problem + (f"; {extra}" if extra else "")
            return None, "missing", problem
    with ThreadPoolExecutor(max_workers=max(1, len(names))) as pool:
        return dict(zip(names, pool.map(one, names), strict=True))


def _local_one(name: str) -> tuple:
    loader = LOCAL_LOADERS.get(name)
    if loader is None:
        return None, "missing", ""
    try:
        return loader(), "local", ""
    except Exception as error:  # noqa: BLE001
        return None, "missing", f"local: {str(error).splitlines()[0][:200]}"


AUTH_HINTS = ("not authenticated", "credentials", "refresherror", "invalid_grant", "reauth")


def _bigquery_status(problem: str) -> str:
    text = problem.lower()
    if not text:
        return "ok"
    return "auth" if any(hint in text for hint in AUTH_HINTS) else "unreachable"


def _finish(results: dict[str, tuple], **kwargs) -> Data:
    tables, source, problems = {}, {}, {}
    for name, (df, src, problem) in results.items():
        tables[name], source[name] = _normalize(df), src
        if problem:
            problems[name] = problem
    if tables.get("v_headline_metrics") is None:
        computed = headline_from_tables(tables)
        tables["v_headline_metrics"] = computed
        if computed is not None:
            source["v_headline_metrics"] = "local" if "local" in source.values() else source.get(
                "v_headline_metrics", "missing")
    return Data(tables, source, problems, **kwargs)


def load_all() -> Data:
    names = list(SOURCES)
    force = os.environ.get("FORCE_MODE", "").lower()
    if os.environ.get("VSE_DATA", "").lower() == "local":
        return _finish({n: _local_one(n) for n in names}, bigquery="off", mode="local")

    snap = find_snapshot()
    final = bool(snap and snap.manifest.get("final"))
    if force in ("snapshot", "final") or (final and force != "live"):
        if snap is None:
            return _finish({n: (None, "missing", "no snapshot") for n in names}, mode="missing")
        mode = "final" if (force == "final" or final) else "saved"
        return _finish(_load_each(names, snap.read, "snapshot"), mode=mode, manifest=snap.manifest,
                       expires_at=snap.manifest.get("bq_expires_at"), bigquery="off")

    # LIVE: one probe first, so an offline laptop doesn't wait for every table to time out
    use_streamlit_secrets()
    probe = _load_each(["v_headline_metrics"], _bq_read, "BigQuery")["v_headline_metrics"]
    status = _bigquery_status(probe[2])
    if status == "ok":
        rest = [n for n in names if n != "v_headline_metrics"]
        results = {"v_headline_metrics": probe, **_load_each(rest, _bq_read, "BigQuery", _local_one)}
        try:
            expiry = _bq_expiry()
        except Exception:  # noqa: BLE001 - expiry is only shown, never required
            expiry = None
        return _finish(results, mode="live", expires_at=expiry.isoformat() if expiry else None,
                       manifest=snap.manifest if snap else {})

    # BigQuery failed: the saved copy, else local data on a dev machine
    if snap is not None and force != "live":
        data = _finish(_load_each(names, snap.read, "snapshot"), mode="saved", manifest=snap.manifest,
                       expires_at=snap.manifest.get("bq_expires_at"), bigquery=status)
        data.problems["BigQuery"] = probe[2]
        return data
    data = _finish({n: _local_one(n) for n in names}, bigquery=status, mode="local")
    data.problems.setdefault("v_headline_metrics", probe[2])
    if all(s == "missing" for s in data.source.values()):
        data.mode = "missing"
    return data


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
    log = tables.get("publish_log")
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
        "last_published_at": last_published(log),
    }
    return pd.DataFrame([row])


def last_published(publish_log: pd.DataFrame | None) -> str | None:
    if publish_log is None or publish_log.empty:
        return None
    return str(pd.to_datetime(publish_log["published_at"], utc=True).max().isoformat())


def facts_from(table: pd.DataFrame | None) -> dict:
    """dataset_facts (key/value text) back into the nested dict the app reads."""
    out: dict = {}
    if table is None:
        return out
    for key, value in zip(table["key"], table["value"], strict=True):
        try:
            parsed: object = int(value)
        except ValueError:
            try:
                parsed = float(value)
            except ValueError:
                parsed = value
        node = out
        parts = str(key).split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = parsed
    return out


def agent_details(data: Data) -> dict[str, dict]:
    """Per brief: complaints on file, monthly counts, and cited complaints' date + excerpt."""
    out: dict[str, dict] = {}
    stats, monthly, cited = data.get("agent_brief_stats"), data.get("agent_monthly"), data.get("agent_complaints")
    if stats is not None:
        for b, n in zip(stats["brief_id"], stats["complaints_on_file"], strict=True):
            out.setdefault(b, {})["complaints_on_file"] = int(n)
    if monthly is not None:
        for b, group in monthly.groupby("brief_id"):
            out.setdefault(b, {})["monthly"] = [{"month": str(pd.Timestamp(m).date()), "complaints": float(c)}
                                                for m, c in zip(group["month"], group["complaints"], strict=True)]
    if cited is not None:
        for _, r in cited.iterrows():
            date = "" if pd.isna(r["received_date"]) else str(pd.Timestamp(r["received_date"]).date())
            out.setdefault(r["brief_id"], {}).setdefault("cited", {})[str(r["odino"])] = {
                "date": date, "excerpt": r["excerpt"]}
    return out


def month_label(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m")
