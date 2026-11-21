"""Streamlit demo (app/streamlit_app.py) with AppTest: no network, BigQuery and local data faked."""

from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app import data as D
from src import bq_reader

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
TABS = ["Overview", "Recall watchlist", "Investigation brief", "Model results", "Trajectory",
        "Data and limitations"]
RESULT_TILES = ["hit", "coverage", "lead", "drivers", "fair", "signals"]
TRAJ_TILES = ["horizon", "types", "struggles"]


def _path(n, dx=1.0, dy=0.0):
    return [[round(dx * i, 2), round(dy * i, 2)] for i in range(1, n + 1)]


def fake_tables() -> dict[str, pd.DataFrame]:
    watch = pd.DataFrame([{
        "month": "2024-12-01", "make": "KIA", "model": "EV9", "model_year": 2024,
        "component": "ADAS AND STABILITY CONTROL", "score": 0.7 - rank / 100, "rank": rank,
        "top_reasons": "vehicle is under a year old", "source": "stg",
        "brief_id": "2024-12-r01" if rank == 1 else None,
        "brief_markdown": ("## Failure pattern (2-3 sentences)\nThe screen goes dark.\n\n## Severity\n- Crashes: 0\n\n"
                           "## Evidence\n- Screen goes dark [ODINO 111], [ODINO 222]\n\n## Similar past recalls\n"
                           "Search: same make (KIA)\nnone relevant\n") if rank == 1 else None,
        "model_used": "gemini-x" if rank == 1 else None,
        "citations_valid": 2 if rank == 1 else None, "citations_total": 2 if rank == 1 else None,
        "uncited_bullets": 0 if rank == 1 else None, "evidence_bullets": 3 if rank == 1 else None,
        "passed_first_try": rank == 1, "passed_after_retry": False, "still_failing": False,
        "similar_recall_search_scope": "same_make" if rank == 1 else None, "has_brief": rank == 1,
    } for rank in (1, 2, 3)] + [{
        "month": "2024-11-01", "make": "FORD", "model": "F150", "model_year": 2023,
        "component": "POWER TRAIN", "score": 0.5, "rank": 1, "top_reasons": "9 complaints",
        "source": "stg", "has_brief": False, "passed_first_try": False, "passed_after_retry": False,
        "still_failing": False}])
    metrics = []
    for scorer, p, auc, weeks in [("lgbm", 0.118, 0.031, 23.2), ("baseline", 0.089, 0.016, 35.1)]:
        for metric, value in [("precision_at_20", p), ("pr_auc", auc), ("median_weeks_warning", weeks),
                              ("first_recalls_with_complaints", 1685)]:
            metrics.append(("recall_metrics_stg.csv", "test", "all", scorer, 20, metric, value))
        metrics.append(("recall_metrics_stg.csv", "val", "all", scorer, 20, "precision_at_20", p / 2))
        for k in (20, 50, 100):
            for metric, value in [("precision", p), ("recalls_caught", k * 2), ("pct_recalls_caught", k / 1000),
                                  ("median_weeks_warning", weeks)]:
                metrics.append(("recall_topk_stg.csv", "test", "all", scorer, k, metric, value))
    recall_metrics = pd.DataFrame(metrics, columns=["source_file", "split", "parts", "scorer", "top_k",
                                                    "metric", "value"])
    audit = pd.DataFrame([
        ("label_audit", "label_audit_stg.csv", "exact label (Justine's y_6m)", "test", "lgbm_precision_at_20", 0.114),
        ("label_audit", "label_audit_stg.csv", "exact label (Justine's y_6m)", "test", "base_rate", 0.0026),
        ("label_audit", "label_audit_stg.csv", "+ model-name variant (4xe, L, Hybrid, ...)", "test",
         "lgbm_precision_at_20", 0.136),
        ("investigations", "x.csv", "v1", "val", "precision_at_20", 0.071),
        ("investigations", "x.csv", "v1 + investigations", "val", "precision_at_20", 0.0875),
        ("investigations", "x.csv", "v1 (Justine's predictions)", "test", "precision_at_20", 0.114),
        ("investigations", "x.csv", "v1 + investigations", "test", "precision_at_20", 0.140),
        ("investigations", "x.csv", "v1 (Justine's predictions)", "test", "catch_rate_first_with_complaints", 0.037),
        ("investigations", "x.csv", "v1 + investigations", "test", "catch_rate_first_with_complaints", 0.026),
    ], columns=["experiment", "source_file", "variant", "split", "metric", "value"])
    citations = pd.DataFrame({"brief_id": ["2024-12-r01"] * 2, "odino": ["111", "222"], "valid": [True, True]})
    waymo = pd.DataFrame([{
        "object_type": t, "horizon_s": h, "n_agents": 10, "predictor": pr,
        "ade_m": 1.0, "fde_m": {"constant_velocity": 2.0 * h, "model_best_guess": 1.5 * h,
                                "model_best_of_6": 0.5 * h}[pr],
        "fde_per_m_mean": None, "fde_per_m_median": None,
        "fde_improvement_vs_cv_pct": None if pr == "constant_velocity" else
        {"model_best_guess": 25.0, "model_best_of_6": 75.0}[pr],
    } for t in ("all", "vehicle", "pedestrian", "cyclist") for h in (3, 5, 8)
        for pr in ("constant_velocity", "model_best_guess", "model_best_of_6")])
    examples = pd.DataFrame([{
        "example_id": f"s{i}_1", "scenario_id": f"s{i}", "track_id": 1, "object_type": kind,
        "frame": "agent frame", "past_path": [[-1.0, 0.0], None, [0.0, 0.0]],
        "true_future": _path(80, 0.5, 0.1 * i), "baseline_path": _path(80, 0.6),
        "model_paths": [_path(80, 0.5, 0.01 * k) for k in range(6)],
        "model_confidences": [0.5, 0.2, 0.1, 0.1, 0.05, 0.05],
        "lane_polylines": [[[0.0, 2.0], [40.0, 2.0]]],
        "fde_8s_cv_m": 9.0, "fde_8s_best_m": 3.0, "fde_8s_min_m": 2.0, "travelled_m": 40.0,
        "checkpoint": "mlp.pt",
    } for i, kind in enumerate(["vehicle", "pedestrian", "cyclist"])])
    tables = {"v_watchlist_with_briefs": watch, "recall_metrics": recall_metrics, "recall_audit": audit,
              "agent_citations": citations, "waymo_results": waymo, "waymo_demo_examples": examples}
    normalized = {k: D._normalize(v) for k, v in tables.items()}
    tables["v_headline_metrics"] = D.headline_from_tables(normalized)
    return tables


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv("VSE_DATA", raising=False)
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def fake_bigquery(monkeypatch, tables, broken=(), error="not found"):
    def read(name, limit=None, timeout=None):
        if name in broken or name not in tables:
            raise RuntimeError(f"{name} {error}")
        return tables[name].copy()
    monkeypatch.setattr(bq_reader, "read_table", read)
    monkeypatch.setattr(bq_reader, "read_view", read)


def fake_local(monkeypatch, tables, broken=()):
    def loader(name):
        def build():
            if name in broken:
                raise FileNotFoundError(f"no local {name}")
            return tables[name].copy()
        return build
    monkeypatch.setattr(D, "LOCAL_LOADERS", {n: loader(n) for n in D.LOCAL_LOADERS})


def run_app(state: dict | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=90)
    for key, value in (state or {}).items():
        at.session_state[key] = value
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert [t.label for t in at.tabs] == TABS
    return at


def text(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_every_tab_renders_with_bigquery_data(monkeypatch):
    tables = fake_tables()
    fake_bigquery(monkeypatch, tables)
    fake_local(monkeypatch, tables, broken=set(tables))  # must not be needed

    at = run_app()
    page = text(at)

    assert "Live results" in page and "Local data" not in page
    assert "11.8%" in page and "2 of 2" in page and "2 unique complaints" in page  # KPI cards
    assert "Screen goes dark" in page and "ODINO 111" in page                     # brief cards
    assert "What drivers report" in page and "Similar past recalls: none found" in page
    assert "Data sources" in page and "Honest limitations" in page
    assert at.selectbox(key="watch_month").value == "Dec 2024"
    assert not at.info


@pytest.mark.parametrize("page_key, key", [("results", k) for k in RESULT_TILES]
                         + [("traj", k) for k in TRAJ_TILES])
def test_every_tile_detail_view_renders(monkeypatch, page_key, key):
    tables = fake_tables()
    fake_bigquery(monkeypatch, tables)

    at = run_app({f"detail_{page_key}": key})

    labels = [b.label for b in at.button]
    assert "Back to overview" in labels or "Back to trajectories" in labels
    assert any(w in text(at) for w in ("What we found", "What it measures"))


def test_tiles_open_and_close(monkeypatch):
    tables = fake_tables()
    fake_bigquery(monkeypatch, tables)
    at = run_app()

    at.button(key="open_results_hit").click().run()
    assert at.session_state["detail_results"] == "hit"
    at.button(key="back_results").click().run()
    assert at.session_state["detail_results"] is None
    assert not at.exception


def test_widgets_work(monkeypatch):
    tables = fake_tables()
    fake_bigquery(monkeypatch, tables)
    at = run_app()

    at.selectbox(key="watch_month").select("Nov 2024").run()
    at.button_group(key="traj_type").set_value("pedestrian").run()

    assert not at.exception, [e.message for e in at.exception]
    assert at.selectbox(key="traj_pick").label.endswith("pedestrians)")


def test_local_fallback_when_bigquery_unreachable(monkeypatch):
    tables = fake_tables()
    fake_bigquery(monkeypatch, {}, broken=set(tables), error="connection timed out")
    fake_local(monkeypatch, tables)

    at = run_app({"detail_results": "fair"})

    page = text(at)
    assert "Local data" in page and "isn't reachable" in page
    assert "Are we grading fairly?" in page


def test_missing_login_shows_one_friendly_line(monkeypatch):
    tables = fake_tables()
    fake_bigquery(monkeypatch, {}, broken=set(tables),
                  error="Not authenticated to BigQuery. Run `gcloud auth application-default login`")
    fake_local(monkeypatch, tables)

    at = run_app()

    page = text(at)
    assert "run gcloud auth application-default login to read from BigQuery" in page
    assert "Not authenticated" not in page                      # raw error only in Technical details
    assert any("Not authenticated" in c.value for c in at.code)


def test_vse_data_local_skips_bigquery(monkeypatch):
    tables = fake_tables()
    monkeypatch.setenv("VSE_DATA", "local")
    monkeypatch.setattr(bq_reader, "read_table", lambda *a, **k: pytest.fail("BigQuery used"))
    monkeypatch.setattr(bq_reader, "read_view", lambda *a, **k: pytest.fail("BigQuery used"))
    fake_local(monkeypatch, tables)

    at = run_app()

    assert "switched off" in text(at)


@pytest.mark.parametrize("name", ["waymo_demo_examples", "agent_citations", "recall_audit",
                                  "v_watchlist_with_briefs", "recall_metrics", "waymo_results"])
def test_missing_table_shows_notice_and_other_tabs_still_work(monkeypatch, name):
    tables = fake_tables()
    fake_bigquery(monkeypatch, tables, broken={name})
    fake_local(monkeypatch, tables, broken={name})

    at = run_app()

    if name not in ("agent_citations", "recall_audit"):  # these two only shrink a card / drop a tile
        assert any(name in i.body for i in at.info), [i.body for i in at.info]
    assert "Data sources" in text(at)  # the rest of the app still renders


def test_headline_from_tables_matches_inputs():
    tables = {k: D._normalize(v) for k, v in fake_tables().items()}
    h = D.headline_from_tables(tables).iloc[0]

    assert h["model_precision_at_20"] == pytest.approx(0.118)
    assert h["briefs"] == 1 and h["briefs_fully_valid"] == 1
    assert h["waymo_best_guess_vs_cv_pct"] == pytest.approx(25.0)


def test_no_secrets_means_local_gcloud_login(monkeypatch):
    monkeypatch.setattr(bq_reader, "_client", None)
    assert D.use_streamlit_secrets() is False
    assert bq_reader._client is None


def test_report_readers_parse_committed_files():
    from app import reports

    assert reports.overlap(20) is not None
    assert len(reports.ablation()) == 4
    assert reports.label_audit_misses()["misses"].sum() > 0
    assert reports.miss_reasons()["recalls"].sum() > 0
