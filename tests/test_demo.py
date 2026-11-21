"""Streamlit demo (app/streamlit_app.py) with AppTest: no network, BigQuery and local data faked."""

from pathlib import Path

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from app import data as D
from src import bq_reader

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")
TABS = ["Overview", "Recall watchlist", "AI agent", "Model results", "Trajectory", "Data and limitations"]
RESULT_TILES = ["hit", "coverage", "lead", "drivers", "fair", "signals"]
TRAJ_TILES = ["horizon", "types", "struggles"]


def _path(n, dx=1.0, dy=0.0):
    return [[round(dx * i, 2), round(dy * i, 2)] for i in range(1, n + 1)]


BRIEF_MD = ("## Failure pattern (2-3 sentences)\nDrivers report the instrument cluster blackout while "
            "driving and sudden loss of power. Warning lights come on.\n\n## Severity\n- Crashes: 1\n- Deaths: 0\n"
            "- Fires: 0\n- Injuries: 2\n\n## Evidence\n- Screen goes dark [ODINO 111], [ODINO 222]\n"
            "- Car stalls on the highway [ODINO 333]\n\n## Watchlist reasons check\n"
            "- vehicle is under a year old: CONTEXT (new model year).\n"
            "- 13 complaints in the last 12 months: SUPPORT (complaints match).\n\n## Similar past recalls\n"
            "Search: same make (KIA)\nnone relevant\n\n## Suggested next step\n- Next step: Ask for dealer "
            "records [ODINO 111].\n")
BRIEF_IDS = ["2024-12-r01", "2024-12-r02"]


def fake_tables() -> dict[str, pd.DataFrame]:
    rows = []
    for rank in (1, 2, 3):
        has = rank <= 2
        rows.append({
            "month": "2024-12-01", "make": "KIA", "model": "EV9", "model_year": 2024,
            "component": "ADAS AND STABILITY CONTROL" if rank != 2 else "POWER TRAIN", "score": 0.7 - rank / 100,
            "rank": rank, "top_reasons": "vehicle is under a year old; 13 complaints in the last 12 months; "
                                         "2 fires in the last 3 months", "source": "stg",
            "brief_id": f"2024-12-r0{rank}" if has else None, "brief_markdown": BRIEF_MD if has else None,
            "model_used": "gemini-x" if has else None, "citations_valid": 3 if has else None,
            "citations_total": 3 if has else None, "uncited_bullets": 0 if has else None,
            "evidence_bullets": 2 if has else None, "passed_first_try": rank == 1, "passed_after_retry": rank == 2,
            "still_failing": False, "similar_recall_search_scope": "same_make" if has else None, "has_brief": has,
        })
    rows.append({"month": "2024-11-01", "make": "FORD", "model": "F150", "model_year": 2023,
                 "component": "POWER TRAIN", "score": 0.5, "rank": 1, "top_reasons": "9 complaints in the last 3 months",
                 "source": "stg", "has_brief": False, "passed_first_try": False, "passed_after_retry": False,
                 "still_failing": False})
    watch = pd.DataFrame(rows)
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
    citations = pd.DataFrame({"brief_id": [b for b in BRIEF_IDS for _ in range(3)],
                              "odino": ["111", "222", "333"] * 2, "valid": [True] * 6})
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
    facts = {"complaints.complaints": 1072108, "complaints.records": 1487991, "complaints.first": "2010-01-01",
             "complaints.last": "2026-11-16", "complaints.per_year": 62676, "recalls.campaigns": 13621,
             "recalls.records": 219987, "recalls.first": "2010-01-01", "recalls.last": "2026-10-29",
             "recalls.per_year": 808, "investigations": 5355, "model.car_parts_ranked_per_month": 92926,
             "model.test_first_month": "2022-01-01", "model.test_last_month": "2024-12-01",
             "waymo.train_scenes": 990, "waymo.test_scenes": 286, "agent.read_limit": 40,
             "generated_at": "2026-11-21T07:40:38+00:00"}
    tables = {
        "v_watchlist_with_briefs": watch, "recall_metrics": recall_metrics, "recall_audit": audit,
        "agent_citations": citations, "waymo_results": waymo,
        "publish_log": pd.DataFrame({"published_at": ["2026-11-19T06:01:58+00:00", "2026-11-21T09:25:47+00:00"],
                                     "table_name": ["watchlist", "watchlist"], "rows": [720, 720]}),
        "dataset_facts": pd.DataFrame({"key": list(facts), "value": [str(v) for v in facts.values()]}),
        "recall_ablation": pd.DataFrame({"model": ["Baseline (3-month complaint count)", "Age only", "Complaints only",
                                                   "v1 (complaints + age)"], "hit_rate_20": [0.089, 0.076, 0.064, 0.118],
                                         "hit_rate_100": [0.056, 0.065, 0.054, 0.079],
                                         "pr_auc": [0.016, 0.026, 0.017, 0.031]}),
        "recall_overlap": pd.DataFrame([{"top": 20, "caught_by_both": 13, "only_baseline": 17, "only_lgbm": 49,
                                         "both_lgbm_earlier": 8, "both_same_month": 1, "both_lgbm_later": 4,
                                         "both_median_weeks_baseline": 33.4, "both_median_weeks_lgbm": 37.0,
                                         "only_baseline_median_weeks": 35.1, "only_lgbm_median_weeks": 22.3}]),
        "recall_miss_reasons": pd.DataFrame({"reason": ["1-2 complaints in the 12 months before",
                                                        "no complaints in the 12 months before",
                                                        "3+ complaints, never in the top 100",
                                                        "3+ complaints, best rank 21-100"],
                                             "recalls": [719, 522, 280, 102], "share": [0.44, 0.32, 0.17, 0.06]}),
        "label_audit_misses": pd.DataFrame({"match": ["model-name variant", "recall 6-12 months out"],
                                            "misses": [16, 50], "share": [0.025, 0.078]}),
        "agent_complaints": pd.DataFrame({"brief_id": BRIEF_IDS, "odino": ["111", "111"],
                                          "received_date": ["2024-11-29", "2024-11-29"],
                                          "excerpt": ["The screen went black on the highway."] * 2}),
        "agent_monthly": pd.DataFrame({"brief_id": [b for b in BRIEF_IDS for _ in range(2)],
                                       "month": ["2024-11-01", "2024-12-01"] * 2, "complaints": [2.0, 5.0] * 2}),
        "agent_brief_stats": pd.DataFrame({"brief_id": BRIEF_IDS, "complaints_on_file": [31, 31]}),
    }
    tables["_examples"] = examples
    normalized = {k: D._normalize(v) for k, v in tables.items() if not k.startswith("_")}
    tables["v_headline_metrics"] = D.headline_from_tables(normalized)
    return tables


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    for var in ("VSE_DATA", "FORCE_MODE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("VSE_SNAPSHOT_REMOTE", "0")                 # no network
    monkeypatch.setenv("VSE_SNAPSHOT_DIR", str(tmp_path / "none"))  # no saved copy unless a test makes one
    examples = fake_tables()["_examples"]
    monkeypatch.setattr(D, "load_local_examples", lambda: examples.copy())
    monkeypatch.setattr(D, "_bq_expiry", lambda: None)
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


def save_snapshot(folder, tables, final=False):
    folder.mkdir(parents=True, exist_ok=True)
    for name in D.SOURCES:
        if name in tables:
            tables[name].to_parquet(folder / f"{name}.parquet", index=False)
    import json
    (folder / "manifest.json").write_text(json.dumps({
        "final": final, "frozen_at": "2027-01-15T18:00:00+00:00", "last_updated": "2026-11-21T09:25:47+00:00",
        "bq_expires_at": "2027-01-18T06:02:02+00:00", "source": "test", "tables": {}}))


def run_app(state: dict | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120)
    for key, value in (state or {}).items():
        at.session_state[key] = value
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    assert [t.label for t in at.tabs] == TABS
    return at


def text(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def all_tabs_text(at: AppTest) -> str:
    """Tabs are lazy (only the open one renders), so visit each and collect its text."""
    out = []
    for name in TABS:
        at.session_state["main_tab"] = name
        at.run()
        assert not at.exception, (name, [e.message for e in at.exception])
        out.append(text(at))
    return "\n".join(out)


def test_every_tab_renders_with_bigquery_data(monkeypatch):
    tables = fake_tables()
    fake_bigquery(monkeypatch, tables)
    fake_local(monkeypatch, tables, broken=set(tables))  # must not be needed

    at = run_app()
    page = all_tabs_text(at)

    assert "Live results" in page and "Local data" not in page
    assert "Sean Gee · Justine Li-Wu · Angelina Yang" in page
    assert "11.8%" in page and "6 of 6" in page and "3 unique complaints" in page  # KPI cards
    assert "WHY THIS PROJECT" in page and "Defects hide in complaints" in page    # overview story
    assert "Spot safety problems early" in page                                  # decision tree
    assert "New car" in page and "Fire reports" in page                          # watchlist tags
    assert "vehicle is under a year old" not in page.split("Flag reasons check")[0]  # never raw on the list
    assert "Built by Sean Gee, Justine Li-Wu, Angelina Yang." in page
    assert not at.info


@pytest.mark.parametrize("brief_id", BRIEF_IDS)
def test_every_agent_car_detail_renders(monkeypatch, brief_id):
    tables = fake_tables()
    fake_bigquery(monkeypatch, tables)

    at = run_app({"agent_car": brief_id, "main_tab": "AI agent"})
    page = text(at)

    assert "How the agent worked" in page and "31 of" in page               # stepper with real counts
    assert "Screens or gauges go dark" in page                              # symptom chips
    assert "Screen goes dark" in page and "Flag reasons check" in page
    assert "Supported" in page and "Context only" in page
    assert ("First try" if brief_id.endswith("1") else "After one retry") in page
    assert any(b.label == "Back to all cars" for b in at.button)


def test_watchlist_brief_button_opens_the_agent(monkeypatch):
    tables = fake_tables()
    fake_bigquery(monkeypatch, tables)
    at = run_app({"main_tab": "Recall watchlist"})

    at.button(key="wopen_0").click().run()

    assert at.session_state["agent_car"] == "2024-12-r01"
    assert at.session_state["main_tab"] == "AI agent"


@pytest.mark.parametrize("page_key, key", [("results", k) for k in RESULT_TILES]
                         + [("traj", k) for k in TRAJ_TILES])
def test_every_tile_detail_view_renders(monkeypatch, page_key, key):
    tables = fake_tables()
    fake_bigquery(monkeypatch, tables)

    tab = "Model results" if page_key == "results" else "Trajectory"
    at = run_app({f"detail_{page_key}": key, "main_tab": tab})

    labels = [b.label for b in at.button]
    assert "Back to overview" in labels or "Back to trajectories" in labels
    assert any(w in text(at) for w in ("What we found", "What it measures"))


@pytest.mark.parametrize("size, scale", [("S", "0.86"), ("M", "1.0"), ("L", "1.18")])
def test_text_size_scales_the_page(monkeypatch, size, scale):
    tables = fake_tables()
    fake_bigquery(monkeypatch, tables)

    at = run_app({"text_size": size})

    assert f"--s: {scale};" in text(at)


def test_tiles_and_widgets(monkeypatch):
    tables = fake_tables()
    fake_bigquery(monkeypatch, tables)
    at = run_app({"main_tab": "Model results"})

    at.button(key="open_results_hit").click().run()
    assert at.session_state["detail_results"] == "hit"
    at.button(key="back_results").click().run()
    assert at.session_state["detail_results"] is None
    at.session_state["main_tab"] = "Recall watchlist"
    at.run()
    at.selectbox(key="watch_month").select("Nov 2024").run()
    at.session_state["main_tab"] = "Trajectory"
    at.run()
    at.button_group(key="traj_type").set_value("pedestrian").run()
    assert not at.exception, [e.message for e in at.exception]
    assert at.session_state["traj_example"] == "s1_1"


def test_local_fallback_when_bigquery_unreachable(monkeypatch):
    tables = fake_tables()
    fake_bigquery(monkeypatch, {}, broken=set(tables), error="connection timed out")
    fake_local(monkeypatch, tables)

    at = run_app({"detail_results": "fair", "main_tab": "Model results"})

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


@pytest.mark.parametrize("name", ["agent_citations", "recall_audit", "v_watchlist_with_briefs",
                                  "recall_metrics", "waymo_results", "agent_complaints", "dataset_facts"])
def test_missing_table_shows_notice_and_other_tabs_still_work(monkeypatch, name):
    tables = fake_tables()
    fake_bigquery(monkeypatch, tables, broken={name})
    fake_local(monkeypatch, tables, broken={name})

    at = run_app()

    notices = []
    for tab in TABS:
        at.session_state["main_tab"] = tab
        at.run()
        assert not at.exception, (tab, [e.message for e in at.exception])
        notices += [i.body for i in at.info]
    if name not in ("agent_citations", "recall_audit", "agent_complaints", "dataset_facts"):  # degrade quietly
        assert any(name in n for n in notices), notices
    assert "Data sources" in text(at)  # the last tab (Data and limitations) still renders


def test_headline_from_tables_matches_inputs():
    tables = {k: D._normalize(v) for k, v in fake_tables().items()}
    h = D.headline_from_tables(tables).iloc[0]

    assert h["model_precision_at_20"] == pytest.approx(0.118)
    assert h["briefs"] == 2 and h["briefs_fully_valid"] == 2
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


# ---------------------------------------------------------------------------------------------
# pure helpers

@pytest.mark.parametrize("reason, tag", [
    ("vehicle is under a year old", "New car"), ("vehicle is 1 year old", "New car"),
    ("vehicle is 3 years old", "Older car"), ("42 complaints in the last 12 months", "Many complaints"),
    ("complaints up 3.1x vs. the prior 9 months", "Complaints rising"),
    ("93% of this car's complaints in the last 3 months", "Complaints rising"),
    ("2 fires in the last 3 months", "Fire reports"), ("3 crashes in the last 3 months", "Crashes reported"),
    ("2 complaints with injuries in the last 3 months", "Injuries reported"),
    ("4 complaints mentioning smoke in the last 3 months", "Smoke reports"),
    ("5 complaints mentioning a tow in the last 3 months", "Towed away"),
    ("7 complaints saying it happened without warning in the last 3 months", "No warning"),
    ("1 complaints mentioning lost power or a shut-off in the last 3 months", "Power loss"),
    ("median 12,000 miles at complaint", "Mileage pattern"),
])
def test_every_reason_pattern_maps_to_a_short_tag(reason, tag):
    from app import components as ui

    assert ui.reason_tag(reason) == tag
    assert len(tag.split()) <= 2


def test_symptom_chips_are_short_and_three():
    from app import briefs

    chips = briefs.symptom_chips("Owners report sudden blackouts of the screen, loss of power and warning lights.")
    assert len(chips) == 3 and all(len(c.split()) <= 8 for c in chips)
    assert chips[0] == "Screens or gauges go dark"
    assert len(briefs.symptom_chips("Something odd happens with the vehicle sometimes.")) >= 1


def test_motion_descriptions():
    from app import motion

    straight = {"object_type": "vehicle", "past_path": [[-1.0, 0.0], [0.0, 0.0]],
                "true_future": _path(80, 1.0, 0.0)}
    left = {"object_type": "cyclist", "past_path": [[-0.5, 0.0], [0.0, 0.0]],
            "true_future": [[i * 0.5, 0.0] for i in range(40)] + [[20.0, i * 0.5] for i in range(40)]}
    assert motion.describe(straight) == "Car going straight · 36 km/h"
    assert motion.describe(left).startswith("Cyclist turning left · 18 km/h")


def test_palette_meets_wcag_aa():
    from app import theme

    for fg, bg in [(theme.INK, theme.BG), (theme.INK, theme.CARD), (theme.MUTED, theme.CARD),
                   (theme.TRACK_A, theme.CARD), (theme.TRACK_B, theme.CARD), (theme.BASELINE, theme.CARD),
                   (theme.BG, theme.TRACK_A), (theme.GOOD, theme.CARD)]:
        assert theme.contrast(fg, bg) >= 4.5, (fg, bg)
    for color in theme.PART_COLORS.values():  # dark text on part chips
        assert theme.contrast(theme.BG, color) >= 4.5, color
    for color in theme.ROAD_USER.values():
        assert theme.contrast(color, theme.CARD) >= 4.5, color
    assert theme.contrast(theme.ALERT, theme.CARD) >= 3.0  # alert is only used for large numbers and icons


# ---------------------------------------------------------------------------------------------
# data modes: live / saved copy / final

def test_live_mode_shows_expiry(monkeypatch):
    tables = fake_tables()
    fake_bigquery(monkeypatch, tables)
    from datetime import UTC, datetime
    monkeypatch.setattr(D, "_bq_expiry", lambda: datetime(2027, 1, 18, 6, 2, tzinfo=UTC))

    page = text(run_app())

    assert "Live results · ends Jan 17, 2027" in page  # Pacific time


def test_saved_copy_when_bigquery_fails(monkeypatch, tmp_path):
    tables = fake_tables()
    save_snapshot(tmp_path / "snap", tables)
    monkeypatch.setenv("VSE_SNAPSHOT_DIR", str(tmp_path / "snap"))
    fake_bigquery(monkeypatch, {}, broken=set(tables), error="403 table expired")

    at = run_app()
    page = all_tabs_text(at)

    assert "Saved copy · Jan 15, 2027, 10:00 AM PT" in page and "Live data ended" not in page
    assert "11.8%" in page


def test_final_snapshot_wins_even_if_bigquery_works(monkeypatch, tmp_path):
    tables = fake_tables()
    save_snapshot(tmp_path / "snap", tables, final=True)
    monkeypatch.setenv("VSE_SNAPSHOT_DIR", str(tmp_path / "snap"))
    monkeypatch.setattr(bq_reader, "read_table", lambda *a, **k: pytest.fail("BigQuery used after final"))
    monkeypatch.setattr(bq_reader, "read_view", lambda *a, **k: pytest.fail("BigQuery used after final"))

    page = all_tabs_text(run_app())

    assert "Live data ended Jan 17, 2027 (BigQuery free tier limit)" in page
    assert "final copy saved Jan 15, 2027, 10:00 AM PT" in page
    assert "Data last updated Nov 21, 2026, 1:25 AM PT" in page


@pytest.mark.parametrize("mode, expected", [("snapshot", "Saved copy"), ("final", "Live data ended")])
def test_force_mode_previews(monkeypatch, tmp_path, mode, expected):
    tables = fake_tables()
    save_snapshot(tmp_path / "snap", tables)
    monkeypatch.setenv("VSE_SNAPSHOT_DIR", str(tmp_path / "snap"))
    monkeypatch.setenv("FORCE_MODE", mode)
    fake_bigquery(monkeypatch, tables)

    assert expected in text(run_app())


def test_public_trajectory_uses_images_not_coordinates(monkeypatch):
    tables = fake_tables()
    fake_bigquery(monkeypatch, tables)
    monkeypatch.setattr(D, "load_local_examples", lambda: None)  # no data/ (public app)

    at = run_app({"main_tab": "Trajectory"})

    assert len(at.get("image")) == 1                # the pre-rendered picture
    assert not at.get("plotly_chart") or all("Meters ahead" not in str(c.proto) for c in at.get("plotly_chart"))
    assert "described from their own paths" in text(at)
