"""The public app against the bundled saved copy (app/snapshot/): no BigQuery, no data/, no secrets.

FORCE_MODE (snapshot | final) comes from the environment; CI runs both with only
app/requirements.txt installed.
"""

import json
import os
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "streamlit_app.py")
SNAPSHOT = ROOT / "app" / "snapshot"
TABS = ["Overview", "Recall watchlist", "AI agent", "Model results", "Trajectory", "Data and limitations"]


@pytest.fixture
def mode(monkeypatch):
    mode = os.environ.get("FORCE_MODE") if os.environ.get("FORCE_MODE") in ("snapshot", "final") else "snapshot"
    monkeypatch.setenv("FORCE_MODE", mode)
    monkeypatch.setenv("VSE_SNAPSHOT_REMOTE", "0")
    monkeypatch.delenv("VSE_SNAPSHOT_DIR", raising=False)
    monkeypatch.delenv("VSE_DATA", raising=False)
    st.cache_data.clear()
    yield mode
    st.cache_data.clear()


def test_every_tab_loads_from_the_bundled_copy(mode):
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    pages = []
    for tab in TABS:
        at.session_state["main_tab"] = tab
        at.run()
        assert not at.exception, (tab, [e.message for e in at.exception])
        assert not at.info, (tab, [i.body for i in at.info])  # no missing table anywhere
        pages.append("\n".join(m.value for m in at.markdown))
    page = "\n".join(pages)
    if mode == "final":
        assert "Live data ended" in page and "BigQuery free tier limit" in page
    else:
        assert "Saved copy ·" in page
    assert "11.8%" in page and "81 of 81" in page


def test_agent_details_and_trajectory_images(mode):
    briefs = json.loads((SNAPSHOT / "manifest.json").read_text())["tables"]
    assert briefs["agent_complaints"] > 0
    at = AppTest.from_file(APP, default_timeout=180)
    at.session_state["main_tab"] = "AI agent"
    at.session_state["agent_car"] = "2024-12-r06"
    at.run()
    assert not at.exception
    page = "\n".join(m.value for m in at.markdown)
    assert "How the agent worked" in page and "31 of 31" in page
    at.session_state["main_tab"] = "Trajectory"
    at.run()
    assert len(at.get("image")) == 1 and not at.exception


def test_bundled_manifest_is_complete_and_clean():
    manifest = json.loads((SNAPSHOT / "manifest.json").read_text())
    assert set(manifest) >= {"final", "frozen_at", "last_updated", "bq_expires_at", "source", "tables"}
    assert all(rows > 0 for rows in manifest["tables"].values())
    text = json.dumps(manifest).lower()
    assert "private_key" not in text and "client_email" not in text
    for name in manifest["tables"]:
        assert (SNAPSHOT / f"{name}.parquet").exists(), name
