"""BigQuery publish with a fake client: no network, no credentials."""

import dataclasses
import re

import pandas as pd
import pytest

from src import bq_publish as bq

EXPECTED = ["watchlist", "recall_metrics", "recall_audit", "agent_briefs", "agent_citations",
            "waymo_results", "dataset_facts", "recall_ablation", "recall_overlap", "recall_miss_reasons",
            "label_audit_misses", "agent_complaints", "agent_monthly", "agent_brief_stats"]


def _row(table):
    sample = {"STRING": "x", "INT64": 1, "FLOAT64": 0.5, "BOOL": True, "DATE": "2024-12-01",
              "TIMESTAMP": "2024-12-01T00:00:00+00:00", "JSON": [[0.0, 0.0]]}
    return {c.name: sample[c.type] for c in table.columns}


@pytest.fixture
def tiny_tables(monkeypatch):
    """Every table builds 3 schema-shaped rows instead of reading local data."""
    tables = {name: dataclasses.replace(t, build=lambda t=t: [_row(t)] * 3)
              for name, t in bq.TABLES.items()}
    monkeypatch.setattr(bq, "TABLES", tables)
    return tables


class FakeJob:
    def __init__(self, rows=()):
        self.rows = list(rows)

    def result(self):
        return self.rows


class FakeClient:
    def __init__(self):
        self.loaded, self.queries, self.datasets = {}, [], []

    def create_dataset(self, dataset, exists_ok=False):
        self.datasets.append(dataset.dataset_id)

    def load_table_from_json(self, rows, table_id, job_config=None):
        assert job_config.write_disposition == "WRITE_TRUNCATE"
        self.loaded[table_id.rsplit(".", 1)[1]] = (list(rows), job_config)
        return FakeJob()

    def query(self, sql):
        self.queries.append(sql)
        match = re.search(r"COUNT\(\*\) AS n FROM `[^`]+\.(\w+)`", sql)
        if match:
            return FakeJob([{"n": len(self.loaded[match.group(1)][0])}])
        if "v_watchlist_with_briefs" in sql and sql.lstrip().startswith("SELECT"):
            return FakeJob([{"n": 3, "with_brief": 3, "latest": "2024-12-01"}])
        return FakeJob([{"model_precision_at_20": 0.1}])


def test_dry_run_lists_exactly_the_final_tables(tiny_tables, monkeypatch, capsys):
    monkeypatch.setattr(bq, "client", lambda: pytest.fail("dry run must not connect"))

    assert bq.main([]) == 0

    out = capsys.readouterr().out
    assert [p.table.name for p in bq.plan()] == EXPECTED
    listed = [line.split()[0] for line in out.splitlines() if line.split() and line.split()[0] in bq.TABLES]
    assert listed == EXPECTED
    assert "Dry run: nothing uploaded" in out


@pytest.mark.parametrize("name", ["complaints", "recalls", "stg_complaints", "raw_recalls",
                                  "monthly_features", "waymo_tracks", "tracks", "investigations"])
def test_forbidden_tables_are_refused(name):
    with pytest.raises(bq.PublishRefused):
        bq.check_allowed(name)


def test_forbidden_table_never_loaded_even_if_added(tiny_tables, monkeypatch):
    rogue = dataclasses.replace(bq.TABLES["watchlist"], name="stg_complaints")
    monkeypatch.setitem(bq.TABLES, "stg_complaints", rogue)
    fake = FakeClient()

    with pytest.raises(bq.PublishRefused):
        bq.plan(["stg_complaints"])
    with pytest.raises(bq.PublishRefused):
        bq.load_table(fake, rogue, [])
    assert fake.loaded == {}


@pytest.mark.parametrize("rows, bytes_each, message", [
    (bq.MAX_TABLE_ROWS + 1, 10, "rows >"),
    (2, bq.MAX_TABLE_BYTES, "MB >"),
])
def test_size_guards_refuse_oversized_tables(rows, bytes_each, message):
    planned = bq.Planned(bq.TABLES["watchlist"], [{"x": "a" * bytes_each}] * rows)
    with pytest.raises(bq.PublishRefused, match=message):
        bq.check_guards([planned], bq.read_ledger(bq.ROOT / "does-not-exist.csv"))


def test_publish_total_and_lifetime_guards():
    row = [{"x": "a" * (bq.MAX_TABLE_BYTES // 2)}]  # each table under 25 MB
    many = [bq.Planned(bq.TABLES["watchlist"], row) for _ in range(9)]  # ~112 MB in total
    empty = pd.DataFrame(columns=bq.LEDGER_COLUMNS)
    with pytest.raises(bq.PublishRefused, match="this publish"):
        bq.check_guards(many, empty)

    full = pd.DataFrame([{"published_at": "t", "git_commit": "c", "table_name": "watchlist",
                          "rows": 1, "bytes": bq.MAX_LIFETIME_BYTES - 10, "cumulative_bytes": 0}])
    with pytest.raises(bq.PublishRefused, match="lifetime"):
        bq.check_guards([bq.Planned(bq.TABLES["watchlist"], [{"x": "abcdefghijk"}])], full)


def test_publish_loads_only_allowed_tables_and_updates_ledger(tiny_tables, tmp_path):
    ledger = tmp_path / "ledger.csv"
    fake = FakeClient()

    first = bq.publish(bq.plan(), fake, ledger_path=ledger)

    assert set(fake.loaded) == set(EXPECTED) | {"publish_log"}
    assert all(not bq.FORBIDDEN.search(name) for name in fake.loaded)
    assert list(first["table_name"]) == EXPECTED + ["publish_log"]
    assert first["cumulative_bytes"].is_monotonic_increasing
    assert first["cumulative_bytes"].iloc[-1] == first["bytes"].sum()
    # views are CREATE OR REPLACE; no DML anywhere
    assert sum("CREATE OR REPLACE VIEW" in q for q in fake.queries) == len(bq.VIEWS)
    assert not any(re.search(r"\b(INSERT|UPDATE|DELETE|MERGE)\b", q) for q in fake.queries)

    second = bq.publish(bq.plan(), FakeClient(), ledger_path=ledger)

    assert len(second) == 2 * len(first)
    assert bq.read_ledger(ledger)["cumulative_bytes"].iloc[-1] == second["bytes"].sum()


def test_schema_has_descriptions_and_rows_match_columns(tiny_tables):
    for table in list(bq.TABLES.values()) + [bq.PUBLISH_LOG]:
        assert table.description
        assert all(c.description for c in table.columns), table.name
    fake = FakeClient()
    bq.load_table(fake, bq.TABLES["agent_briefs"], [_row(bq.TABLES["agent_briefs"])])
    config = fake.loaded["agent_briefs"][1]
    assert config.destination_table_description == bq.TABLES["agent_briefs"].description
    assert [f.name for f in config.schema] == [c.name for c in bq.TABLES["agent_briefs"].columns]


def test_verify_compares_row_counts(tiny_tables):
    fake = FakeClient()
    planned = bq.plan()
    for p in planned:
        bq.load_table(fake, p.table, p.rows)

    lines = bq.verify(fake, planned)

    assert all("OK" in line for line in lines[: len(planned)])
    fake.loaded["watchlist"] = ([], None)
    with pytest.raises(RuntimeError, match="MISMATCH"):
        bq.verify(fake, planned)


def test_briefs_and_citations_from_validation_csv(tmp_path, monkeypatch):
    month_dir = tmp_path / "watchlist_2024-12"
    month_dir.mkdir()
    (month_dir / "validation.csv").write_text(
        "rank,make,model,model_year,component,month,model_used,source,self_correction,recall_search,"
        "citations_total,citations_valid,citations_invalid,evidence_bullets,bullets_without_citations,"
        "invalid_odinos,uncited_bullets,first_try_errors\n"
        "3,KIA,EV9,2024,ADAS AND STABILITY CONTROL,2024-12-01,gemini-x,generated,passed_after_retry,"
        "same_make,3,2,1,3,0,999,,\n", encoding="utf-8")
    (month_dir / "kia_ev9_2024_adas_and_stability_control.md").write_text(
        "## Evidence\n- a [ODINO 111], [ODINO 999]\n- b [ODINO 111]\n", encoding="utf-8")
    monkeypatch.setattr(bq, "BRIEFS_DIR", tmp_path)

    (brief,) = bq.build_agent_briefs()
    citations = bq.build_agent_citations()

    assert brief["brief_id"] == "2024-12-r03" and brief["passed_after_retry"] and not brief["passed_first_try"]
    assert list(brief) == [c.name for c in bq.TABLES["agent_briefs"].columns]
    assert citations == [{"brief_id": "2024-12-r03", "odino": "111", "valid": True},
                         {"brief_id": "2024-12-r03", "odino": "999", "valid": False}]


def test_reader_names_match_publisher_and_reject_unknown():
    from src import bq_reader

    assert set(bq_reader.TABLES) == set(bq.TABLES) | {"publish_log"}
    assert set(bq_reader.VIEWS) == set(bq.VIEWS)
    with pytest.raises(ValueError):
        bq_reader.read_table("stg_complaints")


def test_reader_explains_missing_credentials(monkeypatch):
    from google.auth.exceptions import DefaultCredentialsError
    from google.cloud import bigquery

    from src import bq_reader

    def no_creds(*args, **kwargs):
        raise DefaultCredentialsError("none")

    monkeypatch.setattr(bq_reader, "_client", None)
    monkeypatch.setattr(bigquery, "Client", no_creds)
    with pytest.raises(RuntimeError, match="gcloud auth application-default login"):
        bq_reader.read_table("watchlist")


def test_no_waymo_coordinates_are_published():
    assert "waymo_demo_examples" not in bq.TABLES
    for table in bq.TABLES.values():
        names = {c.name for c in table.columns}
        assert not names & {"past_path", "true_future", "baseline_path", "model_paths", "lane_polylines"}, table.name
