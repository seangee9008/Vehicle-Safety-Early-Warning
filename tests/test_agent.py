import csv
from pathlib import Path
from types import SimpleNamespace

import duckdb

from src.agent import data_access
from src.agent.brief import create_brief
from src.agent.run import WATCHLIST_COLUMNS
from src.agent.validate import validate_brief

ROOT = Path(__file__).resolve().parents[1]


def test_get_complaints_excludes_watchlist_month_and_later(tmp_path, monkeypatch):
    database = tmp_path / "nhtsa.duckdb"
    connection = duckdb.connect(str(database))
    connection.execute("CREATE SCHEMA raw")
    connection.execute(
        """
        CREATE TABLE raw.complaints (
            CMPLID VARCHAR, ODINO VARCHAR, MAKETXT VARCHAR, MODELTXT VARCHAR,
            YEARTXT VARCHAR, COMPDESC VARCHAR, LDATE VARCHAR, CDESCR VARCHAR,
            CRASH VARCHAR, FIRE VARCHAR, INJURED VARCHAR, DEATHS VARCHAR
        )
        """
    )
    connection.executemany(
        "INSERT INTO raw.complaints VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("1", "12345678", "Ford", "Fusion", "2010", "Air Bags", "20180831",
             "Airbag warning light.", "Y", "N", "1", "0"),
            ("2", "23456789", "Ford", "Fusion", "2010", "Air Bags", "20180901",
             "Post-month complaint.", "N", "Y", "0", "0"),
            ("3", "34567890", "Ford", "Fusion", "2010", "Air Bags", "20180902",
             "Later complaint.", "N", "N", "0", "0"),
        ],
    )
    connection.close()
    monkeypatch.setattr(data_access, "DB_PATH", database)
    monkeypatch.setenv("AGENT_SOURCE", "raw")

    rows = data_access.get_complaints(
        "ford", "fusion", 2010, "AIR BAGS", "2018-09-01"
    )

    assert [row["odino"] for row in rows] == ["12345678"]
    assert rows[0]["received_date"] == "2018-08-31"
    assert rows[0]["crash"] is True
    assert tuple(rows[0]) == data_access.CONTRACT_1_COLUMNS


def test_stg_source_reads_contract_1_columns(tmp_path, monkeypatch):
    database = tmp_path / "nhtsa.duckdb"
    connection = duckdb.connect(str(database))
    connection.execute(
        """
        CREATE TABLE main.stg_complaints (
            complaint_id VARCHAR, odino VARCHAR, make VARCHAR, model VARCHAR,
            model_year INTEGER, component VARCHAR, received_date DATE, description VARCHAR,
            crash BOOLEAN, fire BOOLEAN, injuries INTEGER, deaths INTEGER
        )
        """
    )
    connection.execute(
        """
        INSERT INTO main.stg_complaints VALUES
        ('1', '12345678', 'FORD', 'FUSION', 2010, 'AIR BAGS', DATE '2018-08-31',
         'Airbag concern.', TRUE, FALSE, 1, 0)
        """
    )
    connection.close()
    monkeypatch.setattr(data_access, "DB_PATH", database)
    monkeypatch.setenv("AGENT_SOURCE", "stg")

    rows = data_access.get_complaints(
        "FORD", "FUSION", 2010, "AIR BAGS", "2018-09-01"
    )

    assert [row["odino"] for row in rows] == ["12345678"]
    assert tuple(rows[0]) == data_access.CONTRACT_1_COLUMNS


def test_similar_recalls_are_strictly_before_cutoff(tmp_path, monkeypatch):
    database = tmp_path / "nhtsa.duckdb"
    connection = duckdb.connect(str(database))
    connection.execute("CREATE SCHEMA raw")
    connection.execute(
        """
        CREATE TABLE raw.recalls (
            CAMPNO VARCHAR, MAKETXT VARCHAR, MODELTXT VARCHAR, YEARTXT VARCHAR,
            COMPNAME VARCHAR, RCDATE VARCHAR, DESC_DEFECT VARCHAR,
            CONSEQUENCE_DEFECT VARCHAR
        )
        """
    )
    connection.executemany(
        "INSERT INTO raw.recalls VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            ("18V000001", "FORD", "FUSION", "2010", "AIR BAGS", "20180831",
             "Inflator defect.", "May not deploy."),
            ("18V000002", "FORD", "FUSION", "2010", "AIR BAGS", "20180901",
             "On cutoff date.", "May not deploy."),
        ],
    )
    connection.close()
    monkeypatch.setattr(data_access, "DB_PATH", database)

    recalls = data_access.find_similar_recalls(
        "AIR BAGS", ["inflator"], "2018-09-01"
    )

    assert [recall["campaign_id"] for recall in recalls] == ["18V000001"]


def test_validator_rejects_fake_id_and_uncited_bullet():
    complaints = [
        {"odino": "12345678", "received_date": "2018-08-31"},
        {"odino": "87654321", "received_date": "2018-09-01"},
    ]
    brief = """## Evidence
- The documented event involved a crash [ODINO 12345678].
- Fake citation [ODINO 99999999].
- No source is cited for this bullet.
"""

    result = validate_brief(brief, complaints, "2018-09-01")

    assert result["citations_total"] == 2
    assert result["citations_valid"] == 1
    assert result["citations_invalid"] == 1
    assert result["bullets_without_citations"] == 1
    assert result["invalid_odinos"] == "99999999"
    assert result["evidence_bullets"] == 3


def test_watchlist_fixture_matches_contract_3_columns():
    path = ROOT / "src" / "agent" / "fixtures" / "watchlist_sample.csv"
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        assert tuple(reader.fieldnames or ()) == WATCHLIST_COLUMNS
        rows = list(reader)
    assert len(rows) == 5


def test_gemini_function_calling_is_mocked(monkeypatch):
    complaint = {
        "odino": "12345678",
        "received_date": "2018-08-31",
        "crash": True,
        "fire": False,
        "injuries": 1,
        "deaths": 0,
    }
    requested_scopes = []

    def fake_get_complaints(*args, **kwargs):
        requested_scopes.append((args, kwargs))
        return [complaint]

    def fake_severity_counts(rows):
        assert rows == [complaint]
        return {"crashes": 1, "fires": 0, "injuries": 1, "deaths": 0}

    monkeypatch.setattr(data_access, "get_complaints", fake_get_complaints)
    monkeypatch.setattr(data_access, "severity_counts", fake_severity_counts)
    function_part = SimpleNamespace(
        function_call=SimpleNamespace(
            name="get_complaints",
            args={
                "make": "OTHER",
                "model": "OTHER",
                "model_year": 2010,
                "component": "OTHER",
                "before_date": "2099-01-01",
            },
        )
    )
    final_text = """## Failure pattern
An air bag complaint described a warning light. One complaint is available.
## Severity
1 crash, 0 fires, 1 injury, 0 deaths.
## Evidence
- Airbag warning light [ODINO 12345678].
## Similar past recalls
none found
## Suggested next step
Review the complaint details.
"""
    responses = [
        SimpleNamespace(
            candidates=[SimpleNamespace(content=SimpleNamespace(parts=[function_part]))],
            text=None,
        ),
        SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(
                        parts=[SimpleNamespace(function_call=None, text=final_text)]
                    )
                )
            ],
            text=final_text,
        ),
    ]

    class FakeModels:
        def generate_content(self, **kwargs):
            assert kwargs["model"] == "mock-flash"
            return responses.pop(0)

    client = SimpleNamespace(models=FakeModels())
    brief, validation = create_brief(
        {
            "month": "2018-09-01",
            "make": "FORD",
            "model": "FUSION",
            "model_year": "2010",
            "component": "AIR BAGS",
            "score": "0.98",
            "rank": "1",
            "top_reasons": "complaints up 3x in 3 months",
        },
        client=client,
        model_name="mock-flash",
    )

    assert "ODINO 12345678" in brief
    assert validation["citations_valid"] == 1
    assert validation["citations_invalid"] == 0
    assert requested_scopes[0][0][:5] == (
        "FORD",
        "FUSION",
        2010,
        "AIR BAGS",
        "2018-09-01",
    )


def test_rate_limit_retries_with_backoff(monkeypatch):
    from google.genai.errors import ClientError

    from src.agent import brief as agent_brief

    class FakeModels:
        calls = 0

        def generate_content(self, **kwargs):
            assert kwargs["model"] == "mock"
            self.calls += 1
            if self.calls == 1:
                raise ClientError(429, {"message": "rate limited"})
            return "ok"

    models = FakeModels()
    sleeps = []
    monkeypatch.setattr(agent_brief.time, "sleep", sleeps.append)
    monkeypatch.setattr(agent_brief.random, "random", lambda: 0.0)

    result = agent_brief._generate_with_backoff(
        SimpleNamespace(models=models),
        model="mock",
        contents=[],
        config=None,
    )

    assert result == "ok"
    assert models.calls == 2
    assert sleeps == [1]


def test_description_redacts_common_identifiers():
    description = "Call me at 555-123-4567, email a@example.com, VIN 1HGCM82633A004352."
    redacted = data_access._redact_description(description)
    assert "555-123-4567" not in redacted
    assert "a@example.com" not in redacted
    assert "1HGCM82633A004352" not in redacted
