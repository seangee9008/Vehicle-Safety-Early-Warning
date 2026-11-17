import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import sources
from src.agent import data_access
from src.agent.brief import create_brief
from src.agent.run import WATCHLIST_COLUMNS
from src.agent.validate import extract_citations, validate_brief
from tests.nhtsa_fixtures import make_db, raw_complaint, raw_recall, stg_complaint

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def use_db(monkeypatch):
    def _use(path):
        monkeypatch.setattr(sources, "DB_PATH", path)
        data_access._source_sql.cache_clear()
        return path

    yield _use
    data_access._source_sql.cache_clear()


def test_get_complaints_placeholder_excludes_cutoff_and_later(tmp_path, use_db):
    use_db(make_db(tmp_path / "nhtsa.duckdb", complaints=[
        raw_complaint("12345678", "20180831", crash="Y", injured="1"),
        raw_complaint("23456789", "20180901"),
        raw_complaint("34567890", "20180902"),
        raw_complaint("45678901", "20180815", cmpl_type="RC"),  # dropped like stg does
    ]))

    rows = data_access.get_complaints("ford", "fusion", 2010, "AIR BAGS", "2018-09-01")

    assert [row["odi_no"] for row in rows] == ["12345678"]
    assert rows[0]["ldate"] == "2018-08-31"
    assert rows[0]["is_crash"] == 1
    assert rows[0]["component"] == "AIR BAGS"
    assert tuple(rows[0]) == data_access.COMPLAINT_COLUMNS


def test_get_complaints_reads_real_stg_table_with_normalized_model(tmp_path, use_db):
    use_db(make_db(tmp_path / "nhtsa.duckdb", stg_complaints=[
        stg_complaint("11111111", "2019-03-30", crash=1),
        stg_complaint("22222222", "2019-04-01"),
        stg_complaint("33333333", "2019-03-01", comp="ENGINE AND ENGINE COOLING"),
    ], stg_recalls=[]))

    # The sample watchlist spells it F-150 and keeps the subcomponent; stg has F150 / POWER TRAIN.
    rows = data_access.get_complaints(
        "FORD", "F-150", 2013, "POWER TRAIN:AUTOMATIC TRANSMISSION", "2019-04-01"
    )

    assert sources.get_complaints().is_real
    assert [row["odi_no"] for row in rows] == ["11111111"]
    assert tuple(rows[0]) == data_access.COMPLAINT_COLUMNS
    assert data_access.severity_counts(rows)["crashes"] == 1


def test_component_prefix_matches_standardized_name(tmp_path, use_db):
    use_db(make_db(tmp_path / "nhtsa.duckdb", stg_complaints=[
        stg_complaint("1", "2018-09-01", model="SONATA", model_raw="SONATA", year=2011,
                      make="HYUNDAI", comp="ENGINE AND ENGINE COOLING", comp_raw="ENGINE"),
    ], stg_recalls=[]))

    rows = data_access.get_complaints("HYUNDAI", "SONATA", 2011, "ENGINE", "2018-10-01")

    assert [row["odi_no"] for row in rows] == ["1"]


def test_evidence_cutoff_is_end_of_watchlist_month():
    assert data_access.evidence_cutoff("2024-12-01") == "2025-01-01"
    assert data_access.evidence_cutoff("2019-04-01") == "2019-05-01"


def test_similar_recalls_are_strictly_before_cutoff(tmp_path, use_db):
    use_db(make_db(tmp_path / "nhtsa.duckdb", recalls=[
        raw_recall("18V000001", "20180831"),
        raw_recall("18V000002", "20180901", defect="On cutoff date."),
    ]))

    recalls = data_access.find_similar_recalls("AIR BAGS", ["inflator"], "2018-09-01")

    assert [recall["camp_no"] for recall in recalls] == ["18V000001"]


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


GROUPED_COMPLAINTS = [
    {"odino": "11111111", "received_date": "2018-08-31"},
    {"odino": "22222222", "received_date": "2018-08-31"},
]


@pytest.mark.parametrize(
    "citation",
    [
        "[ODINO 11111111, 22222222]",
        "[ODINO 11111111, ODINO 22222222]",
        "[ODINO 11111111; 22222222]",
        "[ODINO 11111111][ODINO 22222222]",
        "[ODINO 11111111], [ODINO 22222222]",
        "[odino 11111111 , 22222222]",
    ],
)
def test_validator_accepts_grouped_citations(citation):
    brief = f"## Evidence\n- Two complaints describe the same failure {citation}.\n"

    result = validate_brief(brief, GROUPED_COMPLAINTS, "2018-09-01")

    assert extract_citations(brief) == ["11111111", "22222222"]
    assert result["citations_total"] == 2
    assert result["citations_valid"] == 2
    assert result["citations_invalid"] == 0
    assert result["bullets_without_citations"] == 0


def test_validator_flags_fake_id_inside_group():
    brief = "## Evidence\n- Grouped [ODINO 11111111, 99999999].\n"

    result = validate_brief(brief, GROUPED_COMPLAINTS, "2018-09-01")

    assert result["citations_total"] == 2
    assert result["citations_valid"] == 1
    assert result["invalid_odinos"] == "99999999"


def test_validator_rejects_malformed_group():
    brief = "## Evidence\n- Not a citation [ODINO 11111111, see above].\n"

    result = validate_brief(brief, GROUPED_COMPLAINTS, "2018-09-01")

    assert result["citations_total"] == 0
    assert result["bullets_without_citations"] == 1


def test_watchlist_fixture_matches_contract_3_columns():
    path = ROOT / "src" / "agent" / "fixtures" / "watchlist_sample.csv"
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        assert tuple(reader.fieldnames or ()) == WATCHLIST_COLUMNS
        rows = list(reader)
    assert len(rows) == 5


def test_gemini_function_calling_is_mocked(monkeypatch):
    complaint = {
        "odi_no": "12345678",
        "ldate": "2018-08-31",
        "is_crash": 1,
        "is_fire": 0,
        "injured_count": 1,
        "deaths_count": 0,
    }
    requested_scopes = []

    def fake_get_complaints(*args, **kwargs):
        requested_scopes.append((args, kwargs))
        return [complaint]

    def fake_severity_counts(rows):
        assert rows == [complaint]
        return {"crashes": 1, "fires": 0, "injuries": 1, "deaths": 0}

    monkeypatch.setattr(data_access, "get_complaints", fake_get_complaints)
    monkeypatch.setattr(data_access, "eligible_complaints", lambda *a, **k: [complaint])
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
    # month 2018-09 -> evidence through the end of September
    assert requested_scopes[0][0][:5] == (
        "FORD",
        "FUSION",
        2010,
        "AIR BAGS",
        "2018-10-01",
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


def _api_error(code):
    from google.genai.errors import ClientError, ServerError

    cls = ClientError if code < 500 else ServerError
    return cls(code, {"message": f"HTTP {code}"})


@pytest.mark.parametrize("code", [429, 503])
def test_falls_back_to_next_flash_model_on_429_or_503(monkeypatch, code):
    from src.agent import brief as agent_brief

    monkeypatch.setattr(agent_brief.time, "sleep", lambda _: None)
    monkeypatch.delenv("GEMINI_FALLBACK_MODELS", raising=False)
    monkeypatch.setattr(data_access, "get_complaints", lambda *a, **k: [])
    monkeypatch.setattr(data_access, "eligible_complaints", lambda *a, **k: [])
    monkeypatch.setattr(data_access, "severity_counts", lambda rows: {})
    text = "## Evidence\nnone\n"
    calls = []

    class FakeModels:
        def list(self):
            return [
                SimpleNamespace(name="models/gemini-3.6-flash", supported_actions=["generateContent"]),
                SimpleNamespace(name="models/gemini-3.6-flash-lite", supported_actions=["generateContent"]),
                SimpleNamespace(name="models/gemini-3.6-pro", supported_actions=["generateContent"]),
                SimpleNamespace(name="models/gemini-3.6-flash-image", supported_actions=["generateContent"]),
                SimpleNamespace(name="models/gemini-3.5-flash", supported_actions=["generateContent"]),
            ]

        def generate_content(self, **kwargs):
            calls.append(kwargs["model"])
            if kwargs["model"] == "gemini-3.6-flash":
                raise _api_error(code)
            part = SimpleNamespace(function_call=None, text=text)
            return SimpleNamespace(
                candidates=[SimpleNamespace(content=SimpleNamespace(parts=[part]))], text=text
            )

    _, validation = create_brief(
        {"month": "2018-09-01", "make": "FORD", "model": "FUSION",
         "model_year": "2010", "component": "AIR BAGS"},
        client=SimpleNamespace(models=FakeModels()),
        model_name="gemini-3.6-flash",
    )

    retries = agent_brief.MAX_RETRIES_PER_MODEL + 1
    assert calls == ["gemini-3.6-flash"] * retries + ["gemini-3.6-flash-lite"]
    assert validation["model_used"] == "gemini-3.6-flash-lite"


def test_fallback_order_prefers_newer_flash_over_lite():
    from src.agent.brief import fallback_models

    models = SimpleNamespace(
        list=lambda: [
            SimpleNamespace(name=f"models/{name}", supported_actions=["generateContent"])
            for name in (
                "gemini-3.5-flash", "gemini-3.6-flash-lite", "gemini-3.6-flash",
                "gemini-3.6-flash-tts", "gemini-3.6-pro", "text-embedding-004",
                "gemini-omni-flash-preview", "gemini-3-flash-preview",
            )
        ]
    )

    assert fallback_models(SimpleNamespace(models=models)) == [
        "gemini-3.6-flash", "gemini-3.6-flash-lite", "gemini-3.5-flash",
        "gemini-3-flash-preview", "gemini-omni-flash-preview",
    ]


def test_non_retryable_error_does_not_fall_back(monkeypatch):
    from src.agent import brief as agent_brief

    class FakeModels:
        def generate_content(self, **kwargs):
            raise _api_error(400)

    with pytest.raises(agent_brief.APIError):
        agent_brief._generate_with_backoff(
            SimpleNamespace(models=FakeModels()), model="m", contents=[], config=None
        )


def test_run_skips_existing_brief_unless_forced(tmp_path, monkeypatch):
    from src.agent import run as agent_run

    monkeypatch.setattr(agent_run, "BRIEFS_DIR", tmp_path)
    monkeypatch.setattr(
        agent_run.data_access,
        "eligible_complaints",
        lambda *a, **k: [{"odi_no": "11111111", "ldate": "2018-08-31"}],
    )
    brief = "## Evidence\n- a [ODINO 11111111]\n- b [ODINO 11111111]\n- c [ODINO 11111111]\n"
    generated = []

    def fake_create_brief(item):
        generated.append(item["make"])
        return brief, {**validate_brief(brief, [{"odino": "11111111",
                                                "received_date": "2018-08-31"}],
                                        "2018-09-01"), "model_used": "mock-flash"}

    monkeypatch.setattr(agent_run, "create_brief", fake_create_brief)
    watchlist = ROOT / "src" / "agent" / "fixtures" / "watchlist_sample.csv"

    agent_run.run(watchlist, 1)
    agent_run.run(watchlist, 1)
    assert generated == ["FORD"]
    with (tmp_path / "validation.csv").open(newline="") as file:
        row = next(csv.DictReader(file))
    assert row["source"] == "cached"
    assert row["model_used"] == "mock-flash"

    agent_run.run(watchlist, 1, force=True)
    assert generated == ["FORD", "FORD"]
