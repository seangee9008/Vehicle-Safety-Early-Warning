import duckdb
import pytest

from src import sources
from src.contracts import check_contract, load_contracts
from tests.nhtsa_fixtures import make_db, raw_complaint, raw_recall, stg_complaint


@pytest.fixture
def db(tmp_path, monkeypatch):
    def _make(**tables):
        path = make_db(tmp_path / "nhtsa.duckdb", **tables)
        monkeypatch.setattr(sources, "DB_PATH", path)
        return path

    return _make


def _add_justine_tables(path, *, drop_column=None):
    features = ("make VARCHAR, model VARCHAR, model_year INTEGER, component VARCHAR, month DATE, "
                "cmpl_1m DOUBLE, cmpl_3m DOUBLE, cmpl_6m DOUBLE, severe_3m DOUBLE, "
                "vehicle_age BIGINT, label BOOLEAN, cmpl_12m DOUBLE")
    watchlist = ("month DATE, make VARCHAR, model VARCHAR, model_year INTEGER, component VARCHAR, "
                 "score DOUBLE, rank BIGINT, top_reasons VARCHAR, source VARCHAR")
    if drop_column:
        watchlist = ", ".join(c for c in watchlist.split(", ") if not c.startswith(drop_column + " "))
    with duckdb.connect(str(path)) as con:
        con.execute(f"CREATE TABLE main.monthly_features ({features})")
        con.execute("INSERT INTO main.monthly_features VALUES ('FORD', 'F150', 2013, 'POWER TRAIN', "
                    "DATE '2024-12-01', 1, 3, 5, 0, 11, false, 9)")
        con.execute(f"CREATE TABLE main.watchlist ({watchlist})")
        values = {"month": "DATE '2024-12-01'", "make": "'FORD'", "model": "'F150'",
                  "model_year": "2013", "component": "'POWER TRAIN'", "score": "0.9",
                  "rank": "1", "top_reasons": "'3 complaints in the last 3 months'",
                  "source": "'stg'"}
        cols = [c.split(" ")[0] for c in watchlist.split(", ")]
        con.execute(f"INSERT INTO main.watchlist VALUES ({', '.join(values[c] for c in cols)})")


def test_real_tables_pass_their_contracts(db):
    path = db(stg_complaints=[stg_complaint("1", "2024-12-15")],
              stg_recalls=[("r1", "24V001", "FORD", "F150", 2013, "POWER TRAIN",
                            "2024-06-01", "Defect.", "Crash risk.")])
    _add_justine_tables(path)

    with sources.connect() as con:
        results = {name: check_contract(con, name) for name in load_contracts()}

    assert all(result.ok for result in results.values()), [r.describe() for r in results.values()]
    assert all(row["kind"] == "real" for row in sources.status())


def test_placeholders_pass_their_contracts(db):
    db(complaints=[raw_complaint("1", "20180831", comp="ENGINE AND ENGINE COOLING:OIL")],
       recalls=[raw_recall("18V000001", "20180801")])

    status = {row["table"]: row for row in sources.status()}
    with sources.connect() as con:
        for name in load_contracts():
            source = sources.resolve(name, con)
            assert source.kind == "placeholder"
            assert check_contract(con, name, relation_sql=source.sql).ok, name

    assert status["watchlist"]["rows"] == 5  # the sample CSV
    assert status["complaints"]["rows"] == 1
    assert status["monthly_features"]["kind"] == "placeholder"


def test_missing_column_fails_contract_and_falls_back(db):
    path = db(complaints=[raw_complaint("1", "20180831")])
    _add_justine_tables(path, drop_column="top_reasons")

    with sources.connect() as con:
        result = check_contract(con, "watchlist")
        source = sources.get_watchlist(con)

    assert not result.ok
    assert result.missing == ["top_reasons"]
    assert source.kind == "placeholder"
    assert "missing top_reasons" in source.problem


def test_wrong_type_fails_contract(db):
    path = db()
    with duckdb.connect(str(path)) as con:
        con.execute("CREATE SCHEMA stg")
        con.execute("CREATE TABLE stg.recalls (camp_no VARCHAR, make VARCHAR, model VARCHAR, "
                    "model_year INTEGER, component VARCHAR, recall_date VARCHAR, "
                    "defect_summary VARCHAR, consequence_summary VARCHAR)")
        con.execute("INSERT INTO stg.recalls VALUES "
                    "('1', 'FORD', 'F150', 2013, 'X', '2024-01-01', 'd', 'c')")

    with sources.connect() as con:
        result = check_contract(con, "recalls")

    assert result.wrong_type == {"recall_date": "VARCHAR, expected date"}


def test_contract_columns_cover_what_consumers_read():
    from src.agent.data_access import COMPLAINT_COLUMNS
    from src.recall_model.sources import COMPLAINT_COLUMNS as MODEL_COMPLAINTS
    from src.recall_model.sources import RECALL_COLUMNS as MODEL_RECALLS

    contracts = load_contracts()
    assert set(MODEL_COMPLAINTS) <= set(contracts["complaints"]["columns"])
    assert set(MODEL_RECALLS) <= set(contracts["recalls"]["columns"])
    assert set(COMPLAINT_COLUMNS) == set(contracts["complaints"]["columns"])
