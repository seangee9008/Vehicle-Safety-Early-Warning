import pandas as pd

from src.recall_audit import investigations, label_audit
from tests.nhtsa_fixtures import make_db


def _recall(model, year, comp, date, camp="r"):
    return (camp, camp, "FORD", model, year, comp, date, "Defect.", "Crash risk.")


def test_label_audit_flags_each_looser_match(tmp_path):
    db = make_db(tmp_path / "nhtsa.duckdb", stg_recalls=[
        _recall("F150", 2020, "BRAKES", "2022-03-01"),          # exact for F150 2020 BRAKES
        _recall("F150", 2021, "BRAKES", "2022-03-01"),          # variant for F150HYBRID 2021
        _recall("RANGER", 2022, "STEERING", "2022-04-01"),      # adj_year for RANGER 2021
        _recall("BRONCO", 2021, "LIGHTING", "2022-10-01"),      # late (9 months out)
        _recall("EDGE", 2020, "AIR BAGS", "2022-02-01"),        # other_part for EDGE 2020 BRAKES
    ])
    rows = [("F150", 2020, "BRAKES", True), ("F150HYBRID", 2021, "BRAKES", False),
            ("RANGER", 2021, "STEERING", False), ("BRONCO", 2021, "LIGHTING", False),
            ("EDGE", 2020, "BRAKES", False)]
    preds = pd.DataFrame([{"make": "FORD", "model": m, "model_year": y, "component": c,
                           "month": pd.Timestamp("2022-01-01"), "split": "test", "y": lab,
                           "baseline": 1.0, "score": 0.5} for m, y, c, lab in rows])
    path = tmp_path / "predictions.parquet"
    preds.to_parquet(path)

    flags = label_audit.load_flags(path, db).set_index("model")

    assert flags.loc["F150", "exact"] and not flags.loc["F150", "variant"]
    assert flags.loc["F150", "adj_year"]  # the 2021 F150 recall is one model year away
    assert flags.loc["F150HYBRID", "variant"]
    assert flags.loc["RANGER", "adj_year"]
    assert flags.loc["BRONCO", "late"]
    assert flags.loc["EDGE", "other_part"] and not flags.loc["EDGE", "variant"]


def test_investigation_is_open_only_between_open_and_close():
    inv = pd.DataFrame({"make": ["FORD"], "odate": [pd.Timestamp("2022-02-10")],
                        "cdate": [pd.Timestamp("2022-04-01")]})
    months = pd.date_range("2022-01-01", "2022-05-01", freq="MS")

    open_months = investigations.monthly_open(inv, ["make"], months)["month"]

    # opened Feb 10 (open at Feb 28), closed Apr 1 (closed by Apr 30)
    assert sorted(pd.to_datetime(open_months).dt.strftime("%Y-%m")) == ["2022-02", "2022-03"]
