import pandas as pd

from src.recall_model.model import KEYS, known_by, warning_weeks
from src.recall_model.watchlist import COLUMNS, EXTRA, build_watchlist

CELL = {"make": "FORD", "model": "F-150", "model_year": 2015, "component": "AIR BAGS"}


def test_known_by_keeps_only_closed_labels():
    df = pd.DataFrame({"month": pd.to_datetime(["2019-05-01", "2019-06-01", "2019-07-01"])})
    kept = known_by(df, pd.Timestamp("2019-12-01"), 6)
    assert list(kept["month"].dt.month) == [5, 6]


def test_warning_weeks_uses_first_flag_before_recall():
    flagged = pd.DataFrame([{**CELL, "month": pd.Timestamp(m)}
                            for m in ["2018-01-01", "2019-02-01", "2019-03-01", "2019-06-01"]])
    events = pd.DataFrame([{**CELL, "recall_date": pd.Timestamp("2019-05-31"), "had_complaints": True},
                           {**CELL, "component": "ENGINE", "recall_date": pd.Timestamp("2019-05-31"),
                            "had_complaints": False}])
    out = warning_weeks(flagged, events).set_index("component")
    # 2018-01 is more than 12 months early and 2019-06 is after the recall
    assert out.loc["AIR BAGS", "first_flag"] == pd.Timestamp("2019-02-28")
    assert round(out.loc["AIR BAGS", "weeks"], 1) == 13.1
    assert pd.isna(out.loc["ENGINE", "weeks"])


def test_watchlist_columns():
    rows = []
    for i in range(25):
        rows.append({**CELL, "model_year": 2000 + i, "month": pd.Timestamp("2022-03-01"),
                     "score": i / 25, "baseline": i, "cmpl_1m": 2.0, "cmpl_3m": 9.0, "cmpl_12m": 18.0,
                     "share_of_car": 0.5, "crash_3m": 1.0, "fire_3m": 0.0,
                     "injured_3m": 0.0, "deaths_3m": 0.0})
    wl = build_watchlist(pd.DataFrame(rows), "raw")
    assert COLUMNS == ["month", "make", "model", "model_year", "component", "score", "rank", "top_reasons"]
    assert list(wl.columns) == COLUMNS + EXTRA
    assert len(wl) == 20 and list(wl["rank"])[:2] == [1, 2]
    assert wl.loc[0, "model_year"] == 2024
    assert wl.loc[0, "reason_1"] == "complaints up 3.0x vs. the prior 9 months"
    assert wl.loc[0, "reason_2"] == "1 crash in the last 3 months"
    assert set(KEYS) <= set(wl.columns)
