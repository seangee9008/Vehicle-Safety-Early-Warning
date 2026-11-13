import duckdb
import pytest

from src.recall_model.features import CONTRACT, FEATURES, build_features

# odino 1 lists two parts; odino 2 shows up twice under AIR BAGS; odino 4 comes after the recall
COMPLAINTS = """
select * from (values
    ('c1', '1', 'FORD', 'F-150', 2015, 'AIR BAGS', date '2016-01-10', true, false, 0, 0, 1000, ''),
    ('c1', '1', 'FORD', 'F-150', 2015, 'ENGINE', date '2016-01-10', true, false, 0, 0, 1000, ''),
    ('c2', '2', 'FORD', 'F-150', 2015, 'AIR BAGS', date '2016-01-20', false, true, 1, 0, null, ''),
    ('c2', '2', 'FORD', 'F-150', 2015, 'AIR BAGS', date '2016-01-20', false, true, 1, 0, null, ''),
    ('c3', '3', 'FORD', 'F-150', 2015, 'AIR BAGS', date '2016-03-05', false, false, 0, 0, 3000, ''),
    ('c4', '4', 'FORD', 'F-150', 2015, 'AIR BAGS', date '2016-06-01', false, false, 0, 0, 5000, '')
) t(complaint_id, odino, make, model, model_year, component, received_date,
    is_crash, is_fire, injured, deaths, miles, narrative)
"""

RECALLS = """
select * from (values
    ('16V001000', 'FORD', 'F-150', 2015, 'AIR BAGS', date '2016-05-20', 'air bag may not deploy')
) t(campaign_id, make, model, model_year, component, recall_date, defect_summary)
"""


@pytest.fixture(scope="module")
def rows():
    con = duckdb.connect()
    df = build_features(con, COMPLAINTS, RECALLS, horizons=(3, 6), start="2015-06-01", end="2016-08-01").df()
    df["month"] = df["month"].astype(str)
    return {(r.component, r.month): r for r in df.itertuples()}


def test_has_contract_columns(rows):
    row = rows[("AIR BAGS", "2016-01-01")]
    for col in CONTRACT + FEATURES:
        assert hasattr(row, col)


def test_distinct_complaint_counts(rows):
    jan = rows[("AIR BAGS", "2016-01-01")]
    assert (jan.cmpl_1m, jan.cmpl_3m) == (2, 2)
    assert (jan.crash_3m, jan.fire_3m, jan.injured_3m, jan.deaths_3m, jan.severe_3m) == (1, 1, 1, 0, 2)
    assert rows[("ENGINE", "2016-01-01")].cmpl_1m == 1


def test_only_complaints_received_by_cutoff(rows):
    assert rows[("AIR BAGS", "2016-02-01")].cmpl_3m == 2
    mar = rows[("AIR BAGS", "2016-03-01")]
    assert (mar.cmpl_1m, mar.cmpl_3m, mar.cmpl_6m) == (1, 3, 3)
    assert rows[("AIR BAGS", "2016-04-01")].cmpl_3m == 1


def test_post_recall_complaints_dropped(rows):
    assert rows[("AIR BAGS", "2016-06-01")].cmpl_1m == 0
    assert rows[("AIR BAGS", "2016-06-01")].cmpl_12m == 3


def test_share_age_miles_label(rows):
    assert rows[("AIR BAGS", "2016-01-01")].share_of_car == 1.0
    assert rows[("ENGINE", "2016-01-01")].share_of_car == 0.5
    assert rows[("AIR BAGS", "2016-03-01")].median_miles_12m == 2000
    assert rows[("AIR BAGS", "2016-01-01")].vehicle_age == 1
    assert rows[("AIR BAGS", "2016-01-01")].label
    assert not rows[("ENGINE", "2016-01-01")].label
