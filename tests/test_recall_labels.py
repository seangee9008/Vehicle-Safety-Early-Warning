import duckdb

from src.recall_model.labels import build_labels

COMPLAINTS = """
select * from (values
    ('FORD', 'F-150', 2015, 'AIR BAGS', date '2016-01-10'),
    ('FORD', 'F-150', 2015, 'AIR BAGS', date '2016-03-05')
) t(make, model, model_year, component, received_date)
"""

RECALLS = """
select * from (values
    ('FORD', 'F-150', 2015, 'AIR BAGS', date '2016-05-20', '16V001000'),
    ('FORD', 'F-150', 2015, 'ENGINE', date '2016-02-01', '16V002000')
) t(make, model, model_year, component, recall_date, campaign_id)
"""


def get_rows():
    con = duckdb.connect()
    rel = build_labels(con, COMPLAINTS, RECALLS, horizons=(3, 6), start="2015-06-01", end="2016-08-01")
    rows = rel.select("month, n_complaints, y_3m, y_6m, recalled_prev_12m, next_recall_campaign_ids").fetchall()
    return {str(r[0]): r[1:] for r in rows}


def test_panel_starts_at_first_complaint():
    rows = get_rows()
    assert min(rows) == "2016-01-01"
    assert max(rows) == "2016-08-01"


def test_labels():
    rows = get_rows()
    assert rows["2016-01-01"][:3] == (1, False, True)
    assert rows["2016-02-01"][:3] == (0, True, True)
    assert rows["2016-05-01"][1:4] == (False, False, True)
    assert rows["2016-01-01"][4] == "16V001000"  # engine recall shouldn't count
