"""Build recall labels: one row per (make, model, model_year, component, month).

y_{h}m is true if the cell gets a recall within h months after the end of that month.
Complaints are dated by LDATE and recalls by RCDATE.

    python -m src.recall_model.labels --source raw
"""
import argparse

from src.recall_model.sources import (
    OUT,
    RAW_COMPLAINTS,
    RAW_RECALLS,
    add_source_arg,
    connect,
    get_sql,
)

KEYS = "make, model, model_year, component"
JOIN = "p.make = r.make and p.model = r.model and p.model_year = r.model_year and p.component = r.component"


def month_range(con, complaints_sql):
    # first and last full month of complaints in the input
    lo, hi = con.sql(f"""
        select date_trunc('month', min(received_date))::date,
               date_trunc('month', max(received_date) + interval 1 day)::date - interval 1 month
        from ({complaints_sql})
    """).fetchall()[0]
    return str(lo), str(hi)[:10]


def build_labels(con, complaints_sql=RAW_COMPLAINTS, recalls_sql=RAW_RECALLS,
                 horizons=(3, 6, 12), start=None, end=None):
    return con.sql(labels_sql(con, complaints_sql, recalls_sql, horizons, start, end) + f" order by {KEYS}, month")


def labels_sql(con, complaints_sql, recalls_sql, horizons=(3, 6, 12), start=None, end=None):
    if start is None or end is None:
        lo, hi = month_range(con, complaints_sql)
        start, end = start or lo, end or hi
    labels = ",\n".join(
        f"coalesce(next_recall_date <= as_of + interval {h} month, false) as y_{h}m" for h in horizons
    )
    return f"""
    with monthly as (
        select {KEYS}, date_trunc('month', received_date)::date as month, count(*) as n_complaints
        from ({complaints_sql})
        where received_date is not null and model_year is not null
        group by all
    ),
    cells as (
        select {KEYS}, min(month) as first_month from monthly group by all
    ),
    panel as (
        select cells.make, cells.model, cells.model_year, cells.component,
               m.month, last_day(m.month) as as_of
        from cells
        join (select unnest(generate_series(date '{start}', date '{end}', interval 1 month))::date as month) m
          on m.month >= cells.first_month
    ),
    recalls as (
        select {KEYS}, recall_date, string_agg(distinct campaign_id, ',' order by campaign_id) as campaign_ids
        from ({recalls_sql})
        where recall_date is not null and model_year is not null
        group by all
    ),
    p as (
        select panel.*, coalesce(monthly.n_complaints, 0) as n_complaints
        from panel left join monthly using ({KEYS}, month)
    ),
    p2 as (
        select p.*, r.recall_date as next_recall_date, r.campaign_ids as next_recall_campaign_ids
        from p asof left join recalls r on {JOIN} and r.recall_date > p.as_of
    )
    select p.*,
           r.recall_date as last_recall_date,
           coalesce(r.recall_date > p.as_of - interval 12 month, false) as recalled_prev_12m,
           {labels}
    from p2 p asof left join recalls r on {JOIN} and r.recall_date <= p.as_of
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_source_arg(parser)
    args = parser.parse_args()

    con = connect()
    out = OUT / f"recall_labels_{args.source}.parquet"
    build_labels(con, *get_sql(args.source)).write_parquet(str(out))

    n, cells, y3, y6, y12 = con.sql(f"""
        select count(*), count(distinct (make, model, model_year, component)),
               avg(y_3m::int), avg(y_6m::int), avg(y_12m::int)
        from read_parquet('{out.as_posix()}')
    """).fetchall()[0]
    print(f"wrote {out}: {n:,} cell-months, {cells:,} cells")
    print(f"positive rate  3m {y3:.2%}  6m {y6:.2%}  12m {y12:.2%}")
