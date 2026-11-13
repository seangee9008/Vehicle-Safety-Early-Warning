"""Build monthly_features: one row per (make, model, model_year, component, month) with labels.

A month's row only uses complaints received (LDATE) by the end of that month. Counts are
distinct odino. Complaints received on or after a cell's first recall are dropped, so they
never count toward its next flag. label = recalled within 6 months (y_6m).

    python -m src.recall_model.features --source raw
"""
import argparse

import duckdb

from src.recall_model.labels import KEYS, labels_sql
from src.recall_model.sources import DB, OUT, add_source_arg, connect, get_sql

# contract columns from AGENTS.md first, then extras
CONTRACT = ["make", "model", "model_year", "component", "month",
            "cmpl_1m", "cmpl_3m", "cmpl_6m", "severe_3m", "vehicle_age", "label"]
FEATURES = ["cmpl_1m", "cmpl_3m", "cmpl_6m", "cmpl_12m", "growth_3m", "share_of_car",
            "severe_3m", "crash_3m", "fire_3m", "injured_3m", "deaths_3m",
            "vehicle_age", "median_miles_12m"]
CAR = "make, model, model_year"


def window(col, months):
    return f"sum({col}) over (partition by {KEYS} order by month rows between {months - 1} preceding and current row)"


def build_features(con, complaints_sql, recalls_sql, horizons=(3, 6, 12), start=None, end=None):
    panel = labels_sql(con, complaints_sql, recalls_sql, horizons, start, end)
    ys = ", ".join(f"y_{h}m" for h in horizons)
    return con.sql(f"""
    with panel_v as ({panel}),
    c as (
        select {KEYS}, odino, received_date,
               date_trunc('month', received_date)::date as month,
               max(coalesce(is_crash::int, 0)) = 1 as crash,
               max(coalesce(is_fire::int, 0)) = 1 as fire,
               max(coalesce(injured, 0)) as injured,
               max(coalesce(deaths, 0)) as deaths,
               max(miles) as miles
        from ({complaints_sql})
        where received_date is not null and model_year is not null
        group by {KEYS}, odino, received_date
    ),
    first_recall as (
        select {KEYS}, min(recall_date) as first_recall_date
        from ({recalls_sql}) where recall_date is not null group by all
    ),
    kept as (
        select c.* from c left join first_recall f using ({KEYS})
        where f.first_recall_date is null or c.received_date < f.first_recall_date
    ),
    cell_month as (
        select {KEYS}, month,
               count(*) as n,
               sum(crash::int) as n_crash,
               sum(fire::int) as n_fire,
               sum((injured > 0)::int) as n_injured,
               sum((deaths > 0)::int) as n_deaths,
               sum((crash or fire or injured > 0 or deaths > 0)::int) as n_severe
        from kept group by all
    ),
    car_month as (
        select {CAR}, date_trunc('month', received_date)::date as month, count(distinct odino) as n
        from c group by all
    ),
    car_panel as (
        select car.*, m.month
        from (select {CAR}, min(month) as first_month from car_month group by all) car
        join (select distinct month from panel_v) m on m.month >= car.first_month
    ),
    car_3m as (
        select {CAR}, month,
               sum(coalesce(cm.n, 0)) over (partition by {CAR} order by month
                                           rows between 2 preceding and current row) as car_n_3m
        from car_panel left join car_month cm using ({CAR}, month)
    ),
    miles as (
        select {KEYS}, m.month, median(miles) as median_miles_12m
        from kept,
             unnest(generate_series(kept.month, kept.month + interval 11 month, interval 1 month)) as m(month)
        where miles > 0
        group by all
    ),
    p as (
        select panel_v.*,
               coalesce(cm.n, 0) as n, coalesce(cm.n_crash, 0) as n_crash, coalesce(cm.n_fire, 0) as n_fire,
               coalesce(cm.n_injured, 0) as n_injured, coalesce(cm.n_deaths, 0) as n_deaths,
               coalesce(cm.n_severe, 0) as n_severe
        from panel_v left join cell_month cm using ({KEYS}, month)
    ),
    w as (
        select {KEYS}, month,
               {window('n', 1)} as cmpl_1m,
               {window('n', 3)} as cmpl_3m,
               {window('n', 6)} as cmpl_6m,
               {window('n', 12)} as cmpl_12m,
               {window('n_severe', 3)} as severe_3m,
               {window('n_crash', 3)} as crash_3m,
               {window('n_fire', 3)} as fire_3m,
               {window('n_injured', 3)} as injured_3m,
               {window('n_deaths', 3)} as deaths_3m,
               {ys}, next_recall_date, next_recall_campaign_ids, last_recall_date
        from p
    )
    select w.make, w.model, w.model_year, w.component, w.month,
           cmpl_1m, cmpl_3m, cmpl_6m, severe_3m,
           year(w.month) - w.model_year as vehicle_age,
           y_6m as label,
           cmpl_12m,
           (cmpl_3m / 3.0) / ((cmpl_12m - cmpl_3m) / 9.0 + 1) as growth_3m,
           coalesce(cmpl_3m / nullif(car_n_3m, 0), 0) as share_of_car,
           crash_3m, fire_3m, injured_3m, deaths_3m,
           median_miles_12m,
           {ys}, next_recall_date, next_recall_campaign_ids, last_recall_date
    from w
    left join car_3m using ({CAR}, month)
    left join miles using ({KEYS}, month)
    order by w.make, w.model, w.model_year, w.component, w.month
    """)


def features_path(source):
    return OUT / f"monthly_features_{source}.parquet"


def publish(table, path):
    # contract tables live in data/nhtsa.duckdb, schema main
    with duckdb.connect(str(DB)) as con:
        con.execute(f"create or replace table main.{table} as select * from read_parquet('{path.as_posix()}')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_source_arg(parser)
    args = parser.parse_args()

    out = features_path(args.source)
    with connect() as con:
        con.execute("set preserve_insertion_order = false")
        build_features(con, *get_sql(args.source)).write_parquet(str(out))
        n, cells, lo, hi, y6 = con.sql(f"""
            select count(*), count(distinct (make, model, model_year, component)), min(month), max(month),
                   avg(label::int)
            from read_parquet('{out.as_posix()}')
        """).fetchall()[0]
    publish("monthly_features", out)
    print(f"wrote {out} and main.monthly_features: {n:,} rows, {cells:,} cells, {lo} to {hi}, "
          f"label rate {y6:.2%}")
