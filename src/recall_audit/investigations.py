"""Experiment: does an open NHTSA investigation help Justine's recall model?

Adds three features to her v1 features, each known at month end:

  inv_open_part       an open PE / EA / DP investigation covers this make + model + year + part
  inv_open_car        ... covers this make + model + year, any part
  inv_open_make_part  ... covers this make + part, any model (shared supplier parts, e.g. inflators)

"Open" = opened (ODATE) on or before month end and not closed (CDATE) by then. Investigation
parts are mapped to Angelina's standardized components with the same mapping stg.complaints
uses. Justine's code is imported, never edited: same rows, label, LightGBM settings and refits.

Honesty rule: compare on the validation years (2020-21) first; run the 2022-24 test once, only
if validation improves.

    python -m src.recall_audit.investigations --stage val
    python -m src.recall_audit.investigations --stage test   # rolling backtest, ~1 fit per month

Leakage caveat: the investigations file is today's snapshot. If NHTSA added vehicles to an
investigation's scope after it opened, a month before that change would wrongly see it as
covered. `--stage val` prints how many covered vehicles have a model year later than the
investigation's open year + 1 (built after it opened, so certainly added later).
"""

from __future__ import annotations

import argparse
import time

import duckdb
import pandas as pd
from sklearn.metrics import average_precision_score

from src.recall_model.model import (
    KEYS,
    MODEL_FEATURES,
    backtest,
    fit,
    flags,
    known_by,
    load_features,
    metrics,
    recall_events,
)
from src.recall_model.sources import OUT
from src.sources import DB_PATH, ROOT, normalize_model_sql

INV_FEATURES = ["inv_open_part", "inv_open_car", "inv_open_make_part"]
TYPES = ("PE", "EA", "DP")  # preliminary evaluation, engineering analysis, defect petition
HORIZON = 6
VAL = ("2020-01-01", "2021-12-01")
TEST = ("2022-01-01", "2024-12-01")
REPORT = ROOT / "reports" / "investigations_experiment_stg.csv"

INVESTIGATIONS_SQL = f"""
with comp_map as (
    -- Angelina's raw top-level part -> standardized component, as used in stg.complaints
    select raw_top, arg_max(component, n) as component
    from (select upper(trim(split_part(component_raw, ':', 1))) as raw_top, component, count(*) as n
          from stg.complaints group by all)
    group by raw_top
),
inv as (
    select ACTION_NUMBER as action,
           upper(trim(MAKE)) as make,
           {normalize_model_sql('MODEL')} as model,
           TRY_CAST(YEAR AS INTEGER) as model_year,
           upper(trim(split_part(COMPNAME, ':', 1))) as raw_top,
           TRY_STRPTIME(ODATE, '%Y%m%d')::DATE as odate,
           TRY_STRPTIME(nullif(trim(CDATE), ''), '%Y%m%d')::DATE as cdate
    from raw.investigations
    where left(ACTION_NUMBER, 2) in {TYPES}
)
select distinct action, make, model, model_year,
       coalesce(comp_map.component, inv.raw_top) as component, odate, cdate
from inv left join comp_map using (raw_top)
where odate is not null and model_year is not null
"""


def investigations() -> pd.DataFrame:
    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        return con.sql(INVESTIGATIONS_SQL).df()


def monthly_open(inv: pd.DataFrame, keys: list[str], months: pd.DatetimeIndex) -> pd.DataFrame:
    """(keys, month) pairs with an investigation open at that month's end."""
    with duckdb.connect() as con:
        con.register("inv", inv)
        con.register("m", pd.DataFrame({"month": months}))
        cols = ", ".join(keys)
        return con.sql(f"""
            select distinct {cols}, m.month
            from inv join m
              on inv.odate <= last_day(m.month::date)
             and (inv.cdate is null or inv.cdate > last_day(m.month::date))
        """).df()


def add_features(df: pd.DataFrame, inv: pd.DataFrame) -> pd.DataFrame:
    months = pd.DatetimeIndex(sorted(df["month"].unique()))
    for name, keys in [("inv_open_part", KEYS), ("inv_open_car", ["make", "model", "model_year"]),
                       ("inv_open_make_part", ["make", "component"])]:
        flags_ = monthly_open(inv[keys + ["odate", "cdate"]].drop_duplicates(), keys, months)
        flags_[name] = 1.0
        for k in keys:  # match the categorical / int dtypes of df before merging
            if isinstance(df[k].dtype, pd.CategoricalDtype):  # vehicles the model never ranks
                flags_ = flags_[flags_[k].isin(df[k].cat.categories)]
            flags_[k] = flags_[k].astype(df[k].dtype)
        df = df.merge(flags_, on=keys + ["month"], how="left")
        df[name] = df[name].fillna(0.0).astype("float32")
    return df


def leakage_check(inv: pd.DataFrame) -> str:
    late = inv["model_year"] > inv["odate"].dt.year + 1
    return (f"{late.sum():,} of {len(inv):,} investigation vehicle rows ({late.mean():.1%}) have a model "
            "year later than the open year + 1, so they were added to the scope after opening")


def evaluate_val(df: pd.DataFrame) -> pd.DataFrame:
    val_start, val_end = map(pd.Timestamp, VAL)
    train = known_by(df, val_start, HORIZON)
    val = df[(df["month"] >= val_start) & (df["month"] <= val_end)].copy()
    rows = []
    for name, features in [("v1", MODEL_FEATURES), ("v1 + investigations", MODEL_FEATURES + INV_FEATURES)]:
        t = time.time()
        model = fit(train, features)
        val["score"] = model.predict_proba(val[features])[:, 1]
        top = flags(val, "score")
        rows.append({"split": "val", "model": name,
                     "precision_at_20": top.groupby("month")["y"].mean().mean(),
                     "pr_auc": average_precision_score(val["y"], val["score"]),
                     "hits_top20": int(top["y"].sum())})
        print(f"  {name}: fit in {time.time() - t:.0f}s", flush=True)
    return pd.DataFrame(rows)


def evaluate_test(df: pd.DataFrame) -> pd.DataFrame:
    start, end = map(pd.Timestamp, TEST)
    events = recall_events("stg", start.date(), (end + pd.offsets.MonthEnd(0)).date())
    pred, _, _ = backtest(df, HORIZON, start, end, refit_every=1, features=MODEL_FEATURES + INV_FEATURES)
    lead_in, test = pred[pred["month"] < start], pred[pred["month"] >= start]
    m_new, _ = metrics(test, events, "score", lead_in)
    v1 = pd.read_parquet(OUT / "predictions_stg.parquet")
    v1["month"] = pd.to_datetime(v1["month"])
    m_v1, _ = metrics(v1[v1["split"] == "test"], events, "score", v1[v1["split"] == "lead_in"])
    rows = []
    for name, m in [("v1 (Justine's predictions)", m_v1), ("v1 + investigations", m_new)]:
        rows.append({"split": "test", "model": name, "precision_at_20": m["precision_at_20"],
                     "pr_auc": m["pr_auc"], "median_weeks_warning": m["median_weeks_warning"],
                     "catch_rate_first_with_complaints": m["catch_rate_first_with_complaints"]})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", choices=["val", "test"], required=True)
    args = parser.parse_args()

    inv = investigations()
    print(f"{inv['action'].nunique():,} investigations, {len(inv):,} vehicle rows")
    print(leakage_check(inv))
    df = add_features(load_features("stg", HORIZON, "2010-01-01", first_only=True), inv)
    for f in INV_FEATURES:
        on = df[f] == 1
        print(f"{f}: {on.mean():.3%} of rows, label rate {df.loc[on, 'y'].mean():.2%} "
              f"vs {df.loc[~on, 'y'].mean():.2%} without")

    out = evaluate_val(df) if args.stage == "val" else evaluate_test(df)
    if REPORT.exists():
        old = pd.read_csv(REPORT)
        out = pd.concat([old[old["split"] != out["split"].iloc[0]], out])
    out.to_csv(REPORT, index=False)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
