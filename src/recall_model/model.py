"""Baseline and LightGBM recall model with a time split and a rolling backtest.

Train on months up to the validation start, check on the validation years, then for each
test month refit on every row whose label was already known (month + horizon <= test month)
and score that month. Writes predictions, metrics and the last model, with the source in
each file name.

    python -m src.recall_model.model --source raw
"""
import argparse
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

from src.recall_model.features import features_path
from src.recall_model.sources import OUT, ROOT, add_source_arg, connect, get_sql

KEYS = ["make", "model", "model_year", "component"]
# v0: count and severity features only
MODEL_FEATURES = ["cmpl_1m", "cmpl_3m", "cmpl_6m", "cmpl_12m", "growth_3m", "share_of_car",
                  "crash_3m", "fire_3m", "injured_3m", "deaths_3m"]
TOP = 20
LOOKBACK_MONTHS = 12


def load_features(source, horizon, start):
    cols = KEYS + ["month"] + MODEL_FEATURES + [f"y_{horizon}m"]
    table = pq.read_table(features_path(source), columns=cols,
                          filters=[("month", ">=", pd.Timestamp(start).date())])
    df = table.to_pandas(strings_to_categorical=True)
    df["month"] = pd.to_datetime(df["month"])
    df[MODEL_FEATURES] = df[MODEL_FEATURES].astype("float32")
    return df.rename(columns={f"y_{horizon}m": "y"})


def baseline_score(df):
    # complaints in the last 3 months; cmpl_1m and cmpl_12m only break ties
    return df["cmpl_3m"] + df["cmpl_1m"] * 1e-3 + df["cmpl_12m"] * 1e-6


def fit(train):
    pos = train["y"].sum()
    # the full neg/pos weight (~200) makes the trees diverge and thousands of scores tie at 1.0;
    # its square root was best on the validation years
    model = lgb.LGBMClassifier(scale_pos_weight=np.sqrt((len(train) - pos) / pos), random_state=0,
                               n_jobs=-1, verbose=-1)
    model.fit(train[MODEL_FEATURES], train["y"])
    return model


def known_by(df, month, horizon):
    return df[df["month"] + pd.DateOffset(months=horizon) <= month]


def validate(df, horizon, val_start, val_end):
    model = fit(known_by(df, val_start, horizon))
    val = df[(df["month"] >= val_start) & (df["month"] <= val_end)].copy()
    val["score"] = model.predict_proba(val[MODEL_FEATURES])[:, 1]
    return val, model


def backtest(df, horizon, test_start, test_end, refit_every):
    months = pd.date_range(test_start - pd.DateOffset(months=LOOKBACK_MONTHS), test_end, freq="MS")
    parts, model = [], None
    for i, month in enumerate(months):
        if i % refit_every == 0:
            t = time.time()
            model = fit(known_by(df, month, horizon))
            print(f"  fit for {month:%Y-%m} in {time.time() - t:.0f}s")
        part = df[df["month"] == month].copy()
        part["score"] = model.predict_proba(part[MODEL_FEATURES])[:, 1]
        parts.append(part)
    return pd.concat(parts), model


def flags(pred, col):
    ranked = pred.sort_values(["month", col], ascending=[True, False])
    return ranked.groupby("month").head(TOP)


def recall_events(source, start, end):
    # every vehicle recall (car + part + date) in the period, and whether the cell had a complaint first
    complaints_sql, recalls_sql = get_sql(source)
    con = connect()
    events = con.sql(f"""
        with r as (
            select distinct make, model, model_year, component, recall_date
            from ({recalls_sql})
            where recall_date between date '{start}' and date '{end}' and model_year is not null
        ),
        c as (
            select make, model, model_year, component, min(received_date) as first_complaint
            from ({complaints_sql}) group by all
        )
        select r.*, coalesce(c.first_complaint < r.recall_date, false) as had_complaints
        from r left join c using (make, model, model_year, component)
    """).df()
    events["recall_date"] = pd.to_datetime(events["recall_date"])
    return events


def warning_weeks(flagged, events):
    # first top-20 flag in the 12 months before each recall; the flag is known at month end
    f = flagged[KEYS + ["month"]].copy()
    f["as_of"] = f["month"] + pd.offsets.MonthEnd(0)
    m = events.merge(f, on=KEYS, how="left")
    m = m[(m["as_of"] < m["recall_date"])
          & (m["as_of"] >= m["recall_date"] - pd.DateOffset(months=LOOKBACK_MONTHS))]
    first = m.groupby(KEYS + ["recall_date"], observed=True)["as_of"].min().rename("first_flag").reset_index()
    out = events.merge(first, on=KEYS + ["recall_date"], how="left")
    out["weeks"] = (out["recall_date"] - out["first_flag"]).dt.days / 7
    return out


def metrics(pred, events, col, lead_in=None):
    flagged = flags(pred, col)
    # months before the period can also give warning for recalls early in it
    pool = flagged if lead_in is None else pd.concat([flags(lead_in, col), flagged])
    caught = warning_weeks(pool, events)
    with_cmpl = caught[caught["had_complaints"]]
    return {
        "precision_at_20": flagged.groupby("month")["y"].mean().mean(),
        "pr_auc": average_precision_score(pred["y"], pred[col]),
        "recalls": len(caught),
        "recalls_with_complaints": len(with_cmpl),
        "catch_rate_all": caught["weeks"].notna().mean(),
        "catch_rate_with_complaints": with_cmpl["weeks"].notna().mean(),
        "median_weeks_warning": caught["weeks"].median(),
    }, caught


def report(pred, events, split, lead_in=None):
    rows, caught = [], None
    no_bags = lambda d: None if d is None else d[d["component"] != "AIR BAGS"]
    for name, subset, ev, lead in [("all", pred, events, lead_in),
                                   ("no air bags", no_bags(pred), no_bags(events), no_bags(lead_in))]:
        for scorer, col in [("baseline", "baseline"), ("lgbm", "score")]:
            m, c = metrics(subset, ev, col, lead)
            rows.append({"split": split, "parts": name, "scorer": scorer, **m})
            if name == "all" and scorer == "lgbm":
                caught = c
    return pd.DataFrame(rows), caught


def main():
    parser = argparse.ArgumentParser()
    add_source_arg(parser)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--train-start", default="2010-01-01")
    parser.add_argument("--val", nargs=2, default=["2020-01-01", "2021-12-01"])
    parser.add_argument("--test", nargs=2, default=["2022-01-01", "2024-12-01"])
    parser.add_argument("--refit-every", type=int, default=1, help="months between refits in the backtest")
    args = parser.parse_args()

    df = load_features(args.source, args.horizon, args.train_start)
    df["baseline"] = baseline_score(df)
    print(f"{len(df):,} rows, {df['month'].min():%Y-%m} to {df['month'].max():%Y-%m}, "
          f"positive rate {df['y'].mean():.2%}")

    val_start, val_end = map(pd.Timestamp, args.val)
    test_start, test_end = map(pd.Timestamp, args.test)
    if test_end + pd.DateOffset(months=args.horizon) > df["month"].max():
        print("warning: test labels run past the end of the data, so late months are missing recalls")

    print("validation")
    val, _ = validate(df, args.horizon, val_start, val_end)
    print("rolling backtest")
    test, model = backtest(df, args.horizon, test_start, test_end, args.refit_every)

    # scored recalls are the ones dated inside each period
    val_events = recall_events(args.source, val_start.date(), (val_end + pd.offsets.MonthEnd(0)).date())
    test_events = recall_events(args.source, test_start.date(), (test_end + pd.offsets.MonthEnd(0)).date())
    val_report, _ = report(val, val_events, "val")
    lead_in, test = test[test["month"] < test_start], test[test["month"] >= test_start]
    test_report, caught = report(test, test_events, "test", lead_in)
    results = pd.concat([val_report, test_report])
    results.insert(0, "source", args.source)

    val["split"], test["split"] = "val", "test"
    pred = pd.concat([val, test])[KEYS + ["month", "split", "y", "baseline", "score"] + MODEL_FEATURES]
    pred.to_parquet(OUT / f"predictions_{args.source}.parquet", index=False)
    caught.to_csv(OUT / f"warning_weeks_{args.source}.csv", index=False)
    (OUT / "models").mkdir(exist_ok=True)
    model.booster_.save_model(str(OUT / "models" / f"lgbm_{args.source}.txt"))
    results.to_csv(ROOT / "reports" / f"recall_metrics_{args.source}.csv", index=False)

    pd.set_option("display.width", 200)
    print(results.round(3).to_string(index=False))
    imp = pd.Series(model.booster_.feature_importance("gain"), MODEL_FEATURES).sort_values(ascending=False)
    print("\nfeature importance (gain, last model):")
    print((imp / imp.sum()).round(3).to_string())


if __name__ == "__main__":
    main()
