"""Baseline and LightGBM recall model with a time split and a rolling backtest.

Only car + parts that have not been recalled yet are trained on and ranked, because a cell's
complaints stop counting at its first recall (--all-rows keeps the rest in). Train on months
up to the validation start, check on the validation years, then for each
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
import pyarrow.compute as pc
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score

from src.recall_model.features import features_path
from src.recall_model.sources import OUT, ROOT, add_source_arg, connect, get_sql

KEYS = ["make", "model", "model_year", "component"]
# v1: counts, severity, vehicle age and mileage, keyword counts from the complaint text.
# Features and settings were picked on the validation years with tune.py (reports/recall_tuning_stg.csv).
MODEL_FEATURES = ["cmpl_1m", "cmpl_3m", "cmpl_6m", "cmpl_12m", "growth_3m", "share_of_car",
                  "crash_3m", "fire_3m", "injured_3m", "deaths_3m",
                  "vehicle_age", "median_miles_12m",
                  "kw_stall_3m", "kw_smoke_3m", "kw_no_warning_3m", "kw_lost_power_3m", "kw_tow_3m"]
PARAMS = {"learning_rate": 0.05, "n_estimators": 200, "num_leaves": 15}
TOP = 20
LOOKBACK_MONTHS = 12


def load_features(source, horizon, start, end="2100-01-01", features=MODEL_FEATURES, first_only=False):
    cols = KEYS + ["month"] + features + [f"y_{horizon}m"]
    keep = (pc.field("month") >= pd.Timestamp(start).date()) & (pc.field("month") <= pd.Timestamp(end).date())
    if first_only:
        # drop months after a car + part's first recall: its complaints stop counting there (rule 3),
        # so those rows have nothing for the model to see
        keep = keep & pc.field("last_recall_date").is_null()
    table = pq.read_table(features_path(source), columns=cols, filters=keep)
    df = table.to_pandas(strings_to_categorical=True)
    df["month"] = pd.to_datetime(df["month"])
    df[features] = df[features].astype("float32")
    return df.rename(columns={f"y_{horizon}m": "y"})


def baseline_score(df):
    # complaints in the last 3 months; cmpl_1m and cmpl_12m only break ties
    return df["cmpl_3m"] + df["cmpl_1m"] * 1e-3 + df["cmpl_12m"] * 1e-6


def fit(train, features=MODEL_FEATURES, params=PARAMS):
    pos = train["y"].sum()
    # the full neg/pos weight (~200) makes the trees diverge and thousands of scores tie at 1.0;
    # its square root was best on the validation years
    model = lgb.LGBMClassifier(scale_pos_weight=np.sqrt((len(train) - pos) / pos), random_state=0,
                               n_jobs=-1, verbose=-1, **params)
    model.fit(train[features], train["y"])
    return model


def known_by(df, month, horizon):
    return df[df["month"] + pd.DateOffset(months=horizon) <= month]


def validate(df, horizon, val_start, val_end):
    model = fit(known_by(df, val_start, horizon))
    val = df[(df["month"] >= val_start) & (df["month"] <= val_end)].copy()
    val["score"] = model.predict_proba(val[MODEL_FEATURES])[:, 1]
    return val, model


def flags(pred, col):
    ranked = pred.sort_values(["month", col], ascending=[True, False])
    return ranked.groupby("month").head(TOP)


def shap_values(model, rows):
    # how much each feature moved each row's score (log-odds), from the model that scored it
    values = model.booster_.predict(rows[MODEL_FEATURES], pred_contrib=True)[:, :-1]
    out = rows[KEYS + ["month"]].reset_index(drop=True)
    out[[f"shap_{f}" for f in MODEL_FEATURES]] = values
    return out


def backtest(df, horizon, test_start, test_end, refit_every):
    months = pd.date_range(test_start - pd.DateOffset(months=LOOKBACK_MONTHS), test_end, freq="MS")
    parts, shap_parts, model = [], [], None
    for i, month in enumerate(months):
        if i % refit_every == 0:
            t = time.time()
            model = fit(known_by(df, month, horizon))
            print(f"  fit for {month:%Y-%m} in {time.time() - t:.0f}s", flush=True)
        part = df[df["month"] == month].copy()
        part["score"] = model.predict_proba(part[MODEL_FEATURES])[:, 1]
        parts.append(part)
        shap_parts.append(shap_values(model, flags(part, "score")))
    return pd.concat(parts), pd.concat(shap_parts), model


def recall_events(source, start, end):
    # every vehicle recall (car + part + date) in the period, the cell's previous recall, and
    # whether the cell had a complaint between the two
    complaints_sql, recalls_sql = get_sql(source)
    with connect() as con:
        events = con.sql(f"""
            with r as (
                select *, lag(recall_date) over (partition by make, model, model_year, component
                                                 order by recall_date) as prev_recall_date
                from (select distinct make, model, model_year, component, recall_date
                      from ({recalls_sql}) where recall_date is not null and model_year is not null)
            ),
            c as (select make, model, model_year, component, ldate as received_date from ({complaints_sql}))
            select r.make, r.model, r.model_year, r.component, r.recall_date, r.prev_recall_date,
                   count(c.received_date) > 0 as had_complaints
            from r left join c
              on r.make = c.make and r.model = c.model and r.model_year = c.model_year
             and r.component = c.component and c.received_date < r.recall_date
             and (r.prev_recall_date is null or c.received_date > r.prev_recall_date)
            where r.recall_date between date '{start}' and date '{end}'
            group by all
        """).df()
    events["recall_date"] = pd.to_datetime(events["recall_date"])
    events["prev_recall_date"] = pd.to_datetime(events["prev_recall_date"])
    return events


def warning_weeks(flagged, events):
    # first top-20 flag in the 12 months before each recall and after the cell's previous recall;
    # the flag is known at month end
    f = flagged[KEYS + ["month"]].copy()
    f["as_of"] = f["month"] + pd.offsets.MonthEnd(0)
    m = events.merge(f, on=KEYS, how="left")
    m = m[(m["as_of"] < m["recall_date"])
          & (m["as_of"] >= m["recall_date"] - pd.DateOffset(months=LOOKBACK_MONTHS))
          & (m["prev_recall_date"].isna() | (m["month"] > m["prev_recall_date"]))]
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
    # after a cell's first recall its complaints never count again, so only first recalls can be caught
    first = with_cmpl[with_cmpl["prev_recall_date"].isna()]
    return {
        "precision_at_20": flagged.groupby("month")["y"].mean().mean(),
        "pr_auc": average_precision_score(pred["y"], pred[col]),
        "recalls": len(caught),
        "recalls_with_complaints": len(with_cmpl),
        "catch_rate_all": caught["weeks"].notna().mean(),
        "catch_rate_with_complaints": with_cmpl["weeks"].notna().mean(),
        "first_recalls_with_complaints": len(first),
        "catch_rate_first_with_complaints": first["weeks"].notna().mean(),
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
    parser.add_argument("--all-rows", action="store_true",
                        help="also train on and rank car + parts that were already recalled")
    args = parser.parse_args()
    # outputs of an --all-rows run get their own names so the two never get mixed up
    name = args.source + ("_all_rows" if args.all_rows else "")

    df = load_features(args.source, args.horizon, args.train_start, first_only=not args.all_rows)
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
    test, shap, model = backtest(df, args.horizon, test_start, test_end, args.refit_every)

    # scored recalls are the ones dated inside each period
    val_events = recall_events(args.source, val_start.date(), (val_end + pd.offsets.MonthEnd(0)).date())
    test_events = recall_events(args.source, test_start.date(), (test_end + pd.offsets.MonthEnd(0)).date())
    val_report, _ = report(val, val_events, "val")
    lead_in, test = test[test["month"] < test_start], test[test["month"] >= test_start]
    test_report, caught = report(test, test_events, "test", lead_in)
    results = pd.concat([val_report, test_report])
    results.insert(0, "source", args.source)

    val["split"], lead_in["split"], test["split"] = "val", "lead_in", "test"
    pred = pd.concat([val, lead_in, test])[KEYS + ["month", "split", "y", "baseline", "score"] + MODEL_FEATURES]
    pred.to_parquet(OUT / f"predictions_{name}.parquet", index=False)
    shap.to_parquet(OUT / f"shap_{name}.parquet", index=False)
    caught.to_csv(OUT / f"warning_weeks_{name}.csv", index=False)
    (OUT / "models").mkdir(exist_ok=True)
    model.booster_.save_model(str(OUT / "models" / f"lgbm_{name}.txt"))
    results.to_csv(ROOT / "reports" / f"recall_metrics_{name}.csv", index=False)

    pd.set_option("display.width", 200)
    print(results.round(3).to_string(index=False))
    imp = pd.Series(model.booster_.feature_importance("gain"), MODEL_FEATURES).sort_values(ascending=False)
    print("\nfeature importance (gain, last model):")
    print((imp / imp.sum()).round(3).to_string())


if __name__ == "__main__":
    main()
