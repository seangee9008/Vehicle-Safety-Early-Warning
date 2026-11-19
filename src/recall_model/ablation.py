"""How much do complaints add over vehicle age? Runs the rolling backtest for an age-only and a
complaints-only LightGBM, with the same split, settings and monthly refits as model.py, and
compares them with v1 (read from predictions_<source>.parquet) and the complaint-count baseline.

Complaints-only leaves out vehicle_age and median_miles_12m (miles at complaint stands in for age).

    python -m src.recall_model.ablation --source stg
"""
import argparse

import numpy as np
import pandas as pd

from src.recall_model.model import (
    KEYS,
    MODEL_FEATURES,
    flags,
    load_features,
    metrics,
    recall_events,
)
from src.recall_model.model import backtest as run_backtest
from src.recall_model.results import SIZES, TEST, first_recalls
from src.recall_model.sources import OUT, ROOT, add_source_arg

AGE = ["vehicle_age"]
COMPLAINTS = [f for f in MODEL_FEATURES if f not in ("vehicle_age", "median_miles_12m")]


def scores(pred, events, name, col):
    test, lead_in = pred[pred["split"] == "test"], pred[pred["split"] == "lead_in"]
    rows = []
    for k in SIZES:
        m, caught = metrics(test, events, col, lead_in, top=k)
        first = first_recalls(caught)
        rows.append({"model": name, "top": k, "precision": m["precision_at_20"], "pr_auc": m["pr_auc"],
                     "recalls_caught": int(first["weeks"].notna().sum()),
                     "pct_recalls_caught": first["weeks"].notna().mean(),
                     "median_weeks_warning": first["weeks"].median(),
                     # % caught rewards a list that changes a lot from month to month, so show how much it does
                     "car_parts_flagged": len(flags(test, col, k)[KEYS].drop_duplicates())})
    return rows


def main():
    parser = argparse.ArgumentParser()
    add_source_arg(parser)
    args = parser.parse_args()
    start, end = pd.Timestamp(TEST[0]), pd.Timestamp("2024-12-01")

    events = recall_events(args.source, *TEST)
    v1 = pd.read_parquet(OUT / f"predictions_{args.source}.parquet")
    v1 = v1[v1["split"].isin(["lead_in", "test"])]
    rows = scores(v1, events, "baseline (3-month complaint count)", "baseline")

    df = load_features(args.source, 6, "2010-01-01", first_only=True)
    for name, features in [("age only", AGE), ("complaints only", COMPLAINTS)]:
        print(name)
        pred, _, _ = run_backtest(df, 6, start, end, refit_every=1, features=features)
        if features == AGE:
            # every car + part of the same age gets the same score; break ties with a fixed random
            # number per car + part so the list doesn't reshuffle every month (that inflates % caught)
            cell = pd.util.hash_pandas_object(pred[KEYS].astype(str), index=False).to_numpy()
            pred["score"] += (cell % 10**6) / 10**6 * 1e-9
        pred["split"] = np.where(pred["month"] < start, "lead_in", "test")
        rows += scores(pred, events, name, "score")
    rows += scores(v1, events, "v1 (complaints + age)", "score")

    out = pd.DataFrame(rows)
    out.insert(0, "source", args.source)
    out.to_csv(ROOT / "reports" / f"recall_ablation_{args.source}.csv", index=False)
    pd.set_option("display.width", 200)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
