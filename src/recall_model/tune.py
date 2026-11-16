"""Pick the feature set and LightGBM settings on the validation years only (2020-21).

Each try is fit on rows whose label was known before 2020 and scored on 2020-21. The test
years are never loaded. Results go to reports/recall_tuning_<source>.csv; the winner is
copied by hand into MODEL_FEATURES and PARAMS in model.py.

    python -m src.recall_model.tune --source stg
"""
import argparse
import time

import pandas as pd
from sklearn.metrics import average_precision_score

from src.recall_model.features import KEYWORDS
from src.recall_model.model import fit, known_by, load_features
from src.recall_model.sources import ROOT, add_source_arg

COUNTS = ["cmpl_1m", "cmpl_3m", "cmpl_6m", "cmpl_12m", "growth_3m", "share_of_car",
          "crash_3m", "fire_3m", "injured_3m", "deaths_3m"]
VEHICLE = ["vehicle_age", "median_miles_12m"]
TEXT = [f"kw_{k}_3m" for k in KEYWORDS]
FEATURE_SETS = {
    "counts (v0)": COUNTS,
    "counts + vehicle": COUNTS + VEHICLE,
    "counts + text": COUNTS + TEXT,
    "counts + vehicle + text": COUNTS + VEHICLE + TEXT,
    # for the team's repeat-recall question, not a candidate until that's decided
    "counts + vehicle + text + months_since_recall": COUNTS + VEHICLE + TEXT + ["months_since_recall"],
}
SETTINGS = [
    {},
    {"num_leaves": 15},
    {"num_leaves": 63},
    {"min_child_samples": 200},
    {"min_child_samples": 1000},
    {"learning_rate": 0.05, "n_estimators": 200},
    {"learning_rate": 0.05, "n_estimators": 200, "num_leaves": 15},
    {"learning_rate": 0.05, "n_estimators": 200, "min_child_samples": 200},
    {"colsample_bytree": 0.7, "subsample": 0.7, "subsample_freq": 1},
    {"reg_lambda": 10},
]


def precision_at(val, k):
    top = val.sort_values(["month", "score"], ascending=[True, False]).groupby("month").head(k)
    return top.groupby("month")["y"].mean().mean()


def score(train, val, features, params):
    t = time.time()
    model = fit(train, features, params)
    val = val.assign(score=model.predict_proba(val[features])[:, 1])
    return {"precision_at_20": precision_at(val, 20), "precision_at_100": precision_at(val, 100),
            "pr_auc": average_precision_score(val["y"], val["score"]), "seconds": round(time.time() - t)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_source_arg(parser)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--val", nargs=2, default=["2020-01-01", "2021-12-01"])
    parser.add_argument("--all-rows", action="store_true",
                        help="also train on and rank car + parts that were already recalled")
    args = parser.parse_args()

    val_start, val_end = map(pd.Timestamp, args.val)
    if not args.all_rows:
        del FEATURE_SETS["counts + vehicle + text + months_since_recall"]  # always empty without recalled rows
    everything = sorted({f for features in FEATURE_SETS.values() for f in features})
    df = load_features(args.source, args.horizon, "2010-01-01", val_end, everything, not args.all_rows)
    train = known_by(df, val_start, args.horizon)
    val = df[df["month"] >= val_start]
    print(f"train {len(train):,} rows, validation {len(val):,} rows")

    rows = []
    for name, features in FEATURE_SETS.items():
        rows.append({"features": name, "settings": "{}", **score(train, val, features, {})})
        print(rows[-1], flush=True)
    candidates = [r for r in rows if "months_since_recall" not in r["features"]]
    # precision@20 only has 480 flags over the two years, so one or two hits move it;
    # precision@100 is the steadier top-of-list number, with PR-AUC as the tie-break
    best = max(candidates, key=lambda r: (round(r["precision_at_100"], 3), r["pr_auc"]))["features"]
    print(f"best feature set: {best}")
    for params in SETTINGS[1:]:
        rows.append({"features": best, "settings": str(params), **score(train, val, FEATURE_SETS[best], params)})
        print(rows[-1], flush=True)

    out = ROOT / "reports" / f"recall_tuning_{args.source}{'_all_rows' if args.all_rows else ''}.csv"
    results = pd.DataFrame(rows)
    results.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(results.round(4).to_string(index=False))
    print(f"wrote {out}")
