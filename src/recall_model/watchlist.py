"""Top 20 car + part rows per month from the model's predictions (watchlist contract in AGENTS.md).

The agent's CSV has exactly the contract columns. A month's row uses complaints received by
the end of that month. top_reasons are the features that pushed the score up the most (SHAP),
written as plain facts; the baseline and other splits fall back to simple facts. A fuller copy
with each reason, source and model version goes next to it.

    python -m src.recall_model.watchlist --source raw
"""
import argparse

import pandas as pd

from src.recall_model.features import publish
from src.recall_model.model import KEYS, TOP
from src.recall_model.sources import OUT, add_source_arg

COLUMNS = ["month", "make", "model", "model_year", "component", "score", "rank", "top_reasons"]
EXTRA = ["cmpl_3m", "reason_1", "reason_2", "reason_3", "source", "model_version"]
VERSIONS = {"lgbm": "lgbm-v1", "baseline": "count3m-v0"}


def plural(n, word):
    n = int(n)
    return f"{n} {word}" + ("" if n == 1 else "es" if word.endswith("sh") else "s")


def reasons(r):
    out = []
    prior = (r.cmpl_12m - r.cmpl_3m) / 9
    if r.cmpl_3m >= 3 and prior > 0 and r.cmpl_3m / 3 >= 1.5 * prior:
        out.append(f"complaints up {r.cmpl_3m / 3 / prior:.1f}x vs. the prior 9 months")
    elif r.cmpl_3m >= 3 and prior == 0:
        out.append(f"{plural(r.cmpl_3m, 'complaint')} in 3 months, none in the prior 9")
    for col, word, detail in [("deaths_3m", "complaint", " with deaths"), ("fire_3m", "fire", ""),
                              ("crash_3m", "crash", ""), ("injured_3m", "complaint", " with injuries")]:
        if getattr(r, col) > 0:
            out.append(f"{plural(getattr(r, col), word)}{detail} in the last 3 months")
    if r.share_of_car >= 0.3 and r.cmpl_3m >= 3:
        out.append(f"{r.share_of_car:.0%} of this car's complaints in the last 3 months")
    out.append(f"{plural(r.cmpl_3m, 'complaint')} in the last 3 months")
    out.append(f"{plural(r.cmpl_12m, 'complaint')} in the last 12 months")
    out.append(f"{r.share_of_car:.0%} of this car's complaints in the last 3 months")
    return out[:3]


PERIOD = {"1m": "the last month", "3m": "the last 3 months", "6m": "the last 6 months", "12m": "the last 12 months"}
WORDS = {"cmpl": ("complaint", ""), "crash": ("crash", ""), "fire": ("fire", ""),
         "injured": ("complaint", " with injuries"), "deaths": ("complaint", " with deaths"),
         "kw_stall": ("complaint", " mentioning stalling"), "kw_smoke": ("complaint", " mentioning smoke"),
         "kw_no_warning": ("complaint", " saying it happened without warning"),
         "kw_lost_power": ("complaint", " mentioning lost power or a shut-off"),
         "kw_tow": ("complaint", " mentioning a tow")}


def describe(feature, r):
    # one model feature as a plain fact about the row, or None if there is nothing to say
    value = getattr(r, feature)
    if feature == "growth_3m":
        prior = (r.cmpl_12m - r.cmpl_3m) / 9
        if r.cmpl_3m > 0 and prior == 0:
            return f"{plural(r.cmpl_3m, 'complaint')} in 3 months, none in the prior 9"
        if prior > 0 and r.cmpl_3m / 3 > prior:
            return f"complaints up {r.cmpl_3m / 3 / prior:.1f}x vs. the prior 9 months"
        return None
    if feature == "share_of_car":
        return f"{value:.0%} of this car's complaints in the last 3 months" if value > 0 else None
    if feature == "vehicle_age":
        return f"vehicle is {plural(value, 'year')} old" if value >= 1 else "vehicle is under a year old"
    if feature == "median_miles_12m":
        return None if pd.isna(value) else f"median {value:,.0f} miles at complaint"
    if feature == "months_since_recall":
        return None if pd.isna(value) else f"last recalled {plural(value, 'month')} ago"
    name, months = feature.rsplit("_", 1)
    word, detail = WORDS[name]
    return f"{plural(value, word)}{detail} in {PERIOD[months]}" if value > 0 else None


def shap_reasons(r, features):
    # the features that pushed this row's score up the most (SHAP), biggest first
    out = []
    for f in sorted(features, key=lambda f: getattr(r, f"shap_{f}"), reverse=True):
        if getattr(r, f"shap_{f}") <= 0:
            break
        text = describe(f, r)
        if text:
            out.append(text)
    out = out or [f"{plural(r.cmpl_3m, 'complaint')} in the last 3 months"]
    return (out + ["", ""])[:3]


def build_watchlist(pred, source, scorer="lgbm", shap=None):
    col = "score" if scorer == "lgbm" else "baseline"
    top = pred.sort_values(["month", col], ascending=[True, False]).groupby("month").head(TOP).copy()
    top["rank"] = top.groupby("month").cumcount() + 1
    top["score"] = top[col]
    if shap is None:
        r = [reasons(row) for row in top.itertuples()]
    else:
        features = [c[5:] for c in shap.columns if c.startswith("shap_")]
        top = top.merge(shap, on=KEYS + ["month"], how="left", validate="one_to_one")
        r = [shap_reasons(row, features) for row in top.itertuples()]
    top["reason_1"], top["reason_2"], top["reason_3"] = zip(*r)
    top["top_reasons"] = ["; ".join(t for t in x if t) for x in r]
    top["month"] = top["month"].dt.date
    top["cmpl_3m"] = top["cmpl_3m"].astype(int)
    top["source"] = source
    top["model_version"] = VERSIONS[scorer]
    return top[COLUMNS + EXTRA].reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_source_arg(parser)
    parser.add_argument("--scorer", choices=list(VERSIONS), default="lgbm")
    parser.add_argument("--split", choices=["test", "val", "all"], default="test")
    args = parser.parse_args()

    pred = pd.read_parquet(OUT / f"predictions_{args.source}.parquet")
    for k in ["make", "model", "component"]:
        pred[k] = pred[k].astype(str)
    if args.split == "all":
        pred = pred[pred["split"] != "lead_in"]  # 2021 is already in val
    else:
        pred = pred[pred["split"] == args.split]
    shap = None
    if args.scorer == "lgbm" and args.split == "test":
        # SHAP values are saved for the backtest's flagged rows only
        shap = pd.read_parquet(OUT / f"shap_{args.source}.parquet")
        for k in ["make", "model", "component"]:
            shap[k] = shap[k].astype(str)
    wl = build_watchlist(pred, args.source, args.scorer, shap)

    out = OUT / f"watchlist_{args.source}.csv"
    wl[COLUMNS].to_csv(out, index=False)
    full = OUT / f"watchlist_{args.source}_full.parquet"
    wl.to_parquet(full, index=False)
    publish("watchlist", full)
    print(f"wrote {out}, {full.name} and main.watchlist: {len(wl):,} rows, {wl['month'].nunique()} months")
    print(wl[COLUMNS].head(5).to_string(index=False))
