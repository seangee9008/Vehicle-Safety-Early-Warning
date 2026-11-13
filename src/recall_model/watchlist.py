"""Top 20 car + part rows per month from the model's predictions (watchlist contract in AGENTS.md).

The agent's CSV has exactly the contract columns. A month's row uses complaints received by
the end of that month. Reasons are simple facts for now; SHAP reasons replace them later in
top_reasons. A fuller copy with each reason, source and model version goes next to it.

    python -m src.recall_model.watchlist --source raw
"""
import argparse

import pandas as pd

from src.recall_model.features import publish
from src.recall_model.model import TOP
from src.recall_model.sources import OUT, add_source_arg

COLUMNS = ["month", "make", "model", "model_year", "component", "score", "rank", "top_reasons"]
EXTRA = ["cmpl_3m", "reason_1", "reason_2", "reason_3", "source", "model_version"]
VERSIONS = {"lgbm": "lgbm-v0", "baseline": "count3m-v0"}


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


def build_watchlist(pred, source, scorer="lgbm"):
    col = "score" if scorer == "lgbm" else "baseline"
    top = pred.sort_values(["month", col], ascending=[True, False]).groupby("month").head(TOP).copy()
    top["rank"] = top.groupby("month").cumcount() + 1
    top["score"] = top[col]
    r = [reasons(row) for row in top.itertuples()]
    top["reason_1"], top["reason_2"], top["reason_3"] = zip(*r)
    top["top_reasons"] = ["; ".join(x) for x in r]
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
    if args.split != "all":
        pred = pred[pred["split"] == args.split]
    wl = build_watchlist(pred, args.source, args.scorer)

    out = OUT / f"watchlist_{args.source}.csv"
    wl[COLUMNS].to_csv(out, index=False)
    full = OUT / f"watchlist_{args.source}_full.parquet"
    wl.to_parquet(full, index=False)
    publish("watchlist", full)
    print(f"wrote {out}, {full.name} and main.watchlist: {len(wl):,} rows, {wl['month'].nunique()} months")
    print(wl[COLUMNS].head(5).to_string(index=False))
