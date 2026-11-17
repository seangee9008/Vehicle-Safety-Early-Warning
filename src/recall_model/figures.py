"""Charts for the write-up: weeks of warning, and which features drive the model (SHAP).

Reads the files model.py wrote and saves PNGs in reports/.

    python -m src.recall_model.figures --source stg
"""
import argparse

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.recall_model.model import MODEL_FEATURES
from src.recall_model.sources import OUT, ROOT, add_source_arg

BLUE, INK, GREY, PAPER = "#2a78d6", "#0b0b0b", "#52514e", "#fcfcfb"


def new_axes(title, note):
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor=PAPER)
    ax.set_facecolor(PAPER)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=22)
    ax.text(0, 1.03, note, transform=ax.transAxes, color=GREY, fontsize=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#c8c7c2")
    ax.tick_params(colors=GREY, length=0, labelsize=9)
    return fig, ax


def warning_chart(source):
    weeks = pd.read_csv(OUT / f"warning_weeks_{source}.csv")["weeks"].dropna()
    fig, ax = new_axes("Weeks of warning before a recall",
                       f"{len(weeks)} test recalls (2022-24) flagged in the top 20 beforehand, LightGBM")
    ax.hist(weeks, bins=np.arange(0, 57, 4), color=BLUE, rwidth=0.92)
    ax.axvline(weeks.median(), color=INK, lw=1)
    ax.text(weeks.median() + 0.8, ax.get_ylim()[1] * 0.95, f"median {weeks.median():.0f} weeks", color=INK, fontsize=9)
    ax.set_xticks(np.arange(0, 57, 8))
    ax.set_xlabel("weeks between the first flag and the recall date", color=GREY, fontsize=9)
    ax.set_ylabel("recalls", color=GREY, fontsize=9)
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / f"recall_warning_weeks_{source}.png", dpi=160)


def shap_chart(source, sample=200_000):
    # mean |SHAP| of the last model over a sample of test rows
    pred = pd.read_parquet(OUT / f"predictions_{source}.parquet", columns=["split", *MODEL_FEATURES])
    rows = pred[pred["split"] == "test"].sample(sample, random_state=0)
    booster = lgb.Booster(model_file=str(OUT / "models" / f"lgbm_{source}.txt"))
    values = booster.predict(rows[MODEL_FEATURES], pred_contrib=True)[:, :-1]
    size = pd.Series(np.abs(values).mean(axis=0), MODEL_FEATURES).sort_values()

    fig, ax = new_axes("What moves the recall score",
                       f"mean |SHAP| in log-odds, {sample:,} test rows, last backtest model")
    ax.barh(size.index, size.values, color=BLUE, height=0.6)
    for y, v in enumerate(size.values):
        ax.text(v, y, f" {v:.2f}", va="center", color=GREY, fontsize=8)
    ax.spines["bottom"].set_visible(False)
    ax.set_xticks([])
    ax.tick_params(axis="y", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / f"recall_shap_summary_{source}.png", dpi=160)
    return size.sort_values(ascending=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_source_arg(parser)
    args = parser.parse_args()
    warning_chart(args.source)
    print(shap_chart(args.source).round(3).to_string())
    print(f"wrote reports/recall_warning_weeks_{args.source}.png and recall_shap_summary_{args.source}.png")
