"""Final results for the write-up, from the files model.py wrote.

- metrics for watchlists of the top 20, 50 and 100 per month, baseline vs LightGBM
- which recalls each scorer caught, and weeks of warning on the ones both caught
- why recalls were missed, plus three caught and three missed case studies
- charts in reports/

"% of recalls caught" counts first recalls of a car + part that had a complaint before them
(after a first recall a cell's complaints stop counting, so later recalls can't be caught).

    python -m src.recall_model.results --source stg
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.recall_model.figures import BLUE, GREY, INK, PAPER, new_axes
from src.recall_model.model import (
    KEYS,
    MODEL_FEATURES,
    fit,
    known_by,
    load_features,
    metrics,
    recall_events,
    shap_values,
)
from src.recall_model.sources import OUT, ROOT, add_source_arg, connect, get_sql
from src.recall_model.watchlist import describe, shap_reasons

ORANGE = "#eb6834"
SIZES = [20, 50, 100]
TEST = ("2022-01-01", "2024-12-31")
HORIZON = 6
SCORERS = [("baseline", "baseline"), ("lgbm", "score")]

# picked by hand from the lists this script prints (stg names)
CAUGHT = [("KIA", "SORENTO", 2021, "POWER TRAIN", "2022-10-11"),
          ("HONDA", "CRV", 2023, "STEERING", "2024-10-03"),
          ("TOYOTA", "COROLLACROSS", 2023, "STEERING", "2023-07-12")]
MISSED = [("RAM", "3500", 2018, "BRAKES", "2024-11-27"),
          ("KIA", "EV6", 2022, "ELECTRICAL SYSTEM", "2024-03-14"),
          ("FORD", "FOCUS", 2016, "ENGINE AND ENGINE COOLING", "2023-12-22")]


def load_test(source):
    pred = pd.read_parquet(OUT / f"predictions_{source}.parquet")
    pred = pred[pred["split"].isin(["lead_in", "test"])].reset_index(drop=True)
    for k in ["make", "model", "component"]:
        pred[k] = pred[k].astype(str)
    for col in ["score", "baseline"]:
        pred[f"rank_{col}"] = pred.groupby("month")[col].rank(ascending=False, method="first")
    return pred


def first_recalls(caught):
    return caught[caught["had_complaints"] & caught["prev_recall_date"].isna()]


def topk_table(pred, events):
    test, lead_in = pred[pred["split"] == "test"], pred[pred["split"] == "lead_in"]
    rows, caught = [], {}
    for k in SIZES:
        for scorer, col in SCORERS:
            m, c = metrics(test, events, col, lead_in, top=k)
            caught[scorer, k] = c
            first = first_recalls(c)
            rows.append({"top": k, "scorer": scorer,
                         "precision": m["precision_at_20"],
                         "recalls_caught": int(first["weeks"].notna().sum()),
                         "pct_recalls_caught": first["weeks"].notna().mean(),
                         "pct_recalls_with_complaints_caught": m["catch_rate_with_complaints"],
                         "median_weeks_warning": first["weeks"].median(),
                         "first_recalls_with_complaints": len(first)})
    return pd.DataFrame(rows), caught


def compare(caught, k):
    b = first_recalls(caught["baseline", k])[KEYS + ["recall_date", "weeks"]]
    l = first_recalls(caught["lgbm", k])[KEYS + ["recall_date", "weeks"]]
    return b.merge(l, on=KEYS + ["recall_date"], suffixes=("_baseline", "_lgbm"))


def overlap_table(caught):
    rows = []
    for k in SIZES:
        m = compare(caught, k)
        b, l = m["weeks_baseline"], m["weeks_lgbm"]
        both = b.notna() & l.notna()
        rows.append({"top": k, "caught_by_both": both.sum(),
                     "only_baseline": (b.notna() & l.isna()).sum(),
                     "only_lgbm": (b.isna() & l.notna()).sum(),
                     "both_median_weeks_baseline": b[both].median(),
                     "both_median_weeks_lgbm": l[both].median(),
                     "both_lgbm_earlier": (l > b)[both].sum(),
                     "both_same_month": (l == b)[both].sum(),
                     "both_lgbm_later": (l < b)[both].sum(),
                     "only_baseline_median_weeks": b[b.notna() & l.isna()].median(),
                     "only_lgbm_median_weeks": l[b.isna() & l.notna()].median()})
    return pd.DataFrame(rows)


def recall_windows(pred, caught):
    # complaints and best ranks in the months that could have warned about each first recall
    first = first_recalls(caught["lgbm", 20])[KEYS + ["recall_date", "first_flag", "weeks"]]
    m = first.merge(pred[KEYS + ["month", "cmpl_1m", "rank_score", "rank_baseline"]], on=KEYS)
    end = m["month"] + pd.offsets.MonthEnd(0)
    m = m[(end < m["recall_date"]) & (end >= m["recall_date"] - pd.DateOffset(months=12))]
    g = m.groupby(KEYS + ["recall_date"]).agg(complaints=("cmpl_1m", "sum"), best_rank=("rank_score", "min"),
                                              best_rank_baseline=("rank_baseline", "min")).reset_index()
    out = first.merge(g, on=KEYS + ["recall_date"], how="left")
    out["complaints"] = out["complaints"].fillna(0).astype(int)
    for k in SIZES[1:]:
        caught_k = first_recalls(caught["lgbm", k])[KEYS + ["recall_date", "weeks"]]
        out = out.merge(caught_k.rename(columns={"weeks": f"weeks_top{k}"}), on=KEYS + ["recall_date"])
    return out


def miss_reason(r):
    if r.complaints == 0:
        return "no complaints in the 12 months before"
    if r.complaints <= 2:
        return "1-2 complaints in the 12 months before"
    if r.best_rank <= 100:
        return "3+ complaints, best rank 21-100"
    return "3+ complaints, never in the top 100"


def monthly_complaints(source, cell, start, end):
    complaints_sql, _ = get_sql(source)
    with connect() as con:
        df = con.execute(f"""
            select date_trunc('month', ldate)::date as month, count(distinct odi_no) as n
            from ({complaints_sql})
            where make = ? and model = ? and model_year = ? and component = ? and ldate between ? and ?
            group by 1""", [*cell, start.date(), end.date()]).df()
    months = pd.DataFrame({"month": pd.date_range(start, end, freq="MS")})
    df["month"] = pd.to_datetime(df["month"])
    return months.merge(df, on="month", how="left").fillna({"n": 0})


def recall_info(source, cell, date):
    _, recalls_sql = get_sql(source)
    with connect() as con:
        df = con.execute(f"""
            select distinct camp_no, defect_summary from ({recalls_sql})
            where make = ? and model = ? and model_year = ? and component = ? and recall_date = ?""",
                         [*cell, date.date()]).df()
    summary = df["defect_summary"].dropna()
    return ", ".join(sorted(df["camp_no"])), (summary.iloc[0] if len(summary) else "")


def pushes(row, features, n=3):
    # features that pushed the score up and down the most, as facts with their SHAP size
    order = sorted(features, key=lambda f: getattr(row, f"shap_{f}"))
    text = lambda f: f"{describe(f, row) or f'{f} = {getattr(row, f):g}'} ({getattr(row, f'shap_{f}'):+.2f})"
    up = [text(f) for f in reversed(order) if getattr(row, f"shap_{f}") > 0][:n]
    down = [text(f) for f in order if getattr(row, f"shap_{f}") < 0][:n]
    return "; ".join(up), "; ".join(down)


def case_rows(source, pred, windows, cases, shap=None, df=None):
    rows, series = [], {}
    features = None
    for make, model, year, part, date in cases:
        cell, date = (make, model, year, part), pd.Timestamp(date)
        w = windows[(windows["make"] == make) & (windows["model"] == model) & (windows["model_year"] == year)
                    & (windows["component"] == part) & (windows["recall_date"] == date)]
        if w.empty:
            print(f"  {cell} {date:%Y-%m-%d} is not a first recall with complaints in this source, skipped")
            continue
        w = w.iloc[0]
        hist = pred[(pred["make"] == make) & (pred["model"] == model) & (pred["model_year"] == year)
                    & (pred["component"] == part)]
        end = hist["month"] + pd.offsets.MonthEnd(0)
        hist = hist[(end < date) & (end >= date - pd.DateOffset(months=12))]
        camp, summary = recall_info(source, cell, date)
        row = {"make": make, "model": model, "model_year": year, "component": part,
               "recall_date": date.date(), "campaign": camp, "complaints_12m_before": w["complaints"],
               "lgbm_first_flag": None if pd.isna(w["first_flag"]) else w["first_flag"].date(),
               "lgbm_weeks": w["weeks"], "lgbm_best_rank": w["best_rank"],
               "baseline_best_rank": w["best_rank_baseline"],
               "lgbm_weeks_top50": w["weeks_top50"], "lgbm_weeks_top100": w["weeks_top100"],
               "months_in_top20": int((hist["rank_score"] <= 20).sum()),
               "baseline_months_in_top20": int((hist["rank_baseline"] <= 20).sum()),
               "defect_summary": summary}
        if shap is not None:
            # caught: the reasons the watchlist gave at the first flag
            month = w["first_flag"].to_period("M").to_timestamp()
            r = hist[hist["month"] == month].merge(shap, on=KEYS + ["month"])
            features = features or [c[5:] for c in shap.columns if c.startswith("shap_")]
            row["flag_month"] = month.date()
            row["top_reasons"] = "; ".join(t for t in shap_reasons(next(r.itertuples()), features) if t)
        else:
            # missed: refit the model the backtest used in the month it ranked the cell highest
            month = hist.loc[hist["rank_score"].idxmin(), "month"]
            model_m = fit(known_by(df, month, HORIZON))
            r = hist[hist["month"] == month]
            s = shap_values(model_m, r)
            check = model_m.predict_proba(r[MODEL_FEATURES])[:, 1][0]
            print(f"  refit {month:%Y-%m}: score {check:.4f}, backtest score {r['score'].iloc[0]:.4f}")
            r = r.merge(s, on=KEYS + ["month"])
            row["best_rank_month"] = month.date()
            row["pushed_up"], row["pushed_down"] = pushes(next(r.itertuples()), MODEL_FEATURES)
        rows.append(row)
        series[cell, date] = (monthly_complaints(source, cell, date - pd.DateOffset(months=18),
                                                 date + pd.DateOffset(months=3)), hist)
    return pd.DataFrame(rows), series


def monthly_table(series, case):
    # complaints per month around each case's recall, with each scorer's rank that month
    rows = []
    for (cell, date), (monthly, hist) in series.items():
        m = monthly.merge(hist[["month", "rank_score", "rank_baseline"]], on="month", how="left")
        m.insert(0, "case", case)
        for k, v in zip(KEYS + ["recall_date"], [*cell, date.date()]):
            m[k] = v
        m["usable_before_recall"] = m["month"] + pd.offsets.MonthEnd(0) < date
        rows.append(m)
    out = pd.concat(rows).rename(columns={"n": "complaints", "rank_score": "lgbm_rank", "rank_baseline": "baseline_rank"})
    out["month"] = out["month"].dt.date
    return out[["case"] + KEYS + ["recall_date", "month", "complaints", "usable_before_recall",
                                  "lgbm_rank", "baseline_rank"]]


def style(ax):
    ax.set_facecolor(PAPER)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#c8c7c2")
    ax.tick_params(colors=GREY, length=0, labelsize=8)
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)


def topk_chart(table, source):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), facecolor=PAPER)
    x = np.arange(len(SIZES))
    for ax, col, title, fmt in [(axes[0], "precision", "Precision", "{:.0%}"),
                                (axes[1], "pct_recalls_caught", "% of recalls caught", "{:.1%}")]:
        style(ax)
        for i, (scorer, label, color) in enumerate([("baseline", "Baseline (complaint count)", ORANGE),
                                                    ("lgbm", "LightGBM", BLUE)]):
            v = table[table["scorer"] == scorer].set_index("top").loc[SIZES, col].values
            ax.bar(x + (i - 0.5) * 0.38, v, width=0.36, color=color, label=label)
            for xi, vi in zip(x + (i - 0.5) * 0.38, v):
                ax.text(xi, vi, fmt.format(vi), ha="center", va="bottom", color=GREY, fontsize=7.5)
        ax.set_xticks(x, [f"top {k}" for k in SIZES])
        ax.set_yticks([])
        ax.set_title(title, loc="left", color=INK, fontsize=10, fontweight="bold", pad=20)
    n = table["first_recalls_with_complaints"].iloc[0]
    axes[0].text(0, 1.03, "share of flagged rows recalled within 6 months", transform=axes[0].transAxes,
                 color=GREY, fontsize=8)
    axes[1].text(0, 1.03, f"of {n:,} first recalls in 2022-24 with a complaint beforehand",
                 transform=axes[1].transAxes, color=GREY, fontsize=8)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right", labelcolor=INK)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / f"recall_topk_{source}.png", dpi=160)


def warning_compare_chart(caught, source, k=20):
    m = compare(caught, k)
    b, l = m["weeks_baseline"], m["weeks_lgbm"]
    both = b.notna() & l.notna()
    groups = [("Caught by both: baseline", b[both], ORANGE), ("Caught by both: LightGBM", l[both], BLUE),
              ("Only baseline caught", b[b.notna() & l.isna()], ORANGE),
              ("Only LightGBM caught", l[b.isna() & l.notna()], BLUE)]
    fig, ax = new_axes(f"Weeks of warning, top {k} per month",
                       "each dot is a 2022-24 recall; bar = median. LightGBM's extra catches come later")
    rng = np.random.default_rng(0)
    for y, (label, v, color) in enumerate(groups):
        ax.scatter(v, y + rng.uniform(-0.18, 0.18, len(v)), s=22, color=color, alpha=0.75,
                   edgecolors=PAPER, linewidths=0.8)
        ax.plot([v.median()] * 2, [y - 0.3, y + 0.3], color=INK, lw=2)
        ax.text(v.median(), y - 0.34, f"median {v.median():.0f} wk", ha="center", color=INK, fontsize=8)
    ax.set_yticks(range(len(groups)), [f"{g[0]} (n={len(g[1])})" for g in groups])
    ax.tick_params(axis="y", labelcolor=INK)
    ax.set_ylim(-0.6, len(groups) - 0.3)
    ax.invert_yaxis()
    ax.set_xlim(0, 54)
    ax.set_xlabel("weeks between the first flag and the recall date", color=GREY, fontsize=9)
    ax.grid(axis="x", color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(ROOT / "reports" / f"recall_warning_compare_{source}.png", dpi=160)


def case_chart(cases, series, title, path, caught=True):
    fig, axes = plt.subplots(len(cases), 1, figsize=(8, 2.4 * len(cases)), facecolor=PAPER)
    fig.suptitle(title, x=0.02, ha="left", color=INK, fontsize=12, fontweight="bold")
    for ax, (_, c) in zip(np.atleast_1d(axes), cases.iterrows()):
        style(ax)
        date = pd.Timestamp(c["recall_date"])
        monthly, hist = series[(c["make"], c["model"], c["model_year"], c["component"]), date]
        before = monthly["month"] + pd.offsets.MonthEnd(0) < date
        ax.bar(monthly["month"] + pd.Timedelta(days=14), monthly["n"], width=24,
               color=np.where(before, "#a3a29c", "#dcdbd5"))
        top = max(monthly["n"].max(), 1)
        for col, color, y in [("rank_score", BLUE, 1.18), ("rank_baseline", ORANGE, 1.08)]:
            f = hist[hist[col] <= 20]["month"] + pd.Timedelta(days=14)
            ax.scatter(f, [top * y] * len(f), marker="v", s=28, color=color, zorder=3)
        if not caught:
            best = hist.loc[hist["rank_score"].idxmin(), "month"] + pd.Timedelta(days=14)
            ax.text(best, top * 1.18, f"#{int(hist['rank_score'].min())}", color=BLUE, fontsize=8,
                    ha="center", va="center", fontweight="bold")
        ax.axvline(date, color=INK, lw=1.2, ls="--")
        ax.text(date, top * 1.3, " recall", color=INK, fontsize=8, va="top")
        if caught:
            note = (f"first LightGBM flag {c['lgbm_weeks']:.0f} weeks before, rank {int(c['lgbm_best_rank'])} at best;"
                    f" baseline best rank {int(c['baseline_best_rank'])}")
        else:
            note = (f"{c['complaints_12m_before']} complaints in the 12 months before; LightGBM best rank "
                    f"{int(c['lgbm_best_rank'])}, baseline {int(c['baseline_best_rank'])}")
        ax.set_title(f"{c['model_year']} {c['make']} {c['model']}, {c['component'].lower()} "
                     f"(campaign {c['campaign']})", loc="left", color=INK, fontsize=9.5, fontweight="bold", pad=16)
        ax.text(0, 1.03, note, transform=ax.transAxes, color=GREY, fontsize=8, va="bottom")
        ax.set_ylim(0, top * 1.35)
        ax.set_ylabel("complaints / month", color=GREY, fontsize=8)
    axes[-1].text(0, -0.32, "▼ blue = LightGBM top 20 that month, ▼ orange = baseline top 20; "
                  "blue # = LightGBM's best rank; pale bars = too late to warn",
                  transform=axes[-1].transAxes, color=GREY, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main():
    parser = argparse.ArgumentParser()
    add_source_arg(parser)
    args = parser.parse_args()
    src = args.source
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    pred = load_test(src)
    events = recall_events(src, *TEST)
    for k in ["make", "model", "component"]:
        events[k] = events[k].astype(str)

    table, caught = topk_table(pred, events)
    overlap = overlap_table(caught)
    windows = recall_windows(pred, caught)
    missed = windows[windows["weeks"].isna()].copy()
    missed["reason"] = [miss_reason(r) for r in missed.itertuples()]
    reasons = missed["reason"].value_counts().rename("recalls").to_frame()
    reasons["share"] = reasons["recalls"] / len(missed)

    print(table.round(4).to_string(index=False))
    print(overlap.round(1).to_string(index=False))
    print(f"\nwhy LightGBM missed {len(missed)} of {len(windows)} first recalls at top 20:")
    print(reasons.round(3).to_string())
    cols = KEYS + ["recall_date", "complaints", "best_rank", "best_rank_baseline", "weeks"]
    print("\ncaught, most warning first:")
    print(windows[windows["weeks"].notna()].sort_values("weeks", ascending=False)[cols].head(25).to_string(index=False))
    print("\nmissed, most complaints first:")
    print(missed.sort_values("complaints", ascending=False)[cols].head(25).to_string(index=False))

    shap = pd.read_parquet(OUT / f"shap_{src}.parquet")
    for k in ["make", "model", "component"]:
        shap[k] = shap[k].astype(str)
    caught_cases, caught_series = case_rows(src, pred, windows, CAUGHT, shap=shap)
    df = load_features(src, HORIZON, "2010-01-01", first_only=True)
    missed_cases, missed_series = case_rows(src, pred, windows, MISSED, df=df)

    reports = ROOT / "reports"
    table.to_csv(reports / f"recall_topk_{src}.csv", index=False)
    overlap.to_csv(reports / f"recall_overlap_{src}.csv", index=False)
    reasons.to_csv(reports / f"recall_miss_reasons_{src}.csv")
    pd.concat([caught_cases.assign(case="caught"), missed_cases.assign(case="missed")]).to_csv(
        reports / f"recall_cases_{src}.csv", index=False)
    pd.concat([monthly_table(caught_series, "caught"), monthly_table(missed_series, "missed")]).to_csv(
        reports / f"recall_cases_monthly_{src}.csv", index=False)
    windows.to_csv(OUT / f"first_recalls_{src}.csv", index=False)

    topk_chart(table, src)
    warning_compare_chart(caught, src)
    if len(caught_cases):
        case_chart(caught_cases, caught_series, "Three recalls LightGBM caught",
                   reports / f"recall_cases_caught_{src}.png")
    if len(missed_cases):
        case_chart(missed_cases, missed_series, "Three recalls LightGBM missed",
                   reports / f"recall_cases_missed_{src}.png", caught=False)
    print(pd.concat([caught_cases, missed_cases]).drop(columns="defect_summary").to_string(index=False))
    print(f"wrote reports/recall_topk_{src}.csv, recall_overlap, recall_miss_reasons, recall_cases (+ 4 PNGs)")


if __name__ == "__main__":
    main()
