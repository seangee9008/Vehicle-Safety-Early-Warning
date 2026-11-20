"""Vehicle Safety Early Warning: demo.

    python run.py demo                       # or: streamlit run app/streamlit_app.py
    VSE_DATA=local python run.py demo        # skip BigQuery, use local files (offline)

Reads the published results from BigQuery (src/bq_reader.py) and falls back to local files.
Every number on screen comes from the data. No Gemini calls: the briefs were written and
validated beforehand.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app import charts
from app import data as D

st.set_page_config(page_title="Vehicle Safety Early Warning", page_icon="🚗", layout="wide")

SCORER_LABELS = {"lgbm": "model (LightGBM)", "baseline": "baseline (complaint count)"}
OUTCOME_LABELS = {"passed_first_try": "passed first try", "passed_after_retry": "passed after retry",
                  "still_failing": "still failing"}
FIGURES = [
    ("recall_topk_stg.png", "Precision and recalls caught at top 20 / 50 / 100"),
    ("recall_warning_compare_stg.png", "Weeks of warning: model vs baseline"),
    ("recall_shap_summary_stg.png", "What drives the model (SHAP)"),
    ("recall_cases_caught_stg.png", "Recalls the model caught"),
]


@st.cache_data(ttl=3600, show_spinner="Loading results…")
def load() -> D.Data:
    return D.load_all()


def pct(x, digits: int = 1) -> str:
    return "–" if x is None or pd.isna(x) else f"{100 * x:.{digits}f}%"


def num(x, digits: int = 1, unit: str = "") -> str:
    return "–" if x is None or pd.isna(x) else f"{x:,.{digits}f}{unit}"


def signed_pct(x) -> str:
    """+6% / -27% / 0% (no "-0%" from rounding)."""
    if x is None or pd.isna(x):
        return "–"
    value = round(100 * x)
    return "0%" if value == 0 else f"{value:+d}%"


def missing(name: str, data: D.Data) -> None:
    why = data.problems.get(name, "")
    st.info(f"`{name}` isn't available right now." + (f" ({why})" if why else ""))


def source_badge(data: D.Data) -> None:
    if data.all_bigquery:
        st.markdown(":blue-badge[BigQuery · waymo-project-513802.vehicle_safety]")
    elif data.any_local:
        st.markdown(":orange-badge[local data]")
        if data.problems:
            with st.expander("Why local data?"):
                for name, why in data.problems.items():
                    st.caption(f"`{name}`: {why}")


def citation_counts(data: D.Data) -> tuple[int, int, int | None]:
    wb, cites = data.get("v_watchlist_with_briefs"), data.get("agent_citations")
    briefs = wb[wb["has_brief"]] if wb is not None else pd.DataFrame()
    valid = int(briefs["citations_valid"].sum()) if len(briefs) else 0
    total = int(briefs["citations_total"].sum()) if len(briefs) else 0
    unique = int(cites["odino"].nunique()) if cites is not None and len(cites) else None
    return valid, total, unique


# ---------------------------------------------------------------------------------------------

def _track_diagram(title: str, color: str, steps: list[str]) -> str:
    nodes = "; ".join(f'n{i} [label="{label}"]' for i, label in enumerate(steps))
    edges = " -> ".join(f"n{i}" for i in range(len(steps)))
    return (f'digraph {{ rankdir=LR; bgcolor="transparent"; label="{title}"; labelloc=t; '
            f'fontname="Helvetica"; fontcolor="{color}"; '
            'node [shape=box, style="rounded,filled", fillcolor="#f4f6f8", color="#c9ced6", '
            'fontname="Helvetica", fontsize=11]; edge [color="#8a94a1"]; '
            f"{nodes}; {edges}; }}")


def tab_overview(data: D.Data) -> None:
    st.markdown("#### Spot vehicle safety problems early: months ahead and seconds ahead")
    h = data.get("v_headline_metrics")
    if h is None or h.empty:
        missing("v_headline_metrics", data)
        return
    h = h.iloc[0]
    rm, wr = data.get("recall_metrics"), data.get("waymo_results")
    base_auc = D.metric(rm, scorer="baseline", metric="pr_auc")
    base_weeks = D.metric(rm, scorer="baseline", metric="median_weeks_warning")
    min_vs_cv = None
    if wr is not None:
        w = wr[(wr["object_type"] == "all") & (wr["horizon_s"] == 8)
               & (wr["predictor"] == "model_best_of_6")]
        min_vs_cv = float(w["fde_improvement_vs_cv_pct"].iloc[0]) if len(w) else None

    st.markdown("**Months ahead: which car + part gets recalled next** (NHTSA complaints, test 2022-24)")
    c = st.columns(4)
    p, b = h["model_precision_at_20"], h["baseline_precision_at_20"]
    c[0].metric("Precision@20", pct(p), f"{100 * (p - b):+.1f} pts vs baseline {pct(b)}",
                help="Share of each month's top-20 watchlist recalled within 6 months.")
    c[1].metric("PR-AUC", num(h["model_pr_auc"], 3),
                None if base_auc is None else f"baseline {base_auc:.3f}", delta_color="off", delta_arrow="off")
    c[2].metric("Median warning", num(h["model_median_weeks_warning"], 1, " weeks"),
                None if base_weeks is None else f"baseline {base_weeks:.1f} (catches half as many)",
                delta_color="off", delta_arrow="off", help="Recall date minus the end of the first month on the list.")
    c[3].metric("Recalls caught", f"{num(h['model_recalls_caught_top20'], 0)} / "
                f"{num(h['model_recalls_caught_top100'], 0)}", "top 20 / top 100 list",
                delta_color="off", delta_arrow="off")

    valid, total, unique = citation_counts(data)
    st.markdown("**Investigation briefs** (Gemini agent, every citation checked)")
    c = st.columns(4)
    c[0].metric("Briefs passing every check", f"{int(h['briefs_fully_valid'])}/{int(h['briefs'])}")
    c[1].metric("Citations valid", f"{valid}/{total}",
                None if unique is None else f"{unique} unique complaints", delta_color="off", delta_arrow="off")

    st.markdown("**Seconds ahead: where road users go next** (Waymo Open Motion, error at 8 s)")
    c = st.columns(4)
    cv = h["waymo_cv_fde_8s_m"]
    c[0].metric("Constant velocity (baseline)", num(cv, 2, " m"))
    c[1].metric("Model, best guess", num(h["waymo_model_fde_8s_m"], 2, " m"),
                f"{-h['waymo_best_guess_vs_cv_pct']:.0f}% vs baseline", delta_color="inverse")
    c[2].metric("Model, best of 6 paths", num(h["waymo_model_min_fde_8s_m"], 2, " m"),
                None if min_vs_cv is None else f"{-min_vs_cv:.0f}% vs baseline", delta_color="inverse")

    st.graphviz_chart(_track_diagram("Track A: months ahead", "#d62728", [
        "NHTSA complaints\\n+ recalls", "Recall model\\n(LightGBM)", "Monthly watchlist\\n(top 20)",
        "Gemini agent\\nbrief + validated citations"]))
    st.graphviz_chart(_track_diagram("Track B: seconds ahead", "#1f77b4", [
        "Waymo scenes\\n(past 1 s + map)", "Trajectory model\\n(MLP, 6 paths)",
        "Paths 8 s ahead\\nvs constant velocity"]))
    if h.get("last_published_at") is not None and not pd.isna(h.get("last_published_at")):
        st.caption(f"Results published to BigQuery {pd.Timestamp(h['last_published_at']):%Y-%m-%d %H:%M} UTC.")


def tab_watchlist(data: D.Data) -> None:
    wb = data.get("v_watchlist_with_briefs")
    if wb is None or wb.empty:
        missing("v_watchlist_with_briefs", data)
        return
    months = {D.month_label(m): m for m in sorted(wb["month"].unique(), reverse=True)}
    label = st.selectbox("Month", list(months), key="watch_month")
    month = months[label]
    rows = wb[wb["month"] == month].sort_values("rank")
    brief_months = sorted({D.month_label(m) for m in wb.loc[wb["has_brief"], "month"]})
    st.caption(f"Top {len(rows)} car + parts for {D.month_label(month)}, using only complaints received "
               f"by the end of that month. Briefs written for: {', '.join(brief_months) or 'none'}.")
    st.dataframe(
        rows[["rank", "make", "model", "model_year", "component", "score", "top_reasons", "has_brief"]],
        hide_index=True, width="stretch", height=36 * (len(rows) + 1) + 3,
        column_config={
            "rank": st.column_config.NumberColumn("Rank", width="small"),
            "make": "Make",
            "model": st.column_config.TextColumn("Model", width="medium"),
            "model_year": st.column_config.NumberColumn("Year", format="%d", width="small"),
            "component": "Part",
            "score": st.column_config.ProgressColumn("Score", format="%.3f", min_value=0.0,
                                                     max_value=float(wb["score"].max())),
            "top_reasons": st.column_config.TextColumn("Why (SHAP)", width="large"),
            "has_brief": st.column_config.CheckboxColumn("Brief"),
        },
    )


def tab_brief(data: D.Data) -> None:
    wb = data.get("v_watchlist_with_briefs")
    if wb is None or wb.empty:
        missing("v_watchlist_with_briefs", data)
        return
    briefs = wb[wb["has_brief"]].sort_values(["month", "rank"], ascending=[False, True])
    if briefs.empty:
        st.info("No briefs have been published yet.")
        return
    labels = {f"{D.month_label(r['month'])} · #{r['rank']} {r['make']} {r['model']} {r['model_year']}"
              f" · {r['component']}": i for i, r in briefs.iterrows()}
    pick = st.selectbox("Watchlist row with a brief", list(labels), key="brief_pick")
    r = briefs.loc[labels[pick]]
    outcome = ("passed_after_retry" if r["passed_after_retry"] else
               "still_failing" if r["still_failing"] else "passed_first_try")
    ok = r["citations_valid"] == r["citations_total"]
    badges = [
        f":{'green' if ok else 'red'}-badge[citations {int(r['citations_valid'])}/{int(r['citations_total'])} valid]",
        f":{'red' if outcome == 'still_failing' else 'green'}-badge[{OUTCOME_LABELS[outcome]}]",
        f":gray-badge[{int(r['evidence_bullets'])} evidence bullets, {int(r['uncited_bullets'])} uncited]",
        f":blue-badge[model {r['model_used']}]",
        f":violet-badge[similar recalls: {r['similar_recall_search_scope']}]",
    ]
    st.markdown(" ".join(badges))
    st.caption(f"On the watchlist because: {r['top_reasons']}")
    with st.container(border=True):
        # display only: the brief's "##" section headings shown smaller; the text is unchanged
        st.markdown(re.sub(r"^## ", "##### ", r["brief_markdown"], flags=re.MULTILINE))
    st.caption("AI-generated by Gemini and not hand-edited. Every [ODINO n] was checked against the "
               "complaints for this car + part received before the watchlist cutoff.")
    cites = data.get("agent_citations")
    if cites is None:
        missing("agent_citations", data)
        return
    mine = cites[cites["brief_id"] == r["brief_id"]]
    with st.expander(f"Cited complaints ({len(mine)} unique ODINO numbers)"):
        st.dataframe(mine[["odino", "valid"]], hide_index=True, width="stretch",
                     column_config={"odino": "ODINO (NHTSA complaint number)",
                                    "valid": st.column_config.CheckboxColumn("Valid")})


def _ablation_section() -> str | None:
    path = D.REPORTS / "recall_results_stg.md"
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    start = text.find("## 7. Vehicle age vs. complaints")
    if start < 0:
        return None
    end = text.find("\n## ", start + 5)
    body = text[start:end if end > 0 else None].split("\n", 1)[1]
    return body.strip()


def _audit_sentences(audit: pd.DataFrame) -> list[str]:
    out = []
    la = audit[(audit["experiment"] == "label_audit") & (audit["metric"] == "lgbm_precision_at_20")]
    by = dict(zip(la["variant"], la["value"], strict=False))
    exact = next((v for k, v in by.items() if k.startswith("exact")), None)
    variant = next((v for k, v in by.items() if "variant" in k), None)
    late = next((v for k, v in by.items() if "6-12" in k), None)
    if exact is not None and variant is not None:
        out.append(f"**Label audit:** counting model-name variants (e.g. a CX-90 Plug-in Hybrid flag when "
                   f"the recall was filed under CX-90) moves precision@20 only from {pct(exact)} to "
                   f"{pct(variant)}, so the grading is fair."
                   + (f" Also counting recalls 6-12 months out lifts it to {pct(late)}: some misses are "
                      "early, not wrong." if late is not None else ""))
    inv = audit[audit["experiment"] == "investigations"]
    if len(inv):
        def val(model, split, metric):
            m = inv[(inv["variant"] == model) & (inv["split"] == split) & (inv["metric"] == metric)]
            return float(m["value"].iloc[0]) if len(m) else None
        base = next((v for v in inv["variant"].unique() if v.startswith("v1 (")), "v1")
        new = "v1 + investigations"
        out.append(
            f"**Open NHTSA investigations as a feature:** precision@20 rose from "
            f"{pct(val('v1', 'val', 'precision_at_20'))} to {pct(val(new, 'val', 'precision_at_20'))} on "
            f"validation and {pct(val(base, 'test', 'precision_at_20'))} to "
            f"{pct(val(new, 'test', 'precision_at_20'))} on test, but distinct recalls caught fell from "
            f"{pct(val(base, 'test', 'catch_rate_first_with_complaints'))} to "
            f"{pct(val(new, 'test', 'catch_rate_first_with_complaints'))} (the same investigated cars "
            "stay on the list), so the model was kept as is.")
    return out


def tab_results(data: D.Data) -> None:
    rm = data.get("recall_metrics")
    if rm is None or rm.empty:
        missing("recall_metrics", data)
        return
    topk = rm[rm["source_file"] == "recall_topk_stg.csv"].pivot_table(
        index=["top_k", "scorer"], columns="metric", values="value").reset_index()
    topk["scorer_label"] = topk["scorer"].map(SCORER_LABELS)
    st.markdown("##### Watchlist size: model vs baseline (test 2022-24)")
    c = st.columns(2)
    c[0].altair_chart(charts.topk_chart(topk, "precision", "precision"), width="content")
    c[1].altair_chart(charts.topk_chart(topk, "recalls_caught", "recalls caught"),
                      width="content")
    table = topk.assign(**{
        "precision": topk["precision"].map(pct),
        "% of recalls caught": topk["pct_recalls_caught"].map(pct),
        "recalls caught": topk["recalls_caught"].map(lambda v: num(v, 0)),
        "median weeks of warning": topk["median_weeks_warning"].map(lambda v: num(v, 1)),
    })[["top_k", "scorer_label", "precision", "recalls caught", "% of recalls caught",
        "median weeks of warning"]].rename(columns={"top_k": "top k", "scorer_label": "scorer"})
    st.dataframe(table, hide_index=True, width="stretch")
    first = D.metric(rm, scorer="lgbm", metric="first_recalls_with_complaints")
    if first:
        st.caption(f"% of recalls caught = share of the {first:,.0f} first recalls in 2022-24 with "
                   "complaints beforehand that were on the list in the 12 months before the recall.")

    c = st.columns(4)
    for col, (label, metric_name, digits) in zip(c, [("PR-AUC (test)", "pr_auc", 3),
                                                    ("Median weeks of warning", "median_weeks_warning", 1)],
                                                 strict=False):
        m, b = D.metric(rm, scorer="lgbm", metric=metric_name), D.metric(rm, scorer="baseline", metric=metric_name)
        col.metric(label, num(m, digits), None if b is None else f"baseline {b:.{digits}f}", delta_color="off", delta_arrow="off")
    vp = D.metric(rm, scorer="lgbm", metric="precision_at_20", split="val")
    vb = D.metric(rm, scorer="baseline", metric="precision_at_20", split="val")
    c[2].metric("Precision@20 (validation 2020-21)", pct(vp), None if vb is None else f"baseline {pct(vb)}",
                delta_color="off", delta_arrow="off")

    st.markdown("##### Vehicle age vs complaints")
    ablation = rm[rm["source_file"] == "recall_ablation_stg.csv"]
    if len(ablation):
        st.dataframe(ablation.pivot_table(index=["top_k", "scorer"], columns="metric", values="value")
                     .reset_index(), hide_index=True, width="stretch")
    elif (section := _ablation_section()) is not None:
        st.markdown(section)
        st.caption("From Justine's committed write-up, reports/recall_results_stg.md (section 7).")
    else:
        st.info("The age vs complaints ablation isn't published yet.")

    audit = data.get("recall_audit")
    st.markdown("##### Checking our own grading")
    if audit is None or audit.empty:
        missing("recall_audit", data)
    else:
        for sentence in _audit_sentences(audit):
            st.markdown(sentence)

    shown = [(f, cap) for f, cap in FIGURES if (D.REPORTS / f).exists()]
    if shown:
        st.markdown("##### Justine's charts")
        cols = st.columns(2)
        for i, (f, cap) in enumerate(shown):
            cols[i % 2].image(str(D.REPORTS / f), caption=cap, width="stretch")


def tab_trajectory(data: D.Data) -> None:
    ex = data.get("waymo_demo_examples")
    if ex is None or ex.empty:
        missing("waymo_demo_examples", data)
    else:
        types = ["all"] + [t for t in ("vehicle", "pedestrian", "cyclist") if (ex["object_type"] == t).any()]
        kind = st.radio("Agent type", types, horizontal=True, key="traj_type")
        pool = ex if kind == "all" else ex[ex["object_type"] == kind]
        # default first: agents that turn (end >= 5 m sideways), then by how much the model beats CV
        turn = pool["true_future"].map(lambda f: abs([p for p in f if p is not None][-1][1]) >= 5)
        pool = (pool.assign(turns=turn, gain=pool["fde_8s_cv_m"] - pool["fde_8s_best_m"])
                .sort_values(["turns", "gain"], ascending=[False, False]))
        labels = {f"{r['object_type']} · scene {r['scenario_id']} track {r['track_id']} · "
                  f"CV {r['fde_8s_cv_m']:.1f} m → model {r['fde_8s_best_m']:.1f} m": i
                  for i, r in pool.iterrows()}
        pick = st.selectbox(f"Agent ({len(pool)})", list(labels), key="traj_pick")
        r = pool.loc[labels[pick]]
        left, right = st.columns([2, 1])
        with left:
            st.altair_chart(charts.trajectory_chart(r), width="content")
        with right:
            cv = r["fde_8s_cv_m"]
            st.metric("Constant velocity, error at 8 s", num(cv, 2, " m"))
            st.metric("Model best guess", num(r["fde_8s_best_m"], 2, " m"),
                      f"{100 * (r['fde_8s_best_m'] / cv - 1):+.0f}%" if cv else None, delta_color="inverse")
            st.metric("Model best of 6", num(r["fde_8s_min_m"], 2, " m"),
                      f"{100 * (r['fde_8s_min_m'] / cv - 1):+.0f}%" if cv else None, delta_color="inverse")
            st.metric("Distance travelled in 8 s", num(r["travelled_m"], 1, " m"))
            conf = ", ".join(f"{c:.2f}" for c in r["model_confidences"])
            st.caption(f"Path confidences (bold = most confident): {conf}. Agent frame: the agent is at "
                       "(0, 0) facing right. Grey lines are lane centres within 60 m.")

    wr = data.get("waymo_results")
    st.markdown("##### All 8 s test agents (Waymo validation shard 0)")
    if wr is None or wr.empty:
        missing("waymo_results", data)
        return
    t = wr.pivot_table(index=["object_type", "horizon_s", "n_agents"], columns="predictor",
                       values="fde_m").reset_index()
    order = {"all": 0, "vehicle": 1, "pedestrian": 2, "cyclist": 3}
    t = t.sort_values(["object_type", "horizon_s"], key=lambda s: s.map(order) if s.name == "object_type" else s)
    t["best guess vs CV"] = (t["model_best_guess"] / t["constant_velocity"] - 1).map(signed_pct)
    t["best of 6 vs CV"] = (t["model_best_of_6"] / t["constant_velocity"] - 1).map(signed_pct)
    st.dataframe(
        t.rename(columns={"object_type": "type", "horizon_s": "horizon (s)", "n_agents": "agents",
                          "constant_velocity": "CV FDE (m)", "model_best_guess": "best guess FDE (m)",
                          "model_best_of_6": "best of 6 FDE (m)"}),
        hide_index=True, width="stretch", height=36 * (len(t) + 1) + 3,
        column_config={c: st.column_config.NumberColumn(format="%.2f")
                       for c in ("CV FDE (m)", "best guess FDE (m)", "best of 6 FDE (m)")})
    st.caption("FDE = distance between the predicted and true position at the horizon.")


def tab_about(data: D.Data) -> None:
    st.markdown("##### How it fits together")
    st.graphviz_chart("""
digraph {
  rankdir=LR; bgcolor="transparent"; compound=true;
  node [shape=box, style="rounded,filled", fillcolor="#f4f6f8", color="#c9ced6", fontname="Helvetica", fontsize=10];
  edge [color="#8a94a1"];
  src [label="NHTSA files\\ncomplaints, recalls,\\ninvestigations"]; raw [label="DuckDB raw.*"];
  stg [label="Cleaning\\nstg.complaints / stg.recalls"]; feat [label="monthly_features\\n+ labels"];
  model [label="LightGBM\\nrolling backtest"]; watch [label="main.watchlist\\n+ SHAP reasons"];
  agent [label="Gemini agent\\n+ citation validator"];
  wsrc [label="Waymo Open Motion\\nshards"]; wpq [label="tracks + map\\nParquet"];
  wmodel [label="MLP, 6 paths\\nvs constant velocity"];
  bq [label="BigQuery sandbox\\n7 tables + 2 views", fillcolor="#e8f0fe"]; demo [label="This demo", fillcolor="#e8f0fe"];
  src -> raw -> stg -> feat -> model -> watch -> agent -> bq;
  wsrc -> wpq -> wmodel -> bq; bq -> demo;
}""")
    st.caption("Every pull request runs CI (ruff + the offline test suite); `python run.py pipeline` "
               "rebuilds every step in order.")
    st.markdown("""
##### Team
| Person | Built |
| --- | --- |
| Angelina Yang | Raw data profiling; cleaned complaints and recalls with standardized make, model and part; demo |
| Justine Li-Wu | Recall labels and EDA, monthly features, LightGBM rolling backtest, watchlist with SHAP reasons, results and case studies; demo |
| Sean Gee | Data download and loading, contracts, Gemini brief agent and validator, label audit, Waymo baseline and model, BigQuery, CI, pipeline |
""")
    st.markdown("##### Honest limitations")
    rm, wr = data.get("recall_metrics"), data.get("waymo_results")
    lines = []
    caught20 = D.metric(rm, scorer="lgbm", metric="pct_recalls_caught", source_file="recall_topk_stg.csv", top_k=20)
    caught100 = D.metric(rm, scorer="lgbm", metric="pct_recalls_caught", source_file="recall_topk_stg.csv", top_k=100)
    if caught20 is not None:
        lines.append(f"- **Most recalls are not caught.** A top-20 list catches {pct(caught20)} of first "
                     f"recalls with complaints ({pct(caught100)} at top 100). It is a triage list for "
                     "investigators, not a recall detector.")
    misses = D.REPORTS / "recall_miss_reasons_stg.csv"
    if misses.exists():
        mr = pd.read_csv(misses)
        thin = mr[mr["reason"].str.contains("no complaints|1-2 complaints")]["recalls"].sum()
        lines.append(f"- **The limit is the data, not the model.** {thin:,} of {mr['recalls'].sum():,} missed "
                     f"recalls ({thin / mr['recalls'].sum():.0%}) had two or fewer complaints in the year "
                     "before: there is little for any model to see.")
    if wr is not None:
        def gain(t):
            w = wr[(wr["object_type"] == t) & (wr["horizon_s"] == 8) & (wr["predictor"] == "model_best_guess")]
            return float(w["fde_improvement_vs_cv_pct"].iloc[0]) if len(w) else None
        ped, cyc = gain("pedestrian"), gain("cyclist")
        if ped is not None:
            lines.append(f"- **Pedestrians and cyclists:** the model's best guess is {abs(ped):.0f}% "
                         f"{'worse' if ped < 0 else 'better'} than constant velocity for pedestrians and "
                         f"{abs(cyc):.0f}% {'worse' if cyc < 0 else 'better'} for cyclists at 8 s; the "
                         "gain comes from vehicles. Few training examples, and people change direction "
                         "abruptly.")
        n = wr[(wr["object_type"] == "all") & (wr["horizon_s"] == 8)]["n_agents"]
        if len(n):
            lines.append(f"- **Small Waymo test set:** {int(n.iloc[0]):,} agents from one validation shard.")
    lines.append("- **Briefs** are AI-generated summaries of what owners reported, checked for real, "
                 "in-scope citations; they are not a verdict on the vehicle.")
    st.markdown("\n".join(lines))
    with st.expander("Data sources"):
        st.markdown("""
- NHTSA [complaints, recalls, investigations](https://www.nhtsa.gov/nhtsa-datasets-and-apis): public domain.
- [Waymo Open Motion Dataset](https://waymo.com/open/about) v1.3.1: non-commercial license. Only summary
  results and 50 example paths are shared, never the raw data.
""")
        st.caption("Data per table: " + ", ".join(f"{n} ({s})" for n, s in data.source.items()))
    if st.button("Reload data"):
        load.clear()
        st.rerun()


# ---------------------------------------------------------------------------------------------

data = load()
title, badge = st.columns([4, 1])
title.title("Vehicle Safety Early Warning")
with badge:
    source_badge(data)

tabs = st.tabs(["Overview", "Recall watchlist", "Investigation brief", "Model results",
                "Trajectory", "About"])
for tab, render in zip(tabs, [tab_overview, tab_watchlist, tab_brief, tab_results, tab_trajectory,
                              tab_about], strict=True):
    with tab:
        render(data)
