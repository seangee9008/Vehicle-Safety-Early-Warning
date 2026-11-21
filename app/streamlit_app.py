"""Vehicle Safety Early Warning: the demo dashboard.

    python run.py demo                       # or: streamlit run app/streamlit_app.py
    VSE_DATA=local python run.py demo        # skip BigQuery, use local files (offline)

Numbers come from the data, in this order: BigQuery (src/bq_reader.py), the local DuckDB, then files
shipped with the app (the committed reports/, app/facts.json, app/agent_snapshot.json).
No AI calls at runtime: the briefs were written and checked beforehand.
"""

from __future__ import annotations

import json
import re
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app import agent_data, charts, motion, reports, theme
from app import briefs as B
from app import components as ui
from app import data as D

st.set_page_config(page_title="Vehicle Safety Early Warning", page_icon=":material/directions_car:",
                   layout="wide")

TABS = ["Overview", "Recall watchlist", "AI agent", "Model results", "Trajectory", "Data and limitations"]
TEAM = ["Sean Gee", "Justine Li-Wu", "Angelina Yang"]
FACTS_PATH = Path(__file__).resolve().parent / "facts.json"
MISS_LABELS = {"1-2 complaints in the 12 months before": "1 or 2 complaints in the year before",
               "no complaints in the 12 months before": "No complaints in the year before",
               "3+ complaints, never in the top 100": "3+ complaints, never in the top 100",
               "3+ complaints, best rank 21-100": "3+ complaints, ranked 21 to 100"}
VERDICTS = {"SUPPORT": (":green[:material/check_circle:]", "Supported"),
            "CONTEXT": (":blue[:material/info:]", "Context only"),
            "CONTRADICT": (":red[:material/cancel:]", "Contradicted")}
try:  # how many complaints the agent reads per brief (its tool's limit)
    from src.agent.brief import COMPLAINT_LIMIT
except Exception:  # noqa: BLE001 - the agent's packages aren't installed on Streamlit Cloud
    COMPLAINT_LIMIT = None


@st.cache_data(ttl=3600, show_spinner="Loading results…")
def load() -> D.Data:
    return D.load_all()


@st.cache_data
def load_facts() -> dict:
    return json.loads(FACTS_PATH.read_text(encoding="utf-8")) if FACTS_PATH.exists() else {}


@st.cache_data(ttl=3600, show_spinner=False)
def load_agent_details(briefs: pd.DataFrame, citations: pd.DataFrame | None, live: bool) -> tuple[dict, str]:
    return agent_data.load(briefs, citations, live=live)


# ---------------------------------------------------------------------------------------------
# formatting and lookups

def pct(x, digits: int = 1) -> str:
    return "–" if x is None or pd.isna(x) else f"{100 * x:.{digits}f}%"


def num(x, digits: int = 0, unit: str = "") -> str:
    return "–" if x is None or pd.isna(x) else f"{x:,.{digits}f}{unit}"


def millions(x) -> str:
    return "–" if not x else f"{x / 1e6:.2f}M"


def month_name(m) -> str:
    return pd.Timestamp(m).strftime("%b %Y")


def vehicle(row) -> str:
    return f"{str(row['make']).title()} {row['model']} {row['model_year']}"


def missing(name: str) -> None:
    st.info(f"`{name}` isn't available right now, so this part can't be shown.")


def muted(text: str) -> None:
    st.markdown(f'<div class="vse-muted">{text}</div>', unsafe_allow_html=True)


def topk(rm: pd.DataFrame | None, scorer: str, metric: str, k: int) -> float | None:
    return D.metric(rm, scorer=scorer, metric=metric, source_file="recall_topk_stg.csv", top_k=k)


def audit_value(audit, experiment: str, variant_prefix: str, metric: str, split: str = "test") -> float | None:
    if audit is None:
        return None
    a = audit[(audit["experiment"] == experiment) & (audit["metric"] == metric) & (audit["split"] == split)
              & audit["variant"].str.startswith(variant_prefix)]
    return float(a["value"].iloc[0]) if len(a) else None


def waymo(wr, predictor: str, object_type: str = "all", horizon: int = 8, col: str = "fde_m") -> float | None:
    if wr is None:
        return None
    w = wr[(wr["predictor"] == predictor) & (wr["object_type"] == object_type) & (wr["horizon_s"] == horizon)]
    return float(w[col].iloc[0]) if len(w) else None


def briefs_of(data: D.Data) -> pd.DataFrame:
    wb = data.get("v_watchlist_with_briefs")
    if wb is None:  # empty, but with the columns the callers read
        return pd.DataFrame(columns=["month", "rank", "make", "model", "model_year", "component", "score",
                                     "top_reasons", "has_brief", *D.BRIEF_COLUMNS])
    return wb[wb["has_brief"]].sort_values(["month", "rank"], ascending=[False, True])


def open_agent(brief_id: str) -> None:
    st.session_state["agent_car"] = brief_id
    st.session_state["main_tab"] = "AI agent"


# ---------------------------------------------------------------------------------------------
# Overview

def tab_overview(data: D.Data, facts: dict) -> None:
    h = data.get("v_headline_metrics")
    rm, wr, audit = data.get("recall_metrics"), data.get("waymo_results"), data.get("recall_audit")
    cf, rf, w = facts.get("complaints", {}), facts.get("recalls", {}), facts.get("waymo", {})

    st.markdown("### Why this project")
    cards = [
        ("why_problem", "report", "The problem",
         f"Safety defects build up in owner complaints long before a recall. Owners file about "
         f"{num(cf.get('per_year'))} complaints a year; NHTSA announces about {num(rf.get('per_year'))} recall "
         "campaigns a year." if cf and rf else "Safety defects build up in owner complaints long before a recall."),
        ("why_goal", "flag", "The goal",
         "Flag risky car parts months before a recall. Predict where road users move seconds ahead."),
        ("why_approach", "account_tree", "Our approach",
         ("Rank every car part monthly and explain each flag with checked evidence. Learn several possible "
         "paths for each road user.")),
    ]
    cols = st.columns(3)
    for col, (key, ic, title, body) in zip(cols, cards, strict=True):
        with col, st.container(key=key):
            st.markdown(f"**:material/{ic}: {title}**")
            st.markdown(body)

    st.markdown("### Results")
    if h is None or h.empty:
        missing("v_headline_metrics")
        return
    h = h.iloc[0]
    hit, base_hit = h["model_precision_at_20"], h["baseline_precision_at_20"]
    base_rate = audit_value(audit, "label_audit", "exact", "base_rate")
    weeks, base_weeks = h["model_median_weeks_warning"], D.metric(rm, scorer="baseline", metric="median_weeks_warning")
    caught, base_caught = topk(rm, "lgbm", "recalls_caught", 20), topk(rm, "baseline", "recalls_caught", 20)
    briefs = briefs_of(data)
    valid, total = int(briefs["citations_valid"].sum()), int(briefs["citations_total"].sum())
    cites = data.get("agent_citations")
    unique = cites["odino"].nunique() if cites is not None else None
    cv, best, closest = h["waymo_cv_fde_8s_m"], h["waymo_model_fde_8s_m"], h["waymo_model_min_fde_8s_m"]

    c = st.columns(4)
    with c[0]:
        random = f" · random: about {20 * base_rate:.2f} of 20" if base_rate else ""
        ui.kpi_card("Hit rate", pct(hit), f"About {20 * hit:.1f} of 20 flagged cars a month were recalled within "
                    "6 months.", f"vs {pct(base_hit)} baseline{random}", theme.TRACK_A,
                    ui.cars_of_twenty(20 * hit), help=theme.BASELINE_TOOLTIP, tall=True)
    with c[1]:
        bars = [("Model", weeks, theme.TRACK_A)] + ([("Baseline", base_weeks, theme.BASELINE)] if base_weeks else [])
        ui.kpi_card("Lead time", num(weeks, 0, " weeks"), "Typical warning before the recall.",
                    f"Baseline catches {num(base_caught)} recalls to the model's {num(caught)}" if base_caught else "",
                    theme.TRACK_A, ui.compare_bars(bars, "wk"), tall=True)
    with c[2]:
        ui.kpi_card("Evidence check", f"{valid} of {total}", "Every complaint the AI quoted is real and "
                    "dated before the flag.", f"{unique} unique complaints across {len(briefs)} briefs"
                    if unique is not None else "", theme.TRACK_A,
                    ui.ring(valid / total if total else 0, f"{valid}/{total}", theme.GOOD, size=74, stroke=9), tall=True)
    with c[3]:
        ui.kpi_card("Position error at 8 s", num(closest, 1, " m"), "Closest of the model's 6 guesses.",
                    f"Top guess {num(best, 1, ' m')}", theme.TRACK_B,
                    ui.compare_bars([("Going straight", cv, theme.BASELINE), ("Closest of 6", closest, theme.TRACK_B)],
                                    "m"), tall=True)

    st.markdown("### How it works")
    wb, ex = data.get("v_watchlist_with_briefs"), data.get("waymo_demo_examples")
    left, right = st.columns(2, gap="medium")
    with left, st.container(key="panel_a"):
        st.markdown("**:material/directions_car: Months ahead: vehicle recalls**")
        per_month = int(wb.groupby("month").size().max()) if wb is not None else None
        months = wb["month"].nunique() if wb is not None else None
        model_facts = facts.get("model", {})
        ui.steps([
            ("Complaints in", millions(cf.get("complaints"))),
            ("Model ranks every car part monthly",
             f"{num(model_facts.get('car_parts_ranked_per_month', 0) / 1000)}k a month" if model_facts else "–"),
            ("Top 20 watchlist", f"{per_month} a month, {months} months" if per_month else "–"),
            ("AI writes an evidence brief", f"{len(briefs)} briefs, {total} citations"),
        ], theme.TRACK_A)
        ui.headline(pct(hit), f"of flagged cars recalled within 6 months (baseline {pct(base_hit)})", theme.TRACK_A)
    with right, st.container(key="panel_b"):
        st.markdown("**:material/add_road: Seconds ahead: road users**")
        first = ex.iloc[0] if ex is not None and len(ex) else None
        ui.steps([
            ("Driving scenes", f"{w.get('train_scenes', '–')} train, {w.get('test_scenes', '–')} test"),
            ("Past 1 second", f"{len(first['past_path'])} positions" if first is not None else "–"),
            ("Model draws 6 possible paths", f"{len(first['model_paths'])} paths, "
             f"{len(first['true_future']) / 10:.0f} s ahead" if first is not None else "–"),
            ("Compare to what happened", f"{num(waymo(wr, 'model_best_guess', col='n_agents'))} road users"),
        ], theme.TRACK_B)
        ui.headline(num(closest, 1, " m"), f"closest of 6 after 8 s (going straight: {num(cv, 1, ' m')})",
                    theme.TRACK_B)

    st.markdown("#### Data at a glance")
    years = f"{cf['first'][:4]}–{cf['last'][:4]}" if cf else "–"
    ui.glance([
        (millions(cf.get("complaints")), "vehicle complaints to NHTSA"),
        (num(rf.get("campaigns")), "recall campaigns"),
        (years, "years covered"),
        (num((w.get("train_scenes") or 0) + (w.get("test_scenes") or 0)) if w else "–", "driving scenes"),
    ])

    st.markdown("### How we built it")
    decision_tree(data, facts)


def decision_tree(data: D.Data, facts: dict) -> None:
    rm = data.get("recall_metrics")
    cf = facts.get("complaints", {})
    hit, cov = topk(rm, "lgbm", "precision", 20), topk(rm, "lgbm", "pct_recalls_caught", 20)
    years = f"Covers {cf['first'][:4]}–{cf['last'][:4]}" if cf else "Covers many years"
    st.markdown('<div class="vse-root">Spot safety problems early</div>', unsafe_allow_html=True)
    track_a = [
        ("Public NHTSA complaints", "Free, and every make reports there", ["Free, every make", years],
         ["Voluntary, noisy reports"]),
        ("Time-based split, no post-recall complaints", "No peeking at the future",
         ["Honest results", "Mirrors real use"], ["Less training data"]),
        ("Tree-based model on monthly counts", "Fast and explainable",
         ["Fast on tables", "Explainable flags"], ["Limited by what complaints say"]),
        ("Top 20 watchlist per month", "A list an investigator can act on",
         ["Short, actionable list", f"About {20 * hit:.1f} real hits a month" if hit else "Real hits each month"],
         [f"Catches {pct(cov)} of recalls" if cov else "Catches a small share of recalls"]),
        ("AI briefs with checked citations", "Explains each flag",
         ["Explains each flag", "Every quote verified"], ["Free-tier rate limits"]),
    ]
    track_b = [
        ("Waymo Open Motion scenes", "Real, labeled driving", ["Real driving data", "Labeled road users"],
         ["Non-commercial license", "Large files"]),
        ("Simple model, 6 possible paths", "Covers several futures",
         ["Trains in seconds", "Covers several futures"], ["Doesn't see lanes or others"]),
        ("Choose on held-out data, test once", "Numbers you can trust",
         ["Honest numbers", "No tuning on test"], ["Less data to train"]),
    ]
    left, right = st.columns(2, gap="large")
    for col, title, nodes, color, key in [(left, "Track A: vehicle recalls", track_a, theme.TRACK_A, "a"),
                                          (right, "Track B: road users", track_b, theme.TRACK_B, "b")]:
        with col:
            st.markdown(f'<div class="vse-branch" style="--accent:{color}">{title}</div>', unsafe_allow_html=True)
            for i, (choice, why, pros, cons) in enumerate(nodes):
                if i:
                    st.markdown('<div class="vse-link"></div>', unsafe_allow_html=True)
                ui.tree_node(choice, why, pros, cons, color, f"{key}{i}")
    st.markdown('<div class="vse-branch" style="--accent:#A9B2BC">Shared</div>', unsafe_allow_html=True)
    ui.tree_node("Free stack: DuckDB, BigQuery sandbox, GitHub CI", "Costs nothing, runs anywhere",
                 ["Costs $0", "Reproducible runs"], ["Storage limits"], theme.MUTED, "shared")


# ---------------------------------------------------------------------------------------------
# Recall watchlist

def tab_watchlist(data: D.Data) -> None:
    wb = data.get("v_watchlist_with_briefs")
    if wb is None or wb.empty:
        missing("v_watchlist_with_briefs")
        return
    months = {month_name(m): m for m in sorted(wb["month"].unique(), reverse=True)}
    c1, c2 = st.columns([1, 3])
    label = c1.selectbox("Month", list(months), key="watch_month")
    rows = wb[wb["month"] == months[label]].sort_values("rank").reset_index(drop=True)
    with c2:
        st.markdown(f'<div class="vse-muted" style="padding-top:2.2rem">The {len(rows)} car parts most likely to be '
                    f'recalled in the 6 months after {label}. The ring shows each score against the month\'s top '
                    "score.</div>", unsafe_allow_html=True)
    widths = [0.45, 2.4, 1.5, 0.8, 3.1, 1.4]
    head = st.columns(widths)
    for col, name in zip(head, ["#", "Vehicle", "Part", "Risk", "Why flagged", "Brief"], strict=True):
        col.markdown(f'<div class="vse-th">{name}</div>', unsafe_allow_html=True)
    top = float(rows["score"].max()) or 1.0
    for i, r in rows.iterrows():
        with st.container(key=f"wrow_{i}"):
            c = st.columns(widths, vertical_alignment="center")
            c[0].markdown(f"**{r['rank']}**")
            c[1].markdown(f"{ui.make_chip(r['make'])} **{escape(str(r['model']))}** {r['model_year']}",
                          unsafe_allow_html=True)
            c[2].markdown(ui.part_chip(r["component"]), unsafe_allow_html=True)
            c[3].markdown(ui.ring(r["score"] / top, f"{r['score']:.2f}"), unsafe_allow_html=True)
            c[4].markdown(" ".join(ui.tag_chip(t) for t in ui.reason_tags(r["top_reasons"])), unsafe_allow_html=True)
            if r["has_brief"]:
                c[5].button("Brief ready", key=f"wopen_{i}", on_click=open_agent, args=(r["brief_id"],),
                            icon=":material/smart_toy:", type="primary")


# ---------------------------------------------------------------------------------------------
# AI agent

def tab_agent(data: D.Data) -> None:
    briefs = briefs_of(data)
    if briefs.empty:
        missing("v_watchlist_with_briefs") if data.get("v_watchlist_with_briefs") is None else \
            st.info("No briefs have been published yet.")
        return
    cites = data.get("agent_citations")
    details, details_source = load_agent_details(
        briefs.reset_index(drop=True), None if cites is None else cites.reset_index(drop=True), live=True)
    pick = st.session_state.get("agent_car")
    if pick in set(briefs["brief_id"]):
        agent_detail(data, briefs[briefs["brief_id"] == pick].iloc[0], details.get(pick, {}), details_source)
        return
    months = {month_name(m): m for m in sorted(briefs["month"].unique(), reverse=True)}
    label = st.selectbox("Watchlist month", list(months), key="agent_month") if len(months) > 1 else next(iter(months))
    cars = briefs[briefs["month"] == months[label]].head(10)
    muted(f"The AI agent wrote a brief for each of these {len(cars)} cars on the {label} watchlist. Pick one to see "
          "how it worked.")
    wb = data.get("v_watchlist_with_briefs")
    top = float(wb.loc[wb["month"] == months[label], "score"].max())
    for start in range(0, len(cars), 5):
        cols = st.columns(5)
        for col, (_, r) in zip(cols, cars.iloc[start:start + 5].iterrows(), strict=False):
            with col, st.container(key=f"car_{r['brief_id']}"):
                st.markdown(ui.car_svg(theme.part_color(r["component"]), 128), unsafe_allow_html=True)
                st.markdown(f"{ui.make_chip(r['make'])}<br><b>{escape(str(r['model']))}</b> {r['model_year']}",
                            unsafe_allow_html=True)
                st.markdown(ui.part_chip(r["component"]), unsafe_allow_html=True)
                ok = r["citations_valid"] == r["citations_total"]
                score_ring = ui.ring(r["score"] / top, f"{r['score']:.2f}")
                checked = ui.chip(f"{int(r['citations_valid'])}/{int(r['citations_total'])} checked", theme.BG,
                                  theme.GOOD if ok else theme.TRACK_A, "verified")
                st.markdown(f'<div style="display:flex;align-items:center;gap:10px">{score_ring}{checked}</div>',
                            unsafe_allow_html=True)
                st.button("Open", key=f"agent_open_{r['brief_id']}", on_click=open_agent, args=(r["brief_id"],),
                          icon=":material/arrow_forward:", width="stretch")


def agent_detail(data: D.Data, r: pd.Series, detail: dict, source: str) -> None:
    st.button("Back to all cars", key="agent_back", icon=":material/arrow_back:",
              on_click=lambda: st.session_state.update(agent_car=None))
    color = theme.part_color(r["component"])
    wb = data.get("v_watchlist_with_briefs")
    n_month = int((wb["month"] == r["month"]).sum()) if wb is not None else 20
    st.markdown(f'<div class="vse-head" style="--accent:{color}"><div style="display:flex;gap:18px;align-items:center">'
                f'{ui.car_svg(color, 110)}<div><div class="v">{escape(vehicle(r))}</div>'
                f'<div class="s">{ui.part_chip(r["component"])} {month_name(r["month"])} watchlist · rank '
                f'{r["rank"]} of {n_month}</div></div></div></div>', unsafe_allow_html=True)
    sec = B.sections(r["brief_markdown"])

    st.markdown("#### How the agent worked")
    on_file = detail.get("complaints_on_file")
    read = f"{min(on_file, COMPLAINT_LIMIT)} of {on_file}" if on_file is not None and COMPLAINT_LIMIT else \
        (f"{on_file} on file" if on_file is not None else "–")
    listed, none_found = B.similar_recalls(sec.get("Similar past recalls", ""))
    scope = {"same_make": "same make", "all_makes": "all makes"}.get(r["similar_recall_search_scope"], "all makes")
    outcome = ("After one retry" if r["passed_after_retry"] else "Still failing" if r["still_failing"]
               else "First try")
    ui.stepper([
        ("menu_book", "Read complaints", read),
        ("history", f"Searched past recalls ({scope})", "None relevant" if none_found else f"{listed} found"),
        ("edit_note", "Wrote the brief", f"{int(r['evidence_bullets'])} evidence points"),
        ("fact_check", "Checked citations", f"{int(r['citations_valid'])}/{int(r['citations_total'])}"),
        ("verified", "Passed all checks", outcome),
    ], theme.TRACK_A)

    left, right = st.columns([1, 1], gap="medium")
    with left:
        st.markdown("#### Severity")
        sev = B.severity(sec.get("Severity", ""))
        cs = st.columns(4)
        for col, (key, ic) in zip(cs, [("crashes", "car_crash"), ("fires", "local_fire_department"),
                                       ("injuries", "personal_injury"), ("deaths", "heart_broken")], strict=True):
            col.markdown(ui.counter(ic, sev[key], key.capitalize(), theme.ALERT), unsafe_allow_html=True)
        st.markdown("#### What drivers report")
        st.markdown(" ".join(ui.chip(c, theme.INK, theme.BORDER, "report") for c in
                             B.symptom_chips(sec.get("What drivers report", ""))), unsafe_allow_html=True)
        with st.expander("Full description"):
            st.markdown(sec.get("What drivers report", ""))
    with right:
        st.markdown("#### Complaints before the flag")
        if detail.get("monthly") is not None:
            st.plotly_chart(charts.complaints_per_month(detail["monthly"], pd.Timestamp(r["month"]), color),
                            config=charts.CONFIG, theme=None, key=f"cpm_{r['brief_id']}")
        else:
            muted("Monthly complaint counts aren't available here.")

    st.markdown("#### Evidence")
    cited = detail.get("cited", {})
    for j, bullet in enumerate(B.bullets(sec.get("Evidence", ""))):
        with st.container(border=True):
            st.markdown(B.strip_citations(bullet))
            with st.container(horizontal=True):
                for odino in B.citation_ids(bullet):
                    with st.popover(f"ODINO {odino}", icon=":material/description:", key=f"pop_{r['brief_id']}_{j}_{odino}"):
                        info = cited.get(odino)
                        if info:
                            st.markdown(f"**Received {info['date']}**")
                            st.markdown(f"“{info['excerpt']}”")
                        else:
                            st.markdown("Complaint text isn't available here.")
                        st.caption("NHTSA complaint ID; personal details removed.")

    st.markdown("#### Flag reasons check")
    checks = B.reason_checks(sec.get("Flag reasons check", ""))
    if checks:
        for reason, verdict, why in checks:
            ic, word = VERDICTS[verdict]
            tag = ui.reason_tag(reason)
            why = re.sub(r"^\(|\)\.?$", "", why.strip()).strip()
            why = why[:1].upper() + why[1:]
            st.markdown(f"{ic} **{word}** · {tag or reason}  \n<span class='vse-muted'>{escape(why)}</span>",
                        unsafe_allow_html=True)
    else:
        muted("This brief was written by an earlier version of the agent, before it checked the flag reasons.")

    st.markdown("#### Suggested next step")
    steps = [B.strip_citations(s) for s in B.bullets(sec.get("Suggested next step", ""))]
    steps = [s.removeprefix("Next step:").strip() for s in steps]
    st.markdown('<div class="vse-next">' + "".join(
        f'<div style="margin:4px 0">{ui.icon("arrow_right", theme.TRACK_A)} {escape(s)}</div>' for s in steps)
        + "</div>", unsafe_allow_html=True)
    muted(f"Written by an AI model and not edited by hand; every quoted complaint was checked automatically. "
          f"Complaint details: {'local database' if source == 'local' else 'saved snapshot'}.")


# ---------------------------------------------------------------------------------------------
# Model results (tiles + detail views)

def results_tiles(data: D.Data) -> list[dict]:
    rm, audit = data.get("recall_metrics"), data.get("recall_audit")
    ks = [20, 50, 100]
    hit = [topk(rm, "lgbm", "precision", k) for k in ks]
    base_hit = [topk(rm, "baseline", "precision", k) for k in ks]
    cov = [topk(rm, "lgbm", "pct_recalls_caught", k) for k in ks]
    base_cov = [topk(rm, "baseline", "pct_recalls_caught", k) for k in ks]
    first = D.metric(rm, scorer="lgbm", metric="first_recalls_with_complaints")
    weeks = D.metric(rm, scorer="lgbm", metric="median_weeks_warning")
    base_weeks = D.metric(rm, scorer="baseline", metric="median_weeks_warning")
    ov, abl, misses, late = reports.overlap(20), reports.ablation(), reports.miss_reasons(), reports.label_audit_misses()
    exact = audit_value(audit, "label_audit", "exact", "lgbm_precision_at_20")
    variant = audit_value(audit, "label_audit", "+ model-name", "lgbm_precision_at_20")
    inv_hit = (audit_value(audit, "investigations", "v1 (", "precision_at_20"),
               audit_value(audit, "investigations", "v1 + investigations", "precision_at_20"))
    inv_cov = (audit_value(audit, "investigations", "v1 (", "catch_rate_first_with_complaints"),
               audit_value(audit, "investigations", "v1 + investigations", "catch_rate_first_with_complaints"))
    tiles = []
    if all(v is not None for v in hit + base_hit):
        tiles.append({
            "key": "hit", "title": "Hit rate", "value": pct(hit[0]),
            "one": f"About {20 * hit[0]:.1f} of 20 flagged cars a month were recalled.",
            "preview": charts.mini_pair(hit, base_hit),
            "draw": lambda: st.plotly_chart(charts.model_vs_baseline(ks, hit, base_hit, ytitle="Recalled within 6 months"),
                                            config=charts.CONFIG, theme=None),
            "text": ["**What it measures.** Of the cars flagged each month, the share recalled within 6 months.",
                     (f"**What we found.** The model beats the baseline at every list size: {pct(hit[0])} vs "
                     f"{pct(base_hit[0])} at 20 cars a month."),
                     ("**Why it matters.** Investigators only open a few cases a month, so the top of the list "
                     "must be right more often.")]})
    if all(v is not None for v in cov + base_cov):
        tiles.append({
            "key": "coverage", "title": "Coverage", "value": pct(cov[0]),
            "one": f"Recalls flagged ahead of time; {pct(cov[2])} with 100 flags.",
            "preview": charts.mini_pair(cov, base_cov),
            "draw": lambda: st.plotly_chart(charts.model_vs_baseline(ks, cov, base_cov,
                                            ytitle="Recalls flagged ahead of time"), config=charts.CONFIG, theme=None),
            "text": [(f"**What it measures.** Of {num(first)} first recalls with complaints beforehand, the share "
                     "on the list in the year before."),
                     f"**What we found.** About {pct(cov[0], 0)} with 20 flags a month, {pct(cov[2])} with 100.",
                     "**Why it matters.** The list helps prioritize; most recalls are never flagged."]})
    if ov is not None and weeks is not None:
        groups = ["Caught by both", "Only the model", "Only the baseline"]
        model_w = [ov["both_median_weeks_lgbm"], ov["only_lgbm_median_weeks"], None]
        base_w = [ov["both_median_weeks_baseline"], None, ov["only_baseline_median_weeks"]]
        tiles.append({
            "key": "lead", "title": "Lead time", "value": num(weeks, 0, " weeks"),
            "one": f"Typical warning. The baseline's {num(base_weeks)} weeks come from fewer recalls.",
            "preview": charts.mini_bars([weeks, base_weeks], [theme.TRACK_A, theme.BASELINE]),
            "draw": lambda: st.plotly_chart(charts.lead_time_groups(groups, model_w, base_w, [
                f"{int(ov['caught_by_both'])} recalls", f"{int(ov['only_lgbm'])} recalls",
                f"{int(ov['only_baseline'])} recalls"]), config=charts.CONFIG, theme=None),
            "text": ["**What it measures.** Weeks from the first flag to the recall.",
                     (f"**What we found.** They catch different recalls. On the {int(ov['caught_by_both'])} both "
                     f"catch, the model flags earlier ({num(ov['both_median_weeks_lgbm'])} vs "
                     f"{num(ov['both_median_weeks_baseline'])} weeks)."),
                     (f"**Why it matters.** The {int(ov['only_lgbm'])} recalls only the model catches are new cars "
                     "with fast-rising complaints.")]})
    if abl is not None:
        rows = {r["model"].lower(): r for _, r in abl.iterrows()}
        age = next((r for k, r in rows.items() if k.startswith("age")), None)
        cmp_ = next((r for k, r in rows.items() if k.startswith("complaints")), None)
        both = next((r for k, r in rows.items() if "+" in k), None)
        base = next((r for k, r in rows.items() if k.startswith("baseline")), None)
        if all(x is not None for x in (age, cmp_, both, base)):
            vals = [base["hit_rate_20"], age["hit_rate_20"], cmp_["hit_rate_20"], both["hit_rate_20"]]
            colors = [theme.BASELINE, "#9C7A3C", "#9C7A3C", theme.TRACK_A]
            gain = both["hit_rate_20"] - max(age["hit_rate_20"], cmp_["hit_rate_20"])
            tiles.append({
                "key": "drivers", "title": "What drives the score", "value": f"+{100 * gain:.1f} pts",
                "one": f"Age alone {pct(age['hit_rate_20'])}, complaints alone {pct(cmp_['hit_rate_20'])}.",
                "preview": charts.mini_bars(vals, colors),
                "draw": lambda: st.plotly_chart(charts.bars(["Baseline", "Car age only", "Complaints only",
                                                             "Both (model)"], vals, colors,
                                                            ytitle="Hit rate, 20 flags a month"), config=charts.CONFIG, theme=None),
                "text": ["**What it measures.** The model trained on car age only, complaints only, and both.",
                         "**What we found.** Neither alone is enough. Age picks the new cars; complaints rank them.",
                         "**Why it matters.** Complaints add most where new models start getting them."]})
    if misses is not None and late is not None and exact is not None and variant is not None:
        late_share = late[late["match"].str.contains("6-12")]["share"]
        late_share = float(late_share.iloc[0]) if len(late_share) else None
        thin = misses[~misses["reason"].str.startswith("3+")]["recalls"].sum() / misses["recalls"].sum()
        miss_colors = [theme.TRACK_A, "#C7883A", "#8C6A3A", "#5A4A33"]
        tiles.append({
            "key": "fair", "title": "Are we grading fairly?", "value": f"+{100 * (variant - exact):.0f} pts",
            "one": f"Looser name matching adds little; {pct(late_share, 0)} of misses came 6–12 months later.",
            "preview": charts.mini_donut(list(misses["recalls"]), miss_colors),
            "draw": lambda: st.plotly_chart(charts.donut([MISS_LABELS.get(r, r) for r in misses["reason"]],
                                                         list(misses["recalls"]), miss_colors), config=charts.CONFIG, theme=None),
            "text": ["**What it measures.** Whether misses are hits under a looser match, and why the rest were missed.",
                     (f"**What we found.** Looser matching moves the hit rate only from {pct(exact)} to {pct(variant)} "
                     f"(on a rerun). {pct(late_share, 0)} of misses were recalled 6–12 months later: early, not wrong."),
                     (f"**Why it matters.** {thin:.0%} of missed recalls had two or fewer complaints in the year "
                     "before. The limit is the data.")]})
    if all(v is not None for v in inv_hit + inv_cov):
        tiles.append({
            "key": "signals", "title": "Did more signals help?", "value": f"{100 * (inv_cov[1] - inv_cov[0]):+.1f} pts",
            "one": "Open NHTSA investigations raised the hit rate but lowered coverage.",
            "preview": charts.mini_pair([inv_hit[0], inv_cov[0]], [inv_hit[1], inv_cov[1]]),
            "draw": lambda: st.plotly_chart(charts.paired(
                ["Hit rate", "Coverage"], [inv_hit[0], inv_cov[0]], [inv_hit[1], inv_cov[1]],
                ("Original model", "With open investigations"), (theme.TRACK_A, "#9C7A3C"),
                ytitle="Share, 20 flags a month"), config=charts.CONFIG, theme=None),
            "text": ["**What it measures.** The model plus one signal: an open NHTSA investigation on the car.",
                     (f"**What we found.** Hit rate rose from {pct(inv_hit[0])} to {pct(inv_hit[1])}; coverage fell "
                     f"from {pct(inv_cov[0])} to {pct(inv_cov[1])}. The same cars stay on the list."),
                     "**Why it matters.** Those cars are already being investigated. We kept the original model."]})
    return tiles


def show_tiles(tiles: list[dict], page: str, color: str) -> None:
    for start in range(0, len(tiles), 3):
        cols = st.columns(3)
        for col, t in zip(cols, tiles[start:start + 3], strict=False):
            with col:
                ui.tile(t["title"], t["value"], t["one"], color, t["key"], page,
                        preview=lambda t=t: st.plotly_chart(t["preview"], config=charts.CONFIG, theme=None,
                                                            key=f"mini_{page}_{t['key']}"))


def tab_results(data: D.Data) -> None:
    if data.get("recall_metrics") is None:
        missing("recall_metrics")
        return
    tiles = results_tiles(data)
    match = next((t for t in tiles if t["key"] == st.session_state.get(ui.detail_key("results"))), None)
    baseline = theme.BASELINE_TOOLTIP[0].lower() + theme.BASELINE_TOOLTIP[1:]
    if match is not None:
        ui.detail_view("results", match["title"], match["draw"], match["text"])
        muted(f"Model in amber, baseline in gray. Baseline: {baseline}")
        return
    muted(f"Test years, 20 cars flagged a month unless noted. Model in amber; baseline in gray ({baseline})")
    show_tiles(tiles, "results", theme.TRACK_A)


# ---------------------------------------------------------------------------------------------
# Trajectory

def traj_tiles(wr: pd.DataFrame) -> list[dict]:
    cv, best, closest = waymo(wr, "constant_velocity"), waymo(wr, "model_best_guess"), waymo(wr, "model_best_of_6")
    ped_cv, ped_best = waymo(wr, "constant_velocity", "pedestrian"), waymo(wr, "model_best_guess", "pedestrian")
    cyc_cv, cyc_best = waymo(wr, "constant_velocity", "cyclist"), waymo(wr, "model_best_guess", "cyclist")
    veh_cv, veh_best = waymo(wr, "constant_velocity", "vehicle"), waymo(wr, "model_best_guess", "vehicle")
    ped_n = waymo(wr, "model_best_guess", "pedestrian", col="n_agents")
    cyc_n = waymo(wr, "model_best_guess", "cyclist", col="n_agents")
    by_h = {p: wr[(wr["object_type"] == "all") & (wr["predictor"] == p)].sort_values("horizon_s")["fde_m"].tolist()
            for p in charts.PREDICTORS}
    horizons = sorted(wr["horizon_s"].unique())
    tiles = [{
        "key": "horizon", "title": "Error at 3, 5 and 8 seconds", "value": num(best, 1, " m"),
        "one": f"Top guess after 8 s, vs {num(cv, 1, ' m')} going straight.",
        "preview": charts.mini_lines(horizons, [(by_h["constant_velocity"], theme.BASELINE, "dash"),
                                                (by_h["model_best_guess"], theme.TRACK_B, "solid")]),
        "draw": lambda: st.plotly_chart(charts.horizon_lines(wr), config=charts.CONFIG, theme=None),
        "text": ["**What it measures.** How far each prediction lands from where the road user really was.",
                 (f"**What we found.** After 8 s: top guess {num(best, 1)} m, closest of 6 {num(closest, 1)} m, going "
                 f"straight {num(cv, 1)} m."),
                 "**Why it matters.** A few seconds of warning is what a self-driving car plans with."]}]
    if veh_cv and veh_best:
        tiles.append({
            "key": "types", "title": "By road user type", "value": f"{100 * (veh_best / veh_cv - 1):+.0f}%",
            "one": "Change in error for cars after 8 s, top guess vs going straight.",
            "preview": charts.mini_bars([veh_cv, veh_best, ped_cv or 0, ped_best or 0, cyc_cv or 0, cyc_best or 0],
                                        [theme.BASELINE, theme.TRACK_B] * 3),
            "draw": lambda: st.plotly_chart(charts.by_road_user(wr), config=charts.CONFIG, theme=None),
            "text": ["**What it measures.** Error after 8 seconds for each kind of road user.",
                     (f"**What we found.** Cars gain most ({num(veh_cv, 1)} m to {num(veh_best, 1)} m). For "
                     "pedestrians and cyclists the top guess matches going straight."),
                     "**Why it matters.** Most of the gain comes from cars, which dominate the data."]})
    if ped_cv and ped_best:
        tiles.append({
            "key": "struggles", "title": "Where it struggles", "value": f"{100 * (ped_best / ped_cv - 1):+.0f}%",
            "one": "Pedestrians: the top guess is no better than going straight.",
            "preview": charts.mini_bars([ped_cv, ped_best], [theme.BASELINE, theme.TRACK_B]),
            "draw": lambda: st.plotly_chart(charts.horizon_lines(wr, "pedestrian"), config=charts.CONFIG, theme=None),
            "text": [(f"**What we found.** Pedestrians after 8 s: top guess {num(ped_best, 2)} m vs {num(ped_cv, 2)} m "
                     f"going straight. Cyclists: {num(cyc_best, 2)} m vs {num(cyc_cv, 2)} m."),
                     (f"**Why.** People stop and turn suddenly, and the test set has only {num(ped_n)} pedestrians and "
                     f"{num(cyc_n)} cyclists.")]})
    return tiles


def pick_example(example_id: str) -> None:
    st.session_state["traj_example"] = example_id


def tab_trajectory(data: D.Data) -> None:
    ex, wr = data.get("waymo_demo_examples"), data.get("waymo_results")
    open_key = st.session_state.get(ui.detail_key("traj"))
    if wr is not None and open_key:
        match = next((t for t in traj_tiles(wr) if t["key"] == open_key), None)
        if match is not None:
            ui.detail_view("traj", match["title"], match["draw"], match["text"], back_label="Back to trajectories")
            return
    if ex is None or ex.empty:
        missing("waymo_demo_examples")
    else:
        types = [t for t in ("vehicle", "pedestrian", "cyclist") if (ex["object_type"] == t).any()]
        kind = st.pills("Road user", types, default=types[0], key="traj_type",
                        format_func=lambda t: motion.TYPE_WORD.get(t, t)) or types[0]
        pool = ex[ex["object_type"] == kind]
        turns = pool["true_future"].map(lambda f: abs([p for p in f if p is not None][-1][1]) >= 5)
        pool = (pool.assign(turns=turns, gain=pool["fde_8s_cv_m"] - pool["fde_8s_best_m"])
                .sort_values(["turns", "gain"], ascending=[False, False]))
        if st.session_state.get("traj_example") not in set(pool["example_id"]):
            st.session_state["traj_example"] = pool["example_id"].iloc[0]
        chosen = st.session_state["traj_example"]
        muted(f"{len(pool)} examples from the test scenes, described from their own paths. Pick one.")
        with st.container(height=190, key="traj_picker"):
            cols = st.columns(4)
            for i, (_, r) in enumerate(pool.iterrows()):
                cols[i % 4].button(motion.describe(r), key=f"ex_{r['example_id']}", on_click=pick_example,
                                   args=(r["example_id"],), icon=motion.TYPE_ICON.get(kind), width="stretch",
                                   type="primary" if r["example_id"] == chosen else "secondary")
        r = pool[pool["example_id"] == chosen].iloc[0]
        color = theme.ROAD_USER.get(kind, theme.TRACK_B)
        c = st.columns(3)
        with c[0]:
            ui.kpi_card("Going straight", num(r["fde_8s_cv_m"], 1, " m"), "Error if it kept its speed and heading.",
                        "after 8 seconds", theme.BASELINE)
        with c[1]:
            ui.kpi_card("Top guess", num(r["fde_8s_best_m"], 1, " m"), "The model's most confident path.",
                        f"{100 * (r['fde_8s_best_m'] / r['fde_8s_cv_m'] - 1):+.0f}% vs going straight", theme.TRACK_B)
        with c[2]:
            ui.kpi_card("Closest of 6", num(r["fde_8s_min_m"], 1, " m"), "The best of the model's 6 paths.",
                        f"{100 * (r['fde_8s_min_m'] / r['fde_8s_cv_m'] - 1):+.0f}% vs going straight", theme.TRACK_B)
        with st.container(border=True):
            st.markdown(f"**{motion.describe(r)}**")
            st.plotly_chart(charts.trajectory(r, color), config=charts.CONFIG, theme=None)
            muted(f"Seen from the {motion.TYPE_WORD.get(kind, kind).lower()}: it is at the dot, facing right. "
                  "Gray lines are lane centers.")
    if wr is None:
        missing("waymo_results")
        return
    st.markdown("#### All test road users")
    show_tiles(traj_tiles(wr), "traj", theme.TRACK_B)


# ---------------------------------------------------------------------------------------------
# Data and limitations

def tab_data(data: D.Data, facts: dict) -> None:
    rm, wr = data.get("recall_metrics"), data.get("waymo_results")
    cf, rf, w = facts.get("complaints", {}), facts.get("recalls", {}), facts.get("waymo", {})
    left, right = st.columns(2, gap="medium")
    with left, st.container(border=True):
        st.markdown("#### :material/database: Data sources")
        lines = []
        if cf:
            lines.append(f"- **NHTSA complaints:** {num(cf['complaints'])} complaints ({num(cf['records'])} "
                         f"complaint–part records), {cf['first'][:4]} to {cf['last'][:4]}.")
        if rf:
            lines.append(f"- **NHTSA recalls:** {num(rf['campaigns'])} campaigns ({num(rf['records'])} "
                         f"vehicle–part records), {rf['first'][:4]} to {rf['last'][:4]}.")
        if facts.get("investigations"):
            lines.append(f"- **NHTSA investigations:** {num(facts['investigations'])}.")
        lines.append("- Source: [NHTSA datasets](https://www.nhtsa.gov/nhtsa-datasets-and-apis), public domain.")
        if w:
            lines.append(f"- **Waymo Open Motion Dataset v1.3.1:** {num(w['train_scenes'])} training and "
                         f"{num(w['test_scenes'])} test scenes. Non-commercial license; "
                         "[waymo.com/open](https://waymo.com/open/about).")
        lines.append("- Results are stored in BigQuery; raw data stays on the laptops it was downloaded to.")
        st.markdown("\n".join(lines))
    with right, st.container(border=True):
        st.markdown("#### :material/report: Honest limitations")
        cov20, cov100 = topk(rm, "lgbm", "pct_recalls_caught", 20), topk(rm, "lgbm", "pct_recalls_caught", 100)
        misses, abl, late = reports.miss_reasons(), reports.ablation(), reports.label_audit_misses()
        out = []
        if cov20 is not None:
            out.append(f"- Flags catch a small share of recalls: {pct(cov20)} with 20 a month, {pct(cov100)} with 100.")
        if misses is not None:
            none = misses[misses["reason"].str.startswith("no complaints")]["recalls"].sum() / misses["recalls"].sum()
            out.append(f"- Recalls with no complaints beforehand can't be predicted from complaints; {none:.0%} of "
                       "missed recalls had none in the year before.")
        if abl is not None:
            age = abl[abl["model"].str.lower().str.startswith("age")]["hit_rate_20"]
            both = abl[abl["model"].str.contains(r"\+")]["hit_rate_20"]
            if len(age) and len(both):
                out.append(f"- Car age drives much of the score: age alone reaches {pct(age.iloc[0])}, the full "
                           f"model {pct(both.iloc[0])}.")
        if late is not None:
            share = late[late["match"].str.contains("6-12")]["share"]
            if len(share):
                out.append(f"- The 6-month window misses some early flags: {pct(share.iloc[0], 0)} of misses were "
                           "recalled 6 to 12 months later.")
        if wr is not None:
            out.append(f"- The top guess ({num(waymo(wr, 'model_best_guess'), 1)} m at 8 s) lags the closest of 6 "
                       f"({num(waymo(wr, 'model_best_of_6'), 1)} m): the model doesn't see lanes or other road users.")
            out.append(f"- Small samples: {num(waymo(wr, 'model_best_guess', 'pedestrian', col='n_agents'))} "
                       f"pedestrians and {num(waymo(wr, 'model_best_guess', 'cyclist', col='n_agents'))} cyclists.")
        out.append("- Briefs come from a free-tier AI model and are checked automatically, not by a person.")
        st.markdown("\n".join(out))
    briefs = briefs_of(data)
    model = facts.get("model", {})
    with st.container(border=True):
        st.markdown("#### :material/verified_user: How we kept it honest")
        window = (f"{month_name(model['test_first_month'])} to {month_name(model['test_last_month'])}"
                  if model else "the most recent years")
        st.markdown("\n".join([
            f"- Time-based split: trained on earlier years, tested on {window}.",
            "- Complaints received after a car part's first recall are never used.",
            "- Test data was never used to choose models or settings.",
            f"- Every AI citation was checked: {int(briefs['citations_valid'].sum())} of "
            f"{int(briefs['citations_total'].sum())} valid." if len(briefs) else "- Every AI citation is checked.",
        ]))
    if facts.get("generated_at"):
        muted(f"Dataset counts as of {facts['generated_at'][:10]}.")
    st.markdown(f"**Built by {', '.join(TEAM)}.**")


# ---------------------------------------------------------------------------------------------

size = st.session_state.get("text_size") or "M"
scale = theme.TEXT_SCALES.get(size, 1.0)
charts.set_scale(scale)
st.markdown(theme.css(scale), unsafe_allow_html=True)

data = load()
facts = load_facts()
head, side = st.columns([4, 1.4])
head.markdown(f'<div class="vse-title">Vehicle Safety Early Warning</div>'
              f'<div class="vse-sub">Spot vehicle safety problems early: months ahead and seconds ahead.</div>'
              f'<div class="vse-team">{" · ".join(TEAM)}</div>', unsafe_allow_html=True)
with side:
    st.segmented_control("Text size", list(theme.TEXT_SCALES), default="M", key="text_size",
                         selection_mode="single")
    if data.notice:
        st.markdown(f'{ui.chip("Local data", theme.BG, theme.TRACK_A, "cloud_off")}'
                    f'<div class="vse-muted">{escape(data.notice.replace("`", ""), quote=False)}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(ui.chip("Live results", theme.BG, theme.GOOD, "cloud_done"), unsafe_allow_html=True)

# Lazy tabs: only the tab on screen is built and sent to the browser, so the page stays light.
tabs = st.tabs(TABS, key="main_tab", on_change="rerun")
for tab, render in zip(tabs, [lambda: tab_overview(data, facts), lambda: tab_watchlist(data),
                              lambda: tab_agent(data), lambda: tab_results(data),
                              lambda: tab_trajectory(data), lambda: tab_data(data, facts)], strict=True):
    if tab.open:
        with tab:
            render()

with st.expander("Technical details"):
    st.markdown("\n".join(f"- `{name}`: {source}" for name, source in data.source.items()))
    for name, why in data.problems.items():
        st.code(f"{name}: {why}", language=None)
    if st.button("Reload data"):
        load.clear()
        load_facts.clear()
        load_agent_details.clear()
        st.rerun()
