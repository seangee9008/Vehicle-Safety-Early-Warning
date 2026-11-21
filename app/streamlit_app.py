"""Vehicle safety early warning: the demo dashboard.

    python run.py demo                       # or: streamlit run app/streamlit_app.py
    VSE_DATA=local python run.py demo        # skip BigQuery, use local files (offline)

Reads the published results from BigQuery (src/bq_reader.py) and falls back to local files.
Every number on screen comes from the data (BigQuery tables, the committed reports/ files, and
app/facts.json, which app/build_facts.py counts from the local data). No AI calls at runtime.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from app import charts, reports, theme
from app import components as ui
from app import data as D

st.set_page_config(page_title="Vehicle safety early warning", page_icon=":material/directions_car:",
                   layout="wide")
st.markdown(theme.CSS, unsafe_allow_html=True)

TABS = ["Overview", "Recall watchlist", "Investigation brief", "Model results", "Trajectory",
        "Data and limitations"]
FACTS_PATH = Path(__file__).resolve().parent / "facts.json"
SECTION_TITLES = {"failure pattern": "What drivers report", "severity": "Severity", "evidence": "Evidence",
                  "watchlist reasons check": "What the flag reasons say",
                  "similar past recalls": "Similar past recalls", "suggested next step": "Suggested next step"}
MISS_LABELS = {"1-2 complaints in the 12 months before": "1 or 2 complaints in the year before",
               "no complaints in the 12 months before": "No complaints in the year before",
               "3+ complaints, never in the top 100": "3+ complaints, never in the top 100",
               "3+ complaints, best rank 21-100": "3+ complaints, ranked 21 to 100"}


@st.cache_data(ttl=3600, show_spinner="Loading results…")
def load() -> D.Data:
    return D.load_all()


@st.cache_data
def load_facts() -> dict:
    return json.loads(FACTS_PATH.read_text(encoding="utf-8")) if FACTS_PATH.exists() else {}


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


def part(component: str) -> str:
    return str(component).capitalize().replace("Adas", "ADAS")


def vehicle(row) -> str:
    return f"{str(row['make']).title()} {row['model']} {row['model_year']}"


def missing(name: str) -> None:
    st.info(f"`{name}` isn't available right now, so this part can't be shown.")


def topk(rm: pd.DataFrame | None, scorer: str, metric: str, k: int) -> float | None:
    return D.metric(rm, scorer=scorer, metric=metric, source_file="recall_topk_stg.csv", top_k=k)


def audit_value(audit: pd.DataFrame | None, experiment: str, variant_prefix: str, metric: str,
                split: str = "test") -> float | None:
    if audit is None:
        return None
    a = audit[(audit["experiment"] == experiment) & (audit["metric"] == metric) & (audit["split"] == split)
              & audit["variant"].str.startswith(variant_prefix)]
    return float(a["value"].iloc[0]) if len(a) else None


def waymo(wr: pd.DataFrame | None, predictor: str, object_type: str = "all", horizon: int = 8,
          col: str = "fde_m") -> float | None:
    if wr is None:
        return None
    w = wr[(wr["predictor"] == predictor) & (wr["object_type"] == object_type) & (wr["horizon_s"] == horizon)]
    return float(w[col].iloc[0]) if len(w) else None


def briefs_of(data: D.Data) -> pd.DataFrame:
    wb = data.get("v_watchlist_with_briefs")
    if wb is None:  # empty, but with the columns the callers sum and read
        return pd.DataFrame(columns=["month", "rank", "make", "model", "model_year", "component",
                                     "top_reasons", "has_brief", *D.BRIEF_COLUMNS])
    return wb[wb["has_brief"]].sort_values(["month", "rank"], ascending=[False, True])


def brief_label(row) -> str:
    return f"{month_name(row['month'])} · #{row['rank']} {vehicle(row)} · {part(row['component'])}"


def open_brief(label: str) -> None:
    st.session_state["brief_pick"] = label
    st.session_state["main_tab"] = "Investigation brief"


# ---------------------------------------------------------------------------------------------
# Overview

def tab_overview(data: D.Data, facts: dict) -> None:
    h = data.get("v_headline_metrics")
    rm, wr, audit = data.get("recall_metrics"), data.get("waymo_results"), data.get("recall_audit")
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
        ui.kpi_card("Hit rate", pct(hit), f"About {20 * hit:.1f} of 20 flagged cars a month were recalled "
                    "within 6 months.", f"vs {pct(base_hit)} baseline{random}",
                    theme.TRACK_A, help=theme.BASELINE_TOOLTIP, tall=True)
    with c[1]:
        ui.kpi_card("Lead time", num(weeks, 0, " weeks"), "Typical warning before the recall was announced.",
                    f"Baseline {num(base_weeks)} weeks, but it catches {num(base_caught)} recalls to the "
                    f"model's {num(caught)}" if base_weeks else "", theme.TRACK_A, tall=True)
    with c[2]:
        ui.kpi_card("Evidence check", f"{valid} of {total}", "Every complaint the AI quoted is real and dated "
                    "before the flag.", f"{unique} unique complaints across {len(briefs)} briefs"
                    if unique is not None else "", theme.TRACK_A, tall=True)
    with c[3]:
        ui.kpi_card("Position error after 8 seconds", num(closest, 1, " m"), "Closest of the model's 6 guesses "
                    "for where a road user will be.", f"vs {num(cv, 1, ' m')} going straight · top guess "
                    f"{num(best, 1, ' m')}", theme.TRACK_B, tall=True)

    st.markdown("#### How it works")
    wb = data.get("v_watchlist_with_briefs")
    ex = data.get("waymo_demo_examples")
    left, right = st.columns(2, gap="medium")
    with left, st.container(key="panel_a"):
        st.markdown(":material/directions_car: **Months ahead: vehicle recalls**")
        per_month = int(wb.groupby("month").size().max()) if wb is not None else None
        months = wb["month"].nunique() if wb is not None else None
        model_facts = facts.get("model", {})
        ui.steps([
            ("Complaints in", millions(facts.get("complaints", {}).get("complaints"))),
            ("Model ranks every car part monthly",
             f"{num(model_facts.get('car_parts_ranked_per_month', 0) / 1000)}k a month"
             if model_facts else "–"),
            ("Top 20 watchlist", f"{per_month} a month, {months} months" if per_month else "–"),
            ("AI writes an evidence brief", f"{len(briefs)} briefs, {total} citations"),
        ], theme.TRACK_A)
        ui.headline(pct(hit), f"of flagged cars recalled within 6 months (baseline {pct(base_hit)})",
                    theme.TRACK_A)
    with right, st.container(key="panel_b"):
        st.markdown(":material/add_road: **Seconds ahead: road users**")
        w = facts.get("waymo", {})
        first = ex.iloc[0] if ex is not None and len(ex) else None
        n_agents = waymo(wr, "model_best_guess", col="n_agents")
        ui.steps([
            ("Driving scenes", f"{w.get('train_scenes', '–')} train, {w.get('test_scenes', '–')} test"),
            ("Past 1 second", f"{len(first['past_path'])} positions" if first is not None else "–"),
            ("Model draws 6 possible paths", f"{len(first['model_paths'])} paths, "
             f"{len(first['true_future']) / 10:.0f} s ahead" if first is not None else "–"),
            ("Compare to what happened", f"{num(n_agents)} road users"),
        ], theme.TRACK_B)
        ui.headline(num(closest, 1, " m"), f"closest of 6 after 8 s (going straight: {num(cv, 1, ' m')})",
                    theme.TRACK_B)

    st.markdown("#### Data at a glance")
    cf, rf, w = facts.get("complaints", {}), facts.get("recalls", {}), facts.get("waymo", {})
    years = f"{cf['first'][:4]}–{cf['last'][:4]}" if cf else "–"
    ui.glance([
        (millions(cf.get("complaints")), "vehicle complaints to NHTSA"),
        (num(rf.get("campaigns")), "recall campaigns"),
        (years, "years covered"),
        (num((w.get("train_scenes") or 0) + (w.get("test_scenes") or 0)) if w else "–", "driving scenes"),
    ])


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
    c2.markdown(f'<div class="vse-muted" style="padding-top:2.1rem">The {len(rows)} car parts most likely to '
                f'be recalled in the 6 months after {label}, using only complaints received by then. Select a '
                "row to open its brief.</div>", unsafe_allow_html=True)
    table = pd.DataFrame({
        "Rank": rows["rank"],
        "Vehicle": rows.apply(vehicle, axis=1),
        "Part": rows["component"].map(part),
        "Risk score": rows["score"],
        "Why flagged": rows["top_reasons"].fillna("").map(lambda s: [t.strip() for t in s.split(";") if t.strip()]),
        "Brief": rows["has_brief"].map(lambda b: "Brief ready" if b else ""),
    })
    event = st.dataframe(
        table, hide_index=True, width="stretch", height=36 * (len(table) + 1) + 3,
        on_select="rerun", selection_mode="single-row", key="watch_table",
        column_config={
            "Rank": st.column_config.NumberColumn(width=60),
            "Vehicle": st.column_config.TextColumn(width=250),
            "Part": st.column_config.TextColumn(width=200),
            "Risk score": st.column_config.ProgressColumn(format="%.2f", min_value=0.0,
                                                          max_value=float(wb["score"].max()), width=120,
                                                          help="Model's chance of a recall within 6 months"),
            "Why flagged": st.column_config.ListColumn(width=600),
            "Brief": st.column_config.TextColumn(width=100),
        })
    picked = event.selection.rows if event and hasattr(event, "selection") else []
    if picked:
        row = rows.iloc[picked[0]]
        with st.container(border=True):
            a, b = st.columns([3, 1])
            a.markdown(f"**#{row['rank']} {vehicle(row)}** · {part(row['component'])}")
            if row["has_brief"]:
                b.button("View brief", type="primary", on_click=open_brief, args=(brief_label(row),),
                         icon=":material/description:", key="view_brief")
            else:
                brief_months = sorted({month_name(m) for m in wb.loc[wb["has_brief"], "month"]})
                a.markdown(f'<span class="vse-muted">No brief for this row. Briefs were written for the top '
                           f'{int(wb["has_brief"].sum())} of {", ".join(brief_months)}.</span>',
                           unsafe_allow_html=True)


# ---------------------------------------------------------------------------------------------
# Investigation brief

def _sections(markdown: str) -> list[tuple[str, str]]:
    out, title, body = [], None, []
    for line in markdown.splitlines():
        if line.startswith("## "):
            if title is not None:
                out.append((title, "\n".join(body).strip()))
            raw = line[3:].strip()
            key = re.sub(r"\s*\(.*\)$", "", raw).lower()
            title, body = SECTION_TITLES.get(key, raw), []
        else:
            body.append(line)
    if title is not None:
        out.append((title, "\n".join(body).strip()))
    return out


def _chip_citations(text: str) -> str:
    """Show [ODINO n] (and grouped citations) as small chips; the words are unchanged."""
    def group(match: re.Match) -> str:
        return " ".join(ui.chip(f"ODINO {n}", theme.TRACK_A, theme.TRACK_A_TINT)
                        for n in re.findall(r"\d+", match.group(0)))
    return re.sub(r"\[ODINO[^\]]*\]", group, text)


def tab_brief(data: D.Data) -> None:
    briefs = briefs_of(data)
    if briefs.empty:
        missing("v_watchlist_with_briefs") if data.get("v_watchlist_with_briefs") is None else \
            st.info("No briefs have been published yet.")
        return
    labels = {brief_label(r): i for i, r in briefs.iterrows()}
    if st.session_state.get("brief_pick") not in labels:
        st.session_state.pop("brief_pick", None)
    pick = st.selectbox("Watchlist row with a brief", list(labels), key="brief_pick")
    r = briefs.loc[labels[pick]]
    sections = _sections(r["brief_markdown"])
    similar = next((body for title, body in sections if title == "Similar past recalls"), "")
    scope = r["similar_recall_search_scope"]
    if "none" in similar.lower() and "-" not in similar.split("\n", 1)[-1]:
        similar_label = "Similar past recalls: none found"
    else:
        similar_label = {"same_make": "Similar past recalls: same make", "all_makes":
                         "Similar past recalls: all makes"}.get(scope, "Similar past recalls: searched all makes")
    outcome = ("Passed after one retry" if r["passed_after_retry"] else
               "Still failing checks" if r["still_failing"] else "Checked on first try")
    ok = r["citations_valid"] == r["citations_total"]
    chips = [
        ui.chip(f"Evidence check {int(r['citations_valid'])}/{int(r['citations_total'])}",
                theme.GOOD if ok else theme.CAUTION, "#E8F3E9" if ok else "#FBF0E0"),
        ui.chip(outcome, theme.GOOD if outcome != "Still failing checks" else theme.CAUTION,
                "#E8F3E9" if outcome != "Still failing checks" else "#FBF0E0"),
        ui.chip(similar_label, theme.TRACK_A, theme.TRACK_A_TINT),
    ]
    st.markdown(
        f'<div class="vse-head"><div class="v">{vehicle(r)}</div><div class="s">{part(r["component"])} · '
        f'{month_name(r["month"])} watchlist · rank {r["rank"]} of '
        f'{int((data.get("v_watchlist_with_briefs")["month"] == r["month"]).sum())}</div>'
        f'<div style="margin-top:8px">{"".join(chips)}</div></div>', unsafe_allow_html=True)
    st.markdown("")

    by_title = dict(sections)
    order = [["What drivers report"], ["Severity", "What the flag reasons say"], ["Evidence"],
             ["Similar past recalls", "Suggested next step"]]
    for group in order:
        present = [t for t in group if t in by_title]
        if not present:
            continue
        cols = st.columns([1, 2] if len(present) == 2 and present[0] == "Severity" else [1] * len(present))
        for col, title in zip(cols, present, strict=False):
            with col, st.container(border=True):
                st.markdown(f"**{title}**")
                st.markdown(_chip_citations(by_title[title]), unsafe_allow_html=True)
    shown = {t for group in order for t in group}
    extra = [t for t in by_title if t not in shown]
    for title in extra:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.markdown(_chip_citations(by_title[title]), unsafe_allow_html=True)

    cites = data.get("agent_citations")
    if cites is not None:
        mine = cites[cites["brief_id"] == r["brief_id"]]
        with st.expander(f"Cited complaints ({len(mine)})"):
            st.markdown(" ".join(ui.chip(f"ODINO {o}", theme.GOOD if v else theme.CAUTION,
                                         "#E8F3E9" if v else "#FBF0E0")
                                 for o, v in zip(mine["odino"], mine["valid"], strict=True)),
                        unsafe_allow_html=True)
            st.markdown('<span class="vse-muted">Each number is an NHTSA complaint ID. Green means it exists '
                        "for this vehicle and part and was received before the flag.</span>",
                        unsafe_allow_html=True)
    st.markdown('<span class="vse-muted">Written by an AI model and not edited by hand. Every quoted complaint '
                "was checked automatically.</span>", unsafe_allow_html=True)


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
    ov = reports.overlap(20)
    abl = reports.ablation()
    misses = reports.miss_reasons()
    late = reports.label_audit_misses()
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
            "one": f"About {20 * hit[0]:.1f} of the 20 cars flagged each month were recalled within 6 months.",
            "draw": lambda: st.plotly_chart(charts.model_vs_baseline(ks, hit, base_hit, color=theme.TRACK_A,
                                         ytitle="Recalled within 6 months"), config=charts.CONFIG),
            "text": [("**What it measures.** Of the cars flagged each month, the share that really got a recall "
                  "within the next 6 months, for lists of 20, 50 and 100."),
                  (f"**What we found.** The model beats the baseline at every list size: {pct(hit[0])} vs "
                  f"{pct(base_hit[0])} at 20, {pct(hit[2])} vs {pct(base_hit[2])} at 100."),
                  ("**Why it matters.** Investigators can only open a few cases a month, so the top of the list "
                  "has to be right more often than chance.")]})
    if all(v is not None for v in cov + base_cov):
        tiles.append({
            "key": "coverage", "title": "Coverage", "value": pct(cov[0]),
            "one": f"Share of recalls flagged ahead of time with 20 flags a month; {pct(cov[2])} with 100.",
            "draw": lambda: st.plotly_chart(charts.model_vs_baseline(ks, cov, base_cov, color=theme.TRACK_A,
                                         ytitle="Recalls flagged ahead of time"), config=charts.CONFIG),
            "text": [(f"**What it measures.** Of the {num(first)} first recalls in the test years that had complaints "
                  "beforehand, the share that appeared on the list in the 12 months before the recall."),
                  (f"**What we found.** About {pct(cov[0], 0)} at 20 flags a month, rising to {pct(cov[2])} at 100. "
                  f"The baseline reaches {pct(base_cov[0])} and {pct(base_cov[2])}."),
                  ("**Why it matters.** The list is a way to prioritize, not a full recall detector: most recalls "
                  "are never flagged.")]})
    if ov is not None and weeks is not None:
        tiles.append({
            "key": "lead", "title": "Lead time", "value": num(weeks, 0, " weeks"),
            "one": f"Typical warning before the recall. The baseline's {num(base_weeks)} weeks come from fewer, "
                "different recalls.",
            "draw": lambda: st.plotly_chart(charts.lead_time_groups(
                ["Caught by both", "Only the model", "Only the baseline"],
                [ov["both_median_weeks_lgbm"], ov["only_lgbm_median_weeks"], None],
                [ov["both_median_weeks_baseline"], None, ov["only_baseline_median_weeks"]],
                [f"{int(ov['caught_by_both'])} recalls", f"{int(ov['only_lgbm'])} recalls",
                 f"{int(ov['only_baseline'])} recalls"], theme.TRACK_A), config=charts.CONFIG),
            "text": [("**What it measures.** Weeks between the first month a car part was flagged and its recall, "
                  "with 20 flags a month."),
                  (f"**What we found.** The model's typical lead time is {num(weeks)} weeks and the baseline's "
                  f"{num(base_weeks)}, but they catch different recalls: only {int(ov['caught_by_both'])} are "
                  f"caught by both, and on those the model flags earlier ({num(ov['both_median_weeks_lgbm'])} vs "
                  f"{num(ov['both_median_weeks_baseline'])} weeks)."),
                  (f"**Why it matters.** The {int(ov['only_lgbm'])} recalls only the model catches are new cars "
                  "with a fast rise in complaints, which come with less time to spare.")]})
    if abl is not None:
        rows = {r["model"].lower(): r for _, r in abl.iterrows()}
        age = next((r for k, r in rows.items() if k.startswith("age")), None)
        cmp_ = next((r for k, r in rows.items() if k.startswith("complaints")), None)
        both = next((r for k, r in rows.items() if "+" in k), None)
        base = next((r for k, r in rows.items() if k.startswith("baseline")), None)
        if all(x is not None for x in (age, cmp_, both, base)):
            gain = both["hit_rate_20"] - max(age["hit_rate_20"], cmp_["hit_rate_20"])
            tiles.append({
                "key": "drivers", "title": "What drives the score", "value": f"+{100 * gain:.1f} pts",
                "one": f"Age alone {pct(age['hit_rate_20'])}, complaints alone {pct(cmp_['hit_rate_20'])}; "
                    f"together {pct(both['hit_rate_20'])}.",
                "draw": lambda: st.plotly_chart(charts.bars(
                    ["Baseline", "Vehicle age only", "Complaints only", "Both (model)"],
                    [base["hit_rate_20"], age["hit_rate_20"], cmp_["hit_rate_20"], both["hit_rate_20"]],
                    [theme.BASELINE, "#8FA9D6", "#8FA9D6", theme.TRACK_A],
                    ytitle="Hit rate with 20 flags a month"), config=charts.CONFIG),
                "text": [("**What it measures.** The same model trained on vehicle age only, on complaints only, "
                      "and on both."),
                      ("**What we found.** Neither alone is enough. Age picks out the new cars; complaints rank "
                      "the cars within that group."),
                      ("**Why it matters.** Much of the score comes from age, so complaints add the most where "
                      "new models start getting them.")]})
    if misses is not None and late is not None and exact is not None and variant is not None:
        late_share = late[late["match"].str.contains("6-12")]["share"]
        late_share = float(late_share.iloc[0]) if len(late_share) else None
        thin = misses[~misses["reason"].str.startswith("3+")]["recalls"].sum() / misses["recalls"].sum()
        tiles.append({
            "key": "fair", "title": "Are we grading fairly?", "value": f"+{100 * (variant - exact):.0f} pts",
            "one": f"Looser name matching adds about {100 * (variant - exact):.0f} points; "
                f"{pct(late_share, 0)} of misses were recalled 6 to 12 months later.",
            "draw": lambda: st.plotly_chart(charts.donut(
                [MISS_LABELS.get(r, r) for r in misses["reason"]], list(misses["recalls"]),
                [theme.TRACK_A, "#5878B4", "#9DB2D8", "#D3DDEE"]), config=charts.CONFIG),
            "text": [("**What it measures.** Whether some misses are really hits under a looser match (model names "
                  "like a plug-in hybrid filed under the base model, or a recall just past 6 months), and why "
                  "the rest were missed."),
                  (f"**What we found.** Looser name matching moves the hit rate only from {pct(exact)} to "
                  f"{pct(variant)} (on a rerun of the model), and {pct(late_share, 0)} of misses were recalled 6 "
                  f"to 12 months later: early, not wrong. {thin:.0%} of missed recalls had two or fewer "
                  "complaints in the year before."),
                  ("**Why it matters.** The grading is fair, and the main limit is the data: many recalls have "
                  "almost no complaints to learn from.")]})
    if all(v is not None for v in inv_hit + inv_cov):
        tiles.append({
            "key": "signals", "title": "Did more signals help?", "value": f"{100 * (inv_cov[1] - inv_cov[0]):+.1f} pts",
            "one": "Open NHTSA investigations raised the hit rate but caught fewer different recalls.",
            "draw": lambda: st.plotly_chart(charts.paired(
                ["Hit rate", "Coverage"], [inv_hit[0], inv_cov[0]], [inv_hit[1], inv_cov[1]],
                ("Original model", "With open investigations"), (theme.TRACK_A, "#8FA9D6"),
                ytitle="Share (20 flags a month)"), config=charts.CONFIG),
            "text": [("**What it measures.** The model with one extra signal: whether NHTSA already had an open "
                  "investigation on the car or part."),
                  (f"**What we found.** The hit rate rose from {pct(inv_hit[0])} to {pct(inv_hit[1])}, but coverage "
                  f"fell from {pct(inv_cov[0])} to {pct(inv_cov[1])}: the same investigated cars stay on the list "
                  "month after month."),
                  ("**Why it matters.** Those cars are already being looked at, so the extra signal adds little "
                  "early warning. We kept the original model.")]})
    return tiles


def tab_results(data: D.Data) -> None:
    if data.get("recall_metrics") is None:
        missing("recall_metrics")
        return
    tiles = results_tiles(data)
    open_key = st.session_state.get(ui.detail_key("results"))
    match = next((t for t in tiles if t["key"] == open_key), None)
    if match is not None:
        ui.detail_view("results", match["title"], match["draw"], match["text"])
        st.markdown(f'<span class="vse-muted">Baseline: {theme.BASELINE_TOOLTIP[0].lower() + theme.BASELINE_TOOLTIP[1:]}</span>', unsafe_allow_html=True)
        return
    st.markdown('<span class="vse-muted">Test years, 20 cars flagged a month unless noted. '
                f'Baseline: {theme.BASELINE_TOOLTIP[0].lower() + theme.BASELINE_TOOLTIP[1:]}</span>', unsafe_allow_html=True)
    for start in range(0, len(tiles), 3):
        cols = st.columns(3)
        for col, t in zip(cols, tiles[start:start + 3], strict=False):
            with col:
                ui.tile(t["title"], t["value"], t["one"], theme.TRACK_A, t["key"], "results")


# ---------------------------------------------------------------------------------------------
# Trajectory

def traj_tiles(wr: pd.DataFrame) -> list[dict]:
    cv, best, closest = waymo(wr, "constant_velocity"), waymo(wr, "model_best_guess"), waymo(wr, "model_best_of_6")
    ped_cv, ped_best = waymo(wr, "constant_velocity", "pedestrian"), waymo(wr, "model_best_guess", "pedestrian")
    cyc_cv, cyc_best = waymo(wr, "constant_velocity", "cyclist"), waymo(wr, "model_best_guess", "cyclist")
    veh_cv, veh_best = waymo(wr, "constant_velocity", "vehicle"), waymo(wr, "model_best_guess", "vehicle")
    ped_n, cyc_n = waymo(wr, "model_best_guess", "pedestrian", col="n_agents"), \
        waymo(wr, "model_best_guess", "cyclist", col="n_agents")
    tiles = [{
        "key": "horizon", "title": "Overall error at 3, 5 and 8 seconds", "value": num(best, 1, " m"),
        "one": f"Top guess after 8 s, against {num(cv, 1, ' m')} going straight.",
        "draw": lambda: st.plotly_chart(charts.horizon_lines(wr), config=charts.CONFIG),
        "text": [("**What it measures.** How far each prediction ends up from where the road user really was, "
              "averaged over every road user in the test scenes."),
              (f"**What we found.** The gap grows with time. After 8 seconds the top guess is {num(best, 1)} m off "
              f"and the closest of 6 {num(closest, 1)} m, against {num(cv, 1)} m for going straight."),
              ("**Why it matters.** A few seconds of warning is what a self-driving car needs to plan around "
              "other road users.")]}]
    if veh_cv and veh_best:
        tiles.append({
            "key": "types", "title": "By road user type", "value": f"{100 * (veh_best / veh_cv - 1):+.0f}%",
            "one": "Change in error for vehicles after 8 s, top guess against going straight.",
            "draw": lambda: st.plotly_chart(charts.by_road_user(wr), config=charts.CONFIG),
            "text": ["**What it measures.** Error after 8 seconds for each kind of road user.",
                  (f"**What we found.** Vehicles gain the most ({num(veh_cv, 1)} m to {num(veh_best, 1)} m). For "
                  "pedestrians and cyclists the top guess is about the same as going straight, while the closest "
                  "of 6 is much better."),
                  "**Why it matters.** Most of the improvement comes from cars, which also dominate the data."]})
    if ped_cv and ped_best:
        tiles.append({
            "key": "struggles", "title": "Where it struggles", "value": f"{100 * (ped_best / ped_cv - 1):+.0f}%",
            "one": "Pedestrians: the top guess is no better than going straight.",
            "draw": lambda: st.plotly_chart(charts.horizon_lines(wr, "pedestrian"), config=charts.CONFIG),
            "text": [(f"**What we found.** For pedestrians the top guess ends {num(ped_best, 2)} m off after 8 s, "
                  f"against {num(ped_cv, 2)} m going straight; for cyclists {num(cyc_best, 2)} m against "
                  f"{num(cyc_cv, 2)} m."),
                  (f"**Why.** People stop and turn abruptly, and the test set has only {num(ped_n)} pedestrians "
                  f"and {num(cyc_n)} cyclists.")]})
    return tiles


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
                        format_func=str.capitalize) or types[0]
        pool = ex[ex["object_type"] == kind]
        turns = pool["true_future"].map(lambda f: abs([p for p in f if p is not None][-1][1]) >= 5)
        pool = (pool.assign(turns=turns, gain=pool["fde_8s_cv_m"] - pool["fde_8s_best_m"])
                .sort_values(["turns", "gain"], ascending=[False, False]))
        labels = {f"Scene {r['scenario_id'][:8]} · moved {r['travelled_m']:.0f} m · going straight off by "
                  f"{r['fde_8s_cv_m']:.1f} m": i for i, r in pool.iterrows()}
        pick = st.selectbox(f"Example ({len(pool)} {kind}s)", list(labels), key="traj_pick")
        r = pool.loc[labels[pick]]
        color = theme.ROAD_USER.get(kind, theme.TRACK_B)
        c = st.columns(3)
        with c[0]:
            ui.kpi_card("Going straight", num(r["fde_8s_cv_m"], 1, " m"), "Error if it kept its current speed "
                        "and heading.", "position error after 8 seconds", theme.BASELINE)
        with c[1]:
            ui.kpi_card("Top guess", num(r["fde_8s_best_m"], 1, " m"), "The model's most confident path.",
                        f"{100 * (r['fde_8s_best_m'] / r['fde_8s_cv_m'] - 1):+.0f}% against going straight",
                        color)
        with c[2]:
            ui.kpi_card("Closest of 6", num(r["fde_8s_min_m"], 1, " m"), "The best of the model's 6 paths.",
                        f"{100 * (r['fde_8s_min_m'] / r['fde_8s_cv_m'] - 1):+.0f}% against going straight", color)
        with st.container(border=True):
            st.plotly_chart(charts.trajectory(r, color), config=charts.CONFIG)
            st.markdown(f'<span class="vse-muted">Seen from the {kind}: it is at the dot, facing right. Gray '
                        "lines are lane centers.</span>", unsafe_allow_html=True)
    if wr is None:
        missing("waymo_results")
        return
    st.markdown("#### All test road users")
    cols = st.columns(3)
    for col, t in zip(cols, traj_tiles(wr), strict=False):
        with col:
            ui.tile(t["title"], t["value"], t["one"], theme.TRACK_B, t["key"], "traj")


# ---------------------------------------------------------------------------------------------
# Data and limitations

def tab_data(data: D.Data, facts: dict) -> None:
    rm, wr = data.get("recall_metrics"), data.get("waymo_results")
    cf, rf, w = facts.get("complaints", {}), facts.get("recalls", {}), facts.get("waymo", {})
    left, right = st.columns(2, gap="medium")
    with left, st.container(border=True):
        st.markdown("**Data sources**")
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
            lines.append(f"- **Waymo Open Motion Dataset v1.3.1:** {num(w['train_scenes'])} training scenes and "
                         f"{num(w['test_scenes'])} test scenes. Non-commercial license; "
                         "[waymo.com/open](https://waymo.com/open/about).")
        lines.append("- Results are stored in BigQuery and read from there; raw data never leaves the "
                     "laptops it was downloaded to.")
        st.markdown("\n".join(lines))
    with right, st.container(border=True):
        st.markdown("**Honest limitations**")
        cov20, cov100 = topk(rm, "lgbm", "pct_recalls_caught", 20), topk(rm, "lgbm", "pct_recalls_caught", 100)
        misses, abl, late = reports.miss_reasons(), reports.ablation(), reports.label_audit_misses()
        out = []
        if cov20 is not None:
            out.append(f"- Flags catch a small share of recalls: {pct(cov20)} with 20 a month, {pct(cov100)} "
                       "with 100.")
        if misses is not None:
            none = misses[misses["reason"].str.startswith("no complaints")]["recalls"].sum() / misses["recalls"].sum()
            out.append(f"- Recalls with no complaints beforehand can't be predicted from complaints; "
                       f"{none:.0%} of missed recalls had none in the year before.")
        if abl is not None:
            age = abl[abl["model"].str.lower().str.startswith("age")]["hit_rate_20"]
            both = abl[abl["model"].str.contains(r"\+")]["hit_rate_20"]
            if len(age) and len(both):
                out.append(f"- Vehicle age drives much of the score: age alone reaches a {pct(age.iloc[0])} hit "
                           f"rate, against {pct(both.iloc[0])} for the full model.")
        if late is not None:
            share = late[late["match"].str.contains("6-12")]["share"]
            if len(share):
                out.append(f"- The 6-month window misses some early flags: {pct(share.iloc[0], 0)} of misses were "
                           "recalled 6 to 12 months later.")
        if wr is not None:
            out.append(f"- The top guess ({num(waymo(wr, 'model_best_guess'), 1)} m after 8 s) lags the closest "
                       f"of 6 ({num(waymo(wr, 'model_best_of_6'), 1)} m) because the model doesn't see lanes or "
                       "other road users.")
            out.append(f"- Pedestrians and cyclists have small samples: "
                       f"{num(waymo(wr, 'model_best_guess', 'pedestrian', col='n_agents'))} and "
                       f"{num(waymo(wr, 'model_best_guess', 'cyclist', col='n_agents'))} in the test set.")
        out.append("- Briefs come from a free-tier AI model and are checked automatically, not by a person.")
        st.markdown("\n".join(out))
    briefs = briefs_of(data)
    model = facts.get("model", {})
    with st.container(border=True):
        st.markdown("**How we kept it honest**")
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
        st.markdown(f'<span class="vse-muted">Dataset counts as of {facts["generated_at"][:10]}.</span>',
                    unsafe_allow_html=True)


# ---------------------------------------------------------------------------------------------

data = load()
facts = load_facts()
head, badge = st.columns([3, 2])
head.markdown("## Vehicle safety early warning")
head.markdown('<div class="vse-muted" style="margin-top:-0.6rem">Spot vehicle safety problems early: months '
              "ahead and seconds ahead.</div>", unsafe_allow_html=True)
if data.notice:
    badge.markdown(f'<div style="text-align:right;padding-top:1.2rem">{ui.chip("Local data", theme.CAUTION, "#FBF0E0")}'
                   f'<div class="vse-muted">{data.notice.replace("`", "")}</div></div>', unsafe_allow_html=True)
else:
    badge.markdown(f'<div style="text-align:right;padding-top:1.4rem">'
                   f'{ui.chip("Live results", theme.GOOD, "#E8F3E9")}</div>', unsafe_allow_html=True)

tabs = st.tabs(TABS, key="main_tab")
for tab, render in zip(tabs, [lambda: tab_overview(data, facts), lambda: tab_watchlist(data),
                              lambda: tab_brief(data), lambda: tab_results(data),
                              lambda: tab_trajectory(data), lambda: tab_data(data, facts)], strict=True):
    with tab:
        render()

with st.expander("Technical details"):
    st.markdown("\n".join(f"- `{name}`: {source}" for name, source in data.source.items()))
    for name, why in data.problems.items():
        st.code(f"{name}: {why}", language=None)
    if st.button("Reload data"):
        load.clear()
        load_facts.clear()
        st.rerun()
