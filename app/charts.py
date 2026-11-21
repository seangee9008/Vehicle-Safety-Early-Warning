"""Every chart in the demo, in one dark house style (theme.py): plain axis labels with units, model in
its track color, baseline in gray, no titles inside charts. Font sizes follow the S / M / L text
size (set_scale)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app import theme

CONFIG = {"displayModeBar": False, "responsive": True}
PREDICTORS = {"constant_velocity": "Going straight", "model_best_guess": "Top guess",
              "model_best_of_6": "Closest of 6"}
_SCALE = {"s": 1.0}


def set_scale(scale: float) -> None:
    _SCALE["s"] = scale


def _layout(fig: go.Figure, height: int = 360, **kwargs) -> go.Figure:
    s = _SCALE["s"]
    base = theme.chart_layout(s)
    legend = {**base.pop("legend"), **kwargs.pop("legend", {})}
    fig.update_layout(**base, legend=legend, height=round(height * (0.9 + 0.1 * s)), **kwargs)
    return fig


def _mini(fig: go.Figure, height: int = 110) -> go.Figure:
    fig.update_layout(template="vse_dark", height=height, showlegend=False,
                      margin={"l": 0, "r": 0, "t": 4, "b": 0}, hovermode=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, rangemode="tozero")
    return fig


def model_vs_baseline(k: list[int], model: list[float], baseline: list[float], *, ytitle: str,
                      percent: bool = True, xtitle: str = "Cars flagged per month") -> go.Figure:
    fmt = ".1%" if percent else ",.0f"
    fig = go.Figure()
    for name, values, c in [("Model", model, theme.TRACK_A), ("Baseline", baseline, theme.BASELINE)]:
        fig.add_bar(x=[f"Top {x}" for x in k], y=values, name=name, marker_color=c,
                    text=[format(v, fmt) for v in values], textposition="outside",
                    textfont={"color": theme.INK},
                    hovertemplate=f"{name}<br>%{{x}}: %{{y:{fmt}}}<extra></extra>")
    fig.update_yaxes(title_text=ytitle, tickformat=".0%" if percent else ",", rangemode="tozero",
                     automargin=True)
    fig.update_xaxes(title_text=xtitle)
    return _layout(fig, barmode="group")


def mini_pair(model: list[float], baseline: list[float]) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(y=model, marker_color=theme.TRACK_A)
    fig.add_bar(y=baseline, marker_color=theme.BASELINE)
    return _mini(fig.update_layout(barmode="group", bargap=0.35))


def mini_bars(values: list[float], colors: list[str]) -> go.Figure:
    return _mini(go.Figure(go.Bar(y=values, marker_color=colors)).update_layout(bargap=0.35))


def mini_donut(values: list[float], colors: list[str]) -> go.Figure:
    fig = go.Figure(go.Pie(values=values, hole=0.6, sort=False, textinfo="none",
                           marker={"colors": colors, "line": {"color": theme.CARD, "width": 2}}))
    return _mini(fig)


def mini_lines(x: list, ys: list[tuple[list[float], str, str]]) -> go.Figure:
    fig = go.Figure()
    for y, color, dash in ys:
        fig.add_scatter(x=x, y=y, mode="lines+markers", line={"color": color, "width": 3, "dash": dash},
                        marker={"size": 6})
    return _mini(fig)


def bars(labels: list[str], values: list[float], colors: list[str], *, ytitle: str,
         percent: bool = True, height: int = 340, xtitle: str = "") -> go.Figure:
    fmt = ".1%" if percent else ",.1f"
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors, textfont={"color": theme.INK},
                           text=[format(v, fmt) for v in values], textposition="outside",
                           hovertemplate=f"%{{x}}: %{{y:{fmt}}}<extra></extra>"))
    fig.update_yaxes(title_text=ytitle, tickformat=".0%" if percent else ",", rangemode="tozero")
    fig.update_xaxes(title_text=xtitle)
    return _layout(fig, height=height, showlegend=False)


def lead_time_groups(groups: list[str], model: list[float | None], baseline: list[float | None],
                     counts: list[str]) -> go.Figure:
    fig = go.Figure()
    for name, values, c in [("Model", model, theme.TRACK_A), ("Baseline", baseline, theme.BASELINE)]:
        fig.add_bar(x=groups, y=values, name=name, marker_color=c, textfont={"color": theme.INK},
                    text=[f"{v:.0f} wk" if v is not None else "" for v in values], textposition="outside",
                    customdata=counts,
                    hovertemplate=f"{name}<br>%{{x}}<br>%{{y:.1f}} weeks<br>%{{customdata}}<extra></extra>")
    fig.update_yaxes(title_text="Median lead time (weeks)", rangemode="tozero")
    return _layout(fig, barmode="group")


def donut(labels: list[str], values: list[float], colors: list[str]) -> go.Figure:
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.62, sort=False, direction="clockwise",
                           marker={"colors": colors, "line": {"color": theme.CARD, "width": 2}},
                           textinfo="percent", textfont={"color": theme.BG},
                           hovertemplate="%{label}<br>%{value:,} recalls (%{percent})<extra></extra>"))
    return _layout(fig, height=360, legend={"orientation": "v", "y": 0.5, "x": 1.02, "yanchor": "middle"})


def paired(categories: list[str], first: list[float], second: list[float], names: tuple[str, str],
           colors: tuple[str, str], ytitle: str) -> go.Figure:
    fig = go.Figure()
    for name, values, c in [(names[0], first, colors[0]), (names[1], second, colors[1])]:
        fig.add_bar(x=categories, y=values, name=name, marker_color=c, textfont={"color": theme.INK},
                    text=[f"{v:.1%}" for v in values], textposition="outside",
                    hovertemplate=f"{name}<br>%{{x}}: %{{y:.1%}}<extra></extra>")
    fig.update_yaxes(title_text=ytitle, tickformat=".0%", rangemode="tozero")
    return _layout(fig, barmode="group")


def complaints_per_month(monthly: list[dict], flag_month: pd.Timestamp, color: str) -> go.Figure:
    """Complaints per month over the 12 months up to the flag; months with none show as 0."""
    months = pd.date_range(flag_month - pd.DateOffset(months=11), flag_month, freq="MS")
    counts = {pd.Timestamp(m["month"]): m["complaints"] for m in monthly}
    y = [counts.get(m, 0.0) for m in months]
    colors = [theme.TRACK_A if m == flag_month else color for m in months]
    fig = go.Figure(go.Bar(x=[m.strftime("%b %y") for m in months], y=y, marker_color=colors,
                           hovertemplate="%{x}: %{y:.0f} complaints<extra></extra>"))
    fig.add_annotation(x=flag_month.strftime("%b %y"), y=max(y) if y else 0, text="Flagged",
                       showarrow=False, yshift=16, font={"color": theme.TRACK_A, "size": round(14 * _SCALE["s"])})
    fig.update_yaxes(title_text="Complaints per month", rangemode="tozero")
    return _layout(fig, height=280, showlegend=False, margin={"l": 10, "r": 10, "t": 30, "b": 10})


def horizon_lines(results: pd.DataFrame, object_type: str = "all") -> go.Figure:
    df = results[results["object_type"] == object_type]
    fig = go.Figure()
    styles = {"constant_velocity": (theme.BASELINE, "dash"), "model_best_guess": (theme.TRACK_B, "solid"),
              "model_best_of_6": (theme.TRACK_B, "dot")}
    for predictor, (color, dash) in styles.items():
        d = df[df["predictor"] == predictor].sort_values("horizon_s")
        fig.add_scatter(x=d["horizon_s"], y=d["fde_m"], name=PREDICTORS[predictor], mode="lines+markers",
                        line={"color": color, "dash": dash, "width": 3}, marker={"size": 9},
                        hovertemplate=f"{PREDICTORS[predictor]}<br>after %{{x}} s: %{{y:.2f}} m<extra></extra>")
    fig.update_xaxes(title_text="Seconds ahead", tickvals=sorted(df["horizon_s"].unique()), ticksuffix=" s")
    fig.update_yaxes(title_text="Position error (meters)", rangemode="tozero")
    return _layout(fig)


def by_road_user(results: pd.DataFrame, horizon_s: int = 8) -> go.Figure:
    df = results[(results["horizon_s"] == horizon_s) & (results["object_type"] != "all")]
    order = [t for t in ("vehicle", "pedestrian", "cyclist") if t in set(df["object_type"])]
    fig = go.Figure()
    colors = {"constant_velocity": theme.BASELINE, "model_best_guess": theme.TRACK_B,
              "model_best_of_6": "#8FE3DA"}
    for predictor, label in PREDICTORS.items():
        d = df[df["predictor"] == predictor].set_index("object_type").reindex(order)
        fig.add_bar(x=[t.capitalize() for t in order], y=d["fde_m"], name=label,
                    marker_color=colors[predictor], customdata=d["n_agents"],
                    hovertemplate=f"{label}<br>%{{x}}: %{{y:.2f}} m<br>%{{customdata:,}} road users<extra></extra>")
    fig.update_yaxes(title_text=f"Position error after {horizon_s} s (meters)", rangemode="tozero")
    return _layout(fig, barmode="group")


def _xy(points) -> tuple[list, list]:
    pts = [p for p in points if p is not None]
    return [p[0] for p in pts], [p[1] for p in pts]


def trajectory(example: pd.Series, color: str) -> go.Figure:
    """Lanes, past, what really happened, going straight, 6 guesses; equal axis scale."""
    fig = go.Figure()
    xs, ys = [], []
    for line in example["lane_polylines"]:
        x, y = _xy(line)
        xs += x + [None]
        ys += y + [None]
    fig.add_scatter(x=xs, y=ys, mode="lines", name="Lanes", line={"color": theme.LANE, "width": 1.2},
                    hoverinfo="skip")
    paths = list(example["model_paths"])
    conf = list(example["model_confidences"])
    for k in range(len(paths) - 1, -1, -1):  # the top guess is drawn last among the guesses
        x, y = _xy(paths[k])
        top = k == 0
        fig.add_scatter(x=x, y=y, mode="lines", name="Top guess" if top else "Other guesses",
                        legendgroup="top" if top else "other", showlegend=top or k == 1,
                        line={"color": theme.TRACK_B, "width": 4.5 if top else 1.6}, opacity=1 if top else 0.55,
                        hovertemplate=f"{'Top guess' if top else 'Guess'} (confidence {conf[k]:.0%})<extra></extra>")
    x, y = _xy(example["baseline_path"])
    fig.add_scatter(x=x, y=y, mode="lines", name="Going straight",
                    line={"color": theme.BASELINE, "width": 2.5, "dash": "dash"}, hoverinfo="skip")
    x, y = _xy(example["past_path"])
    fig.add_scatter(x=x, y=y, mode="lines", name="Past second", line={"color": theme.PAST, "width": 3.5},
                    hoverinfo="skip")
    x, y = _xy(example["true_future"])
    fig.add_scatter(x=x, y=y, mode="lines", name="What really happened", line={"color": color, "width": 5},
                    hovertemplate="What really happened<extra></extra>")
    fig.add_scatter(x=[0], y=[0], mode="markers", name="Now", marker={"color": theme.INK, "size": 10},
                    hovertemplate="Position now<extra></extra>", showlegend=False)

    pts = np.array([p for path in [example["past_path"], example["true_future"], example["baseline_path"],
                                   *paths] for p in path if p is not None], dtype=float)
    lo, hi = pts.min(0), pts.max(0)
    center, half = (lo + hi) / 2, max(float((hi - lo).max()) / 2 * 1.2, 12.0)
    fig.update_xaxes(range=[center[0] - half, center[0] + half], title_text="Meters ahead",
                     showgrid=False, zeroline=False)
    fig.update_yaxes(range=[center[1] - half, center[1] + half], title_text="Meters to the left",
                     scaleanchor="x", scaleratio=1, showgrid=False, zeroline=False)
    return _layout(fig, height=620, legend={"y": -0.2}, margin={"l": 10, "r": 10, "t": 10, "b": 95})
