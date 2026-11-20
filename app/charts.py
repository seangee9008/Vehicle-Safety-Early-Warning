"""Charts for the demo (Altair, bundled with Streamlit)."""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

KINDS = ["past (1 s)", "true future (8 s)", "constant velocity", "model: most confident",
         "model: other 5 paths"]
COLORS = ["#3b3b3b", "#1f77b4", "#ff7f0e", "#d62728", "#f2a3a3"]
MODEL = "#d62728"


def _points(path, kind: str, line: str, width: float, opacity: float = 1.0) -> list[dict]:
    return [{"x": p[0], "y": p[1], "kind": kind, "line": line, "order": i, "w": width, "o": opacity}
            for i, p in enumerate(path) if p is not None]


def trajectory_frame(example: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(paths, lanes) long frames for one waymo_demo_examples row."""
    rows = []
    rows += _points(example["past_path"], KINDS[0], "past", 3.0)
    rows += _points(example["true_future"], KINDS[1], "future", 2.5)
    rows += _points(example["baseline_path"], KINDS[2], "cv", 2.5)
    conf = list(example["model_confidences"])
    for k, path in enumerate(example["model_paths"]):  # highest confidence first
        kind = KINDS[3] if k == 0 else KINDS[4]
        rows += _points(path, kind, f"model{k}", 4.5 if k == 0 else 1.2 + 3 * conf[k],
                        1.0 if k == 0 else 0.85)
    lanes = [{"x": p[0], "y": p[1], "line": f"lane{i}", "order": j}
             for i, line in enumerate(example["lane_polylines"]) for j, p in enumerate(line)]
    return pd.DataFrame(rows), pd.DataFrame(lanes, columns=["x", "y", "line", "order"])


def trajectory_chart(example: pd.Series, size: int = 620) -> alt.LayerChart:
    paths, lanes = trajectory_frame(example)
    xy = paths[["x", "y"]].to_numpy()
    lo, hi = xy.min(0), xy.max(0)
    center = (lo + hi) / 2
    half = max(float(np.max(hi - lo)) / 2 * 1.25, 12.0)  # equal scales on both axes, 12 m minimum
    xdom = [center[0] - half, center[0] + half]
    ydom = [center[1] - half, center[1] + half]
    x = alt.X("x:Q", scale=alt.Scale(domain=xdom, nice=False), title="metres forward (heading now)")
    y = alt.Y("y:Q", scale=alt.Scale(domain=ydom, nice=False), title="metres left")

    lane_layer = alt.Chart(lanes).mark_line(clip=True, color="#c9ced6", strokeWidth=1).encode(
        x=x, y=y, detail="line:N", order="order:Q")
    # other paths first so the most confident one is drawn on top
    # true future last: it is the answer, so it must stay visible on top of the model's paths
    draw_order = {k: i for i, k in enumerate([KINDS[4], KINDS[0], KINDS[2], KINDS[3], KINDS[1]])}
    paths = (paths.assign(z=paths["kind"].map(draw_order))
             .sort_values(["z", "line", "order"], kind="stable"))
    path_layer = alt.Chart(paths).mark_line(clip=True, strokeCap="round").encode(
        x=x, y=y, detail="line:N", order="order:Q",
        color=alt.Color("kind:N", scale=alt.Scale(domain=KINDS, range=COLORS),
                        legend=alt.Legend(title=None, orient="bottom", columns=3)),
        strokeWidth=alt.StrokeWidth("w:Q", scale=None, legend=None),
        opacity=alt.Opacity("o:Q", scale=None, legend=None),
        strokeDash=alt.condition(alt.datum.kind == KINDS[2], alt.value([6, 4]), alt.value([1, 0])),
    )
    now = alt.Chart(pd.DataFrame({"x": [0.0], "y": [0.0]})).mark_point(
        filled=True, size=90, color="black").encode(x=x, y=y)
    return (lane_layer + path_layer + now).properties(width=size, height=size)


def topk_chart(topk: pd.DataFrame, value: str, title: str) -> alt.Chart:
    """Grouped bars: model vs baseline at each watchlist size."""
    fmt = ".1%" if value in ("precision", "pct_recalls_caught") else ".0f"
    base = alt.Chart(topk).encode(
        x=alt.X("scorer_label:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
        y=alt.Y(f"{value}:Q", title=title, axis=alt.Axis(format=fmt)),
        color=alt.Color("scorer_label:N", title=None,
                        scale=alt.Scale(domain=["model (LightGBM)", "baseline (complaint count)"],
                                        range=[MODEL, "#9aa5b1"]),
                        legend=alt.Legend(orient="bottom")),
    )
    bars = base.mark_bar()
    text = base.mark_text(dy=-6).encode(text=alt.Text(f"{value}:Q", format=fmt))
    return (bars + text).properties(width=90, height=220).facet(
        column=alt.Column("top_k:O", title="watchlist size per month (top k)"))
