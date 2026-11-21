"""The demo's design system: one palette, one chart template, one stylesheet. Defined once here."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# Track A: vehicle recalls. Track B: Waymo road users.
TRACK_A = "#1F4E9A"
TRACK_A_TINT = "#E8EEF8"
TRACK_B = "#0F8B7D"
TRACK_B_TINT = "#E3F4F1"
BASELINE = "#9AA0A6"
GOOD = "#2E7D32"
CAUTION = "#C77700"
ROAD_USER = {"vehicle": "#1F4E9A", "pedestrian": "#C2185B", "cyclist": "#E07B00"}

INK = "#1F2933"
MUTED = "#5F6B7A"
LINE = "#E3E7EC"
LANE = "#D5DAE1"
PAST = "#3A4250"
FONT = "Inter, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

BASELINE_TOOLTIP = "Rank cars by how many complaints they got in the last 3 months."

pio.templates["vse"] = go.layout.Template(layout=go.Layout(
    font={"family": FONT, "size": 13, "color": INK},
    paper_bgcolor="white", plot_bgcolor="white",
    colorway=[TRACK_A, BASELINE, TRACK_B, CAUTION],
    margin={"l": 8, "r": 8, "t": 8, "b": 8},
    xaxis={"showgrid": False, "zeroline": False, "linecolor": LINE, "ticks": "", "title_font": {"color": MUTED}},
    yaxis={"showgrid": True, "gridcolor": "#F0F2F5", "zeroline": False, "linecolor": LINE, "ticks": "",
               "title_font": {"color": MUTED}},
    legend={"orientation": "h", "yanchor": "top", "y": -0.18, "xanchor": "left", "x": 0, "title_text": "",
                "font": {"color": MUTED}},
    hoverlabel={"bgcolor": "white", "bordercolor": LINE, "font": {"family": FONT, "color": INK}},
    bargap=0.35, bargroupgap=0.08,
))
pio.templates.default = "vse"

CSS = f"""
<style>
  html, body, [class*="css"] {{ font-family: {FONT}; }}
  h1, h2, h3, h4 {{ letter-spacing: -0.01em; }}
  .block-container {{ padding-top: 2.2rem; max-width: 1280px; }}
  .vse-kpi {{ border: 1px solid {LINE}; border-left: 5px solid var(--accent); border-radius: 8px;
             padding: 14px 16px 12px; background: white; }}
  .vse-kpi.tall {{ min-height: 232px; }}
  .vse-kpi .t {{ font-size: 0.82rem; color: {MUTED}; font-weight: 600; margin-bottom: 4px; }}
  .vse-kpi .v {{ font-size: 2.0rem; font-weight: 700; color: {INK}; line-height: 1.15; }}
  .vse-kpi .m {{ font-size: 0.86rem; color: {INK}; margin-top: 6px; }}
  .vse-kpi .c {{ font-size: 0.78rem; color: {MUTED}; margin-top: 6px; }}
  .vse-tile {{ border-top: 4px solid var(--accent); padding: 2px 2px 4px; }}
  .vse-tile .t {{ font-size: 0.9rem; font-weight: 600; color: {INK}; }}
  .vse-tile .v {{ font-size: 1.7rem; font-weight: 700; color: var(--accent); margin: 4px 0; }}
  .vse-tile .o {{ font-size: 0.85rem; color: {MUTED}; min-height: 2.6em; }}
  .vse-chip {{ display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 0.78rem;
              font-weight: 600; margin: 2px 4px 2px 0; background: var(--bg); color: var(--fg);
              white-space: nowrap; }}
  .vse-steps {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 6px 0 10px; }}
  .vse-step {{ flex: 1 1 120px; background: white; border-radius: 8px; padding: 10px 10px 9px;
              border: 1px solid {LINE}; }}
  .vse-step .n {{ display: inline-flex; width: 20px; height: 20px; border-radius: 50%;
                 align-items: center; justify-content: center; font-size: 0.72rem; font-weight: 700;
                 color: white; background: var(--accent); margin-bottom: 6px; }}
  .vse-step .l {{ font-size: 0.82rem; color: {INK}; font-weight: 600; line-height: 1.25; }}
  .vse-step .x {{ font-size: 0.95rem; color: var(--accent); font-weight: 700; margin-top: 4px; }}
  .vse-headline {{ display: flex; align-items: baseline; gap: 10px; margin-top: 4px; }}
  .vse-headline .v {{ font-size: 2.3rem; font-weight: 700; color: var(--accent); }}
  .vse-headline .d {{ font-size: 0.88rem; color: {MUTED}; }}
  .vse-glance {{ display: flex; gap: 0; border: 1px solid {LINE}; border-radius: 8px; overflow: hidden; }}
  .vse-glance > div {{ flex: 1; padding: 10px 14px; border-right: 1px solid {LINE}; }}
  .vse-glance > div:last-child {{ border-right: none; }}
  .vse-glance .v {{ font-size: 1.15rem; font-weight: 700; color: {INK}; }}
  .vse-glance .l {{ font-size: 0.78rem; color: {MUTED}; }}
  .vse-head {{ border: 1px solid {LINE}; border-left: 5px solid {TRACK_A}; border-radius: 8px;
              padding: 12px 16px; background: white; }}
  .vse-head .v {{ font-size: 1.35rem; font-weight: 700; color: {INK}; }}
  .vse-head .s {{ font-size: 0.88rem; color: {MUTED}; margin-top: 2px; }}
  .vse-muted {{ color: {MUTED}; font-size: 0.85rem; }}
  .st-key-panel_a {{ background: {TRACK_A_TINT}; border-radius: 10px; padding: 14px 16px; }}
  .st-key-panel_b {{ background: {TRACK_B_TINT}; border-radius: 10px; padding: 14px 16px; }}
</style>
"""


def tint(color: str) -> str:
    return {TRACK_A: TRACK_A_TINT, TRACK_B: TRACK_B_TINT}.get(color, "#F1F3F5")
