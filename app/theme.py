"""The demo's design system, defined once: the dark "asphalt" palette, part and road-user colors,
text sizes (with an S / M / L scale), the Plotly template and the stylesheet."""

from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# Surfaces and text
BG = "#15191E"
CARD = "#1E242B"
BORDER = "#2C343D"
INK = "#EEF1F4"
MUTED = "#A9B2BC"
GRID = "#262D35"
LANE = "#3A434D"
PAST = "#C9D1D9"

# Tracks and comparisons
TRACK_A = "#F5A623"        # vehicle recalls: signal amber
TRACK_A_TINT = "#3A2E17"
TRACK_B = "#2EC4B6"        # Waymo road users: electric teal
TRACK_B_TINT = "#133733"
BASELINE = "#8A9199"
GOOD = "#46A758"
ALERT = "#E5484D"
ROAD_USER = {"vehicle": "#4C9AFF", "pedestrian": "#FF6FB5", "cyclist": "#FFB547"}

# One color per vehicle part (bright enough for dark text on the chip; the label always shows
# too, so color is never the only cue).
PART_COLORS = {
    "ADAS": "#56B4E9", "Electrical": "#F0E442", "Power train": "#E69F00", "Engine": "#FF8C69",
    "Air bags": "#CC79A7", "Brakes": "#FF7A7A", "Steering": "#3BC48F", "Suspension": "#B39DDB",
    "Seat belts": "#9CCC65", "Structure": "#BCAAA4", "Fuel": "#80DEEA", "Visibility": "#FFE0B2",
    "Other": "#B0B7BF",
}
PART_GROUPS = [  # (prefix of the standardized component, part name)
    ("ADAS", "ADAS"), ("ELECTRICAL", "Electrical"), ("POWER TRAIN", "Power train"),
    ("ENGINE", "Engine"), ("AIR BAG", "Air bags"), ("BRAKE", "Brakes"), ("STEERING", "Steering"),
    ("SUSPENSION", "Suspension"), ("SEAT BELT", "Seat belts"), ("STRUCTURE", "Structure"),
    ("FUEL", "Fuel"), ("VISIBILITY", "Visibility"),
]
MAKE_TINTS = ["#2A3A55", "#3A2E4F", "#24423A", "#4A3328", "#3B3F22", "#2E3B4E", "#47283A", "#2F3640"]

FONT = "Inter, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
TEXT_SCALES = {"S": 0.86, "M": 1.0, "L": 1.18}
BASE_PX = 18

BASELINE_TOOLTIP = "Rank cars by how many complaints they got in the last 3 months."


def rgba(hex_color: str, alpha: float) -> str:
    r, g, b = (int(hex_color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def part_name(component: str) -> str:
    c = str(component).upper()
    return next((name for prefix, name in PART_GROUPS if c.startswith(prefix)), "Other")


def part_color(component: str) -> str:
    return PART_COLORS[part_name(component)]


def make_tint(make: str) -> str:
    return MAKE_TINTS[sum(map(ord, str(make).upper())) % len(MAKE_TINTS)]


# ---------------------------------------------------------------------------------------------
# contrast (WCAG 2.1)

def _luminance(hex_color: str) -> float:
    rgb = [int(hex_color.lstrip("#")[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(fg: str, bg: str) -> float:
    hi, lo = sorted([_luminance(fg), _luminance(bg)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


# ---------------------------------------------------------------------------------------------
# Plotly

def chart_layout(scale: float = 1.0) -> dict:
    s = scale
    return {
        "template": "vse_dark",
        "font": {"family": FONT, "size": round(14 * s), "color": INK},
        "hoverlabel": {"bgcolor": BG, "bordercolor": BORDER,
                       "font": {"family": FONT, "size": round(16 * s), "color": INK}},
        "legend": {"font": {"size": round(14 * s), "color": MUTED}},
        "xaxis": {"title": {"font": {"size": round(16 * s), "color": MUTED}},
                  "tickfont": {"size": round(14 * s), "color": MUTED}},
        "yaxis": {"title": {"font": {"size": round(16 * s), "color": MUTED}},
                  "tickfont": {"size": round(14 * s), "color": MUTED}},
    }


pio.templates["vse_dark"] = go.layout.Template(layout=go.Layout(
    font={"family": FONT, "size": 14, "color": INK},
    paper_bgcolor=CARD, plot_bgcolor=CARD,
    colorway=[TRACK_A, BASELINE, TRACK_B, GOOD],
    margin={"l": 10, "r": 10, "t": 10, "b": 10},
    xaxis={"showgrid": False, "zeroline": False, "linecolor": BORDER, "ticks": "", "automargin": True},
    yaxis={"showgrid": True, "gridcolor": GRID, "zeroline": False, "linecolor": BORDER, "ticks": "",
           "automargin": True},
    legend={"orientation": "h", "yanchor": "top", "y": -0.2, "xanchor": "left", "x": 0, "title_text": "",
            "bgcolor": "rgba(0,0,0,0)"},
    bargap=0.32, bargroupgap=0.08,
))


# ---------------------------------------------------------------------------------------------
# stylesheet

def css(scale: float = 1.0) -> str:
    return f"""
<style>
  /* icons use the Material Symbols font Streamlit already ships (no extra download) */
  :root {{ --s: {scale}; }}
  html {{ font-size: calc({BASE_PX}px * var(--s)); }}
  html, body, [class*="css"], .stApp {{ font-family: {FONT}; }}
  .stApp {{ background: {BG}; color: {INK}; }}
  .block-container {{ padding-top: 3.6rem; max-width: 1360px; }}
  .stButton button[kind="primary"], .stButton button[kind="primary"] p,
  .stButton button[kind="primary"] span {{ color: {BG} !important; font-weight: 650; }}
  svg.vse-ring {{ flex: none; }}
  p, li, label, .stMarkdown {{ font-size: 1rem; }}
  h3 {{ font-size: 1.4rem !important; }}
  h4 {{ font-size: 1.22rem !important; }}
  .ms {{ font-family: 'Material Symbols Rounded'; font-weight: normal; font-style: normal;
        font-size: 1.15em; line-height: 1; vertical-align: -0.2em; display: inline-block;
        -webkit-font-feature-settings: 'liga'; font-feature-settings: 'liga'; }}
  .vse-title {{ font-size: 2.45rem; font-weight: 750; color: {INK}; line-height: 1.1; letter-spacing: -0.02em; }}
  .vse-sub {{ font-size: 1.05rem; color: {MUTED}; margin-top: 6px; }}
  .vse-team {{ font-size: 0.9rem; color: {MUTED}; margin-top: 4px; }}
  .vse-muted {{ color: {MUTED}; font-size: 0.9rem; }}
  .vse-card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px; padding: 16px 18px; }}
  .vse-kpi {{ background: {CARD}; border: 1px solid {BORDER}; border-top: 4px solid var(--accent);
             border-radius: 12px; padding: 16px 18px 14px; }}
  .vse-kpi.tall {{ min-height: 21rem; }}
  .vse-kpi .t {{ font-size: 1.22rem; color: {INK}; font-weight: 650; }}
  .vse-kpi .v {{ font-size: 2.5rem; font-weight: 780; color: var(--accent); line-height: 1.15; margin-top: 2px; }}
  .vse-kpi .m {{ font-size: 0.95rem; color: {INK}; margin-top: 4px; }}
  .vse-kpi .c {{ font-size: 0.85rem; color: {MUTED}; margin-top: 6px; }}
  .vse-kpi .viz {{ margin-top: 12px; }}
  .vse-tile .t {{ font-size: 1.22rem; font-weight: 650; color: {INK}; }}
  .vse-tile .v {{ font-size: 2.2rem; font-weight: 780; color: var(--accent); margin: 2px 0; }}
  .vse-tile .o {{ font-size: 0.92rem; color: {MUTED}; min-height: 2.8em; }}
  .vse-chip {{ display: inline-flex; align-items: center; gap: 4px; padding: 3px 10px; border-radius: 999px;
              font-size: 0.82rem; font-weight: 650; margin: 2px 5px 2px 0; background: var(--bg);
              color: var(--fg); white-space: nowrap; }}
  .vse-steps {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 8px 0 12px; }}
  .vse-step {{ flex: 1 1 calc(50% - 10px); min-width: 0; background: {CARD}; border-radius: 10px; padding: 12px; border: 1px solid {BORDER}; }}
  .vse-step .n {{ display: inline-flex; width: 1.5rem; height: 1.5rem; border-radius: 50%; align-items: center;
                 justify-content: center; font-size: 0.8rem; font-weight: 750; color: {BG};
                 background: var(--accent); margin-bottom: 8px; }}
  .vse-step .l {{ font-size: 0.92rem; color: {INK}; font-weight: 600; line-height: 1.3; }}
  .vse-step .x {{ font-size: 1.05rem; color: var(--accent); font-weight: 750; margin-top: 4px; }}
  .vse-headline {{ display: flex; align-items: baseline; gap: 12px; margin-top: 6px; flex-wrap: wrap; }}
  .vse-headline .v {{ font-size: 2.6rem; font-weight: 780; color: var(--accent); }}
  .vse-headline .d {{ font-size: 0.95rem; color: {MUTED}; }}
  .vse-glance {{ display: flex; border: 1px solid {BORDER}; border-radius: 12px; overflow: hidden; background: {CARD}; }}
  .vse-glance > div {{ flex: 1; padding: 14px 18px; border-right: 1px solid {BORDER}; }}
  .vse-glance > div:last-child {{ border-right: none; }}
  .vse-glance .v {{ font-size: 1.5rem; font-weight: 750; color: {INK}; }}
  .vse-glance .l {{ font-size: 0.85rem; color: {MUTED}; }}
  .vse-head {{ background: {CARD}; border: 1px solid {BORDER}; border-left: 6px solid var(--accent);
              border-radius: 12px; padding: 14px 18px; }}
  .vse-head .v {{ font-size: 1.6rem; font-weight: 750; color: {INK}; }}
  .vse-head .s {{ font-size: 0.95rem; color: {MUTED}; margin-top: 2px; }}
  .vse-node {{ background: {CARD}; border: 1px solid {BORDER}; border-left: 4px solid var(--accent);
              border-radius: 10px; padding: 10px 14px; }}
  .vse-node .c {{ font-weight: 680; color: {INK}; font-size: 1rem; }}
  .vse-node .w {{ color: {MUTED}; font-size: 0.88rem; margin-top: 2px; }}
  .vse-root {{ text-align: center; background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px;
              padding: 12px; font-weight: 750; font-size: 1.15rem; color: {INK}; }}
  .vse-branch {{ font-weight: 750; font-size: 1.05rem; color: var(--accent); margin: 10px 0 6px; }}
  .vse-link {{ height: 16px; border-left: 2px solid #4A5562; margin-left: 22px; }}
  .vse-stepper {{ display: flex; align-items: stretch; flex-wrap: wrap; gap: 14px; }}
  .vse-stepper .st {{ flex: 1 1 150px; background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px;
                     padding: 12px; }}
  .vse-stepper .ic {{ color: var(--accent); font-size: 1.6rem; }}
  .vse-stepper .lb {{ font-size: 0.9rem; color: {MUTED}; margin-top: 4px; }}
  .vse-stepper .nm {{ font-size: 1.15rem; font-weight: 750; color: {INK}; }}
  .vse-counter {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 10px; padding: 12px;
                 text-align: center; }}
  .vse-counter .ms {{ font-size: 1.9rem; }}
  .vse-counter .n {{ font-size: 1.8rem; font-weight: 780; }}
  .vse-counter .l {{ font-size: 0.85rem; color: {MUTED}; }}
  .vse-next {{ background: {TRACK_A_TINT}; border: 1px solid {TRACK_A}; border-radius: 12px; padding: 14px 18px; }}
  .vse-th {{ color: {MUTED}; font-size: 0.82rem; font-weight: 650; text-transform: uppercase; letter-spacing: 0.04em; }}
  .vse-eyebrow {{ color: {TRACK_A}; font-size: 0.78rem; font-weight: 750; letter-spacing: 0.12em; margin-top: 0.6rem; }}
  .vse-bigtitle {{ font-size: 1.78rem; font-weight: 760; color: {INK}; line-height: 1.2; margin: 4px 0 14px;
                  letter-spacing: -0.01em; }}
  .vse-story {{ display: flex; align-items: stretch; gap: 10px; margin-bottom: 8px; }}
  .vse-why {{ flex: 1 1 0; min-width: 0; border: 1px solid {BORDER}; border-top: 5px solid var(--accent);
             border-radius: 14px; padding: 18px 20px;
             background: linear-gradient(180deg, var(--glow) 0%, rgba(0,0,0,0) 75%), {CARD}; }}
  .vse-why .ic {{ width: 3.1rem; height: 3.1rem; border-radius: 50%; background: var(--accent);
                 display: flex; align-items: center; justify-content: center; color: {BG}; margin-bottom: 12px; }}
  .vse-why .ic .ms {{ font-size: 1.75rem; vertical-align: 0; }}
  .vse-why .h {{ font-size: 1.44rem; font-weight: 760; color: {INK}; line-height: 1.2; margin-bottom: 10px; }}
  .vse-why .stat {{ font-size: 2.1rem; font-weight: 800; color: var(--accent); line-height: 1.1; }}
  .vse-why .stat .u {{ font-size: 1rem; font-weight: 650; color: {INK}; }}
  .vse-why .sub {{ font-size: 0.9rem; color: {MUTED}; margin-top: 2px; }}
  .vse-why .chips {{ display: flex; flex-direction: column; align-items: flex-start; gap: 2px; }}
  .vse-why .ln {{ font-size: 0.89rem; color: {MUTED}; margin-top: 12px; }}
  .vse-chev {{ align-self: center; color: {MUTED}; font-size: 1.8rem; font-weight: 700; flex: none; }}
  @media (max-width: 900px) {{
    .vse-story {{ flex-direction: column; }}
    .vse-chev {{ transform: rotate(90deg); margin: -4px auto; }}
  }}
  .st-key-panel_a {{ background: {TRACK_A_TINT}; border-radius: 14px; padding: 16px 18px; }}
  .st-key-panel_b {{ background: {TRACK_B_TINT}; border-radius: 14px; padding: 16px 18px; }}
  [class*="st-key-why_"], [class*="st-key-car_"], [class*="st-key-tile_"], [class*="st-key-wrow_"] {{
      background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px; padding: 12px 16px; }}
  [class*="st-key-car_"] {{ min-height: 19.5rem; }}
  [class*="st-key-tile_"] {{ min-height: 19rem; }}
</style>
"""
