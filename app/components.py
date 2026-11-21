"""Reusable dashboard pieces: KPI cards with small visuals, tiles, chips, tags, rings, a generic
car silhouette, steppers and decision-tree nodes. All SVG is inline and generic (no logos)."""

from __future__ import annotations

import re
from collections.abc import Callable
from html import escape

import streamlit as st

from app import theme


def icon(name: str, color: str | None = None) -> str:
    style = f' style="color:{color}"' if color else ""
    return f'<span class="ms"{style}>{name}</span>'


def chip(text: str, fg: str = theme.INK, bg: str = theme.BORDER, icon_name: str | None = None) -> str:
    ic = icon(icon_name) if icon_name else ""
    return f'<span class="vse-chip" style="--fg:{fg};--bg:{bg}">{ic}{escape(text)}</span>'


def part_chip(component: str) -> str:
    return chip(theme.part_name(component), theme.BG, theme.part_color(component))


def make_chip(make: str) -> str:
    return chip(str(make).title(), theme.INK, theme.make_tint(make))


# ---------------------------------------------------------------------------------------------
# short "why flagged" tags from the watchlist's reason strings (never shown raw)

TAG_ICONS = {"New car": "new_releases", "Complaints rising": "trending_up", "Many complaints": "forum",
             "Crashes reported": "car_crash", "Fire reports": "local_fire_department",
             "Injuries reported": "personal_injury", "Smoke reports": "mode_heat", "Towed away": "rv_hookup",
             "No warning": "warning", "Power loss": "power_off", "Older car": "history",
             "Mileage pattern": "speed"}
TAG_RULES = [  # (pattern on the reason string, short tag); first match wins
    (r"under a year old|\b1 year old", "New car"),
    (r"\d+ years? old", "Older car"),
    (r"crash", "Crashes reported"),
    (r"\bfires?\b", "Fire reports"),
    (r"injur", "Injuries reported"),
    (r"smoke", "Smoke reports"),
    (r"\btow", "Towed away"),
    (r"without warning", "No warning"),
    (r"lost power|shut-?off", "Power loss"),
    (r"complaints up|none in the prior|% of this car's complaints", "Complaints rising"),
    (r"miles at complaint", "Mileage pattern"),
    (r"complaints? in the last", "Many complaints"),
]


def reason_tag(reason: str) -> str | None:
    text = reason.lower()
    return next((tag for pattern, tag in TAG_RULES if re.search(pattern, text)), None)


def reason_tags(top_reasons: str | None, limit: int = 3) -> list[str]:
    tags = []
    for reason in str(top_reasons or "").split(";"):
        tag = reason_tag(reason.strip()) if reason.strip() else None
        if tag and tag not in tags:
            tags.append(tag)
    return tags[:limit]


def tag_chip(tag: str) -> str:
    return chip(tag, theme.INK, theme.BORDER, TAG_ICONS.get(tag))


# ---------------------------------------------------------------------------------------------
# small SVG visuals

def ring(fraction: float, label: str, color: str = theme.TRACK_A, size: int = 46, stroke: int = 6) -> str:
    """Filled ring (donut) with a label in the middle."""
    r = (size - stroke) / 2
    circ = 2 * 3.14159 * r
    f = max(0.0, min(1.0, float(fraction)))
    return (f'<svg class="vse-ring" width="{size}" height="{size}" viewBox="0 0 {size} {size}" role="img" '
            f'aria-label="{escape(label)}">'
            f'<circle cx="{size / 2}" cy="{size / 2}" r="{r}" fill="none" stroke="{theme.BORDER}" stroke-width="{stroke}"/>'
            f'<circle cx="{size / 2}" cy="{size / 2}" r="{r}" fill="none" stroke="{color}" stroke-width="{stroke}" '
            f'stroke-dasharray="{f * circ:.1f} {circ:.1f}" stroke-linecap="round" '
            f'transform="rotate(-90 {size / 2} {size / 2})"/>'
            f'<text x="50%" y="54%" text-anchor="middle" dominant-baseline="middle" fill="{theme.INK}" '
            f'font-size="{size * 0.27:.0f}" font-weight="700" font-family="Inter, sans-serif">{escape(label)}</text></svg>')


_CAR_PATH = ("M6 30 L10 21 Q12 17 17 16 L32 14 Q38 13 44 17 L52 22 L60 23 Q66 24 66 30 L66 34 "
             "L6 34 Z")


def car_svg(color: str, width: int = 120) -> str:
    """Generic side-view car silhouette (no brand design), tinted."""
    h = width * 44 / 72
    return (f'<svg width="{width}" height="{h:.0f}" viewBox="0 0 72 44" role="img" aria-label="car">'
            f'<path d="{_CAR_PATH}" fill="{color}" opacity="0.9"/>'
            f'<path d="M19 18 L31 16 L31 22 L15 22 Z M34 16 L42 17 L49 22 L34 22 Z" fill="{theme.BG}" opacity="0.55"/>'
            f'<circle cx="19" cy="34" r="6" fill="{theme.BG}"/><circle cx="19" cy="34" r="3" fill="{theme.MUTED}"/>'
            f'<circle cx="53" cy="34" r="6" fill="{theme.BG}"/><circle cx="53" cy="34" r="3" fill="{theme.MUTED}"/></svg>')


def _mini_car(fill: str, opacity: float = 1.0) -> str:
    return (f'<svg width="34" height="21" viewBox="0 0 72 44" style="margin:1px 2px"><path d="{_CAR_PATH}" '
            f'fill="{fill}" opacity="{opacity}"/><circle cx="19" cy="34" r="6" fill="{fill}" opacity="{opacity}"/>'
            f'<circle cx="53" cy="34" r="6" fill="{fill}" opacity="{opacity}"/></svg>')


def cars_of_twenty(hits: float, color: str = theme.TRACK_A) -> str:
    """20 small cars, the hits filled in (a partial car shows the fraction)."""
    full, part = int(hits), hits - int(hits)
    cars = []
    for i in range(20):
        if i < full:
            cars.append(_mini_car(color))
        elif i == full and part > 0.05:
            cars.append(_mini_car(color, 0.25 + 0.75 * part))
        else:
            cars.append(_mini_car(theme.BORDER))
    rows = "<br>".join("".join(cars[i:i + 5]) for i in range(0, 20, 5))
    return f'<div style="line-height:0">{rows}</div>'


def compare_bars(rows: list[tuple[str, float, str]], unit: str, width: int = 260) -> str:
    """Horizontal bars: (label, value, color)."""
    top = max(v for _, v, _ in rows) or 1
    out = []
    for label, value, color in rows:
        w = max(4, value / top * width * 0.9)
        out.append(f'<div style="margin:4px 0"><div style="font-size:0.8rem;color:{theme.MUTED}">'
                   f'{escape(label)} <b style="color:{theme.INK}">{value:,.1f} {unit}</b></div>'
                   f'<div style="width:{w:.0f}px;height:12px;border-radius:6px;background:{color};margin-top:2px">'
                   '</div></div>')
    return "".join(out)


# ---------------------------------------------------------------------------------------------
# cards and layout pieces

def kpi_card(title: str, value: str, meaning: str, comparison: str = "", color: str = theme.TRACK_A,
             visual: str = "", help: str | None = None, tall: bool = False) -> None:
    tip = f' title="{escape(help)}"' if help else ""
    st.markdown(
        f'<div class="vse-kpi{" tall" if tall else ""}" style="--accent:{color}"{tip}>'
        f'<div class="t">{escape(title)}</div><div class="v">{escape(value)}</div>'
        f'<div class="m">{escape(meaning)}</div>'
        + (f'<div class="c">{escape(comparison)}</div>' if comparison else "")
        + (f'<div class="viz">{visual}</div>' if visual else "")
        + "</div>", unsafe_allow_html=True)


def detail_key(page: str) -> str:
    return f"detail_{page}"


def open_detail(page: str, key: str) -> None:
    st.session_state[detail_key(page)] = key


def close_detail(page: str) -> None:
    st.session_state[detail_key(page)] = None


def tile(title: str, value: str, one_liner: str, color: str, key: str, page: str,
         preview: Callable[[], None] | None = None) -> None:
    """A card with a headline number and a small preview chart; its button opens the detail view."""
    with st.container(key=f"tile_{page}_{key}"):
        st.markdown(f'<div class="vse-tile" style="--accent:{color}"><div class="t">{escape(title)}</div>'
                    f'<div class="v">{escape(value)}</div><div class="o">{escape(one_liner)}</div></div>',
                    unsafe_allow_html=True)
        if preview is not None:
            preview()
        st.button("View details", key=f"open_{page}_{key}", on_click=open_detail, args=(page, key),
                  type="tertiary", icon=":material/arrow_forward:")


def detail_view(page: str, title: str, draw: Callable[[], None], sentences: list[str],
                back_label: str = "Back to overview") -> None:
    st.button(back_label, key=f"back_{page}", on_click=close_detail, args=(page,),
              icon=":material/arrow_back:")
    st.markdown(f"### {title}")
    with st.container(border=True):
        draw()
    for sentence in sentences:
        st.markdown(sentence)


def steps(items: list[tuple[str, str]], color: str) -> None:
    cells = "".join(
        f'<div class="vse-step" style="--accent:{color}"><div class="n">{i}</div>'
        f'<div class="l">{escape(label)}</div><div class="x">{escape(number)}</div></div>'
        for i, (label, number) in enumerate(items, 1))
    st.markdown(f'<div class="vse-steps">{cells}</div>', unsafe_allow_html=True)


def headline(value: str, description: str, color: str) -> None:
    st.markdown(f'<div class="vse-headline" style="--accent:{color}"><span class="v">{escape(value)}'
                f'</span><span class="d">{escape(description)}</span></div>', unsafe_allow_html=True)


def glance(items: list[tuple[str, str]]) -> None:
    cells = "".join(f'<div><div class="v">{escape(v)}</div><div class="l">{escape(label)}</div></div>'
                    for v, label in items)
    st.markdown(f'<div class="vse-glance">{cells}</div>', unsafe_allow_html=True)


def stepper(items: list[tuple[str, str, str]], color: str) -> None:
    """Horizontal steps: (material icon, label, value)."""
    cells = "".join(f'<div class="st" style="--accent:{color}"><div class="ic">{icon(ic)}</div>'
                    f'<div class="nm">{escape(value)}</div><div class="lb">{escape(label)}</div></div>'
                    for ic, label, value in items)
    st.markdown(f'<div class="vse-stepper">{cells}</div>', unsafe_allow_html=True)


def counter(icon_name: str, value: int, label: str, color: str) -> str:
    c = color if value > 0 else theme.MUTED
    return (f'<div class="vse-counter">{icon(icon_name, c)}<div class="n" style="color:{c}">{value}</div>'
            f'<div class="l">{escape(label)}</div></div>')


def tree_node(choice: str, why: str, pros: list[str], cons: list[str], color: str, key: str) -> None:
    st.markdown(f'<div class="vse-node" style="--accent:{color}"><div class="c">{escape(choice)}</div>'
                f'<div class="w">{escape(why)}</div></div>', unsafe_allow_html=True)
    with st.popover("Pros and cons", icon=":material/balance:", key=f"pop_{key}"):
        for p in pros:
            st.markdown(f":green[:material/check_circle:] {p}")
        for c in cons:
            st.markdown(f":orange[:material/warning:] {c}")
