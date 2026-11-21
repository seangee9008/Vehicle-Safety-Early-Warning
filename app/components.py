"""Reusable dashboard pieces: KPI cards, clickable tiles, chips, step chips."""

from __future__ import annotations

from collections.abc import Callable
from html import escape

import streamlit as st

from app import theme


def kpi_card(title: str, value: str, meaning: str, comparison: str = "", color: str = theme.TRACK_A,
             help: str | None = None, tall: bool = False) -> None:
    """Accent bar, big number, one line of meaning, a small comparison line."""
    tip = f' title="{escape(help)}"' if help else ""
    st.markdown(
        f'<div class="vse-kpi{" tall" if tall else ""}" style="--accent:{color}"{tip}>'
        f'<div class="t">{escape(title)}</div><div class="v">{escape(value)}</div>'
        f'<div class="m">{escape(meaning)}</div>'
        + (f'<div class="c">{escape(comparison)}</div>' if comparison else "")
        + "</div>", unsafe_allow_html=True)


def chip(text: str, fg: str = theme.INK, bg: str = "#F1F3F5") -> str:
    return f'<span class="vse-chip" style="--fg:{fg};--bg:{bg}">{escape(text)}</span>'


def detail_key(page: str) -> str:
    return f"detail_{page}"


def open_detail(page: str, key: str) -> None:
    st.session_state[detail_key(page)] = key


def close_detail(page: str) -> None:
    st.session_state[detail_key(page)] = None


def tile(title: str, value: str, one_liner: str, color: str, key: str, page: str) -> None:
    """A card with a headline number; its button opens the tile's detail view."""
    with st.container(border=True):
        st.markdown(
            f'<div class="vse-tile" style="--accent:{color}"><div class="t">{escape(title)}</div>'
            f'<div class="v">{escape(value)}</div><div class="o">{escape(one_liner)}</div></div>',
            unsafe_allow_html=True)
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
    """Numbered step chips in a row: (label, real number)."""
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
