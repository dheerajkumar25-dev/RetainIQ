"""Small HTML/Streamlit helpers so every page looks consistent."""
from html import escape

import streamlit as st

PLOT_CONFIG = {"displayModeBar": False}


def card():
    """A bordered container styled as a card. The unique key gives it a stable CSS class (st-key-riq-card-N)."""
    counter = st.session_state.get("_riq_card_counter", 0) + 1
    st.session_state["_riq_card_counter"] = counter
    return st.container(border=True, key=f"riq-card-{counter}")


def pill(text: str, tone: str = "") -> str:
    return f'<span class="riq-pill {tone}">{escape(text)}</span>'


def page_header(title: str, description: str, pills: list[tuple[str, str]] | None = None) -> None:
    pill_html = "".join(pill(text, tone) for text, tone in (pills or []))
    st.markdown(
        f'<div class="riq-header"><h1>{escape(title)}</h1><p>{escape(description)}</p>{pill_html}</div>',
        unsafe_allow_html=True,
    )


def section_title(title: str, caption: str | None = None) -> None:
    caption_html = f"<p>{escape(caption)}</p>" if caption else ""
    st.markdown(f'<div class="riq-section"><h3>{escape(title)}</h3>{caption_html}</div>', unsafe_allow_html=True)


def kpi_card(label: str, value: str, caption: str = "", tone: str = "neutral") -> None:
    caption_html = f'<div class="riq-kpi-caption">{escape(caption)}</div>' if caption else ""
    st.markdown(
        f'<div class="riq-kpi riq-tone-{tone}"><div class="riq-kpi-label">{escape(label)}</div>'
        f'<div class="riq-kpi-value">{escape(value)}</div>{caption_html}</div>',
        unsafe_allow_html=True,
    )


def kpi_row(items: list[dict]) -> None:
    """items: [{'label':..., 'value':..., 'caption':..., 'tone':...}, ...]"""
    for column, item in zip(st.columns(len(items)), items):
        with column:
            kpi_card(**item)


def callout(text_html: str, tone: str = "") -> None:
    """text_html may contain <b> tags; pass only trusted strings."""
    st.markdown(f'<div class="riq-callout {tone}">{text_html}</div>', unsafe_allow_html=True)


def chart_card(figure, key: str | None = None) -> None:
    with card():
        st.plotly_chart(figure, width="stretch", config=PLOT_CONFIG, key=key)


def bullet_list(lines: list[str]) -> None:
    items = "".join(f"<li>{escape(line)}</li>" for line in lines)
    st.markdown(f'<ul class="riq-list">{items}</ul>', unsafe_allow_html=True)
