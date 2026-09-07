"""
Reusable Metric Card Component for SIH26054 Dashboard.
Renders KPI cards with status-colored accents, values, units, and availability indicators.
"""

import streamlit as st
from dashboard.schemas.view_model import MetricCardModel, StatusLevel, AvailabilityStatus
from dashboard.utils.styles import STATUS_COLORS, render_status_badge


def render_metric_card(card: MetricCardModel):
    """Render a styled KPI metric card without multiline markdown indentation bugs."""
    accent_color = STATUS_COLORS.get(card.status, "#30363d")
    status_badge_html = render_status_badge(card.status)

    value_style = "color: #f0f6fc;"
    if card.availability == AvailabilityStatus.UNAVAILABLE or card.value == "Unavailable":
        value_style = "color: #8b949e; font-style: italic; font-size: 20px;"

    unit_span = (
        f'<span style="color: #8b949e; font-size: 13px; margin-left: 4px; font-weight: 500;">{card.unit}</span>'
        if card.unit
        else ""
    )
    subtext_div = (
        f'<div style="font-size: 11px; color: #8b949e; margin-top: 6px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{card.subtext}</div>'
        if card.subtext
        else '<div style="height: 17px;"></div>'
    )

    # Note: Using a single continuous string without internal newlines/spaces prevents CommonMark
    # from erroneously classifying indented HTML tags as markdown code blocks (<pre><code>).
    card_html = (
        f'<div style="background: linear-gradient(145deg, #161b22, #0d1117); '
        f'border: 1px solid #30363d; border-top: 3px solid {accent_color}; '
        f'box-shadow: 0 4px 12px rgba(0,0,0,0.25), 0 0 10px {accent_color}1a; '
        f'border-radius: 8px; padding: 14px 16px; min-height: 118px; '
        f'display: flex; flex-direction: column; justify-content: space-between;">'
        f'<div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">'
        f'<div style="font-size: 11px; font-weight: 700; color: #8b949e; text-transform: uppercase; letter-spacing: 0.6px;">{card.label}</div>'
        f'<div>{status_badge_html}</div>'
        f'</div>'
        f'<div>'
        f'<span style="{value_style} font-weight: 700; font-family: monospace; font-size: 23px; letter-spacing: -0.5px;">{card.value}</span>'
        f'{unit_span}'
        f'</div>'
        f'{subtext_div}'
        f'</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

