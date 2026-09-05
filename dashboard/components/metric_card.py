"""
Reusable Metric Card Component for SIH26054 Dashboard.
Renders KPI cards with status-colored accents, values, units, and availability indicators.
"""

import streamlit as st
from dashboard.schemas.view_model import MetricCardModel, StatusLevel, AvailabilityStatus
from dashboard.utils.styles import STATUS_COLORS, render_status_badge


def render_metric_card(card: MetricCardModel):
    """Render a styled KPI metric card."""
    accent_color = STATUS_COLORS.get(card.status, "#30363d")
    status_badge_html = render_status_badge(card.status)

    value_style = "color: #f0f6fc;"
    if card.availability == AvailabilityStatus.UNAVAILABLE or card.value == "Unavailable":
        value_style = "color: #8b949e; font-style: italic; font-size: 20px;"

    card_html = f"""
    <div style="
        background-color: #161b22;
        border: 1px solid #30363d;
        border-top: 3px solid {accent_color};
        border-radius: 8px;
        padding: 16px;
        min-height: 115px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    ">
        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
            <div style="font-size: 11px; font-weight: 600; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px;">
                {card.label}
            </div>
            <div>{status_badge_html}</div>
        </div>
        <div style="margin-top: 8px; margin-bottom: 4px;">
            <span style="{value_style} font-weight: 700; font-family: monospace; font-size: 22px;">
                {card.value}
            </span>
            {"<span style='color: #8b949e; font-size: 13px; margin-left: 4px;'>" + card.unit + "</span>" if card.unit else ""}
        </div>
        <div style="font-size: 11px; color: #8b949e; margin-top: 2px;">
            {card.subtext or ""}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)
