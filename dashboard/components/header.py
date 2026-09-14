"""
Header Component for SIH26054 Dashboard.
Renders the top operational banner with mission metadata, execution mode, and overall status.
"""

import streamlit as st
from dashboard.schemas.view_model import OverviewViewModel
from dashboard.utils.formatters import format_timestamp
from dashboard.utils.styles import render_status_badge, STATUS_COLORS


def render_header(overview: OverviewViewModel):
    """Render compact operational mission header bar."""
    if overview.is_synthetic_demo or overview.simulation_mode == "SYNTHETIC_SIMULATION":
        st.markdown(
            '<div class="demo-watermark-compact">'
            'SIMULATION ENVIRONMENT · SYNTHETIC ENGINE TELEMETRY'
            '</div>',
            unsafe_allow_html=True,
        )

    col_title, col_status = st.columns([3, 1])

    with col_title:
        st.markdown(
            '<div style="display: flex; align-items: baseline; gap: 10px; margin-bottom: 4px;">'
            '<span style="font-size: 20px; font-weight: 800; color: #f0f6fc; letter-spacing: -0.3px;">Avekshak</span>'
            '<span style="font-size: 11px; font-weight: 700; color: #58a6ff; font-family: monospace; background: #161b22; border: 1px solid #21262d; border-radius: 3px; padding: 2px 6px;">SIH26054</span>'
            '<span style="font-size: 12px; color: #8b949e;">Aero-Piston Engine Health & Prognostics</span>'
            '</div>',
            unsafe_allow_html=True,
        )

    with col_status:
        status_html = render_status_badge(overview.overall_status, f"SYSTEM: {overview.overall_status.value}")
        st.markdown(
            f'<div style="text-align: right; padding-top: 2px;">{status_html}</div>',
            unsafe_allow_html=True,
        )

    # Compact Single-Line Operational Context Strip
    meta_strip = (
        f'<div style="background-color: #11151c; border: 1px solid #21262d; border-radius: 4px; '
        f'padding: 6px 12px; font-size: 11px; color: #8b949e; font-family: monospace; display: flex; '
        f'flex-wrap: wrap; gap: 16px; align-items: center; margin-top: 4px; margin-bottom: 12px;">'
        f'<div><span style="color: #6e7681;">ENGINE:</span> <b style="color: #58a6ff;">{overview.engine_id}</b></div>'
        f'<div><span style="color: #6e7681;">MISSION:</span> <b style="color: #f0f6fc;">{overview.mission_id or "NOT_ASSIGNED"}</b></div>'
        f'<div><span style="color: #6e7681;">FLIGHT PHASE:</span> <b style="color: #f0f6fc;">{overview.mission_phase}</b></div>'
        f'<div><span style="color: #6e7681;">MISSION TIME:</span> <b style="color: #f0f6fc;">{format_timestamp(overview.timestamp)}</b></div>'
        f'<div><span style="color: #6e7681;">DATA SOURCE:</span> <b style="color: #8b949e;">{overview.simulation_mode}</b></div>'
        f'</div>'
    )
    st.markdown(meta_strip, unsafe_allow_html=True)

    # Demonstration Notice (strictly hides injected ground truth answer key from main judge-facing view)
    st.markdown(
        """
        <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 4px; padding: 5px 12px; margin-top: -4px; margin-bottom: 12px; font-size: 11px; color: #8b949e;">
            <b style="color: #58a6ff;">Avekshak:</b> AI-Enabled Real-Time Digital Twin for Aero-Piston Engine Health & Prognostics.
            <span style="color: #6e7681; margin-left: 8px;">Results represent simulated engine behaviour for engineering evaluation and decision support.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


