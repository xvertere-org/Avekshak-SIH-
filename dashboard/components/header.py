"""
Header Component for SIH26054 Dashboard.
Renders the top operational banner with mission metadata, execution mode, and overall status.
"""

import streamlit as st
from dashboard.schemas.view_model import OverviewViewModel
from dashboard.utils.formatters import format_timestamp
from dashboard.utils.styles import render_status_badge, STATUS_COLORS


def render_header(overview: OverviewViewModel):
    """Render top operational mission banner."""
    if overview.is_synthetic_demo or overview.simulation_mode == "SYNTHETIC_SIMULATION":
        st.markdown(
            """
            <div class="demo-watermark">
                ⚠️ SYNTHETIC SIMULATION / DEMONSTRATION MODE — NOT LIVE AIRCRAFT TELEMETRY
            </div>
            """,
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(
            f"""
            <div style="margin-bottom: 8px;">
                <h2 style="margin: 0; color: #f0f6fc; font-weight: 700;">
                    Aero-Piston Engine Digital Twin <span style="font-size: 16px; color: #58a6ff; font-weight: 500;">SIH26054</span>
                </h2>
                <div style="color: #8b949e; font-size: 13px; margin-top: 4px;">
                    MALE UAV Propulsion Health Monitoring, Fault Diagnosis & Prognostics
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        status_html = render_status_badge(overview.overall_status, f"SYSTEM: {overview.overall_status.value}")
        st.markdown(
            f"""
            <div style="text-align: right; padding-top: 6px;">
                {status_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Operational Context Ribbon (engine_id + mission_id define execution context)
    meta_cols = st.columns(5)
    with meta_cols[0]:
        st.caption("ENGINE IDENTIFIER")
        st.markdown(f"**`{overview.engine_id}`**")
    with meta_cols[1]:
        st.caption("MISSION IDENTIFIER")
        st.markdown(f"**`{overview.mission_id or 'NOT_ASSIGNED'}`**")
    with meta_cols[2]:
        st.caption("MISSION PHASE")
        st.markdown(f"**`{overview.mission_phase}`**")
    with meta_cols[3]:
        st.caption("MISSION ELAPSED TIME")
        st.markdown(f"**`{format_timestamp(overview.timestamp)}`**")
    with meta_cols[4]:
        st.caption("EXECUTION MODE")
        st.markdown(f"**`{overview.simulation_mode}`**")

    # If scenario metadata is present, render strictly as Simulation Scenario / Ground Truth
    if overview.scenario_metadata:
        sc_name = overview.scenario_metadata.get("scenario_name", overview.scenario_metadata.get("name", "Standard Mission"))
        st.markdown(
            f"""
            <div style="background-color: #161b22; border: 1px dashed #30363d; border-radius: 4px; padding: 6px 12px; margin-top: 6px; font-size: 11px; color: #8b949e;">
                <b style="color: #d29922;">Simulation Scenario / Ground Truth:</b> <code>{sc_name}</code>
                <span style="color: #6e7681; margin-left: 8px;">(Ground truth simulation control only — never used as an inferred system result)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<hr style='border-color: #30363d; margin-top: 10px; margin-bottom: 20px;' />", unsafe_allow_html=True)
