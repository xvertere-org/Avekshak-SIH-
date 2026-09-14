"""
Mission Replay Page for SIH26054 Dashboard.
Chronological playback and scrubbing of completed synthetic missions.
Consumes authoritative Phase 13 payloads without recomputing PHM logic.
"""

import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from typing import Dict, Any, List, Optional
import streamlit as st
import plotly.graph_objects as go
import numpy as np

from orchestrator.schema import DashboardStatePayload
from dashboard.schemas.view_model import DashboardViewModel
from dashboard.services.adapter import DashboardAdapter
from dashboard.utils.formatters import (
    format_value,
    format_rul,
    format_action,
    format_status,
    format_channel,
    format_error,
)
from dashboard.utils.styles import PLOT_COLORS, STATUS_COLORS
from phase14.replay import MissionReplayManager


def render_replay_page(
    payloads: List[DashboardStatePayload],
    scenario_name: str = "Active Mission",
):
    """Render 6. MISSION REPLAY section."""
    st.markdown("### Mission Replay")
    st.markdown(
        """
        <div style="font-size: 13px; color: #8b949e; margin-bottom: 16px;">
            Step through the mission and see how engine behaviour, diagnostics, health, and recommendations changed over time.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not payloads:
        st.warning("No mission data available for replay. Run a scenario from the sidebar first.")
        return

    # Initialize Replay Session via MissionReplayManager
    mgr = MissionReplayManager()
    try:
        session = mgr.create_session_from_payloads(payloads, scenario_name=scenario_name)
    except Exception as e:
        st.error(format_error("Mission replay session could not be initialized", e))
        return

    # Playback Controls Bar
    max_step = session.total_steps - 1
    col_ctrl1, col_ctrl2 = st.columns([3, 1])

    with col_ctrl1:
        step_idx = st.slider(
            "Mission Timeline",
            min_value=0,
            max_value=max_step,
            value=min(max_step, max(0, max_step // 2)),
            step=1,
            help="Step through the mission to see engine health, diagnostics and decisions at each point.",
        )

    with col_ctrl2:
        st.markdown(
            f"""
            <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; margin-top: 5px; text-align: center;">
                <div style="font-size: 10px; color: #8b949e; text-transform: uppercase;">Mission Time</div>
                <div style="font-size: 18px; font-weight: 700; color: #58a6ff;">T+{step_idx * session.dt:.1f}s <span style="font-size: 12px; color: #8b949e;">({step_idx + 1}/{session.total_steps})</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Current Frame Payload & Adaptation
    current_payload = session.get_step(step_idx)
    adapter = DashboardAdapter()
    vm: DashboardViewModel = adapter.adapt(current_payload)

    # Mission & Provenance Ribbon
    clean_phase = str(current_payload.mission_phase).replace('_', ' ').title()
    clean_anomaly = format_status(current_payload.anomaly_status)
    st.markdown(
        f"""
        <div style="background: linear-gradient(90deg, #161b22 0%, #21262d 100%); border: 1px solid #30363d; border-radius: 6px; padding: 8px 14px; margin-bottom: 15px; display: flex; justify-content: space-between; font-size: 12px; color: #c9d1d9;">
            <div><b>Engine ID:</b> <code>{current_payload.engine_id}</code> | <b>Mission ID:</b> <code>{current_payload.mission_id}</code> | <b>Phase:</b> <code>{clean_phase}</code></div>
            <div><b>Anomaly Status:</b> <code style="color: {'#f85149' if current_payload.anomaly_status in ('WARNING', 'ANOMALY') else '#3fb950'};">{clean_anomaly}</code> | <b>Health Index:</b> <code>{current_payload.smoothed_health_index:.3f}</code></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Synchronized Metric Cards (Telemetry, Diagnosis, Health, RUL)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        rpm_val = current_payload.observed_telemetry.get("rpm", float("nan"))
        st.metric("Engine Speed (RPM)", format_value(rpm_val, decimals=0, unit=" rpm"))
    with c2:
        cht_val = current_payload.observed_telemetry.get("cht", float("nan"))
        st.metric("Cylinder Head Temp (CHT)", format_value(cht_val, decimals=1, unit=" °C"))
    with c3:
        egt_val = current_payload.observed_telemetry.get("egt", float("nan"))
        st.metric("Exhaust Gas Temp (EGT)", format_value(egt_val, decimals=1, unit=" °C"))
    with c4:
        hi_val = current_payload.smoothed_health_index
        st.metric("Health Index", format_value(hi_val, decimals=3))
    with c5:
        rul_val = current_payload.point_rul_seconds
        st.metric(
            "Projected RUL",
            format_rul(rul_val),
            help="RUL unavailable — insufficient continuous history for a valid prognostic estimate. The system withholds RUL rather than extrapolating from insufficient evidence." if rul_val is None else None,
        )

    # Synchronized Advisory Banner
    if current_payload.advisory is not None:
        adv = current_payload.advisory
        urgency_label = adv.urgency.title() if adv.urgency else "Routine"
        st.markdown(
            f"""
            <div style="background-color: #161b22; border-left: 4px solid #d29922; border: 1px solid #30363d; border-radius: 6px; padding: 10px 14px; margin: 15px 0;">
                <div style="font-size: 11px; font-weight: 700; color: #8b949e;">DECISION SUPPORT: <code>{format_action(adv.action_code)}</code> (Urgency: {urgency_label})</div>
                <div style="font-size: 14px; font-weight: 600; color: #f0f6fc; margin: 4px 0;">{adv.headline}</div>
                <div style="font-size: 12px; color: #c9d1d9;"><b>Action:</b> {adv.recommended_action}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Synchronized History Timeline Plot
    st.markdown("#### Engine Health Timeline")
    history = session.get_history(step_idx)
    ts = history.get("timestamps", [])

    if ts:
        fig = go.Figure()

        # CHT trace
        fig.add_trace(go.Scatter(
            x=ts,
            y=history.get("cht", []),
            mode="lines+markers",
            name="CHT (°C)",
            line=dict(color="#58a6ff", width=2),
            marker=dict(size=4),
        ))

        # EGT trace / 5 to fit scale nicely or dual axis
        fig.add_trace(go.Scatter(
            x=ts,
            y=[y / 5.0 for y in history.get("egt", [])],
            mode="lines",
            name="EGT / 5 (°C)",
            line=dict(color="#f0883e", width=1.5, dash="dot"),
        ))

        # Health Index trace * 100 on secondary scale
        fig.add_trace(go.Scatter(
            x=ts,
            y=[y * 100.0 for y in history.get("health_index", {}).get("hi", [])],
            mode="lines",
            name="Health Index (%)",
            line=dict(color="#3fb950", width=2),
        ))

        fig.update_layout(
            paper_bgcolor="#161b22",
            plot_bgcolor="#0d1117",
            font=dict(color="#c9d1d9", family="sans-serif", size=11),
            margin=dict(l=40, r=20, t=30, b=30),
            height=300,
            xaxis=dict(title="Flight Time Elapsed (s)", gridcolor="#21262d"),
            yaxis=dict(title="Scaled Units", gridcolor="#21262d"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)
