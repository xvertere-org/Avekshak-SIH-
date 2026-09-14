"""
Data Quality and System Status Component for SIH26054 Dashboard.
Renders sensor availability matrix, missingness, invalid observations,
sampling regularity, and artifact provenance.
"""

from typing import Dict, Any, List
import streamlit as st
from dashboard.schemas.view_model import DataQualityViewModel, StatusLevel, AvailabilityStatus
from dashboard.utils.formatters import (
    format_value,
    format_percent,
    format_channel,
    format_data_quality_status,
)
from dashboard.utils.styles import STATUS_COLORS, render_status_badge


def render_quality_summary(quality: DataQualityViewModel):
    """Render high-level data quality KPIs and issue summary."""
    st.markdown("#### Data Quality Summary")

    cols = st.columns(4)
    with cols[0]:
        st.metric(
            label="Overall Quality Score",
            value=format_percent(quality.quality_score),
        )
    with cols[1]:
        st.metric(
            label="Sampling Regularity",
            value="Regular (10 Hz)" if quality.is_regular_sampling else "Irregular",
        )
    with cols[2]:
        st.metric(
            label="Average Sampling Interval",
            value=format_value(quality.sampling_interval_mean, decimals=3, unit="s"),
        )
    with cols[3]:
        st.metric(
            label="Active Integrity Issues",
            value=str(len(quality.issues_detected)),
        )

    if quality.issues_detected:
        st.warning("⚠️ **Detected Data Quality Issues:**\n- " + "\n- ".join(quality.issues_detected))


def render_sensor_status_matrix(quality: DataQualityViewModel):
    """Render table of channel-level sensor statuses."""
    st.markdown("#### Sensor Status")

    if not quality.channel_summaries:
        st.info("ℹ️ Granular channel quality summaries currently unavailable.")
        return

    table_data = []
    for ch, sum_data in quality.channel_summaries.items():
        missing = sum_data.get("missing_count", 0)
        invalid = sum_data.get("invalid_count", 0)
        warning = sum_data.get("warning_count", 0)
        isolated = "Yes" if ch in quality.isolated_sensors else "No"

        if missing > 0:
            status = "DROPOUT"
        elif invalid > 0:
            status = "PHYSICALLY_INVALID"
        elif isolated == "Yes":
            status = "ISOLATED_BY_PHM"
        elif warning > 0:
            status = "WARNING_ENVELOPE"
        else:
            status = "NOMINAL"

        table_data.append({
            "Sensor": format_channel(ch),
            "Integrity Status": format_data_quality_status(status),
            "Missing Count": missing,
            "Invalid Count": invalid,
            "Caution Count": warning,
            "Channel Isolated": isolated,
        })

    st.dataframe(table_data, use_container_width=True, hide_index=True)


def render_provenance_card(quality: DataQualityViewModel):
    """Render telemetry and model provenance details."""
    st.markdown("#### System Information")

    dq = quality
    engine_raw = dq.provenance.get('pipeline', 'SystemPipelineOrchestrator')
    engine_clean = "Avekshak Real-Time Digital Twin Pipeline" if engine_raw == "SystemPipelineOrchestrator" else engine_raw
    source_clean = "Synthetic Aero-Engine Test Bench" if "synthetic" in dq.provenance_source.lower() else dq.provenance_source

    prov_html = (
        f'<div class="console-panel" style="padding: 12px 14px;">'
        f'<div style="font-size: 12px; color: #c9d1d9; line-height: 1.8;">'
        f'<div><span style="color: #8b949e;">Analysis Engine:</span> <b style="color: #58a6ff;">{engine_clean}</b></div>'
        f'<div><span style="color: #8b949e;">Data Feed Source:</span> <b style="color: #f0f6fc;">{source_clean}</b></div>'
        f'<div><span style="color: #8b949e;">Pipeline Latency:</span> <code style="color: #3fb950;">{dq.execution_latency_ms:.2f} ms</code></div>'
        f'<div><span style="color: #8b949e;">Stream Quality:</span> <b style="color: #f0f6fc;">{format_data_quality_status(dq.quality_status)}</b></div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(prov_html, unsafe_allow_html=True)

