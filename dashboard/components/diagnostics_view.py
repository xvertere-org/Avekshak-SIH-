"""
Diagnostics View Component for SIH26054 Dashboard.
Renders anomaly breakdown, fault diagnosis probabilities, digital twin residuals,
and sensor fault isolation status.
"""

from typing import Dict, Any, List
import streamlit as st
import plotly.graph_objects as go
from dashboard.schemas.view_model import DiagnosticsViewModel, StatusLevel, AvailabilityStatus
from dashboard.utils.formatters import format_value, format_fault_name, format_percent
from dashboard.utils.styles import PLOT_COLORS, STATUS_COLORS, render_status_badge


def render_anomaly_diagnostics(diag: DiagnosticsViewModel):
    """Render anomaly detection scores and detector decomposition."""
    st.markdown("#### Phase 7 Hybrid Anomaly Detection")

    if diag.anomaly_status == "Unavailable":
        st.info("ℹ️ Anomaly detection outputs currently unavailable.")
        return

    cols = st.columns(4)
    with cols[0]:
        st.metric(
            label="Active Anomaly Status",
            value=diag.anomaly_status,
        )
    with cols[1]:
        st.metric(
            label="Composite Anomaly Score",
            value=format_value(diag.anomaly_score, decimals=2),
        )
    with cols[2]:
        threshold_score = diag.detector_scores.get("threshold")
        st.metric(
            label="Threshold Score",
            value=format_value(threshold_score, decimals=2),
        )
    with cols[3]:
        ewma_score = diag.detector_scores.get("ewma")
        st.metric(
            label="EWMA Score",
            value=format_value(ewma_score, decimals=2),
        )

    if diag.contributing_channels:
        st.markdown(
            f"**Contributing Degradation Channels:** `"
            + "`, `".join(diag.contributing_channels)
            + "`"
        )


def render_fault_classification(diag: DiagnosticsViewModel):
    """Render fault classification and class probabilities."""
    st.markdown("#### Phase 8 Multi-Class Fault Diagnosis")

    if diag.predicted_fault == "Unavailable":
        st.info("ℹ️ Fault diagnosis classification currently unavailable.")
        return

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown(
            f"""
            <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px;">
                <div style="font-size: 11px; color: #8b949e; text-transform: uppercase;">Predicted Fault Class</div>
                <div style="font-size: 20px; font-weight: 700; color: #f0f6fc; margin: 8px 0;">
                    {format_fault_name(diag.predicted_fault_class)}
                </div>
                <div style="font-size: 13px; color: #8b949e; margin-bottom: 4px;">
                    Diagnosis Probability: <b>{format_percent(diag.diagnostic_confidence)}</b>
                </div>
                <div style="font-size: 12px; color: #8b949e; margin-bottom: 8px;">
                    Data Quality: <code>{diag.diagnosis_data_quality}</code>
                </div>
                <div style="margin-top: 10px; border-top: 1px solid #21262d; padding-top: 8px; font-size: 12px;">
                    {"⚠️ <b>Sensor Fault Indicated:</b> Channel isolated from physical Twin" if diag.sensor_fault_indicated else "✅ Physical evidence consistent with engine state"}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        if diag.class_probabilities:
            # Horizontal bar chart of diagnosis probabilities
            sorted_probs = sorted(diag.class_probabilities.items(), key=lambda x: x[1], reverse=True)
            labels = [format_fault_name(k) for k, v in sorted_probs]
            values = [v * 100.0 for k, v in sorted_probs]

            fig = go.Figure(go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker=dict(
                    color=["#f85149" if i == 0 and labels[0] != "Nominal / None" else "#58a6ff" for i in range(len(labels))],
                ),
            ))
            fig.update_layout(
                title=dict(text="Diagnosis Probability Distribution (%)", font=dict(size=12, color=PLOT_COLORS["text"])),
                margin=dict(l=10, r=20, t=30, b=20),
                height=200,
                paper_bgcolor=PLOT_COLORS["paper_bg"],
                plot_bgcolor=PLOT_COLORS["plot_bg"],
                font=dict(color=PLOT_COLORS["text"], size=10),
                xaxis=dict(gridcolor=PLOT_COLORS["grid"], range=[0, 100], title="Diagnosis Probability (%)"),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_residual_table(diag: DiagnosticsViewModel):
    """Render table of digital twin expected values, residuals, and normalized excursions."""
    st.markdown("#### Phase 6 Digital Twin Expected States & Residuals")

    if not diag.residuals and not diag.expected_telemetry:
        st.info("ℹ️ Digital Twin state residuals currently unavailable.")
        return

    # Table of expected telemetry, raw residuals, and normalized residuals
    rows = []
    # Combine channel keys from residuals and expected
    all_channels = sorted(set(list(diag.residuals.keys()) + list(diag.expected_telemetry.keys())))
    for k in all_channels:
        clean_name = k.replace("_residual", "").replace("_norm", "").replace("expected_", "").upper()
        exp_val = diag.expected_telemetry.get(f"expected_{k.lower()}", diag.expected_telemetry.get(k.lower()))
        res_val = diag.residuals.get(f"{k.lower()}_residual", diag.residuals.get(k.lower()))
        norm_val = diag.normalized_residuals.get(f"{k.lower()}_norm", diag.normalized_residuals.get(k.lower()))

        rows.append({
            "Channel": clean_name,
            "Expected Value": format_value(exp_val, decimals=2),
            "Raw Residual": format_value(res_val, decimals=3),
            "Normalized Residual (σ)": format_value(norm_val, decimals=2),
        })

    st.dataframe(rows, use_container_width=True, hide_index=True)
