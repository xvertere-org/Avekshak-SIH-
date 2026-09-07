"""
Prognostics View Component for SIH26054 Dashboard.
Renders Health Index, degradation rate/trend, Remaining Useful Life (RUL),
uncertainty bounds, and future trajectory forecasts.
"""

from typing import Dict, Any, List, Optional
import streamlit as st
import plotly.graph_objects as go
from dashboard.schemas.view_model import PrognosticsViewModel, StatusLevel, AvailabilityStatus
from dashboard.utils.formatters import (
    format_value,
    format_health_index,
    format_rul,
    format_percent,
    map_health_to_status,
    map_rul_status_to_status,
)
from dashboard.utils.styles import PLOT_COLORS, render_status_badge


def render_health_prognostics(
    prog: PrognosticsViewModel,
    history_timestamps: Optional[List[float]] = None,
    history_hi: Optional[List[float]] = None,
):
    """Render current health index, degradation rate, and historical trajectory."""
    st.markdown("#### Health Index & Degradation Velocity")

    cols = st.columns(4)
    with cols[0]:
        st.metric(
            label="Smoothed Health Index",
            value=format_health_index(prog.health_index),
            help="Causally filtered Health Index on [0.0, 1.0] scale",
        )
    with cols[1]:
        st.metric(
            label="Operational Health State",
            value=prog.health_state,
        )
    with cols[2]:
        rate_str = f"{prog.degradation_rate * 1e4:.2f} ×10⁻⁴ s⁻¹" if prog.degradation_rate is not None else "Unavailable"
        st.metric(
            label="Degradation Rate (dHI/dt)",
            value=rate_str,
        )
    with cols[3]:
        st.metric(
            label="Degradation Trend",
            value=prog.degradation_trend,
        )

    # Health Index history chart
    if history_timestamps and history_hi:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history_timestamps,
            y=history_hi,
            mode="lines+markers",
            name="Health Index (HI)",
            line=dict(color="#3fb950", width=2.5),
            marker=dict(size=4),
        ))
        # Threshold lines (Critical: 0.35, Degraded: 0.60, Healthy: 0.85)
        fig.add_hline(y=0.85, line_dash="dash", line_color="#2ea043", annotation_text="Healthy Threshold (0.85)")
        fig.add_hline(y=0.60, line_dash="dash", line_color="#d29922", annotation_text="Degraded Threshold (0.60)")
        fig.add_hline(y=0.35, line_dash="dash", line_color="#f85149", annotation_text="Critical / EOL Threshold (0.35)")

        fig.update_layout(
            title=dict(text="Historical Health Index Trajectory", font=dict(size=12, color=PLOT_COLORS["text"])),
            margin=dict(l=40, r=20, t=30, b=30),
            height=240,
            paper_bgcolor=PLOT_COLORS["paper_bg"],
            plot_bgcolor=PLOT_COLORS["plot_bg"],
            font=dict(color=PLOT_COLORS["text"], size=10),
            xaxis=dict(gridcolor=PLOT_COLORS["grid"], title="Time (s)"),
            yaxis=dict(gridcolor=PLOT_COLORS["grid"], range=[0.0, 1.05], title="Health Index"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_rul_panel(prog: PrognosticsViewModel):
    """Render RUL estimation, confidence intervals, and EOL limiting factor."""
    st.markdown("#### Remaining Useful Life (RUL) Prognostics")

    if prog.rul_state == "Unavailable" and prog.rul_status == "Unavailable":
        st.info("ℹ️ Prognostics RUL estimation currently unavailable.")
        return

    cols = st.columns(4)
    with cols[0]:
        rul_display = f"{prog.rul_hours:.1f} hrs ({prog.point_rul_seconds:.0f}s)" if prog.rul_hours is not None else prog.rul_state
        st.metric(
            label="Median RUL Estimate",
            value=rul_display,
        )
    with cols[1]:
        bounds_str = "Unavailable"
        if prog.rul_p05_hours is not None and prog.rul_p95_hours is not None:
            bounds_str = f"[{prog.rul_p05_hours:.1f}h - {prog.rul_p95_hours:.1f}h]"
        st.metric(
            label="Uncertainty Range [P05 - P95]",
            value=bounds_str,
        )
    with cols[2]:
        st.metric(
            label="RUL State / Limiting Factor",
            value=f"{prog.rul_state} / {prog.limiting_factor or 'NONE'}",
        )
    with cols[3]:
        st.metric(
            label="Forecast-Assisted Mode",
            value="ACTIVE" if prog.forecast_assisted_mode else "OFF (Causal Trend)",
        )

    # EOL Provenance & Disclaimer box
    eol_info = ""
    if prog.eol_provenance:
        eol_info = f" | <b>EOL Provenance:</b> <code>{prog.eol_provenance.get('method', 'Weibull-Degradation')}</code>"

    eol_html = (
        f'<div style="background-color: #11151c; border: 1px solid #21262d; '
        f'border-radius: 4px; padding: 12px 14px; margin-top: 10px; font-size: 12px; color: #8b949e;">'
        f'<b>Backend RUL State:</b> <code style="color: #f0f6fc;">{prog.rul_state}</code> | '
        f'<b>Limiting Factor:</b> <code style="color: #f0f6fc;">{prog.limiting_factor}</code>{eol_info}'
        f'<div style="margin-top: 6px; color: #d29922;">'
        f'⚠️ <b>Airworthiness Disclaimer:</b> End-of-Life (EOL) criteria and redlines are project-defined simulated criteria, NOT certified OEM or FAA flight airworthiness limits.'
        f'</div>'
        f'</div>'
    )
    st.markdown(eol_html, unsafe_allow_html=True)



def render_forecast_panel(prog: PrognosticsViewModel):
    """Render telemetry forecasting status, source, and forecast trajectories."""
    st.markdown("#### Telemetry Forecasting (TimesFM / Baseline)")


    # Status / Source callout box distinguishing LOADED_PRETRAINED vs BLOCKED_UNAUTHENTICATED_GATED
    if prog.forecast_status == "BLOCKED_UNAUTHENTICATED_GATED":
        st.warning(
            "⚠️ **TimesFM Pretrained Weights:** `BLOCKED_UNAUTHENTICATED_GATED` (Google Cloud authentication gated / unavailable). "
            "Causal baseline fallback is active — **NOT a TimesFM foundation model prediction**."
        )
    elif prog.forecast_status == "LOCAL_UNCHECKPOINTED_GRAPH":
        st.error(
            "⚠️ **Untrained Architecture:** `LOCAL_UNCHECKPOINTED_GRAPH` detected. Untrained local graph weights are rejected and must never be presented as a usable flight forecast."
        )
    elif prog.forecast_status == "LOADED_PRETRAINED":
        st.success(
            "✅ **Pretrained Foundation Model:** `LOADED_PRETRAINED` — TimesFM-3 inference active."
        )
    elif prog.forecast_status == "BUFFERING":
        st.info(
            f"ℹ️ **Buffering Forecast Context:** Accumulating timesteps for horizon H={prog.forecast_horizon}."
        )
    else:
        st.info(f"Forecast Status: `{prog.forecast_status}` | Source: `{prog.forecast_source}`")

    f_cols = st.columns(4)
    with f_cols[0]:
        st.caption("FORECAST STATUS")
        st.markdown(f"**`{prog.forecast_status}`**")
    with f_cols[1]:
        st.caption("FORECAST SOURCE")
        st.markdown(f"**`{prog.forecast_source}`**")
    with f_cols[2]:
        st.caption("FORECAST HORIZON")
        st.markdown(f"**`{prog.forecast_horizon} steps`**")
    with f_cols[3]:
        st.caption("FORECAST QUALITY")
        st.markdown(f"**`{prog.forecast_quality}`**")

    # Render multi-channel predicted curves if predicted_telemetry is present
    if prog.predicted_telemetry and prog.forecast_timestamps:
        fig = go.Figure()
        for ch, vals in prog.predicted_telemetry.items():
            if vals and len(vals) == len(prog.forecast_timestamps):
                fig.add_trace(go.Scatter(
                    x=prog.forecast_timestamps,
                    y=vals,
                    mode="lines",
                    name=f"Forecast {ch.upper()}",
                ))
        fig.update_layout(
            title=dict(text=f"Future Telemetry Forecast ({prog.forecast_source})", font=dict(size=12, color=PLOT_COLORS["text"])),
            margin=dict(l=40, r=20, t=30, b=30),
            height=240,
            paper_bgcolor=PLOT_COLORS["paper_bg"],
            plot_bgcolor=PLOT_COLORS["plot_bg"],
            font=dict(color=PLOT_COLORS["text"], size=10),
            xaxis=dict(gridcolor=PLOT_COLORS["grid"], title="Future Timestamp (s)"),
            yaxis=dict(gridcolor=PLOT_COLORS["grid"], title="Forecast Sensor Value"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.caption("Telemetry trajectory forecast arrays buffering or currently unavailable.")


# Backward compatibility alias
render_projected_trajectory = render_forecast_panel
