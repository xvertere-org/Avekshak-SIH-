"""
Prognostics View Component for SIH26054 Dashboard.
Renders Health Index, degradation rate/trend, Remaining Useful Life (RUL),
engineering uncertainty bounds, and future trajectory forecasts.
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
    format_health_state,
    format_health_trend,
    format_rul_state,
    format_limiting_factor,
    format_forecast_status,
    format_forecast_quality,
    format_channel,
    map_health_to_status,
    map_rul_status_to_status,
)
from dashboard.utils.styles import PLOT_COLORS


def render_health_prognostics(
    prog: PrognosticsViewModel,
    history_timestamps: Optional[List[float]] = None,
    history_hi: Optional[List[float]] = None,
):
    """Render current health index, degradation rate, and historical trajectory."""
    st.markdown("#### Engine Health & Degradation")

    hi_val = prog.engine_health_score if prog.engine_health_score is not None else prog.health_index
    trend_display = prog.trend_state if prog.trend_state != "Unavailable" else prog.degradation_trend

    cols = st.columns(4)
    with cols[0]:
        st.metric(
            label="Engine Health",
            value=format_health_index(hi_val),
            help="Bounded engineering health indicator on [0.0, 1.0]. 1.0 = nominal baseline, 0.0 = functional failure threshold (HI <= 0.35).",
        )
    with cols[1]:
        st.metric(
            label="Engine Condition",
            value=format_health_state(prog.health_state),
        )
    with cols[2]:
        rate_str = f"{prog.degradation_rate * 6000.0:.2f} %/min" if prog.degradation_rate is not None else "Steady"
        st.metric(
            label="Decline Rate",
            value=rate_str,
            help="Estimated rate of health decline per minute of mission operation.",
        )
    with cols[3]:
        st.metric(
            label="Degradation Trend",
            value=format_health_trend(prog.degradation_trend),
        )

    # Subsystem Health Breakdown if present
    if prog.subsystem_scores:
        st.markdown("<div style='font-size: 11px; font-weight: 600; color: #8b949e; margin-top: 10px; margin-bottom: 4px; text-transform: uppercase;'>Subsystem Health Indicators</div>", unsafe_allow_html=True)
        sub_cols = st.columns(len(prog.subsystem_scores))
        for idx, (sub_name, sub_score) in enumerate(prog.subsystem_scores.items()):
            with sub_cols[idx]:
                st.metric(
                    label=sub_name.capitalize(),
                    value=f"{sub_score:.2f}",
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
    """Render the conditional RUL estimate and engineering uncertainty range."""
    st.markdown("#### Life Prediction (Remaining Useful Life)")

    if prog.rul_hours is None or prog.rul_state == "UNAVAILABLE":
        reason_text = prog.prognostics_reason or "Insufficient continuous monotonic degradation history for a valid prognostic estimate."
        st.info(f"ℹ️ **RUL unavailable — {reason_text}** The system withholds RUL rather than extrapolating unsupported estimates.")

    cols = st.columns(4)
    with cols[0]:
        rul_display = f"{prog.rul_hours:.1f} hrs ({prog.point_rul_seconds / 60.0:.0f} min)" if prog.rul_hours is not None else format_rul_state(prog.rul_state)
        st.metric(
            label="Conditional RUL estimate",
            value=rul_display,
            help="Shown only when there is sufficient degradation history and valid telemetry evidence.",
        )
    with cols[1]:
        bounds_str = "Unavailable"
        if prog.rul_p05_hours is not None and prog.rul_p95_hours is not None:
            bounds_str = f"[{prog.rul_p05_hours:.1f}h - {prog.rul_p95_hours:.1f}h]"
        st.metric(
            label="Engineering uncertainty interval",
            value=bounds_str,
            help="Engineering uncertainty estimate covering 90% of simulated outcomes (P05–P95).",
        )
    with cols[2]:
        st.metric(
            label="Life Prediction Status",
            value=format_rul_state(prog.rul_state),
        )
    with cols[3]:
        st.metric(
            label="Limiting Factor",
            value=format_limiting_factor(prog.limiting_factor),
        )

    # EOL Provenance & Disclaimer box
    eol_info = ""
    if prog.eol_provenance:
        eol_info = f" | <b>Model Provenance:</b> <code>{prog.eol_provenance.get('method', 'Empirical Degradation Projections')}</code>"

    eol_html = (
        f'<div style="background-color: #11151c; border: 1px solid #21262d; '
        f'border-radius: 4px; padding: 12px 14px; margin-top: 10px; font-size: 12px; color: #8b949e;">'
        f'<b>Prognostic Condition:</b> <b style="color: #f0f6fc;">{format_rul_state(prog.rul_state)}</b> | '
        f'<b>Limiting Threshold:</b> <b style="color: #58a6ff;">{format_limiting_factor(prog.limiting_factor)}</b>{eol_info}'
        f'<div style="margin-top: 6px; color: #d29922;">'
        f'⚠️ <b>Simulation Disclaimer:</b> Time-to-threshold calculations are engineering demonstrations on simulated degradation scenarios. These are not certified OEM or regulatory airworthiness limits.'
        f'</div>'
        f'</div>'
    )
    st.markdown(eol_html, unsafe_allow_html=True)


def render_forecast_panel(prog: PrognosticsViewModel):
    """Render telemetry forecasting status, source, and forecast trajectories."""
    st.markdown("#### Telemetry Forecast")

    # Status / Source callout box distinguishing LOADED_PRETRAINED vs BLOCKED_UNAUTHENTICATED_GATED
    if prog.forecast_status == "BLOCKED_UNAUTHENTICATED_GATED":
        st.warning(
            "⚠️ **Advanced forecast unavailable.** Pretrained weights are not loaded. "
            "Avekshak is using baseline persistence forecasting."
        )
    elif prog.forecast_status == "LOCAL_UNCHECKPOINTED_GRAPH":
        st.info(
            "ℹ️ **Local forecast graph active.** Graph initialized for local trajectory estimation."
        )
    elif prog.forecast_status == "LOADED_PRETRAINED":
        st.success(
            "✅ **Forecast model active.** Pretrained weights loaded — multi-step forecasting enabled."
        )
    elif prog.forecast_status == "BUFFERING":
        st.info(
            f"ℹ️ **Preparing forecast.** Accumulating {prog.forecast_horizon} timesteps before generating trajectory predictions."
        )
    else:
        st.info(f"Forecast method: `{prog.forecast_source}` (Status: `{format_forecast_status(prog.forecast_status)}`)")

    f_cols = st.columns(4)
    with f_cols[0]:
        st.caption("FORECAST METHOD")
        st.markdown(f"**{prog.forecast_source or 'Baseline Forecaster'}**")
    with f_cols[1]:
        st.caption("FORECAST STATUS")
        st.markdown(f"**{format_forecast_status(prog.forecast_status)}**")
    with f_cols[2]:
        st.caption("PREDICTION WINDOW")
        st.markdown(f"**{prog.forecast_horizon} timesteps**")
    with f_cols[3]:
        st.caption("FORECAST QUALITY")
        st.markdown(f"**{format_forecast_quality(prog.forecast_quality)}**")

    # Render multi-channel predicted curves if predicted_telemetry is present
    if prog.predicted_telemetry and prog.forecast_timestamps:
        fig = go.Figure()
        for ch, vals in prog.predicted_telemetry.items():
            if vals and len(vals) == len(prog.forecast_timestamps):
                channel_label = format_channel(ch)
                fig.add_trace(go.Scatter(
                    x=prog.forecast_timestamps,
                    y=vals,
                    mode="lines",
                    name=f"Forecast: {channel_label}",
                ))
        fig.update_layout(
            title=dict(text=f"Future Telemetry Forecast ({prog.forecast_source or 'Baseline'})", font=dict(size=12, color=PLOT_COLORS["text"])),
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
