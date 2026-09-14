"""
Telemetry Charts Component for SIH26054 Dashboard.
Renders interactive Plotly visualizers for the canonical 7 channels and operating context.
Strictly distinguishes observed, digital twin expected, and forecast data.
"""

from typing import Dict, Any, List, Optional
import streamlit as st
import plotly.graph_objects as go
from dashboard.schemas.view_model import (
    TelemetryViewModel,
    ChannelTelemetryModel,
    CANONICAL_CHANNELS,
    CANONICAL_CHANNEL_METADATA,
    AvailabilityStatus,
)
from dashboard.utils.formatters import format_value
from dashboard.utils.styles import PLOT_COLORS, render_status_badge


def render_operating_context_cards(telemetry_vm: TelemetryViewModel):
    """Render top operating context indicators (throttle, load, altitude, ambient temp)."""
    cols = st.columns(4)
    with cols[0]:
        st.metric(
            label="Throttle",
            value=format_value(telemetry_vm.throttle, decimals=1, unit="%"),
        )
    with cols[1]:
        st.metric(
            label="Engine Load",
            value=format_value(telemetry_vm.load, decimals=1, unit="%"),
        )
    with cols[2]:
        st.metric(
            label="Altitude",
            value=format_value(telemetry_vm.altitude, decimals=0, unit="m"),
        )
    with cols[3]:
        st.metric(
            label="Ambient Temperature",
            value=format_value(telemetry_vm.ambient_temp, decimals=1, unit="°C"),
        )


def build_channel_figure(
    model: ChannelTelemetryModel,
    history_timestamps: Optional[List[float]] = None,
    history_observed: Optional[List[float]] = None,
    history_expected: Optional[List[float]] = None,
) -> go.Figure:
    """
    Build an individual channel line chart clearly separating:
    - Observed Telemetry (solid blue)
    - Digital Twin Expected (dashed green)
    - Future Forecast (dotted purple with uncertainty band)
    """
    fig = go.Figure()

    # 1. Historical & Current Observed Telemetry
    if history_timestamps and history_observed:
        fig.add_trace(go.Scatter(
            x=history_timestamps,
            y=history_observed,
            mode="lines+markers",
            name="Observed",
            line=dict(color=PLOT_COLORS["observed"], width=2.5),
            marker=dict(size=4),
        ))
    elif model.observed_value is not None:
        fig.add_trace(go.Scatter(
            x=[0],
            y=[model.observed_value],
            mode="markers",
            name="Observed",
            marker=dict(color=PLOT_COLORS["observed"], size=9, symbol="circle"),
        ))

    # 2. Historical & Current Digital Twin Expected (Physics Estimate)
    if history_timestamps and history_expected:
        fig.add_trace(go.Scatter(
            x=history_timestamps,
            y=history_expected,
            mode="lines",
            name="Physics estimate",
            line=dict(color=PLOT_COLORS["expected"], width=2, dash="dash"),
        ))
    elif model.physics_estimate is not None or model.expected_value is not None:
        val = model.physics_estimate if model.physics_estimate is not None else model.expected_value
        fig.add_trace(go.Scatter(
            x=[0],
            y=[val],
            mode="markers",
            name="Physics estimate",
            marker=dict(color=PLOT_COLORS["expected"], size=8, symbol="diamond"),
        ))

    # 2b. Corrected Prediction Marker (Sensor-informed Grey-Box)
    if model.corrected_prediction is not None and model.observed_value is None:
        fig.add_trace(go.Scatter(
            x=[0],
            y=[model.corrected_prediction],
            mode="markers",
            name="Corrected prediction",
            marker=dict(color="#56d364", size=8, symbol="star"),
        ))


    # 3. Future Forecast Trajectory (Phase 10)
    if model.forecast_values and model.forecast_timestamps:
        # Uncertainty band (P05 to P95) if supplied
        if model.forecast_lower_bounds and model.forecast_upper_bounds:
            fig.add_trace(go.Scatter(
                x=model.forecast_timestamps + model.forecast_timestamps[::-1],
                y=model.forecast_upper_bounds + model.forecast_lower_bounds[::-1],
                fill="toself",
                fillcolor=PLOT_COLORS["forecast_bound"],
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
                name="Confidence Bound",
            ))

        fig.add_trace(go.Scatter(
            x=model.forecast_timestamps,
            y=model.forecast_values,
            mode="lines+markers",
            name="Forecast (TimesFM)",
            line=dict(color=PLOT_COLORS["forecast"], width=2, dash="dot"),
            marker=dict(size=4),
        ))

    # Operational Redline Limits from Metadata
    meta = CANONICAL_CHANNEL_METADATA.get(model.channel, {})
    if "critical_max" in meta:
        fig.add_hline(
            y=meta["critical_max"],
            line_dash="dot",
            line_color=PLOT_COLORS["critical_limit"],
            annotation_text=f"Max Limit ({meta['critical_max']} {model.unit})",
            annotation_position="top right",
            annotation_font=dict(size=9, color=PLOT_COLORS["critical_limit"]),
        )
    if "critical_min" in meta:
        fig.add_hline(
            y=meta["critical_min"],
            line_dash="dot",
            line_color=PLOT_COLORS["critical_limit"],
            annotation_text=f"Min Limit ({meta['critical_min']} {model.unit})",
            annotation_position="bottom right",
            annotation_font=dict(size=9, color=PLOT_COLORS["critical_limit"]),
        )

    fig.update_layout(
        title=dict(
            text=f"<b>{model.display_name}</b> ({model.unit})",
            font=dict(size=13, color=PLOT_COLORS["text"]),
        ),
        margin=dict(l=40, r=20, t=35, b=30),
        height=240,
        paper_bgcolor=PLOT_COLORS["paper_bg"],
        plot_bgcolor=PLOT_COLORS["plot_bg"],
        font=dict(color=PLOT_COLORS["text"], size=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=9),
        ),
        xaxis=dict(
            gridcolor=PLOT_COLORS["grid"],
            zeroline=False,
            title="Time (s)",
        ),
        yaxis=dict(
            gridcolor=PLOT_COLORS["grid"],
            zeroline=False,
        ),
    )
    return fig


def render_canonical_telemetry_grid(
    telemetry_vm: TelemetryViewModel,
    history_data: Optional[Dict[str, Dict[str, List[float]]]] = None,
):
    """
    Render 7 canonical telemetry channels with comparison metrics and charts.
    """
    st.markdown("#### Engine Sensor Channels")

    # Grid layout: 2 columns for the first 6, then full-width for the 7th or 3-column layout
    cols = st.columns(2)
    for idx, ch in enumerate(CANONICAL_CHANNELS):
        col = cols[idx % 2]
        ch_model = telemetry_vm.channels.get(ch)
        if ch_model is None:
            continue

        with col:
            # Top summary row for the channel
            hdr_col1, hdr_col2 = st.columns([3, 1])
            with hdr_col1:
                phys_val = ch_model.physics_estimate if ch_model.physics_estimate is not None else ch_model.expected_value
                phys_str = format_value(phys_val, decimals=1, unit=ch_model.unit)
                corr_str = format_value(ch_model.corrected_prediction, decimals=1, unit=ch_model.unit)
                sc_val = ch_model.sensor_correction if ch_model.sensor_correction is not None else ch_model.residual
                sc_str = format_value(sc_val, decimals=2, unit=ch_model.unit)
                dev_val = ch_model.detected_deviation if ch_model.detected_deviation is not None else ch_model.residual
                dev_str = format_value(dev_val, decimals=2, unit=ch_model.unit)
                conf_str = f"{ch_model.model_confidence * 100.0:.0f}%" if ch_model.model_confidence is not None else "100%"
                pi_str = ""
                if ch_model.prediction_lower is not None and ch_model.prediction_upper is not None:
                    pi_str = f" | Prediction range: <b style='color: #79c0ff; font-family: monospace;'>[{ch_model.prediction_lower:.1f} – {ch_model.prediction_upper:.1f}]</b>"

                hdr_html = (
                    f'<div style="font-size: 13px; font-weight: 700; color: #f0f6fc;">{ch_model.display_name}</div>'
                    f'<div style="font-size: 11px; color: #8b949e; margin-bottom: 2px;">'
                    f'Physics estimate: <b style="color: {PLOT_COLORS["expected"]}; font-family: monospace;">{phys_str}</b> | '
                    f'Sensor-informed correction: <b style="color: #79c0ff; font-family: monospace;">{sc_str}</b> | '
                    f'Corrected prediction: <b style="color: #56d364; font-family: monospace;">{corr_str}</b>'
                    f'</div>'
                    f'<div style="font-size: 10px; color: #8b949e; margin-bottom: 4px;">'
                    f'Detected deviation: <b style="color: #f0f6fc; font-family: monospace;">{dev_str}</b> | '
                    f'Model confidence: <b style="color: #e6edf3; font-family: monospace;">{conf_str}</b>'
                    f'{pi_str}'
                    f'</div>'
                )
                st.markdown(hdr_html, unsafe_allow_html=True)

            with hdr_col2:
                status_text = "ISOLATED" if ch_model.is_isolated else ch_model.status.value
                st.markdown(
                    f"<div style='text-align: right;'>{render_status_badge(ch_model.status, status_text)}</div>",
                    unsafe_allow_html=True,
                )


            # Chart
            ch_hist = history_data.get(ch, {}) if history_data else {}
            fig = build_channel_figure(
                model=ch_model,
                history_timestamps=ch_hist.get("timestamps"),
                history_observed=ch_hist.get("observed"),
                history_expected=ch_hist.get("expected"),
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
