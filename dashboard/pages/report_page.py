"""
Mission Engineering Report Page for SIH26054 Dashboard.
Renders executive summaries, operational peaks, PHM metrics, and provides
downloadable Markdown and JSON engineering reports.
"""

import sys
from pathlib import Path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from typing import Dict, Any, List, Optional
import streamlit as st

from orchestrator.schema import DashboardStatePayload
from phase14.reporting import MissionReportGenerator
from phase14.schema import MissionReportSummary
from dashboard.utils.formatters import (
    format_action,
    format_fault,
    format_limiting_factor,
    format_health_trend,
    format_forecast_quality,
    format_forecast_status,
    format_evidence_status,
    format_channel,
    format_error,
)


def render_report_page(
    payloads: List[DashboardStatePayload],
    scenario_metadata: Optional[Dict[str, Any]] = None,
):
    """Render 7. MISSION REPORT section."""
    st.markdown("### Mission Report")
    st.markdown(
        """
        <div style="font-size: 13px; color: #8b949e; margin-bottom: 16px;">
            Review the complete mission health assessment, key events, degradation trends, and recommended actions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not payloads:
        st.warning("No mission data available to generate a report. Run a mission from the sidebar first.")
        return

    # Generate Report
    try:
        report: MissionReportSummary = MissionReportGenerator.generate_report(
            payloads, scenario_metadata=scenario_metadata
        )
    except Exception as e:
        st.error(format_error("Mission report compilation failed", e))
        return

    # Download Buttons Bar
    col_dl1, col_dl2, col_meta = st.columns([1.5, 1.5, 3])
    with col_dl1:
        st.download_button(
            label="📥 Download Mission Report",
            data=report.to_markdown(),
            file_name=f"mission_report_{report.mission_id}.md",
            mime="text/markdown",
            help="Download the full mission report as a Markdown document.",
            use_container_width=True,
        )
    with col_dl2:
        st.download_button(
            label="📥 Download Mission Data (.JSON)",
            data=report.to_json(),
            file_name=f"mission_report_{report.mission_id}.json",
            mime="application/json",
            help="Download structured JSON mission data.",
            use_container_width=True,
        )
    with col_meta:
        st.markdown(
            f"""
            <div style="text-align: right; font-size: 11px; color: #8b949e; padding-top: 8px;">
                Engine: <code>{report.engine_id}</code> | Mission: <code>{report.mission_id}</code> | Samples: <code>{len(payloads)}</code>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Executive Advisory Assessment Banner
    adv_colors = {
        "GO": "#3fb950",
        "CAUTION": "#d29922",
        "MAINTENANCE": "#f85149",
        "INSUFFICIENT_DATA": "#8b949e",
    }
    adv_col = adv_colors.get(report.advisory_assessment, "#8b949e")
    clean_action = format_action(report.advisory_action_code)
    clean_urgency = report.advisory_urgency.title() if report.advisory_urgency else "Routine"

    st.markdown(
        f"""
        <div style="background-color: #161b22; border-left: 5px solid {adv_col}; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <div style="font-size: 11px; font-weight: 700; color: #8b949e; text-transform: uppercase;">
                    POST-MISSION ASSESSMENT — ACTION: <code>{clean_action}</code>
                </div>
                <span style="background-color: {adv_col}22; color: {adv_col}; border: 1px solid {adv_col}; padding: 2px 10px; border-radius: 4px; font-size: 12px; font-weight: 700;">
                    {format_action(report.advisory_assessment)} (Urgency: {clean_urgency})
                </span>
            </div>
            <div style="font-size: 16px; font-weight: 700; color: #f0f6fc; margin-bottom: 6px;">
                {report.advisory_headline}
            </div>
            <div style="font-size: 13px; color: #c9d1d9;">
                <b>Recommended Action:</b> {report.recommended_operator_action}
            </div>
            <div style="font-size: 10px; color: #8b949e; border-top: 1px solid #21262d; padding-top: 8px; margin-top: 8px;">
                ℹ️ <i>{report.disclaimer}</i>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Telemetry Peaks Summary
    st.markdown("#### 1. Engine Telemetry Summary")
    cols_t1, cols_t2, cols_t3, cols_t4 = st.columns(4)
    with cols_t1:
        st.metric("Peak CHT", f"{report.cht_peak:.1f} °C", help="Cylinder Head Temperature maximum")
        st.metric("Min Oil Pressure", f"{report.oil_pressure_min:.2f} bar", help="Minimum lubrication pressure")
    with cols_t2:
        st.metric("Peak EGT", f"{report.egt_peak:.1f} °C", help="Exhaust Gas Temperature maximum")
        st.metric("Max Vibration", f"{report.vibration_max:.3f} g", help="Peak mechanical vibration")
    with cols_t3:
        st.metric("Peak Oil Temp", f"{report.oil_temp_peak:.1f} °C", help="Maximum engine oil temperature")
        st.metric("RPM Range", f"{report.rpm_min:.0f} - {report.rpm_max:.0f}", help="Engine RPM boundary")
    with cols_t4:
        st.metric("Fuel Flow (Mean)", f"{report.fuel_flow_mean:.1f} L/h", help="Mean fuel flow rate")
        st.metric("Total Fuel Burn", f"{report.fuel_flow_total:.2f} L", help="Integrated total fuel consumption")

    st.markdown("---")

    # PHM & Prognostics Summary
    st.markdown("#### 2. Health & Prognostics")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.markdown(
            f"""
            <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px;">
                <div style="font-size: 11px; color: #8b949e; text-transform: uppercase;">Fault Diagnosis</div>
                <div style="font-size: 16px; font-weight: 700; color: #58a6ff; margin: 4px 0;">{format_fault(report.final_diagnosis)}</div>
                <div style="font-size: 11px; color: #c9d1d9;">
                    <b>Probability:</b> {report.final_diagnosis_probability:.3f}<br/>
                    <b>Evidence Quality:</b> {format_forecast_quality(report.evidence_quality)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_p2:
        st.markdown(
            f"""
            <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px;">
                <div style="font-size: 11px; color: #8b949e; text-transform: uppercase;">Engine Health & Degradation</div>
                <div style="font-size: 16px; font-weight: 700; color: #3fb950; margin: 4px 0;">Health Index: {report.final_health_index:.3f}</div>
                <div style="font-size: 11px; color: #c9d1d9;">
                    <b>Minimum Health Index:</b> {report.min_health_index:.3f}<br/>
                    <b>Degradation Trend:</b> {format_health_trend(report.degradation_trend)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_p3:
        rul_text = f"{report.final_rul_seconds:.1f} s" if report.final_rul_seconds is not None else "Unavailable"
        rul_desc = f"<b>Limiting Factor:</b> {format_limiting_factor(report.limiting_factor)}<br/><b>Forecaster:</b> {format_forecast_status(report.forecast_source)}" if report.final_rul_seconds is not None else "<i>RUL unavailable — insufficient continuous history for a valid prognostic estimate. Withheld to prevent extrapolation.</i>"
        st.markdown(
            f"""
            <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px;">
                <div style="font-size: 11px; color: #8b949e; text-transform: uppercase;">Life Prediction (RUL)</div>
                <div style="font-size: 16px; font-weight: 700; color: #bc8cff; margin: 4px 0;">{rul_text}</div>
                <div style="font-size: 11px; color: #c9d1d9;">
                    {rul_desc}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Multi-source Explainability Preview
    st.markdown("#### 3. Evidence & Explainability")
    clean_anomaly_channels = ', '.join(format_channel(c) for c in report.dominant_anomaly_channels) if report.dominant_anomaly_channels else 'None'
    st.markdown(
        f"""
        <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px; font-size: 12px; color: #c9d1d9; line-height: 1.6;">
            <b>Fused Synthesis:</b> {report.fused_evidence_headline}<br/>
            <b>Physics Evidence:</b> <code>{format_evidence_status(report.physics_evidence_status)}</code> ({report.physics_consistency_reason or 'Nominal dynamics'})<br/>
            <b>Temporal Evidence:</b> <code>{format_evidence_status(report.temporal_evidence_status)}</code><br/>
            <b>Persistent Anomalies:</b> {report.anomaly_events_count} event(s) across {clean_anomaly_channels}.
        </div>
        """,
        unsafe_allow_html=True,
    )
