"""
Mission What-If Comparison Page for SIH26054 Dashboard.
Evaluates comparative mission condition changes and simulated fault-stress scenarios.
Strictly reuses authoritative Simulation -> Phase 13 -> Phase 11 RUL pathway.
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

from orchestrator.schema import SimulationScenario, ScenarioFaultType, DashboardStatePayload
from orchestrator.pipeline import SystemPipelineOrchestrator
from phase14.what_if import WhatIfAnalyzer
from phase14.schema import WhatIfComparisonResult


def render_what_if_page(
    orchestrator: SystemPipelineOrchestrator,
):
    """Render 8. WHAT-IF ANALYSIS section."""
    st.markdown("### Mission comparison")
    st.markdown(
        """
        <div style="font-size: 13px; color: #8b949e; margin-bottom: 16px;">
            Compare alternative mission conditions or simulated fault scenarios with a baseline mission.
            Results are inferred from the digital-twin and life-prediction pipeline; fault labels are not exposed to the model.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Conceptually Separated Controls
    st.markdown(
        """
        <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 10px 14px; margin-bottom: 16px; font-size: 12px; color: #c9d1d9;">
            <b style="color: #58a6ff;">BASELINE MISSION:</b> <code>Nominal Healthy Cruise</code> &nbsp;|&nbsp; 
            <b style="color: #bc8cff;">WHAT-IF MISSION:</b> <code>Planned Alternative / Simulated Fault-Stress</code>
            <div style="font-size: 11px; color: #8b949e; margin-top: 4px;">
                Demonstration mode compares a baseline mission with alternative conditions or simulated stress. Results are inferred; fault labels remain hidden from the model.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Compare mission conditions")
    col_base, col_whatif = st.columns(2)

    with col_base:
        st.markdown(
            """
            <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 10px; margin-bottom: 10px; font-weight: 700; color: #58a6ff;">
                BASELINE MISSION
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<b>Conditions:</b>", unsafe_allow_html=True)
        b_altitude = st.slider("Baseline Altitude (m)", 500, 5000, 2000, 250, key="wi_b_alt")
        b_oat = st.slider("Baseline Ambient Temperature (°C)", -20, 50, 15, 1, key="wi_b_oat")
        b_throttle = st.slider("Baseline Throttle (%)", 50, 100, 75, 5, key="wi_b_throttle")
        b_duration = st.slider("Baseline Duration (s)", 15, 60, 30, 5, key="wi_b_dur")

        st.markdown("<b style='color: #8b949e;'>Mission profile:</b>", unsafe_allow_html=True)
        b_fault_str = st.selectbox(
            "Baseline Profile",
            ["Nominal Healthy Cruise", "Cooling Degradation (Thermal Loss)", "Lubrication Degradation (Oil Loss)"],
            index=0,
            key="wi_b_fault",
        )

    with col_whatif:
        st.markdown(
            """
            <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 10px; margin-bottom: 10px; font-weight: 700; color: #bc8cff;">
                WHAT-IF MISSION
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<b>Conditions:</b>", unsafe_allow_html=True)
        w_altitude = st.slider("What-If Altitude (m)", 500, 5000, 3000, 250, key="wi_w_alt")
        w_oat = st.slider("What-If Ambient Temperature (°C)", -20, 50, 30, 1, key="wi_w_oat")
        w_throttle = st.slider("What-If Throttle (%)", 50, 100, 85, 5, key="wi_w_throttle")
        w_duration = st.slider("What-If Duration (s)", 15, 60, 30, 5, key="wi_w_dur")

        st.markdown("<b style='color: #8b949e;'>Simulated condition:</b>", unsafe_allow_html=True)
        w_fault_str = st.selectbox(
            "Simulated Stress Condition",
            [
                "Nominal Healthy Cruise",
                "Cooling Degradation (Thermal Loss)",
                "Lubrication Degradation (Oil Loss)",
                "Fuel Abnormality",
                "Mechanical Degradation (Vibration)",
            ],
            index=1,
            key="wi_w_fault",
            help="Select a simulated condition for comparison. Fault labels are not available to the inference system.",
        )

    # Fault mapping helper
    fault_map = {
        "Nominal Healthy Cruise": ScenarioFaultType.HEALTHY,
        "Cooling Degradation": ScenarioFaultType.COOLING_DEGRADATION,
        "Cooling Degradation (Thermal Loss)": ScenarioFaultType.COOLING_DEGRADATION,
        "Lubrication Degradation": ScenarioFaultType.LUBRICATION_DEGRADATION,
        "Lubrication Degradation (Oil Loss)": ScenarioFaultType.LUBRICATION_DEGRADATION,
        "Fuel Abnormality": ScenarioFaultType.FUEL_ABNORMALITY,
        "Mechanical Degradation (Vibration)": ScenarioFaultType.MECHANICAL_DEGRADATION,
    }

    st.markdown("")
    btn_run = st.button("Run mission comparison", type="primary", use_container_width=True)

    if btn_run or "last_whatif_result" in st.session_state:
        if btn_run:
            with st.spinner("Running the simulated mission comparison..."):
                b_sc = SimulationScenario(
                    name="baseline_mission",
                    duration_s=float(b_duration),
                    throttle_pct=float(b_throttle),
                    altitude_m=float(b_altitude),
                    ambient_temp_c=float(b_oat),
                    fault_type=fault_map.get(b_fault_str, ScenarioFaultType.HEALTHY),
                    fault_start_s=12.0,
                    fault_severity=0.6,
                    seed=42,
                    engine_id="UAV_AERO_01",
                    mission_id="MIS_BASELINE",
                )
                w_sc = SimulationScenario(
                    name="whatif_mission",
                    duration_s=float(w_duration),
                    throttle_pct=float(w_throttle),
                    altitude_m=float(w_altitude),
                    ambient_temp_c=float(w_oat),
                    fault_type=fault_map.get(w_fault_str, ScenarioFaultType.HEALTHY),
                    fault_start_s=12.0,
                    fault_severity=0.7,
                    seed=42,
                    engine_id="UAV_AERO_01",
                    mission_id="MIS_WHATIF",
                )
                res = WhatIfAnalyzer.run_comparison(orchestrator, b_sc, w_sc)
                st.session_state["last_whatif_result"] = res
        else:
            res = st.session_state["last_whatif_result"]

        st.markdown("---")

        # Top Comparison Summary (baseline versus what-if simulated projection)
        st.markdown("#### Comparison summary")
        st.markdown(
            f"""
            <div style="background-color: #161b22; border-left: 5px solid #58a6ff; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 20px;">
                <div style="font-size: 15px; font-weight: 700; color: #f0f6fc; margin-bottom: 6px;">
                    {res.comparison_summary_headline}
                </div>
                <div style="font-size: 13px; color: #c9d1d9; margin-bottom: 8px;">
                    {res.simulated_projection_narrative}
                </div>
                <div style="font-size: 11px; color: #8b949e; border-top: 1px solid #21262d; padding-top: 8px;">
                    ℹ️ <i>{res.disclaimer}</i>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Comparative Side-by-Side Cards (Baseline vs What-If with Delta)
        c_hi, c_rul, c_cht, c_adv = st.columns(4)

        with c_hi:
            delta_str = f"{res.delta_health_index:+.3f}" if not np.isnan(res.delta_health_index) else "N/A"
            st.metric(
                label="Health Index (HI)",
                value=f"{res.whatif_health_index:.3f}",
                delta=f"Baseline: {res.baseline_health_index:.3f} (Δ {delta_str})",
                delta_color="normal" if res.delta_health_index >= 0 else "inverse",
            )

        with c_rul:
            b_rul_str = f"{res.baseline_rul_seconds:.0f} s" if res.baseline_rul_seconds is not None else "Unavailable"
            w_rul_str = f"{res.whatif_rul_seconds:.0f} s" if res.whatif_rul_seconds is not None else "Unavailable"
            d_rul_str = f"{res.delta_rul_seconds:+.0f} s" if res.delta_rul_seconds is not None else "N/A"
            st.metric(
                label="Projected RUL",
                value=w_rul_str,
                delta=f"Baseline: {b_rul_str} (Δ {d_rul_str})" if res.delta_rul_seconds is not None else None,
                delta_color="normal" if (res.delta_rul_seconds or 0) >= 0 else "inverse",
                help="RUL unavailable — insufficient continuous history for a valid prognostic estimate. The system withholds RUL rather than extrapolating from insufficient evidence." if (res.whatif_rul_seconds is None or res.baseline_rul_seconds is None) else None,
            )
            if res.whatif_rul_seconds is None or res.baseline_rul_seconds is None:
                st.caption("ℹ️ *RUL unavailable — insufficient continuous history for a valid prognostic estimate. The system withholds RUL rather than extrapolating from insufficient evidence.*")

        with c_cht:
            delta_cht_str = f"{res.delta_peak_cht:+.1f} °C" if not np.isnan(res.delta_peak_cht) else "N/A"
            st.metric(
                label="Peak CHT",
                value=f"{res.whatif_peak_cht:.1f} °C",
                delta=f"Baseline: {res.baseline_peak_cht:.1f} °C (Δ {delta_cht_str})",
                delta_color="inverse" if res.delta_peak_cht > 0 else "normal",
            )

        with c_adv:
            st.markdown(
                f"""
                <div style="background-color: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 10px; text-align: center;">
                    <div style="font-size: 11px; color: #8b949e; text-transform: uppercase;">Advisory Shift</div>
                    <div style="font-size: 14px; font-weight: 700; color: #f0f6fc; margin: 4px 0;">
                        <code>{res.baseline_advisory_assessment}</code> ➔ <code>{res.whatif_advisory_assessment}</code>
                    </div>
                    <div style="font-size: 10px; color: #8b949e;">Limiting: {res.whatif_limiting_factor}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Comparative Trajectory Overlay Plots
        st.markdown("#### Trajectory comparison")
        b_ts = [p.timestamp for p in res.baseline_payloads]
        w_ts = [p.timestamp for p in res.whatif_payloads]

        plot_c1, plot_c2 = st.columns(2)

        with plot_c1:
            fig_hi = go.Figure()
            fig_hi.add_trace(go.Scatter(
                x=b_ts,
                y=[p.smoothed_health_index for p in res.baseline_payloads],
                mode="lines",
                name="Baseline HI",
                line=dict(color="#58a6ff", width=2, dash="dash"),
            ))
            fig_hi.add_trace(go.Scatter(
                x=w_ts,
                y=[p.smoothed_health_index for p in res.whatif_payloads],
                mode="lines",
                name="What-If HI",
                line=dict(color="#f85149" if res.delta_health_index < 0 else "#3fb950", width=2.5),
            ))
            fig_hi.update_layout(
                title="Health Index Trajectory Comparison",
                paper_bgcolor="#161b22",
                plot_bgcolor="#0d1117",
                font=dict(color="#c9d1d9", size=10),
                margin=dict(l=30, r=20, t=40, b=30),
                height=260,
                xaxis=dict(title="Time (s)", gridcolor="#21262d"),
                yaxis=dict(title="Health Index", gridcolor="#21262d", range=[0.0, 1.05]),
                legend=dict(orientation="h", y=1.1, x=1, xanchor="right"),
            )
            st.plotly_chart(fig_hi, use_container_width=True)

        with plot_c2:
            fig_cht = go.Figure()
            fig_cht.add_trace(go.Scatter(
                x=b_ts,
                y=[p.observed_telemetry.get("cht", np.nan) for p in res.baseline_payloads],
                mode="lines",
                name="Baseline CHT",
                line=dict(color="#58a6ff", width=2, dash="dash"),
            ))
            fig_cht.add_trace(go.Scatter(
                x=w_ts,
                y=[p.observed_telemetry.get("cht", np.nan) for p in res.whatif_payloads],
                mode="lines",
                name="What-If CHT",
                line=dict(color="#f0883e", width=2.5),
            ))
            fig_cht.update_layout(
                title="Cylinder Head Temperature (CHT) Trajectory",
                paper_bgcolor="#161b22",
                plot_bgcolor="#0d1117",
                font=dict(color="#c9d1d9", size=10),
                margin=dict(l=30, r=20, t=40, b=30),
                height=260,
                xaxis=dict(title="Time (s)", gridcolor="#21262d"),
                yaxis=dict(title="CHT (°C)", gridcolor="#21262d"),
                legend=dict(orientation="h", y=1.1, x=1, xanchor="right"),
            )
            st.plotly_chart(fig_cht, use_container_width=True)
