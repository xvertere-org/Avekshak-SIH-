"""
NIRVANAA — SIH26054 Aero Engine Digital Twin Dashboard.

Authoritative Operator Presentation & Interaction Layer.
Consumes outputs from Phase 13 via DashboardAdapter.
"""

import os
import sys
from pathlib import Path

# Ensure repository root is in sys.path regardless of Streamlit invocation directory
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from typing import Dict, Any, Optional, List
import streamlit as st

from orchestrator.pipeline import SystemPipelineOrchestrator
from orchestrator.schema import (
    OrchestratorConfig,
    SimulationScenario,
    ScenarioFaultType,
    DashboardStatePayload,
)
from dashboard.schemas.contracts import Phase13OutputContract
from dashboard.schemas.view_model import DashboardViewModel
from dashboard.services.adapter import DashboardAdapter
from dashboard.services.demo_provider import DemoScenarioProvider
from dashboard.utils.styles import AEROSPACE_CSS
from dashboard.components.header import render_header
from dashboard.pages.overview_page import render_overview_page
from dashboard.pages.telemetry_page import render_telemetry_page
from dashboard.pages.diagnostics_page import render_diagnostics_page
from dashboard.pages.prognostics_page import render_prognostics_page
from dashboard.pages.system_status_page import render_system_status_page
from dashboard.pages.replay_page import render_replay_page
from dashboard.pages.report_page import render_report_page
from dashboard.pages.what_if_page import render_what_if_page


@st.cache_resource(show_spinner="Bootstrapping Phase 13 Pipeline Orchestrator (XGBoost + Isolation Forest)...")
def get_cached_orchestrator() -> SystemPipelineOrchestrator:
    """Initialize singleton production orchestrator with deterministic bootstrap."""
    cfg = OrchestratorConfig(auto_bootstrap_on_init=True, deterministic_seed=42)
    return SystemPipelineOrchestrator(config=cfg)


@st.cache_data(show_spinner="Executing Phase 13 End-to-End Simulation Pipeline...")
def run_live_simulation(
    scenario_fault_str: str,
    duration_s: float,
    fault_start_s: float,
    fault_severity: float,
    throttle_pct: float = 75.0,
    altitude_m: float = 2000.0,
    ambient_temp_c: float = 15.0,
    seed: int = 42,
) -> List[DashboardStatePayload]:
    """Execute end-to-end mission simulation through full Phase 6-12 causal pipeline."""
    orch = get_cached_orchestrator()
    fault_map = {
        "1. Nominal Healthy Cruise": ScenarioFaultType.HEALTHY,
        "2. Cooling Degradation (Thermal Conductance Loss)": ScenarioFaultType.COOLING_DEGRADATION,
        "3. Lubrication Degradation (Oil Pressure Loss)": ScenarioFaultType.LUBRICATION_DEGRADATION,
        "4. Fuel Injection Abnormality": ScenarioFaultType.FUEL_ABNORMALITY,
        "5. Mechanical Degradation (High Vibration)": ScenarioFaultType.MECHANICAL_DEGRADATION,
        "6. Sensor Drift & Isolation": ScenarioFaultType.SENSOR_FAULT,
    }
    ft = fault_map.get(scenario_fault_str, ScenarioFaultType.HEALTHY)
    sc = SimulationScenario(
        engine_id="UAV_AERO_ROT912_01",
        mission_id="MIS_ISR_PATROL_01",
        duration_s=float(duration_s),
        fault_type=ft,
        fault_start_s=float(fault_start_s),
        fault_severity=float(fault_severity),
        throttle_pct=float(throttle_pct),
        altitude_m=float(altitude_m),
        ambient_temp_c=float(ambient_temp_c),
        seed=int(seed),
    )
    return orch.run_simulation(sc)


def extract_history_from_payloads(payloads_subset: List[DashboardStatePayload]) -> Dict[str, Any]:
    """Extract multi-channel time series from live orchestrator payloads for charting."""
    channels = ["rpm", "cht", "egt", "oil_temp", "oil_pressure", "fuel_flow", "vibration"]
    history: Dict[str, Any] = {
        "timestamps": [p.timestamp for p in payloads_subset],
        "health_index": {
            "timestamps": [p.timestamp for p in payloads_subset],
            "hi": [p.smoothed_health_index if p.smoothed_health_index is not None else float("nan") for p in payloads_subset],
        },
    }
    for ch in channels:
        history[ch] = {
            "timestamps": [p.timestamp for p in payloads_subset],
            "observed": [p.observed_telemetry.get(ch, float("nan")) for p in payloads_subset],
            "expected": [p.expected_telemetry.get(ch, float("nan")) for p in payloads_subset],
        }
    return history


class DashboardInterface:
    """
    Backward-compatible interface connecting pipeline outputs to dashboard models.
    Preserves Phase 1 main.py dry-run contract while supporting modern Phase 13 schemas.
    """

    def __init__(self, title: str = "Aero Piston Engine Digital Twin - SIH26054"):
        self.title = title
        self.adapter = DashboardAdapter()

    def render_state(
        self,
        telemetry: Any = None,
        twin_state: Any = None,
        health: Any = None,
        rul: Any = None,
        explanation: Any = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Package pipeline state for UI verification and backward compatibility.
        """
        contract = Phase13OutputContract(
            timestamp=getattr(telemetry, "timestamp", 0.0),
            engine_id=getattr(telemetry, "engine_id", "UNKNOWN"),
            telemetry=telemetry,
            digital_twin=twin_state,
            health_index=health,
            prognostics=rul,
            explainability=explanation,
        )
        vm = self.adapter.adapt(contract)
        return {
            "title": self.title,
            "engine_id": vm.engine_id,
            "timestamp": vm.timestamp,
            "mission_phase": vm.telemetry.mission_phase,
            "health_index": vm.prognostics.health_index,
            "anomaly_detected": vm.diagnostics.anomaly_status == "ANOMALY",
            "fault_category": vm.diagnostics.predicted_fault,
            "estimated_rul_hours": vm.prognostics.rul_hours,
            "explanation": vm.diagnostics.summary_explanation,
            "provenance": {
                "source": vm.data_quality.provenance_source,
                "source_type": vm.data_quality.provenance_source_type,
                "simulation_version": vm.data_quality.simulation_version,
            },
        }


def main():
    """Main Streamlit application entrypoint."""
    st.set_page_config(
        page_title="SIH26054 Aero Engine Digital Twin",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Inject aerospace dark theme CSS
    st.markdown(AEROSPACE_CSS, unsafe_allow_html=True)

    # Sidebar: Mode Selection & Navigation
    with st.sidebar:
        st.markdown(
            '<div style="text-align: center; margin-bottom: 20px; padding: 10px 0; border-bottom: 1px solid #21262d;">'
            '<h3 style="margin: 0; color: #58a6ff; font-weight: 700; letter-spacing: 1px;">SIH26054</h3>'
            '<div style="font-size: 11px; color: #8b949e; margin-top: 4px;">MALE UAV Aero-Piston Engine Twin</div>'
            '</div>',
            unsafe_allow_html=True,
        )

        # Navigation / System Views
        st.markdown(
            '<div style="font-size: 11px; font-weight: 700; color: #8b949e; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px;">VIEWS</div>',
            unsafe_allow_html=True,
        )
        active_tab = st.radio(
            "System Views",
            [
                "Overview",
                "Live Telemetry",
                "Diagnostics",
                "Prognostics",
                "System Status",
                "Mission Replay",
                "Mission Report",
                "What-If Comparison",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown('<hr style="border: none; border-top: 1px solid #21262d; margin: 16px 0;" />', unsafe_allow_html=True)

        # Simulation Controls
        st.markdown(
            '<div style="font-size: 11px; font-weight: 700; color: #8b949e; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px;">OPERATIONAL FEED</div>',
            unsafe_allow_html=True,
        )

        feed_mode = st.radio(
            "Telemetry Source",
            [
                "Phase 13 Live Pipeline Orchestrator",
                "Pre-Packaged Demo Scenarios",
            ],
            index=0,
            help="Select live end-to-end backend orchestrator or pre-packaged static demo.",
        )

        history_data = None
        payloads: List[DashboardStatePayload] = []
        adapter = DashboardAdapter()

        if feed_mode == "Phase 13 Live Pipeline Orchestrator":
            st.markdown("---")
            st.markdown("#### Live Mission Scenarios")
            live_scenarios = [
                "1. Nominal Healthy Cruise",
                "2. Cooling Degradation (Thermal Conductance Loss)",
                "3. Lubrication Degradation (Oil Pressure Loss)",
                "4. Fuel Injection Abnormality",
                "5. Mechanical Degradation (High Vibration)",
                "6. Sensor Drift & Isolation",
            ]
            selected_scenario = st.selectbox("Scenario", live_scenarios, index=1)

            with st.expander("⚙️ Mission & Fault Settings", expanded=False):
                duration = st.slider("Duration (s)", min_value=15, max_value=90, value=35, step=5)
                fault_start = st.slider("Fault Injection Time (s)", min_value=5, max_value=max(6, duration - 5), value=15, step=1)
                severity = st.slider("Fault Severity", min_value=0.1, max_value=1.0, value=0.7, step=0.05)
                throttle = st.slider("Throttle (%)", min_value=50, max_value=100, value=75, step=5)
                altitude = st.slider("Altitude (m)", min_value=500, max_value=5000, value=2000, step=250, help="Operating flight altitude in meters.")
                ambient_temp = st.slider("Ambient Temperature (°C)", min_value=-20, max_value=50, value=15, step=1, help="Operating ambient air temperature.")

            # Execute end-to-end backend pipeline
            payloads = run_live_simulation(
                scenario_fault_str=selected_scenario,
                duration_s=float(duration),
                fault_start_s=float(fault_start),
                fault_severity=float(severity),
                throttle_pct=float(throttle),
                altitude_m=float(altitude),
                ambient_temp_c=float(ambient_temp),
            )

            max_step = len(payloads) - 1
            default_step = min(max_step, 25)

            step_slider = st.slider(
                "Mission Elapsed Time (s)",
                min_value=0,
                max_value=max_step,
                value=default_step,
                step=1,
                help="Scrub flight time to observe causal real-time pipeline inference.",
            )

            current_payload = payloads[step_slider]
            history_data = extract_history_from_payloads(payloads[:step_slider + 1])
            vm = adapter.adapt(current_payload)

        else:
            st.markdown("---")
            st.markdown("#### Test Scenarios")
            scenarios = DemoScenarioProvider.get_available_scenarios()
            selected_scenario = st.selectbox("Scenario", scenarios, index=0)

            step_slider = st.slider(
                "Mission Elapsed Time (s)",
                min_value=5,
                max_value=120,
                value=35,
                step=1,
                help="Advance flight time step to observe progressive degradation.",
            )

            contract, history_data = DemoScenarioProvider.generate_scenario_payload(
                scenario_name=selected_scenario,
                step=step_slider,
            )
            vm = adapter.adapt(contract)

            # Map pre-packaged test scenario to authoritative pipeline simulation for Mission Replay & Reporting
            demo_to_live_map = {
                "1. Nominal Healthy Cruise": ("1. Nominal Healthy Cruise", 15.0, 0.0),
                "2. Thermal Degradation (Cooling Conductance Loss)": ("2. Cooling Degradation (Thermal Conductance Loss)", 15.0, 0.7),
                "3. Lubrication Pressure Loss (Hydraulic Failure)": ("3. Lubrication Degradation (Oil Pressure Loss)", 15.0, 0.7),
                "4. CHT Sensor Dropout & Isolation (Instrumentation Fault)": ("6. Sensor Drift & Isolation", 15.0, 0.7),
            }
            mapped = demo_to_live_map.get(selected_scenario, ("1. Nominal Healthy Cruise", 15.0, 0.0))
            sim_duration = max(float(step_slider), 35.0)
            try:
                payloads = run_live_simulation(
                    scenario_fault_str=mapped[0],
                    duration_s=sim_duration,
                    fault_start_s=mapped[1],
                    fault_severity=mapped[2],
                    throttle_pct=75.0,
                )
            except Exception:
                payloads = []

        if st.button("Reset Simulation", use_container_width=True):
            st.session_state.sim_time = 35
            st.session_state.scenario_idx = 0
            st.rerun()

        st.markdown('<hr style="border: none; border-top: 1px solid #21262d; margin: 16px 0;" />', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size: 11px; color: #8b949e; line-height: 1.5; background: #161b22; '
            'border: 1px solid #21262d; border-radius: 4px; padding: 8px 10px;">'
            '<b style="color: #c9d1d9;">Engineering Reference Architecture:</b><br/>'
            'Rotax 914 UL/F (Reduced-Order Grey-Box Prototype).<br/>'
            '<span style="color: #6e7681;">Strict non-fabrication presentation layer.</span>'

            '</div>',
            unsafe_allow_html=True,
        )

    # Render Header Ribbon
    render_header(vm.overview)

    # Render Active View
    if active_tab == "Overview" or "Overview" in active_tab:
        render_overview_page(vm, history_data=history_data)
    elif active_tab == "Live Telemetry" or "Live Telemetry" in active_tab:
        render_telemetry_page(vm, history_data=history_data)
    elif active_tab == "Diagnostics" or "Diagnostics" in active_tab:
        render_diagnostics_page(vm)
    elif active_tab == "Prognostics" or "Prognostics" in active_tab:
        hist_hi_ts = history_data.get("health_index", {}).get("timestamps") if history_data else None
        hist_hi = history_data.get("health_index", {}).get("hi") if history_data else None
        render_prognostics_page(vm, history_timestamps=hist_hi_ts, history_hi=hist_hi)
    elif active_tab == "System Status" or "Status" in active_tab or "Data Quality" in active_tab:
        render_system_status_page(vm)
    elif active_tab == "Mission Replay" or "Replay" in active_tab:
        render_replay_page(payloads, scenario_name=selected_scenario)
    elif active_tab == "Mission Report" or "Report" in active_tab:
        scenario_meta = {"scenario_name": selected_scenario, "feed_mode": feed_mode}
        render_report_page(payloads, scenario_metadata=scenario_meta)
    elif active_tab == "What-If Comparison" or "What-If" in active_tab:
        orch = get_cached_orchestrator()
        render_what_if_page(orchestrator=orch)


if __name__ == "__main__":
    main()
