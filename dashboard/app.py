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
            """
            <div style="text-align: center; margin-bottom: 15px;">
                <h3 style="margin: 0; color: #58a6ff;">SIH26054</h3>
                <div style="font-size: 11px; color: #8b949e;">MALE UAV Aero-Piston Engine Twin</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Operational Feed")
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

            # Execute end-to-end backend pipeline
            payloads = run_live_simulation(
                scenario_fault_str=selected_scenario,
                duration_s=float(duration),
                fault_start_s=float(fault_start),
                fault_severity=float(severity),
                throttle_pct=float(throttle),
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

        st.markdown("---")
        st.markdown("### Navigation")
        active_tab = st.radio(
            "System Views",
            [
                "1. Overview",
                "2. Live Telemetry",
                "3. Diagnostics",
                "4. Prognostics",
                "5. Data Quality / Status",
                "6. Mission Replay",
                "7. Mission Report",
                "8. What-If Comparison",
            ],
            index=0,
        )

        st.markdown("---")
        st.markdown(
            """
            <div style="font-size: 10px; color: #8b949e; line-height: 1.4;">
                <b>Engineering Reference Anchor:</b><br/>
                Rotax 912 ULS grey-box baseline.<br/>
                Strict non-fabrication presentation layer.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Render Header Ribbon
    render_header(vm.overview)

    # Render Active View
    if active_tab == "1. Overview":
        render_overview_page(vm, history_data=history_data)
    elif active_tab == "2. Live Telemetry":
        render_telemetry_page(vm, history_data=history_data)
    elif active_tab == "3. Diagnostics":
        render_diagnostics_page(vm)
    elif active_tab == "4. Prognostics":
        hist_hi_ts = history_data.get("health_index", {}).get("timestamps") if history_data else None
        hist_hi = history_data.get("health_index", {}).get("hi") if history_data else None
        render_prognostics_page(vm, history_timestamps=hist_hi_ts, history_hi=hist_hi)
    elif active_tab == "5. Data Quality / Status":
        render_system_status_page(vm)
    elif active_tab == "6. Mission Replay":
        render_replay_page(payloads, scenario_name=selected_scenario)
    elif active_tab == "7. Mission Report":
        scenario_meta = {"scenario_name": selected_scenario, "feed_mode": feed_mode}
        render_report_page(payloads, scenario_metadata=scenario_meta)
    elif active_tab == "8. What-If Comparison":
        orch = get_cached_orchestrator()
        render_what_if_page(orchestrator=orch)


if __name__ == "__main__":
    main()

