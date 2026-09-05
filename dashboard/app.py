"""
NIRVANAA — SIH26054 Aero Engine Digital Twin Dashboard.

Authoritative Operator Presentation & Interaction Layer.
Consumes outputs from Phase 13 via DashboardAdapter.
"""

from typing import Dict, Any, Optional
import streamlit as st

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
            ["Demo / Synthetic Bench Simulation", "Phase 13 Ingestion Boundary"],
            index=0,
            help="Select live or synthetic demonstration stream.",
        )

        history_data = None
        if feed_mode == "Demo / Synthetic Bench Simulation":
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

            # Generate demo contract
            contract, history_data = DemoScenarioProvider.generate_scenario_payload(
                scenario_name=selected_scenario,
                step=step_slider,
            )
        else:
            st.info("Awaiting live Phase 13 output contract stream...")
            contract = Phase13OutputContract(
                timestamp=0.0,
                engine_id="UAV_AERO_01",
                execution_status="AWAITING_INPUT",
                is_synthetic_demo=False,
            )

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

    # Adapt contract to ViewModel
    adapter = DashboardAdapter()
    vm = adapter.adapt(contract)

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


if __name__ == "__main__":
    main()
