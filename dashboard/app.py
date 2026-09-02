"""
Dashboard entrypoint and interface stub for SIH26054 Digital Twin.

Phase 1 provides a clean interface/stub structure for future Streamlit/Plotly visualization.
"""

from typing import Dict, Any, Optional
from telemetry.schema import (
    TelemetryRecord,
    DigitalTwinState,
    HealthAssessment,
    RULPrediction,
    ExplanationReport,
)


class DashboardInterface:
    """
    Interface connecting pipeline outputs to visual dashboard components.
    """

    def __init__(self, title: str = "Aero Piston Engine Digital Twin - SIH26054"):
        self.title = title

    def render_state(
        self,
        telemetry: TelemetryRecord,
        twin_state: DigitalTwinState,
        health: HealthAssessment,
        rul: RULPrediction,
        explanation: ExplanationReport,
    ) -> Dict[str, Any]:
        """
        Package pipeline state for UI rendering / verification.
        """
        payload = {
            "title": self.title,
            "engine_id": telemetry.engine_id,
            "timestamp": telemetry.timestamp,
            "mission_phase": telemetry.mission_phase,
            "health_index": health.health_index,
            "anomaly_detected": health.anomaly_detected,
            "fault_category": health.fault_category,
            "estimated_rul_hours": rul.estimated_rul_hours,
            "explanation": explanation.explanation_text,
            "provenance": {
                "source": telemetry.source,
                "source_type": telemetry.source_type,
                "simulation_version": telemetry.simulation_version,
            },
        }
        return payload


def main():
    """
    Streamlit application entrypoint stub.
    """
    try:
        import streamlit as st
        st.set_page_config(page_title="SIH26054 Aero Engine Digital Twin", layout="wide")
        st.title("Aero Piston Engine Digital Twin — SIH26054")
        st.info("Phase 1 Architecture Stub: Dashboard UI components will be connected in future phases.")
    except ImportError:
        print("Streamlit not installed or running in headless CLI mode.")


if __name__ == "__main__":
    main()
