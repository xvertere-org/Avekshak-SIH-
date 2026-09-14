"""
Prognostics and Health Management (PHM) interface.

Handles health index estimation, anomaly detection, and fault classification.
"""

from typing import Optional
from telemetry.schema import DigitalTwinState, HealthAssessment, FaultCategory


class HealthDetector:
    """
    PHM Anomaly and Fault Detection interface.
    Phase 1: Stub providing schema-compliant health assessment without ML models.
    """

    def __init__(self, sensitivity: float = 0.5):
        self.sensitivity = sensitivity

    def assess(self, twin_state: DigitalTwinState) -> HealthAssessment:
        """
        Assess engine health from Digital Twin state and residuals.
        """
        # Phase 1: Stub health assessment
        fault_type = twin_state.observed_telemetry.fault_type
        is_faulty = fault_type != FaultCategory.NONE.value

        health_index = 0.65 if is_faulty else 0.98
        anomaly_score = 0.75 if is_faulty else 0.05
        warnings = [f"Warning: {fault_type}"] if is_faulty else []

        return HealthAssessment(
            timestamp=twin_state.timestamp,
            engine_id=twin_state.engine_id,
            health_index=health_index,
            anomaly_detected=is_faulty,
            anomaly_score=anomaly_score,
            fault_category=fault_type,
            fault_confidence=0.88 if is_faulty else 0.0,
            active_warnings=warnings,
        )
