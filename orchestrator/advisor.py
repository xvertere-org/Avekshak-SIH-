"""
Operator decision-support advisory module for Phase 13.
Conforms strictly to Section 15 requirements: provides advisory guidance based on existing
authoritative Phase 7–12 outputs.

IMPORTANT:
These are decision-support advisory recommendations only based on simulated engine telemetry.
They are NOT certified airworthiness directives, FAA/OEM failure limits, or flight safety clearances.
"""

from typing import Dict, Any, Optional, List
from orchestrator.schema import OperatorAdvisory, AdvisoryActionCode


class OperatorActionAdvisor:
    """
    Synthesizes actionable, explainable advisory recommendations from authoritative system states.
    Does not invent new algorithms or mutate upstream decisions.
    """

    @staticmethod
    def generate_advisory(
        anomaly_status: str,
        diagnosis_fault: str,
        diagnosis_confidence: float,
        health_state: str,
        rul_status: str,
        dominant_channels: Optional[List[str]] = None,
        recommended_action_from_phase12: Optional[str] = None,
    ) -> OperatorAdvisory:
        """
        Generate standardized decision-support advisory from authoritative phase outputs.
        """
        dominant = dominant_channels or []
        subsystem_str = ", ".join(dominant) if dominant else None

        # 1. CRITICAL EOL REACHED
        if (
            rul_status == "CRITICAL_EOL_REACHED"
            or health_state == "CRITICAL"
            or anomaly_status == "CRITICAL"
        ):
            action_code = AdvisoryActionCode.CRITICAL_ABORT_ACTION
            headline = "CRITICAL LIMIT REACHED / IMMEDIATE INTERVENTION ADVISED"
            rec = (
                recommended_action_from_phase12
                or "Terminate operation according to simulated mission policy. Immediate maintenance inspection required."
            )
            urgency = "CRITICAL"

        # 2. ACTIVE DEGRADATION / DEGRADED PROGNOSTICS
        elif (
            rul_status in ("ACTIVE_DEGRADATION", "DEGRADED_PROGNOSTIC")
            or health_state == "WARNING"
            or (diagnosis_fault != "none" and diagnosis_confidence >= 0.6)
        ):
            action_code = AdvisoryActionCode.MAINTENANCE_INSPECTION
            headline = f"ACTIVE DEGRADATION DETECTED ({diagnosis_fault.upper()})"
            rec = (
                recommended_action_from_phase12
                or f"Consider mission reassessment or scheduled inspection for {diagnosis_fault}."
            )
            urgency = "HIGH"

        # 3. WARNING / CAUTION / UNVERIFIED ANOMALY
        elif (
            anomaly_status in ("WARNING", "ANOMALY")
            or health_state == "CAUTION"
            or (diagnosis_fault != "none" and diagnosis_confidence < 0.6)
        ):
            action_code = AdvisoryActionCode.ADVISORY_CAUTION
            headline = f"SUBSYSTEM CAUTION / ANOMALY DETECTED"
            rec = (
                recommended_action_from_phase12
                or f"Monitor affected subsystem ({subsystem_str or diagnosis_fault}). Maintain situational awareness."
            )
            urgency = "MEDIUM"

        # 4. INSUFFICIENT DATA
        elif anomaly_status == "INSUFFICIENT_DATA" or health_state == "INSUFFICIENT_DATA":
            action_code = AdvisoryActionCode.INSUFFICIENT_DATA
            headline = "INSUFFICIENT TELEMETRY DATA"
            rec = "Verify sensor signal integrity and telemetry stream connectivity."
            urgency = "LOW"

        # 5. NORMAL
        else:
            action_code = AdvisoryActionCode.NORMAL_MONITORING
            headline = "NOMINAL CRUISE / STABLE OPERATION"
            rec = "Continue nominal flight monitoring."
            urgency = "LOW"

        return OperatorAdvisory(
            action_code=action_code,
            headline=headline,
            recommended_action=rec,
            affected_subsystem=subsystem_str or (diagnosis_fault if diagnosis_fault != "none" else None),
            urgency=urgency,
        )
