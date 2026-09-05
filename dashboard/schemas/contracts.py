"""
Phase 13 Ingestion Contract for SIH26054 Dashboard.

This defines the clean architectural boundary between the Phase 13 Orchestrator
and the downstream DashboardAdapter.

The dashboard NEVER reaches past this boundary into internal orchestration logic,
and NEVER executes ML models or computes PHM metrics.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from telemetry.schema import TelemetryRecord, DigitalTwinState
    from telemetry.quality import DataQualityReport
    from anomaly_detection.schema import AnomalyRecord
    from fault_diagnosis.schema import FaultDiagnosisResult
    from health_index.schema import HealthIndexResult
    from forecasting.schema import ForecastResult
    from prognostics.schema import RULResult
    from explainability.schema import ExplainabilityResult


@dataclass
class Phase13OutputContract:
    """
    Authoritative container for single-step or streaming outputs emitted by Phase 13 Orchestrator.

    All sub-phase outputs are typed where available. If any phase is omitted, disabled,
    or degraded, that field is None, and the DashboardAdapter handles it without fabricating values.
    """
    # System metadata
    timestamp: float
    engine_id: str
    mission_id: Optional[str] = None
    execution_status: str = "COMPLETED"  # "COMPLETED", "PARTIAL", "ERROR", "UNAVAILABLE"
    is_synthetic_demo: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Sub-phase authoritative outputs
    telemetry: Optional[Union[TelemetryRecord, Dict[str, Any]]] = None
    data_quality: Optional[Union[DataQualityReport, Dict[str, Any]]] = None
    digital_twin: Optional[Union[DigitalTwinState, Dict[str, Any]]] = None
    anomaly: Optional[Union[AnomalyRecord, Dict[str, Any]]] = None
    fault_diagnosis: Optional[Union[FaultDiagnosisResult, Dict[str, Any]]] = None
    health_index: Optional[Union[HealthIndexResult, Dict[str, Any]]] = None
    forecast: Optional[Union[ForecastResult, Dict[str, Any]]] = None
    prognostics: Optional[Union[RULResult, Dict[str, Any]]] = None
    explainability: Optional[Union[ExplainabilityResult, Dict[str, Any]]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Phase13OutputContract:
        """Safely instantiate from dictionary payload with standard or alternative key names."""
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict for Phase13OutputContract, got {type(data).__name__}")

        # Support both flat keys and nested 'mission' structure from DashboardStatePayload.to_dict()
        mission = data.get("mission") if isinstance(data.get("mission"), dict) else {}
        timestamp = data.get("timestamp", mission.get("timestamp", 0.0))
        engine_id = data.get("engine_id", mission.get("engine_id", "UNKNOWN"))
        mission_id = data.get("mission_id", mission.get("mission_id"))

        # Handle potential key aliases emitted by various orchestrators
        telemetry = data.get("telemetry") or data.get("observed_telemetry") or data.get("telemetry_record") or data.get("_raw_telemetry")
        data_quality = data.get("data_quality") or data.get("quality_report") or data.get("telemetry_quality") or data.get("quality_status")
        digital_twin = data.get("digital_twin") or data.get("twin_state") or data.get("digital_twin_state") or data.get("_residual_frame")
        anomaly = data.get("anomaly") or data.get("anomaly_record") or data.get("anomaly_detection")
        fault_diagnosis = data.get("fault_diagnosis") or data.get("diagnosis") or data.get("diagnosis_result") or data.get("_diagnosis_result")
        health_index = data.get("health_index") or data.get("health") or data.get("health_result") or data.get("_health_result")
        forecast = data.get("forecast") or data.get("forecasting") or data.get("forecast_result") or data.get("_forecast_result")
        prognostics = data.get("prognostics") or data.get("rul") or data.get("rul_result") or data.get("_rul_result")
        explainability = data.get("explainability") or data.get("xai") or data.get("explainability_result") or data.get("_explainability_result")

        return cls(
            timestamp=float(timestamp) if timestamp is not None else 0.0,
            engine_id=str(engine_id),
            mission_id=str(mission_id) if mission_id is not None else None,
            execution_status=str(data.get("execution_status", "COMPLETED")),
            is_synthetic_demo=bool(data.get("is_synthetic_demo", False)),
            metadata=dict(data.get("metadata", {})),
            telemetry=telemetry,
            data_quality=data_quality,
            digital_twin=digital_twin,
            anomaly=anomaly,
            fault_diagnosis=fault_diagnosis,
            health_index=health_index,
            forecast=forecast,
            prognostics=prognostics,
            explainability=explainability,
        )

    @classmethod
    def from_object(cls, obj: Any) -> Phase13OutputContract:
        """Safely instantiate from an arbitrary Phase 13 orchestrator result object."""
        if isinstance(obj, Phase13OutputContract):
            return obj
        if isinstance(obj, dict):
            return cls.from_dict(obj)

        # Attribute-based extraction (including backing fields and properties from DashboardStatePayload)
        timestamp = getattr(obj, "timestamp", 0.0)
        engine_id = getattr(obj, "engine_id", "UNKNOWN")
        mission_id = getattr(obj, "mission_id", None)

        telemetry = (
            getattr(obj, "telemetry", None)
            or getattr(obj, "telemetry_record", None)
            or getattr(obj, "_raw_telemetry", None)
            or getattr(obj, "observed_telemetry", None)
        )
        data_quality = getattr(obj, "data_quality", getattr(obj, "quality_report", getattr(obj, "quality_status", None)))

        digital_twin = (
            getattr(obj, "digital_twin", None)
            or getattr(obj, "twin_state", None)
            or getattr(obj, "_residual_frame", None)
        )
        if digital_twin is None and (hasattr(obj, "expected_telemetry") or hasattr(obj, "residuals")):
            digital_twin = {
                "expected": getattr(obj, "expected_telemetry", {}),
                "nominal_estimates": getattr(obj, "expected_telemetry", {}),
                "residuals": getattr(obj, "residuals", {}),
                "normalized_residuals": getattr(obj, "normalized_residuals", {}),
            }

        anomaly = getattr(obj, "anomaly", getattr(obj, "anomaly_record", None))
        if anomaly is None and hasattr(obj, "anomaly_status"):
            anomaly = {
                "anomaly_status": getattr(obj, "anomaly_status", "NORMAL"),
                "status": getattr(obj, "anomaly_status", "NORMAL"),
                "anomaly_score": getattr(obj, "anomaly_score", 0.0),
                "score": getattr(obj, "anomaly_score", 0.0),
                "contributing_channels": getattr(obj, "anomaly_contributing_channels", []),
                "persistence_count": getattr(obj, "persistence_count", 0),
                "evidence": getattr(obj, "anomaly_evidence", {}),
            }

        fault_diagnosis = (
            getattr(obj, "fault_diagnosis", None)
            or getattr(obj, "diagnosis", None)
            or getattr(obj, "authoritative_diagnosis", None)
            or getattr(obj, "_diagnosis_result", None)
        )
        if fault_diagnosis is None and hasattr(obj, "predicted_fault_class"):
            fault_diagnosis = {
                "predicted_fault_type": getattr(obj, "predicted_fault_class", "none"),
                "predicted_fault_class": getattr(obj, "predicted_fault_class", "none"),
                "diagnostic_confidence": getattr(obj, "diagnostic_confidence", 1.0),
                "confidence": getattr(obj, "diagnostic_confidence", 1.0),
                "class_probabilities": getattr(obj, "diagnosis_probabilities", {}),
                "data_quality": getattr(obj, "diagnosis_data_quality", "VALID"),
            }

        health_index = (
            getattr(obj, "health_index", None)
            or getattr(obj, "health", None)
            or getattr(obj, "authoritative_health", None)
            or getattr(obj, "_health_result", None)
        )
        if health_index is None and hasattr(obj, "smoothed_health_index"):
            health_index = {
                "smoothed_health_index": getattr(obj, "smoothed_health_index", 1.0),
                "raw_health_index": getattr(obj, "raw_health_index", 1.0),
                "health_state": getattr(obj, "health_state", "NORMAL"),
                "degradation_rate": getattr(obj, "degradation_rate", 0.0),
                "degradation_trend": getattr(obj, "degradation_trend", "STABLE"),
                "dominant_channels": getattr(obj, "dominant_channels", []),
                "channel_contributions": getattr(obj, "channel_contributions", {}),
            }

        forecast = (
            getattr(obj, "forecast", None)
            or getattr(obj, "forecast_result", None)
            or getattr(obj, "authoritative_forecast", None)
            or getattr(obj, "_forecast_result", None)
        )
        if forecast is None and hasattr(obj, "forecast_status"):
            forecast = {
                "status": getattr(obj, "forecast_status", "BUFFERING"),
                "source": getattr(obj, "forecast_source", ""),
                "horizon": getattr(obj, "forecast_horizon", 16),
                "predicted_telemetry": getattr(obj, "predicted_telemetry", None),
                "forecast_timestamps": getattr(obj, "forecast_timestamps", None),
                "forecast_quality": getattr(obj, "forecast_quality", "INSUFFICIENT_CONTEXT"),
            }

        prognostics = (
            getattr(obj, "prognostics", None)
            or getattr(obj, "rul", None)
            or getattr(obj, "authoritative_rul", None)
            or getattr(obj, "_rul_result", None)
        )
        if prognostics is None and hasattr(obj, "rul_state"):
            prognostics = {
                "status": getattr(obj, "rul_state", "INSUFFICIENT_HISTORY"),
                "state": getattr(obj, "rul_state", "INSUFFICIENT_HISTORY"),
                "rul_seconds_median": getattr(obj, "point_rul_seconds", None),
                "point_rul_seconds": getattr(obj, "point_rul_seconds", None),
                "rul_seconds_p05": getattr(obj, "rul_uncertainty_p05", None),
                "rul_seconds_p95": getattr(obj, "rul_uncertainty_p95", None),
                "limiting_factor": getattr(obj, "limiting_factor", "NONE"),
            }

        explainability = (
            getattr(obj, "explainability", None)
            or getattr(obj, "explainability_result", None)
            or getattr(obj, "authoritative_explainability", None)
            or getattr(obj, "_explainability_result", None)
        )
        if explainability is None and hasattr(obj, "summary_explanation"):
            explainability = {
                "summary_explanation": getattr(obj, "summary_explanation", ""),
                "shap_attribution": getattr(obj, "shap_attribution", None),
                "physics_evidence": getattr(obj, "physics_evidence", None),
                "temporal_evidence": getattr(obj, "temporal_evidence", None),
                "fused_evidence": getattr(obj, "fused_evidence", None),
                "recommended_operator_action": getattr(obj, "recommended_operator_action", ""),
            }

        return cls(
            timestamp=float(timestamp) if timestamp is not None else 0.0,
            engine_id=str(engine_id),
            mission_id=str(mission_id) if mission_id is not None else None,
            execution_status=str(getattr(obj, "execution_status", "COMPLETED")),
            is_synthetic_demo=bool(getattr(obj, "is_synthetic_demo", False)),
            metadata=dict(getattr(obj, "metadata", {})),
            telemetry=telemetry,
            data_quality=data_quality,
            digital_twin=digital_twin,
            anomaly=anomaly,
            fault_diagnosis=fault_diagnosis,
            health_index=health_index,
            forecast=forecast,
            prognostics=prognostics,
            explainability=explainability,
        )
