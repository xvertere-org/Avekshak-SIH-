"""
Schema and state contracts for Phase 13 Unified System Pipeline Orchestrator.
Defines DashboardStatePayload, OrchestratorConfig, SimulationScenario, and advisory structures.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd

from telemetry.schema import TelemetryRecord, DigitalTwinState
from digital_twin.residuals import ResidualFrame
from anomaly_detection.schema import AnomalyRecord, AnomalyFrame, AnomalyStatus
from fault_diagnosis.schema import FaultDiagnosisResult, DiagnosisDataQuality, CANONICAL_FAULT_LABELS
from health_index.schema import HealthIndexResult, HealthState, DegradationTrend
from forecasting.schema import ForecastResult, ForecastQuality, ModelStatus
from prognostics.schema import RULResult, RULStatus
from explainability.schema import ExplainabilityResult


class ScenarioFaultType(str, Enum):
    """Supported simulation fault scenarios for demo and testing."""
    HEALTHY = "none"
    COOLING_DEGRADATION = "cooling_degradation"
    LUBRICATION_DEGRADATION = "lubrication_degradation"
    FUEL_ABNORMALITY = "fuel_injection_abnormality"
    MECHANICAL_DEGRADATION = "mechanical_degradation"
    SENSOR_FAULT = "sensor_fault"


class AdvisoryActionCode(str, Enum):
    """Standardized advisory action codes for operator decision support."""
    NORMAL_MONITORING = "NORMAL_MONITORING"
    ADVISORY_CAUTION = "ADVISORY_CAUTION"
    MAINTENANCE_INSPECTION = "MAINTENANCE_INSPECTION"
    CRITICAL_ABORT_ACTION = "CRITICAL_ABORT_ACTION"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass
class OperatorAdvisory:
    """
    Operator decision-support advisory.
    Advisory only: never certified OEM/FAA airworthiness limits.
    """
    action_code: AdvisoryActionCode
    headline: str
    recommended_action: str
    affected_subsystem: Optional[str] = None
    urgency: str = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL
    disclaimer: str = (
        "Decision-support advisory recommendation based on simulated telemetry. "
        "Not certified airworthiness limits or OEM/FAA flight-clearance directives."
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_code": self.action_code.value,
            "headline": self.headline,
            "recommended_action": self.recommended_action,
            "affected_subsystem": self.affected_subsystem,
            "urgency": self.urgency,
            "disclaimer": self.disclaimer,
        }


@dataclass
class DashboardStatePayload:
    """
    Unified system payload consumed by the Streamlit dashboard.
    Passes through authoritative Phase 6–12 dataclass outputs without duplicating business logic.
    """
    # MISSION
    engine_id: str
    mission_id: Optional[str]
    timestamp: float
    mission_phase: str
    simulation_mode: str = "SYNTHETIC_SIMULATION"
    scenario_metadata: Dict[str, Any] = field(default_factory=dict)

    # TELEMETRY
    observed_telemetry: Dict[str, float] = field(default_factory=dict)
    quality_status: str = "NOMINAL"

    # DIGITAL TWIN (Phase 6)
    expected_telemetry: Dict[str, float] = field(default_factory=dict)
    residuals: Dict[str, float] = field(default_factory=dict)
    normalized_residuals: Dict[str, float] = field(default_factory=dict)

    # ANOMALY (Phase 7)
    anomaly_status: str = "NORMAL"
    anomaly_score: float = 0.0
    anomaly_contributing_channels: List[str] = field(default_factory=list)
    persistence_count: int = 0
    anomaly_evidence: Dict[str, Any] = field(default_factory=dict)

    # DIAGNOSIS (Phase 8)
    predicted_fault_class: str = "none"
    diagnosis_probabilities: Dict[str, float] = field(default_factory=dict)
    diagnostic_confidence: float = 1.0
    diagnosis_data_quality: str = "VALID"
    suspect_sensor: Optional[str] = None
    suspect_sensors: List[str] = field(default_factory=list)
    sensor_isolation_status: str = "NONE"

    # HEALTH (Phase 9)
    raw_health_index: float = 1.0
    smoothed_health_index: float = 1.0
    health_state: str = "NORMAL"
    degradation_rate: float = 0.0
    degradation_trend: str = "STABLE"
    dominant_channels: List[str] = field(default_factory=list)
    channel_contributions: Dict[str, float] = field(default_factory=dict)

    # FORECAST (Phase 10)
    forecast_status: str = "BUFFERING"
    forecast_source: str = "TIMESFM_OR_BASELINE"
    forecast_horizon: int = 16
    predicted_telemetry: Optional[Dict[str, List[float]]] = None
    forecast_timestamps: Optional[List[float]] = None
    is_pretrained: bool = False
    forecast_quality: str = "INSUFFICIENT_CONTEXT"
    projected_health_trajectory: Optional[List[float]] = None

    # RUL (Phase 11)
    rul_state: str = "INSUFFICIENT_HISTORY"
    point_rul_seconds: Optional[float] = None
    rul_uncertainty_p05: Optional[float] = None
    rul_uncertainty_p95: Optional[float] = None
    limiting_factor: str = "NONE"
    forecast_assisted_mode: bool = False
    forecast_mode_status: str = "OFF"
    eol_provenance: Dict[str, Any] = field(default_factory=dict)

    # EXPLAINABILITY (Phase 12)
    summary_explanation: str = ""
    shap_attribution: Optional[Dict[str, Any]] = None
    physics_evidence: Optional[Dict[str, Any]] = None
    temporal_evidence: Optional[Dict[str, Any]] = None
    fused_evidence: Optional[Dict[str, Any]] = None
    recommended_operator_action: str = "Continue nominal monitoring."

    # ADVISORY DECISION SUPPORT (Section 15)
    advisory: Optional[OperatorAdvisory] = None

    # PROVENANCE & TIMING
    provenance: Dict[str, Any] = field(default_factory=dict)
    execution_latency_ms: float = 0.0

    # AUTHORITATIVE BACKING INSTANCES (Preserving exact Phase 6–12 objects)
    _raw_telemetry: Optional[Any] = field(default=None, repr=False)
    _residual_frame: Optional[ResidualFrame] = field(default=None, repr=False)
    _diagnosis_result: Optional[FaultDiagnosisResult] = field(default=None, repr=False)
    _health_result: Optional[HealthIndexResult] = field(default=None, repr=False)
    _forecast_result: Optional[ForecastResult] = field(default=None, repr=False)
    _rul_result: Optional[RULResult] = field(default=None, repr=False)
    _explainability_result: Optional[ExplainabilityResult] = field(default=None, repr=False)

    @property
    def authoritative_diagnosis(self) -> Optional[FaultDiagnosisResult]:
        return self._diagnosis_result

    @property
    def authoritative_health(self) -> Optional[HealthIndexResult]:
        return self._health_result

    @property
    def authoritative_forecast(self) -> Optional[ForecastResult]:
        return self._forecast_result

    @property
    def authoritative_rul(self) -> Optional[RULResult]:
        return self._rul_result

    @property
    def authoritative_explainability(self) -> Optional[ExplainabilityResult]:
        return self._explainability_result

    def to_dict(self) -> Dict[str, Any]:
        """Serialize payload to a clean dictionary for Streamlit or API serialization."""
        return {
            "mission": {
                "engine_id": self.engine_id,
                "mission_id": self.mission_id,
                "timestamp": self.timestamp,
                "mission_phase": self.mission_phase,
                "simulation_mode": self.simulation_mode,
                "scenario_metadata": self.scenario_metadata,
            },
            "telemetry": {
                "observed": self.observed_telemetry,
                "quality_status": self.quality_status,
            },
            "digital_twin": {
                "expected": self.expected_telemetry,
                "residuals": self.residuals,
                "normalized_residuals": self.normalized_residuals,
            },
            "anomaly": {
                "status": self.anomaly_status,
                "score": self.anomaly_score,
                "contributing_channels": self.anomaly_contributing_channels,
                "persistence_count": self.persistence_count,
                "evidence": self.anomaly_evidence,
            },
            "diagnosis": {
                "predicted_fault_class": self.predicted_fault_class,
                "class_probabilities": self.diagnosis_probabilities,
                "confidence": self.diagnostic_confidence,
                "data_quality": self.diagnosis_data_quality,
                "suspect_sensor": self.suspect_sensor,
                "suspect_channel": self.suspect_sensor,
                "suspect_sensors": self.suspect_sensors,
                "sensor_isolation_status": self.sensor_isolation_status,
            },
            "health": {
                "raw_health_index": self.raw_health_index,
                "smoothed_health_index": self.smoothed_health_index,
                "health_state": self.health_state,
                "degradation_rate": self.degradation_rate,
                "degradation_trend": self.degradation_trend,
                "dominant_channels": self.dominant_channels,
                "channel_contributions": self.channel_contributions,
            },
            "forecast": {
                "status": self.forecast_status,
                "source": self.forecast_source,
                "horizon": self.forecast_horizon,
                "predicted_telemetry": self.predicted_telemetry,
                "timestamps": self.forecast_timestamps,
                "is_pretrained": self.is_pretrained,
                "quality": self.forecast_quality,
            },
            "rul": {
                "state": self.rul_state,
                "point_rul_seconds": self.point_rul_seconds,
                "p05": self.rul_uncertainty_p05,
                "p95": self.rul_uncertainty_p95,
                "limiting_factor": self.limiting_factor,
                "forecast_assisted": self.forecast_assisted_mode,
                "forecast_assisted_mode": self.forecast_assisted_mode,
                "forecast_mode_status": self.forecast_mode_status,
                "eol_provenance": self.eol_provenance,
            },
            "explainability": {
                "summary_explanation": self.summary_explanation,
                "shap_attribution": self.shap_attribution,
                "physics_evidence": self.physics_evidence,
                "temporal_evidence": self.temporal_evidence,
                "fused_evidence": self.fused_evidence,
                "recommended_operator_action": self.recommended_operator_action,
            },
            "advisory": self.advisory.to_dict() if self.advisory else None,
            "provenance": self.provenance,
            "execution_latency_ms": self.execution_latency_ms,
        }


@dataclass
class OrchestratorConfig:
    """Configuration options for SystemPipelineOrchestrator."""
    default_engine_id: str = "ENG_001"
    default_mission_id: str = "MISSION_001"
    sampling_dt: float = 1.0

    # Phase 7 configuration
    min_valid_features: int = 4
    anomaly_threshold_warning: float = 1.5
    anomaly_threshold_anomaly: float = 3.0
    anomaly_ewma_alpha: float = 0.2
    anomaly_persistence_min: int = 3

    # Phase 8 configuration
    enable_phase7_gating: bool = True
    xgboost_n_estimators: int = 30
    xgboost_max_depth: int = 4

    # Phase 9 configuration
    health_ewma_alpha: float = 0.15

    # Phase 10 configuration
    forecast_context_length: int = 32
    forecast_horizon: int = 16

    # Phase 11 configuration
    rul_warmup_duration_s: float = 30.0
    theil_sen_window_s: float = 60.0

    # Phase 12 configuration
    shap_top_k: int = 5
    physics_tau_threshold: float = 1.5

    # Bootstrap configuration
    deterministic_seed: int = 42
    auto_bootstrap_on_init: bool = True


@dataclass
class SimulationScenario:
    """Defines a simulation flight run with optional fault injection."""
    name: str = "nominal_cruise"
    duration_s: float = 60.0
    throttle_pct: float = 75.0
    altitude_m: float = 2000.0
    ambient_temp_c: float = 15.0
    fault_type: ScenarioFaultType = ScenarioFaultType.HEALTHY
    fault_start_s: float = 20.0
    fault_severity: float = 0.5
    seed: int = 42
    dt: float = 1.0
    engine_id: str = "ENG_001"
    mission_id: str = "MISSION_001"
