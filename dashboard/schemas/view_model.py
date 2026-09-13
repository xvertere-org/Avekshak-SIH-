"""
Typed View Models and Status Enums for SIH26054 Dashboard UI.

Every value presented to the UI strictly originates from these structures,
guaranteeing complete separation of presentation from upstream calculations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional


class StatusLevel(str, Enum):
    """Semantic status classifications for UI display, badges, and alerts."""
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"
    UNKNOWN = "UNKNOWN"


class AvailabilityStatus(str, Enum):
    """Explicit availability tracking to prevent silent fabrication."""
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    DEGRADED = "DEGRADED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


# Canonical 7 aero-piston telemetry channels with display metadata
CANONICAL_CHANNEL_METADATA: Dict[str, Dict[str, Any]] = {
    "rpm": {
        "display_name": "Engine Speed",
        "unit": "RPM",
        "nominal_min": 2000.0,
        "nominal_max": 5800.0,
        "critical_max": 5850.0,
    },
    "cht": {
        "display_name": "Cylinder Head Temp",
        "unit": "°C",
        "nominal_min": 60.0,
        "nominal_max": 135.0,
        "critical_max": 150.0,
    },
    "egt": {
        "display_name": "Exhaust Gas Temp",
        "unit": "°C",
        "nominal_min": 500.0,
        "nominal_max": 850.0,
        "critical_max": 950.0,
    },
    "oil_temp": {
        "display_name": "Oil Temperature",
        "unit": "°C",
        "nominal_min": 70.0,
        "nominal_max": 110.0,
        "critical_max": 130.0,
    },
    "oil_pressure": {
        "display_name": "Oil Pressure",
        "unit": "bar",
        "nominal_min": 2.0,
        "nominal_max": 5.5,
        "critical_min": 1.2,
    },
    "fuel_flow": {
        "display_name": "Fuel Flow",
        "unit": "L/h",
        "nominal_min": 5.0,
        "nominal_max": 30.0,
        "critical_max": 45.0,
    },
    "vibration": {
        "display_name": "Vibration Amplitude",
        "unit": "g",
        "nominal_min": 0.1,
        "nominal_max": 1.5,
        "critical_max": 3.5,
    },
}

CANONICAL_CHANNELS: List[str] = list(CANONICAL_CHANNEL_METADATA.keys())


@dataclass
class MetricCardModel:
    """Standardized KPI card model for overview and section summaries."""
    label: str
    value: str
    status: StatusLevel = StatusLevel.UNKNOWN
    subtext: Optional[str] = None
    unit: Optional[str] = None
    availability: AvailabilityStatus = AvailabilityStatus.AVAILABLE


@dataclass
class ChannelTelemetryModel:
    """Unified representation of a single telemetry channel for live charts."""
    channel: str
    display_name: str
    unit: str
    observed_value: Optional[float] = None
    expected_value: Optional[float] = None
    residual: Optional[float] = None
    forecast_values: List[float] = field(default_factory=list)
    forecast_timestamps: List[float] = field(default_factory=list)
    forecast_lower_bounds: List[float] = field(default_factory=list)
    forecast_upper_bounds: List[float] = field(default_factory=list)
    status: StatusLevel = StatusLevel.HEALTHY
    is_isolated: bool = False
    is_missing: bool = False
    availability: AvailabilityStatus = AvailabilityStatus.AVAILABLE


@dataclass
class OperatorAdvisoryModel:
    """Operator decision-support advisory model."""
    action_code: str = "NORMAL_MONITORING"
    headline: str = "Nominal Operations"
    recommended_action: str = "Continue nominal monitoring."
    affected_subsystem: Optional[str] = None
    urgency: str = "LOW"  # LOW, MEDIUM, HIGH, CRITICAL
    disclaimer: str = (
        "Decision-support advisory recommendation based on simulated telemetry. "
        "Not certified airworthiness limits or OEM/FAA flight-clearance directives."
    )


@dataclass
class TelemetryViewModel:
    """Live Telemetry view model including 7 canonical channels and operating context."""
    timestamp: float
    channels: Dict[str, ChannelTelemetryModel] = field(default_factory=dict)
    observed_telemetry: Dict[str, Optional[float]] = field(default_factory=dict)
    quality_status: str = "NOMINAL"
    throttle: Optional[float] = None
    load: Optional[float] = None
    altitude: Optional[float] = None
    ambient_temp: Optional[float] = None
    mission_phase: str = "CRUISE"
    availability: AvailabilityStatus = AvailabilityStatus.AVAILABLE


@dataclass
class DiagnosticsViewModel:
    """Diagnostics view model covering digital twin, anomaly detection, fault diagnosis, and XAI."""
    timestamp: float

    # Digital Twin (Phase 6)
    expected_telemetry: Dict[str, Optional[float]] = field(default_factory=dict)
    residuals: Dict[str, Optional[float]] = field(default_factory=dict)
    normalized_residuals: Dict[str, Optional[float]] = field(default_factory=dict)

    # Anomaly Detection (Phase 7)
    anomaly_status: str = "Unavailable"
    anomaly_score: Optional[float] = None
    contributing_channels: List[str] = field(default_factory=list)
    anomaly_contributing_channels: List[str] = field(default_factory=list)
    persistence_count: int = 0
    anomaly_evidence: Dict[str, Any] = field(default_factory=dict)
    detector_scores: Dict[str, Optional[float]] = field(default_factory=dict)

    # Fault Diagnosis (Phase 8)
    predicted_fault: str = "Unavailable"
    predicted_fault_class: str = "none"
    diagnostic_confidence: Optional[float] = None
    class_probabilities: Dict[str, float] = field(default_factory=dict)
    diagnosis_data_quality: str = "VALID"
    sensor_fault_indicated: bool = False
    isolated_channels: List[str] = field(default_factory=list)
    suspect_sensor: Optional[str] = None
    suspect_sensors: List[str] = field(default_factory=list)
    sensor_isolation_status: str = "NONE"

    # Explainability (Phase 12)
    summary_explanation: Optional[str] = None
    recommended_operator_action: Optional[str] = None
    shap_top_features: List[Dict[str, Any]] = field(default_factory=list)
    shap_attribution: Optional[Dict[str, Any]] = None
    physics_evidence: Optional[Dict[str, Any]] = None
    physics_evidence_status: Optional[str] = None
    physics_consistency_reason: Optional[str] = None
    temporal_evidence: Optional[Dict[str, Any]] = None
    fused_evidence: Optional[Dict[str, Any]] = None
    shap_disclaimer: Optional[str] = None

    availability: AvailabilityStatus = AvailabilityStatus.AVAILABLE


@dataclass
class PrognosticsViewModel:
    """Prognostics view model covering health index, degradation rate, RUL, and forecast."""
    timestamp: float

    # Health Index (Phase 9)
    health_index: Optional[float] = None
    smoothed_health_index: Optional[float] = None
    raw_health_index: Optional[float] = None
    health_state: str = "Unavailable"
    degradation_rate: Optional[float] = None
    degradation_trend: str = "Unavailable"
    dominant_channels: List[str] = field(default_factory=list)
    channel_contributions: Dict[str, float] = field(default_factory=dict)

    # RUL & Prognostics (Phase 11)
    rul_state: str = "Unavailable"
    rul_status: str = "Unavailable"
    point_rul_seconds: Optional[float] = None
    rul_hours: Optional[float] = None
    rul_uncertainty_p05: Optional[float] = None
    rul_p05_hours: Optional[float] = None
    rul_uncertainty_p95: Optional[float] = None
    rul_p95_hours: Optional[float] = None
    limiting_factor: Optional[str] = None
    forecast_assisted_mode: bool = False
    forecast_mode_status: str = "OFF"
    eol_provenance: Dict[str, Any] = field(default_factory=dict)
    prognostic_confidence: Optional[float] = None

    # Forecast (Phase 10)
    forecast_status: str = "BUFFERING"
    forecast_source: str = "TIMESFM_OR_BASELINE"
    forecast_horizon: int = 16
    forecast_quality: Optional[str] = None
    model_name: Optional[str] = None
    model_status: Optional[str] = None
    is_pretrained: bool = False
    predicted_telemetry: Optional[Dict[str, List[float]]] = None
    forecast_timestamps: List[float] = field(default_factory=list)
    projected_health_trajectory: Optional[List[float]] = None

    channel_degradation_evidence: Dict[str, float] = field(default_factory=dict)
    dominant_degraded_channels: List[str] = field(default_factory=list)

    availability: AvailabilityStatus = AvailabilityStatus.AVAILABLE


@dataclass
class DataQualityViewModel:
    """Data quality and system status view model."""
    timestamp: float
    quality_score: Optional[float] = None
    quality_status: str = "NOMINAL"
    is_regular_sampling: bool = True
    sampling_interval_mean: Optional[float] = None
    missing_sensors: List[str] = field(default_factory=list)
    invalid_sensors: List[str] = field(default_factory=list)
    warning_sensors: List[str] = field(default_factory=list)
    isolated_sensors: List[str] = field(default_factory=list)
    channel_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    issues_detected: List[str] = field(default_factory=list)

    provenance: Dict[str, Any] = field(default_factory=dict)
    provenance_source: str = "Unavailable"
    provenance_source_type: str = "Unavailable"
    simulation_version: str = "Unavailable"
    execution_latency_ms: float = 0.0

    availability: AvailabilityStatus = AvailabilityStatus.AVAILABLE


@dataclass
class OverviewViewModel:
    """Overview view model containing top-level operational health indicators."""
    overall_status: StatusLevel
    engine_id: str
    mission_id: Optional[str]
    timestamp: float
    mission_phase: str
    simulation_mode: str
    is_synthetic_demo: bool
    scenario_metadata: Dict[str, Any] = field(default_factory=dict)

    health_card: MetricCardModel = field(default_factory=lambda: MetricCardModel("Health", "Unavailable"))
    rul_card: MetricCardModel = field(default_factory=lambda: MetricCardModel("RUL", "Unavailable"))
    anomaly_card: MetricCardModel = field(default_factory=lambda: MetricCardModel("Anomaly", "Unavailable"))
    fault_card: MetricCardModel = field(default_factory=lambda: MetricCardModel("Fault", "Unavailable"))
    data_quality_card: MetricCardModel = field(default_factory=lambda: MetricCardModel("Quality", "Unavailable"))
    advisory: Optional[OperatorAdvisoryModel] = None


@dataclass
class DashboardViewModel:
    """Root composite view model for the entire dashboard UI state."""
    timestamp: float
    engine_id: str
    mission_id: Optional[str]
    mission_phase: str
    simulation_mode: str
    scenario_metadata: Dict[str, Any]
    is_synthetic_demo: bool
    execution_status: str

    overview: OverviewViewModel
    telemetry: TelemetryViewModel
    diagnostics: DiagnosticsViewModel
    prognostics: PrognosticsViewModel
    data_quality: DataQualityViewModel
    advisory: Optional[OperatorAdvisoryModel] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
    execution_latency_ms: float = 0.0
