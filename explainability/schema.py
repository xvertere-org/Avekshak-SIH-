"""
Data schemas, enums, and typed structures for Phase 12 Explainability & Evidence Fusion.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


class EvidenceStatus(str, Enum):
    """Evaluation status of physical consistency for a diagnosed condition."""
    SUPPORTED = "SUPPORTED"
    PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
    CONFLICTING = "CONFLICTING"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class EvidenceQuality(str, Enum):
    """Categorical composite confidence and concordance across all evidence streams."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass(frozen=True)
class SHAPFeatureContribution:
    """
    Individual local feature attribution from TreeSHAP.
    NOTE: SHAP feature attribution does not establish causality.
    """
    feature_name: str
    feature_value: Optional[float]
    shap_value: float
    direction: str                     # "TOWARD_PREDICTED_CLASS" or "AWAY_FROM_PREDICTED_CLASS"
    relative_weight: float             # |shap_value| / sum(|shap_values|)


@dataclass(frozen=True)
class SHAPEvidence:
    """
    Local model attribution evidence for Phase 8 fault classification.
    """
    predicted_fault: Optional[str]
    diagnostic_confidence: Optional[float]
    class_probabilities: Dict[str, float]
    top_features: List[SHAPFeatureContribution]
    base_value: Optional[float]
    status: str                        # "AVAILABLE", "MODEL_UNAVAILABLE", "INSUFFICIENT_DATA"
    disclaimer: str = "SHAP feature attribution does not establish causality."


@dataclass(frozen=True)
class PhysicsEvidence:
    """
    Deterministic rule-based consistency check between observed residuals and physical relationships.
    NOTE: Physics consistency is supporting engineering evidence, not proof of fault causation.
    """
    status: EvidenceStatus
    diagnosed_fault: str
    evidence_channels: List[str]
    observed_residual_directions: Dict[str, str]   # e.g. {"cht": "ELEVATED", "oil_temp": "ELEVATED"}
    expected_residual_directions: Dict[str, str]   # e.g. {"cht": "ELEVATED", "oil_temp": "ELEVATED"}
    consistency_reason: str
    supporting_channels: List[str]
    conflicting_channels: List[str]
    missing_channels: List[str]
    provenance: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HealthEvidence:
    """
    Passthrough interpretation of Phase 9 Health Index and channel degradation contributions.
    """
    current_health_index: Optional[float]
    health_state: Optional[str]
    dominant_degraded_channels: List[str]
    channel_contributions: Dict[str, float]
    valid_channels: List[str]
    missing_channels: List[str]
    excluded_channels: List[str]
    effective_channel_weights: Dict[str, float]
    data_quality: Optional[str]
    status: str                        # "AVAILABLE", "SENSOR_ISOLATED", "INSUFFICIENT_DATA"


@dataclass(frozen=True)
class TemporalEvidence:
    """
    Causal historical trajectory evidence from Phase 9 and Phase 10.
    """
    degradation_rate: Optional[float]
    degradation_trend: Optional[str]
    trend_direction: str               # "WORSENING", "STABLE", "IMPROVING", "INDETERMINATE"
    forecast_status: Optional[str]
    forecast_horizon_s: Optional[float]
    status: str                        # "AVAILABLE", "INSUFFICIENT_HISTORY", "UNAVAILABLE"


@dataclass(frozen=True)
class RULEvidence:
    """
    Interpretation of Phase 11 Remaining Useful Life and EOL limiting factors.
    NOTE: Project-defined EOL criteria are not certified OEM/FAA limits.
    """
    rul_seconds_median: Optional[float]
    rul_seconds_p05: Optional[float]
    rul_seconds_p95: Optional[float]
    rul_status: Optional[str]
    limiting_factor: Optional[str]
    active_flight_phase: Optional[str]
    handoff_source: Optional[str]
    eol_provenance: Dict[str, Any]
    status: str                        # "AVAILABLE", "NON_DEGRADING", "UNAVAILABLE"
    disclaimer: str = "Project-defined EOL criteria are not certified OEM/FAA limits."


@dataclass(frozen=True)
class EvidenceProvenance:
    """Metadata tracking upstream artifact provenance and state isolation."""
    engine_id: str
    mission_id: Optional[str]
    timestamp: float
    phase8_present: bool
    phase9_present: bool
    phase10_present: bool
    phase11_present: bool


@dataclass(frozen=True)
class ExplainabilityResult:
    """
    Unified authoritative explainability and evidence fusion result per observation point.
    """
    engine_id: str
    mission_id: Optional[str]
    timestamp: float
    overall_quality: EvidenceQuality
    summary_explanation: str
    shap_evidence: Optional[SHAPEvidence]
    physics_evidence: PhysicsEvidence
    health_evidence: Optional[HealthEvidence]
    temporal_evidence: Optional[TemporalEvidence]
    rul_evidence: Optional[RULEvidence]
    provenance: EvidenceProvenance
