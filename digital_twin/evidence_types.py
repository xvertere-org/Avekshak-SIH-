"""
Phase 11 Evidence, Traceability & Engineering Explainability: Data Models & Contracts.

Provides strictly typed, immutable data structures, enums, reason codes,
and provenance tracking for auditable engineering explainability over the
Rotax 914 grey-box Digital Twin pipeline.

SCIENTIFIC & REGULATORY DISCLAIMER:
These contracts represent grey-box digital twin research abstractions.
All explanations, reason codes, hypothesis scores, and RUL estimates are
ENGINEERING HEURISTICS and MODEL SCENARIOS. They do NOT represent certified
OEM maintenance thresholds, airworthiness limits, formal causal proofs, or
certified flight safety diagnostics.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
import json
import math
from typing import Any, Dict, List, Optional, Tuple, Union


class DataClassification(str, Enum):
    """
    Epistemic classification of data values to prevent conflating measured with synthetic/modeled.
    """
    MEASURED = "MEASURED"
    ESTIMATED = "ESTIMATED"
    PREDICTED = "PREDICTED"
    DERIVED = "DERIVED"
    DEFAULT_ASSUMED = "DEFAULT_ASSUMED"
    UNAVAILABLE = "UNAVAILABLE"


class EpistemicProvenance(str, Enum):
    """
    Authoritative provenance source for thresholds, limits, and model parameters.
    """
    OEM_REFERENCE = "OEM_REFERENCE"
    MODEL_CALIBRATION = "MODEL_CALIBRATION"
    ENGINEERING_HEURISTIC = "ENGINEERING_HEURISTIC"
    SYNTHETIC_VALIDATION = "SYNTHETIC_VALIDATION"
    DATA_QUALITY_RULE = "DATA_QUALITY_RULE"


class ChannelRole(str, Enum):
    """
    Channel architectural role in health assessment to prevent double-counting.
    """
    PRIMARY_ENGINE_HEALTH = "PRIMARY_ENGINE_HEALTH"
    SECONDARY_MODEL_CONSISTENCY = "SECONDARY_MODEL_CONSISTENCY"
    CYLINDER_LOCALIZATION_DIAGNOSTIC = "CYLINDER_LOCALIZATION_DIAGNOSTIC"


class ExplanationReasonCode(str, Enum):
    """
    Deterministic machine-readable reason codes explaining Digital Twin outputs.
    """
    # Health and Subsystem Triggers
    HEALTH_CHANNEL_DEGRADED = "HEALTH_CHANNEL_DEGRADED"
    SUBSYSTEM_HEALTH_REDUCED = "SUBSYSTEM_HEALTH_REDUCED"
    ENGINE_HEALTH_REDUCED = "ENGINE_HEALTH_REDUCED"
    NOMINAL_OPERATION = "NOMINAL_OPERATION"

    # Temporal Anomaly Triggers
    ANOMALY_THRESHOLD_CROSSED = "ANOMALY_THRESHOLD_CROSSED"
    ANOMALY_PERSISTENCE_SATISFIED = "ANOMALY_PERSISTENCE_SATISFIED"
    ANOMALY_RECOVERY = "ANOMALY_RECOVERY"

    # Quality and Observability Triggers
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    SENSOR_QUALITY_DEGRADED = "SENSOR_QUALITY_DEGRADED"
    SENSOR_DROPOUT = "SENSOR_DROPOUT"
    SENSOR_STUCK = "SENSOR_STUCK"
    SENSOR_BIAS = "SENSOR_BIAS"
    SENSOR_DRIFT = "SENSOR_DRIFT"
    SENSOR_LOCALIZATION_FAVORED = "SENSOR_LOCALIZATION_FAVORED"

    # Diagnosis and Physics Signatures
    FAULT_SIGNATURE_MATCH = "FAULT_SIGNATURE_MATCH"
    CYLINDER_LOCALIZATION = "CYLINDER_LOCALIZATION"
    RESIDUAL_DIRECTIONAL_SUPPORT = "RESIDUAL_DIRECTIONAL_SUPPORT"
    THERMAL_LIMIT_APPROACH = "THERMAL_LIMIT_APPROACH"
    LUBRICATION_PRESSURE_DEVIATION = "LUBRICATION_PRESSURE_DEVIATION"
    COMBUSTION_EGT_DEVIATION = "COMBUSTION_EGT_DEVIATION"
    MECHANICAL_VIBRATION_DEVIATION = "MECHANICAL_VIBRATION_DEVIATION"

    # Prognostics and RUL Triggers
    RUL_TREND_SUPPORTED = "RUL_TREND_SUPPORTED"
    RUL_UNAVAILABLE = "RUL_UNAVAILABLE"
    RUL_NON_DEGRADING = "RUL_NON_DEGRADING"

    # Mission and Environmental Triggers
    MISSION_ENVELOPE_EVENT = "MISSION_ENVELOPE_EVENT"
    MODEL_ASSUMPTION_ACTIVE = "MODEL_ASSUMPTION_ACTIVE"


@dataclass(frozen=True)
class TraceabilityNode:
    """
    Individual step in an evidence lineage / computational trace.
    """
    stage: str
    source_entity: str
    output_entity: str
    value_summary: str
    relationship: str  # e.g., 'evaluated_by', 'normalized_from', 'aggregated_into'
    provenance: EpistemicProvenance = EpistemicProvenance.ENGINEERING_HEURISTIC


@dataclass(frozen=True)
class TraceabilityChain:
    """
    Complete machine-readable lineage connecting source telemetry to final output.
    """
    chain_id: str
    target_metric: str
    nodes: Tuple[TraceabilityNode, ...] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "target_metric": self.target_metric,
            "nodes": [
                {
                    "stage": n.stage,
                    "source_entity": n.source_entity,
                    "output_entity": n.output_entity,
                    "value_summary": n.value_summary,
                    "relationship": n.relationship,
                    "provenance": n.provenance.value,
                }
                for n in self.nodes
            ],
        }


@dataclass(frozen=True)
class ChannelEvidence:
    """
    Structured evidence record for an individual telemetry / physics channel.
    """
    channel: str
    unit: str
    observed_value: Optional[float]
    predicted_value: Optional[float]
    raw_residual: Optional[float]
    normalized_residual: Optional[float]
    classification: DataClassification
    role: ChannelRole
    quality_status: str
    observability_status: str
    provenance: EpistemicProvenance = EpistemicProvenance.MODEL_CALIBRATION
    rejection_reason: Optional[str] = None
    threshold_limit: Optional[float] = None
    threshold_direction: Optional[str] = None
    threshold_provenance: Optional[EpistemicProvenance] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel,
            "unit": self.unit,
            "observed_value": round(self.observed_value, 4) if self.observed_value is not None and not math.isnan(self.observed_value) else None,
            "predicted_value": round(self.predicted_value, 4) if self.predicted_value is not None and not math.isnan(self.predicted_value) else None,
            "raw_residual": round(self.raw_residual, 4) if self.raw_residual is not None and not math.isnan(self.raw_residual) else None,
            "normalized_residual": round(self.normalized_residual, 4) if self.normalized_residual is not None and not math.isnan(self.normalized_residual) else None,
            "classification": self.classification.value,
            "role": self.role.value,
            "quality_status": self.quality_status,
            "observability_status": self.observability_status,
            "provenance": self.provenance.value,
            "rejection_reason": self.rejection_reason,
            "threshold_limit": self.threshold_limit,
            "threshold_direction": self.threshold_direction,
            "threshold_provenance": self.threshold_provenance.value if self.threshold_provenance else None,
        }


@dataclass(frozen=True)
class SubsystemEvidence:
    """
    Subsystem-level health evidence reflecting Phase 5 aggregation.
    """
    subsystem: str
    health_score: Optional[float]
    contributing_primary_channels: Tuple[str, ...]
    channel_health_scores: Dict[str, Optional[float]]
    diagnostic_channels: Tuple[str, ...]
    cylinder_channels: Tuple[str, ...]
    reason_codes: Tuple[ExplanationReasonCode, ...]
    provenance: EpistemicProvenance = EpistemicProvenance.ENGINEERING_HEURISTIC

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subsystem": self.subsystem,
            "health_score": round(self.health_score, 4) if self.health_score is not None and not math.isnan(self.health_score) else None,
            "contributing_primary_channels": list(self.contributing_primary_channels),
            "channel_health_scores": {
                k: round(v, 4) if v is not None and not math.isnan(v) else None
                for k, v in self.channel_health_scores.items()
            },
            "diagnostic_channels": list(self.diagnostic_channels),
            "cylinder_channels": list(self.cylinder_channels),
            "reason_codes": [rc.value for rc in self.reason_codes],
            "provenance": self.provenance.value,
        }


@dataclass(frozen=True)
class AnomalyEvidence:
    """
    Temporal anomaly detection evidence reflecting Phase 6 outputs.
    """
    anomaly_score: Optional[float]
    threshold: float
    status: str
    is_anomalous: bool
    persistence_seconds: float
    recovery_seconds: float
    contributing_channels: Tuple[str, ...]
    contributing_subsystems: Tuple[str, ...]
    evidence_coverage: float
    hi_raw_ref: Optional[float]
    reason_codes: Tuple[ExplanationReasonCode, ...]
    provenance: EpistemicProvenance = EpistemicProvenance.ENGINEERING_HEURISTIC

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomaly_score": round(self.anomaly_score, 4) if self.anomaly_score is not None and not math.isnan(self.anomaly_score) else None,
            "threshold": self.threshold,
            "status": self.status,
            "is_anomalous": self.is_anomalous,
            "persistence_seconds": round(self.persistence_seconds, 2),
            "recovery_seconds": round(self.recovery_seconds, 2),
            "contributing_channels": list(self.contributing_channels),
            "contributing_subsystems": list(self.contributing_subsystems),
            "evidence_coverage": round(self.evidence_coverage, 4),
            "hi_raw_ref": round(self.hi_raw_ref, 4) if self.hi_raw_ref is not None and not math.isnan(self.hi_raw_ref) else None,
            "reason_codes": [rc.value for rc in self.reason_codes],
            "provenance": self.provenance.value,
        }


@dataclass(frozen=True)
class DiagnosisEvidence:
    """
    Fault diagnosis evidence reflecting Phase 6 outputs without probability claims.
    """
    primary_fault: str
    status: str
    confidence_heuristic: float
    ranked_hypotheses: Tuple[Dict[str, Any], ...]
    supporting_channels: Tuple[str, ...]
    residual_directions: Dict[str, str]
    affected_cylinder: Optional[int]
    is_sensor_fault: bool
    uncertainty_heuristic: float
    competing_hypotheses_count: int
    rule_evidence: Tuple[str, ...]
    reason_codes: Tuple[ExplanationReasonCode, ...]
    provenance: EpistemicProvenance = EpistemicProvenance.ENGINEERING_HEURISTIC

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_fault": self.primary_fault,
            "status": self.status,
            "confidence_heuristic": round(self.confidence_heuristic, 4),
            "ranked_hypotheses": list(self.ranked_hypotheses),
            "supporting_channels": list(self.supporting_channels),
            "residual_directions": dict(self.residual_directions),
            "affected_cylinder": self.affected_cylinder,
            "is_sensor_fault": self.is_sensor_fault,
            "uncertainty_heuristic": round(self.uncertainty_heuristic, 4),
            "competing_hypotheses_count": self.competing_hypotheses_count,
            "rule_evidence": list(self.rule_evidence),
            "reason_codes": [rc.value for rc in self.reason_codes],
            "provenance": self.provenance.value,
        }


@dataclass(frozen=True)
class PrognosticEvidence:
    """
    Degradation and Remaining Useful Life (RUL) evidence reflecting Phase 8 outputs.
    """
    degradation_index: Optional[float]
    trend_slope_per_second: Optional[float]
    trend_slope_per_hour: Optional[float]
    trend_window_seconds: float
    observation_count: int
    valid_fraction: float
    data_confidence: float
    dominant_subsystem: Optional[str]
    rul_status: str
    rul_estimate_hours: Optional[float]
    rul_low_hours: Optional[float]
    rul_high_hours: Optional[float]
    eol_threshold: float
    stress_multiplier: float
    rejection_reasons: Tuple[str, ...]
    reason_codes: Tuple[ExplanationReasonCode, ...]
    provenance: EpistemicProvenance = EpistemicProvenance.ENGINEERING_HEURISTIC

    def to_dict(self) -> Dict[str, Any]:
        return {
            "degradation_index": round(self.degradation_index, 4) if self.degradation_index is not None and not math.isnan(self.degradation_index) else None,
            "trend_slope_per_second": round(self.trend_slope_per_second, 6) if self.trend_slope_per_second is not None and not math.isnan(self.trend_slope_per_second) else None,
            "trend_slope_per_hour": round(self.trend_slope_per_hour, 4) if self.trend_slope_per_hour is not None and not math.isnan(self.trend_slope_per_hour) else None,
            "trend_window_seconds": round(self.trend_window_seconds, 2),
            "observation_count": self.observation_count,
            "valid_fraction": round(self.valid_fraction, 4),
            "data_confidence": round(self.data_confidence, 4),
            "dominant_subsystem": self.dominant_subsystem,
            "rul_status": self.rul_status,
            "rul_estimate_hours": round(self.rul_estimate_hours, 3) if self.rul_estimate_hours is not None else None,
            "rul_low_hours": round(self.rul_low_hours, 3) if self.rul_low_hours is not None else None,
            "rul_high_hours": round(self.rul_high_hours, 3) if self.rul_high_hours is not None else None,
            "eol_threshold": self.eol_threshold,
            "stress_multiplier": round(self.stress_multiplier, 2),
            "rejection_reasons": list(self.rejection_reasons),
            "reason_codes": [rc.value for rc in self.reason_codes],
            "provenance": self.provenance.value,
        }


@dataclass(frozen=True)
class CompletenessAudit:
    """
    Explicit, non-arbitrary evidence completeness quantification.
    """
    scope: str
    required_fields: Tuple[str, ...]
    present_fields: Tuple[str, ...]
    missing_fields: Tuple[str, ...]
    total_required: int
    total_present: int
    completeness_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scope": self.scope,
            "total_required": self.total_required,
            "total_present": self.total_present,
            "completeness_ratio": round(self.completeness_ratio, 4),
            "required_fields": list(self.required_fields),
            "present_fields": list(self.present_fields),
            "missing_fields": list(self.missing_fields),
        }


@dataclass(frozen=True)
class StepEvidenceRecord:
    """
    Complete point-in-time structured evidence snapshot.
    Immutable, deterministically serializable, and auditable.
    """
    timestamp: float
    engine_id: str
    mission_id: Optional[str]
    source_telemetry_timestamp: float
    canonical_timestamp: float
    hi_raw: Optional[float]
    hi_smooth: Optional[float]
    engine_health_state: str
    channel_evidence: Dict[str, ChannelEvidence]
    subsystem_evidence: Dict[str, SubsystemEvidence]
    anomaly_evidence: Optional[AnomalyEvidence]
    diagnosis_evidence: Optional[DiagnosisEvidence]
    prognostic_evidence: Optional[PrognosticEvidence]
    traceability_chains: Dict[str, TraceabilityChain]
    overall_reason_codes: Tuple[ExplanationReasonCode, ...]
    data_quality_status: str
    observability_coverage: float
    synchronization_status: str
    completeness: CompletenessAudit
    provenance_metadata: Dict[str, str] = field(default_factory=dict)
    limitations: Tuple[str, ...] = (
        "Synthetic grey-box validation; not operational or certified engine diagnosis.",
        "Diagnostic confidence and compatibility scores are engineering heuristics, not calibrated probabilities.",
        "RUL estimates are model-defined projections to horizon D_EOL and do not represent OEM maintenance limits.",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to stable dictionary for deterministic serialization."""
        return {
            "timestamp": round(self.timestamp, 4),
            "engine_id": self.engine_id,
            "mission_id": self.mission_id,
            "source_telemetry_timestamp": round(self.source_telemetry_timestamp, 4),
            "canonical_timestamp": round(self.canonical_timestamp, 4),
            "hi_raw": round(self.hi_raw, 4) if self.hi_raw is not None and not math.isnan(self.hi_raw) else None,
            "hi_smooth": round(self.hi_smooth, 4) if self.hi_smooth is not None and not math.isnan(self.hi_smooth) else None,
            "engine_health_state": self.engine_health_state,
            "channel_evidence": {k: v.to_dict() for k, v in sorted(self.channel_evidence.items())},
            "subsystem_evidence": {k: v.to_dict() for k, v in sorted(self.subsystem_evidence.items())},
            "anomaly_evidence": self.anomaly_evidence.to_dict() if self.anomaly_evidence else None,
            "diagnosis_evidence": self.diagnosis_evidence.to_dict() if self.diagnosis_evidence else None,
            "prognostic_evidence": self.prognostic_evidence.to_dict() if self.prognostic_evidence else None,
            "traceability_chains": {k: v.to_dict() for k, v in sorted(self.traceability_chains.items())},
            "overall_reason_codes": [rc.value for rc in self.overall_reason_codes],
            "data_quality_status": self.data_quality_status,
            "observability_coverage": round(self.observability_coverage, 4),
            "synchronization_status": self.synchronization_status,
            "completeness": self.completeness.to_dict(),
            "provenance_metadata": dict(sorted(self.provenance_metadata.items())),
            "limitations": list(self.limitations),
        }

    def compute_sha256(self) -> str:
        """Compute deterministic SHA-256 hash of serialized JSON bytes."""
        serialized = json.dumps(self.to_dict(), sort_keys=True, indent=None)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class WhatIfScenarioDeltaEvidence:
    """
    Comparative scenario evidence reflecting Phase 10 What-If evaluation.
    """
    scenario_id: str
    baseline_scenario_id: str
    delta_metrics: Dict[str, Optional[float]]
    modeled_contributions: Dict[str, str]
    envelope_event_count_delta: int
    risk_index_delta: Optional[float]
    reason_codes: Tuple[ExplanationReasonCode, ...]
    provenance: EpistemicProvenance = EpistemicProvenance.SYNTHETIC_VALIDATION
    limitations: Tuple[str, ...] = (
        "Comparative differences represent simulated grey-box scenario effects, not empirical flight test data.",
        "Attributed modeled contributions are computational sensitivity partitions, not formal causal inference.",
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "baseline_scenario_id": self.baseline_scenario_id,
            "delta_metrics": {
                k: round(v, 4) if v is not None and not math.isnan(v) else None
                for k, v in sorted(self.delta_metrics.items())
            },
            "modeled_contributions": dict(sorted(self.modeled_contributions.items())),
            "envelope_event_count_delta": self.envelope_event_count_delta,
            "risk_index_delta": round(self.risk_index_delta, 4) if self.risk_index_delta is not None and not math.isnan(self.risk_index_delta) else None,
            "reason_codes": [rc.value for rc in self.reason_codes],
            "provenance": self.provenance.value,
            "limitations": list(self.limitations),
        }
