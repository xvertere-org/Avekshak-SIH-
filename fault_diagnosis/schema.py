"""
Core data schemas and contracts for Phase 8 Fault Diagnosis.

Defines the output data structures, data quality enum, and canonical fault taxonomy.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional

from simulator.fault_interface import FaultType


# Canonical 6-class fault taxonomy — sourced directly from FaultType enum values
CANONICAL_FAULT_LABELS: List[str] = sorted([ft.value for ft in FaultType])
# Result: ['cooling_degradation', 'fuel_injection_abnormality', 'lubrication_degradation',
#          'mechanical_degradation', 'none', 'sensor_fault']


class DiagnosisDataQuality(str, Enum):
    """Data quality status for fault diagnosis."""
    VALID = "VALID"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass
class FaultDiagnosisResult:
    """
    Structured fault diagnosis output for an individual telemetry sample.

    Phase 8 answers: "What is the most likely fault causing the abnormal behavior?"

    Notes:
    - predicted_fault_type uses canonical FaultType label strings.
    - diagnostic_confidence = max(class_probabilities) — the predicted probability
      of the most likely class. It is NOT certainty.
    - model_feature_importance is global XGBoost model-level importance (gain metric),
      NOT per-sample causal attribution. Per-sample SHAP is deferred to the
      Explainability phase.
    - anomaly_status and anomaly_score are optional provenance fields populated by
      the integration layer when Phase 7 results are available. They are NEVER
      computed by the classifier.
    """
    # Provenance
    timestamp: float
    engine_id: str
    mission_id: str
    mission_phase: str

    # Phase 8 core outputs
    predicted_fault_type: str
    class_probabilities: Dict[str, float]
    diagnostic_confidence: float

    # Data quality
    data_quality: str = DiagnosisDataQuality.VALID.value

    # Global model feature importance (NOT per-sample attribution)
    model_feature_importance: Dict[str, float] = field(default_factory=dict)

    # Optional Phase 7 context (populated by integration layer, NOT by classifier)
    anomaly_status: Optional[str] = None
    anomaly_score: Optional[float] = None

    # Sensor fault identification outputs (populated when predicted_fault_type == "sensor_fault")
    suspect_sensor: Optional[str] = None
    suspect_sensors: List[str] = field(default_factory=list)
    sensor_isolation_status: str = "NONE"  # "CONFIRMED", "UNCERTAIN", "NONE"

    @property
    def suspect_channel(self) -> Optional[str]:
        """Backward-compatible alias for suspect_sensor."""
        return self.suspect_sensor

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp,
            "engine_id": self.engine_id,
            "mission_id": self.mission_id,
            "mission_phase": self.mission_phase,
            "predicted_fault_type": self.predicted_fault_type,
            "class_probabilities": dict(self.class_probabilities),
            "diagnostic_confidence": self.diagnostic_confidence,
            "data_quality": self.data_quality,
            "model_feature_importance": dict(self.model_feature_importance),
            "anomaly_status": self.anomaly_status,
            "anomaly_score": self.anomaly_score,
            "suspect_sensor": self.suspect_sensor,
            "suspect_channel": self.suspect_sensor,
            "suspect_sensors": list(self.suspect_sensors),
            "sensor_isolation_status": self.sensor_isolation_status,
        }
