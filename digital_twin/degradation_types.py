"""
Phase 8 Degradation State and Remaining Useful Life (RUL) Types & Contracts.

Provides typed, immutable data models, enums, and provenance tracking for:
- Subsystem degradation dimensions
- Degradation regimes and operational states
- Uncertainty-aware RUL projections
- Stress scenario representations
- Evidence-based explainability contracts

SCIENTIFIC DISCLAIMER:
These contracts represent grey-box digital twin research abstractions.
They DO NOT represent certified Rotax 914 OEM maintenance thresholds,
airworthiness limits, or certified flight-hour predictions.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional


class DegradationSubsystem(str, Enum):
    """
    Subsystem degradation dimensions corresponding to engine subsystems.
    NOTE: These are digital twin model-level dimensions, not certified OEM failure modes.
    """
    THERMAL_DEGRADATION = "THERMAL_DEGRADATION"
    LUBRICATION_DEGRADATION = "LUBRICATION_DEGRADATION"
    COMBUSTION_DEGRADATION = "COMBUSTION_DEGRADATION"
    MECHANICAL_DEGRADATION = "MECHANICAL_DEGRADATION"
    FUEL_SYSTEM_DEGRADATION = "FUEL_SYSTEM_DEGRADATION"
    COOLING_DEGRADATION = "COOLING_DEGRADATION"


class DegradationRegime(str, Enum):
    """
    Classification of degradation trend magnitude and persistence.
    """
    STABLE = "STABLE"
    DEGRADING = "DEGRADING"
    RAPID_DEGRADATION = "RAPID_DEGRADATION"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class RULStatus(str, Enum):
    """
    Status of Remaining Useful Life estimation.
    A successful system must know when it does NOT have enough evidence.
    """
    COMPUTED = "COMPUTED"
    STABLE = "STABLE"
    NON_DEGRADING = "NON_DEGRADING"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    ALREADY_BEYOND_MODEL_HORIZON = "ALREADY_BEYOND_MODEL_HORIZON"
    DATA_QUALITY_DEGRADED = "DATA_QUALITY_DEGRADED"


class RULScenario(str, Enum):
    """
    Operating scenario profiles for remaining-life projections.
    Scenarios model future operational exposure multipliers, not certified flight envelopes.
    """
    CURRENT_PROFILE = "CURRENT_PROFILE"
    NORMAL_MISSION = "NORMAL_MISSION"
    HIGH_ALTITUDE = "HIGH_ALTITUDE"
    HOT_DAY = "HOT_DAY"
    HIGH_LOAD = "HIGH_LOAD"


class ProvenanceTag(str, Enum):
    """
    Epistemic provenance of configuration parameters and model thresholds.
    """
    EASA_REFERENCE = "EASA_REFERENCE"
    OEM_DOCUMENT = "OEM_DOCUMENT"
    MODEL_CALIBRATION = "MODEL_CALIBRATION"
    ENGINEERING_HEURISTIC = "ENGINEERING_HEURISTIC"
    SYNTHETIC_VARIATION = "SYNTHETIC_VARIATION"


# Scenario stress factors (relative degradation rate multiplier)
# Provenance: ENGINEERING_HEURISTIC
DEFAULT_SCENARIO_STRESS_FACTORS: Dict[RULScenario, float] = {
    RULScenario.CURRENT_PROFILE: 1.00,
    RULScenario.NORMAL_MISSION: 1.00,
    RULScenario.HIGH_ALTITUDE: 1.15,   # Reduced coolant air density / higher turbo pressure ratio
    RULScenario.HOT_DAY: 1.30,         # Elevated ambient heat rejection limit
    RULScenario.HIGH_LOAD: 1.50,       # Sustained max continuous / takeoff rating
}


@dataclass(frozen=True)
class SubsystemDegradationState:
    """
    Degradation state for an individual engine subsystem dimension.
    """
    subsystem: DegradationSubsystem
    degradation_index: float            # 0.0 (pristine) to 1.0 (fully degraded)
    trend_per_second: float             # Rate of degradation change per second
    trend_per_hour: float               # Rate of degradation change per hour
    regime: DegradationRegime
    confidence: float                   # 0.0 to 1.0 based on data quality & sample count
    observation_count: int
    window_start: float                 # Seconds timestamp
    window_end: float                   # Seconds timestamp
    data_quality: float                 # 0.0 to 1.0
    contributing_channels: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subsystem": self.subsystem.value,
            "degradation_index": round(self.degradation_index, 4) if not math.isnan(self.degradation_index) else None,
            "trend_per_second": round(self.trend_per_second, 6) if not math.isnan(self.trend_per_second) else None,
            "trend_per_hour": round(self.trend_per_hour, 4) if not math.isnan(self.trend_per_hour) else None,
            "regime": self.regime.value,
            "confidence": round(self.confidence, 4),
            "observation_count": self.observation_count,
            "window_start": round(self.window_start, 2),
            "window_end": round(self.window_end, 2),
            "data_quality": round(self.data_quality, 4),
            "contributing_channels": list(self.contributing_channels),
        }


@dataclass
class DegradationAssessment:
    """
    Comprehensive engine degradation state evaluated at a single time step.
    Derived strictly from observable Phase 5 residuals, health indices, and data quality.
    """
    timestamp: float
    engine_id: str
    degradation_index: float            # Primary index D(t) = 1.0 - HI_smooth (0.0 to 1.0)
    degradation_raw: float              # Raw instantaneous index D_raw(t) = 1.0 - HI_raw
    trend_slope_per_sec: float          # Robust Theil-Sen slope (dD/dt)
    trend_slope_per_hour: float         # Scaled slope (dD/d hour)
    slope_low_per_sec: float            # Pessimistic / lower bound slope
    slope_high_per_sec: float           # Optimistic / upper bound slope
    regime: DegradationRegime
    confidence: float                   # Overall confidence [0.0, 1.0]
    observation_count: int
    window_duration_s: float
    window_start_s: float
    window_end_s: float
    data_quality_factor: float          # From Phase 3 TelemetryQualityReport / C_data
    subsystems: Dict[str, SubsystemDegradationState] = field(default_factory=dict)
    dominant_subsystem: Optional[str] = None
    provenance: str = ProvenanceTag.ENGINEERING_HEURISTIC.value
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "engine_id": self.engine_id,
            "degradation_index": round(self.degradation_index, 4) if not math.isnan(self.degradation_index) else None,
            "degradation_raw": round(self.degradation_raw, 4) if not math.isnan(self.degradation_raw) else None,
            "trend_slope_per_sec": round(self.trend_slope_per_sec, 6) if not math.isnan(self.trend_slope_per_sec) else None,
            "trend_slope_per_hour": round(self.trend_slope_per_hour, 4) if not math.isnan(self.trend_slope_per_hour) else None,
            "slope_low_per_sec": round(self.slope_low_per_sec, 6) if not math.isnan(self.slope_low_per_sec) else None,
            "slope_high_per_sec": round(self.slope_high_per_sec, 6) if not math.isnan(self.slope_high_per_sec) else None,
            "regime": self.regime.value,
            "confidence": round(self.confidence, 4),
            "observation_count": self.observation_count,
            "window_duration_s": round(self.window_duration_s, 2),
            "window_start_s": round(self.window_start_s, 2),
            "window_end_s": round(self.window_end_s, 2),
            "data_quality_factor": round(self.data_quality_factor, 4),
            "subsystems": {k: v.to_dict() for k, v in self.subsystems.items()},
            "dominant_subsystem": self.dominant_subsystem,
            "provenance": self.provenance,
            "metadata": self.metadata,
        }


@dataclass
class RULAssessment:
    """
    Contract for uncertainty-aware Remaining Useful Life (RUL) estimation.
    Adheres strictly to the contract defined in Section 23 of the Phase 8 specification.
    """
    status: RULStatus
    degradation_index: float                    # Current D(t)
    degradation_trend: float                    # Robust slope (Theil-Sen)
    trend_unit: str = "delta_D_per_hour"
    trend_confidence: float = 0.0               # Confidence in the trend slope [0.0, 1.0]
    rul_low: Optional[float] = None             # Pessimistic projection (fastest degradation)
    rul_median: Optional[float] = None          # Central projection
    rul_high: Optional[float] = None            # Optimistic projection (slowest degradation)
    rul_unit: str = "hours"
    eol_threshold: float = 0.50                 # Model-defined horizon D_EOL
    data_confidence: float = 1.0                # Observability & sensor quality confidence
    sample_count: int = 0
    window_duration: float = 0.0                # History window in seconds
    scenario: str = RULScenario.CURRENT_PROFILE.value
    assumptions: str = (
        "Linear extrapolation of robust Theil-Sen degradation slope to model-defined D_EOL horizon. "
        "Not an OEM maintenance limit or airworthiness prediction."
    )
    provenance: str = ProvenanceTag.ENGINEERING_HEURISTIC.value
    explanation_evidence: List[Dict[str, Any]] = field(default_factory=list)
    scenario_projections: Dict[str, Dict[str, Optional[float]]] = field(default_factory=dict)
    rejection_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "degradation_index": round(self.degradation_index, 4) if not math.isnan(self.degradation_index) else None,
            "degradation_trend": round(self.degradation_trend, 6) if not math.isnan(self.degradation_trend) else None,
            "trend_unit": self.trend_unit,
            "trend_confidence": round(self.trend_confidence, 4),
            "rul_low": round(self.rul_low, 3) if self.rul_low is not None else None,
            "rul_median": round(self.rul_median, 3) if self.rul_median is not None else None,
            "rul_high": round(self.rul_high, 3) if self.rul_high is not None else None,
            "rul_unit": self.rul_unit,
            "eol_threshold": self.eol_threshold,
            "data_confidence": round(self.data_confidence, 4),
            "sample_count": self.sample_count,
            "window_duration": round(self.window_duration, 2),
            "scenario": self.scenario,
            "assumptions": self.assumptions,
            "provenance": self.provenance,
            "explanation_evidence": self.explanation_evidence,
            "scenario_projections": self.scenario_projections,
            "rejection_reason": self.rejection_reason,
            "metadata": self.metadata,
        }
