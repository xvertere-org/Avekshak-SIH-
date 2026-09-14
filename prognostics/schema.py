"""
Data schemas, enums, and configuration for Phase 11 Remaining Useful Life (RUL) & Prognostics.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


class RULStatus(str, Enum):
    """Prognostic health trajectory status and RUL availability state."""
    ACTIVE_DEGRADATION = "ACTIVE_DEGRADATION"       # Valid degrading trajectory approaching EOL
    NOT_DEGRADING = "NOT_DEGRADING"                 # Nominal healthy cruise condition (RUL = NaN)
    RECOVERING = "RECOVERING"                       # Thermal or dynamic recovery (RUL = NaN)
    INDETERMINATE_TREND = "INDETERMINATE_TREND"     # Near-zero slope with degraded health (RUL = NaN)
    CRITICAL_EOL_REACHED = "CRITICAL_EOL_REACHED"   # Already at/below EOL or redline breached (RUL = 0.0 s)
    DEGRADED_PROGNOSTIC = "DEGRADED_PROGNOSTIC"     # Calculated with single sensor isolated
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"         # Missing >= 4 channels from Phase 9 (RUL = NaN)
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"   # Elapsed flight time < 32.0 s (RUL = NaN)
    EXCEEDS_HORIZON = "EXCEEDS_HORIZON"             # Projected RUL > reporting horizon (reported as >24h)


DEFAULT_HI_EOL_TOLERANCE: float = 1e-5  # Numerical tolerance for HI EOL threshold boundary


@dataclass
class EOLCriterion:
    """
    Individual End-of-Life (EOL) functional failure boundary with explicit provenance.
    """
    name: str
    channel: str                          # "health_index", "cht", "oil_pressure", etc.
    threshold_value: float
    comparison: str                       # "<=" or ">="
    source: str                           # Provenance tag
    rationale: str
    tolerance: float = 0.0                # Numerical tolerance for boundary evaluation

    def is_breached(self, value: float) -> bool:
        """Check if a measured or forecasted value breaches this EOL criterion."""
        if value is None or (isinstance(value, float) and (value != value)):  # NaN check
            return False
        v = float(value)
        if self.comparison == "<=":
            return v <= (self.threshold_value + self.tolerance)
        elif self.comparison == ">=":
            return v >= (self.threshold_value - self.tolerance)
        return False


@dataclass
class EOLCriteriaConfig:
    """
    Configured EOL thresholds for the weakest-link composite evaluator.
    All criteria are project-defined simulated engineering boundaries, NOT certified OEM limits.
    """
    hi_eol: EOLCriterion = field(
        default_factory=lambda: EOLCriterion(
            name="Functional Health Index EOL",
            channel="health_index",
            threshold_value=0.35,
            comparison="<=",
            source="phase_9_critical_state_boundary",
            rationale="Phase 9 boundary for CRITICAL health state; multi-subsystem divergence beyond 4-5 sigma.",
            tolerance=DEFAULT_HI_EOL_TOLERANCE,
        )
    )
    cht_redline: EOLCriterion = field(
        default_factory=lambda: EOLCriterion(
            name="Cylinder Head Temperature Redline",
            channel="cht",
            threshold_value=150.0,
            comparison=">=",
            source="telemetry_warning_bound_repurposed",
            rationale="Repurposed operational warning bound representing simulated cylinder head thermal ceiling.",
        )
    )
    oil_pressure_redline: EOLCriterion = field(
        default_factory=lambda: EOLCriterion(
            name="Minimum Oil Pressure Redline",
            channel="oil_pressure",
            threshold_value=1.2,
            comparison="<=",
            source="project_defined_failure_assumption",
            rationale="Simulated hydrodynamic film collapse threshold in flight, safely above 0.8 bar idle minimum.",
        )
    )
    oil_temp_redline: EOLCriterion = field(
        default_factory=lambda: EOLCriterion(
            name="Maximum Oil Temperature Redline",
            channel="oil_temp",
            threshold_value=140.0,
            comparison=">=",
            source="project_defined_failure_assumption",
            rationale="Simulated lubricant thermal cracking and viscosity failure limit exceeding 130 C warning bound.",
        )
    )
    vibration_redline: EOLCriterion = field(
        default_factory=lambda: EOLCriterion(
            name="Maximum Structural Vibration Redline",
            channel="vibration",
            threshold_value=3.5,
            comparison=">=",
            source="telemetry_warning_bound_repurposed",
            rationale="Repurposed operational warning bound representing severe mechanical unbalance limit.",
        )
    )

    def get_all_criteria(self) -> List[EOLCriterion]:
        return [
            self.hi_eol,
            self.cht_redline,
            self.oil_pressure_redline,
            self.oil_temp_redline,
            self.vibration_redline,
        ]


@dataclass
class RULConfig:
    """Configuration parameters for RUL inference and uncertainty estimation."""
    warmup_duration_s: float = 32.0              # Authoritative warmup lockout (Phase 10 context length)
    theil_sen_window_s: float = 60.0             # Historical time window for robust slope calculation
    theil_sen_min_samples: int = 5               # Minimum valid historical samples required for slope
    theil_sen_max_gap_s: float = 5.0             # Maximum timestamp gap before breaking history
    maximum_reportable_rul_s: float = 86400.0    # 24 hours: prognostic reporting horizon, NOT platform endurance
    slope_deadband: float = 0.0005               # s^-1: boundary for stationary / indeterminate slope
    mc_samples: int = 500                        # Default Monte Carlo trajectory realization count
    confidence_discount_sensor_fault: float = 0.20 # Confidence penalty when an isolated sensor fault is present
    eol_criteria: EOLCriteriaConfig = field(default_factory=EOLCriteriaConfig)

    # Engineering uncertainty assumptions for synthetic-data prognostics (NOT calibrated real-engine distributions)
    assumed_residual_std: float = 0.02           # Initial health state standard deviation
    assumed_slope_rel_std: float = 0.15          # Relative standard deviation of degradation slope
    assumed_threshold_half_width: float = 0.02   # EOL threshold tolerance half-width (+/- 0.02)


@dataclass
class RULResult:
    """
    Structured Remaining Useful Life (RUL) estimation output per telemetry observation.
    """
    engine_id: str
    mission_id: Optional[str]
    timestamp: float
    status: RULStatus
    rul_seconds_median: Optional[float]          # 50th percentile (point estimate) or None
    rul_seconds_p05: Optional[float]             # 5th percentile (conservative lower bound) or None
    rul_seconds_p95: Optional[float]             # 95th percentile (optimistic upper bound) or None
    limiting_factor: str                         # "GLOBAL_HEALTH_INDEX", "REDLINE_CHT", etc.
    confidence_score: float                      # [0.0, 1.0] based on history length and sensor isolation
    active_flight_phase: str                     # e.g. "CRUISE", "CLIMB", "DESCENT"
    handoff_horizon_s: float                     # 0.0, 16.0, or 32.0 s
    trajectory_type: str                         # "ROBUST_LINEAR_PRIMARY", "ACCELERATED_STRESS", "BASELINE_EWMA"
    provenance: Dict[str, Any] = field(default_factory=dict)
    forecast_assisted: bool = False              # True only if forecast trajectory or redline crossing was actually used
    forecast_mode_status: str = "OFF"            # "ACTIVE", "OFF", "BLOCKED", or "UNAVAILABLE"

    @property
    def forecast_assisted_mode(self) -> bool:
        """Alias for forecast_assisted for full naming consistency."""
        return self.forecast_assisted

    def to_dict(self) -> Dict[str, Any]:
        """Convert RULResult to standard dictionary representation."""
        return {
            "engine_id": self.engine_id,
            "mission_id": self.mission_id,
            "timestamp": self.timestamp,
            "status": self.status.value if hasattr(self.status, "value") else str(self.status),
            "state": self.status.value if hasattr(self.status, "value") else str(self.status),
            "rul_seconds_median": self.rul_seconds_median,
            "median": self.rul_seconds_median,
            "rul_seconds": self.rul_seconds_median,
            "point_rul_seconds": self.rul_seconds_median,
            "rul_seconds_p05": self.rul_seconds_p05,
            "p05": self.rul_seconds_p05,
            "rul_seconds_p95": self.rul_seconds_p95,
            "p95": self.rul_seconds_p95,
            "limiting_factor": self.limiting_factor,
            "confidence_score": self.confidence_score,
            "active_flight_phase": self.active_flight_phase,
            "handoff_horizon_s": self.handoff_horizon_s,
            "trajectory_type": self.trajectory_type,
            "forecast_assisted": self.forecast_assisted,
            "forecast_assisted_mode": self.forecast_assisted,
            "forecast_mode_status": self.forecast_mode_status,
            "eol_provenance": dict(self.provenance),
            "provenance": dict(self.provenance),
        }

