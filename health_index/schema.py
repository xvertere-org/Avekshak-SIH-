"""
Data schemas, enums, and configuration for Phase 9 Health Index + Degradation Tracking.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional
import numpy as np


class HealthState(str, Enum):
    """Engine operational health classification bands."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    SEVERELY_DEGRADED = "SEVERELY_DEGRADED"
    CRITICAL = "CRITICAL"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class DegradationTrend(str, Enum):
    """Direction and velocity of engine condition change."""
    STABLE = "STABLE"
    IMPROVING = "IMPROVING"
    DEGRADING = "DEGRADING"
    RAPIDLY_DEGRADING = "RAPIDLY_DEGRADING"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class HealthDataQuality(str, Enum):
    """Data quality and observation integrity status."""
    VALID = "VALID"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    SENSOR_ISOLATED = "SENSOR_ISOLATED"


# Project-defined engineering baseline weights for SIH26054 prototype
# Note: These are engineering assumptions, not certified aerospace airworthiness limits.
DEFAULT_CHANNEL_WEIGHTS: Dict[str, float] = {
    "oil_pressure": 0.20,
    "cht": 0.18,
    "egt": 0.16,
    "oil_temp": 0.16,
    "vibration": 0.14,
    "rpm": 0.10,
    "fuel_flow": 0.06,
}


@dataclass
class HealthIndexConfig:
    """Configuration parameters for Health Index and Degradation Tracking."""
    channel_weights: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_CHANNEL_WEIGHTS)
    )
    tau_nominal: float = 1.5            # Deadband threshold in normalized residual units
    tau_critical: float = 5.0           # Full physical degradation threshold in normalized residual units
    min_valid_channels: int = 4         # Minimum valid physical channels required for health index
    ewma_alpha: float = 0.15            # Causal EWMA filter coefficient (0 < alpha <= 1)
    rate_horizon_s: float = 10.0        # Elapsed history horizon required to compute degradation rate
    rapid_degrade_rate: float = -0.010  # Rate threshold for RAPIDLY_DEGRADING (s^-1)
    degrade_rate: float = -0.001        # Rate threshold for DEGRADING (s^-1)
    improving_rate: float = 0.001       # Rate threshold for IMPROVING (s^-1)
    healthy_threshold: float = 0.85     # Lower bound for HEALTHY state
    degraded_threshold: float = 0.60    # Lower bound for DEGRADED state
    severely_degraded_threshold: float = 0.35  # Lower bound for SEVERELY_DEGRADED state

    # Deterministic observation-quality heuristic parameters
    sensor_isolation_persist_s: float = 5.0      # Continuous duration required for heuristic isolation
    sensor_isolation_max_gap_s: float = 2.0      # Maximum allowed timestamp gap before continuity resets
    sensor_isolation_outlier_sigma: float = 4.5  # Suspect channel residual threshold
    sensor_isolation_correlated_max_sigma: float = 1.2  # Max allowable residual on non-suspect channels

    # Timestamp robustness parameters
    max_timestamp_gap_s: float = 5.0             # Maximum allowed gap before breaking degradation history

    def __post_init__(self):
        # Validate weights
        total_w = sum(self.channel_weights.values())
        if abs(total_w - 1.0) > 1e-4:
            raise ValueError(f"Channel weights must sum to 1.0, got {total_w:.4f}")
        for ch, w in self.channel_weights.items():
            if w < 0.0:
                raise ValueError(f"Channel weight for '{ch}' must be non-negative, got {w}")


@dataclass
class HealthIndexResult:
    """
    Structured Health Index and degradation tracking output per telemetry sample.
    """
    # Provenance
    timestamp: float
    engine_id: str
    mission_id: str
    mission_phase: str

    # Core Health Outputs
    raw_health_index: float                # Instantaneous health [0.0, 1.0] or NaN
    smoothed_health_index: float           # Causally filtered health [0.0, 1.0] or NaN
    health_state: str                      # HealthState enum value
    raw_degradation_score: float           # D_raw [0.0, 1.0] or NaN

    # Degradation Tracking
    degradation_rate: float                # dHI/dt in s^-1 or NaN
    degradation_trend: str                 # DegradationTrend enum value

    # Degradation Attribution (NOT Fault Diagnosis)
    channel_contributions: Dict[str, float]       # Per-channel c_i contribution to D_raw
    channel_degradation_evidence: Dict[str, float]# Raw d_i evidence per channel
    dominant_degraded_channels: List[str]         # Channels with significant degradation evidence

    # Channel Audit & Data Quality
    valid_channels: List[str]                     # Channels with finite numeric residual
    missing_channels: List[str]                   # Channels with NaN residual
    excluded_channels: List[str]                  # Isolated sensor-fault channels
    effective_channel_weights: Dict[str, float]   # Dynamically renormalized weights w'_i
    data_quality: str                             # HealthDataQuality enum value

    # Optional Upstream Context (Passthrough only, NOT computed by Phase 9)
    diagnosed_fault: Optional[str] = None         # Phase 8 fault label if passed
    diagnostic_confidence: Optional[float] = None # Phase 8 confidence if passed
    anomaly_status: Optional[str] = None          # Phase 7 status if passed
    anomaly_score: Optional[float] = None         # Phase 7 score if passed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp,
            "engine_id": self.engine_id,
            "mission_id": self.mission_id,
            "mission_phase": self.mission_phase,
            "raw_health_index": self.raw_health_index,
            "smoothed_health_index": self.smoothed_health_index,
            "health_state": self.health_state,
            "raw_degradation_score": self.raw_degradation_score,
            "degradation_rate": self.degradation_rate,
            "degradation_trend": self.degradation_trend,
            "channel_contributions": dict(self.channel_contributions),
            "channel_degradation_evidence": dict(self.channel_degradation_evidence),
            "dominant_degraded_channels": list(self.dominant_degraded_channels),
            "valid_channels": list(self.valid_channels),
            "missing_channels": list(self.missing_channels),
            "excluded_channels": list(self.excluded_channels),
            "effective_channel_weights": dict(self.effective_channel_weights),
            "data_quality": self.data_quality,
            "diagnosed_fault": self.diagnosed_fault,
            "diagnostic_confidence": self.diagnostic_confidence,
            "anomaly_status": self.anomaly_status,
            "anomaly_score": self.anomaly_score,
        }
