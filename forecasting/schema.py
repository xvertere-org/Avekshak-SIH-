"""
Schema and data structures for Phase 10: TimesFM-3 Future Telemetry Forecasting.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


# Canonical 7 aero-piston telemetry channels
DEFAULT_FORECAST_CHANNELS: List[str] = [
    "rpm",
    "cht",
    "egt",
    "oil_temp",
    "oil_pressure",
    "fuel_flow",
    "vibration",
]

# Fixed pre-established engineering reference ranges (operational warning span)
# Defined in telemetry/quality.py (DEFAULT_QUALITY_ENVELOPES warning bounds)
# Denominators are strictly constant across models, splits, and evaluation windows.
DEFAULT_NRMSE_DENOMINATORS: Dict[str, float] = {
    "rpm": 4850.0,           # 1000.0 to 5850.0 RPM
    "cht": 130.0,            # 20.0 to 150.0 °C
    "egt": 550.0,            # 400.0 to 950.0 °C
    "oil_temp": 110.0,       # 20.0 to 130.0 °C
    "oil_pressure": 6.2,     # 0.8 to 7.0 bar
    "fuel_flow": 44.5,       # 0.5 to 45.0 L/h
    "vibration": 3.45,       # 0.05 to 3.5 g
}

# Canonical nominal expected baseline values for healthy cruise physics (Tier A / Digital Twin)
DEFAULT_NOMINAL_EXPECTED: Dict[str, float] = {
    "rpm": 5000.0,
    "cht": 100.0,
    "egt": 680.0,
    "oil_temp": 85.0,
    "oil_pressure": 4.5,
    "fuel_flow": 18.0,
    "vibration": 0.5,
}

# Physical bounds for sanity clamping where physically non-negative
PHYSICAL_LOWER_BOUNDS: Dict[str, float] = {
    "rpm": 0.0,
    "cht": -50.0,
    "egt": 0.0,
    "oil_temp": -50.0,
    "oil_pressure": 0.0,
    "fuel_flow": 0.0,
    "vibration": 0.0,
}


class ForecastQuality(str, Enum):
    """Quality classification for context data and generated forecasts."""
    VALID = "VALID"
    IMPUTED = "IMPUTED"
    INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"
    DEGRADED_INPUT = "DEGRADED_INPUT"


class ModelStatus(str, Enum):
    """Runtime status of forecasting model adapter."""
    LOADED_PRETRAINED = "LOADED_PRETRAINED"
    LOCAL_UNCHECKPOINTED_GRAPH = "LOCAL_UNCHECKPOINTED_GRAPH"
    BLOCKED_UNAUTHENTICATED_GATED = "BLOCKED_UNAUTHENTICATED_GATED"
    BASELINE = "BASELINE"


@dataclass
class ForecastingConfig:
    """Configuration parameters for telemetry trajectory forecasting."""
    context_length: int = 32
    forecast_horizon: int = 16
    sampling_interval_s: float = 1.0
    target_channels: List[str] = field(
        default_factory=lambda: list(DEFAULT_FORECAST_CHANNELS)
    )
    max_timestamp_gap_s: float = 5.0
    model_name: str = "timesfm-3.0"
    checkpoint_path: str = "google/timesfm-3.0-pytorch"
    device: Optional[str] = None
    clamp_to_physical_limits: bool = True
    enable_projected_health: bool = True

    def __post_init__(self):
        if self.context_length <= 0:
            raise ValueError(f"context_length must be positive, got {self.context_length}")
        if self.forecast_horizon <= 0:
            raise ValueError(f"forecast_horizon must be positive, got {self.forecast_horizon}")
        if self.sampling_interval_s <= 0:
            raise ValueError(f"sampling_interval_s must be positive, got {self.sampling_interval_s}")
        if not self.target_channels:
            raise ValueError("target_channels must not be empty")


@dataclass
class ForecastResult:
    """
    Structured trajectory forecasting result per telemetry observation point.
    """
    engine_id: str
    mission_id: Optional[str]
    forecast_start_timestamp: float
    context_start_timestamp: float
    context_length: int
    forecast_horizon: int
    sampling_interval: float
    target_channels: List[str]
    forecast_timestamps: List[float]
    predicted_telemetry: Dict[str, List[float]]
    lower_bounds: Optional[Dict[str, List[float]]] = None
    upper_bounds: Optional[Dict[str, List[float]]] = None
    projected_health_trajectory: Optional[List[float]] = None
    forecast_quality: str = ForecastQuality.VALID.value
    model_name: str = ""
    model_status: str = ""
    baseline_comparison: Optional[Dict[str, Any]] = None
    provenance: Dict[str, Any] = field(default_factory=dict)
