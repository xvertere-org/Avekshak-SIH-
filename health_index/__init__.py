"""
Phase 9 Health Index and Degradation Tracking Package for SIH26054.

Provides deterministic, causal Health Index (HI) calculation, channel attribution,
causal EWMA smoothing, and degradation rate / trend tracking.
"""

from health_index.schema import (
    HealthState,
    DegradationTrend,
    HealthDataQuality,
    DEFAULT_CHANNEL_WEIGHTS,
    HealthIndexConfig,
    HealthIndexResult,
)
from health_index.calculator import (
    HealthCalculator,
    compute_channel_evidence,
    SensorIsolationTracker,
)
from health_index.smoothing import CausalEWMASmoother
from health_index.degradation import DegradationTracker
from health_index.pipeline import HealthIndexPipeline

__all__ = [
    "HealthState",
    "DegradationTrend",
    "HealthDataQuality",
    "DEFAULT_CHANNEL_WEIGHTS",
    "HealthIndexConfig",
    "HealthIndexResult",
    "HealthCalculator",
    "compute_channel_evidence",
    "SensorIsolationTracker",
    "CausalEWMASmoother",
    "DegradationTracker",
    "HealthIndexPipeline",
]
