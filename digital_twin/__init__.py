"""
Digital Twin module for SIH26054 Aero Piston Engine Digital Twin.

Includes:
- Physics-informed nominal state estimation (twin_model.py)
- Dynamic residual generation and schema (residuals.py)
- Canonical state representation (state.py)
- Telemetry data quality and sensor validity (quality.py)
- Authoritative observability taxonomy and registry (observability.py)
- State estimation and deterministic synchronization (synchronizer.py)
- Deterministic historical replay (replay.py)
"""

from digital_twin.twin_model import DigitalTwin, DigitalTwinModel
from digital_twin.residuals import (
    ResidualFrame,
    ResidualGenerator,
    DEFAULT_RESIDUAL_SCALES,
    SUPPORTED_RESIDUAL_CHANNELS,
)
from digital_twin.state import (
    CanonicalTwinState,
    PhysicalQuantity,
    QuantityStatus,
    EngineOperatingRegime,
    SynchronizationStatus,
    RotationalState,
    AirBoostState,
    CombustionState,
    ThermalState,
    LubricationState,
    VibrationState,
    ElectricalState,
)
from digital_twin.quality import (
    DataQualityStatus,
    ChannelQuality,
    TelemetryQualityReport,
    TelemetryQualityValidator,
    PHYSICAL_INSTRUMENT_LIMITS,
)
from digital_twin.observability import (
    ObservabilityType,
    ObservabilityEntry,
    ObservabilityRegistry,
    CANONICAL_OBSERVABILITY_CATALOG,
)
from digital_twin.synchronizer import (
    StateEstimator,
    EstimatorConfig,
    DEFAULT_MODEL_RESIDUAL_SCALES,
)
from digital_twin.replay import (
    DigitalTwinReplay,
    ReplayStepRecord,
)

__all__ = [
    "DigitalTwin",
    "DigitalTwinModel",
    "ResidualFrame",
    "ResidualGenerator",
    "DEFAULT_RESIDUAL_SCALES",
    "SUPPORTED_RESIDUAL_CHANNELS",
    "CanonicalTwinState",
    "PhysicalQuantity",
    "QuantityStatus",
    "EngineOperatingRegime",
    "SynchronizationStatus",
    "RotationalState",
    "AirBoostState",
    "CombustionState",
    "ThermalState",
    "LubricationState",
    "VibrationState",
    "ElectricalState",
    "DataQualityStatus",
    "ChannelQuality",
    "TelemetryQualityReport",
    "TelemetryQualityValidator",
    "PHYSICAL_INSTRUMENT_LIMITS",
    "ObservabilityType",
    "ObservabilityEntry",
    "ObservabilityRegistry",
    "CANONICAL_OBSERVABILITY_CATALOG",
    "StateEstimator",
    "EstimatorConfig",
    "DEFAULT_MODEL_RESIDUAL_SCALES",
    "DigitalTwinReplay",
    "ReplayStepRecord",
]
