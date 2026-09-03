"""
Telemetry module for SIH26054 Digital Twin.

Includes:
- Typed data contracts and schemas (schema.py)
- Ingestion and buffering interfaces (streamer.py, ingestion.py)
- Non-destructive data quality pipeline (quality.py)
- Synchronization and causal resampling (preprocessing.py)
- Causal feature engineering and vibration analytics (features.py)
- External benchmark dataset adapters (adapters.py)
- End-to-end pipeline coordinator (pipeline.py)
"""

from telemetry.schema import (
    MissionPhase,
    FaultCategory,
    EngineConfig,
    MissionConfig,
    TelemetryRecord,
    DigitalTwinState,
    HealthAssessment,
    RULPrediction,
    ExplanationReport,
)
from telemetry.streamer import TelemetryStreamer
from telemetry.ingestion import (
    CanonicalTelemetryFrame,
    TelemetryIngestor,
    REQUIRED_COLUMNS,
    OPTIONAL_PHYSICAL_CHANNELS,
    OPTIONAL_METADATA_COLUMNS,
    ALL_CANONICAL_COLUMNS,
)
from telemetry.quality import (
    QualityStatus,
    QualityEnvelope,
    ChannelQualitySummary,
    DataQualityReport,
    DataQualityChecker,
    DEFAULT_QUALITY_ENVELOPES,
)
from telemetry.preprocessing import (
    TelemetrySynchronizer,
    TelemetryResampler,
    TelemetryCleaner,
)
from telemetry.features import (
    TelemetryFeatureExtractor,
    LeakageSafeSplitter,
)
from telemetry.adapters import (
    ExternalDatasetAdapter,
    GenericCSVAdapter,
    VibrationBenchmarkAdapter,
    CMAPSSBenchmarkAdapter,
)
from telemetry.pipeline import (
    TelemetryPipeline,
    PipelineResult,
)

__all__ = [
    # Schema
    "MissionPhase",
    "FaultCategory",
    "EngineConfig",
    "MissionConfig",
    "TelemetryRecord",
    "DigitalTwinState",
    "HealthAssessment",
    "RULPrediction",
    "ExplanationReport",
    # Streamer
    "TelemetryStreamer",
    # Ingestion
    "CanonicalTelemetryFrame",
    "TelemetryIngestor",
    "REQUIRED_COLUMNS",
    "OPTIONAL_PHYSICAL_CHANNELS",
    "OPTIONAL_METADATA_COLUMNS",
    "ALL_CANONICAL_COLUMNS",
    # Quality
    "QualityStatus",
    "QualityEnvelope",
    "ChannelQualitySummary",
    "DataQualityReport",
    "DataQualityChecker",
    "DEFAULT_QUALITY_ENVELOPES",
    # Preprocessing
    "TelemetrySynchronizer",
    "TelemetryResampler",
    "TelemetryCleaner",
    # Features
    "TelemetryFeatureExtractor",
    "LeakageSafeSplitter",
    # Adapters
    "ExternalDatasetAdapter",
    "GenericCSVAdapter",
    "VibrationBenchmarkAdapter",
    "CMAPSSBenchmarkAdapter",
    # Pipeline
    "TelemetryPipeline",
    "PipelineResult",
]
