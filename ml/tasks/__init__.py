"""
Reusable Baseline ML Task Pipelines for SIH26054 Grey-Box Digital Twin.

Exports:
- Adapters: Schema-aware dataset loaders and target/group/feature resolvers.
- Preprocessing: Training-only LeakageSafePreprocessor.
- Pipelines:
    * FaultClassificationPipeline
    * AnomalyDetectionPipeline
    * DegradationPipeline
- Evaluation: Standardized metric calculators.
"""

from ml.tasks.exceptions import (
    MLTaskError,
    TargetNotFoundError,
    FeatureSelectionError,
    HealthyReferenceNotFoundError,
    InsufficientEntityError,
    TemporalOrderingError,
    IncompatibleDomainError,
)

from ml.tasks.adapters import (
    DatasetAdapter,
    AdapterConfig,
    DATASET_ADAPTER_REGISTRY,
)

from ml.tasks.preprocessing import (
    LeakageSafePreprocessor,
)

from ml.tasks.evaluation import (
    evaluate_classification,
    evaluate_anomaly_detection,
    evaluate_regression,
)

from ml.tasks.classification import (
    FaultClassificationPipeline,
    FaultClassificationReport,
    ClassificationModelResult,
)

from ml.tasks.anomaly_detection import (
    AnomalyDetectionPipeline,
    AnomalyDetectionReport,
    AnomalyModelResult,
)

from ml.tasks.degradation import (
    DegradationPipeline,
    DegradationReport,
    RegressionModelResult,
)

from ml.tasks.residual_correction import (
    GreyBoxResidualPipeline,
    ResidualCorrectionConfig,
    ModelComparisonResult,
    ChannelComparisonMetrics,
    ExplanationContribution,
    CANONICAL_TARGET_CHANNELS,
    TARGET_TO_PHYSICS_MAP,
    CHANNEL_UNITS,
)

from ml.tasks.health_prognostics import (
    EngineHealthState,
    AlertClassification,
    DegradationTrendState,
    ChannelUncertainty,
    HealthMonitoringOutput,
    FaultScenarioDefinition,
    get_standard_fault_catalog,
    HealthPrognosticsPipeline,
)

__all__ = [
    "MLTaskError",
    "TargetNotFoundError",
    "FeatureSelectionError",
    "HealthyReferenceNotFoundError",
    "InsufficientEntityError",
    "TemporalOrderingError",
    "IncompatibleDomainError",
    "DatasetAdapter",
    "AdapterConfig",
    "DATASET_ADAPTER_REGISTRY",
    "LeakageSafePreprocessor",
    "evaluate_classification",
    "evaluate_anomaly_detection",
    "evaluate_regression",
    "FaultClassificationPipeline",
    "FaultClassificationReport",
    "ClassificationModelResult",
    "AnomalyDetectionPipeline",
    "AnomalyDetectionReport",
    "AnomalyModelResult",
    "DegradationPipeline",
    "DegradationReport",
    "RegressionModelResult",
    "GreyBoxResidualPipeline",
    "ResidualCorrectionConfig",
    "ModelComparisonResult",
    "ChannelComparisonMetrics",
    "ExplanationContribution",
    "CANONICAL_TARGET_CHANNELS",
    "TARGET_TO_PHYSICS_MAP",
    "CHANNEL_UNITS",
    "EngineHealthState",
    "AlertClassification",
    "DegradationTrendState",
    "ChannelUncertainty",
    "HealthMonitoringOutput",
    "FaultScenarioDefinition",
    "get_standard_fault_catalog",
    "HealthPrognosticsPipeline",
]

