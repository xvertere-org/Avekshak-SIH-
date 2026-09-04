"""
Phase 7 — Hybrid Anomaly Detection Package for SIH26054.

Provides deterministic, explainable anomaly detection combining:
- Normalized residual thresholding
- EWMA temporal smoothing
- Persistence gating
- Unsupervised Isolation Forest multivariate scoring
- Rule-based evidence fusion
"""

from anomaly_detection.schema import (
    AnomalyStatus,
    AnomalyRecord,
    AnomalyFrame,
    PRIMARY_NORMALIZED_RESIDUAL_CHANNELS,
    RAW_RESIDUAL_CHANNELS,
)
from anomaly_detection.preprocessing import (
    ResidualPreprocessor,
    DEFAULT_MIN_VALID_FEATURES,
)
from anomaly_detection.detectors import (
    ResidualThresholdDetector,
    EWMADetector,
    PersistenceGate,
    IsolationForestDetector,
)
from anomaly_detection.fusion import EvidenceFusionEngine
from anomaly_detection.pipeline import HybridAnomalyDetector

__all__ = [
    "AnomalyStatus",
    "AnomalyRecord",
    "AnomalyFrame",
    "PRIMARY_NORMALIZED_RESIDUAL_CHANNELS",
    "RAW_RESIDUAL_CHANNELS",
    "ResidualPreprocessor",
    "DEFAULT_MIN_VALID_FEATURES",
    "ResidualThresholdDetector",
    "EWMADetector",
    "PersistenceGate",
    "IsolationForestDetector",
    "EvidenceFusionEngine",
    "HybridAnomalyDetector",
]
