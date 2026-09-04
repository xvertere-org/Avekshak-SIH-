"""
Phase 8 Fault Diagnosis package for SIH26054.

Provides supervised multiclass XGBoost fault-diagnosis trained on
physics-informed synthetic aero-piston telemetry.
"""

from fault_diagnosis.schema import (
    CANONICAL_FAULT_LABELS,
    DiagnosisDataQuality,
    FaultDiagnosisResult,
)
from fault_diagnosis.features import FeatureExtractor
from fault_diagnosis.classifier import XGBoostFaultClassifier, FaultClassifierConfig
from fault_diagnosis.evaluation import evaluate_classifier
from fault_diagnosis.pipeline import FaultDiagnosisPipeline

__all__ = [
    "CANONICAL_FAULT_LABELS",
    "DiagnosisDataQuality",
    "FaultDiagnosisResult",
    "FeatureExtractor",
    "XGBoostFaultClassifier",
    "FaultClassifierConfig",
    "evaluate_classifier",
    "FaultDiagnosisPipeline",
]
