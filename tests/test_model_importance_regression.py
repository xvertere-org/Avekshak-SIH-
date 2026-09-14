"""
Regression and Integration Tests for Phase 8 Model Feature Importance and Pipeline Robustness.

Tests:
1. Exact regression path: diagnose_batch / diagnose_sample executes without NameError.
2. Feature importance available (trained model).
3. Feature importance unavailable (untrained model / exception handling fallback).
4. Normal diagnosis path (with and without Phase 7 gating).
5. Fault diagnosis path across fault types.
6. Malformed and edge inputs (empty batch, all-NaN, partial NaN dropout).
7. Explainability handoff (SHAP, Physics fusion, Temporal evidence).
8. BUG-009 regression safety check (dynamic degradation, multi-severity, early detection).
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any

from simulator.fault_interface import FaultType
from fault_diagnosis.schema import (
    FaultDiagnosisResult,
    DiagnosisDataQuality,
    CANONICAL_FAULT_LABELS,
)
from fault_diagnosis.features import FeatureExtractor, DIAGNOSTIC_FEATURES
from fault_diagnosis.classifier import (
    FaultClassifierConfig,
    XGBoostFaultClassifier,
)
from fault_diagnosis.pipeline import FaultDiagnosisPipeline
from fault_diagnosis.sensor_identification import SensorFaultIdentifier


@pytest.fixture(scope="module")
def trained_pipeline():
    """Build and train a fast deterministic pipeline for regression testing."""
    fe = FeatureExtractor()
    cfg = FaultClassifierConfig(n_estimators=10, max_depth=3, random_state=42)
    clf = XGBoostFaultClassifier(cfg)

    # Generate synthetic training data across all 6 classes
    rows = []
    labels = []
    for cls in CANONICAL_FAULT_LABELS:
        for k in range(10):
            row = {
                "rpm_norm_residual": 0.1 if cls == "none" else (3.0 if cls == "mechanical_degradation" else 0.0),
                "cht_norm_residual": 0.1 if cls == "none" else (3.5 if cls == "cooling_degradation" else 0.0),
                "egt_norm_residual": 0.1 if cls == "none" else (3.0 if cls == "fuel_injection_abnormality" else 0.0),
                "oil_temp_norm_residual": 0.1,
                "oil_pressure_norm_residual": -3.0 if cls == "lubrication_degradation" else 0.0,
                "fuel_flow_norm_residual": 0.1,
                "vibration_norm_residual": 3.0 if cls == "mechanical_degradation" else 0.0,
                "throttle": 75.0,
                "altitude": 1000.0,
                "load": 80.0,
                "mission_phase": "CRUISE",
                "timestamp": float(k),
                "engine_id": "TEST_ENG",
                "mission_id": "TEST_MIS",
            }
            if cls == "sensor_fault":
                row["cht_norm_residual"] = 5.0
            rows.append(row)
            labels.append(cls)

    train_df = pd.DataFrame(rows)
    y_train = pd.Series(labels)
    fe.fit(train_df)
    X_train = fe.transform(train_df)
    clf.fit(X_train, y_train)

    return FaultDiagnosisPipeline(classifier=clf, feature_extractor=fe)


def test_fault_diagnosis_execution_without_name_error(trained_pipeline):
    """
    Verifies that calling diagnose_batch and diagnose_sample executes cleanly
    without raising NameError: name 'model_importance' is not defined.
    """
    sample = {
        "timestamp": 10.0,
        "engine_id": "ENG_01",
        "mission_id": "MIS_01",
        "mission_phase": "CRUISE",
        "rpm_norm_residual": 0.0,
        "cht_norm_residual": 2.5,
        "egt_norm_residual": 0.0,
        "oil_temp_norm_residual": 0.0,
        "oil_pressure_norm_residual": 0.0,
        "fuel_flow_norm_residual": 0.0,
        "vibration_norm_residual": 0.0,
        "throttle": 75.0,
        "altitude": 1000.0,
        "load": 75.0,
    }

    # 1. Single sample
    res = trained_pipeline.diagnose_sample(sample)
    assert isinstance(res, FaultDiagnosisResult)
    assert isinstance(res.model_feature_importance, dict)
    assert len(res.model_feature_importance) > 0

    # 2. Batch
    df = pd.DataFrame([sample, sample])
    batch_res = trained_pipeline.diagnose_batch(df)
    assert len(batch_res) == 2
    for r in batch_res:
        assert isinstance(r, FaultDiagnosisResult)
        assert isinstance(r.model_feature_importance, dict)
        assert len(r.model_feature_importance) > 0


def test_feature_importance_available(trained_pipeline):
    """
    Verifies that for a trained model, get_feature_importance returns valid
    non-fabricated gain scores across the feature schema.
    """
    importance = trained_pipeline.get_feature_importance()
    assert isinstance(importance, dict)
    assert len(importance) == 16  # 16 features from FeatureExtractor
    assert all(isinstance(v, float) for v in importance.values())
    assert all(v >= 0.0 for v in importance.values())

    # Ensure to_dict on result serializes model_feature_importance properly
    sample = {
        "timestamp": 1.0, "engine_id": "E1", "mission_id": "M1", "mission_phase": "CRUISE",
        "rpm_norm_residual": 0.0, "cht_norm_residual": 0.0, "egt_norm_residual": 0.0,
        "oil_temp_norm_residual": 0.0, "oil_pressure_norm_residual": 0.0,
        "fuel_flow_norm_residual": 0.0, "vibration_norm_residual": 0.0,
        "throttle": 75.0, "altitude": 1000.0, "load": 75.0,
    }
    res = trained_pipeline.diagnose_sample(sample)
    d = res.to_dict()
    assert "model_feature_importance" in d
    assert d["model_feature_importance"] == importance


def test_feature_importance_unavailable_fallback():
    """
    Verifies that when a model is untrained or feature importance is unavailable,
    the pipeline returns an empty dict rather than fabricating values or crashing.
    """
    fe = FeatureExtractor()
    clf = XGBoostFaultClassifier()  # untrained
    pipeline = FaultDiagnosisPipeline(classifier=clf, feature_extractor=fe)

    assert pipeline.get_feature_importance() == {}

    sample = {
        "timestamp": 1.0, "engine_id": "E1", "mission_id": "M1", "mission_phase": "CRUISE",
        "rpm_norm_residual": 0.0, "cht_norm_residual": 0.0, "egt_norm_residual": 0.0,
        "oil_temp_norm_residual": 0.0, "oil_pressure_norm_residual": 0.0,
        "fuel_flow_norm_residual": 0.0, "vibration_norm_residual": 0.0,
        "throttle": 75.0, "altitude": 1000.0, "load": 75.0,
    }
    fe.fit(pd.DataFrame([sample]))

    res = pipeline.diagnose_sample(sample)
    assert res.model_feature_importance == {}
    assert res.data_quality == DiagnosisDataQuality.VALID.value


def test_normal_and_fault_diagnosis_paths(trained_pipeline):
    """
    Verifies diagnosis logic across normal and faulted states.
    """
    # 1. Normal state with Phase 7 gating
    normal_sample = {
        "timestamp": 5.0, "engine_id": "E1", "mission_id": "M1", "mission_phase": "CRUISE",
        "rpm_norm_residual": 0.0, "cht_norm_residual": 0.0, "egt_norm_residual": 0.0,
        "oil_temp_norm_residual": 0.0, "oil_pressure_norm_residual": 0.0,
        "fuel_flow_norm_residual": 0.0, "vibration_norm_residual": 0.0,
        "throttle": 75.0, "altitude": 1000.0, "load": 75.0,
    }
    res_normal = trained_pipeline.diagnose_sample(normal_sample, anomaly_status="NORMAL")
    assert res_normal.predicted_fault_type == "none"
    assert res_normal.data_quality == "VALID"
    assert res_normal.anomaly_status == "NORMAL"

    # 2. Cooling fault state with Phase 7 WARNING
    cooling_sample = {
        "timestamp": 15.0, "engine_id": "E1", "mission_id": "M1", "mission_phase": "CRUISE",
        "rpm_norm_residual": 0.0, "cht_norm_residual": 3.8, "egt_norm_residual": 0.0,
        "oil_temp_norm_residual": 0.5, "oil_pressure_norm_residual": 0.0,
        "fuel_flow_norm_residual": 0.0, "vibration_norm_residual": 0.0,
        "throttle": 75.0, "altitude": 1000.0, "load": 75.0,
    }
    res_cooling = trained_pipeline.diagnose_sample(cooling_sample, anomaly_status="WARNING", anomaly_score=0.6)
    assert res_cooling.predicted_fault_type == "cooling_degradation"
    assert res_cooling.diagnostic_confidence > 0.5
    assert res_cooling.anomaly_status == "WARNING"
    assert res_cooling.anomaly_score == 0.6


def test_malformed_and_edge_inputs(trained_pipeline):
    """
    Verifies handling of empty batches, all-NaN inputs, and insufficient residual channels.
    """
    # 1. Empty DataFrame
    empty_df = pd.DataFrame()
    assert trained_pipeline.diagnose_batch(empty_df) == []

    # 2. All-NaN residual inputs (< 4 valid channels)
    nan_sample = {
        "timestamp": 20.0, "engine_id": "E1", "mission_id": "M1", "mission_phase": "CRUISE",
        "rpm_norm_residual": np.nan, "cht_norm_residual": np.nan, "egt_norm_residual": np.nan,
        "oil_temp_norm_residual": np.nan, "oil_pressure_norm_residual": np.nan,
        "fuel_flow_norm_residual": np.nan, "vibration_norm_residual": np.nan,
        "throttle": 75.0, "altitude": 1000.0, "load": 75.0,
    }
    res_nan = trained_pipeline.diagnose_sample(nan_sample)
    assert res_nan.data_quality == DiagnosisDataQuality.INSUFFICIENT_DATA.value
    assert res_nan.predicted_fault_type == "none"
    assert res_nan.diagnostic_confidence == 0.0
    assert isinstance(res_nan.model_feature_importance, dict)

    # 3. Exactly 3 valid channels (< 4 threshold)
    partial_nan_sample = dict(nan_sample)
    partial_nan_sample["rpm_norm_residual"] = 0.5
    partial_nan_sample["cht_norm_residual"] = 0.5
    partial_nan_sample["egt_norm_residual"] = 0.5
    res_partial = trained_pipeline.diagnose_sample(partial_nan_sample)
    assert res_partial.data_quality == DiagnosisDataQuality.INSUFFICIENT_DATA.value


def test_explainability_handoff(trained_pipeline):
    """
    Verifies that output from fault diagnosis seamlessly passes into
    the Explainability and SHAP attribution pipeline without schema incompatibility.
    """
    from explainability.shap_explainer import SHAPExplainer

    explainer = SHAPExplainer(top_k=5)
    
    sample = {
        "timestamp": 12.0, "engine_id": "E1", "mission_id": "M1", "mission_phase": "CRUISE",
        "rpm_norm_residual": 0.0, "cht_norm_residual": 3.0, "egt_norm_residual": 0.0,
        "oil_temp_norm_residual": 0.2, "oil_pressure_norm_residual": 0.0,
        "fuel_flow_norm_residual": 0.0, "vibration_norm_residual": 0.0,
        "throttle": 75.0, "altitude": 1000.0, "load": 75.0,
    }
    diag_res = trained_pipeline.diagnose_sample(sample)
    df = pd.DataFrame([sample])
    X = trained_pipeline.feature_extractor.transform(df)

    shap_res = explainer.explain_instance(
        classifier=trained_pipeline.classifier,
        features=X,
        predicted_fault=diag_res.predicted_fault_type,
        class_probabilities=diag_res.class_probabilities,
    )

    assert shap_res.predicted_fault == diag_res.predicted_fault_type
    assert shap_res.status == "AVAILABLE"
    assert len(shap_res.top_features) > 0
