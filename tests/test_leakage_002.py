import pytest
import numpy as np
import pandas as pd
from fault_diagnosis.pipeline import FaultDiagnosisPipeline
from orchestrator.schema import DiagnosisDataQuality

class MockClassifier:
    def __init__(self):
        self.is_trained = True
        self.classes = ["none", "compressor_degradation"]
        
    def predict(self, X):
        return ["compressor_degradation"] * len(X)
        
    def predict_proba(self, X):
        return np.array([[0.05, 0.95] for _ in range(len(X))])
        
    def get_feature_importance(self):
        return {"rpm_norm_residual": 0.5, "egt_norm_residual": 0.5}

class MockFeatureExtractor:
    def transform(self, X):
        return X

@pytest.fixture
def base_pipeline():
    return FaultDiagnosisPipeline(
        classifier=MockClassifier(),
        feature_extractor=MockFeatureExtractor(),
        enable_phase7_gating=True
    )

@pytest.fixture
def sample_residual():
    return {
        'rpm_residual': 100.0,
        'timestamp': 1.0,
        'rpm_norm_residual': 10.0,
        'egt_norm_residual': 5.0,
        'vibration_norm_residual': 2.0,
        'oil_pressure_norm_residual': 1.0,
        'cht_norm_residual': 1.0,
        'fuel_flow_norm_residual': 1.0,
        'oil_temp_norm_residual': 1.0,
        'engine_id': 'ENG_001',
        'mission_id': 'MISS_001',
        'mission_phase': 'CRUISE'
    }

def test_sudden_fault(base_pipeline, sample_residual):
    # Sudden fault triggers anomaly detector correctly
    res = base_pipeline.diagnose_sample(sample_residual, anomaly_status='ANOMALY', anomaly_score=0.9)
    assert res.predicted_fault_type == 'compressor_degradation'
    assert res.diagnostic_confidence == 0.95

def test_gradual_fault(base_pipeline, sample_residual):
    # Gradual fault triggers WARNING first
    res = base_pipeline.diagnose_sample(sample_residual, anomaly_status='WARNING', anomaly_score=0.6)
    assert res.predicted_fault_type == 'compressor_degradation'

def test_false_negative_anomaly_detector(base_pipeline, sample_residual):
    # Anomaly detector misses it (NORMAL), but classifier is highly confident (0.95)
    res = base_pipeline.diagnose_sample(sample_residual, anomaly_status='NORMAL', anomaly_score=0.1)
    assert res.predicted_fault_type == 'compressor_degradation', "Genuine fault was suppressed!"
    assert res.diagnostic_confidence == 0.95

def test_anomaly_detector_unavailable(base_pipeline, sample_residual):
    # Anomaly detector fails or returns INSUFFICIENT_DATA
    res = base_pipeline.diagnose_sample(sample_residual, anomaly_status='INSUFFICIENT_DATA', anomaly_score=float('nan'))
    assert res.predicted_fault_type == 'compressor_degradation'
    assert res.diagnostic_confidence == 0.95

def test_anomaly_recovery(base_pipeline, sample_residual):
    # Recovers from ANOMALY back to NORMAL but fault still present
    res = base_pipeline.diagnose_sample(sample_residual, anomaly_status='NORMAL', anomaly_score=0.2)
    assert res.predicted_fault_type == 'compressor_degradation'

def test_repeated_false_negatives(base_pipeline, sample_residual):
    # Repeatedly misses it
    for i in range(5):
        sample_residual['timestamp'] += 1.0
        res = base_pipeline.diagnose_sample(sample_residual, anomaly_status='NORMAL', anomaly_score=0.1)
        assert res.predicted_fault_type == 'compressor_degradation'

def test_sensor_dropout(base_pipeline, sample_residual):
    # Missing residual keys simulate sensor dropout.
    # The pipeline should handle the missing key safely.
    sample_residual.pop('rpm_norm_residual')
    res = base_pipeline.diagnose_sample(sample_residual, anomaly_status='NORMAL', anomaly_score=0.1)
    # The MockFeatureExtractor just returns X, which will miss the key. 
    # Real feature extractor would handle imputation.
    assert res.predicted_fault_type == 'compressor_degradation'

def test_overlapping_faults(base_pipeline, sample_residual):
    # For overlapping faults, classifier would return the most likely one.
    # The gating behavior should not block it.
    res = base_pipeline.diagnose_sample(sample_residual, anomaly_status='ANOMALY', anomaly_score=0.99)
    assert res.predicted_fault_type == 'compressor_degradation'

def test_fault_during_recovery(base_pipeline, sample_residual):
    # Anomaly detector is recovering (e.g. NORMAL) but classifier detects it
    res = base_pipeline.diagnose_sample(sample_residual, anomaly_status='NORMAL', anomaly_score=0.4)
    assert res.predicted_fault_type == 'compressor_degradation'

def test_fresh_process_execution(base_pipeline, sample_residual):
    # Simulates first execution
    res = base_pipeline.diagnose_sample(sample_residual, anomaly_status='NORMAL', anomaly_score=0.1)
    assert res.predicted_fault_type == 'compressor_degradation'

def test_normal_operation_is_not_flagged(base_pipeline, sample_residual):
    # If the classifier predicts none (high confidence), it should remain none
    base_pipeline.classifier = type('Mock', (), {
        'is_trained': True,
        'classes': ["none", "compressor_degradation"],
        'get_feature_importance': lambda self: {},
        'predict': lambda self, X: ["none"] * len(X),
        'predict_proba': lambda self, X: np.array([[0.99, 0.01] for _ in range(len(X))])
    })()
    res = base_pipeline.diagnose_sample(sample_residual, anomaly_status='NORMAL', anomaly_score=0.1)
    assert res.predicted_fault_type == 'none'
    assert res.diagnostic_confidence == 1.0 # Due to NORMAL gating

def test_low_confidence_override_rejected(base_pipeline, sample_residual):
    # If classifier predicts fault but with low confidence (<0.75), NORMAL gating suppresses it
    base_pipeline.classifier = type('Mock', (), {
        'is_trained': True,
        'classes': ["none", "compressor_degradation"],
        'get_feature_importance': lambda self: {},
        'predict': lambda self, X: ["compressor_degradation"] * len(X),
        'predict_proba': lambda self, X: np.array([[0.4, 0.6] for _ in range(len(X))])
    })()
    res = base_pipeline.diagnose_sample(sample_residual, anomaly_status='NORMAL', anomaly_score=0.1)
    assert res.predicted_fault_type == 'none'
    assert res.diagnostic_confidence == 1.0 # Suppressed
