"""
Tests for Phase 8 Supervised Fault Diagnosis with XGBoost.

Verifies:
- Canonical 6-class fault taxonomy
- Simulator dataset generation matrix (121 runs, 6 classes, sensor coverage)
- Mission-aware grouped stratified splitting (no mission_run_id overlap)
- Leakage prevention (no targets or generation metadata in features)
- Feature extraction with 16 columns (7 residuals + 3 context + 6 one-hot phase)
- Unseen phase handling (deterministic zeros, no crashes, 16 columns preserved)
- Native NaN preservation (no zero-imputation)
- Insufficient data handling (< 4 valid residuals -> INSUFFICIENT_DATA)
- Deterministic XGBoost training and model save/load
- Correct diagnosis across all fault types
- Evaluation metrics, confusion matrix, and NONE false positive rate
- Independence from Phase 7 and full provenance tracking
"""

import os
import shutil
import tempfile
import pytest
import numpy as np
import pandas as pd

from simulator.fault_interface import FaultType
from simulator.sensor_faults import SensorChannel, SensorFaultMode
from simulator.subsystems.mission import MissionProfile, PhaseSegment, FlightPhase
from fault_diagnosis.schema import (
    CANONICAL_FAULT_LABELS,
    DiagnosisDataQuality,
    FaultDiagnosisResult,
)
from fault_diagnosis.dataset import (
    DatasetConfig,
    generate_fault_diagnosis_dataset,
    get_dataset_summary,
    _run_mission,
    HEALTHY_SEEDS,
)
from fault_diagnosis.features import (
    FeatureExtractor,
    DIAGNOSTIC_FEATURES,
    CONTEXT_CONTINUOUS_FEATURES,
    EXPECTED_FEATURE_COUNT,
    split_by_mission_run,
    compute_class_weights,
    get_split_summary,
)
from fault_diagnosis.classifier import (
    FaultClassifierConfig,
    XGBoostFaultClassifier,
)
from fault_diagnosis.evaluation import (
    evaluate_classifier,
    format_confusion_matrix_ascii,
)
from fault_diagnosis.pipeline import FaultDiagnosisPipeline


# Fixture for fast 2-step profile to test generation matrix without simulation lag
@pytest.fixture(scope="module")
def fast_dataset_config():
    prof = MissionProfile(
        mission_id="FAST_TEST_PROFILE",
        segments=[
            PhaseSegment(
                phase=FlightPhase.CRUISE,
                duration_s=2.0,
                throttle_start_pct=75.0,
                throttle_end_pct=75.0,
                altitude_start_m=1000.0,
                altitude_end_m=1000.0,
            )
        ],
    )
    return DatasetConfig(dt=1.0, fault_onset_time=1.0, mission_profile=prof)


@pytest.fixture(scope="module")
def fast_generated_dataset(fast_dataset_config):
    """Generates the full 121-run dataset matrix using a fast 2-step profile (~1.5s total)."""
    return generate_fault_diagnosis_dataset(config=fast_dataset_config, verbose=False)


# =====================================================================
# Test A: Canonical Fault Taxonomy
# =====================================================================
def test_a_canonical_fault_taxonomy():
    """Verify exactly 6 canonical classes match FaultType enum values."""
    expected_classes = sorted([ft.value for ft in FaultType])
    assert len(CANONICAL_FAULT_LABELS) == 6
    assert CANONICAL_FAULT_LABELS == expected_classes
    assert "none" in CANONICAL_FAULT_LABELS
    assert "cooling_degradation" in CANONICAL_FAULT_LABELS
    assert "lubrication_degradation" in CANONICAL_FAULT_LABELS
    assert "fuel_injection_abnormality" in CANONICAL_FAULT_LABELS
    assert "mechanical_degradation" in CANONICAL_FAULT_LABELS
    assert "sensor_fault" in CANONICAL_FAULT_LABELS


# =====================================================================
# Test B: Dataset Generation Produces All Classes
# =====================================================================
def test_b_dataset_generation_produces_all_classes(fast_generated_dataset):
    """Generated dataset contains all 6 fault classes."""
    classes_in_dataset = set(fast_generated_dataset["fault_type"].unique())
    assert classes_in_dataset == set(CANONICAL_FAULT_LABELS)


# =====================================================================
# Test C: Mission Run ID Uniqueness
# =====================================================================
def test_c_mission_run_id_uniqueness(fast_generated_dataset):
    """Every independent mission run has a unique mission_run_id."""
    run_ids = fast_generated_dataset["mission_run_id"].unique()
    assert len(run_ids) == 121
    # Check that each run_id has records
    counts = fast_generated_dataset.groupby("mission_run_id").size()
    assert (counts > 0).all()


# =====================================================================
# Test D: Class Run Counts
# =====================================================================
def test_d_class_run_counts(fast_generated_dataset):
    """Expected number of runs per class: 6/6/6/6/6/91 = 121 total."""
    runs_per_class = fast_generated_dataset.groupby("fault_type")["mission_run_id"].nunique().to_dict()
    assert runs_per_class["none"] == 6
    assert runs_per_class["cooling_degradation"] == 6
    assert runs_per_class["lubrication_degradation"] == 6
    assert runs_per_class["fuel_injection_abnormality"] == 6
    assert runs_per_class["mechanical_degradation"] == 6
    assert runs_per_class["sensor_fault"] == 91
    assert sum(runs_per_class.values()) == 121


# =====================================================================
# Test E: Sensor Channel and Mode Coverage
# =====================================================================
def test_e_sensor_channel_mode_coverage(fast_generated_dataset):
    """All 7 channels and all 5 modes are covered in sensor_fault runs."""
    sensor_df = fast_generated_dataset[fast_generated_dataset["fault_type"] == "sensor_fault"]
    channels = set(sensor_df["generation_sensor_channel"].unique())
    modes = set(sensor_df["generation_sensor_mode"].unique())

    expected_channels = {sc.value for sc in SensorChannel}
    expected_modes = {sm.value for sm in SensorFaultMode}

    assert channels == expected_channels
    assert modes == expected_modes


# =====================================================================
# Test F: Severity Coverage for Physical Faults
# =====================================================================
def test_f_severity_coverage(fast_generated_dataset):
    """Physical faults cover severities 0.3, 0.5, 0.7."""
    phys_types = [
        "cooling_degradation",
        "lubrication_degradation",
        "fuel_injection_abnormality",
        "mechanical_degradation",
    ]
    for ft in phys_types:
        df_ft = fast_generated_dataset[fast_generated_dataset["fault_type"] == ft]
        sevs = set(np.round(df_ft["generation_severity"].unique(), 2))
        assert {0.3, 0.5, 0.7}.issubset(sevs)


# =====================================================================
# Test G: Grouped Split Integrity
# =====================================================================
def test_g_grouped_split_integrity(fast_generated_dataset):
    """No mission_run_id appears in more than one split."""
    train_df, val_df, test_df = split_by_mission_run(fast_generated_dataset, random_state=42)

    train_runs = set(train_df["mission_run_id"].unique())
    val_runs = set(val_df["mission_run_id"].unique())
    test_runs = set(test_df["mission_run_id"].unique())

    assert len(train_runs & val_runs) == 0
    assert len(train_runs & test_runs) == 0
    assert len(val_runs & test_runs) == 0


# =====================================================================
# Test H: Class Representation Per Split
# =====================================================================
def test_h_class_representation_per_split(fast_generated_dataset):
    """Every class has at least 1 run in train, val, and test splits."""
    train_df, val_df, test_df = split_by_mission_run(fast_generated_dataset, random_state=42)

    for cls in CANONICAL_FAULT_LABELS:
        assert (train_df["fault_type"] == cls).sum() > 0
        assert (val_df["fault_type"] == cls).sum() > 0
        assert (test_df["fault_type"] == cls).sum() > 0


# =====================================================================
# Test I: No Train-Test Mission Overlap
# =====================================================================
def test_i_no_train_test_mission_overlap(fast_generated_dataset):
    """Total runs partition exactly into train, val, test with zero overlap."""
    train_df, val_df, test_df = split_by_mission_run(fast_generated_dataset, random_state=42)
    all_runs = set(fast_generated_dataset["mission_run_id"].unique())
    union_runs = set(train_df["mission_run_id"]) | set(val_df["mission_run_id"]) | set(test_df["mission_run_id"])
    assert union_runs == all_runs
    assert len(all_runs) == len(train_df["mission_run_id"].unique()) + len(val_df["mission_run_id"].unique()) + len(test_df["mission_run_id"].unique())


# =====================================================================
# Test J: Feature Leakage Prevention
# =====================================================================
def test_j_feature_leakage_prevention(fast_generated_dataset):
    """Target and generation metadata columns are strictly excluded from features."""
    fe = FeatureExtractor()
    fe.fit(fast_generated_dataset)
    X = fe.transform(fast_generated_dataset)

    forbidden_cols = [
        "fault_type",
        "fault_severity",
        "generation_severity",
        "generation_sensor_channel",
        "generation_sensor_mode",
        "anomaly_status",
        "anomaly_score",
        "parameters",
    ]
    for col in forbidden_cols:
        assert col not in X.columns


# =====================================================================
# Test K: Training-Only Feature Fitting
# =====================================================================
def test_k_training_only_feature_fitting(fast_generated_dataset):
    """FeatureExtractor.fit uses only training data, transform reproduces schema."""
    train_df, _, test_df = split_by_mission_run(fast_generated_dataset, random_state=42)

    fe = FeatureExtractor()
    with pytest.raises(RuntimeError):
        # Transform before fit must fail
        fe.transform(test_df)

    fe.fit(train_df)
    X_test = fe.transform(test_df)
    assert list(X_test.columns) == fe.feature_names_


# =====================================================================
# Test L: One-Hot Mission Phase & Unseen Phase Handling
# =====================================================================
def test_l_onehot_mission_phase(fast_generated_dataset):
    """mission_phase is encoded as binary columns; unseen phase yields all zeros."""
    fe = FeatureExtractor()
    fe.fit(fast_generated_dataset)

    # Regular transform
    X = fe.transform(fast_generated_dataset)
    phase_cols = [c for c in X.columns if c.startswith("phase_")]
    assert len(phase_cols) == 6
    # Binary check
    for pc in phase_cols:
        assert set(X[pc].unique()).issubset({0, 1})

    # Test unseen phase
    test_sample = fast_generated_dataset.iloc[:2].copy()
    test_sample["mission_phase"] = "HYPERSONIC_CRUISE"  # Completely unseen
    X_unseen = fe.transform(test_sample)

    assert X_unseen.shape[1] == EXPECTED_FEATURE_COUNT
    for pc in phase_cols:
        assert (X_unseen[pc] == 0).all()


# =====================================================================
# Test M: Feature Count Exactly 16
# =====================================================================
def test_m_feature_count(fast_generated_dataset):
    """Feature matrix has exactly 16 columns (7 residuals + 3 context + 6 phases)."""
    fe = FeatureExtractor()
    fe.fit(fast_generated_dataset)
    X = fe.transform(fast_generated_dataset)
    assert X.shape[1] == 16
    assert X.shape[1] == EXPECTED_FEATURE_COUNT


# =====================================================================
# Test N: NaN Preservation (No Zero-Imputation)
# =====================================================================
def test_n_nan_preservation(fast_generated_dataset):
    """NaNs in residual columns are preserved and not replaced by 0.0."""
    fe = FeatureExtractor()
    fe.fit(fast_generated_dataset)

    sample = fast_generated_dataset.iloc[:5].copy()
    sample.loc[0, "rpm_norm_residual"] = np.nan
    sample.loc[1, "cht_norm_residual"] = np.nan

    X = fe.transform(sample)
    assert np.isnan(X.loc[0, "rpm_norm_residual"])
    assert np.isnan(X.loc[1, "cht_norm_residual"])


# =====================================================================
# Test O: Insufficient Data Handling
# =====================================================================
def test_o_insufficient_data_handling():
    """Samples with <4 valid residuals produce INSUFFICIENT_DATA and zero confidence."""
    fe = FeatureExtractor()
    clf = XGBoostFaultClassifier()
    # Dummy fit for fe
    dummy_df = pd.DataFrame([{
        "rpm_norm_residual": 0.1, "cht_norm_residual": 0.1, "egt_norm_residual": 0.1,
        "oil_temp_norm_residual": 0.1, "oil_pressure_norm_residual": 0.1,
        "fuel_flow_norm_residual": 0.1, "vibration_norm_residual": 0.1,
        "throttle": 75.0, "altitude": 1000.0, "load": 80.0, "mission_phase": "CRUISE"
    }])
    fe.fit(dummy_df)
    pipeline = FaultDiagnosisPipeline(classifier=clf, feature_extractor=fe)

    # Sample with only 2 valid residuals (<4)
    insufficient_sample = {
        "timestamp": 12.0,
        "engine_id": "ENG_001",
        "mission_id": "TEST_RUN",
        "mission_phase": "CRUISE",
        "rpm_norm_residual": 1.0,
        "cht_norm_residual": 2.0,
        "egt_norm_residual": np.nan,
        "oil_temp_norm_residual": np.nan,
        "oil_pressure_norm_residual": np.nan,
        "fuel_flow_norm_residual": np.nan,
        "vibration_norm_residual": np.nan,
        "throttle": 75.0,
        "altitude": 1000.0,
        "load": 80.0,
    }

    result = pipeline.diagnose_sample(insufficient_sample)
    assert result.data_quality == DiagnosisDataQuality.INSUFFICIENT_DATA.value
    assert result.diagnostic_confidence == 0.0


# =====================================================================
# Test P: XGBoost Train and Predict
# =====================================================================
def test_p_xgboost_train_predict():
    """Classifier trains and predicts canonical labels on synthetic test features."""
    rng = np.random.RandomState(42)
    n = 60
    # Create synthetic features for 6 classes
    feature_cols = [
        "rpm_norm_residual", "cht_norm_residual", "egt_norm_residual",
        "oil_temp_norm_residual", "oil_pressure_norm_residual", "fuel_flow_norm_residual",
        "vibration_norm_residual", "throttle", "altitude", "load",
        "phase_TAKEOFF", "phase_CLIMB", "phase_CRUISE", "phase_LOITER", "phase_DESCENT", "phase_LANDING",
    ]
    X = pd.DataFrame(rng.randn(n, 16), columns=feature_cols)
    y = pd.Series([CANONICAL_FAULT_LABELS[i % 6] for i in range(n)])

    clf = XGBoostFaultClassifier(FaultClassifierConfig(n_estimators=10, max_depth=3))
    clf.fit(X, y)
    assert clf.is_trained

    preds = clf.predict(X)
    assert len(preds) == n
    for p in preds:
        assert p in CANONICAL_FAULT_LABELS


# =====================================================================
# Test Q: Predict Proba Sums to 1.0
# =====================================================================
def test_q_predict_proba_sums_to_one():
    """predict_proba outputs dictionaries whose probabilities sum to 1.0."""
    rng = np.random.RandomState(42)
    n = 20
    feature_cols = [
        "rpm_norm_residual", "cht_norm_residual", "egt_norm_residual",
        "oil_temp_norm_residual", "oil_pressure_norm_residual", "fuel_flow_norm_residual",
        "vibration_norm_residual", "throttle", "altitude", "load",
        "phase_TAKEOFF", "phase_CLIMB", "phase_CRUISE", "phase_LOITER", "phase_DESCENT", "phase_LANDING",
    ]
    X = pd.DataFrame(rng.randn(n, 16), columns=feature_cols)
    y = pd.Series([CANONICAL_FAULT_LABELS[i % 6] for i in range(n)])

    clf = XGBoostFaultClassifier(FaultClassifierConfig(n_estimators=10, max_depth=3))
    clf.fit(X, y)

    prob_dicts = clf.predict_proba(X, return_dict=True)
    assert len(prob_dicts) == n
    for pd_item in prob_dicts:
        assert len(pd_item) == 6
        prob_sum = sum(pd_item.values())
        assert abs(prob_sum - 1.0) < 1e-4


# =====================================================================
# Test R: Deterministic Training
# =====================================================================
def test_r_deterministic_training():
    """Two identical training sessions with seed=42 yield identical predictions."""
    rng = np.random.RandomState(42)
    feature_cols = [
        "rpm_norm_residual", "cht_norm_residual", "egt_norm_residual",
        "oil_temp_norm_residual", "oil_pressure_norm_residual", "fuel_flow_norm_residual",
        "vibration_norm_residual", "throttle", "altitude", "load",
        "phase_TAKEOFF", "phase_CLIMB", "phase_CRUISE", "phase_LOITER", "phase_DESCENT", "phase_LANDING",
    ]
    X = pd.DataFrame(rng.randn(30, 16), columns=feature_cols)
    y = pd.Series([CANONICAL_FAULT_LABELS[i % 6] for i in range(30)])

    clf1 = XGBoostFaultClassifier(FaultClassifierConfig(n_estimators=15, random_state=42))
    clf1.fit(X, y)
    preds1 = clf1.predict(X)

    clf2 = XGBoostFaultClassifier(FaultClassifierConfig(n_estimators=15, random_state=42))
    clf2.fit(X, y)
    preds2 = clf2.predict(X)

    assert np.array_equal(preds1, preds2)


# =====================================================================
# Test S: Model Save and Load
# =====================================================================
def test_s_model_save_load():
    """Saved and reloaded model produces identical predictions and metadata."""
    rng = np.random.RandomState(42)
    feature_cols = [
        "rpm_norm_residual", "cht_norm_residual", "egt_norm_residual",
        "oil_temp_norm_residual", "oil_pressure_norm_residual", "fuel_flow_norm_residual",
        "vibration_norm_residual", "throttle", "altitude", "load",
        "phase_TAKEOFF", "phase_CLIMB", "phase_CRUISE", "phase_LOITER", "phase_DESCENT", "phase_LANDING",
    ]
    X = pd.DataFrame(rng.randn(30, 16), columns=feature_cols)
    y = pd.Series([CANONICAL_FAULT_LABELS[i % 6] for i in range(30)])

    clf = XGBoostFaultClassifier(FaultClassifierConfig(n_estimators=10, random_state=42))
    clf.fit(X, y)
    orig_preds = clf.predict(X)

    with tempfile.TemporaryDirectory() as tmpdir:
        clf.save_model(tmpdir)
        loaded_clf = XGBoostFaultClassifier()
        loaded_clf.load_model(tmpdir)

        loaded_preds = loaded_clf.predict(X)
        assert np.array_equal(orig_preds, loaded_preds)
        assert loaded_clf.is_trained
        assert loaded_clf.feature_names == clf.feature_names


# =====================================================================
# Tests T-Y: Physics Fault-Specific Diagnoses (T, U, V, W, X, Y)
# =====================================================================
@pytest.fixture(scope="module")
def trained_pipeline_and_data():
    """Train a pipeline on multi-step profile data with physical fault signatures."""
    prof = MissionProfile(
        mission_id="DIAGNOSIS_PROFILE",
        segments=[
            PhaseSegment(
                phase=FlightPhase.CRUISE,
                duration_s=60.0,
                throttle_start_pct=75.0,
                throttle_end_pct=75.0,
                altitude_start_m=1000.0,
                altitude_end_m=1000.0,
            )
        ],
    )
    cfg = DatasetConfig(
        dt=1.0,
        fault_onset_time=0.0,
        mission_profile=prof,
        healthy_seeds=list(HEALTHY_SEEDS),
        physical_severities=[0.3, 0.5, 0.7],
        physical_seed_pairs={0.3: [42, 100], 0.5: [200, 300], 0.7: [400, 500]},
        sensor_severities=[0.5, 0.7],
        binary_sensor_seeds=[42],
    )
    dataset = generate_fault_diagnosis_dataset(cfg)
    train_df, _, test_df = split_by_mission_run(dataset, random_state=42)

    fe = FeatureExtractor()
    fe.fit(train_df)
    X_train = fe.transform(train_df)
    y_train = train_df["fault_type"]

    weights = compute_class_weights(y_train)
    clf = XGBoostFaultClassifier(FaultClassifierConfig(n_estimators=50, max_depth=5, random_state=42))
    clf.fit(X_train, y_train, sample_weight=weights)

    pipeline = FaultDiagnosisPipeline(classifier=clf, feature_extractor=fe, enable_phase7_gating=False)
    return pipeline, train_df, test_df


def test_t_none_diagnosis(trained_pipeline_and_data):
    """Healthy data with near-zero residuals diagnosed as none."""
    pipeline, _, test_df = trained_pipeline_and_data
    none_data = test_df[test_df["fault_type"] == "none"]
    if len(none_data) > 0:
        results = pipeline.diagnose_batch(none_data)
        preds = [r.predicted_fault_type for r in results]
        accuracy = sum(1 for p in preds if p == "none") / len(preds)
        assert accuracy >= 0.70


def test_u_cooling_diagnosis(trained_pipeline_and_data):
    """Cooling degradation (elevated CHT) diagnosed correctly in active fault window."""
    pipeline, _, test_df = trained_pipeline_and_data
    cooling_data = test_df[(test_df["fault_type"] == "cooling_degradation") & (test_df["timestamp"] > 25.0)]
    if len(cooling_data) > 0:
        results = pipeline.diagnose_batch(cooling_data)
        preds = [r.predicted_fault_type for r in results]
        cooling_matches = sum(1 for p in preds if p == "cooling_degradation")
        assert cooling_matches > 0


def test_v_lubrication_diagnosis(trained_pipeline_and_data):
    """Lubrication degradation diagnosed correctly in active fault window."""
    pipeline, _, test_df = trained_pipeline_and_data
    lub_data = test_df[(test_df["fault_type"] == "lubrication_degradation") & (test_df["timestamp"] > 25.0)]
    if len(lub_data) > 0:
        results = pipeline.diagnose_batch(lub_data)
        preds = [r.predicted_fault_type for r in results]
        lub_matches = sum(1 for p in preds if p == "lubrication_degradation")
        assert lub_matches > 0


def test_w_fuel_diagnosis(trained_pipeline_and_data):
    """Fuel injection abnormality diagnosed correctly in active fault window."""
    pipeline, _, test_df = trained_pipeline_and_data
    fuel_data = test_df[(test_df["fault_type"] == "fuel_injection_abnormality") & (test_df["timestamp"] > 25.0)]
    if len(fuel_data) > 0:
        results = pipeline.diagnose_batch(fuel_data)
        preds = [r.predicted_fault_type for r in results]
        fuel_matches = sum(1 for p in preds if p == "fuel_injection_abnormality")
        assert fuel_matches > 0


def test_x_mechanical_diagnosis(trained_pipeline_and_data):
    """Mechanical degradation (high vibration) diagnosed correctly in active fault window."""
    pipeline, _, test_df = trained_pipeline_and_data
    mech_data = test_df[(test_df["fault_type"] == "mechanical_degradation") & (test_df["timestamp"] > 25.0)]
    if len(mech_data) > 0:
        results = pipeline.diagnose_batch(mech_data)
        preds = [r.predicted_fault_type for r in results]
        mech_matches = sum(1 for p in preds if p == "mechanical_degradation")
        assert mech_matches > 0


def test_y_sensor_diagnosis(trained_pipeline_and_data):
    """Sensor fault recognized (recall explicitly checked)."""
    pipeline, _, test_df = trained_pipeline_and_data
    sensor_data = test_df[(test_df["fault_type"] == "sensor_fault") & (test_df["timestamp"] > 25.0)]
    if len(sensor_data) > 0:
        results = pipeline.diagnose_batch(sensor_data)
        preds = [r.predicted_fault_type for r in results]
        sensor_matches = sum(1 for p in preds if p == "sensor_fault")
        recall = sensor_matches / len(preds)
        assert recall >= 0.50


# =====================================================================
# Test Z: Evaluation Metrics
# =====================================================================
def test_z_evaluation_metrics(trained_pipeline_and_data):
    """evaluate_classifier computes Macro F1, confusion matrix, and per-class metrics."""
    pipeline, _, test_df = trained_pipeline_and_data
    fe = pipeline.feature_extractor
    clf = pipeline.classifier

    X_test = fe.transform(test_df)
    y_test = test_df["fault_type"]

    eval_out = evaluate_classifier(clf, X_test, y_test)
    assert "macro_f1" in eval_out
    assert "confusion_matrix" in eval_out
    assert "per_class" in eval_out
    assert len(eval_out["confusion_matrix"]) == 6
    assert len(eval_out["per_class"]) == 6
    assert 0.0 <= eval_out["macro_f1"] <= 1.0


# =====================================================================
# Test AA: NONE False Positive Rate
# =====================================================================
def test_aa_none_false_positive_rate(trained_pipeline_and_data):
    """NONE False Positive Rate is computed and finite between 0 and 1."""
    pipeline, _, test_df = trained_pipeline_and_data
    fe = pipeline.feature_extractor
    clf = pipeline.classifier

    X_test = fe.transform(test_df)
    y_test = test_df["fault_type"]

    eval_out = evaluate_classifier(clf, X_test, y_test)
    fpr = eval_out["none_false_positive_rate"]
    far = eval_out["none_false_alarm_rate"]

    assert isinstance(fpr, float)
    assert isinstance(far, float)
    assert 0.0 <= fpr <= 1.0
    assert 0.0 <= far <= 1.0


# =====================================================================
# Test AB: Provenance Fields Preserved
# =====================================================================
def test_ab_provenance_fields():
    """FaultDiagnosisResult preserves timestamp, engine_id, mission_id, mission_phase."""
    fe = FeatureExtractor()
    clf = XGBoostFaultClassifier()
    pipeline = FaultDiagnosisPipeline(classifier=clf, feature_extractor=fe)

    sample = {
        "timestamp": 123.45,
        "engine_id": "ENG_UAV_09",
        "mission_id": "FLIGHT_ALPHA",
        "mission_phase": "LOITER",
        "rpm_norm_residual": 0.0,
        "cht_norm_residual": 0.0,
        "egt_norm_residual": 0.0,
        "oil_temp_norm_residual": 0.0,
        "oil_pressure_norm_residual": 0.0,
        "fuel_flow_norm_residual": 0.0,
        "vibration_norm_residual": 0.0,
        "throttle": 70.0,
        "altitude": 1500.0,
        "load": 75.0,
    }
    # Fit fe dummy
    fe.fit(pd.DataFrame([sample]))
    res = pipeline.diagnose_sample(sample)

    assert res.timestamp == 123.45
    assert res.engine_id == "ENG_UAV_09"
    assert res.mission_id == "FLIGHT_ALPHA"
    assert res.mission_phase == "LOITER"


# =====================================================================
# Test AC: Phase 7 Independence
# =====================================================================
def test_ac_phase7_independence():
    """Classifier operates independently without Phase 7 inputs or imports."""
    fe = FeatureExtractor()
    sample = {
        "rpm_norm_residual": 0.2, "cht_norm_residual": 0.1, "egt_norm_residual": -0.1,
        "oil_temp_norm_residual": 0.0, "oil_pressure_norm_residual": -0.2,
        "fuel_flow_norm_residual": 0.0, "vibration_norm_residual": 0.1,
        "throttle": 75.0, "altitude": 1000.0, "load": 80.0, "mission_phase": "CRUISE"
    }
    df = pd.DataFrame([sample] * 12)
    fe.fit(df)
    X = fe.transform(df)
    y = pd.Series([CANONICAL_FAULT_LABELS[i % 6] for i in range(12)])

    clf = XGBoostFaultClassifier(FaultClassifierConfig(n_estimators=5, max_depth=2))
    clf.fit(X, y)

    # Predict without Phase 7
    preds = clf.predict(X)
    probs = clf.predict_proba(X)
    assert len(preds) == 12
    assert len(probs) == 12


# =====================================================================
# Test AD: Class Probabilities Labels
# =====================================================================
def test_ad_class_probabilities_labels():
    """class_probabilities dictionary contains exactly the 6 canonical labels."""
    fe = FeatureExtractor()
    sample = {
        "rpm_norm_residual": 0.2, "cht_norm_residual": 0.1, "egt_norm_residual": -0.1,
        "oil_temp_norm_residual": 0.0, "oil_pressure_norm_residual": -0.2,
        "fuel_flow_norm_residual": 0.0, "vibration_norm_residual": 0.1,
        "throttle": 75.0, "altitude": 1000.0, "load": 80.0, "mission_phase": "CRUISE"
    }
    df = pd.DataFrame([sample] * 12)
    fe.fit(df)
    X = fe.transform(df)
    y = pd.Series([CANONICAL_FAULT_LABELS[i % 6] for i in range(12)])

    clf = XGBoostFaultClassifier(FaultClassifierConfig(n_estimators=5, max_depth=2))
    clf.fit(X, y)

    probs = clf.predict_proba(X, return_dict=True)
    for p_dict in probs:
        assert sorted(p_dict.keys()) == CANONICAL_FAULT_LABELS


# =====================================================================
# Test AE: Existing Tests Regression Safety
# =====================================================================
def test_ae_existing_tests_intact():
    """Verifies that Phase 8 imports do not interfere with Phase 1-7 schemas."""
    from simulator.fault_interface import FaultType
    from digital_twin.twin_model import DigitalTwin
    from anomaly_detection import HybridAnomalyDetector

    assert len(FaultType) == 6
    twin = DigitalTwin()
    assert twin is not None
    detector = HybridAnomalyDetector()
    assert detector is not None
