"""
Comprehensive Unit and Validation Tests for Phase 7: Hybrid Anomaly Detection.

Covers:
- Test A: Healthy baseline — predominantly NORMAL, bounded false alarms
- Test B: Cooling degradation — CHT normalized residual triggers anomaly evidence
- Test C: Lubrication degradation — oil pressure normalized residual triggers anomaly evidence
- Test D: Fuel abnormality — EGT / fuel flow normalized residuals trigger anomaly evidence
- Test E: Mechanical degradation — vibration normalized residual triggers anomaly evidence
- Test F: Sensor fault — sensor observation deviation flagged without changing engine physics
- Test G: Dropout / NaN — NaN remains NaN, no zero imputation, no crash
- Test H: Missing normalized feature — raw physical channel is NOT substituted, min_valid rule applied
- Test I: Insufficient valid features — produces INSUFFICIENT_DATA, score is NaN, never NORMAL
- Test J: Isolated transient — single spike does not produce sustained ANOMALY due to persistence gate
- Test K: Sustained anomaly — consecutive abnormal samples produce ANOMALY
- Test L: EWMA temporal detector — verify noise smoothing and sustained shift detection
- Test M: Persistence gate — counter tracking, reset on return to nominal, ignores missing/insufficient data
- Test N: Isolation Forest — healthy-only fitting, determinism with fixed random_state, [0, 1] scoring
- Test O: No leakage — fault metadata/type/severity never enter detector features
- Test P: Provenance preservation — flight/engine identifiers survive into AnomalyFrame
- Test Q: Determinism — identical inputs produce identical output statuses and scores
"""

import math
import numpy as np
import pandas as pd
import pytest

from telemetry.schema import TelemetryRecord, DigitalTwinState
from telemetry.ingestion import CanonicalTelemetryFrame
from digital_twin.twin_model import DigitalTwin
from digital_twin.residuals import ResidualGenerator, ResidualFrame, SUPPORTED_RESIDUAL_CHANNELS
from simulator.engine_simulator import EngineSimulator
from simulator.subsystems.mission import MissionProfile, PhaseSegment, FlightPhase
from simulator.fault_interface import (
    FaultState,
    FaultType,
    FaultSubsystem,
    FuelMixtureMode,
)

from anomaly_detection.schema import (
    AnomalyStatus,
    AnomalyRecord,
    AnomalyFrame,
    PRIMARY_NORMALIZED_RESIDUAL_CHANNELS,
)
from anomaly_detection.preprocessing import ResidualPreprocessor
from anomaly_detection.detectors import (
    ResidualThresholdDetector,
    EWMADetector,
    PersistenceGate,
    IsolationForestDetector,
)
from anomaly_detection.fusion import EvidenceFusionEngine
from anomaly_detection.pipeline import HybridAnomalyDetector


# ==============================================================================
# TEST FIXTURES & HELPERS
# ==============================================================================

def _generate_telemetry(
    duration_s: float = 30.0,
    throttle: float = 75.0,
    altitude: float = 2000.0,
    fault_schedule=None,
    seed: int = 42,
    dt: float = 0.5,
) -> CanonicalTelemetryFrame:
    """Helper to run simulator and produce CanonicalTelemetryFrame."""
    sim = EngineSimulator(seed=seed)
    seg = PhaseSegment(
        phase=FlightPhase.CRUISE,
        duration_s=duration_s,
        throttle_start_pct=throttle,
        throttle_end_pct=throttle,
        altitude_start_m=altitude,
        altitude_end_m=altitude,
    )
    profile = MissionProfile(segments=[seg])
    records = sim.run(profile, dt=dt, fault_schedule=fault_schedule)
    df = pd.DataFrame([r.to_dict() for r in records])
    return CanonicalTelemetryFrame(df)


def _generate_residuals(
    duration_s: float = 40.0,
    throttle: float = 75.0,
    altitude: float = 2000.0,
    fault_schedule=None,
    seed: int = 42,
    dt: float = 0.5,
    skip_transient: bool = True,
) -> ResidualFrame:
    """Helper to generate residuals through Phase 6 Digital Twin + ResidualGenerator."""
    telemetry = _generate_telemetry(
        duration_s=duration_s,
        throttle=throttle,
        altitude=altitude,
        fault_schedule=fault_schedule,
        seed=seed,
        dt=dt,
    )
    twin = DigitalTwin()
    res_frame = twin.process_frame(telemetry)
    if skip_transient:
        df = res_frame.to_dataframe()
        steady_df = df[df["timestamp"] >= 5.0].reset_index(drop=True)
        return ResidualFrame(steady_df, metadata=res_frame.metadata)
    return res_frame


@pytest.fixture(scope="module")
def healthy_baseline_residuals() -> ResidualFrame:
    """Provides a calibrated healthy baseline ResidualFrame for detector training."""
    return _generate_residuals(duration_s=45.0, seed=100)


@pytest.fixture(scope="module")
def fitted_detector(healthy_baseline_residuals) -> HybridAnomalyDetector:
    """Provides a pre-fitted HybridAnomalyDetector."""
    detector = HybridAnomalyDetector(
        min_valid_features=4,
        threshold_warning=1.5,
        threshold_anomaly=3.0,
        threshold_critical=4.5,
        ewma_alpha=0.2,
        persistence_min_consecutive=3,
        iforest_random_state=42,
    )
    detector.fit(healthy_baseline_residuals)
    return detector


# ==============================================================================
# A. HEALTHY BASELINE
# ==============================================================================

def test_a_healthy_baseline_predominantly_normal(fitted_detector):
    """Test A: Healthy simulator residuals produce predominantly NORMAL with low false alarms."""
    healthy_test = _generate_residuals(duration_s=45.0, seed=200)
    anom_frame = fitted_detector.detect(healthy_test)

    summary = anom_frame.summary()
    assert summary["total_samples"] == len(healthy_test)
    assert summary["insufficient_data_count"] == 0

    # Healthy engine should be overwhelmingly NORMAL (>= 95%) and 0 false ANOMALY
    normal_pct = summary["normal_count"] / summary["total_samples"]
    assert normal_pct >= 0.95, f"Expected >= 95% NORMAL on healthy data, got {normal_pct*100:.1f}%"
    assert summary["anomaly_count"] == 0, f"Expected 0 false ANOMALY on healthy baseline, got {summary['anomaly_count']}"


# ==============================================================================
# B. COOLING DEGRADATION
# ==============================================================================

def test_b_cooling_degradation_detected(fitted_detector):
    """Test B: Cooling degradation generates anomaly evidence on CHT residual."""
    fault_state = FaultState(
        fault_type=FaultType.COOLING_DEGRADATION,
        severity=0.85,
        start_time=10.0,
        end_time=35.0,
        affected_subsystem=FaultSubsystem.THERMAL,
    )
    rf = _generate_residuals(duration_s=45.0, fault_schedule=fault_state, seed=300)
    anom_frame = fitted_detector.detect(rf)

    df_anom = anom_frame.anomalies()
    assert len(df_anom) > 0, "Expected ANOMALY detections during cooling fault."

    # Post-fault samples should have cht_norm_residual in contributing channels
    post_fault = anom_frame.to_dataframe()[anom_frame.to_dataframe()["timestamp"] >= 15.0]
    cht_contributing = any("cht_norm_residual" in chs for chs in post_fault["contributing_channels"])
    assert cht_contributing, "cht_norm_residual must be identified as a contributing channel."


# ==============================================================================
# C. LUBRICATION DEGRADATION
# ==============================================================================

def test_c_lubrication_degradation_detected(fitted_detector):
    """Test C: Lubrication degradation generates anomaly evidence on oil pressure residual."""
    fault_state = FaultState(
        fault_type=FaultType.LUBRICATION_DEGRADATION,
        severity=0.80,
        start_time=10.0,
        end_time=35.0,
        affected_subsystem=FaultSubsystem.LUBRICATION,
    )
    rf = _generate_residuals(duration_s=45.0, fault_schedule=fault_state, seed=400)
    anom_frame = fitted_detector.detect(rf)

    df_anom = anom_frame.anomalies()
    assert len(df_anom) > 0, "Expected ANOMALY detections during lubrication fault."

    post_fault = anom_frame.to_dataframe()[anom_frame.to_dataframe()["timestamp"] >= 15.0]
    oil_contributing = any("oil_pressure_norm_residual" in chs for chs in post_fault["contributing_channels"])
    assert oil_contributing, "oil_pressure_norm_residual must be identified as a contributing channel."


# ==============================================================================
# D. FUEL ABNORMALITY
# ==============================================================================

def test_d_fuel_abnormality_detected(fitted_detector):
    """Test D: Fuel abnormality generates anomaly evidence on EGT and/or fuel flow."""
    fault_state = FaultState(
        fault_type=FaultType.FUEL_INJECTION_ABNORMALITY,
        severity=0.85,
        start_time=10.0,
        end_time=35.0,
        affected_subsystem=FaultSubsystem.FUEL,
        parameters={"mode": "lean"},
    )
    rf = _generate_residuals(duration_s=45.0, fault_schedule=fault_state, seed=500)
    anom_frame = fitted_detector.detect(rf)

    df_anom = anom_frame.anomalies()
    assert len(df_anom) > 0, "Expected ANOMALY detections during fuel fault."

    post_fault = anom_frame.to_dataframe()[anom_frame.to_dataframe()["timestamp"] >= 15.0]
    fuel_or_egt = any(
        ("fuel_flow_norm_residual" in chs or "egt_norm_residual" in chs)
        for chs in post_fault["contributing_channels"]
    )
    assert fuel_or_egt, "fuel_flow_norm_residual or egt_norm_residual must contribute anomaly evidence."


# ==============================================================================
# E. MECHANICAL DEGRADATION
# ==============================================================================

def test_e_mechanical_degradation_detected(fitted_detector):
    """Test E: Mechanical degradation generates anomaly evidence on vibration residual."""
    fault_state = FaultState(
        fault_type=FaultType.MECHANICAL_DEGRADATION,
        severity=0.85,
        start_time=10.0,
        end_time=35.0,
        affected_subsystem=FaultSubsystem.DYNAMICS,
    )
    rf = _generate_residuals(duration_s=45.0, fault_schedule=fault_state, seed=600)
    anom_frame = fitted_detector.detect(rf)

    df_anom = anom_frame.anomalies()
    assert len(df_anom) > 0, "Expected ANOMALY detections during mechanical fault."

    post_fault = anom_frame.to_dataframe()[anom_frame.to_dataframe()["timestamp"] >= 15.0]
    vib_contributing = any("vibration_norm_residual" in chs for chs in post_fault["contributing_channels"])
    assert vib_contributing, "vibration_norm_residual must be identified as a contributing channel."


# ==============================================================================
# F. SENSOR FAULT
# ==============================================================================

def test_f_sensor_fault_generates_anomaly_without_physics_change(fitted_detector):
    """Test F: Sensor fault generates anomaly evidence on that channel while physics is nominal."""
    fault_state = FaultState(
        fault_type=FaultType.SENSOR_FAULT,
        severity=1.0,
        start_time=10.0,
        end_time=35.0,
        affected_subsystem=FaultSubsystem.SENSOR,
        parameters={"sensor_channel": "cht", "sensor_mode": "bias", "bias_magnitude": 30.0},
    )
    rf = _generate_residuals(duration_s=45.0, fault_schedule=fault_state, seed=700)
    anom_frame = fitted_detector.detect(rf)

    df_anom = anom_frame.anomalies()
    assert len(df_anom) > 0, "Sensor bias must produce anomaly evidence."

    post_fault = anom_frame.to_dataframe()[anom_frame.to_dataframe()["timestamp"] >= 15.0]
    cht_found = any("cht_norm_residual" in chs for chs in post_fault["contributing_channels"])
    assert cht_found, "Sensor fault on CHT must report cht_norm_residual as contributing."


# ==============================================================================
# G. DROPOUT / NaN PRESERVATION
# ==============================================================================

def test_g_nan_preservation_and_no_zero_imputation(fitted_detector):
    """Test G: NaNs remain NaNs, never converted to 0, and detector does not crash."""
    rf = _generate_residuals(duration_s=20.0, seed=800)
    df = rf.to_dataframe()

    # Artificially inject NaNs into cht_norm_residual at selected rows
    df.loc[5:8, "cht_norm_residual"] = np.nan
    mod_rf = ResidualFrame(df)

    anom_frame = fitted_detector.detect(mod_rf)
    res_df = anom_frame.to_dataframe()

    # Check that missing feature is explicitly captured in record metadata
    assert "cht_norm_residual" in res_df.loc[5, "missing_features"]
    assert "cht_norm_residual" in res_df.loc[8, "missing_features"]
    assert "cht_norm_residual" not in res_df.loc[4, "missing_features"]


# ==============================================================================
# H. MISSING NORMALIZED FEATURE (NO RAW PHYSICAL SUBSTITUTION)
# ==============================================================================

def test_h_no_raw_physical_substitution():
    """Test H: Raw physical channel is NOT silently substituted for missing normalized residual."""
    preprocessor = ResidualPreprocessor(min_valid_features=4)

    # Provide raw physical channels but omit normalized residual
    df = pd.DataFrame({
        "timestamp": [0.0, 1.0],
        "cht": [120.0, 125.0],              # Raw physical channel
        "oil_pressure": [4.0, 4.1],          # Raw physical channel
        "rpm_norm_residual": [0.1, 0.2],
        "egt_norm_residual": [0.0, -0.1],
        "oil_temp_norm_residual": [0.2, 0.1],
        # cht_norm_residual is missing
        # oil_pressure_norm_residual is missing
        "fuel_flow_norm_residual": [0.0, 0.0],
        "vibration_norm_residual": [0.1, 0.1],
    })

    feat_df, valid_mask, missing_per_row = preprocessor.extract_features(df)

    # cht_norm_residual MUST be NaN, NOT 120.0!
    assert np.isnan(feat_df.loc[0, "cht_norm_residual"])
    assert np.isnan(feat_df.loc[1, "cht_norm_residual"])
    assert "cht_norm_residual" in missing_per_row[0]


# ==============================================================================
# I. INSUFFICIENT VALID FEATURES
# ==============================================================================

def test_i_insufficient_valid_features_yields_insufficient_data(fitted_detector):
    """Test I: Row with < min_valid_features produces INSUFFICIENT_DATA and NaN score, never NORMAL."""
    df = pd.DataFrame({
        "timestamp": [0.0, 1.0, 2.0],
        "engine_id": ["UAV_01"] * 3,
        "mission_id": ["TEST_MISS"] * 3,
        "mission_phase": ["CRUISE"] * 3,
        "source": ["sim"] * 3,
        "source_type": ["simulated"] * 3,
        # Only 2 valid features out of 7 (min required is 4)
        "rpm_norm_residual": [0.05, 0.02, 0.01],
        "cht_norm_residual": [0.10, 0.12, 0.09],
        "egt_norm_residual": [np.nan, np.nan, np.nan],
        "oil_temp_norm_residual": [np.nan, np.nan, np.nan],
        "oil_pressure_norm_residual": [np.nan, np.nan, np.nan],
        "fuel_flow_norm_residual": [np.nan, np.nan, np.nan],
        "vibration_norm_residual": [np.nan, np.nan, np.nan],
    })

    anom_frame = fitted_detector.detect(df)
    res_df = anom_frame.to_dataframe()

    for idx in range(3):
        assert res_df.loc[idx, "anomaly_status"] == AnomalyStatus.INSUFFICIENT_DATA.value
        assert np.isnan(res_df.loc[idx, "anomaly_score"])
        assert res_df.loc[idx, "anomaly_status"] != AnomalyStatus.NORMAL.value


# ==============================================================================
# J. ISOLATED TRANSIENT (PERSISTENCE REJECTION)
# ==============================================================================

def test_j_isolated_transient_does_not_produce_sustained_anomaly(fitted_detector):
    """Test J: Single isolated spike produces WARNING at most, not sustained ANOMALY."""
    # Create baseline with 0.0 residuals
    data = {ch: [0.0] * 10 for ch in PRIMARY_NORMALIZED_RESIDUAL_CHANNELS}
    data["timestamp"] = list(range(10))
    df = pd.DataFrame(data)

    # Inject single moderate anomaly-level spike at index 4 (3.5 sigma, below critical 4.5)
    df.loc[4, "cht_norm_residual"] = 3.5

    anom_frame = fitted_detector.detect(df)
    res_df = anom_frame.to_dataframe()

    # Row 4 has an isolated spike: must be WARNING, not ANOMALY
    assert res_df.loc[4, "anomaly_status"] == AnomalyStatus.WARNING.value
    # Rows 3 and 5 are normal
    assert res_df.loc[3, "anomaly_status"] == AnomalyStatus.NORMAL.value
    assert res_df.loc[5, "anomaly_status"] == AnomalyStatus.NORMAL.value


# ==============================================================================
# K. SUSTAINED ANOMALY (PERSISTENCE CONFIRMATION)
# ==============================================================================

def test_k_sustained_anomaly_produces_anomaly_status(fitted_detector):
    """Test K: Sustained abnormal samples confirm persistence and trigger ANOMALY."""
    data = {ch: [0.0] * 20 for ch in PRIMARY_NORMALIZED_RESIDUAL_CHANNELS}
    data["timestamp"] = list(range(20))
    df = pd.DataFrame(data)

    # Inject 4 consecutive abnormal samples at indices 3, 4, 5, 6 (min_consecutive=3)
    df.loc[3:6, "cht_norm_residual"] = 3.5

    anom_frame = fitted_detector.detect(df)
    res_df = anom_frame.to_dataframe()

    # Sample 3 (1st hit): WARNING
    assert res_df.loc[3, "anomaly_status"] == AnomalyStatus.WARNING.value
    assert res_df.loc[3, "persistence_count"] == 1

    # Sample 4 (2nd hit): WARNING
    assert res_df.loc[4, "anomaly_status"] == AnomalyStatus.WARNING.value
    assert res_df.loc[4, "persistence_count"] == 2

    # Sample 5 (3rd hit): Reaches min_consecutive=3 -> ANOMALY
    assert res_df.loc[5, "anomaly_status"] == AnomalyStatus.ANOMALY.value
    assert res_df.loc[5, "persistence_count"] == 3

    # Sample 6 (4th hit): Sustained -> ANOMALY
    assert res_df.loc[6, "anomaly_status"] == AnomalyStatus.ANOMALY.value
    assert res_df.loc[6, "persistence_count"] == 4

    # Sample 7 (returns to 0.0): Persistence counter immediately resets to 0
    assert res_df.loc[7, "persistence_count"] == 0

    # Sample 15 (after EWMA exponentially decays below warning threshold): status returns to NORMAL
    assert res_df.loc[15, "anomaly_status"] == AnomalyStatus.NORMAL.value
    assert res_df.loc[15, "persistence_count"] == 0


# ==============================================================================
# L. EWMA TEMPORAL SMOOTHING
# ==============================================================================

def test_l_ewma_temporal_smoothing():
    """Test L: EWMA smooths alternating spikes and detects sustained drift."""
    ewma = EWMADetector(alpha=0.2, warning_threshold=1.2, anomaly_threshold=2.2)

    # 1. Alternating noisy spike: +2.0, -2.0, +2.0
    res1 = ewma.update_sample({"cht_norm_residual": 2.0})
    # Initial state = 2.0
    assert abs(res1["smoothed_values"]["cht_norm_residual"] - 2.0) < 1e-3

    res2 = ewma.update_sample({"cht_norm_residual": -2.0})
    # Smoothed = 0.2 * (-2.0) + 0.8 * (2.0) = 1.2
    assert abs(res2["smoothed_values"]["cht_norm_residual"] - 1.2) < 1e-3

    # 2. Reset and verify sustained step
    ewma.reset()
    for _ in range(15):
        res = ewma.update_sample({"cht_norm_residual": 3.0})

    # After 15 samples of 3.0, smoothed value converges near 3.0
    assert res["smoothed_values"]["cht_norm_residual"] > 2.8
    assert res["is_anomaly"] is True


# ==============================================================================
# M. PERSISTENCE GATE DETECTOR UNIT TESTS
# ==============================================================================

def test_m_persistence_gate_counter_and_reset():
    """Test M: PersistenceGate correctly counts, gates, and resets; ignores invalid data."""
    gate = PersistenceGate(min_consecutive=3)

    assert gate.update(is_abnormal=True, is_valid=True)["persistence_count"] == 1
    assert gate.update(is_abnormal=True, is_valid=True)["persistence_count"] == 2
    res = gate.update(is_abnormal=True, is_valid=True)
    assert res["persistence_count"] == 3
    assert res["persistence_satisfied"] is True

    # When missing data occurs (is_valid=False), counter must reset and NOT increment
    res_inv = gate.update(is_abnormal=True, is_valid=False)
    assert res_inv["persistence_count"] == 0
    assert res_inv["persistence_satisfied"] is False

    # Return to normal
    res_norm = gate.update(is_abnormal=False, is_valid=True)
    assert res_norm["persistence_count"] == 0


# ==============================================================================
# N. ISOLATION FOREST DETECTOR
# ==============================================================================

def test_n_isolation_forest_healthy_fit_and_determinism(healthy_baseline_residuals):
    """Test N: Isolation Forest fits only on healthy baseline and produces deterministic scores in [0, 1]."""
    iso1 = IsolationForestDetector(n_estimators=50, random_state=42)
    iso2 = IsolationForestDetector(n_estimators=50, random_state=42)

    iso1.fit(healthy_baseline_residuals)
    iso2.fit(healthy_baseline_residuals)

    sample = {ch: 0.05 for ch in PRIMARY_NORMALIZED_RESIDUAL_CHANNELS}
    res1 = iso1.score_sample(sample, is_valid=True)
    res2 = iso2.score_sample(sample, is_valid=True)

    # Identical score
    assert res1["isolation_score"] == res2["isolation_score"]
    assert 0.0 <= res1["isolation_score"] <= 1.0

    # Highly anomalous sample
    anom_sample = {ch: 5.0 for ch in PRIMARY_NORMALIZED_RESIDUAL_CHANNELS}
    anom_res = iso1.score_sample(anom_sample, is_valid=True)
    assert anom_res["isolation_score"] > res1["isolation_score"]
    assert anom_res["is_anomaly"] is True


# ==============================================================================
# O. NO LEAKAGE OF FAULT METADATA
# ==============================================================================

def test_o_no_fault_leakage(healthy_baseline_residuals):
    """Test O: Fault metadata is never used in features; fitting with fault data raises ValueError."""
    iso = IsolationForestDetector(random_state=42)
    df_faulty = healthy_baseline_residuals.to_dataframe().copy()
    df_faulty.loc[0, "fault_type"] = "cooling_degradation"

    with pytest.raises(ValueError, match="Fault leakage detected"):
        iso.fit(df_faulty)


# ==============================================================================
# P. PROVENANCE PRESERVATION
# ==============================================================================

def test_p_provenance_preservation(fitted_detector, healthy_baseline_residuals):
    """Test P: Flight and engine identifiers survive into AnomalyFrame."""
    anom_frame = fitted_detector.detect(healthy_baseline_residuals)
    df = anom_frame.to_dataframe()

    for col in ["timestamp", "engine_id", "mission_id", "mission_phase", "source", "source_type"]:
        assert col in df.columns, f"Missing provenance column '{col}' in AnomalyFrame."
        assert df[col].notna().all(), f"Provenance column '{col}' contains unexpected NaNs."


# ==============================================================================
# Q. DETERMINISM
# ==============================================================================

def test_q_determinism(fitted_detector, healthy_baseline_residuals):
    """Test Q: Identical input + identical configuration yields deterministic output."""
    res1 = fitted_detector.detect(healthy_baseline_residuals, reset_state=True).to_dataframe()
    res2 = fitted_detector.detect(healthy_baseline_residuals, reset_state=True).to_dataframe()

    # Exact equality for string/discrete columns
    assert (res1["anomaly_status"] == res2["anomaly_status"]).all()
    assert (res1["persistence_count"] == res2["persistence_count"]).all()
    assert (res1["contributing_channels"] == res2["contributing_channels"]).all()

    # Float equality within numerical tolerance
    for col in ["anomaly_score", "threshold_score", "ewma_score", "persistence_score", "isolation_score"]:
        diff = np.abs(res1[col].values - res2[col].values)
        assert np.nanmax(diff) < 1e-6, f"Non-deterministic floating point score in column {col}"


# ==============================================================================
# S. MULTIVARIATE ANOMALY FUSION BEHAVIOR & CRITICAL OVERRIDE
# ==============================================================================

def test_s_multivariate_anomaly_fusion_behavior(fitted_detector):
    """
    Test S: Verify multivariate anomaly fusion dynamics:
    - Simultaneous sub-critical multi-channel elevation triggers WARNING on initial samples
    - Sustained multi-channel elevation reaches persistence (3 samples) and becomes ANOMALY
    - Single-channel catastrophic excursion (>= 4.5 sigma) triggers immediate ANOMALY override
    """
    # 1. Sub-critical multivariate elevation (4 channels at 2.0 sigma, below critical 4.5 sigma)
    n_samples = 6
    multi_data = {ch: [0.0] * n_samples for ch in PRIMARY_NORMALIZED_RESIDUAL_CHANNELS}
    multi_data["timestamp"] = [float(i) for i in range(n_samples)]

    # Elevate 4 channels simultaneously for all 6 samples
    elevated_channels = [
        "cht_norm_residual",
        "egt_norm_residual",
        "oil_temp_norm_residual",
        "oil_pressure_norm_residual",
    ]
    for ch in elevated_channels:
        multi_data[ch] = [2.0] * n_samples

    df_multi = pd.DataFrame(multi_data)
    anom_frame_multi = fitted_detector.detect(df_multi, reset_state=True)
    res_multi = anom_frame_multi.to_dataframe()

    # Sample 0 (1st hit, persistence count 1 < 3): MUST be WARNING (not suppressed, not yet sustained ANOMALY)
    assert res_multi.loc[0, "anomaly_status"] == AnomalyStatus.WARNING.value
    assert res_multi.loc[0, "persistence_count"] == 1
    assert 0.35 <= res_multi.loc[0, "anomaly_score"] <= 0.74

    # Sample 1 (2nd hit, persistence count 2 < 3): MUST be WARNING
    assert res_multi.loc[1, "anomaly_status"] == AnomalyStatus.WARNING.value
    assert res_multi.loc[1, "persistence_count"] == 2

    # Sample 2 (3rd hit, persistence count 3 >= 3): Confirms sustained ANOMALY
    assert res_multi.loc[2, "anomaly_status"] == AnomalyStatus.ANOMALY.value
    assert res_multi.loc[2, "persistence_count"] == 3
    assert res_multi.loc[2, "anomaly_score"] >= 0.75

    # Contributing channels must reflect the elevated channels
    contributing = res_multi.loc[2, "contributing_channels"]
    assert any(ch in contributing for ch in elevated_channels)

    # 2. Critical single-channel excursion (>= 4.5 sigma) MUST trigger immediate ANOMALY at sample 0
    crit_data = {ch: [0.0] * 3 for ch in PRIMARY_NORMALIZED_RESIDUAL_CHANNELS}
    crit_data["timestamp"] = [0.0, 1.0, 2.0]
    crit_data["vibration_norm_residual"] = [5.0, 0.0, 0.0]  # Catastrophic 5.0 sigma spike on sample 0

    df_crit = pd.DataFrame(crit_data)
    anom_frame_crit = fitted_detector.detect(df_crit, reset_state=True)
    res_crit = anom_frame_crit.to_dataframe()

    # Sample 0: Instantaneous critical override triggers ANOMALY immediately (bypasses persistence requirement)
    assert res_crit.loc[0, "anomaly_status"] == AnomalyStatus.ANOMALY.value
    assert res_crit.loc[0, "anomaly_score"] >= 0.90
    assert "vibration_norm_residual" in res_crit.loc[0, "contributing_channels"]
