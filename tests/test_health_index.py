"""
Behavioral and validation tests for Phase 9 Health Index + Degradation Tracking.
"""

import math
import pytest
import numpy as np
import pandas as pd

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

from simulator.fault_interface import FaultType, FaultState
from simulator.subsystems.mission import MissionProfile, PhaseSegment, FlightPhase
from fault_diagnosis.dataset import _run_mission


@pytest.fixture
def sample_cruise_profile():
    return MissionProfile(
        mission_id="HI_TEST_CRUISE",
        segments=[
            PhaseSegment(
                phase=FlightPhase.CRUISE,
                duration_s=50.0,
                throttle_start_pct=75.0,
                throttle_end_pct=75.0,
                altitude_start_m=1000.0,
                altitude_end_m=1000.0,
            )
        ],
    )


# =====================================================================
# Test 1: Weights Specification
# =====================================================================
def test_1_weights_specification():
    """Weights sum to 1.0, are non-negative, and custom weights work deterministically."""
    total_w = sum(DEFAULT_CHANNEL_WEIGHTS.values())
    assert abs(total_w - 1.0) < 1e-5
    for ch, w in DEFAULT_CHANNEL_WEIGHTS.items():
        assert w >= 0.0

    # Custom valid weights
    custom_w = {
        "oil_pressure": 0.30, "cht": 0.20, "egt": 0.10, "oil_temp": 0.20,
        "vibration": 0.10, "rpm": 0.05, "fuel_flow": 0.05,
    }
    cfg = HealthIndexConfig(channel_weights=custom_w)
    assert cfg.channel_weights == custom_w

    # Invalid weights: sum != 1.0
    with pytest.raises(ValueError):
        HealthIndexConfig(channel_weights={"oil_pressure": 0.5})

    # Invalid weights: negative
    with pytest.raises(ValueError):
        HealthIndexConfig(channel_weights={
            "oil_pressure": -0.1, "cht": 0.4, "egt": 0.2, "oil_temp": 0.2,
            "vibration": 0.1, "rpm": 0.1, "fuel_flow": 0.1,
        })


# =====================================================================
# Test 2: Healthy Baseline Flight Health
# =====================================================================
def test_2_healthy_baseline_health(sample_cruise_profile):
    """Healthy flight maintains high smoothed Health Index >= 0.85 and HEALTHY state."""
    df = _run_mission(seed=42, dt=1.0, mission_profile=sample_cruise_profile)
    pipeline = HealthIndexPipeline()
    results = pipeline.process_dataframe(df)

    assert len(results) == len(df)
    steady_state = results[15:]
    for r in steady_state:
        assert r.smoothed_health_index >= 0.85
        assert r.health_state == HealthState.HEALTHY.value
        assert r.data_quality == HealthDataQuality.VALID.value


# =====================================================================
# Tests 3-6: Severity Progression (Controlled Comparison)
# =====================================================================
def test_3_cooling_severity_progression(sample_cruise_profile):
    """Under controlled flight conditions, high cooling severity produces lower HI than low severity."""
    fs_low = FaultState(FaultType.COOLING_DEGRADATION, severity=0.3, start_time=0.0)
    fs_high = FaultState(FaultType.COOLING_DEGRADATION, severity=0.7, start_time=0.0)

    df_low = _run_mission(seed=42, dt=1.0, fault_state=fs_low, mission_profile=sample_cruise_profile)
    df_high = _run_mission(seed=42, dt=1.0, fault_state=fs_high, mission_profile=sample_cruise_profile)

    pipe_low = HealthIndexPipeline()
    pipe_high = HealthIndexPipeline()

    res_low = pipe_low.process_dataframe(df_low)
    res_high = pipe_high.process_dataframe(df_high)

    mean_hi_low = np.mean([r.smoothed_health_index for r in res_low[20:]])
    mean_hi_high = np.mean([r.smoothed_health_index for r in res_high[20:]])

    assert mean_hi_high < mean_hi_low
    assert "cht" in res_high[-1].dominant_degraded_channels


def test_4_lubrication_severity_progression(sample_cruise_profile):
    """Under controlled flight conditions, high lubrication severity produces lower HI than low severity."""
    fs_low = FaultState(FaultType.LUBRICATION_DEGRADATION, severity=0.3, start_time=0.0)
    fs_high = FaultState(FaultType.LUBRICATION_DEGRADATION, severity=0.7, start_time=0.0)

    df_low = _run_mission(seed=42, dt=1.0, fault_state=fs_low, mission_profile=sample_cruise_profile)
    df_high = _run_mission(seed=42, dt=1.0, fault_state=fs_high, mission_profile=sample_cruise_profile)

    res_low = HealthIndexPipeline().process_dataframe(df_low)
    res_high = HealthIndexPipeline().process_dataframe(df_high)

    mean_hi_low = np.mean([r.smoothed_health_index for r in res_low[20:]])
    mean_hi_high = np.mean([r.smoothed_health_index for r in res_high[20:]])

    assert mean_hi_high < mean_hi_low


def test_5_fuel_abnormality_progression(sample_cruise_profile):
    """Fuel injection abnormality degrades steady-state HI."""
    fs_fuel = FaultState(FaultType.FUEL_INJECTION_ABNORMALITY, severity=0.7, start_time=0.0, parameters={"mode": "lean"})
    df_healthy = _run_mission(seed=42, dt=1.0, mission_profile=sample_cruise_profile)
    df_fuel = _run_mission(seed=42, dt=1.0, fault_state=fs_fuel, mission_profile=sample_cruise_profile)

    res_healthy = HealthIndexPipeline().process_dataframe(df_healthy)
    res_fuel = HealthIndexPipeline().process_dataframe(df_fuel)

    mean_hi_h = np.mean([r.smoothed_health_index for r in res_healthy[20:]])
    mean_hi_f = np.mean([r.smoothed_health_index for r in res_fuel[20:]])

    assert mean_hi_f < mean_hi_h


def test_6_mechanical_severity_progression(sample_cruise_profile):
    """Mechanical degradation increases vibration and degrades HI."""
    fs_mech = FaultState(FaultType.MECHANICAL_DEGRADATION, severity=0.7, start_time=0.0)
    df_healthy = _run_mission(seed=42, dt=1.0, mission_profile=sample_cruise_profile)
    df_mech = _run_mission(seed=42, dt=1.0, fault_state=fs_mech, mission_profile=sample_cruise_profile)

    res_healthy = HealthIndexPipeline().process_dataframe(df_healthy)
    res_mech = HealthIndexPipeline().process_dataframe(df_mech)

    mean_hi_h = np.mean([r.smoothed_health_index for r in res_healthy[20:]])
    mean_hi_m = np.mean([r.smoothed_health_index for r in res_mech[20:]])

    assert mean_hi_m < mean_hi_h
    assert "vibration" in res_mech[-1].dominant_degraded_channels


# =====================================================================
# Test 7: Sensor Fault Isolation via Explicit Upstream Context
# =====================================================================
def test_7_sensor_fault_isolation_explicit_context():
    """Upstream sensor-fault diagnosis isolates corrupted channel, preventing false physical HI drop."""
    pipeline = HealthIndexPipeline()

    # Telemetry with extreme CHT residual but all other channels normal
    sample = {
        "timestamp": 15.0,
        "engine_id": "ENG_001",
        "rpm_norm_residual": 0.1,
        "cht_norm_residual": 8.5,  # Extreme sensor spike
        "egt_norm_residual": 0.0,
        "oil_temp_norm_residual": 0.1,
        "oil_pressure_norm_residual": -0.1,
        "fuel_flow_norm_residual": 0.0,
        "vibration_norm_residual": 0.1,
    }

    # Pass Phase 8 sensor_fault context
    context = {
        "predicted_fault_type": "sensor_fault",
        "diagnostic_confidence": 0.85,
        "sensor_channel": "cht",
    }

    res = pipeline.process_sample(sample, optional_context=context)

    assert "cht" in res.excluded_channels
    assert res.data_quality == HealthDataQuality.SENSOR_ISOLATED.value
    # Because CHT was isolated, physical HI remains high (>= 0.85)
    assert res.raw_health_index >= 0.85
    assert res.health_state == HealthState.HEALTHY.value


# =====================================================================
# Test 8: Sensor Fault Isolation via Deterministic Disconnect Heuristic
# =====================================================================
def test_8_sensor_fault_isolation_deterministic_rule():
    """Single channel >4.5 sigma with all others <=1.2 sigma continuously for >=5s triggers heuristic isolation."""
    pipeline = HealthIndexPipeline()

    records = []
    for t in range(10):
        records.append({
            "timestamp": float(t),
            "engine_id": "ENG_001",
            "rpm_norm_residual": 0.1,
            "cht_norm_residual": 5.5,  # Extreme single channel outlier
            "egt_norm_residual": 0.2,
            "oil_temp_norm_residual": 0.1,
            "oil_pressure_norm_residual": -0.1,
            "fuel_flow_norm_residual": 0.0,
            "vibration_norm_residual": 0.1,
        })
    df = pd.DataFrame(records)
    results = pipeline.process_dataframe(df)

    # Before 5 seconds (e.g. t=2s), CHT is not yet isolated
    assert "cht" not in results[2].excluded_channels

    # At t=6s (>= 5.0s of continuous persistence), CHT is isolated
    assert "cht" in results[6].excluded_channels
    assert results[6].data_quality == HealthDataQuality.SENSOR_ISOLATED.value
    assert results[6].raw_health_index >= 0.85


# =====================================================================
# Test 9: Correlated Physical Fault Not Isolated
# =====================================================================
def test_9_correlated_physical_fault_not_isolated():
    """Genuine physical fault with multiple elevated channels (CHT + Oil Temp) is NOT falsely isolated."""
    pipeline = HealthIndexPipeline()

    records = []
    for t in range(10):
        records.append({
            "timestamp": float(t),
            "engine_id": "ENG_001",
            "rpm_norm_residual": 0.2,
            "cht_norm_residual": 5.5,       # High
            "oil_temp_norm_residual": 3.0,  # Correlated thermal elevation (> 1.2 sigma)
            "egt_norm_residual": 1.5,
            "oil_pressure_norm_residual": -0.8,
            "fuel_flow_norm_residual": 0.1,
            "vibration_norm_residual": 0.2,
        })
    df = pd.DataFrame(records)
    results = pipeline.process_dataframe(df)

    # No channel should be isolated as a sensor fault
    for r in results:
        assert len(r.excluded_channels) == 0
        assert r.data_quality == HealthDataQuality.VALID.value
        assert r.raw_health_index < 0.85  # Properly registers as physically degraded


# =====================================================================
# Test 10: NaN Preservation
# =====================================================================
def test_10_nan_preservation():
    """NaN residuals are preserved, not converted to 0.0, and tracked in missing_channels."""
    pipeline = HealthIndexPipeline()
    sample = {
        "timestamp": 1.0,
        "engine_id": "ENG_001",
        "rpm_norm_residual": 0.1,
        "cht_norm_residual": np.nan,  # Missing
        "egt_norm_residual": 0.0,
        "oil_temp_norm_residual": 0.1,
        "oil_pressure_norm_residual": 0.0,
        "fuel_flow_norm_residual": 0.1,
        "vibration_norm_residual": np.nan,  # Missing
    }
    res = pipeline.process_sample(sample)

    assert "cht" in res.missing_channels
    assert "vibration" in res.missing_channels
    assert math.isnan(res.channel_degradation_evidence["cht"])
    assert math.isnan(res.channel_degradation_evidence["vibration"])
    assert res.data_quality == HealthDataQuality.VALID.value
    # Dynamic weights allocated over the 5 remaining valid channels
    assert len(res.valid_channels) == 5
    assert abs(sum(res.effective_channel_weights.values()) - 1.0) < 1e-4


# =====================================================================
# Test 11: Insufficient Data Handling (< 4 valid channels)
# =====================================================================
def test_11_insufficient_data_handling():
    """Fewer than 4 valid channels produces INSUFFICIENT_DATA and NaN health."""
    pipeline = HealthIndexPipeline()
    sample = {
        "timestamp": 1.0,
        "engine_id": "ENG_001",
        "rpm_norm_residual": 0.1,
        "cht_norm_residual": 0.2,
        "egt_norm_residual": 0.1,
        # Remaining 4 channels are missing (only 3 valid)
        "oil_temp_norm_residual": np.nan,
        "oil_pressure_norm_residual": np.nan,
        "fuel_flow_norm_residual": np.nan,
        "vibration_norm_residual": np.nan,
    }
    res = pipeline.process_sample(sample)

    assert res.health_state == HealthState.INSUFFICIENT_DATA.value
    assert res.data_quality == HealthDataQuality.INSUFFICIENT_DATA.value
    assert math.isnan(res.raw_health_index)
    assert math.isnan(res.smoothed_health_index)
    assert math.isnan(res.degradation_rate)
    assert res.degradation_trend == DegradationTrend.INSUFFICIENT_DATA.value


# =====================================================================
# Test 12: Causal Prefix Invariance
# =====================================================================
def test_12_causal_prefix_invariance():
    """Running on [0...T] produces the exact same health at T as running on [0...T...T_future]."""
    rng = np.random.RandomState(42)
    n_total = 40
    t_cutoff = 20

    records = []
    for t in range(n_total):
        records.append({
            "timestamp": float(t),
            "engine_id": "ENG_001",
            "rpm_norm_residual": rng.randn() * 0.5,
            "cht_norm_residual": rng.randn() * 0.5,
            "egt_norm_residual": rng.randn() * 0.5,
            "oil_temp_norm_residual": rng.randn() * 0.5,
            "oil_pressure_norm_residual": rng.randn() * 0.5,
            "fuel_flow_norm_residual": rng.randn() * 0.5,
            "vibration_norm_residual": rng.randn() * 0.5,
        })
    df_full = pd.DataFrame(records)
    df_prefix = df_full.iloc[:t_cutoff + 1].copy()

    pipe_prefix = HealthIndexPipeline()
    pipe_full = HealthIndexPipeline()

    res_prefix = pipe_prefix.process_dataframe(df_prefix)
    res_full = pipe_full.process_dataframe(df_full)

    # Result at t_cutoff must be bit-for-bit identical
    r_pre = res_prefix[t_cutoff]
    r_full = res_full[t_cutoff]

    assert r_pre.raw_health_index == r_full.raw_health_index
    assert r_pre.smoothed_health_index == r_full.smoothed_health_index
    assert r_pre.health_state == r_full.health_state
    assert r_pre.degradation_trend == r_full.degradation_trend


# =====================================================================
# Test 13: Future Sample Modification Invariance
# =====================================================================
def test_13_future_sample_modification_invariance():
    """Modifying future samples after time t has zero effect on the output at time t."""
    rng = np.random.RandomState(42)
    n = 30
    t_eval = 15

    base_records = []
    for t in range(n):
        base_records.append({
            "timestamp": float(t),
            "engine_id": "ENG_001",
            "rpm_norm_residual": 0.2, "cht_norm_residual": 0.2, "egt_norm_residual": 0.2,
            "oil_temp_norm_residual": 0.2, "oil_pressure_norm_residual": 0.2,
            "fuel_flow_norm_residual": 0.2, "vibration_norm_residual": 0.2,
        })

    df_a = pd.DataFrame(base_records)
    df_b = pd.DataFrame(base_records)
    # Corrupt future samples in df_b (after t_eval)
    for idx in range(t_eval + 1, n):
        df_b.loc[idx, "cht_norm_residual"] = 10.0
        df_b.loc[idx, "vibration_norm_residual"] = 10.0

    res_a = HealthIndexPipeline().process_dataframe(df_a)
    res_b = HealthIndexPipeline().process_dataframe(df_b)

    assert res_a[t_eval].raw_health_index == res_b[t_eval].raw_health_index
    assert res_a[t_eval].smoothed_health_index == res_b[t_eval].smoothed_health_index
    assert res_a[t_eval].degradation_rate == res_b[t_eval].degradation_rate


# =====================================================================
# Test 14: Initial History Degradation Rate (INSUFFICIENT_HISTORY)
# =====================================================================
def test_14_initial_history_degradation_rate():
    """When elapsed history < rate_horizon (10s), degradation_rate is NaN and trend is INSUFFICIENT_HISTORY."""
    pipeline = HealthIndexPipeline()

    records = []
    for t in range(8):  # 0 to 7 seconds (< 10s horizon)
        records.append({
            "timestamp": float(t),
            "engine_id": "ENG_001",
            "rpm_norm_residual": 0.1, "cht_norm_residual": 0.1, "egt_norm_residual": 0.1,
            "oil_temp_norm_residual": 0.1, "oil_pressure_norm_residual": 0.1,
            "fuel_flow_norm_residual": 0.1, "vibration_norm_residual": 0.1,
        })
    df = pd.DataFrame(records)
    results = pipeline.process_dataframe(df)

    for r in results:
        assert math.isnan(r.degradation_rate)
        assert r.degradation_trend == DegradationTrend.INSUFFICIENT_HISTORY.value


# =====================================================================
# Test 15: Sustained Degradation Trend
# =====================================================================
def test_15_sustained_degradation_trend():
    """Sustained physical deterioration produces negative rate and DEGRADING or RAPIDLY_DEGRADING trend."""
    pipeline = HealthIndexPipeline()

    records = []
    for t in range(25):
        # Progressively worsen CHT and Oil Temp
        res_val = 0.5 + (t * 0.25)
        records.append({
            "timestamp": float(t),
            "engine_id": "ENG_001",
            "rpm_norm_residual": 0.1,
            "cht_norm_residual": res_val,
            "oil_temp_norm_residual": res_val * 0.8,
            "egt_norm_residual": 0.1,
            "oil_pressure_norm_residual": -0.1,
            "fuel_flow_norm_residual": 0.1,
            "vibration_norm_residual": 0.1,
        })
    df = pd.DataFrame(records)
    results = pipeline.process_dataframe(df)

    later_results = results[15:]
    trends = [r.degradation_trend for r in later_results]
    rates = [r.degradation_rate for r in later_results]

    assert any(t in (DegradationTrend.DEGRADING.value, DegradationTrend.RAPIDLY_DEGRADING.value) for t in trends)
    assert all(r < 0.0 for r in rates if not math.isnan(r))


# =====================================================================
# Test 16: Transient Spike Resilience
# =====================================================================
def test_16_transient_spike_resilience():
    """A single-second outlier spike does not trigger RAPIDLY_DEGRADING trend."""
    pipeline = HealthIndexPipeline()

    records = []
    for t in range(20):
        # Normal data with a single 1s spike at t=12
        spike = 6.0 if t == 12 else 0.2
        records.append({
            "timestamp": float(t),
            "engine_id": "ENG_001",
            "rpm_norm_residual": 0.1,
            "cht_norm_residual": spike,
            "egt_norm_residual": 0.1,
            "oil_temp_norm_residual": 0.1,
            "oil_pressure_norm_residual": 0.1,
            "fuel_flow_norm_residual": 0.1,
            "vibration_norm_residual": 0.1,
        })
    df = pd.DataFrame(records)
    results = pipeline.process_dataframe(df)

    # At and after t=12, trend must not be RAPIDLY_DEGRADING
    for r in results[10:]:
        assert r.degradation_trend != DegradationTrend.RAPIDLY_DEGRADING.value


# =====================================================================
# Test 17: Health Index Bounded Range [0.0, 1.0]
# =====================================================================
def test_17_hi_bounded_range():
    """raw_health_index and smoothed_health_index remain bounded within [0.0, 1.0]."""
    pipeline = HealthIndexPipeline()

    # Extreme positive residuals
    extreme_pos = {
        "timestamp": 1.0, "engine_id": "ENG_001",
        "rpm_norm_residual": 100.0, "cht_norm_residual": 100.0, "egt_norm_residual": 100.0,
        "oil_temp_norm_residual": 100.0, "oil_pressure_norm_residual": 100.0,
        "fuel_flow_norm_residual": 100.0, "vibration_norm_residual": 100.0,
    }
    r_pos = pipeline.process_sample(extreme_pos)
    assert 0.0 <= r_pos.raw_health_index <= 1.0
    assert 0.0 <= r_pos.smoothed_health_index <= 1.0
    assert r_pos.health_state == HealthState.CRITICAL.value

    # Extreme negative residuals
    extreme_neg = {
        "timestamp": 2.0, "engine_id": "ENG_001",
        "rpm_norm_residual": -100.0, "cht_norm_residual": -100.0, "egt_norm_residual": -100.0,
        "oil_temp_norm_residual": -100.0, "oil_pressure_norm_residual": -100.0,
        "fuel_flow_norm_residual": -100.0, "vibration_norm_residual": -100.0,
    }
    r_neg = pipeline.process_sample(extreme_neg)
    assert 0.0 <= r_neg.raw_health_index <= 1.0
    assert 0.0 <= r_neg.smoothed_health_index <= 1.0


# =====================================================================
# Test 18: Phase 7 Independence
# =====================================================================
def test_18_phase7_independence():
    """Pipeline operates identically without Phase 7 inputs."""
    sample = {
        "timestamp": 1.0, "engine_id": "ENG_001",
        "rpm_norm_residual": 0.1, "cht_norm_residual": 0.2, "egt_norm_residual": 0.1,
        "oil_temp_norm_residual": 0.1, "oil_pressure_norm_residual": -0.1,
        "fuel_flow_norm_residual": 0.0, "vibration_norm_residual": 0.1,
    }
    # No Phase 7 inputs passed
    res = HealthIndexPipeline().process_sample(sample)
    assert res.raw_health_index >= 0.85
    assert res.anomaly_status is None
    assert res.anomaly_score is None


# =====================================================================
# Test 19: Phase 8 Independence
# =====================================================================
def test_19_phase8_independence():
    """Pipeline operates identically without Phase 8 inputs."""
    sample = {
        "timestamp": 1.0, "engine_id": "ENG_001",
        "rpm_norm_residual": 0.1, "cht_norm_residual": 0.2, "egt_norm_residual": 0.1,
        "oil_temp_norm_residual": 0.1, "oil_pressure_norm_residual": -0.1,
        "fuel_flow_norm_residual": 0.0, "vibration_norm_residual": 0.1,
    }
    # No Phase 8 inputs passed
    res = HealthIndexPipeline().process_sample(sample)
    assert res.raw_health_index >= 0.85
    assert res.diagnosed_fault is None
    assert res.diagnostic_confidence is None


# =====================================================================
# Test 20: Effective Weights Tracking & Channel Audit
# =====================================================================
def test_20_effective_weights_tracking():
    """Renormalized weights and audit fields reflect missing and active channels."""
    pipeline = HealthIndexPipeline()
    sample = {
        "timestamp": 1.0, "engine_id": "ENG_001",
        "rpm_norm_residual": 0.1,
        "cht_norm_residual": np.nan,  # Missing
        "egt_norm_residual": 0.1,
        "oil_temp_norm_residual": 0.1,
        "oil_pressure_norm_residual": 0.1,
        "fuel_flow_norm_residual": 0.1,
        "vibration_norm_residual": 0.1,
    }
    res = pipeline.process_sample(sample)

    assert "cht" not in res.effective_channel_weights
    assert "cht" in res.missing_channels
    assert abs(sum(res.effective_channel_weights.values()) - 1.0) < 1e-4
    # The sum of active channel base weights was 1.0 - 0.18 = 0.82
    # Oil pressure base weight is 0.20 -> effective weight is 0.20 / 0.82
    expected_oil_p_eff = 0.20 / 0.82
    assert abs(res.effective_channel_weights["oil_pressure"] - expected_oil_p_eff) < 1e-4


# =====================================================================
# Test 21: Existing Tests Regression Safety
# =====================================================================
def test_21_existing_tests_intact():
    """Phase 9 imports do not modify or interfere with Phase 1-8 modules."""
    from simulator.fault_interface import FaultType
    from digital_twin.twin_model import DigitalTwin
    from anomaly_detection import HybridAnomalyDetector
    from fault_diagnosis import XGBoostFaultClassifier

    assert len(FaultType) == 6
    assert DigitalTwin() is not None
    assert HybridAnomalyDetector() is not None
    assert XGBoostFaultClassifier() is not None


# =====================================================================
# Test 22: Mission State Isolation (Audit Check 1)
# =====================================================================
def test_mission_state_isolation():
    """
    Mission B must NEVER inherit Mission A's EWMA state, degradation history,
    or timestamps when running on the same engine_id.
    """
    pipeline_shared = HealthIndexPipeline()

    # 1. Process Mission A using engine_id="UAV-01" with severe degradation
    mission_a_samples = []
    for i in range(20):
        mission_a_samples.append({
            "timestamp": float(i),
            "engine_id": "UAV-01",
            "mission_id": "M001",
            "rpm_norm_residual": 4.5 + 0.1 * i,
            "cht_norm_residual": 5.0 + 0.1 * i,
            "egt_norm_residual": 4.5 + 0.1 * i,
            "oil_temp_norm_residual": 4.0 + 0.1 * i,
            "oil_pressure_norm_residual": -4.8 - 0.1 * i,
            "fuel_flow_norm_residual": 3.5,
            "vibration_norm_residual": 4.0,
        })
    df_mission_a = pd.DataFrame(mission_a_samples)
    out_a = pipeline_shared.process_dataframe(df_mission_a)

    # Verify Mission A actually degraded significantly
    assert out_a[-1].smoothed_health_index < 0.40
    assert out_a[-1].health_state in [
        HealthState.SEVERELY_DEGRADED.value,
        HealthState.CRITICAL.value,
    ]

    # 2. Then process Mission B using the same engine_id on the same pipeline
    mission_b_samples = []
    for i in range(20):
        mission_b_samples.append({
            "timestamp": float(i),
            "engine_id": "UAV-01",
            "mission_id": "M002",
            "rpm_norm_residual": 0.05,
            "cht_norm_residual": 0.05,
            "egt_norm_residual": 0.05,
            "oil_temp_norm_residual": 0.05,
            "oil_pressure_norm_residual": -0.05,
            "fuel_flow_norm_residual": 0.05,
            "vibration_norm_residual": 0.05,
        })
    df_mission_b = pd.DataFrame(mission_b_samples)
    out_b_shared = pipeline_shared.process_dataframe(df_mission_b)

    # 3. Separately process Mission B from a fresh Phase 9 pipeline
    pipeline_fresh = HealthIndexPipeline()
    out_b_fresh = pipeline_fresh.process_dataframe(df_mission_b)

    # 4. Compare Mission B outputs - they must be strictly identical
    assert len(out_b_shared) == len(out_b_fresh) == 20
    for i in range(20):
        res_shared = out_b_shared[i]
        res_fresh = out_b_fresh[i]

        assert res_shared.raw_health_index == pytest.approx(res_fresh.raw_health_index, abs=1e-6)
        assert res_shared.smoothed_health_index == pytest.approx(res_fresh.smoothed_health_index, abs=1e-6)
        assert res_shared.health_state == res_fresh.health_state
        assert res_shared.degradation_trend == res_fresh.degradation_trend

        if math.isnan(res_fresh.degradation_rate):
            assert math.isnan(res_shared.degradation_rate)
        else:
            assert res_shared.degradation_rate == pytest.approx(res_fresh.degradation_rate, abs=1e-6)

    # Ensure Mission B started fresh at healthy HI (~0.99), not inheriting Mission A's degraded state (~0.40)
    assert out_b_shared[0].smoothed_health_index > 0.95
    assert out_b_shared[0].health_state == HealthState.HEALTHY.value

    # Verify behavior when mission_id is absent: pipeline executes without crash
    sample_no_mission = {
        "timestamp": 100.0,
        "engine_id": "UAV-01",
        "rpm_norm_residual": 0.1,
        "cht_norm_residual": 0.1,
        "egt_norm_residual": 0.1,
        "oil_temp_norm_residual": 0.1,
        "oil_pressure_norm_residual": -0.1,
        "fuel_flow_norm_residual": 0.1,
        "vibration_norm_residual": 0.1,
    }
    res_no_mission = pipeline_shared.process_sample(sample_no_mission)
    assert res_no_mission.raw_health_index > 0.85
    assert res_no_mission.mission_id is None


# =====================================================================
# Test 23: Duplicate and Out-of-Order Timestamps (Audit Check 2)
# =====================================================================
def test_duplicate_and_out_of_order_timestamps():
    """
    Verify deterministic behavior for dt <= 0 (duplicates and out-of-order).
    Must return NaN rate, INSUFFICIENT_DATA trend, and never divide by zero
    or corrupt causal history.
    """
    pipeline = HealthIndexPipeline()

    base_sample = {
        "engine_id": "UAV-01",
        "mission_id": "M_ROBUST",
        "rpm_norm_residual": 0.2,
        "cht_norm_residual": 0.2,
        "egt_norm_residual": 0.2,
        "oil_temp_norm_residual": 0.2,
        "oil_pressure_norm_residual": -0.2,
        "fuel_flow_norm_residual": 0.1,
        "vibration_norm_residual": 0.1,
    }

    # Step 1: Normal observation at t=20.0
    s1 = dict(base_sample, timestamp=20.0)
    r1 = pipeline.process_sample(s1)
    assert not math.isnan(r1.smoothed_health_index)

    # Step 2: Duplicate timestamp at t=20.0 (dt = 0)
    s2 = dict(base_sample, timestamp=20.0)
    r2 = pipeline.process_sample(s2)
    assert math.isnan(r2.degradation_rate)
    assert r2.degradation_trend == DegradationTrend.INSUFFICIENT_DATA.value
    assert not math.isnan(r2.smoothed_health_index)

    # Step 3: Out-of-order timestamp at t=19.5 (dt = -0.5)
    s3 = dict(base_sample, timestamp=19.5)
    r3 = pipeline.process_sample(s3)
    assert math.isnan(r3.degradation_rate)
    assert r3.degradation_trend == DegradationTrend.INSUFFICIENT_DATA.value
    assert not math.isnan(r3.smoothed_health_index)

    # Step 4: Subsequent causal observation at t=21.0 (valid dt = 1.0 > 0)
    s4 = dict(base_sample, timestamp=21.0)
    r4 = pipeline.process_sample(s4)
    # The valid observation must process normally without being corrupted by the invalid dt
    assert not math.isnan(r4.smoothed_health_index)
    assert r4.degradation_trend == DegradationTrend.INSUFFICIENT_HISTORY.value


# =====================================================================
# Test 24: Large Timestamp Gap and History Reset (Audit Check 2)
# =====================================================================
def test_large_timestamp_gap_history_reset():
    """
    A timestamp gap larger than max_timestamp_gap_s must break degradation history,
    reset smoother state, and require sufficient new history before calculating rate/trend.
    """
    config = HealthIndexConfig(
        max_timestamp_gap_s=5.0,
        rate_horizon_s=10.0,
    )
    pipeline = HealthIndexPipeline(config=config)

    # 1. Accumulate continuous history from t=0.0 to t=15.0 at 1 Hz
    history_samples = []
    for i in range(16):
        history_samples.append({
            "timestamp": float(i),
            "engine_id": "UAV-01",
            "mission_id": "M_GAP_TEST",
            "rpm_norm_residual": 0.5 + 0.05 * i,
            "cht_norm_residual": 0.5 + 0.05 * i,
            "egt_norm_residual": 0.5,
            "oil_temp_norm_residual": 0.5,
            "oil_pressure_norm_residual": -0.5,
            "fuel_flow_norm_residual": 0.5,
            "vibration_norm_residual": 0.5,
        })
    res_cont = pipeline.process_dataframe(pd.DataFrame(history_samples))

    # At t=15.0, history has 16 samples spanning 15 seconds (> 10s horizon)
    assert res_cont[-1].degradation_trend in [
        DegradationTrend.STABLE.value,
        DegradationTrend.DEGRADING.value,
    ]
    assert not math.isnan(res_cont[-1].degradation_rate)

    # 2. Introduce a large timestamp gap: jump from t=15.0 to t=60.0 (gap = 45s > 5s max gap)
    gap_sample = {
        "timestamp": 60.0,
        "engine_id": "UAV-01",
        "mission_id": "M_GAP_TEST",
        "rpm_norm_residual": 0.5,
        "cht_norm_residual": 0.5,
        "egt_norm_residual": 0.5,
        "oil_temp_norm_residual": 0.5,
        "oil_pressure_norm_residual": -0.5,
        "fuel_flow_norm_residual": 0.5,
        "vibration_norm_residual": 0.5,
    }
    res_gap = pipeline.process_sample(gap_sample)

    # History must be broken: rate is NaN and trend is INSUFFICIENT_HISTORY
    assert math.isnan(res_gap.degradation_rate)
    assert res_gap.degradation_trend == DegradationTrend.INSUFFICIENT_HISTORY.value

    # 3. Verify causal behavior after timestamp gap:
    # Adding 4 more samples (t=61..64) -> only 4s elapsed since reset (< 10s horizon)
    post_gap_samples = []
    for i in range(1, 5):
        post_gap_samples.append({
            "timestamp": 60.0 + float(i),
            "engine_id": "UAV-01",
            "mission_id": "M_GAP_TEST",
            "rpm_norm_residual": 0.5,
            "cht_norm_residual": 0.5,
            "egt_norm_residual": 0.5,
            "oil_temp_norm_residual": 0.5,
            "oil_pressure_norm_residual": -0.5,
            "fuel_flow_norm_residual": 0.5,
            "vibration_norm_residual": 0.5,
        })
    res_partial = pipeline.process_dataframe(pd.DataFrame(post_gap_samples))
    for r in res_partial:
        assert r.degradation_trend == DegradationTrend.INSUFFICIENT_HISTORY.value

    # Add samples up to t=71.0 (11s elapsed since t=60.0 >= 10s horizon)
    sufficient_samples = []
    for i in range(5, 12):
        sufficient_samples.append({
            "timestamp": 60.0 + float(i),
            "engine_id": "UAV-01",
            "mission_id": "M_GAP_TEST",
            "rpm_norm_residual": 0.5,
            "cht_norm_residual": 0.5,
            "egt_norm_residual": 0.5,
            "oil_temp_norm_residual": 0.5,
            "oil_pressure_norm_residual": -0.5,
            "fuel_flow_norm_residual": 0.5,
            "vibration_norm_residual": 0.5,
        })
    res_sufficient = pipeline.process_dataframe(pd.DataFrame(sufficient_samples))
    # Once sufficient history is re-established, trend and rate become active again
    assert not math.isnan(res_sufficient[-1].degradation_rate)
    assert res_sufficient[-1].degradation_trend in [
        DegradationTrend.STABLE.value,
        DegradationTrend.DEGRADING.value,
    ]


# =====================================================================
# Test 25: No Division by Zero or Negative dt in Tracker
# =====================================================================
def test_no_division_by_zero_or_negative_dt():
    """
    Directly verify that DegradationTracker handles dt <= 0 deterministically
    without ZeroDivisionError, returning (nan, INSUFFICIENT_DATA, health_state).
    """
    tracker = DegradationTracker()

    # Initial sample
    rate1, trend1, state1 = tracker.update_and_evaluate(timestamp=10.0, smoothed_hi=0.90)
    assert math.isnan(rate1)
    assert trend1 == DegradationTrend.INSUFFICIENT_HISTORY.value
    assert state1 == HealthState.HEALTHY.value

    # Duplicate timestamp (dt = 0)
    rate2, trend2, state2 = tracker.update_and_evaluate(timestamp=10.0, smoothed_hi=0.85)
    assert math.isnan(rate2)
    assert trend2 == DegradationTrend.INSUFFICIENT_DATA.value
    assert state2 == HealthState.HEALTHY.value

    # Out-of-order timestamp (dt = -2.0)
    rate3, trend3, state3 = tracker.update_and_evaluate(timestamp=8.0, smoothed_hi=0.85)
    assert math.isnan(rate3)
    assert trend3 == DegradationTrend.INSUFFICIENT_DATA.value
    assert state3 == HealthState.HEALTHY.value

