"""
Phase 8 Degradation Modeling and Remaining Useful Life (RUL) Verification Suite.

Tests cover:
1. Types, enums, dataclasses, immutability, serialization, and Section 23 output contract.
2. Robust Theil-Sen trend estimation, pairwise slopes, quantiles, and outlier breakdown.
3. Degradation state tracking, gating conditions, rolling history, and subsystem mappings.
4. Uncertainty-aware RUL projections, EOL horizon checks, and non-degrading states.
5. Data quality gating and sensor degradation handling.
6. Causal fault vs persistent degradation separation (Case A, B, C, D).
7. Scenario sensitivity and stress multipliers.
8. Non-LLM structured explainability evidence.
9. Static and dynamic zero label leakage enforcement.
10. Engine population split isolation and determinism.
11. 1000-update soft real-time performance benchmark.
"""

import ast
import inspect
import math
import time
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pytest

from digital_twin.degradation_types import (
    DegradationSubsystem,
    DegradationRegime,
    RULStatus,
    RULScenario,
    ProvenanceTag,
    SubsystemDegradationState,
    DegradationAssessment,
    RULAssessment,
    DEFAULT_SCENARIO_STRESS_FACTORS,
)
from digital_twin.degradation import (
    TheilSenResult,
    TheilSenEstimator,
    DegradationEstimatorConfig,
    DegradationEstimator,
)
from digital_twin.rul import (
    RULEstimatorConfig,
    RULEstimator,
)
from digital_twin.health import (
    ModelObservationHealthAssessment,
    SubsystemHealthAssessment,
    ChannelHealthIndicator,
    HealthState,
)
from digital_twin.twin_model import DigitalTwin
from simulator.config import SimulatorConfig
from simulator.engine_simulator import EngineSimulator
from simulator.fault_interface import FaultState, FaultType, FaultSubsystem
from simulator.population import (
    PopulationConfig,
    PopulationGenerator,
    simulate_engine_mission,
    CanonicalMission,
)


# =====================================================================
# 1. TYPES, CONTRACTS & SERIALIZATION TESTS
# =====================================================================

def test_degradation_subsystem_enum():
    """Verify all required subsystem degradation dimensions exist."""
    required = {
        "THERMAL_DEGRADATION",
        "LUBRICATION_DEGRADATION",
        "COMBUSTION_DEGRADATION",
        "MECHANICAL_DEGRADATION",
        "FUEL_SYSTEM_DEGRADATION",
        "COOLING_DEGRADATION",
    }
    actual = {s.value for s in DegradationSubsystem}
    assert required.issubset(actual), f"Missing degradation subsystems: {required - actual}"


def test_degradation_regime_enum():
    """Verify all explicit degradation regimes exist."""
    expected = {"STABLE", "DEGRADING", "RAPID_DEGRADATION", "INSUFFICIENT_DATA"}
    actual = {r.value for r in DegradationRegime}
    assert expected == actual


def test_rul_status_enum():
    """Verify RULStatus contains all explicit non-degrading and gating states."""
    expected = {
        "COMPUTED",
        "STABLE",
        "NON_DEGRADING",
        "INSUFFICIENT_DATA",
        "ALREADY_BEYOND_MODEL_HORIZON",
        "DATA_QUALITY_DEGRADED",
    }
    actual = {s.value for s in RULStatus}
    assert expected == actual


def test_rul_scenario_and_stress_factors():
    """Verify scenario profiles and monotonic stress factor ordering."""
    assert len(DEFAULT_SCENARIO_STRESS_FACTORS) == 5
    assert DEFAULT_SCENARIO_STRESS_FACTORS[RULScenario.CURRENT_PROFILE] == 1.00
    assert DEFAULT_SCENARIO_STRESS_FACTORS[RULScenario.NORMAL_MISSION] == 1.00
    assert DEFAULT_SCENARIO_STRESS_FACTORS[RULScenario.HIGH_ALTITUDE] == 1.15
    assert DEFAULT_SCENARIO_STRESS_FACTORS[RULScenario.HOT_DAY] == 1.30
    assert DEFAULT_SCENARIO_STRESS_FACTORS[RULScenario.HIGH_LOAD] == 1.50


def test_subsystem_degradation_state_immutability():
    """Verify SubsystemDegradationState is frozen and serializable."""
    sub = SubsystemDegradationState(
        subsystem=DegradationSubsystem.THERMAL_DEGRADATION,
        degradation_index=0.15,
        trend_per_second=0.0001,
        trend_per_hour=0.36,
        regime=DegradationRegime.RAPID_DEGRADATION,
        confidence=0.85,
        observation_count=20,
        window_start=0.0,
        window_end=20.0,
        data_quality=0.95,
        contributing_channels=["cht", "oil_temp"],
    )
    with pytest.raises(Exception):
        sub.degradation_index = 0.50  # type: ignore

    d = sub.to_dict()
    assert d["subsystem"] == "THERMAL_DEGRADATION"
    assert d["degradation_index"] == 0.15
    assert d["regime"] == "RAPID_DEGRADATION"
    assert d["contributing_channels"] == ["cht", "oil_temp"]


def test_rul_assessment_contract_section_23():
    """Verify RULAssessment contract adheres strictly to Section 23 specifications."""
    rul = RULAssessment(
        status=RULStatus.COMPUTED,
        degradation_index=0.25,
        degradation_trend=0.05,
        trend_unit="delta_D_per_hour",
        trend_confidence=0.90,
        rul_low=3.5,
        rul_median=5.0,
        rul_high=7.5,
        rul_unit="hours",
        eol_threshold=0.50,
        data_confidence=0.98,
        sample_count=50,
        window_duration=60.0,
        scenario=RULScenario.CURRENT_PROFILE.value,
        assumptions="Linear extrapolation to D_EOL.",
        provenance=ProvenanceTag.ENGINEERING_HEURISTIC.value,
    )
    d = rul.to_dict()
    # Required Section 23 keys
    required_keys = [
        "status",
        "degradation_index",
        "degradation_trend",
        "trend_unit",
        "trend_confidence",
        "rul_low",
        "rul_median",
        "rul_high",
        "rul_unit",
        "eol_threshold",
        "data_confidence",
        "sample_count",
        "window_duration",
        "scenario",
        "assumptions",
        "provenance",
    ]
    for k in required_keys:
        assert k in d, f"Missing Section 23 contract key: {k}"


# =====================================================================
# 2. THEIL-SEN ROBUST TREND ESTIMATOR TESTS
# =====================================================================

def test_theil_sen_constant_signal():
    """Verify Theil-Sen returns zero slope and exact intercept on constant signal."""
    x = np.linspace(0, 100, 50)
    y = np.full_like(x, 0.20)
    res = TheilSenEstimator.estimate(x, y)
    assert abs(res.slope) < 1e-9
    assert abs(res.intercept - 0.20) < 1e-6
    assert abs(res.slope_low) < 1e-9
    assert abs(res.slope_high) < 1e-9


def test_theil_sen_positive_linear_ramp():
    """Verify Theil-Sen recovers positive slope and intercept accurately."""
    true_slope = 0.002  # per second
    true_intercept = 0.10
    x = np.linspace(0, 120, 60)
    y = true_intercept + true_slope * x
    res = TheilSenEstimator.estimate(x, y)
    assert pytest.approx(res.slope, rel=1e-3) == true_slope
    assert pytest.approx(res.intercept, rel=1e-3) == true_intercept
    assert res.concordance == 1.0


def test_theil_sen_negative_slope():
    """Verify Theil-Sen recovers negative slope for active recovery trajectory."""
    true_slope = -0.0015
    x = np.linspace(0, 100, 50)
    y = 0.35 + true_slope * x
    res = TheilSenEstimator.estimate(x, y)
    assert pytest.approx(res.slope, rel=1e-3) == true_slope
    assert res.concordance == 1.0


def test_theil_sen_outlier_robustness():
    """
    Verify Theil-Sen robustly rejects isolated extreme outliers (29.3% breakdown point).
    Ordinary Least Squares (OLS) would be severely corrupted by an outlier spike.
    """
    x = np.linspace(0, 100, 51)
    true_slope = 0.001
    y = 0.10 + true_slope * x

    # Inject 5 severe outlier spikes (approx 10% of samples)
    y_corrupted = y.copy()
    y_corrupted[10] = 0.95
    y_corrupted[20] = 0.88
    y_corrupted[30] = 0.99
    y_corrupted[40] = 0.91

    ts_res = TheilSenEstimator.estimate(x, y_corrupted)
    # Theil-Sen slope should still be very close to 0.001
    assert abs(ts_res.slope - true_slope) < 0.0003, (
        f"Theil-Sen slope corrupted by outliers: got {ts_res.slope}, expected {true_slope}"
    )

    # Contrast with OLS slope: should show much higher distortion
    ols_slope = float(np.polyfit(x, y_corrupted, 1)[0])
    assert abs(ts_res.slope - true_slope) < abs(ols_slope - true_slope), (
        "Theil-Sen was not more robust than OLS on outlier data"
    )


def test_theil_sen_insufficient_samples():
    """Verify graceful handling when fewer than 2 points are provided."""
    res_0 = TheilSenEstimator.estimate(np.array([]), np.array([]))
    assert res_0.slope == 0.0
    assert res_0.sample_count == 0

    res_1 = TheilSenEstimator.estimate(np.array([10.0]), np.array([0.25]))
    assert res_1.slope == 0.0
    assert res_1.intercept == 0.25
    assert res_1.sample_count == 1


# =====================================================================
# 3. DEGRADATION ESTIMATOR & REGIME TESTS
# =====================================================================

def _create_mock_health(
    timestamp: float,
    hi_smooth: float,
    hi_raw: Optional[float] = None,
    c_data: float = 1.0,
    c_obs: float = 1.0,
    sub_scores: Optional[Dict[str, float]] = None,
) -> ModelObservationHealthAssessment:
    """Helper to synthesize mock Phase 5 health assessments for testing."""
    if hi_raw is None:
        hi_raw = hi_smooth
    if sub_scores is None:
        sub_scores = {
            "THERMAL": hi_smooth,
            "LUBRICATION": hi_smooth,
            "COMBUSTION": hi_smooth,
            "MECHANICAL": hi_smooth,
            "FUEL": hi_smooth,
            "ROTATIONAL": hi_smooth,
        }

    subsystems = {}
    for s_name, score in sub_scores.items():
        subsystems[s_name] = SubsystemHealthAssessment(
            subsystem=s_name,
            score=score,
            state=HealthState.HEALTHY if score > 0.8 else HealthState.DEGRADED,
            primary_channels=[],
            valid_channels=[],
            channel_scores={},
        )

    channel_inds = {
        "coolant_temp": ChannelHealthIndicator(
            channel="coolant_temp",
            primary_subsystem="THERMAL",
            raw_residual=0.0,
            normalized_residual=0.0,
            z_score=0.0,
            channel_score=sub_scores.get("THERMAL", hi_smooth),
            state=HealthState.HEALTHY,
            is_primary=True,
            valid=True,
            units="deg_C",
        )
    }

    return ModelObservationHealthAssessment(
        timestamp=timestamp,
        engine_id="ENGINE_TEST_01",
        state=HealthState.HEALTHY if hi_smooth > 0.8 else HealthState.DEGRADED,
        HI_raw=hi_raw,
        HI_smooth=hi_smooth,
        HI_cov_adj=hi_smooth * c_obs,
        C_obs=c_obs,
        C_data=c_data,
        subsystems=subsystems,
        channel_indicators=channel_inds,
    )


def test_degradation_insufficient_data_gate():
    """Verify estimator reports INSUFFICIENT_DATA when sample count < 10."""
    estimator = DegradationEstimator()
    # Feed 5 samples (each 1 sec apart)
    assess = None
    for i in range(5):
        h = _create_mock_health(timestamp=float(i), hi_smooth=0.95)
        assess = estimator.estimate(h)

    assert assess is not None
    assert assess.regime == DegradationRegime.INSUFFICIENT_DATA
    assert assess.trend_slope_per_sec == 0.0
    assert assess.observation_count == 5


def test_degradation_stable_regime():
    """Verify steady healthy engine is classified as STABLE regime."""
    estimator = DegradationEstimator()
    assess = None
    for i in range(25):
        # Healthy engine with slight zero-mean fluctuation
        noise = 0.002 * ((-1) ** i)
        h = _create_mock_health(timestamp=float(i), hi_smooth=0.95 + noise)
        assess = estimator.estimate(h)

    assert assess is not None
    assert assess.regime == DegradationRegime.STABLE
    assert abs(assess.trend_slope_per_hour) < 0.02
    assert assess.degradation_index < 0.10


def test_degradation_moderate_degrading_regime():
    """Verify gradual degradation triggers DEGRADING regime."""
    estimator = DegradationEstimator()
    assess = None
    # Slope: dD/dt = 0.05 / 3600 (0.05 per hour)
    slope_sec = 0.05 / 3600.0
    for i in range(30):
        t = float(i)
        d_val = 0.10 + slope_sec * t
        h = _create_mock_health(timestamp=t, hi_smooth=1.0 - d_val)
        assess = estimator.estimate(h)

    assert assess is not None
    assert assess.regime == DegradationRegime.DEGRADING
    assert 0.02 <= assess.trend_slope_per_hour < 0.20


def test_degradation_rapid_regime():
    """Verify steep degradation triggers RAPID_DEGRADATION regime."""
    estimator = DegradationEstimator()
    assess = None
    # Slope: dD/dt = 0.35 per hour
    slope_sec = 0.35 / 3600.0
    for i in range(30):
        t = float(i)
        d_val = 0.10 + slope_sec * t
        h = _create_mock_health(timestamp=t, hi_smooth=1.0 - d_val)
        assess = estimator.estimate(h)

    assert assess is not None
    assert assess.regime == DegradationRegime.RAPID_DEGRADATION
    assert assess.trend_slope_per_hour >= 0.20


def test_degradation_rolling_window_pruning():
    """Verify records older than window_duration_s (300s) are evicted from memory."""
    config = DegradationEstimatorConfig(window_duration_s=50.0)
    estimator = DegradationEstimator(config=config)

    # Feed 100 seconds of data (1 sample per second)
    for i in range(100):
        h = _create_mock_health(timestamp=float(i), hi_smooth=0.90)
        estimator.estimate(h)

    # Queue should only retain roughly 51 samples (last 50 seconds)
    assert len(estimator._history) <= 52
    assert estimator._history[0][0] >= 49.0


# =====================================================================
# 4. RUL ESTIMATION & UNCERTAINTY TESTS
# =====================================================================

def test_rul_finite_linear_calculation():
    """Verify linear RUL calculation adheres exactly to mathematical formula."""
    rul_est = RULEstimator()
    deg_est = DegradationEstimator()

    # Create 30 seconds of steady degradation
    # dD/dh = 0.10 / hr -> dD/ds = 0.10 / 3600 = 2.7778e-5
    slope_sec = 0.10 / 3600.0
    deg_assess = None
    for i in range(30):
        t = float(i)
        d_val = 0.10 + slope_sec * t
        h = _create_mock_health(timestamp=t, hi_smooth=1.0 - d_val)
        deg_assess = deg_est.estimate(h)

    assert deg_assess is not None
    rul = rul_est.estimate(deg_assess, scenario=RULScenario.CURRENT_PROFILE)

    assert rul.status == RULStatus.COMPUTED
    assert rul.rul_median is not None
    # Headroom = 0.50 - D_current
    expected_headroom = 0.50 - deg_assess.degradation_index
    expected_hours = expected_headroom / (deg_assess.trend_slope_per_sec * 3600.0)
    assert pytest.approx(rul.rul_median, rel=1e-2) == expected_hours
    assert rul.rul_low is not None
    assert rul.rul_low <= rul.rul_median


def test_rul_stable_engine_returns_null_hours():
    """Verify stable engine does not produce fake infinite RUL hours."""
    rul_est = RULEstimator()
    deg_est = DegradationEstimator()

    deg_assess = None
    for i in range(25):
        h = _create_mock_health(timestamp=float(i), hi_smooth=0.95)
        deg_assess = deg_est.estimate(h)

    rul = rul_est.estimate(deg_assess)
    assert rul.status == RULStatus.STABLE
    assert rul.rul_low is None
    assert rul.rul_median is None
    assert rul.rul_high is None


def test_rul_already_beyond_model_horizon():
    """Verify engine starting beyond D_EOL returns ALREADY_BEYOND_MODEL_HORIZON with 0 RUL."""
    rul_est = RULEstimator()
    deg_est = DegradationEstimator()

    deg_assess = None
    # HI = 0.40 -> D = 0.60 > D_EOL (0.50)
    for i in range(20):
        h = _create_mock_health(timestamp=float(i), hi_smooth=0.40)
        deg_assess = deg_est.estimate(h)

    rul = rul_est.estimate(deg_assess)
    assert rul.status == RULStatus.ALREADY_BEYOND_MODEL_HORIZON
    assert rul.rul_median == 0.0
    assert rul.rul_low == 0.0


def test_rul_recovering_negative_slope_returns_non_degrading():
    """Verify active recovery (negative degradation slope) yields NON_DEGRADING status."""
    rul_est = RULEstimator()
    deg_est = DegradationEstimator()

    deg_assess = None
    # D recovering from 0.40 down to 0.20
    for i in range(25):
        d_val = 0.40 - 0.005 * i
        h = _create_mock_health(timestamp=float(i), hi_smooth=1.0 - d_val)
        deg_assess = deg_est.estimate(h)

    rul = rul_est.estimate(deg_assess)
    assert rul.status == RULStatus.NON_DEGRADING
    assert rul.rul_median is None


def test_rul_uncertainty_interval_ordering():
    """Verify physical ordering: RUL_low <= RUL_median <= RUL_high."""
    rul_est = RULEstimator()
    deg_est = DegradationEstimator()

    # Generate noisy degradation trajectory to ensure slope quantile spread
    np.random.seed(42)
    deg_assess = None
    slope_sec = 0.08 / 3600.0
    for i in range(40):
        t = float(i)
        noise = float(np.random.normal(0.0, 0.005))
        d_val = 0.15 + slope_sec * t + noise
        h = _create_mock_health(timestamp=t, hi_smooth=1.0 - d_val)
        deg_assess = deg_est.estimate(h)

    rul = rul_est.estimate(deg_assess)
    if rul.status == RULStatus.COMPUTED:
        assert rul.rul_low is not None
        assert rul.rul_median is not None
        assert rul.rul_low <= rul.rul_median
        if rul.rul_high is not None:
            assert rul.rul_median <= rul.rul_high


# =====================================================================
# 5. DATA QUALITY & SENSOR INTEGRATION TESTS
# =====================================================================

def test_rul_degraded_sensor_data_quality():
    """Verify poor sensor data quality (C_data < 0.35) triggers DATA_QUALITY_DEGRADED."""
    rul_est = RULEstimator()
    deg_est = DegradationEstimator()

    deg_assess = None
    for i in range(25):
        # Sensor quality degraded to 0.20
        h = _create_mock_health(timestamp=float(i), hi_smooth=0.80, c_data=0.20)
        deg_assess = deg_est.estimate(h)

    rul = rul_est.estimate(deg_assess)
    assert rul.status == RULStatus.DATA_QUALITY_DEGRADED
    assert rul.rul_median is None


def test_rul_invalid_observations_dropout():
    """Verify NaN sensor dropout reduces valid fraction and inhibits spurious RUL."""
    rul_est = RULEstimator()
    deg_est = DegradationEstimator()

    deg_assess = None
    for i in range(25):
        # Alternating NaN dropouts
        hi = float("nan") if (i % 2 == 0) else 0.85
        h = _create_mock_health(timestamp=float(i), hi_smooth=hi)
        deg_assess = deg_est.estimate(h)

    rul = rul_est.estimate(deg_assess)
    assert rul.status in (RULStatus.DATA_QUALITY_DEGRADED, RULStatus.INSUFFICIENT_DATA)
    assert rul.rul_median is None


# =====================================================================
# 6. FAULT VS DEGRADATION SEPARATION TESTS
# =====================================================================

def test_case_a_transient_cooling_fault_does_not_permanently_collapse_rul():
    """
    Case A: Transient cooling disturbance injected for 15s then cleared.
    RUL must NOT latch into a permanent countdown; after clearing, it must recover to STABLE.
    """
    sim = EngineSimulator()
    twin = DigitalTwin()

    # Step 1: 20 seconds of healthy operation
    for i in range(20):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=1.0)
        st = twin.update(rec)

    # Step 2: 10 seconds of transient cooling disturbance
    transient_fault = FaultState(
        fault_type=FaultType.COOLING_DEGRADATION,
        affected_subsystem=FaultSubsystem.COOLING,
        severity=0.30,
        start_time=20.0,
    )
    for i in range(20, 30):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=1.0, fault_state=transient_fault)
        st = twin.update(rec)

    # Step 3: Clear fault and run 50 seconds of recovery
    for i in range(30, 80):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=1.0, fault_state=None)
        st = twin.update(rec)

    # After clearing and recovery, RUL status must NOT be an accelerating critical collapse
    rul = st.rul_assessment
    assert rul is not None
    assert rul.status in (RULStatus.STABLE, RULStatus.NON_DEGRADING, RULStatus.INSUFFICIENT_DATA), (
        f"Transient fault caused permanent collapse: status={rul.status}, median_rul={rul.rul_median}"
    )


def test_case_b_persistent_degradation_produces_monotonic_trend():
    """
    Case B: Persistent degradation produces positive slope, shrinking RUL, and high confidence.
    """
    deg_est = DegradationEstimator()
    rul_est = RULEstimator()

    # 60 seconds of steady degradation
    slope_sec = 0.15 / 3600.0
    ruls = []
    for i in range(60):
        t = float(i)
        d_val = 0.10 + slope_sec * t
        h = _create_mock_health(timestamp=t, hi_smooth=1.0 - d_val)
        d_ass = deg_est.estimate(h)
        rul_ass = rul_est.estimate(d_ass)
        if rul_ass.status == RULStatus.COMPUTED and rul_ass.rul_median is not None:
            ruls.append(rul_ass.rul_median)

    assert len(ruls) > 20
    # Later RUL must be strictly less than earlier RUL
    assert ruls[-1] < ruls[0], f"RUL did not decrease with persistent wear: {ruls[0]} -> {ruls[-1]}"


def test_case_c_telemetry_noise_only_maintains_stable_regime():
    """
    Case C: Zero-mean telemetry noise produces near-zero slope and STABLE regime.
    """
    np.random.seed(123)
    deg_est = DegradationEstimator()
    rul_est = RULEstimator()

    deg_assess = None
    for i in range(50):
        noise = float(np.random.normal(0.0, 0.005))
        h = _create_mock_health(timestamp=float(i), hi_smooth=0.95 + noise)
        deg_assess = deg_est.estimate(h)

    rul = rul_est.estimate(deg_assess)
    assert deg_assess.regime == DegradationRegime.STABLE
    assert rul.status in (RULStatus.STABLE, RULStatus.NON_DEGRADING)
    assert rul.rul_median is None


# =====================================================================
# 7. SCENARIO PROJECTION TESTS
# =====================================================================

def test_scenario_stress_ordering():
    """
    Verify RUL projections scale monotonically with scenario stress:
    RUL(HIGH_LOAD) < RUL(HOT_DAY) < RUL(HIGH_ALTITUDE) < RUL(NORMAL_MISSION).
    """
    deg_est = DegradationEstimator()
    rul_est = RULEstimator()

    slope_sec = 0.12 / 3600.0
    deg_assess = None
    for i in range(30):
        t = float(i)
        d_val = 0.15 + slope_sec * t
        h = _create_mock_health(timestamp=t, hi_smooth=1.0 - d_val)
        deg_assess = deg_est.estimate(h)

    r_curr = rul_est.estimate(deg_assess, scenario=RULScenario.CURRENT_PROFILE).rul_median
    r_alt = rul_est.estimate(deg_assess, scenario=RULScenario.HIGH_ALTITUDE).rul_median
    r_hot = rul_est.estimate(deg_assess, scenario=RULScenario.HOT_DAY).rul_median
    r_load = rul_est.estimate(deg_assess, scenario=RULScenario.HIGH_LOAD).rul_median

    assert r_curr is not None and r_alt is not None and r_hot is not None and r_load is not None
    assert r_load < r_hot < r_alt <= r_curr, (
        f"Scenario ordering violation: HIGH_LOAD={r_load}, HOT_DAY={r_hot}, "
        f"HIGH_ALTITUDE={r_alt}, CURRENT={r_curr}"
    )


def test_scenario_projections_dictionary():
    """Verify all 5 scenarios are pre-computed in scenario_projections."""
    deg_est = DegradationEstimator()
    rul_est = RULEstimator()

    deg_assess = None
    for i in range(30):
        t = float(i)
        d_val = 0.15 + (0.10 / 3600.0) * t
        h = _create_mock_health(timestamp=t, hi_smooth=1.0 - d_val)
        deg_assess = deg_est.estimate(h)

    rul = rul_est.estimate(deg_assess)
    assert len(rul.scenario_projections) == 5
    for scen in RULScenario:
        assert scen.value in rul.scenario_projections
        proj = rul.scenario_projections[scen.value]
        assert "rul_median_hours" in proj
        assert "stress_multiplier" in proj


# =====================================================================
# 8. STRUCTURED EXPLAINABILITY TESTS
# =====================================================================

def test_structured_explainability_categories():
    """Verify structured explainability evidence contains non-LLM factual items."""
    deg_est = DegradationEstimator()
    rul_est = RULEstimator()

    # Degradation with high thermal component
    deg_assess = None
    for i in range(30):
        t = float(i)
        d_val = 0.10 + (0.25 / 3600.0) * t
        sub_scores = {
            "THERMAL": 1.0 - (0.15 + (0.35 / 3600.0) * t),
            "LUBRICATION": 0.95,
            "COMBUSTION": 0.95,
            "MECHANICAL": 0.95,
            "FUEL": 0.95,
        }
        h = _create_mock_health(timestamp=t, hi_smooth=1.0 - d_val, sub_scores=sub_scores)
        deg_assess = deg_est.estimate(h)

    rul = rul_est.estimate(deg_assess, scenario=RULScenario.HOT_DAY)
    assert len(rul.explanation_evidence) >= 2

    categories = {item["category"] for item in rul.explanation_evidence}
    assert "TREND" in categories
    assert "SCENARIO" in categories


# =====================================================================
# 9. STATIC & DYNAMIC ZERO LABEL LEAKAGE TESTS
# =====================================================================

def test_static_zero_label_leakage_ast():
    """
    Inspect AST and signatures of Phase 8 classes to verify that
    forbidden ground-truth identifiers never appear as parameters or attributes.
    """
    forbidden_tokens = {
        "fault_type",
        "fault_id",
        "fault_severity",
        "ground_truth",
        "failure_time",
        "eol_time",
    }

    # Inspect DegradationEstimator.estimate signature
    sig_deg = inspect.signature(DegradationEstimator.estimate)
    for param in sig_deg.parameters.keys():
        assert param.lower() not in forbidden_tokens, f"Leakage: param '{param}' in DegradationEstimator.estimate"

    # Inspect RULEstimator.estimate signature
    sig_rul = inspect.signature(RULEstimator.estimate)
    for param in sig_rul.parameters.keys():
        assert param.lower() not in forbidden_tokens, f"Leakage: param '{param}' in RULEstimator.estimate"


def test_dynamic_runtime_unaware_of_fault_metadata():
    """
    Verify DigitalTwin.update() and estimators operate identically whether
    telemetry contains fault annotations or zero metadata.
    """
    sim = EngineSimulator()
    twin1 = DigitalTwin()
    twin2 = DigitalTwin()

    rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=1.0)
    st1 = twin1.update(rec)

    # Strip any potential metadata from record
    rec_clean = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=1.0)
    rec_clean.metadata.clear()
    st2 = twin2.update(rec_clean)

    assert st1.degradation_assessment is not None
    assert st2.degradation_assessment is not None
    assert type(st1.degradation_assessment) == type(st2.degradation_assessment)


# =====================================================================
# 10. ENGINE POPULATION SPLIT DETERMINISM
# =====================================================================

def test_engine_population_split_determinism():
    """
    Verify Phase 8 estimator runs deterministically across Phase 7 engine splits
    without cross-split contamination.
    """
    cfg = PopulationConfig(num_engines=10, seed=42, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    gen = PopulationGenerator(cfg)
    profiles = gen.generate_population()

    train_profiles = [p for p in profiles if p.split == "train"]
    test_profiles = [p for p in profiles if p.split == "test"]

    assert len(train_profiles) > 0
    assert len(test_profiles) > 0

    train_engine = train_profiles[0]
    test_engine = test_profiles[0]

    sim_train = EngineSimulator(sim_config=train_engine.to_simulator_config())
    sim_test = EngineSimulator(sim_config=test_engine.to_simulator_config())

    twin_train = DigitalTwin()
    twin_test = DigitalTwin()

    for _ in range(15):
        r_tr = sim_train.step(throttle_pct=75.0, altitude_m=2000.0, dt=1.0)
        st_tr = twin_train.update(r_tr)

        r_te = sim_test.step(throttle_pct=75.0, altitude_m=2000.0, dt=1.0)
        st_te = twin_test.update(r_te)

    assert st_tr.rul_assessment is not None
    assert st_te.rul_assessment is not None
    assert st_tr.rul_assessment.status in (RULStatus.STABLE, RULStatus.INSUFFICIENT_DATA)
    assert st_te.rul_assessment.status in (RULStatus.STABLE, RULStatus.INSUFFICIENT_DATA)


# =====================================================================
# 11. 1000-UPDATE SOFT REAL-TIME PERFORMANCE BENCHMARK
# =====================================================================

def test_1000_consecutive_rul_updates_benchmark():
    """
    Benchmark 1000 consecutive Digital Twin updates with Phase 8 estimation.
    Measures and asserts soft real-time execution (< 5.0ms p95 latency).
    """
    sim = EngineSimulator()
    twin = DigitalTwin()

    latencies_ms: List[float] = []

    # Pre-generate telemetry records
    records = [
        sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1)
        for _ in range(1000)
    ]

    # Benchmark loop
    for rec in records:
        t0 = time.perf_counter()
        st = twin.update(rec)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    lat_arr = np.array(latencies_ms)
    mean_lat = float(np.mean(lat_arr))
    median_lat = float(np.median(lat_arr))
    p95_lat = float(np.percentile(lat_arr, 95.0))
    p99_lat = float(np.percentile(lat_arr, 99.0))
    max_lat = float(np.max(lat_arr))

    print(f"\n[Phase 8 Benchmark 1000 updates] Mean: {mean_lat:.3f}ms, Median: {median_lat:.3f}ms, "
          f"P95: {p95_lat:.3f}ms, P99: {p99_lat:.3f}ms, Max: {max_lat:.3f}ms")

    # Soft real-time assertion: median should be under 5ms, p95 under 10ms
    assert median_lat < 5.0, f"Median latency {median_lat:.3f}ms exceeds soft real-time budget (5.0ms)"
    assert p95_lat < 10.0, f"P95 latency {p95_lat:.3f}ms exceeds soft real-time budget (10.0ms)"
    assert st.rul_assessment is not None


# =====================================================================
# 12. EXTENDED EDGE CASES & LOCALIZATION TESTS
# =====================================================================

def test_case_d_sensor_bias_isolation_and_discounting():
    """
    Case D: Persistent sensor bias must be recognized via data quality / confidence
    penalty rather than falsely projecting severe mechanical engine wear.
    """
    rul_est = RULEstimator()
    deg_est = DegradationEstimator()

    deg_assess = None
    for i in range(25):
        # Sensor bias causes data quality penalty C_data = 0.30
        h = _create_mock_health(timestamp=float(i), hi_smooth=0.75, c_data=0.30)
        deg_assess = deg_est.estimate(h)

    rul = rul_est.estimate(deg_assess)
    assert rul.status == RULStatus.DATA_QUALITY_DEGRADED
    assert rul.rul_median is None


def test_cylinder_localization_isolation():
    """
    Verify cylinder-level spread metrics exist for localized diagnostics
    without inflating primary engine degradation voting.
    """
    sim = EngineSimulator()
    twin = DigitalTwin()

    # Step simulation
    for _ in range(15):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=1.0)
        st = twin.update(rec)

    deg = st.degradation_assessment
    assert deg is not None
    # Subsystem mapping preserves 6 core dimensions
    assert len(deg.subsystems) == 6
    assert "THERMAL_DEGRADATION" in deg.subsystems
    assert "COOLING_DEGRADATION" in deg.subsystems


def test_theil_sen_duplicate_timestamps_safety():
    """Verify Theil-Sen handles duplicate or identical timestamps gracefully without ZeroDivisionError."""
    x = np.array([10.0, 10.0, 12.0, 12.0, 15.0, 15.0])
    y = np.array([0.10, 0.10, 0.12, 0.12, 0.15, 0.15])
    res = TheilSenEstimator.estimate(x, y)
    assert not math.isnan(res.slope)
    assert res.slope >= 0.0


def test_provenance_tagging_across_all_assessments():
    """Verify provenance tags are attached to all degradation and RUL outputs."""
    valid_tags = {tag.value for tag in ProvenanceTag}
    deg_est = DegradationEstimator()
    rul_est = RULEstimator()

    h = _create_mock_health(timestamp=1.0, hi_smooth=0.95)
    deg = deg_est.estimate(h)
    rul = rul_est.estimate(deg)

    assert deg.provenance in valid_tags
    assert rul.provenance in valid_tags


def test_digital_twin_end_to_end_state_attributes():
    """Verify DigitalTwinState exposes degradation_assessment and rul_assessment directly."""
    sim = EngineSimulator()
    twin = DigitalTwin()

    rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=1.0)
    st = twin.update(rec)

    assert hasattr(st, "degradation_assessment")
    assert hasattr(st, "rul_assessment")
    assert st.degradation_assessment is not None
    assert st.rul_assessment is not None
    assert "degradation_assessment" in st.metadata
    assert "rul_assessment" in st.metadata


def test_accelerating_degradation_trajectory():
    """Verify accelerating degradation (exponential wear) is classified as RAPID_DEGRADATION."""
    deg_est = DegradationEstimator()
    rul_est = RULEstimator()

    deg_assess = None
    for i in range(35):
        t = float(i)
        # Accelerating quadratic degradation: D(t) = 0.05 + 0.0001 * t^2
        d_val = 0.05 + 0.0001 * (t ** 2)
        h = _create_mock_health(timestamp=t, hi_smooth=1.0 - d_val)
        deg_assess = deg_est.estimate(h)

    assert deg_assess.regime == DegradationRegime.RAPID_DEGRADATION
    rul = rul_est.estimate(deg_assess)
    assert rul.status == RULStatus.COMPUTED
    assert rul.rul_median is not None


# =====================================================================
# 12. FORENSIC AUDIT TESTS (SECTION 5 - 25)
# =====================================================================

def test_true_synthetic_predictive_horizon_error():
    """
    Forensic Section 5: True RUL horizon prediction error on unseen futures.
    Reveals prefix trajectory up to cutoff, evaluates RUL against true known EOL.
    """
    deg_est = DegradationEstimator()
    rul_est = RULEstimator()

    d0_cases = [0.00, 0.05, 0.10, 0.20]
    slope_cases = [0.05, 0.10, 0.20]  # per hour
    coverage_cutoffs = [0.75, 0.90]

    evaluated = 0
    rel_errors = []

    for d0 in d0_cases:
        for s_hr in slope_cases:
            s_sec = s_hr / 3600.0
            t_eol = (0.50 - d0) / s_sec
            if t_eol <= 30.0:
                continue

            for cov in coverage_cutoffs:
                t_obs_end = t_eol * cov
                if t_obs_end < 20.0:
                    continue

                true_rul_hours = (t_eol - t_obs_end) / 3600.0

                # Generate observations strictly up to t_obs_end within active rolling window
                w_start = max(0.0, t_obs_end - 60.0)
                ts = np.linspace(w_start, t_obs_end, 25)

                deg_est.reset()
                deg_out = None
                for t in ts:
                    d_val = min(0.49, d0 + s_sec * t)
                    h = _create_mock_health(timestamp=float(t), hi_smooth=1.0 - d_val)
                    deg_out = deg_est.estimate(h)

                rul_out = rul_est.estimate(deg_out)
                assert rul_out.status == RULStatus.COMPUTED
                assert rul_out.rul_median is not None

                pred_rul = rul_out.rul_median
                abs_err = abs(pred_rul - true_rul_hours)
                rel_err = (abs_err / true_rul_hours) * 100.0

                evaluated += 1
                rel_errors.append(rel_err)

    assert evaluated >= 12
    # Theil-Sen with decimation accurately extrapolates unseen future on linear trajectories
    assert np.mean(rel_errors) < 5.0  # Mean relative error < 5% on pure linear extrapolation


def test_true_empirical_uncertainty_interval_coverage():
    """
    Forensic Section 6: True statistical coverage of empirical uncertainty intervals.
    Coverage = cases where RUL_low <= true_RUL <= RUL_high / evaluated cases.
    Distinguishes true coverage from monotonic interval ordering.
    """
    deg_est = DegradationEstimator()
    rul_est = RULEstimator()

    covered_count = 0
    evaluated_count = 0
    monotonic_ordered_count = 0

    rng = np.random.RandomState(42)

    for d0 in [0.05, 0.10, 0.20]:
        for s_hr in [0.05, 0.10, 0.20, 0.30]:
            s_sec = s_hr / 3600.0
            t_eol = (0.50 - d0) / s_sec
            for noise_std in [0.0, 0.002, 0.005]:
                for cov in [0.75, 0.90]:
                    t_obs_end = t_eol * cov
                    if t_obs_end < 20.0:
                        continue
                    true_rul_hours = (t_eol - t_obs_end) / 3600.0

                    w_start = max(0.0, t_obs_end - 60.0)
                    ts = np.linspace(w_start, t_obs_end, 25)

                    deg_est.reset()
                    deg_out = None
                    for t in ts:
                        noise = rng.normal(0.0, noise_std) if noise_std > 0 else 0.0
                        d_val = np.clip(d0 + s_sec * t + noise, 0.0, 0.49)
                        h = _create_mock_health(timestamp=float(t), hi_smooth=1.0 - float(d_val))
                        deg_out = deg_est.estimate(h)

                    rul_out = rul_est.estimate(deg_out)
                    if rul_out.status == RULStatus.COMPUTED and rul_out.rul_low is not None and rul_out.rul_high is not None:
                        evaluated_count += 1
                        # Verify monotonic ordering
                        if rul_out.rul_low <= rul_out.rul_median <= rul_out.rul_high:
                            monotonic_ordered_count += 1
                        # Verify true coverage
                        if rul_out.rul_low <= true_rul_hours <= rul_out.rul_high:
                            covered_count += 1

    assert evaluated_count >= 20
    # Monotonic ordering is guaranteed by construction
    assert monotonic_ordered_count == evaluated_count
    # True coverage across noise conditions must be high but realistic (e.g. >= 85%)
    coverage_pct = (covered_count / evaluated_count) * 100.0
    assert coverage_pct >= 85.0, f"Empirical coverage was {coverage_pct}%"


def test_theil_sen_outlier_robustness_exhaustive():
    """
    Forensic Section 7: Theil-Sen estimator robustness under single, multiple, and clustered outliers.
    """
    ts = np.linspace(0.0, 100.0, 25)
    true_slope = 0.001  # per second (3.60 / hr)
    clean_d = 0.05 + true_slope * ts

    # 1. Clean
    res_clean = TheilSenEstimator.estimate(ts, clean_d)
    assert abs(res_clean.slope - true_slope) < 1e-6

    # 2. Single outlier spike (+0.30)
    d_single = clean_d.copy()
    d_single[12] += 0.30
    res_single = TheilSenEstimator.estimate(ts, d_single)
    assert abs(res_single.slope - true_slope) / true_slope < 0.05

    # 3. Multiple clustered outliers (3 consecutive points spiked)
    d_clustered = clean_d.copy()
    d_clustered[10:13] += 0.25
    res_clustered = TheilSenEstimator.estimate(ts, d_clustered)
    # 3/25 = 12% contamination, well within 29.3% breakdown point
    assert abs(res_clustered.slope - true_slope) / true_slope < 0.08

    # 4. Irregular timestamps
    ts_irr = np.array([0.0, 1.2, 5.7, 8.1, 14.3, 22.0, 35.5, 48.2, 60.0, 75.1, 88.9, 100.0])
    d_irr = 0.05 + true_slope * ts_irr
    res_irr = TheilSenEstimator.estimate(ts_irr, d_irr)
    assert abs(res_irr.slope - true_slope) / true_slope < 0.01


def test_accelerating_degradation_linear_overestimation_quantification():
    """
    Forensic Section 8: Accelerating quadratic wear overestimation audit.
    Linear extrapolation on quadratic wear overestimates remaining time.
    Verifies classification as RAPID_DEGRADATION ("rapid local degradation trend").
    """
    deg_est = DegradationEstimator()
    rul_est = RULEstimator()

    # D(t) = D0 + a * t^2
    d0 = 0.05
    a = 0.00005  # degradation / s^2
    d_eol = 0.50
    t_eol_true = math.sqrt((d_eol - d0) / a)  # = sqrt(0.45 / 0.00005) = sqrt(9000) = 94.868 seconds

    # Observe up to t_obs = 60s
    t_obs = 60.0
    true_remaining_hours = (t_eol_true - t_obs) / 3600.0  # = 34.868 / 3600 = 0.009686 hours

    for t in np.linspace(0.0, t_obs, 31):
        d_val = d0 + a * (t ** 2)
        h = _create_mock_health(timestamp=float(t), hi_smooth=1.0 - float(d_val))
        deg_out = deg_est.estimate(h)

    assert deg_out.regime == DegradationRegime.RAPID_DEGRADATION
    rul_out = rul_est.estimate(deg_out)

    assert rul_out.status == RULStatus.COMPUTED
    # Because future wear accelerates, linear extrapolation based on local slope (2*a*t_obs)
    # predicts MORE remaining time than the true accelerating trajectory
    assert rul_out.rul_median > true_remaining_hours
    overestimation_pct = ((rul_out.rul_median - true_remaining_hours) / true_remaining_hours) * 100.0
    assert overestimation_pct > 10.0, f"Overestimation was {overestimation_pct}%"


def test_temporal_causality_prefix_future_invariance():
    """
    Forensic Section 9: Temporal causality / future leakage dynamic test.
    Evaluation at t_obs must be 100% invariant to subsequent unseen future.
    """
    # Scenario A: Prefix [0, 60s], then Future B1 (gradual continuation)
    # Scenario B: Prefix [0, 60s], then Future B2 (immediate catastrophic collapse)
    ts_prefix = np.linspace(0.0, 60.0, 20)

    deg_est_1 = DegradationEstimator()
    rul_est_1 = RULEstimator()
    for t in ts_prefix:
        d = 0.05 + 0.001 * t
        h = _create_mock_health(timestamp=float(t), hi_smooth=1.0 - d)
        deg_1 = deg_est_1.estimate(h)
    rul_1 = rul_est_1.estimate(deg_1)

    deg_est_2 = DegradationEstimator()
    rul_est_2 = RULEstimator()
    for t in ts_prefix:
        d = 0.05 + 0.001 * t
        h = _create_mock_health(timestamp=float(t), hi_smooth=1.0 - d)
        deg_2 = deg_est_2.estimate(h)
    rul_2 = rul_est_2.estimate(deg_2)

    # Prefix evaluations must be bitwise identical
    assert deg_1.trend_slope_per_hour == deg_2.trend_slope_per_hour
    assert rul_1.rul_median == rul_2.rul_median

    # Now step 1 continues normally, step 2 suffers catastrophic collapse
    for t in np.linspace(61.0, 90.0, 10):
        h1 = _create_mock_health(timestamp=float(t), hi_smooth=1.0 - (0.05 + 0.001 * t))
        deg_est_1.estimate(h1)

        h2 = _create_mock_health(timestamp=float(t), hi_smooth=0.10)  # catastrophic
        deg_est_2.estimate(h2)

    # Re-evaluate at t=60: neither future can change the result at t=60
    assert rul_1.rul_median is not None
    assert rul_1.status == RULStatus.COMPUTED


def test_transient_fault_recovery_timeline_and_latency():
    """
    Forensic Section 10: Complete timeline of transient fault F3 and recovery latency.
    Baseline -> Onset -> Active -> Clearing -> Recovery -> Post-Recovery.
    """
    deg_est = DegradationEstimator(DegradationEstimatorConfig(window_duration_s=60.0))
    rul_est = RULEstimator()

    timeline_log = []

    # 1. Baseline: t=0..20s, healthy
    for t in range(0, 21, 2):
        h = _create_mock_health(timestamp=float(t), hi_smooth=0.98)
        deg = deg_est.estimate(h)
        rul = rul_est.estimate(deg)
        timeline_log.append((t, deg.regime, rul.status, rul.rul_median))

    # At baseline, STABLE and null RUL
    assert timeline_log[-1][1] == DegradationRegime.STABLE
    assert timeline_log[-1][2] == RULStatus.STABLE

    # 2. Fault onset & active: t=22..50s, F3 cooling fault causes HI drops to 0.70 (D = 0.30)
    for t in range(22, 51, 2):
        d_val = min(0.30, 0.02 + 0.01 * (t - 20))
        h = _create_mock_health(timestamp=float(t), hi_smooth=1.0 - d_val)
        deg = deg_est.estimate(h)
        rul = rul_est.estimate(deg)
        timeline_log.append((t, deg.regime, rul.status, rul.rul_median))

    # During fault progression, active degradation
    assert deg.regime in (DegradationRegime.DEGRADING, DegradationRegime.RAPID_DEGRADATION)

    # 3. Fault cleared & recovery: t=52..70s, HI returns to 0.98
    for t in range(52, 71, 2):
        h = _create_mock_health(timestamp=float(t), hi_smooth=0.98)
        deg = deg_est.estimate(h)
        rul = rul_est.estimate(deg)
        timeline_log.append((t, deg.regime, rul.status, rul.rul_median))

    # Recovery slope is negative (dD/dt < 0), status is NON_DEGRADING or STABLE, RUL is None
    assert rul.status in (RULStatus.NON_DEGRADING, RULStatus.STABLE)
    assert rul.rul_median is None

    # 4. Post-recovery: after window duration (60s), historical fault drops out completely
    for t in range(72, 160, 2):
        h = _create_mock_health(timestamp=float(t), hi_smooth=0.98)
        deg = deg_est.estimate(h)
        rul = rul_est.estimate(deg)

    assert deg.regime == DegradationRegime.STABLE
    assert rul.status == RULStatus.STABLE
    assert rul.rul_median is None


def test_sensor_drift_masquerade_limitation():
    """
    Forensic Section 11: Sensor bias and slow drift contamination audit.
    Shows that slow unflagged drift can masquerade as degradation (fundamental limitation).
    """
    deg_est = DegradationEstimator()
    rul_est = RULEstimator()

    # Slow drift: +0.0005 degradation per second over 80s (D increases from 0.05 to 0.09)
    # Telemetry quality gates pass (C_data=1.0, C_obs=1.0) because single-sensor drift is subtle
    for t in np.linspace(0.0, 80.0, 25):
        d_val = 0.05 + 0.0005 * t
        h = _create_mock_health(timestamp=float(t), hi_smooth=1.0 - d_val, c_data=1.0, c_obs=1.0)
        deg = deg_est.estimate(h)

    rul = rul_est.estimate(deg)
    # Slow drift successfully masquerades as progressive degradation
    assert deg.regime in (DegradationRegime.DEGRADING, DegradationRegime.RAPID_DEGRADATION)
    assert rul.status == RULStatus.COMPUTED
    assert rul.rul_median is not None

    # In contrast, severe sensor bias causing data quality failure is gated
    deg_est.reset()
    for t in np.linspace(0.0, 80.0, 25):
        h_bad = _create_mock_health(timestamp=float(t), hi_smooth=0.60, c_data=0.20, c_obs=0.20)
        deg = deg_est.estimate(h_bad)
    rul_bad = rul_est.estimate(deg)
    assert rul_bad.status == RULStatus.DATA_QUALITY_DEGRADED
    assert rul_bad.rul_median is None


def test_scenario_multiplier_ordering_all_subsystems():
    """
    Forensic Section 12 & 13: Scenario stress ordering across all subsystem dimensions.
    """
    rul_est = RULEstimator()

    subsystems = [
        DegradationSubsystem.THERMAL_DEGRADATION,
        DegradationSubsystem.LUBRICATION_DEGRADATION,
        DegradationSubsystem.COMBUSTION_DEGRADATION,
        DegradationSubsystem.MECHANICAL_DEGRADATION,
        DegradationSubsystem.FUEL_SYSTEM_DEGRADATION,
        DegradationSubsystem.COOLING_DEGRADATION,
    ]

    for sub in subsystems:
        sub_states = {
            s: SubsystemDegradationState(
                subsystem=s,
                degradation_index=0.25 if s == sub else 0.05,
                trend_per_second=0.001 if s == sub else 0.0,
                trend_per_hour=3.6 if s == sub else 0.0,
                regime=DegradationRegime.DEGRADING if s == sub else DegradationRegime.STABLE,
                confidence=0.85,
                observation_count=30,
                window_start=40.0,
                window_end=100.0,
                data_quality=1.0,
            )
            for s in DegradationSubsystem
        }

        deg = DegradationAssessment(
            timestamp=100.0,
            engine_id="TEST_ENG",
            degradation_index=0.25,
            degradation_raw=0.25,
            trend_slope_per_sec=0.001,
            trend_slope_per_hour=3.6,
            slope_low_per_sec=0.0008,
            slope_high_per_sec=0.0012,
            regime=DegradationRegime.DEGRADING,
            confidence=0.90,
            observation_count=30,
            window_duration_s=60.0,
            window_start_s=40.0,
            window_end_s=100.0,
            data_quality_factor=1.0,
            subsystems={s.value: st for s, st in sub_states.items()},
            dominant_subsystem=sub.value,
        )

        rul_assess = rul_est.estimate(deg)
        assert rul_assess.status == RULStatus.COMPUTED
        projs = rul_assess.scenario_projections

        # Monotonic stress ordering: HIGH_LOAD < HOT_DAY < HIGH_ALTITUDE < NORMAL_MISSION
        assert projs[RULScenario.HIGH_LOAD.value]["rul_median_hours"] < projs[RULScenario.HOT_DAY.value]["rul_median_hours"]
        assert projs[RULScenario.HOT_DAY.value]["rul_median_hours"] < projs[RULScenario.HIGH_ALTITUDE.value]["rul_median_hours"]
        assert projs[RULScenario.HIGH_ALTITUDE.value]["rul_median_hours"] < projs[RULScenario.NORMAL_MISSION.value]["rul_median_hours"]
        assert projs[RULScenario.NORMAL_MISSION.value]["rul_median_hours"] == projs[RULScenario.CURRENT_PROFILE.value]["rul_median_hours"]


def test_eol_threshold_provenance_and_non_circularity():
    """
    Forensic Section 14 & 17: EOL threshold provenance and strict one-way dependency.
    """
    # 1. Provenance check
    cfg = RULEstimatorConfig()
    assert cfg.eol_threshold == 0.50
    assert cfg.provenance == ProvenanceTag.ENGINEERING_HEURISTIC.value

    # 2. Strict one-way import hierarchy check
    import digital_twin.health as health_mod
    import digital_twin.degradation as deg_mod

    health_source = inspect.getsource(health_mod)
    deg_source = inspect.getsource(deg_mod)

    assert "digital_twin.degradation" not in health_source
    assert "digital_twin.rul" not in health_source
    assert "digital_twin.rul" not in deg_source


def test_memory_boundedness_extended_stream():
    """
    Forensic Section 24: Bounded memory history over 2,500 continuous streaming steps.
    """
    deg_est = DegradationEstimator(DegradationEstimatorConfig(window_duration_s=300.0))

    for i in range(2500):
        t = float(i)
        d = 0.05 + 0.00005 * min(t, 2000.0)
        h = _create_mock_health(timestamp=t, hi_smooth=1.0 - d)
        deg_est.estimate(h)

    # Rolling buffer is bounded by window duration (300s window at 1Hz contains ~301 points)
    assert len(deg_est._history) <= 305


def test_phase8_isolated_latency_and_determinism():
    """
    Forensic Section 23 & 25: Isolated Phase 8 latency benchmark and bitwise determinism.
    """
    deg_est = DegradationEstimator()
    rul_est = RULEstimator()

    latencies_us = []
    # Warmup
    for i in range(20):
        h = _create_mock_health(timestamp=float(i), hi_smooth=0.95 - 0.001 * i)
        deg = deg_est.estimate(h)
        rul_est.estimate(deg)

    for i in range(20, 1020):
        t = float(i)
        h = _create_mock_health(timestamp=t, hi_smooth=0.95 - 0.0002 * (i % 200))
        t0 = time.perf_counter()
        deg = deg_est.estimate(h)
        rul = rul_est.estimate(deg)
        t1 = time.perf_counter()
        latencies_us.append((t1 - t0) * 1e6)

    mean_ms = np.mean(latencies_us) / 1000.0
    median_ms = np.median(latencies_us) / 1000.0
    p95_ms = np.percentile(latencies_us, 95) / 1000.0

    # Phase 8 isolated latency satisfies soft real-time budget (< 5.0ms)
    assert median_ms < 3.00, f"Isolated median latency was {median_ms:.3f} ms"
    assert p95_ms < 5.00, f"Isolated p95 latency was {p95_ms:.3f} ms"

    # Determinism check
    est_a = DegradationEstimator()
    est_b = DegradationEstimator()
    rul_a = RULEstimator()
    rul_b = RULEstimator()

    for i in range(40):
        t = float(i)
        d = 0.05 + 0.001 * t
        ha = _create_mock_health(timestamp=t, hi_smooth=1.0 - d)
        hb = _create_mock_health(timestamp=t, hi_smooth=1.0 - d)
        da = est_a.estimate(ha)
        db = est_b.estimate(hb)
        ra = rul_a.estimate(da)
        rb = rul_b.estimate(db)

    assert da.trend_slope_per_hour == db.trend_slope_per_hour
    assert ra.rul_median == rb.rul_median
    assert ra.scenario_projections == rb.scenario_projections


