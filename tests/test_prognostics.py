"""
Dedicated Unit & Behavioral Test Suite for Phase 11: Remaining Useful Life (RUL) & Prognostics.
Tests state machine exhaustiveness, boundary conditions, Phase 10 handoff safety,
weakest-link redlines, sensor isolation, Theil-Sen estimation, and NASA PHM08 metrics.
"""

import math
import numpy as np
import pytest

from health_index.schema import HealthIndexResult, HealthState, DegradationTrend, HealthDataQuality
from forecasting.schema import ForecastResult, ForecastQuality, ModelStatus
from prognostics.schema import (
    RULStatus,
    EOLCriterion,
    EOLCriteriaConfig,
    RULConfig,
    RULResult,
)
from prognostics.threshold import WeakestLinkEOLEvaluator
from prognostics.trajectory import TheilSenExtrapolator, DualHorizonSynthesizer
from prognostics.uncertainty import MonteCarloTrajectoryPropagator
from prognostics.pipeline import RULPipeline
from prognostics.evaluation import (
    compute_phm08_score,
    compute_picp_and_mpiw,
    compute_prognostic_metrics,
)


def make_health_result(
    timestamp: float,
    hi_smooth: float,
    engine_id: str = "ENG_01",
    mission_id: str = "MSN_01",
    excluded_channels=None,
    valid_channels=None,
    mission_phase: str = "CRUISE",
) -> HealthIndexResult:
    """Helper to construct valid HealthIndexResult fixtures."""
    all_ch = ["rpm", "cht", "egt", "oil_temp", "oil_pressure", "fuel_flow", "vibration"]
    v_ch = valid_channels if valid_channels is not None else list(all_ch)
    ex_ch = excluded_channels if excluded_channels is not None else []
    return HealthIndexResult(
        timestamp=timestamp,
        engine_id=engine_id,
        mission_id=mission_id,
        mission_phase=mission_phase,
        raw_health_index=hi_smooth,
        smoothed_health_index=hi_smooth,
        health_state=HealthState.HEALTHY.value if hi_smooth >= 0.85 else HealthState.DEGRADED.value,
        raw_degradation_score=1.0 - hi_smooth,
        degradation_rate=0.0,
        degradation_trend=DegradationTrend.STABLE.value,
        channel_contributions={},
        channel_degradation_evidence={},
        dominant_degraded_channels=[],
        valid_channels=v_ch,
        missing_channels=[],
        excluded_channels=ex_ch,
        effective_channel_weights={},
        data_quality=HealthDataQuality.VALID.value,
    )


def prime_pipeline_history(
    pipeline: RULPipeline,
    engine_id: str,
    mission_id: str,
    duration_s: float,
    start_hi: float,
    end_hi: float,
    n_points: int = 40,
    excluded_channels=None,
    current_telemetry=None,
) -> RULResult:
    """Helper to feed monotonic synthetic sequence up to duration_s."""
    ts = np.linspace(0.0, duration_s, n_points)
    his = np.linspace(start_hi, end_hi, n_points)
    last_res = None
    for t, h in zip(ts, his):
        hr = make_health_result(
            timestamp=float(t),
            hi_smooth=float(h),
            engine_id=engine_id,
            mission_id=mission_id,
            excluded_channels=excluded_channels,
        )
        last_res = pipeline.process_assessment(hr, current_telemetry=current_telemetry)
    return last_res


# =========================================================================
# 1. PHASE 10 HANDOFF SAFETY TESTS
# =========================================================================

def make_forecast_result(
    model_name: str,
    model_status: str,
    horizon: int = 16,
    quality: str = ForecastQuality.VALID.value,
    predicted_telemetry=None,
    projected_hi=None,
) -> ForecastResult:
    """Helper to build ForecastResult matching Phase 10 schema."""
    pred_tel = predicted_telemetry if predicted_telemetry is not None else {"cht": [120.0] * horizon}
    return ForecastResult(
        engine_id="ENG_01",
        mission_id="MSN_01",
        forecast_start_timestamp=33.0,
        context_start_timestamp=1.0,
        context_length=32,
        forecast_horizon=horizon,
        sampling_interval=1.0,
        target_channels=list(pred_tel.keys()),
        forecast_timestamps=[33.0 + i for i in range(horizon)],
        predicted_telemetry=pred_tel,
        projected_health_trajectory=projected_hi,
        forecast_quality=quality,
        model_name=model_name,
        model_status=model_status,
    )


def test_phase10_handoff_rejects_uncheckpointed_graph():
    """Verify LOCAL_UNCHECKPOINTED_GRAPH is rejected and falls back to Theil-Sen (H=0)."""
    synth = DualHorizonSynthesizer()
    forecast = make_forecast_result(
        model_name="timesfm-3.0",
        model_status=ModelStatus.LOCAL_UNCHECKPOINTED_GRAPH.value,
        horizon=16,
    )
    H, traj_type = synth.evaluate_phase10_handoff(forecast)
    assert H == 0.0
    assert traj_type == "ROBUST_LINEAR_PRIMARY"


def test_phase10_handoff_rejects_blocked_gated():
    """Verify BLOCKED_UNAUTHENTICATED_GATED is rejected and falls back to Theil-Sen (H=0)."""
    synth = DualHorizonSynthesizer()
    forecast = make_forecast_result(
        model_name="timesfm-3.0",
        model_status=ModelStatus.BLOCKED_UNAUTHENTICATED_GATED.value,
        horizon=16,
        quality=ForecastQuality.DEGRADED_INPUT.value,
        predicted_telemetry={},
    )
    H, traj_type = synth.evaluate_phase10_handoff(forecast)
    assert H == 0.0
    assert traj_type == "ROBUST_LINEAR_PRIMARY"


def test_phase10_handoff_accepts_loaded_pretrained():
    """Verify LOADED_PRETRAINED with valid horizon is accepted (H=16 or 32)."""
    synth = DualHorizonSynthesizer()
    forecast = make_forecast_result(
        model_name="timesfm-3.0",
        model_status=ModelStatus.LOADED_PRETRAINED.value,
        horizon=16,
    )
    H, traj_type = synth.evaluate_phase10_handoff(forecast)
    assert H == 16.0
    assert traj_type == "TIMESFM_FORECAST_ASSISTED"


def test_phase10_handoff_baseline_mode_labeled_correctly():
    """Verify BASELINE is usable only when permitted, and explicitly labeled BASELINE_EWMA_ASSISTED."""
    synth = DualHorizonSynthesizer()
    forecast = make_forecast_result(
        model_name="baseline_ewma",
        model_status=ModelStatus.BASELINE.value,
        horizon=32,
    )
    # Default allows baseline
    H, traj_type = synth.evaluate_phase10_handoff(forecast, allow_baseline=True)
    assert H == 32.0
    assert traj_type == "BASELINE_EWMA_ASSISTED"

    # Disallowing baseline falls back to 0.0
    H_disallowed, traj_disallowed = synth.evaluate_phase10_handoff(forecast, allow_baseline=False)
    assert H_disallowed == 0.0
    assert traj_disallowed == "ROBUST_LINEAR_PRIMARY"


# =========================================================================
# 2. STATE MACHINE EXHAUSTIVENESS & BOUNDARY TESTS
# =========================================================================

def test_state_machine_boundary_hi_035():
    """Verify HI = 0.35 boundary condition maps to CRITICAL_EOL_REACHED (RUL = 0.0s)."""
    pipe = RULPipeline()
    # Feed warmup history with slope
    prime_pipeline_history(pipe, "ENG_01", "MSN_01", duration_s=40.0, start_hi=0.50, end_hi=0.36, n_points=40)

    # Exactly at boundary HI = 0.35
    hr = make_health_result(timestamp=41.0, hi_smooth=0.35, engine_id="ENG_01", mission_id="MSN_01")
    res = pipe.process_assessment(hr)
    assert res.status == RULStatus.CRITICAL_EOL_REACHED
    assert res.rul_seconds_median == 0.0
    assert res.limiting_factor == "GLOBAL_HEALTH_INDEX"


def test_state_machine_boundary_hi_085():
    """Verify HI = 0.85 boundary under various slopes."""
    # Sub-case A: dHI/dt = +0.0005 at HI = 0.85 -> NOT_DEGRADING (since -0.001 <= slope <= +0.0005)
    pipe_a = RULPipeline()
    res_a = prime_pipeline_history(pipe_a, "ENG_01", "MSN_A", duration_s=40.0, start_hi=0.83, end_hi=0.85, n_points=41)
    assert res_a.status == RULStatus.NOT_DEGRADING

    # Sub-case B: dHI/dt = -0.0005 at HI = 0.85 -> NOT_DEGRADING (since -0.001 <= slope <= +0.0005)
    pipe_b = RULPipeline()
    res_b = prime_pipeline_history(pipe_b, "ENG_01", "MSN_B", duration_s=40.0, start_hi=0.87, end_hi=0.85, n_points=41)
    assert res_b.status == RULStatus.NOT_DEGRADING

    # Sub-case C: dHI/dt = -0.001 at HI = 0.85 -> NOT_DEGRADING (boundary included in [-0.001, +0.0005])
    pipe_c = RULPipeline()
    res_c = prime_pipeline_history(pipe_c, "ENG_01", "MSN_C", duration_s=40.0, start_hi=0.89, end_hi=0.85, n_points=41)
    assert res_c.status == RULStatus.NOT_DEGRADING

    # Sub-case D: dHI/dt = -0.002 at HI = 0.85 -> ACTIVE_DEGRADATION (acute drop < -0.001)
    pipe_d = RULPipeline()
    res_d = prime_pipeline_history(pipe_d, "ENG_01", "MSN_D", duration_s=40.0, start_hi=0.93, end_hi=0.85, n_points=41)
    assert res_d.status == RULStatus.ACTIVE_DEGRADATION
    assert res_d.rul_seconds_median is not None
    assert res_d.rul_seconds_median > 0.0


def test_state_machine_boundary_dhi_pos_0005():
    """Verify dHI/dt = +0.0005 boundary: NOT_DEGRADING at HI >= 0.85, INDETERMINATE_TREND at degraded health."""
    # At HI = 0.60: slope +0.0005 -> INDETERMINATE_TREND (since -0.0005 <= slope <= +0.0005)
    pipe = RULPipeline()
    res = prime_pipeline_history(pipe, "ENG_01", "MSN_01", duration_s=40.0, start_hi=0.58, end_hi=0.60, n_points=41)
    assert res.status == RULStatus.INDETERMINATE_TREND
    assert res.rul_seconds_median is None


def test_state_machine_boundary_dhi_neg_0005():
    """Verify dHI/dt = -0.0005 boundary: NOT_DEGRADING at HI >= 0.85, INDETERMINATE_TREND at degraded health."""
    # At HI = 0.60: slope -0.0005 -> INDETERMINATE_TREND (since -0.0005 <= slope <= +0.0005)
    pipe = RULPipeline()
    res = prime_pipeline_history(pipe, "ENG_01", "MSN_01", duration_s=40.0, start_hi=0.62, end_hi=0.60, n_points=41)
    assert res.status == RULStatus.INDETERMINATE_TREND
    assert res.rul_seconds_median is None


def test_state_machine_boundary_dhi_neg_001():
    """Verify dHI/dt = -0.001 boundary: NOT_DEGRADING at HI >= 0.85, ACTIVE_DEGRADATION at degraded health."""
    # At HI = 0.60: slope -0.001 < -0.0005 -> ACTIVE_DEGRADATION
    pipe = RULPipeline()
    res = prime_pipeline_history(pipe, "ENG_01", "MSN_01", duration_s=40.0, start_hi=0.64, end_hi=0.60, n_points=41)
    assert res.status == RULStatus.ACTIVE_DEGRADATION
    assert res.rul_seconds_median is not None
    # Slope is ~ -0.001 / s, HI drop to 0.35 is 0.60 - 0.35 = 0.25 -> ~250 s
    assert 200.0 < res.rul_seconds_median < 300.0


def test_state_machine_recovery_trigger():
    """Verify slope > +0.0005 triggers RECOVERING status."""
    pipe = RULPipeline()
    # Slope = +0.002 / s
    res = prime_pipeline_history(pipe, "ENG_01", "MSN_01", duration_s=40.0, start_hi=0.50, end_hi=0.58, n_points=41)
    assert res.status == RULStatus.RECOVERING
    assert res.rul_seconds_median is None


# =========================================================================
# 3. WEAKEST-LINK & SENSOR ISOLATION TESTS
# =========================================================================

def test_weakest_link_redline_immediate_breach():
    """Verify physical redlines (CHT >= 150, Oil Pressure <= 1.2, Oil Temp >= 140, Vib >= 3.5) trigger immediate EOL."""
    evaluator = WeakestLinkEOLEvaluator()

    # CHT breach
    breached, factor = evaluator.check_immediate_eol(0.80, {"cht": 150.5})
    assert breached is True
    assert factor == "REDLINE_CHT"

    # Oil pressure breach
    breached, factor = evaluator.check_immediate_eol(0.80, {"oil_pressure": 1.1})
    assert breached is True
    assert factor == "REDLINE_OIL_PRESSURE"

    # Oil temp breach
    breached, factor = evaluator.check_immediate_eol(0.80, {"oil_temp": 142.0})
    assert breached is True
    assert factor == "REDLINE_OIL_TEMP"

    # Vibration breach
    breached, factor = evaluator.check_immediate_eol(0.80, {"vibration": 3.6})
    assert breached is True
    assert factor == "REDLINE_VIBRATION"


def test_weakest_link_redline_sensor_isolation():
    """Verify isolated sensor channel is excluded from physical redlines."""
    evaluator = WeakestLinkEOLEvaluator()

    # CHT is isolated
    breached, factor = evaluator.check_immediate_eol(
        current_health_index=0.80,
        current_telemetry={"cht": 160.0, "oil_pressure": 4.0},
        excluded_channels={"cht"},
    )
    # Should NOT breach because CHT is isolated
    assert breached is False
    assert factor == "NONE"


def test_degraded_prognostic_confidence_discount():
    """Verify that isolated sensor fault switches status to DEGRADED_PROGNOSTIC with 20% discount."""
    pipe = RULPipeline()
    # Degrade with CHT isolated
    prime_pipeline_history(
        pipe, "ENG_01", "MSN_01", duration_s=40.0, start_hi=0.70, end_hi=0.60, n_points=41,
        excluded_channels=["cht"]
    )
    hr = make_health_result(
        timestamp=41.0, hi_smooth=0.60, engine_id="ENG_01", mission_id="MSN_01", excluded_channels=["cht"]
    )
    res = pipe.process_assessment(hr)
    assert res.status == RULStatus.DEGRADED_PROGNOSTIC
    assert res.rul_seconds_median is not None
    # Confidence must be discounted (at most 0.80)
    assert res.confidence_score <= 0.80


# =========================================================================
# 4. THEIL-SEN EXTRAPOLATOR TESTS
# =========================================================================

def test_theil_sen_actual_timestamps_and_gaps():
    """Verify Theil-Sen breaks history when timestamp gap > 5.0s and respects min samples."""
    extrapolator = TheilSenExtrapolator(window_s=60.0, min_samples=5, max_gap_s=5.0)

    # 1. Normal continuous sequence
    ts = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    hi = np.array([0.80, 0.79, 0.78, 0.77, 0.76, 0.75])
    slope, se = extrapolator.estimate_slope(ts, hi)
    assert math.isclose(slope, -0.01, rel_tol=1e-3)
    assert se > 0.0

    # 2. Large gap > 5.0s: ts = [1.0, 2.0, 3.0, 12.0, 13.0, 14.0]
    # Breaks after gap, leaving only 3 points (12.0, 13.0, 14.0) -> fewer than min_samples (5)
    ts_gap = np.array([1.0, 2.0, 3.0, 12.0, 13.0, 14.0])
    hi_gap = np.array([0.80, 0.79, 0.78, 0.77, 0.76, 0.75])
    slope_gap, se_gap = extrapolator.estimate_slope(ts_gap, hi_gap)
    assert np.isnan(slope_gap)
    assert np.isnan(se_gap)


# =========================================================================
# 5. UNCERTAINTY & MONTE CARLO CONVERGENCE TESTS
# =========================================================================

def test_monte_carlo_propagation_and_percentile_ordering():
    """Verify P05 <= P50 <= P95 and finite sample generation."""
    propagator = MonteCarloTrajectoryPropagator(seed=42)
    p50, p05, p95, samples = propagator.propagate_linear_trajectory(
        anchor_time=50.0,
        anchor_hi=0.70,
        median_slope=-0.001,
        slope_se=0.0001,
        current_time=50.0,
        m_samples=500,
    )
    assert p50 is not None and p05 is not None and p95 is not None
    assert p05 <= p50 <= p95
    # Theoretical median RUL = (0.70 - 0.35) / 0.001 = 350.0 s
    assert 300.0 < p50 < 400.0
    assert len(samples) == 500


def test_monte_carlo_convergence_check():
    """Verify Monte Carlo convergence across M in {100, 500, 1000}."""
    propagator = MonteCarloTrajectoryPropagator(seed=42)
    results = propagator.check_mc_convergence(
        anchor_time=50.0,
        anchor_hi=0.70,
        median_slope=-0.001,
        slope_se=0.0001,
        current_time=50.0,
        sample_sizes=(100, 500, 1000),
    )
    m100_p50 = results["M_100"]["p50"]
    m500_p50 = results["M_500"]["p50"]
    m1000_p50 = results["M_1000"]["p50"]
    # Verify estimates are within 5% of each other
    assert abs(m500_p50 - m100_p50) / m500_p50 < 0.05
    assert abs(m1000_p50 - m500_p50) / m1000_p50 < 0.05


# =========================================================================
# 6. NASA PHM08 & METRICS VERIFICATION TESTS
# =========================================================================

def test_nasa_phm08_exact_hand_calculated():
    """
    Verify exact hand-calculated NASA PHM08 asymmetric loss formula (Saxena et al., 2008):
    d = RUL_pred - RUL_true
    d = 0: exp(0) - 1 = 0.0
    d = -20: exp(20/13) - 1 = 3.656834...
    d = +20: exp(20/10) - 1 = exp(2) - 1 = 6.389056...
    d = +70: exp(70/10) - 1 = exp(7) - 1 = 1095.633...
    d = +700 (late in seconds): exp(700/10) - 1 = exp(70) - 1 ~ 2.515e30
    """
    y_pred = np.array([100.0, 80.0, 120.0, 170.0, 800.0])
    y_true = np.array([100.0, 100.0, 100.0, 100.0, 100.0])  # d = [0, -20, +20, +70, +700]

    score, penalties = compute_phm08_score(y_pred, y_true)

    # Individual penalties
    assert math.isclose(penalties[0], 0.0, abs_tol=1e-5)
    assert math.isclose(penalties[1], math.exp(20.0 / 13.0) - 1.0, rel_tol=1e-4)
    assert math.isclose(penalties[2], math.exp(2.0) - 1.0, rel_tol=1e-4)
    assert math.isclose(penalties[3], math.exp(7.0) - 1.0, rel_tol=1e-4)
    assert math.isclose(penalties[4], math.exp(70.0) - 1.0, rel_tol=1e-4)

    # Aggregated score is sum of penalties
    expected_sum = sum(penalties)
    assert math.isclose(score, expected_sum, rel_tol=1e-4)


def test_picp_and_mpiw_reporting():
    """Verify PICP calculation and MPIW reporting alongside PICP."""
    y_true = np.array([100.0, 150.0, 200.0, 250.0])
    y_p05 = np.array([90.0, 140.0, 210.0, 240.0])   # 200 is outside [210, ...]
    y_p95 = np.array([110.0, 160.0, 230.0, 260.0])

    picp, mpiw = compute_picp_and_mpiw(y_true, y_p05, y_p95)
    # 3 out of 4 inside: 100 in [90, 110], 150 in [140, 160], 250 in [240, 260] -> 75%
    assert math.isclose(picp, 0.75, abs_tol=1e-4)
    # Width is 20 for all 4 -> MPIW = 20.0
    assert math.isclose(mpiw, 20.0, abs_tol=1e-4)

    # Full metric dictionary
    metrics = compute_prognostic_metrics(y_true, np.array([100.0, 150.0, 200.0, 250.0]), y_p05, y_p95)
    assert metrics["picp"] == 0.75
    assert metrics["picp_meets_target"] is False  # 75% < 90%
    assert metrics["mpiw_s"] == 20.0


# =========================================================================
# 7. MULTI-ENGINE / MULTI-MISSION ISOLATION TESTS
# =========================================================================

def test_engine_mission_isolation():
    """Verify state is completely isolated across (engine_id, mission_id) pairs."""
    pipe = RULPipeline()

    # Engine 1 degrading
    prime_pipeline_history(pipe, "ENG_01", "MSN_01", duration_s=40.0, start_hi=0.70, end_hi=0.55, n_points=41)
    hr_1 = make_health_result(timestamp=41.0, hi_smooth=0.55, engine_id="ENG_01", mission_id="MSN_01")
    res_1 = pipe.process_assessment(hr_1)
    assert res_1.status == RULStatus.ACTIVE_DEGRADATION

    # Engine 2 healthy cruise
    prime_pipeline_history(pipe, "ENG_02", "MSN_01", duration_s=40.0, start_hi=0.92, end_hi=0.92, n_points=41)
    hr_2 = make_health_result(timestamp=41.0, hi_smooth=0.92, engine_id="ENG_02", mission_id="MSN_01")
    res_2 = pipe.process_assessment(hr_2)
    assert res_2.status == RULStatus.NOT_DEGRADING

    # Reset only Engine 1
    pipe.reset(engine_id="ENG_01", mission_id="MSN_01")
    hr_1_reset = make_health_result(timestamp=1.0, hi_smooth=0.55, engine_id="ENG_01", mission_id="MSN_01")
    res_1_reset = pipe.process_assessment(hr_1_reset)
    assert res_1_reset.status == RULStatus.INSUFFICIENT_HISTORY

    # Engine 2 should remain unaffected
    hr_2_cont = make_health_result(timestamp=42.0, hi_smooth=0.92, engine_id="ENG_02", mission_id="MSN_01")
    res_2_cont = pipe.process_assessment(hr_2_cont)
    assert res_2_cont.status == RULStatus.NOT_DEGRADING
