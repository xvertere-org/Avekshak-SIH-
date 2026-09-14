"""
Tests for mathematical derivation of projected Health Index from forecasted telemetry
and forecast-assisted Remaining Useful Life (RUL) dynamics.
"""

import math
from typing import Dict, List
import numpy as np
import pandas as pd
import pytest

from forecasting.schema import (
    ForecastingConfig,
    ForecastResult,
    ModelStatus,
    ForecastQuality,
    DEFAULT_NOMINAL_EXPECTED,
)
from forecasting.pipeline import ForecastingPipeline
from prognostics.trajectory import DualHorizonSynthesizer
from prognostics.threshold import WeakestLinkEOLEvaluator
from prognostics.pipeline import RULPipeline
from prognostics.schema import RULConfig, RULStatus
from orchestrator.pipeline import SystemPipelineOrchestrator
from orchestrator.schema import OrchestratorConfig
from dashboard.services.adapter import DashboardAdapter


def _create_feeding_records(n_steps: int = 35, **sensor_overrides) -> List[Dict[str, float]]:
    """Helper to generate contiguous telemetry records for buffer filling."""
    base = {
        "engine_id": "ENG_001",
        "mission_id": "M_TEST",
        "rpm": 5000.0,
        "cht": 100.0,
        "egt": 680.0,
        "oil_temp": 85.0,
        "oil_pressure": 4.5,
        "fuel_flow": 18.0,
        "vibration": 0.5,
    }
    base.update(sensor_overrides)
    records = []
    for i in range(n_steps):
        rec = dict(base)
        rec["timestamp"] = float(i)
        records.append(rec)
    return records


def test_forecasted_health_not_hardcoded():
    """Verify that projected health trajectory is mathematically derived, not hardcoded 1.0."""
    pipe = ForecastingPipeline(force_local_graph=True)
    records = _create_feeding_records(35, cht=135.0, oil_temp=120.0, oil_pressure=2.0)
    for rec in records:
        res = pipe.process_sample(rec)

    assert res is not None
    assert res.projected_health_trajectory is not None
    assert len(res.projected_health_trajectory) == res.forecast_horizon

    # Must NOT be all 1.0 (hardcoded placeholder)
    assert not all(v == 1.0 for v in res.projected_health_trajectory)
    # Severely degraded sensor inputs must yield a degraded health index
    assert res.projected_health_trajectory[-1] < 0.80


def test_forecasted_sensor_changes_affect_health_trajectory():
    """Verify that varying sensor degradation levels produces monotonic changes in projected HI."""
    pipe_mild = ForecastingPipeline(force_local_graph=True)
    pipe_severe = ForecastingPipeline(force_local_graph=True)

    # Mild: CHT = 112°C (+1.2 sigma)
    for rec in _create_feeding_records(35, cht=112.0):
        res_mild = pipe_mild.process_sample(rec)

    # Severe: CHT = 140°C (+4.0 sigma)
    for rec in _create_feeding_records(35, cht=140.0):
        res_severe = pipe_severe.process_sample(rec)

    assert res_mild is not None and res_severe is not None
    hi_mild = res_mild.projected_health_trajectory[-1]
    hi_severe = res_severe.projected_health_trajectory[-1]

    # Severe degradation must produce strictly lower health than mild degradation
    assert hi_severe < hi_mild


def test_degrading_forecast_produces_strictly_lower_rul_than_healthy():
    """A degrading forecast must produce a lower RUL than a healthy forecast under identical history."""
    synth = DualHorizonSynthesizer(RULConfig())
    history_ts = np.arange(30, dtype=float)
    history_hi = np.linspace(0.85, 0.55, 30)

    # Forecast 1: Healthy projected health
    fc_healthy = ForecastResult(
        engine_id="ENG_001", mission_id="M1",
        forecast_start_timestamp=30.0, context_start_timestamp=0.0,
        context_length=30, forecast_horizon=16, sampling_interval=1.0,
        target_channels=["cht"],
        forecast_timestamps=[30.0 + i for i in range(1, 17)],
        predicted_telemetry={"cht": [100.0] * 16},
        projected_health_trajectory=[0.55] * 16,
        model_status=ModelStatus.BASELINE.value,
    )

    # Forecast 2: Degraded projected health (dropping to 0.40 at t+16)
    fc_degraded = ForecastResult(
        engine_id="ENG_001", mission_id="M1",
        forecast_start_timestamp=30.0, context_start_timestamp=0.0,
        context_length=30, forecast_horizon=16, sampling_interval=1.0,
        target_channels=["cht"],
        forecast_timestamps=[30.0 + i for i in range(1, 17)],
        predicted_telemetry={"cht": [140.0] * 16},
        projected_health_trajectory=list(np.linspace(0.55, 0.40, 16)),
        model_status=ModelStatus.BASELINE.value,
    )

    res_h = synth.synthesize_trajectory(
        current_time=30.0, current_hi=0.55,
        history_timestamps=history_ts, history_hi=history_hi,
        forecast=fc_healthy,
    )
    res_d = synth.synthesize_trajectory(
        current_time=30.0, current_hi=0.55,
        history_timestamps=history_ts, history_hi=history_hi,
        forecast=fc_degraded,
    )

    rul_h = synth.solve_linear_crossing(
        anchor_time=res_h["anchor_time"], anchor_hi=res_h["anchor_hi"],
        slope=res_h["theil_sen_slope"], target_hi=0.35, current_time=30.0,
    )
    rul_d = synth.solve_linear_crossing(
        anchor_time=res_d["anchor_time"], anchor_hi=res_d["anchor_hi"],
        slope=res_d["theil_sen_slope"], target_hi=0.35, current_time=30.0,
    )

    assert res_d["anchor_hi"] < res_h["anchor_hi"]
    assert rul_d is not None and rul_h is not None
    assert rul_d < rul_h


def test_current_health_index_continuity_preservation():
    """Verify that when current HI is 0.55, projected health transitions smoothly from 0.55."""
    pipe = ForecastingPipeline(force_local_graph=True)
    records = _create_feeding_records(35, cht=125.0)
    for rec in records[:-1]:
        pipe.process_sample(rec)

    # Pass current_health_index = 0.55 in optional context
    res = pipe.process_sample(
        records[-1],
        optional_context={"current_health_index": 0.55}
    )

    assert res is not None
    traj = res.projected_health_trajectory
    assert traj is not None
    # Step 0 must be contiguous with 0.55 (within EWMA alpha transition), NOT jumping to 1.0
    assert abs(traj[0] - 0.55) < 0.10
    assert traj[0] < 0.70


def test_healthy_forecast_behavior():
    """Healthy nominal forecast keeps projected HI at ~1.0."""
    pipe = ForecastingPipeline(force_local_graph=True)
    for rec in _create_feeding_records(35):
        res = pipe.process_sample(rec)

    assert res is not None
    for val in res.projected_health_trajectory:
        assert val >= 0.99


def test_slowly_vs_rapidly_degrading_forecast():
    """Rapidly degrading forecast produces lower end-of-horizon health than slowly degrading forecast."""
    pipe_slow = ForecastingPipeline(force_local_graph=True)
    pipe_fast = ForecastingPipeline(force_local_graph=True)

    # Slow drift
    for rec in _create_feeding_records(35, cht=110.0, oil_temp=95.0):
        res_slow = pipe_slow.process_sample(rec)

    # Fast acute spike
    for rec in _create_feeding_records(35, cht=145.0, oil_temp=128.0, vibration=2.8):
        res_fast = pipe_fast.process_sample(rec)

    assert res_slow.projected_health_trajectory[-1] > res_fast.projected_health_trajectory[-1]


def test_constant_forecast_stability():
    """Constant telemetry input yields stable, strictly monotonic convergence towards steady state."""
    pipe = ForecastingPipeline(force_local_graph=True)
    for rec in _create_feeding_records(35, cht=120.0):
        res = pipe.process_sample(rec)

    traj = res.projected_health_trajectory
    assert len(traj) == 16
    # Differences must be unidirectional (monotonic smoothing)
    diffs = np.diff(traj)
    assert np.all(diffs <= 1e-4)


def test_noisy_forecast_smoothing():
    """EWMA smoothing filters high-frequency predicted sensor noise without numerical instability."""
    pipe = ForecastingPipeline(force_local_graph=True)
    rng = np.random.default_rng(42)
    records = []
    for i in range(35):
        noise = rng.normal(0.0, 5.0)
        rec = {
            "timestamp": float(i),
            "engine_id": "ENG_001",
            "mission_id": "M_NOISE",
            "rpm": 5000.0,
            "cht": 110.0 + noise,
            "egt": 680.0,
            "oil_temp": 85.0,
            "oil_pressure": 4.5,
            "fuel_flow": 18.0,
            "vibration": 0.5,
        }
        records.append(rec)

    for rec in records:
        res = pipe.process_sample(rec)

    assert res is not None
    traj = res.projected_health_trajectory
    assert all(0.0 <= v <= 1.0 for v in traj)


def test_adversarial_nan_forecast_handling():
    """NaN telemetry in forecast does not crash and preserves valid channel state."""
    pipe = ForecastingPipeline(force_local_graph=True)
    records = _create_feeding_records(35)
    for rec in records:
        res = pipe.process_sample(rec)
    assert res is not None

    # Test projected health trajectory computation when predicted telemetry contains NaNs
    pred_nan = {ch: [float("nan")] * 16 for ch in DEFAULT_NOMINAL_EXPECTED}
    traj = pipe._compute_projected_health_trajectory(
        predicted_telemetry=pred_nan,
        forecast_timestamps=res.forecast_timestamps,
        optional_context={"current_health_index": 0.55},
    )
    assert traj is not None
    assert len(traj) == 16
    assert all(np.isfinite(v) for v in traj)


def test_adversarial_inf_forecast_handling():
    """Inf telemetry in forecast is treated as invalid and handled without crash."""
    pipe = ForecastingPipeline(force_local_graph=True)
    records = _create_feeding_records(35)
    for rec in records:
        res = pipe.process_sample(rec)
    assert res is not None

    pred_inf = {ch: [float("inf")] * 16 for ch in DEFAULT_NOMINAL_EXPECTED}
    traj = pipe._compute_projected_health_trajectory(
        predicted_telemetry=pred_inf,
        forecast_timestamps=res.forecast_timestamps,
        optional_context={"current_health_index": 0.55},
    )
    assert traj is not None
    assert len(traj) == 16
    assert all(np.isfinite(v) for v in traj)


def test_missing_forecast_channels_renormalization():
    """When non-critical channels are missing, health calculator renormalizes active weights."""
    pipe = ForecastingPipeline(force_local_graph=True)
    records = _create_feeding_records(35)
    for rec in records:
        res = pipe.process_sample(rec)
    assert res is not None

    # Omit vibration and fuel_flow from predicted telemetry
    pred_subset = {ch: list(res.predicted_telemetry[ch]) for ch in ["rpm", "cht", "egt", "oil_temp", "oil_pressure"]}
    traj = pipe._compute_projected_health_trajectory(
        predicted_telemetry=pred_subset,
        forecast_timestamps=res.forecast_timestamps,
        optional_context={"current_health_index": 0.60},
    )
    assert traj is not None
    assert len(traj) == 16
    assert all(0.0 <= v <= 1.0 for v in traj)


def test_forecast_ending_before_required_horizon():
    """Forecast with horizon not in {16, 32} is rejected from influencing RUL (H=0)."""
    synth = DualHorizonSynthesizer(RULConfig())
    forecast_short = ForecastResult(
        engine_id="ENG_001", mission_id="M1",
        forecast_start_timestamp=30.0, context_start_timestamp=0.0,
        context_length=30, forecast_horizon=8, sampling_interval=1.0,
        target_channels=["cht"],
        forecast_timestamps=[31.0 + i for i in range(8)],
        predicted_telemetry={"cht": [120.0] * 8},
        projected_health_trajectory=[0.7] * 8,
        model_status=ModelStatus.LOADED_PRETRAINED.value,
    )
    H, traj_type = synth.evaluate_phase10_handoff(forecast_short)
    assert H == 0.0
    assert traj_type == "ROBUST_LINEAR_PRIMARY"


def test_forecast_assisted_mode_disabled():
    """When enable_projected_health=False, projected health is None and trajectory falls back to slope."""
    cfg = ForecastingConfig(enable_projected_health=False)
    pipe = ForecastingPipeline(config=cfg, force_local_graph=True)
    records = _create_feeding_records(35)
    for rec in records:
        res = pipe.process_sample(rec)

    assert res.projected_health_trajectory is None


def test_timesfm_fallback_produces_derived_health():
    """Causal EWMA baseline produces valid mathematically derived health trajectory."""
    cfg = ForecastingConfig(enable_projected_health=True)
    pipe = ForecastingPipeline(config=cfg, force_local_graph=False)
    # In force_local_graph=False without token, timesfm runtime status is BLOCKED_UNAUTHENTICATED_GATED
    records = _create_feeding_records(35, cht=125.0)
    for rec in records:
        res = pipe.process_sample(rec)

    assert res.model_status in (
        ModelStatus.BLOCKED_UNAUTHENTICATED_GATED.value,
        ModelStatus.LOADED_PRETRAINED.value,
    )
    assert res.projected_health_trajectory is not None
    assert all(0.0 <= v <= 1.0 for v in res.projected_health_trajectory)


def test_uncertainty_propagation_with_derived_anchor():
    """Monte Carlo trajectory propagation preserves percentile ordering P05 <= P50 <= P95."""
    synth = DualHorizonSynthesizer(RULConfig())
    rul_pipe = RULPipeline(RULConfig())
    history_ts = np.arange(30, dtype=float)
    history_hi = np.linspace(0.85, 0.55, 30)

    fc = ForecastResult(
        engine_id="ENG_001", mission_id="M1",
        forecast_start_timestamp=30.0, context_start_timestamp=0.0,
        context_length=30, forecast_horizon=16, sampling_interval=1.0,
        target_channels=["cht"],
        forecast_timestamps=[31.0 + i for i in range(16)],
        predicted_telemetry={"cht": [130.0] * 16},
        projected_health_trajectory=list(np.linspace(0.55, 0.45, 16)),
        model_status=ModelStatus.BASELINE.value,
    )

    traj_dict = synth.synthesize_trajectory(
        current_time=30.0, current_hi=0.55,
        history_timestamps=history_ts, history_hi=history_hi,
        forecast=fc,
    )

    p50, p05, p95, _ = rul_pipe.mc_propagator.propagate_linear_trajectory(
        anchor_time=traj_dict["anchor_time"],
        anchor_hi=traj_dict["anchor_hi"],
        median_slope=traj_dict["theil_sen_slope"],
        slope_se=traj_dict["slope_std_error"],
        current_time=30.0,
    )

    assert p05 is not None and p50 is not None and p95 is not None
    assert p05 <= p50 <= p95


def test_orchestrator_dashboard_payload_receives_projected_health():
    """Verify that the end-to-end SystemPipelineOrchestrator wires projected health to DashboardStatePayload."""
    orch = SystemPipelineOrchestrator(
        config=OrchestratorConfig(auto_bootstrap_on_init=True, deterministic_seed=42)
    )
    records = _create_feeding_records(35, cht=120.0)
    last_payload = None
    for rec in records:
        last_payload = orch.step(rec)

    assert last_payload is not None
    # DashboardStatePayload must contain projected_health_trajectory
    assert hasattr(last_payload, "projected_health_trajectory")
    assert last_payload.projected_health_trajectory is not None
    assert len(last_payload.projected_health_trajectory) == 16

    # DashboardAdapter must adapt it into PrognosticsViewModel
    adapter = DashboardAdapter()
    vm = adapter.adapt(last_payload)
    assert vm.prognostics.projected_health_trajectory is not None
    assert len(vm.prognostics.projected_health_trajectory) == 16


def test_weakest_link_redline_breach_detection_in_forecast_horizon():
    """A severe forecast trajectory crossing HI <= 0.35 inside the 16s horizon triggers immediate EOL."""
    evaluator = WeakestLinkEOLEvaluator()
    future_ts = np.array([31.0 + i for i in range(16)], dtype=float)
    # Crosses 0.35 at index 8 (t = 39.0s)
    future_hi = np.array([0.50 - 0.02 * i for i in range(16)], dtype=float)

    rul, factor = evaluator.find_earliest_trajectory_crossing(
        future_timestamps=future_ts,
        future_health_index=future_hi,
        current_time=30.0,
    )

    assert rul is not None
    assert factor == "GLOBAL_HEALTH_INDEX"
    # Crossing at index 8: t = 39.0s -> RUL = 39.0 - 30.0 = 9.0s
    assert abs(rul - 9.0) < 1e-4
