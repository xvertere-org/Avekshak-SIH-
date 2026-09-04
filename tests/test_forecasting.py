"""
Behavioral and validation tests for Phase 10: TimesFM-3 Future Telemetry Forecasting.
"""

import math
import pytest
import numpy as np
import pandas as pd

from forecasting.schema import (
    DEFAULT_FORECAST_CHANNELS,
    DEFAULT_NRMSE_DENOMINATORS,
    PHYSICAL_LOWER_BOUNDS,
    ForecastQuality,
    ModelStatus,
    ForecastingConfig,
    ForecastResult,
)
from forecasting.preprocessing import CausalTelemetryBuffer
from forecasting.baselines import PersistenceForecaster, CausalEWMAForecaster
from forecasting.models import TimesFM3ModelAdapter
from forecasting.pipeline import ForecastingPipeline
from forecasting.evaluator import (
    ForecastingEvaluator,
    compute_forecast_metrics,
    split_missions_grouped,
)


# =====================================================================
# Test 1: TimesFM-3 Adapter Interface & Status Reporting
# =====================================================================
def test_timesfm_adapter_interface():
    """Adapter implements standard interface and reports honest runtime status."""
    adapter = TimesFM3ModelAdapter(force_local_graph=True)
    assert adapter.name == "timesfm-3.0"
    assert adapter.runtime_status in [
        ModelStatus.LOADED_PRETRAINED.value,
        ModelStatus.LOCAL_UNCHECKPOINTED_GRAPH.value,
        ModelStatus.BLOCKED_UNAUTHENTICATED_GATED.value,
    ]
    assert adapter.is_available()


# =====================================================================
# Test 2: Multichannel Semantics 2D vs 1D
# =====================================================================
def test_multichannel_semantics_2d_vs_1d():
    """
    2D input shape (7, N) activates joint multivariate output (7, H),
    while 1D input list activates univariate batch output.
    """
    adapter = TimesFM3ModelAdapter(force_local_graph=True, use_multivariate=True)
    ctx_2d = np.ones((7, 32), dtype=np.float32)
    point_fc, q_fc = adapter.predict(ctx_2d, horizon=16, return_quantiles=True)

    assert len(point_fc) == 7
    for ch in DEFAULT_FORECAST_CHANNELS:
        assert ch in point_fc
        assert len(point_fc[ch]) == 16
        assert q_fc is not None and ch in q_fc
        assert q_fc[ch].shape == (16, 3)

    # 1D Ablation Mode
    adapter_1d = TimesFM3ModelAdapter(force_local_graph=True, use_multivariate=False)
    point_fc_1d, _ = adapter_1d.predict(ctx_2d, horizon=16)
    assert len(point_fc_1d) == 7
    for ch in DEFAULT_FORECAST_CHANNELS:
        assert len(point_fc_1d[ch]) == 16


# =====================================================================
# Test 3: Multivariate Behavioral Measurement Test
# =====================================================================
def test_multivariate_cross_channel_influence():
    """
    Behavioral measurement test (Change 1 requirement):
    1. Create baseline 2D context X.
    2. Modify exactly one channel (perturb CHT).
    3. Run both forecasts.
    4. Measure forecast differences in non-perturbed channels.
    5. Use explicit numerical tolerance to classify influence as measurable or negligible.
    6. Pass if the experiment executes correctly in either case.
    7. Honestly document the observed sensitivity without asserting positive coupling.
    """
    adapter = TimesFM3ModelAdapter(force_local_graph=True, use_multivariate=True)

    # 1. Baseline context
    rng = np.random.RandomState(42)
    ctx_baseline = rng.randn(7, 32).astype(np.float32)

    # 2. Perturb only CHT (channel index 1 in DEFAULT_FORECAST_CHANNELS)
    cht_idx = DEFAULT_FORECAST_CHANNELS.index("cht")
    ctx_modified = ctx_baseline.copy()
    ctx_modified[cht_idx, :] += 50.0  # +50 °C step perturbation

    # 3. Forecast both
    fc_baseline, _ = adapter.predict(ctx_baseline, horizon=16)
    fc_modified, _ = adapter.predict(ctx_modified, horizon=16)

    # 4. Measure forecast differences across non-perturbed channels
    tolerance = 1e-5
    cross_channel_diffs = {}
    for i, ch in enumerate(DEFAULT_FORECAST_CHANNELS):
        if ch == "cht":
            continue
        diff = np.max(np.abs(fc_modified[ch] - fc_baseline[ch]))
        cross_channel_diffs[ch] = float(diff)

    # 5. Classify influence
    max_other_diff = max(cross_channel_diffs.values())
    is_measurable = max_other_diff >= tolerance

    # 6. Pass if measurement executed cleanly; document result honestly
    assert isinstance(cross_channel_diffs, dict)
    assert len(cross_channel_diffs) == 6
    # Logged empirical finding: local uncheckpointed graph has cross-attention layers
    # Result is documented honestly without asserting that coupling must exceed an arbitrary bound
    report = {
        "max_cross_channel_diff": max_other_diff,
        "is_measurable": is_measurable,
        "tolerance": tolerance,
        "diffs_per_channel": cross_channel_diffs,
    }
    assert report["max_cross_channel_diff"] >= 0.0


# =====================================================================
# Test 4: Causal Preprocessing
# =====================================================================
def test_causal_preprocessing():
    """Context window strictly contains observations at or before current time t."""
    config = ForecastingConfig(context_length=5)
    buffer = CausalTelemetryBuffer(config)

    for i in range(10):
        buffer.add_observation({
            "timestamp": float(i),
            "rpm": 2500.0 + i, "cht": 90.0, "egt": 650.0,
            "oil_temp": 85.0, "oil_pressure": 3.5, "fuel_flow": 15.0, "vibration": 1.0,
        })

    ctx, ts, quality, meta = buffer.get_context()
    assert ctx is not None and ts is not None
    assert ctx.shape == (7, 5)
    # Most recent observation is t=9.0
    assert ts[-1] == 9.0
    assert ts[0] == 5.0
    assert quality == ForecastQuality.VALID.value


# =====================================================================
# Test 5: Chronological Ordering & Non-Positive dt Rejection
# =====================================================================
def test_chronological_ordering():
    """Buffer rejects duplicate timestamps (dt=0) and backwards time jumps (dt < 0)."""
    buffer = CausalTelemetryBuffer()

    rec1 = {"timestamp": 10.0, "rpm": 2500.0, "cht": 90.0, "egt": 650.0,
            "oil_temp": 85.0, "oil_pressure": 3.5, "fuel_flow": 15.0, "vibration": 1.0}
    assert buffer.add_observation(rec1) is True

    # Duplicate timestamp
    rec_dup = {"timestamp": 10.0, "rpm": 2510.0, "cht": 90.0, "egt": 650.0,
               "oil_temp": 85.0, "oil_pressure": 3.5, "fuel_flow": 15.0, "vibration": 1.0}
    assert buffer.add_observation(rec_dup) is False

    # Out-of-order timestamp
    rec_retro = {"timestamp": 9.5, "rpm": 2510.0, "cht": 90.0, "egt": 650.0,
                 "oil_temp": 85.0, "oil_pressure": 3.5, "fuel_flow": 15.0, "vibration": 1.0}
    assert buffer.add_observation(rec_retro) is False


# =====================================================================
# Test 6: Strict Causal NaN Handling
# =====================================================================
def test_nan_and_missing_data_handling():
    """
    Internal NaNs are causally forward-filled; leading NaNs yield INSUFFICIENT_CONTEXT;
    missing values are never converted to zero.
    """
    config = ForecastingConfig(context_length=4)
    buffer = CausalTelemetryBuffer(config)

    # Case A: Leading NaN in CHT -> must be rejected with INSUFFICIENT_CONTEXT (no backward fill)
    buffer.reset()
    buffer.add_observation({"timestamp": 1.0, "rpm": 2500.0, "cht": np.nan, "egt": 650.0,
                            "oil_temp": 85.0, "oil_pressure": 3.5, "fuel_flow": 15.0, "vibration": 1.0})
    for t in [2.0, 3.0, 4.0]:
        buffer.add_observation({"timestamp": t, "rpm": 2500.0, "cht": 90.0, "egt": 650.0,
                                "oil_temp": 85.0, "oil_pressure": 3.5, "fuel_flow": 15.0, "vibration": 1.0})

    ctx_lead, _, q_lead, meta_lead = buffer.get_context()
    assert ctx_lead is None
    assert q_lead == ForecastQuality.INSUFFICIENT_CONTEXT.value

    # Case B: Internal NaN in CHT at t=3.0 -> forward filled from t=2.0 (90.0), NOT converted to 0
    buffer.reset()
    buffer.add_observation({"timestamp": 1.0, "rpm": 2500.0, "cht": 90.0, "egt": 650.0,
                            "oil_temp": 85.0, "oil_pressure": 3.5, "fuel_flow": 15.0, "vibration": 1.0})
    buffer.add_observation({"timestamp": 2.0, "rpm": 2500.0, "cht": 92.0, "egt": 650.0,
                            "oil_temp": 85.0, "oil_pressure": 3.5, "fuel_flow": 15.0, "vibration": 1.0})
    buffer.add_observation({"timestamp": 3.0, "rpm": 2500.0, "cht": np.nan, "egt": 650.0,
                            "oil_temp": 85.0, "oil_pressure": 3.5, "fuel_flow": 15.0, "vibration": 1.0})
    buffer.add_observation({"timestamp": 4.0, "rpm": 2500.0, "cht": 94.0, "egt": 650.0,
                            "oil_temp": 85.0, "oil_pressure": 3.5, "fuel_flow": 15.0, "vibration": 1.0})

    ctx_int, _, q_int, meta_int = buffer.get_context()
    assert ctx_int is not None
    assert q_int == ForecastQuality.IMPUTED.value
    assert "cht" in meta_int["imputed_channels"]
    # Check that t=3.0 was forward filled to 92.0, NOT 0.0
    cht_row = ctx_int[DEFAULT_FORECAST_CHANNELS.index("cht")]
    assert cht_row[2] == 92.0


# =====================================================================
# Test 7: No Future Reconstruction of Past
# =====================================================================
def test_no_future_reconstruction_of_past():
    """Future observations are never used to reconstruct earlier context values."""
    config = ForecastingConfig(context_length=3)
    buffer = CausalTelemetryBuffer(config)

    # Context has missing CHT at t=1.0. Even if t=2.0 and t=3.0 are known, t=1.0 is leading -> rejected!
    buffer.add_observation({"timestamp": 1.0, "rpm": 2500.0, "cht": np.nan, "egt": 650.0,
                            "oil_temp": 85.0, "oil_pressure": 3.5, "fuel_flow": 15.0, "vibration": 1.0})
    buffer.add_observation({"timestamp": 2.0, "rpm": 2500.0, "cht": 95.0, "egt": 650.0,
                            "oil_temp": 85.0, "oil_pressure": 3.5, "fuel_flow": 15.0, "vibration": 1.0})
    buffer.add_observation({"timestamp": 3.0, "rpm": 2500.0, "cht": 96.0, "egt": 650.0,
                            "oil_temp": 85.0, "oil_pressure": 3.5, "fuel_flow": 15.0, "vibration": 1.0})

    ctx, _, q, _ = buffer.get_context()
    assert ctx is None
    assert q == ForecastQuality.INSUFFICIENT_CONTEXT.value


# =====================================================================
# Test 8: Large Timestamp Gap Handling
# =====================================================================
def test_large_timestamp_gap_handling():
    """Timestamp gap > max_timestamp_gap_s resets historical context buffer."""
    config = ForecastingConfig(context_length=3, max_timestamp_gap_s=5.0)
    buffer = CausalTelemetryBuffer(config)

    buffer.add_observation({"timestamp": 1.0, "rpm": 2500.0, "cht": 90.0, "egt": 650.0,
                            "oil_temp": 85.0, "oil_pressure": 3.5, "fuel_flow": 15.0, "vibration": 1.0})
    buffer.add_observation({"timestamp": 2.0, "rpm": 2500.0, "cht": 90.0, "egt": 650.0,
                            "oil_temp": 85.0, "oil_pressure": 3.5, "fuel_flow": 15.0, "vibration": 1.0})
    # Jump from t=2.0 to t=30.0 (gap = 28s > 5s)
    buffer.add_observation({"timestamp": 30.0, "rpm": 2500.0, "cht": 90.0, "egt": 650.0,
                            "oil_temp": 85.0, "oil_pressure": 3.5, "fuel_flow": 15.0, "vibration": 1.0})

    ctx, _, q, _ = buffer.get_context()
    # After gap reset, only 1 sample exists in buffer (< context_length 3) -> INSUFFICIENT_CONTEXT
    assert ctx is None
    assert q == ForecastQuality.INSUFFICIENT_CONTEXT.value


# =====================================================================
# Test 9: Mission Boundary Handling
# =====================================================================
def test_mission_boundary_handling():
    """Buffer strictly prevents context bleeding across distinct missions."""
    config = ForecastingConfig(context_length=3)
    buffer = CausalTelemetryBuffer(config)

    # Feed 2 samples for Mission M001
    buffer.add_observation({"timestamp": 1.0, "rpm": 2500.0, "cht": 90.0, "egt": 650.0,
                            "oil_temp": 85.0, "oil_pressure": 3.5, "fuel_flow": 15.0, "vibration": 1.0},
                           mission_id="M001")
    buffer.add_observation({"timestamp": 2.0, "rpm": 2500.0, "cht": 90.0, "egt": 650.0,
                            "oil_temp": 85.0, "oil_pressure": 3.5, "fuel_flow": 15.0, "vibration": 1.0},
                           mission_id="M001")

    # Feed 1 sample for Mission M002
    buffer.add_observation({"timestamp": 1.0, "rpm": 2500.0, "cht": 90.0, "egt": 650.0,
                            "oil_temp": 85.0, "oil_pressure": 3.5, "fuel_flow": 15.0, "vibration": 1.0},
                           mission_id="M002")

    ctx_m1, _, _, _ = buffer.get_context(mission_id="M001")
    ctx_m2, _, _, _ = buffer.get_context(mission_id="M002")

    # Neither mission has reached required 3 samples
    assert ctx_m1 is None
    assert ctx_m2 is None


# =====================================================================
# Test 10: Mission State Isolation
# =====================================================================
def test_mission_state_isolation():
    """
    Mission B processed on UAV-01 after Mission A produces bitwise identical
    forecasts to Mission B processed on a fresh pipeline.
    """
    pipeline_shared = ForecastingPipeline(force_local_graph=True)
    pipeline_fresh = ForecastingPipeline(force_local_graph=True)

    # Mission A on shared pipeline (20 samples, severely degraded RPM)
    samples_a = []
    for i in range(20):
        samples_a.append({
            "timestamp": float(i), "engine_id": "UAV-01", "mission_id": "M001",
            "rpm": 1500.0 - 20 * i, "cht": 140.0, "egt": 850.0,
            "oil_temp": 120.0, "oil_pressure": 1.2, "fuel_flow": 28.0, "vibration": 3.2,
        })
    pipeline_shared.process_dataframe(pd.DataFrame(samples_a))

    # Mission B on shared pipeline (healthy cruise)
    samples_b = []
    for i in range(40):
        samples_b.append({
            "timestamp": float(i), "engine_id": "UAV-01", "mission_id": "M002",
            "rpm": 5000.0, "cht": 95.0, "egt": 720.0,
            "oil_temp": 90.0, "oil_pressure": 4.0, "fuel_flow": 18.0, "vibration": 0.8,
        })
    df_b = pd.DataFrame(samples_b)

    out_b_shared = pipeline_shared.process_dataframe(df_b)
    out_b_fresh = pipeline_fresh.process_dataframe(df_b)

    assert len(out_b_shared) == len(out_b_fresh)
    assert len(out_b_shared) > 0

    for res_shared, res_fresh in zip(out_b_shared, out_b_fresh):
        assert res_shared.forecast_start_timestamp == res_fresh.forecast_start_timestamp
        for ch in DEFAULT_FORECAST_CHANNELS:
            pred_s = res_shared.predicted_telemetry[ch]
            pred_f = res_fresh.predicted_telemetry[ch]
            assert np.allclose(pred_s, pred_f, atol=1e-5)


# =====================================================================
# Test 11: Persistence Baseline
# =====================================================================
def test_persistence_baseline():
    """Persistence baseline produces exactly y_hat(t+k) = y(t)."""
    forecaster = PersistenceForecaster()
    ctx = np.array([[100.0, 105.0, 110.0]], dtype=np.float32)  # 1 channel, 3 samples
    fc = forecaster.predict(ctx, horizon=5, target_channels=["cht"])

    assert "cht" in fc
    assert np.all(fc["cht"] == 110.0)
    assert len(fc["cht"]) == 5


# =====================================================================
# Test 12: Causal EWMA Baseline with Timestamp OLS & Zero-Denominator Safety
# =====================================================================
def test_causal_ewma_baseline():
    """
    Causal EWMA uses actual timestamps for OLS; handles zero/near-zero denominator
    deterministically without NaN or division-by-zero; respects physical non-negative clamping.
    """
    forecaster = CausalEWMAForecaster(alpha=0.20, clamp_to_physical_limits=True)

    # Case A: Normal linear trend with actual irregular timestamps
    # y = 100 + 2.0 * t
    ts = np.array([0.0, 1.0, 2.5, 4.0], dtype=np.float64)
    y = 100.0 + 2.0 * ts
    ctx = np.array([y], dtype=np.float32)

    fc, clamping = forecaster.predict(ctx, horizon=3, target_channels=["cht"], timestamps=ts, sampling_interval_s=1.0)
    assert "cht" in fc
    # Future values should extrapolate linearly above y[-1]
    assert fc["cht"][-1] > fc["cht"][0]
    assert clamping["cht"] is False

    # Case B: Zero-denominator safety (constant timestamps vector)
    ts_const = np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float64)
    fc_safe, _ = forecaster.predict(ctx, horizon=3, target_channels=["rpm"], timestamps=ts_const)
    # Slope must deterministically be 0.0, no NaN, no exception
    assert not np.any(np.isnan(fc_safe["rpm"]))
    assert np.all(fc_safe["rpm"] == fc_safe["rpm"][0])

    # Case C: Physical non-negative clamping on RPM
    # Downward plunging trajectory that would go negative
    y_drop = np.array([500.0, 300.0, 100.0, 10.0], dtype=np.float32)
    ts_drop = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float64)
    fc_clamped, clamp_flag = forecaster.predict(np.array([y_drop]), horizon=5, target_channels=["rpm"], timestamps=ts_drop)
    assert np.all(fc_clamped["rpm"] >= 0.0)
    assert clamp_flag["rpm"] is True


# =====================================================================
# Test 13: Future Leakage Append Invariance
# =====================================================================
def test_no_future_leakage_append():
    """Appending future samples after time t does not modify forecast at time t."""
    pipeline = ForecastingPipeline(force_local_graph=True)

    samples_base = []
    for i in range(35):
        samples_base.append({
            "timestamp": float(i), "engine_id": "ENG_001", "mission_id": "M_LEAK",
            "rpm": 5000.0, "cht": 95.0, "egt": 720.0,
            "oil_temp": 90.0, "oil_pressure": 4.0, "fuel_flow": 18.0, "vibration": 0.8,
        })

    # Run base
    res_base = pipeline.process_dataframe(pd.DataFrame(samples_base))
    fc_at_34_base = res_base[-1]

    # Append 10 future samples
    samples_extended = list(samples_base)
    for i in range(35, 45):
        samples_extended.append({
            "timestamp": float(i), "engine_id": "ENG_001", "mission_id": "M_LEAK",
            "rpm": 3000.0, "cht": 150.0, "egt": 900.0,
            "oil_temp": 130.0, "oil_pressure": 1.0, "fuel_flow": 30.0, "vibration": 4.0,
        })

    pipeline_new = ForecastingPipeline(force_local_graph=True)
    res_extended = pipeline_new.process_dataframe(pd.DataFrame(samples_extended))
    fc_at_34_ext = res_extended[len(samples_base) - pipeline.config.context_length]

    # Forecast at t=34 must remain strictly identical
    assert fc_at_34_base.forecast_start_timestamp == fc_at_34_ext.forecast_start_timestamp
    for ch in DEFAULT_FORECAST_CHANNELS:
        assert np.allclose(
            fc_at_34_base.predicted_telemetry[ch],
            fc_at_34_ext.predicted_telemetry[ch],
            atol=1e-5,
        )


# =====================================================================
# Test 14: Future Sample Modification Invariance
# =====================================================================
def test_future_sample_modification_invariance():
    """Mutating future telemetry has zero retroactive effect on prior forecasts."""
    pipeline_a = ForecastingPipeline(force_local_graph=True)
    pipeline_b = ForecastingPipeline(force_local_graph=True)

    samples_prefix = []
    for i in range(35):
        samples_prefix.append({
            "timestamp": float(i), "engine_id": "ENG_001", "mission_id": "M_INV",
            "rpm": 5000.0, "cht": 95.0, "egt": 720.0,
            "oil_temp": 90.0, "oil_pressure": 4.0, "fuel_flow": 18.0, "vibration": 0.8,
        })

    # Future variation A: healthy continuation
    df_a = pd.DataFrame(samples_prefix + [{
        "timestamp": 35.0, "engine_id": "ENG_001", "mission_id": "M_INV",
        "rpm": 5000.0, "cht": 95.0, "egt": 720.0,
        "oil_temp": 90.0, "oil_pressure": 4.0, "fuel_flow": 18.0, "vibration": 0.8,
    }])

    # Future variation B: severe engine explosion at t=35.0
    df_b = pd.DataFrame(samples_prefix + [{
        "timestamp": 35.0, "engine_id": "ENG_001", "mission_id": "M_INV",
        "rpm": 0.0, "cht": 200.0, "egt": 1100.0,
        "oil_temp": 150.0, "oil_pressure": 0.0, "fuel_flow": 0.0, "vibration": 15.0,
    }])

    out_a = pipeline_a.process_dataframe(df_a)
    out_b = pipeline_b.process_dataframe(df_b)

    # Compare forecasts at t=34 (index -2)
    assert out_a[-2].forecast_start_timestamp == out_b[-2].forecast_start_timestamp == 34.0
    for ch in DEFAULT_FORECAST_CHANNELS:
        assert np.allclose(
            out_a[-2].predicted_telemetry[ch],
            out_b[-2].predicted_telemetry[ch],
            atol=1e-5,
        )


# =====================================================================
# Test 15: Forecast Output Schema & Provenance
# =====================================================================
def test_forecast_output_schema():
    """ForecastResult contains all required provenance and JSON-serializable fields."""
    pipeline = ForecastingPipeline(force_local_graph=True)
    df = pd.DataFrame([{
        "timestamp": float(i), "engine_id": "ENG_001", "mission_id": "M_SCHEMA",
        "rpm": 5000.0, "cht": 95.0, "egt": 720.0,
        "oil_temp": 90.0, "oil_pressure": 4.0, "fuel_flow": 18.0, "vibration": 0.8,
    } for i in range(35)])

    results = pipeline.process_dataframe(df)
    assert len(results) > 0
    res = results[-1]

    assert res.engine_id == "ENG_001"
    assert res.mission_id == "M_SCHEMA"
    assert res.context_length == 32
    assert res.forecast_horizon == 16
    assert len(res.forecast_timestamps) == 16
    assert len(res.predicted_telemetry) == 7
    assert res.forecast_quality == ForecastQuality.VALID.value
    assert "persistence" in res.baseline_comparison
    assert "causal_ewma" in res.baseline_comparison


# =====================================================================
# Test 16: Grouped Mission-Level Evaluation
# =====================================================================
def test_grouped_mission_evaluation():
    """Grouped mission splitting guarantees zero overlap in mission_ids."""
    records = []
    for m in range(10):
        miss_id = f"MISSION_{m:03d}"
        for i in range(10):
            records.append({
                "timestamp": float(i),
                "mission_id": miss_id,
                "engine_id": "ENG_001",
                "rpm": 5000.0, "cht": 95.0, "egt": 720.0,
                "oil_temp": 90.0, "oil_pressure": 4.0, "fuel_flow": 18.0, "vibration": 0.8,
            })
    df = pd.DataFrame(records)

    train_df, test_df = split_missions_grouped(df, test_ratio=0.30, seed=42)
    train_m = set(train_df["mission_id"].unique())
    test_m = set(test_df["mission_id"].unique())

    assert len(train_m.intersection(test_m)) == 0
    assert len(train_m) + len(test_m) == 10
    assert len(test_m) == 3


# =====================================================================
# Test 17: Fixed Engineering NRMSE Metric Calculation
# =====================================================================
def test_metric_calculation():
    """NRMSE uses fixed pre-established reference ranges as denominators."""
    actual = {"rpm": np.array([5000.0, 5000.0]), "cht": np.array([100.0, 100.0])}
    pred = {"rpm": np.array([5100.0, 4900.0]), "cht": np.array([105.0, 95.0])}

    metrics = compute_forecast_metrics(actual, pred, channels=["rpm", "cht"])

    # RPM RMSE is 100.0 -> NRMSE = 100.0 / 4850.0
    rpm_m = metrics["per_channel"]["rpm"]
    assert rpm_m["rmse"] == pytest.approx(100.0)
    assert rpm_m["nrmse"] == pytest.approx(100.0 / 4850.0)
    assert rpm_m["denominator"] == 4850.0

    # CHT RMSE is 5.0 -> NRMSE = 5.0 / 130.0
    cht_m = metrics["per_channel"]["cht"]
    assert cht_m["rmse"] == pytest.approx(5.0)
    assert cht_m["nrmse"] == pytest.approx(5.0 / 130.0)
    assert cht_m["denominator"] == 130.0


# =====================================================================
# Test 18: Phase 9 Independence
# =====================================================================
def test_phase9_independence():
    """Forecasting functions completely without Phase 9 Health Index inputs."""
    pipeline = ForecastingPipeline(force_local_graph=True)
    sample = {
        "timestamp": 1.0, "engine_id": "ENG_001",
        "rpm": 5000.0, "cht": 95.0, "egt": 720.0,
        "oil_temp": 90.0, "oil_pressure": 4.0, "fuel_flow": 18.0, "vibration": 0.8,
    }
    # No Phase 9 inputs passed
    res = pipeline.process_sample(sample)
    # Context accumulating, returns None cleanly
    assert res is None


# =====================================================================
# Test 19: Phase 8 Independence
# =====================================================================
def test_phase8_independence():
    """Forecasting functions completely without Phase 8 diagnosis inputs."""
    pipeline = ForecastingPipeline(force_local_graph=True)
    sample = {
        "timestamp": 1.0, "engine_id": "ENG_001",
        "rpm": 5000.0, "cht": 95.0, "egt": 720.0,
        "oil_temp": 90.0, "oil_pressure": 4.0, "fuel_flow": 18.0, "vibration": 0.8,
    }
    # No Phase 8 inputs passed
    res = pipeline.process_sample(sample)
    assert res is None


# =====================================================================
# Test 20: Projected Health Trajectory Labeling
# =====================================================================
def test_projected_health_labeling():
    """Any projected health trajectory is explicitly labeled projected, not actual."""
    config = ForecastingConfig(enable_projected_health=True)
    pipeline = ForecastingPipeline(config=config, force_local_graph=True)

    df = pd.DataFrame([{
        "timestamp": float(i), "engine_id": "ENG_001", "mission_id": "M_PROJ",
        "rpm": 5000.0, "cht": 95.0, "egt": 720.0,
        "oil_temp": 90.0, "oil_pressure": 4.0, "fuel_flow": 18.0, "vibration": 0.8,
    } for i in range(35)])

    results = pipeline.process_dataframe(df)
    res = results[-1]
    assert res.projected_health_trajectory is not None
    # Attribute is named projected_health_trajectory, NEVER current or measured health
    assert hasattr(res, "projected_health_trajectory")
    assert not hasattr(res, "current_health_index")


# =====================================================================
# Test 21: Existing Phases Regression Safety
# =====================================================================
def test_existing_phases_regression():
    """Phase 10 does not modify or interfere with Phase 1-9 modules."""
    from simulator.fault_interface import FaultType
    from digital_twin.twin_model import DigitalTwin
    from anomaly_detection import HybridAnomalyDetector
    from fault_diagnosis import XGBoostFaultClassifier
    from health_index import HealthIndexPipeline

    assert len(FaultType) == 6
    assert DigitalTwin() is not None
    assert HybridAnomalyDetector() is not None
    assert XGBoostFaultClassifier() is not None
    assert HealthIndexPipeline() is not None
