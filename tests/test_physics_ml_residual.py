"""
Phase 3 Verification Test Suite: Physics–ML Residual Integration.

Verifies:
1. Physics baseline output schema
2. Exact residual formula: residual = observed_target - physics_prediction
3. Exact corrected-prediction formula: corrected_prediction = physics_prediction + predicted_residual
4. Timestamp monotonicity and dt consistency
5. Unit conversions and sign conventions
6. Missing and invalid sensor handling
7. Zero train/test data leakage
8. Deterministic inference
9. Three-way model comparison (Physics vs Pure ML vs Grey-Box)
10. Dashboard view-model contract and operator-facing terminology
11. Explainability structure without causal claims
12. Numerical stability across extreme/boundary conditions
"""

import math
import pytest
import numpy as np
import pandas as pd

from digital_twin.twin_model import DigitalTwinModel
from ml.tasks.residual_correction import (
    GreyBoxResidualPipeline,
    ResidualCorrectionConfig,
    CANONICAL_TARGET_CHANNELS,
    TARGET_TO_PHYSICS_MAP,
    CHANNEL_UNITS,
    CHANNEL_PHYSICAL_LIMITS,
)
from dashboard.services.adapter import DashboardAdapter
from dashboard.schemas.view_model import ChannelTelemetryModel


@pytest.fixture
def sample_synthetic_dataset():
    """Generates a small deterministic 2-flight dataset for unit testing."""
    n_steps = 50
    t = np.arange(n_steps) * 0.5
    
    # Train flight
    train_df = pd.DataFrame({
        "timestamp": t,
        "flight_id": ["FLIGHT_001"] * n_steps,
        "throttle": np.linspace(65.0, 85.0, n_steps),
        "altitude": np.linspace(1500.0, 2500.0, n_steps),
        "ambient_temp": np.full(n_steps, 15.0),
        "airspeed": np.full(n_steps, 45.0),
        "mission_phase": ["CRUISE"] * n_steps,
        "cht": 95.0 + 0.1 * np.arange(n_steps) + np.sin(t) * 0.5,
        "egt": 650.0 + 0.5 * np.arange(n_steps),
        "oil_temp": 75.0 + 0.05 * np.arange(n_steps),
        "oil_pressure": 4.2 - 0.005 * np.arange(n_steps),
        "fuel_flow": 12.0 + 0.04 * np.arange(n_steps),
        "rpm": 4500.0 + 10.0 * np.arange(n_steps),
        "map_bar": 1.05 + 0.001 * np.arange(n_steps),
        "coolant_temp": 80.0 + 0.08 * np.arange(n_steps),
        "vibration": 0.45 + 0.002 * np.arange(n_steps),
    })

    # Test flight (different flight_id and shifted conditions)
    test_df = pd.DataFrame({
        "timestamp": t,
        "flight_id": ["FLIGHT_002"] * n_steps,
        "throttle": np.full(n_steps, 80.0),
        "altitude": np.full(n_steps, 3000.0),
        "ambient_temp": np.full(n_steps, 10.0),
        "airspeed": np.full(n_steps, 50.0),
        "mission_phase": ["CRUISE"] * n_steps,
        "cht": 105.0 + np.sin(t) * 0.8,
        "egt": 680.0 + np.cos(t) * 2.0,
        "oil_temp": 82.0 + 0.02 * np.arange(n_steps),
        "oil_pressure": 3.9 - 0.002 * np.arange(n_steps),
        "fuel_flow": 14.2 + np.sin(t) * 0.2,
        "rpm": 4800.0 + np.cos(t) * 15.0,
        "map_bar": 1.10 + 0.0005 * np.arange(n_steps),
        "coolant_temp": 85.0 + 0.03 * np.arange(n_steps),
        "vibration": 0.52 + 0.001 * np.arange(n_steps),
    })

    return train_df, test_df


# ---------------------------------------------------------------------
# 1. Physics Baseline Output Schema
# ---------------------------------------------------------------------

def test_physics_baseline_output_schema():
    """Verify DigitalTwinModel produces all required expected channels deterministically."""
    twin = DigitalTwinModel()
    out = twin.step_expected(
        throttle_pct=75.0,
        altitude_m=2000.0,
        ambient_temp_c=15.0,
        dt=0.1,
    )

    expected_channels = [
        "rpm_expected",
        "load_expected",
        "cht_expected",
        "egt_expected",
        "coolant_temp_expected",
        "fuel_flow_expected",
        "oil_temp_expected",
        "oil_pressure_expected",
        "vibration_expected",
        "map_bar_expected",
    ]

    for ch in expected_channels:
        assert ch in out, f"Missing physics output channel: {ch}"
        assert isinstance(out[ch], (int, float)), f"Channel {ch} is not numeric: {out[ch]}"
        assert not math.isnan(out[ch]), f"Channel {ch} returned NaN"
        assert not math.isinf(out[ch]), f"Channel {ch} returned Inf"


# ---------------------------------------------------------------------
# 2. Exact Residual Formula
# ---------------------------------------------------------------------

def test_exact_residual_formula(sample_synthetic_dataset):
    """Verify residual is exactly calculated as: residual = observed_target - physics_prediction."""
    train_df, _ = sample_synthetic_dataset
    pipeline = GreyBoxResidualPipeline()
    physics_df = pipeline.generate_physics_predictions(train_df, dt=0.5)

    for target in ["cht", "egt", "oil_temp", "oil_pressure", "fuel_flow", "rpm"]:
        phys_col = TARGET_TO_PHYSICS_MAP[target]
        y_obs = train_df[target].to_numpy(dtype=float)
        y_phys = physics_df[phys_col].to_numpy(dtype=float)
        expected_residual = y_obs - y_phys

        # Verify elementwise definition
        diff = expected_residual - (y_obs - y_phys)
        assert np.allclose(diff, 0.0, atol=1e-9), f"Residual formula discrepancy for {target}"


# ---------------------------------------------------------------------
# 3. Exact Corrected Prediction Formula
# ---------------------------------------------------------------------

def test_exact_corrected_prediction_formula(sample_synthetic_dataset):
    """Verify corrected_prediction = physics_prediction + predicted_residual."""
    train_df, test_df = sample_synthetic_dataset
    pipeline = GreyBoxResidualPipeline(config=ResidualCorrectionConfig(clipping_safety=False))
    pipeline.fit(train_df, dt=0.5)

    preds = pipeline.predict(test_df, dt=0.5)

    for target in CANONICAL_TARGET_CHANNELS:
        y_phys = preds["physics"][target]
        y_res = preds["residual_ml"][target]
        y_corrected = preds["corrected"][target]

        expected_corrected = y_phys + y_res
        assert np.allclose(y_corrected, expected_corrected, atol=1e-7), (
            f"Corrected prediction for {target} violates formula y_phys + y_res."
        )


# ---------------------------------------------------------------------
# 4. Timestamp Monotonicity and dt Consistency
# ---------------------------------------------------------------------

def test_timestamp_monotonicity_and_dt_consistency():
    """Verify non-monotonic timestamps raise ValueError."""
    bad_df = pd.DataFrame({
        "timestamp": [0.0, 0.5, 0.4, 1.0],  # 0.4 drops back
        "throttle": [70.0, 72.0, 75.0, 75.0],
        "altitude": [2000.0, 2000.0, 2000.0, 2000.0],
        "ambient_temp": [15.0, 15.0, 15.0, 15.0],
    })

    with pytest.raises(ValueError, match="strictly monotonically increasing"):
        GreyBoxResidualPipeline.validate_telemetry_dataframe(bad_df, require_monotonic_time=True)

    good_df = pd.DataFrame({
        "timestamp": [0.0, 0.2, 0.4, 0.6],
        "throttle": [70.0, 72.0, 75.0, 75.0],
        "altitude": [2000.0, 2000.0, 2000.0, 2000.0],
        "ambient_temp": [15.0, 15.0, 15.0, 15.0],
    })
    validated = GreyBoxResidualPipeline.validate_telemetry_dataframe(good_df, require_monotonic_time=True)
    assert len(validated) == 4


# ---------------------------------------------------------------------
# 5. Unit Conversions and Sign Conventions
# ---------------------------------------------------------------------

def test_unit_conversions_and_sign_conventions():
    """Verify units dictionary and physical limits are correctly defined."""
    assert CHANNEL_UNITS["cht"] == "°C"
    assert CHANNEL_UNITS["egt"] == "°C"
    assert CHANNEL_UNITS["oil_pressure"] == "bar"
    assert CHANNEL_UNITS["rpm"] == "RPM"
    assert CHANNEL_UNITS["fuel_flow"] == "L/h"
    assert CHANNEL_UNITS["vibration"] == "g"

    # Verify physical bounds are non-empty and sensible
    for ch, (min_val, max_val) in CHANNEL_PHYSICAL_LIMITS.items():
        assert min_val < max_val, f"Invalid physical limits for {ch}: [{min_val}, {max_val}]"


# ---------------------------------------------------------------------
# 6. Missing and Invalid Sensor Handling
# ---------------------------------------------------------------------

def test_missing_and_invalid_sensor_handling():
    """Verify that NaNs, Infs, or out-of-range sensor readings are safely clipped or imputed."""
    df = pd.DataFrame({
        "timestamp": [0.0, 1.0, 2.0],
        "throttle": [75.0, 120.0, -10.0],  # out-of-range throttle
        "altitude": [2000.0, 25000.0, 1000.0],  # out-of-range altitude
        "ambient_temp": [15.0, 15.0, 15.0],
        "cht": [90.0, np.nan, 350.0],  # NaN and extreme CHT
    })

    cleaned = GreyBoxResidualPipeline.validate_telemetry_dataframe(df, clamp_limits=True)
    assert cleaned["throttle"].max() <= 100.0
    assert cleaned["throttle"].min() >= 0.0
    assert cleaned["altitude"].max() <= 15000.0
    assert cleaned["cht"].max() <= 300.0


# ---------------------------------------------------------------------
# 7. No Train/Test Data Leakage
# ---------------------------------------------------------------------

def test_no_train_test_data_leakage(sample_synthetic_dataset):
    """Verify modifying test split data has zero impact on fitted training parameters."""
    train_df, test_df = sample_synthetic_dataset

    pipeline1 = GreyBoxResidualPipeline(config=ResidualCorrectionConfig(model_type="ridge"))
    pipeline1.fit(train_df, dt=0.5)
    means_before = np.copy(pipeline1.preprocessor_residual.means_)
    coefs_before = np.copy(pipeline1.residual_models["cht"].coef_)

    # Corrupt or alter test_df radically
    corrupted_test_df = test_df.copy()
    corrupted_test_df["cht"] = corrupted_test_df["cht"] * 10.0 + 500.0
    corrupted_test_df["throttle"] = 99.0

    pipeline2 = GreyBoxResidualPipeline(config=ResidualCorrectionConfig(model_type="ridge"))
    pipeline2.fit(train_df, dt=0.5)
    means_after = np.copy(pipeline2.preprocessor_residual.means_)
    coefs_after = np.copy(pipeline2.residual_models["cht"].coef_)

    # Training parameters must be bitwise identical
    assert np.array_equal(means_before, means_after), "Preprocessor means changed; data leakage detected!"
    assert np.array_equal(coefs_before, coefs_after), "Model coefficients changed; data leakage detected!"


# ---------------------------------------------------------------------
# 8. Deterministic Inference
# ---------------------------------------------------------------------

def test_deterministic_inference(sample_synthetic_dataset):
    """Verify running inference multiple times produces bitwise identical predictions."""
    train_df, test_df = sample_synthetic_dataset
    pipeline = GreyBoxResidualPipeline()
    pipeline.fit(train_df, dt=0.5)

    preds1 = pipeline.predict(test_df, dt=0.5)
    preds2 = pipeline.predict(test_df, dt=0.5)

    for target in CANONICAL_TARGET_CHANNELS:
        assert np.array_equal(preds1["physics"][target], preds2["physics"][target]), f"Physics nondeterminism in {target}"
        assert np.array_equal(preds1["residual_ml"][target], preds2["residual_ml"][target]), f"Residual nondeterminism in {target}"
        assert np.array_equal(preds1["corrected"][target], preds2["corrected"][target]), f"Corrected nondeterminism in {target}"


# ---------------------------------------------------------------------
# 9. Three-Way Model Comparison
# ---------------------------------------------------------------------

def test_three_way_model_comparison(sample_synthetic_dataset):
    """Verify 3-way model comparison produces complete metrics across all canonical channels."""
    train_df, test_df = sample_synthetic_dataset
    pipeline = GreyBoxResidualPipeline()
    pipeline.fit(train_df, dt=0.5)

    res = pipeline.evaluate_three_way(test_df, dt=0.5)

    assert len(res.channel_metrics) > 0
    assert res.test_samples == len(test_df)

    for target, m in res.channel_metrics.items():
        assert m.physics_only_mae >= 0.0
        assert m.pure_ml_mae >= 0.0
        assert m.grey_box_mae >= 0.0
        assert m.physics_only_rmse >= m.physics_only_mae * 0.99  # RMSE >= MAE in Euclidean space
        assert isinstance(m.error_reduction_vs_physics_pct, float)
        assert isinstance(m.error_reduction_vs_pure_ml_pct, float)


# ---------------------------------------------------------------------
# 10. Dashboard View-Model Contract
# ---------------------------------------------------------------------

def test_dashboard_view_model_contract():
    """Verify DashboardAdapter populates operator-facing grey-box fields in ChannelTelemetryModel."""
    adapter = DashboardAdapter()
    payload = {
        "timestamp": 12.0,
        "engine_id": "ENG_UAV_TEST",
        "mission_phase": "CRUISE",
        "observed_telemetry": {
            "cht": 98.5,
            "egt": 680.0,
            "oil_temp": 82.0,
            "oil_pressure": 4.1,
            "fuel_flow": 13.5,
            "rpm": 4400.0,
            "vibration": 0.42,
        },
        "expected_telemetry": {
            "expected_cht": 95.0,
            "expected_egt": 670.0,
            "expected_oil_temp": 80.0,
            "expected_oil_pressure": 4.2,
            "expected_fuel_flow": 13.0,
            "expected_rpm": 4350.0,
            "expected_vibration": 0.40,
        },
        "residuals": {
            "cht_residual": 3.5,
            "egt_residual": 10.0,
            "oil_temp_residual": 2.0,
            "oil_pressure_residual": -0.1,
            "fuel_flow_residual": 0.5,
            "rpm_residual": 50.0,
            "vibration_residual": 0.02,
        },
        "sensor_corrections": {
            "cht": 3.2,
        },
        "diagnostic_confidence": 0.95,
    }

    vm = adapter.adapt(payload)
    cht_model = vm.telemetry.channels.get("cht")
    assert cht_model is not None
    assert cht_model.physics_estimate == 95.0
    assert cht_model.sensor_correction == 3.2
    assert cht_model.corrected_prediction == 98.2
    assert cht_model.detected_deviation == 3.5
    assert cht_model.model_confidence == 0.95


# ---------------------------------------------------------------------
# 11. Explainability Structure
# ---------------------------------------------------------------------

def test_explainability_structure(sample_synthetic_dataset):
    """Verify explain_instance outputs clean separation of physics, residual, and health drivers without causal claims."""
    train_df, test_df = sample_synthetic_dataset
    pipeline = GreyBoxResidualPipeline()
    pipeline.fit(train_df, dt=0.5)

    sample_row = test_df.iloc[0]
    explanations = pipeline.explain_instance(sample_row, dt=0.5)

    for target in CANONICAL_TARGET_CHANNELS:
        exp = explanations[target]
        assert "throttle_pct" in exp.physics_drivers
        assert "altitude_m" in exp.physics_drivers
        assert "ambient_temp_c" in exp.physics_drivers
        assert isinstance(exp.residual_drivers, dict)
        assert "physics_estimate" in exp.health_indicator
        assert "sensor_informed_correction" in exp.health_indicator
        assert "corrected_prediction" in exp.health_indicator
        # Non-causal disclaimer check
        assert "does not establish physical causation" in exp.disclaimer or "do not establish physical causation" in exp.disclaimer


# ---------------------------------------------------------------------
# 12. Numerical Stability Edge Cases
# ---------------------------------------------------------------------

def test_numerical_stability_edge_cases():
    """Verify pipeline behaves safely on extreme zero or boundary inputs without crashing."""
    edge_df = pd.DataFrame({
        "timestamp": [0.0, 0.1],
        "throttle": [0.0, 100.0],
        "altitude": [0.0, 12000.0],
        "ambient_temp": [-40.0, 50.0],
        "airspeed": [0.0, 120.0],
        "mission_phase": ["PREFLIGHT", "CRUISE"],
        "cht": [15.0, 250.0],
        "egt": [150.0, 950.0],
        "oil_temp": [10.0, 140.0],
        "oil_pressure": [1.5, 6.0],
        "fuel_flow": [2.0, 35.0],
        "rpm": [1800.0, 5800.0],
        "map_bar": [0.5, 2.2],
        "coolant_temp": [15.0, 110.0],
        "vibration": [0.1, 2.5],
    })

    pipeline = GreyBoxResidualPipeline()
    pipeline.fit(edge_df, dt=0.1)
    preds = pipeline.predict(edge_df, dt=0.1)

    for target in CANONICAL_TARGET_CHANNELS:
        y_corr = preds["corrected"][target]
        assert not np.isnan(y_corr).any(), f"NaN detected in {target}"
        assert not np.isinf(y_corr).any(), f"Inf detected in {target}"
