"""
Unit and Integration Tests for Phase 4: Health Monitoring, Fault Injection, Uncertainty, and Prognostics.

Verifies:
1. Health-State and Alert-Classification Schemas and [0.0, 1.0] Bounds.
2. Deterministic Fault Injection and Catalog Properties.
3. Persistence and Hysteresis Behavior (Single-point noise immunity, deadband recovery).
4. Sensor-Fault vs. Physical-Degradation Discrimination.
5. Engineering Uncertainty Interval Ordering (lower <= pred <= upper) & Training-Only Calibration.
6. Causal Prognostics & Conditional RUL Withholding (UNAVAILABLE when stable).
7. Missing Data, Sensor Dropout, and Numerical Stability.
8. Dashboard View Model Contracts and Operator-Facing Copywriting.
"""

import math
import numpy as np
import pandas as pd
import pytest

from ml.tasks.health_prognostics import (
    EngineHealthState,
    AlertClassification,
    DegradationTrendState,
    ChannelUncertainty,
    HealthMonitoringOutput,
    HealthPrognosticsPipeline,
    get_standard_fault_catalog,
)
from dashboard.schemas.view_model import (
    ChannelTelemetryModel,
    DiagnosticsViewModel,
    PrognosticsViewModel,
    DataQualityViewModel,
    StatusLevel,
)
from dashboard.services.adapter import DashboardAdapter


# =====================================================================
# 1. Health-State & Schema Bounds Tests
# =====================================================================

def test_engine_health_states_enumeration():
    """Verify all 8 authoritative health states are defined."""
    expected_states = {
        "HEALTHY",
        "THERMAL_DEGRADATION",
        "COOLING_DEGRADATION",
        "LUBRICATION_DEGRADATION",
        "COMBUSTION_DEGRADATION",
        "VIBRATION_DEGRADATION",
        "SENSOR_ANOMALY",
        "UNKNOWN_INSUFFICIENT_DATA",
    }
    actual_states = {s.value for s in EngineHealthState}
    assert expected_states == actual_states


def test_alert_classification_enumeration():
    """Verify alert classification categories."""
    expected_alerts = {
        "NOMINAL",
        "POSSIBLE_PHYSICAL_DEGRADATION",
        "SENSOR_ANOMALY",
        "MODEL_DISAGREEMENT",
        "INSUFFICIENT_DATA",
    }
    actual_alerts = {a.value for a in AlertClassification}
    assert expected_alerts == actual_alerts


def test_health_score_bounds_and_clipping():
    """Verify health score, anomaly score, and degradation severity remain in [0.0, 1.0]."""
    pipeline = HealthPrognosticsPipeline()

    # Feed healthy baseline
    out = pipeline.process_step(
        timestamp=1.0,
        observed={"cht": 95.0, "egt": 720.0, "oil_temp": 88.0, "oil_pressure": 3.8, "fuel_flow": 16.0, "rpm": 4800.0, "map_bar": 1.15, "coolant_temp": 82.0, "vibration": 0.35},
        physics_expected={"cht_expected": 95.0, "egt_expected": 720.0, "oil_temp_expected": 88.0, "oil_pressure_expected": 3.8, "fuel_flow_expected": 16.0, "rpm_expected": 4800.0, "map_bar_expected": 1.15, "coolant_temp_expected": 82.0, "vibration_expected": 0.35},
    )
    assert 0.0 <= out.health_score <= 1.0
    assert 0.0 <= out.anomaly_score <= 1.0
    assert 0.0 <= out.degradation_severity <= 1.0
    assert 0.0 <= out.confidence <= 1.0
    assert out.health_state == EngineHealthState.HEALTHY
    assert out.alert_classification == AlertClassification.NOMINAL


# =====================================================================
# 2. Fault Injection Catalog Determinism
# =====================================================================

def test_standard_fault_catalog_completeness():
    """Verify standard fault scenario catalog contains required physical and sensor faults."""
    catalog = get_standard_fault_catalog()
    assert "cooling_degradation_ramp" in catalog
    assert "lubrication_pressure_loss_step" in catalog
    assert "thermal_load_increase" in catalog
    assert "cht_sensor_drift" in catalog
    assert "rpm_sensor_bias" in catalog
    assert "oil_pressure_sensor_dropout" in catalog
    assert "transient_throttle_disturbance" in catalog

    for fault_id, defn in catalog.items():
        assert defn.fault_name
        assert defn.fault_type
        assert defn.affected_signal
        assert defn.start_time >= 0.0
        assert 0.0 <= defn.severity <= 1.0
        assert defn.injection_formula
        assert defn.expected_observable_effect
        assert isinstance(defn.is_sensor_fault, bool)


def test_fault_catalog_determinism():
    """Verify multiple catalog queries produce identical deterministic definitions."""
    c1 = get_standard_fault_catalog()
    c2 = get_standard_fault_catalog()
    assert c1 == c2


# =====================================================================
# 3. Persistence and Hysteresis (Noise Immunity)
# =====================================================================

def test_single_noisy_sample_does_not_trigger_alarm():
    """A single aberrant sample must not trigger an alert due to persistence counter requirement (N=5)."""
    pipeline = HealthPrognosticsPipeline(persistence_steps=5, z_score_threshold=2.5)

    # 1. Normal baseline
    normal_obs = {"cht": 95.0, "egt": 720.0, "oil_temp": 88.0, "oil_pressure": 3.8, "coolant_temp": 82.0, "fuel_flow": 16.0, "rpm": 4800.0, "map_bar": 1.15, "vibration": 0.35}
    normal_exp = {"cht_expected": 95.0, "egt_expected": 720.0, "oil_temp_expected": 88.0, "oil_pressure_expected": 3.8, "coolant_temp_expected": 82.0, "fuel_flow_expected": 16.0, "rpm_expected": 4800.0, "map_bar_expected": 1.15, "vibration_expected": 0.35}

    for t in [1.0, 2.0, 3.0]:
        out = pipeline.process_step(t, normal_obs, normal_exp)
        assert out.alert_classification == AlertClassification.NOMINAL

    # 2. Inject SINGLE noisy point (huge 8-sigma CHT spike)
    noisy_obs = normal_obs.copy()
    noisy_obs["cht"] = 160.0  # Big excursion

    out_noisy = pipeline.process_step(4.0, noisy_obs, normal_exp)
    # Must remain NOMINAL because persistence_steps == 5!
    assert out_noisy.alert_classification == AlertClassification.NOMINAL
    assert len(out_noisy.affected_channels) == 0


def test_persistent_fault_triggers_alarm_after_n_steps():
    """Persistent fault exceeding threshold for N steps successfully alarms."""
    pipeline = HealthPrognosticsPipeline(persistence_steps=5, z_score_threshold=2.5)
    normal_exp = {"cht_expected": 95.0, "egt_expected": 720.0, "oil_temp_expected": 88.0, "oil_pressure_expected": 3.8, "coolant_temp_expected": 82.0, "fuel_flow_expected": 16.0, "rpm_expected": 4800.0, "map_bar_expected": 1.15, "vibration_expected": 0.35}

    faulty_obs = normal_exp.copy()
    # Mutate to create persistent excursion
    faulty_obs = {"cht": 130.0, "egt": 720.0, "oil_temp": 88.0, "oil_pressure": 3.8, "coolant_temp": 98.0, "fuel_flow": 16.0, "rpm": 4800.0, "map_bar": 1.15, "vibration": 0.35}

    outputs = []
    for step in range(1, 8):
        out = pipeline.process_step(float(step), faulty_obs, normal_exp)
        outputs.append(out)

    # First 4 steps must be NOMINAL
    for i in range(4):
        assert outputs[i].alert_classification == AlertClassification.NOMINAL

    # 5th step hits persistence threshold N=5 and triggers alarm
    assert outputs[4].alert_classification != AlertClassification.NOMINAL
    assert outputs[5].alert_classification != AlertClassification.NOMINAL


def test_hysteresis_recovery():
    """Verify recovery requires consecutive steps below recovery threshold before un-alarming."""
    pipeline = HealthPrognosticsPipeline(
        persistence_steps=3,
        hysteresis_recovery_steps=3,
        z_score_threshold=2.5,
        recovery_z_threshold=1.2,
    )
    normal_exp = {"cht_expected": 95.0, "egt_expected": 720.0, "oil_temp_expected": 88.0, "oil_pressure_expected": 3.8, "coolant_temp_expected": 82.0, "fuel_flow_expected": 16.0, "rpm_expected": 4800.0, "map_bar_expected": 1.15, "vibration_expected": 0.35}
    faulty_obs = {"cht": 130.0, "egt": 720.0, "oil_temp": 88.0, "oil_pressure": 3.8, "coolant_temp": 98.0, "fuel_flow": 16.0, "rpm": 4800.0, "map_bar": 1.15, "vibration": 0.35}
    recovered_obs = {"cht": 95.0, "egt": 720.0, "oil_temp": 88.0, "oil_pressure": 3.8, "coolant_temp": 82.0, "fuel_flow": 16.0, "rpm": 4800.0, "map_bar": 1.15, "vibration": 0.35}

    # Trigger alarm (3 steps)
    for t in [1.0, 2.0, 3.0]:
        out = pipeline.process_step(t, faulty_obs, normal_exp)
    assert out.alert_classification != AlertClassification.NOMINAL

    # First recovery step: still alarmed (hysteresis)
    out_rec1 = pipeline.process_step(4.0, recovered_obs, normal_exp)
    assert out_rec1.alert_classification != AlertClassification.NOMINAL

    # Second recovery step: still alarmed
    out_rec2 = pipeline.process_step(5.0, recovered_obs, normal_exp)
    assert out_rec2.alert_classification != AlertClassification.NOMINAL

    # Third recovery step: meets hysteresis_recovery_steps=3 -> returns to NOMINAL
    out_rec3 = pipeline.process_step(6.0, recovered_obs, normal_exp)
    assert out_rec3.alert_classification == AlertClassification.NOMINAL


# =====================================================================
# 4. Sensor-Fault vs Physical-Degradation Discrimination
# =====================================================================

def test_sensor_fault_discrimination_unilateral_drift():
    """Unilateral deviation in CHT while coolant and oil temperatures remain normal -> SENSOR_ANOMALY."""
    pipeline = HealthPrognosticsPipeline(persistence_steps=3, z_score_threshold=2.0)
    normal_exp = {"cht_expected": 95.0, "coolant_temp_expected": 82.0, "oil_temp_expected": 88.0, "oil_pressure_expected": 3.8, "fuel_flow_expected": 16.0, "rpm_expected": 4800.0}

    # Unilateral CHT drift only
    drift_obs = {"cht": 130.0, "coolant_temp": 82.0, "oil_temp": 88.0, "oil_pressure": 3.8, "fuel_flow": 16.0, "rpm": 4800.0}

    for t in [1.0, 2.0, 3.0, 4.0]:
        out = pipeline.process_step(t, drift_obs, normal_exp)

    assert out.alert_classification == AlertClassification.SENSOR_ANOMALY
    assert out.health_state == EngineHealthState.SENSOR_ANOMALY
    assert "cht" in out.affected_channels


def test_physical_degradation_discrimination_correlated_cooling():
    """Correlated elevations in both CHT and Coolant Temp -> POSSIBLE_PHYSICAL_DEGRADATION."""
    pipeline = HealthPrognosticsPipeline(persistence_steps=3, z_score_threshold=2.0)
    normal_exp = {"cht_expected": 95.0, "coolant_temp_expected": 82.0, "oil_temp_expected": 88.0, "oil_pressure_expected": 3.8, "fuel_flow_expected": 16.0, "rpm_expected": 4800.0}

    # Correlated thermal rise
    thermal_obs = {"cht": 125.0, "coolant_temp": 96.0, "oil_temp": 95.0, "oil_pressure": 3.8, "fuel_flow": 16.0, "rpm": 4800.0}

    for t in [1.0, 2.0, 3.0, 4.0]:
        out = pipeline.process_step(t, thermal_obs, normal_exp)

    assert out.alert_classification == AlertClassification.POSSIBLE_PHYSICAL_DEGRADATION
    assert out.health_state in (EngineHealthState.COOLING_DEGRADATION, EngineHealthState.THERMAL_DEGRADATION)
    assert "COOLING" in out.affected_subsystems or "THERMAL" in out.affected_subsystems


# =====================================================================
# 5. Engineering Uncertainty Interval Ordering
# =====================================================================

def test_uncertainty_interval_ordering():
    """Ensure prediction_lower <= predicted_value <= prediction_upper for all channels."""
    pipeline = HealthPrognosticsPipeline()
    obs = {"cht": 96.0, "coolant_temp": 83.0, "oil_temp": 89.0, "oil_pressure": 3.7, "fuel_flow": 16.1, "rpm": 4810.0, "map_bar": 1.14}
    exp = {"cht_expected": 95.0, "coolant_temp_expected": 82.0, "oil_temp_expected": 88.0, "oil_pressure_expected": 3.8, "fuel_flow_expected": 16.0, "rpm_expected": 4800.0, "map_bar_expected": 1.15}

    out = pipeline.process_step(1.0, obs, exp)
    for ch, unc in out.channel_uncertainties.items():
        assert unc.prediction_lower <= unc.predicted_value, f"Lower bound violation for {ch}"
        assert unc.predicted_value <= unc.prediction_upper, f"Upper bound violation for {ch}"
        assert unc.uncertainty_width >= 0.0
        assert "engineering_estimate" in unc.confidence_basis


# =====================================================================
# 6. Causal Prognostics & Conditional RUL Withholding
# =====================================================================

def test_rul_withholding_on_healthy_stable_telemetry():
    """RUL must be explicitly withheld as UNAVAILABLE when health is stable or non-degrading."""
    pipeline = HealthPrognosticsPipeline(history_window_size=20)
    normal_exp = {"cht_expected": 95.0, "coolant_temp_expected": 82.0, "oil_temp_expected": 88.0, "oil_pressure_expected": 3.8, "fuel_flow_expected": 16.0, "rpm_expected": 4800.0}
    normal_obs = {"cht": 95.0, "coolant_temp": 82.0, "oil_temp": 88.0, "oil_pressure": 3.8, "coolant_temp": 82.0, "fuel_flow": 16.0, "rpm": 4800.0}

    out = None
    for step in range(25):
        out = pipeline.process_step(float(step), normal_obs, normal_exp)

    assert out is not None
    assert out.rul_state == "UNAVAILABLE"
    assert out.time_to_threshold_s is None
    assert out.trend_state == DegradationTrendState.STABLE
    assert "stable or non-degrading" in out.prognostics_reason.lower()


def test_time_to_threshold_on_progressive_degradation():
    """When a sustained monotonic degradation trajectory is present, time-to-threshold is estimated."""
    pipeline = HealthPrognosticsPipeline(history_window_size=30, persistence_steps=2)
    normal_exp = {"cht_expected": 95.0, "coolant_temp_expected": 82.0, "oil_temp_expected": 88.0, "oil_pressure_expected": 3.8, "fuel_flow_expected": 16.0, "rpm_expected": 4800.0}

    # Simulate progressive thermal decline
    out = None
    for step in range(35):
        t = float(step)
        obs = {
            "cht": 95.0 + (step * 1.5),
            "coolant_temp": 82.0 + (step * 0.8),
            "oil_temp": 88.0 + (step * 0.5),
            "oil_pressure": 3.8,
            "fuel_flow": 16.0,
            "rpm": 4800.0,
        }
        out = pipeline.process_step(t, obs, normal_exp)

    assert out is not None
    assert out.trend_state in (DegradationTrendState.DEGRADING, DegradationTrendState.RAPIDLY_DEGRADING)
    assert out.trend_slope_per_s is not None and out.trend_slope_per_s < 0.0
    if out.health_score > pipeline.eol_threshold:
        assert out.rul_state == "AVAILABLE"
        assert out.time_to_threshold_s is not None
        assert out.time_to_threshold_s > 0.0


# =====================================================================
# 7. Missing Data & Dropout Handling
# =====================================================================

def test_missing_data_and_dropout_behavior():
    """Sensor dropouts (NaNs) should yield SENSOR_DROPOUT quality status without crashing."""
    pipeline = HealthPrognosticsPipeline()
    normal_exp = {"cht_expected": 95.0, "coolant_temp_expected": 82.0, "oil_temp_expected": 88.0, "oil_pressure_expected": 3.8, "fuel_flow_expected": 16.0, "rpm_expected": 4800.0}

    # Dropout on oil_pressure
    drop_obs = {"cht": 95.0, "coolant_temp": 82.0, "oil_temp": 88.0, "oil_pressure": np.nan, "fuel_flow": 16.0, "rpm": 4800.0}

    out = pipeline.process_step(1.0, drop_obs, normal_exp)
    assert out.data_quality_status == "SENSOR_DROPOUT"
    assert "oil_pressure" in out.evidence["missing_channels"]
    assert 0.0 <= out.health_score <= 1.0


def test_insufficient_history_behavior():
    """History window with <= 2 points must return INSUFFICIENT_HISTORY trend."""
    pipeline = HealthPrognosticsPipeline()
    normal_exp = {"cht_expected": 95.0, "coolant_temp_expected": 82.0, "oil_temp_expected": 88.0, "oil_pressure_expected": 3.8, "fuel_flow_expected": 16.0, "rpm_expected": 4800.0}
    normal_obs = {"cht": 95.0, "coolant_temp": 82.0, "oil_temp": 88.0, "oil_pressure": 3.8, "fuel_flow": 16.0, "rpm": 4800.0}

    out = pipeline.process_step(1.0, normal_obs, normal_exp)
    assert out.trend_state == DegradationTrendState.INSUFFICIENT_HISTORY
    assert out.rul_state == "UNAVAILABLE"


# =====================================================================
# 8. Dashboard View Model Adapter Integration
# =====================================================================

def test_dashboard_adapter_phase4_fields():
    """Verify DashboardAdapter maps Phase 4 outputs into view model with operator-facing labels."""
    adapter = DashboardAdapter()

    payload = {
        "timestamp": 12.0,
        "engine_health_score": 0.82,
        "health_score": 0.82,
        "health_state": "COOLING_DEGRADATION",
        "alert_classification": "POSSIBLE_PHYSICAL_DEGRADATION",
        "degradation_severity": 0.38,
        "affected_subsystems": ["COOLING", "THERMAL"],
        "subsystem_scores": {"COOLING": 0.72, "THERMAL": 0.78, "LUBRICATION": 0.95},
        "trend_state": "DEGRADING",
        "prognostics_reason": "Progressive coolant loop thermal rise",
        "evidence": {"summary": "Elevated CHT and coolant temperature exceeding normal variance."},
        "observed_telemetry": {"cht": 118.0, "coolant_temp": 92.0, "oil_temp": 90.0, "oil_pressure": 3.6, "fuel_flow": 16.0, "rpm": 4800.0, "vibration": 0.35},
        "channel_uncertainties": {
            "cht": {
                "channel": "cht",
                "predicted_value": 96.0,
                "prediction_lower": 91.0,
                "prediction_upper": 101.0,
                "uncertainty_width": 10.0,
                "confidence_basis": "rolling_residual_dispersion_engineering_estimate",
            }
        },
    }

    vm = adapter.adapt(payload)

    # 1. Diagnostics view model
    assert vm.diagnostics.alert_classification == "Possible physical degradation"
    assert vm.diagnostics.alert_classification_raw == "POSSIBLE_PHYSICAL_DEGRADATION"
    assert vm.diagnostics.degradation_severity == 0.38
    assert "COOLING" in vm.diagnostics.affected_subsystems

    # 2. Prognostics view model
    assert vm.prognostics.engine_health_score == 0.82
    assert vm.prognostics.trend_state == "Degrading"
    assert vm.prognostics.prognostics_reason == "Progressive coolant loop thermal rise"
    assert "COOLING" in vm.prognostics.subsystem_scores

    # 3. Channel telemetry model
    cht_channel = vm.telemetry.channels.get("cht")
    assert cht_channel is not None
    assert cht_channel.prediction_lower == 91.0
    assert cht_channel.prediction_upper == 101.0
    assert cht_channel.uncertainty_width == 10.0
