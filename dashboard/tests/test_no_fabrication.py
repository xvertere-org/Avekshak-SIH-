"""
Architectural verification tests: Absolute Non-Fabrication Rules.

Verifies that the dashboard adapter NEVER computes, fabricates, or substitutes
missing ML/PHM outputs (Health Index, RUL, Anomaly scores, Fault diagnosis, Forecast).
"""

import math
import pytest
from dashboard.services.adapter import DashboardAdapter
from dashboard.schemas.view_model import StatusLevel, AvailabilityStatus


def test_missing_health_index_not_fabricated():
    """When Phase 9 is omitted, Health Index must remain None / Unavailable."""
    adapter = DashboardAdapter()
    payload = {
        "timestamp": 10.0,
        "engine_id": "ENGINE_01",
        "telemetry": {
            "timestamp": 10.0,
            "rpm": 5000.0,
            "cht": 100.0,
            "egt": 600.0,
            "oil_temp": 80.0,
            "oil_pressure": 4.0,
            "fuel_flow": 15.0,
            "vibration": 0.5,
        },
        # health_index intentionally omitted!
    }

    vm = adapter.adapt(payload)
    assert vm.prognostics.health_index is None
    assert vm.prognostics.health_state == "Unavailable"
    assert vm.overview.health_card.value == "Unavailable"
    assert vm.overview.health_card.status == StatusLevel.UNAVAILABLE
    assert vm.overview.health_card.availability == AvailabilityStatus.UNAVAILABLE


def test_missing_prognostics_rul_not_fabricated():
    """When Phase 11 is omitted, RUL must remain None / Unavailable."""
    adapter = DashboardAdapter()
    payload = {
        "timestamp": 20.0,
        "engine_id": "ENGINE_01",
        "health_index": {
            "smoothed_health_index": 0.95,
            "health_state": "HEALTHY",
        },
        # prognostics intentionally omitted!
    }

    vm = adapter.adapt(payload)
    assert vm.prognostics.rul_hours is None
    assert vm.prognostics.rul_status == "Unavailable"
    assert vm.overview.rul_card.value == "Unavailable"
    assert vm.overview.rul_card.status == StatusLevel.UNAVAILABLE
    assert vm.overview.rul_card.availability == AvailabilityStatus.UNAVAILABLE


def test_missing_anomaly_detection_not_fabricated():
    """When Phase 7 is omitted, anomaly status must remain Unavailable and score None."""
    adapter = DashboardAdapter()
    payload = {
        "timestamp": 30.0,
        "engine_id": "ENGINE_01",
        # anomaly intentionally omitted!
    }

    vm = adapter.adapt(payload)
    assert vm.diagnostics.anomaly_status == "Unavailable"
    assert vm.diagnostics.anomaly_score is None
    assert vm.overview.anomaly_card.value == "Unavailable"
    assert vm.overview.anomaly_card.status == StatusLevel.UNAVAILABLE


def test_missing_fault_diagnosis_not_fabricated():
    """When Phase 8 is omitted, predicted fault must remain Unavailable and confidence None."""
    adapter = DashboardAdapter()
    payload = {
        "timestamp": 40.0,
        "engine_id": "ENGINE_01",
        # fault_diagnosis intentionally omitted!
    }

    vm = adapter.adapt(payload)
    assert vm.diagnostics.predicted_fault == "Unavailable"
    assert vm.diagnostics.diagnostic_confidence is None
    assert vm.diagnostics.class_probabilities == {}
    assert vm.overview.fault_card.value == "Unavailable"
    assert vm.overview.fault_card.status == StatusLevel.UNAVAILABLE


def test_missing_forecast_not_fabricated():
    """When Phase 10 is omitted, forecast predictions must remain empty without executing TimesFM."""
    adapter = DashboardAdapter()
    payload = {
        "timestamp": 50.0,
        "engine_id": "ENGINE_01",
        "telemetry": {
            "timestamp": 50.0,
            "rpm": 5000.0,
            "cht": 100.0,
            "egt": 600.0,
            "oil_temp": 80.0,
            "oil_pressure": 4.0,
            "fuel_flow": 15.0,
            "vibration": 0.5,
        },
        # forecast intentionally omitted!
    }

    vm = adapter.adapt(payload)
    for ch_model in vm.telemetry.channels.values():
        assert ch_model.forecast_values == []
        assert ch_model.forecast_timestamps == []
    assert vm.prognostics.forecast_quality is None
    assert vm.prognostics.projected_health_trajectory is None


def test_nan_sensor_values_not_replaced_with_zeros_or_nominal():
    """NaN telemetry values must be preserved as None / Unavailable, not replaced with 0 or nominal."""
    adapter = DashboardAdapter()
    payload = {
        "timestamp": 60.0,
        "engine_id": "ENGINE_01",
        "telemetry": {
            "timestamp": 60.0,
            "rpm": float("nan"),
            "cht": float("nan"),
            "egt": None,
            "oil_temp": float("inf"),
            "oil_pressure": 3.0,
            "fuel_flow": 12.0,
            "vibration": 0.4,
        },
    }

    vm = adapter.adapt(payload)
    assert vm.telemetry.channels["rpm"].observed_value is None
    assert vm.telemetry.channels["rpm"].availability == AvailabilityStatus.UNAVAILABLE

    assert vm.telemetry.channels["cht"].observed_value is None
    assert vm.telemetry.channels["cht"].availability == AvailabilityStatus.UNAVAILABLE

    assert vm.telemetry.channels["egt"].observed_value is None
    assert vm.telemetry.channels["egt"].availability == AvailabilityStatus.UNAVAILABLE

    assert vm.telemetry.channels["oil_temp"].observed_value is None
    assert vm.telemetry.channels["oil_temp"].availability == AvailabilityStatus.UNAVAILABLE

    # Valid numeric channels remain valid
    assert vm.telemetry.channels["oil_pressure"].observed_value == 3.0
    assert vm.telemetry.channels["fuel_flow"].observed_value == 12.0
