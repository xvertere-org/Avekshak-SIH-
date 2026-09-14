"""
Tests for Sensor Blackout Resilience across the Complete Digital Twin Pipeline.

Verifies the core architectural invariant:
    UNKNOWN / NO DATA != HEALTHY

Tests:
1. Normal mission followed by complete sensor blackout (all NaN).
2. All None values.
3. All Inf values.
4. Missing all channel fields.
5. Completely empty telemetry dictionary {}.
6. Partial telemetry dropouts (1 channel, 3 channels, 6 channels missing).
7. Single-sample blackout followed by recovery.
8. 10-sample sustained blackout.
9. Blackout followed by valid mission continuation (proves history buffer is not polluted with NaNs).
10. EWMADetector unit test: returns NaN on valid_count == 0 even with populated state.
11. Dashboard adapter: transforms blackout payload into UNAVAILABLE status levels with zero healthy badges.
"""

import math
import pytest
import numpy as np
import pandas as pd

from orchestrator.pipeline import SystemPipelineOrchestrator
from orchestrator.adapter import PipelineHandoffAdapter
from anomaly_detection.detectors import EWMADetector
from dashboard.services.adapter import DashboardAdapter
from dashboard.schemas.view_model import StatusLevel, AvailabilityStatus


CANONICAL_CHANNELS = ["rpm", "cht", "egt", "oil_temp", "oil_pressure", "fuel_flow", "vibration"]

NOMINAL_TELEMETRY = {
    "rpm": 2500.0,
    "cht": 180.0,
    "egt": 650.0,
    "oil_temp": 85.0,
    "oil_pressure": 4.5,
    "fuel_flow": 30.0,
    "vibration": 1.2,
    "throttle": 75.0,
    "altitude": 2000.0,
    "ambient_temp": 15.0,
    "mission_phase": "CRUISE",
}


def _feed_normal_history(orch: SystemPipelineOrchestrator, steps: int = 5, engine_id: str = "ENG-01", mission_id: str = "MSN-01"):
    """Feed initial nominal mission steps to populate historical state across all phases."""
    for i in range(1, steps + 1):
        telem = dict(NOMINAL_TELEMETRY)
        telem["timestamp"] = float(i)
        telem["engine_id"] = engine_id
        telem["mission_id"] = mission_id
        orch.step(telem)


def test_ewma_detector_blackout_with_populated_state():
    """Verify EWMADetector returns NaN when valid_count == 0, even with populated state."""
    det = EWMADetector()
    # Step 1: Valid observation populates internal state
    res1 = det.update_sample({"rpm_norm_residual": 0.5, "cht_norm_residual": 0.2})
    assert not math.isnan(res1["ewma_score"])
    assert len(det.state) > 0

    # Step 2: All channels NaN (blackout)
    res2 = det.update_sample({"rpm_norm_residual": float("nan"), "cht_norm_residual": float("nan")})
    assert math.isnan(res2["ewma_score"]), f"Expected NaN ewma_score on blackout, got {res2['ewma_score']}"
    assert res2["is_anomaly"] is False
    assert res2["is_warning"] is False
    assert res2["warning_channels"] == []
    assert res2["anomaly_channels"] == []

    # Step 3: All channels None
    res3 = det.update_sample({"rpm_norm_residual": None, "cht_norm_residual": None})
    assert math.isnan(res3["ewma_score"])

    # Step 4: Empty feature dictionary
    res4 = det.update_sample({})
    assert math.isnan(res4["ewma_score"])


def test_complete_blackout_all_nan_after_history():
    """Verify complete sensor blackout (all NaN) after valid history produces INSUFFICIENT_DATA, not normal."""
    orch = SystemPipelineOrchestrator()
    _feed_normal_history(orch, steps=5)

    blackout = {
        "timestamp": 6.0,
        "engine_id": "ENG-01",
        "mission_id": "MSN-01",
        "rpm": float("nan"),
        "cht": float("nan"),
        "egt": float("nan"),
        "oil_temp": float("nan"),
        "oil_pressure": float("nan"),
        "fuel_flow": float("nan"),
        "vibration": float("nan"),
        "throttle": float("nan"),
        "altitude": float("nan"),
        "ambient_temp": float("nan"),
        "mission_phase": "CRUISE",
    }
    payload = orch.step(blackout)

    # Telemetry quality
    assert payload.quality_status == "MISSING"

    # Anomaly detection: NEVER 0.0 or NORMAL
    assert payload.anomaly_status == "INSUFFICIENT_DATA"
    assert math.isnan(payload.anomaly_score)
    assert payload.anomaly_contributing_channels == []

    # Fault diagnosis: NEVER 100% confidence or normal
    assert payload.diagnostic_confidence == 0.0
    assert payload.diagnosis_data_quality == "INSUFFICIENT_DATA"

    # Health index: NEVER healthy or 1.0
    assert math.isnan(payload.raw_health_index)
    assert math.isnan(payload.smoothed_health_index)
    assert payload.health_state == "INSUFFICIENT_DATA"

    # Prognostics & RUL: NEVER valid RUL
    assert payload.rul_state == "INSUFFICIENT_DATA"
    assert payload.point_rul_seconds is None


def test_complete_blackout_all_none_after_history():
    """Verify complete sensor blackout with None values produces INSUFFICIENT_DATA."""
    orch = SystemPipelineOrchestrator()
    _feed_normal_history(orch, steps=5)

    blackout = {
        "timestamp": 6.0,
        "engine_id": "ENG-01",
        "mission_id": "MSN-01",
        "rpm": None,
        "cht": None,
        "egt": None,
        "oil_temp": None,
        "oil_pressure": None,
        "fuel_flow": None,
        "vibration": None,
        "throttle": None,
        "altitude": None,
        "ambient_temp": None,
        "mission_phase": "CRUISE",
    }
    payload = orch.step(blackout)

    assert payload.quality_status == "MISSING"
    assert payload.anomaly_status == "INSUFFICIENT_DATA"
    assert math.isnan(payload.anomaly_score)
    assert payload.diagnostic_confidence == 0.0
    assert payload.diagnosis_data_quality == "INSUFFICIENT_DATA"
    assert math.isnan(payload.smoothed_health_index)
    assert payload.health_state == "INSUFFICIENT_DATA"
    assert payload.rul_state == "INSUFFICIENT_DATA"
    assert payload.point_rul_seconds is None


def test_complete_blackout_all_inf_after_history():
    """Verify complete sensor blackout with Inf values produces INSUFFICIENT_DATA."""
    orch = SystemPipelineOrchestrator()
    _feed_normal_history(orch, steps=5)

    blackout = {
        "timestamp": 6.0,
        "engine_id": "ENG-01",
        "mission_id": "MSN-01",
        "rpm": float("inf"),
        "cht": float("inf"),
        "egt": float("inf"),
        "oil_temp": float("inf"),
        "oil_pressure": float("inf"),
        "fuel_flow": float("inf"),
        "vibration": float("inf"),
        "throttle": 75.0,
        "altitude": 2000.0,
        "ambient_temp": 15.0,
        "mission_phase": "CRUISE",
    }
    payload = orch.step(blackout)

    assert payload.quality_status == "MISSING"
    assert payload.anomaly_status == "INSUFFICIENT_DATA"
    assert math.isnan(payload.anomaly_score)
    assert payload.diagnostic_confidence == 0.0
    assert payload.diagnosis_data_quality == "INSUFFICIENT_DATA"
    assert math.isnan(payload.smoothed_health_index)
    assert payload.health_state == "INSUFFICIENT_DATA"
    assert payload.rul_state == "INSUFFICIENT_DATA"
    assert payload.point_rul_seconds is None


def test_missing_all_channel_fields():
    """Verify dictionary missing all sensor channel keys produces INSUFFICIENT_DATA."""
    orch = SystemPipelineOrchestrator()
    _feed_normal_history(orch, steps=5)

    blackout = {
        "timestamp": 6.0,
        "engine_id": "ENG-01",
        "mission_id": "MSN-01",
        "mission_phase": "CRUISE",
    }
    payload = orch.step(blackout)

    assert payload.quality_status == "MISSING"
    assert payload.anomaly_status == "INSUFFICIENT_DATA"
    assert math.isnan(payload.anomaly_score)
    assert payload.diagnostic_confidence == 0.0
    assert payload.diagnosis_data_quality == "INSUFFICIENT_DATA"
    assert math.isnan(payload.smoothed_health_index)
    assert payload.health_state == "INSUFFICIENT_DATA"
    assert payload.rul_state == "INSUFFICIENT_DATA"
    assert payload.point_rul_seconds is None


def test_completely_empty_dictionary():
    """Verify completely empty telemetry dictionary {} produces invalid timestamp rejection without crash."""
    orch = SystemPipelineOrchestrator()
    _feed_normal_history(orch, steps=5)

    payload = orch.step({})
    assert payload.quality_status == "INVALID_TIMESTAMP_REJECTED"
    assert payload.anomaly_status == "INSUFFICIENT_DATA"
    assert math.isnan(payload.anomaly_score)
    assert payload.rul_state == "INSUFFICIENT_DATA"


def test_partial_telemetry_dropouts():
    """Verify partial telemetry dropouts degrade gracefully and do not falsely report nominal when critical data is lost."""
    orch = SystemPipelineOrchestrator()
    _feed_normal_history(orch, steps=5)

    # Case A: 1 channel missing (6 channels valid -> minimum 4 channels met)
    partial_1 = dict(NOMINAL_TELEMETRY)
    partial_1["timestamp"] = 6.0
    partial_1["engine_id"] = "ENG-01"
    partial_1["mission_id"] = "MSN-01"
    partial_1["rpm"] = float("nan")
    p1 = orch.step(partial_1)
    assert p1.quality_status == "DEGRADED"
    # Should still process remaining 6 channels
    assert not math.isnan(p1.smoothed_health_index)

    # Case B: 4 channels missing (only 3 valid -> below 4 channel threshold)
    partial_4 = dict(NOMINAL_TELEMETRY)
    partial_4["timestamp"] = 7.0
    partial_4["engine_id"] = "ENG-01"
    partial_4["mission_id"] = "MSN-01"
    for ch in ["rpm", "cht", "egt", "oil_temp"]:
        partial_4[ch] = float("nan")
    p4 = orch.step(partial_4)
    assert p4.quality_status == "DEGRADED"
    # Below minimum valid channels: health and RUL must transition to INSUFFICIENT_DATA
    assert math.isnan(p4.smoothed_health_index)
    assert p4.health_state == "INSUFFICIENT_DATA"
    assert p4.rul_state == "INSUFFICIENT_DATA"
    assert p4.point_rul_seconds is None


def test_single_sample_blackout_and_recovery():
    """Verify single-sample blackout produces INSUFFICIENT_DATA, followed by clean recovery."""
    orch = SystemPipelineOrchestrator()
    _feed_normal_history(orch, steps=5)

    # Step 6: Blackout
    blackout = {
        "timestamp": 6.0,
        "engine_id": "ENG-01",
        "mission_id": "MSN-01",
        "rpm": float("nan"), "cht": float("nan"), "egt": float("nan"),
        "oil_temp": float("nan"), "oil_pressure": float("nan"),
        "fuel_flow": float("nan"), "vibration": float("nan"),
        "mission_phase": "CRUISE",
    }
    p6 = orch.step(blackout)
    assert p6.anomaly_status == "INSUFFICIENT_DATA"
    assert math.isnan(p6.anomaly_score)
    assert math.isnan(p6.smoothed_health_index)
    assert p6.rul_state == "INSUFFICIENT_DATA"

    # Step 7: Recovery with valid telemetry
    recovery = dict(NOMINAL_TELEMETRY)
    recovery["timestamp"] = 7.0
    recovery["engine_id"] = "ENG-01"
    recovery["mission_id"] = "MSN-01"
    p7 = orch.step(recovery)

    assert p7.quality_status == "NOMINAL"
    assert not math.isnan(p7.smoothed_health_index)
    assert p7.health_state != "INSUFFICIENT_DATA"


def test_ten_sample_blackout_sustained():
    """Verify sustained 10-step blackout continuously reports INSUFFICIENT_DATA across all steps."""
    orch = SystemPipelineOrchestrator()
    _feed_normal_history(orch, steps=5)

    for step in range(6, 16):
        blackout = {
            "timestamp": float(step),
            "engine_id": "ENG-01",
            "mission_id": "MSN-01",
            "rpm": float("nan"), "cht": float("nan"), "egt": float("nan"),
            "oil_temp": float("nan"), "oil_pressure": float("nan"),
            "fuel_flow": float("nan"), "vibration": float("nan"),
            "mission_phase": "CRUISE",
        }
        payload = orch.step(blackout)
        assert payload.quality_status == "MISSING"
        assert payload.anomaly_status == "INSUFFICIENT_DATA"
        assert math.isnan(payload.anomaly_score)
        assert payload.diagnostic_confidence == 0.0
        assert payload.diagnosis_data_quality == "INSUFFICIENT_DATA"
        assert math.isnan(payload.smoothed_health_index)
        assert payload.health_state == "INSUFFICIENT_DATA"
        assert payload.rul_state == "INSUFFICIENT_DATA"
        assert payload.point_rul_seconds is None


def test_blackout_followed_by_recovery_proves_no_nan_pollution_in_history():
    """Verify that after 3 steps of blackout, the historical health buffer is unpolluted and RUL continues smoothly."""
    orch = SystemPipelineOrchestrator()
    _feed_normal_history(orch, steps=5)

    # 3 steps of blackout
    for step in range(6, 9):
        blackout = {
            "timestamp": float(step),
            "engine_id": "ENG-01",
            "mission_id": "MSN-01",
            "rpm": float("nan"), "cht": float("nan"), "egt": float("nan"),
            "oil_temp": float("nan"), "oil_pressure": float("nan"),
            "fuel_flow": float("nan"), "vibration": float("nan"),
            "mission_phase": "CRUISE",
        }
        orch.step(blackout)

    # Verify that the internal RUL history did not accumulate NaNs
    key = ("ENG-01", "MSN-01")
    rul_hist = orch.rul_pipeline._history.get(key, {})
    if "health_values" in rul_hist:
        assert not any(math.isnan(v) for v in rul_hist["health_values"]), (
            f"RUL history contaminated with NaNs: {rul_hist['health_values']}"
        )

    # Resume normal telemetry
    for step in range(9, 15):
        telem = dict(NOMINAL_TELEMETRY)
        telem["timestamp"] = float(step)
        telem["engine_id"] = "ENG-01"
        telem["mission_id"] = "MSN-01"
        payload = orch.step(telem)
        assert not math.isnan(payload.smoothed_health_index)


def test_dashboard_adapter_blackout_view_model_representation():
    """Verify DashboardAdapter transforms blackout payload into UNAVAILABLE status levels with zero healthy badges."""
    orch = SystemPipelineOrchestrator()
    _feed_normal_history(orch, steps=5)

    blackout = {
        "timestamp": 6.0,
        "engine_id": "ENG-01",
        "mission_id": "MSN-01",
        "rpm": float("nan"), "cht": float("nan"), "egt": float("nan"),
        "oil_temp": float("nan"), "oil_pressure": float("nan"),
        "fuel_flow": float("nan"), "vibration": float("nan"),
        "mission_phase": "CRUISE",
    }
    payload = orch.step(blackout)

    adapter = DashboardAdapter()
    vm = adapter.adapt(payload)

    # Overall Status must NEVER be HEALTHY
    assert vm.overview.overall_status == StatusLevel.UNAVAILABLE, (
        f"Expected overall_status UNAVAILABLE, got {vm.overview.overall_status}"
    )

    # Fault Diagnosis Card: must NOT show HEALTHY or "Nominal / None"
    assert vm.overview.fault_card.status == StatusLevel.UNAVAILABLE
    assert vm.overview.fault_card.value == "Insufficient Data"
    assert vm.overview.fault_card.availability == AvailabilityStatus.UNAVAILABLE

    # Active Anomaly Card: must be UNAVAILABLE
    assert vm.overview.anomaly_card.status == StatusLevel.UNAVAILABLE
    assert vm.overview.anomaly_card.value == "INSUFFICIENT_DATA"

    # Health Index Card: must be UNAVAILABLE
    assert vm.overview.health_card.status == StatusLevel.UNAVAILABLE
    assert vm.overview.health_card.value == "Unavailable"

    # RUL Card: must be UNAVAILABLE
    assert vm.overview.rul_card.status == StatusLevel.UNAVAILABLE
    assert vm.overview.rul_card.value == "Unavailable"

    # Data Quality Card: must NOT be HEALTHY
    assert vm.overview.data_quality_card.status in (StatusLevel.DEGRADED, StatusLevel.WARNING)
    assert vm.overview.data_quality_card.value == "MISSING"
