"""
Regression and verification tests for centralized telemetry timestamp integrity policy.

Verifies that duplicate timestamps (dt == 0), backwards/out-of-order timestamps (dt < 0),
and missing/non-finite timestamps are rejected according to contract without advancing
or corrupting pipeline internal states:
- Digital Twin dynamic thermal states
- Anomaly Detection EWMA and persistence counters
- Health Index EWMA smoother and degradation tracker
- Forecasting causal telemetry buffer
- Prognostics trajectory synthesizer and RUL
- System Orchestrator step counting and session tracking
- Dashboard Adapter view model generation
"""

import math
import pytest
from typing import Dict, Any, List

from orchestrator.pipeline import SystemPipelineOrchestrator
from orchestrator.schema import DashboardStatePayload
from health_index.smoothing import CausalEWMASmoother
from health_index.pipeline import HealthIndexPipeline
from health_index.schema import DegradationTrend
from digital_twin.twin_model import DigitalTwin
from dashboard.services.adapter import DashboardAdapter
from dashboard.schemas.view_model import StatusLevel


def _make_telemetry_sample(
    timestamp: float,
    cht: float = 110.0,
    rpm: float = 5200.0,
    engine_id: str = "ENG_TIME_01",
    mission_id: str = "MIS_TIME_01",
) -> Dict[str, Any]:
    """Helper to construct valid telemetry record dictionary."""
    return {
        "timestamp": timestamp,
        "engine_id": engine_id,
        "mission_id": mission_id,
        "rpm": rpm,
        "cht": cht,
        "egt": 685.0,
        "oil_temp": 87.0,
        "oil_pressure": 4.4,
        "fuel_flow": 18.1,
        "vibration": 0.55,
        "throttle": 75.0,
        "load": 65.0,
        "altitude": 2000.0,
        "ambient_temp": 15.0,
        "mission_phase": "CRUISE",
    }


# ==============================================================================
# Test 1: Duplicate Identical Sample Rejection
# ==============================================================================
def test_duplicate_identical_sample_rejected():
    """An exact duplicate identical sample returns DUPLICATE_TIMESTAMP_REJECTED and does not advance state."""
    orch = SystemPipelineOrchestrator()
    orch.reset(engine_id="ENG_01", mission_id="MIS_01")

    s1 = _make_telemetry_sample(1000.0, cht=110.0)
    s2 = _make_telemetry_sample(1001.0, cht=112.0)
    s2_dup = _make_telemetry_sample(1001.0, cht=112.0)

    p1 = orch.step(s1)
    p2 = orch.step(s2)
    step_count_before = orch.step_count
    last_ts_before = orch.last_timestamp

    p2_rejected = orch.step(s2_dup)

    assert p2_rejected.quality_status == "DUPLICATE_TIMESTAMP_REJECTED"
    assert p2_rejected.diagnosis_data_quality == "DUPLICATE_TIMESTAMP"
    assert p2_rejected.forecast_status == "REJECTED_DUPLICATE"
    assert math.isnan(p2_rejected.raw_health_index)
    assert math.isnan(p2_rejected.smoothed_health_index)
    assert orch.step_count == step_count_before
    assert orch.last_timestamp == last_ts_before


# ==============================================================================
# Test 2: Duplicate Timestamp with Different Values Rejection
# ==============================================================================
def test_duplicate_timestamp_different_values_rejected():
    """Duplicate timestamp with conflicting/adversarial sensor values is rejected without corrupting states."""
    orch = SystemPipelineOrchestrator()
    orch.reset(engine_id="ENG_01", mission_id="MIS_01")

    s1 = _make_telemetry_sample(1000.0, cht=100.0)
    s2 = _make_telemetry_sample(1001.0, cht=105.0)
    # Adversarial duplicate trying to inject extreme redline CHT at same timestamp
    s2_adv = _make_telemetry_sample(1001.0, cht=280.0, rpm=7200.0)
    s3 = _make_telemetry_sample(1002.0, cht=106.0)

    # Reference clean run
    orch_ref = SystemPipelineOrchestrator()
    orch_ref.reset(engine_id="ENG_01", mission_id="MIS_01")
    ref_p1 = orch_ref.step(s1)
    ref_p2 = orch_ref.step(s2)
    ref_p3 = orch_ref.step(s3)

    # Injected run
    orch.step(s1)
    orch.step(s2)
    adv_rejected = orch.step(s2_adv)
    inj_p3 = orch.step(s3)

    assert adv_rejected.quality_status == "DUPLICATE_TIMESTAMP_REJECTED"
    # State at t=1002 must match clean reference run with zero corruption
    assert inj_p3.smoothed_health_index == ref_p3.smoothed_health_index
    assert inj_p3.anomaly_status == ref_p3.anomaly_status
    assert inj_p3.anomaly_score == ref_p3.anomaly_score
    assert orch.step_count == orch_ref.step_count


# ==============================================================================
# Test 3: Ten Repeated Duplicates Stability
# ==============================================================================
def test_ten_repeated_duplicates_stability():
    """Flooding with 10 repeated duplicate timestamps causes zero state drift or counter advancement."""
    orch = SystemPipelineOrchestrator()
    orch.reset(engine_id="ENG_01", mission_id="MIS_01")

    orch.step(_make_telemetry_sample(1000.0, cht=100.0))
    p_baseline = orch.step(_make_telemetry_sample(1001.0, cht=105.0))
    baseline_step = orch.step_count
    baseline_ts = orch.last_timestamp

    for i in range(10):
        dup_payload = orch.step(_make_telemetry_sample(1001.0, cht=105.0))
        assert dup_payload.quality_status == "DUPLICATE_TIMESTAMP_REJECTED"
        assert orch.step_count == baseline_step
        assert orch.last_timestamp == baseline_ts

    # Next valid timestep
    p_next = orch.step(_make_telemetry_sample(1002.0, cht=106.0))

    # Compare with reference run
    orch_clean = SystemPipelineOrchestrator()
    orch_clean.reset(engine_id="ENG_01", mission_id="MIS_01")
    orch_clean.step(_make_telemetry_sample(1000.0, cht=100.0))
    orch_clean.step(_make_telemetry_sample(1001.0, cht=105.0))
    clean_next = orch_clean.step(_make_telemetry_sample(1002.0, cht=106.0))

    assert p_next.smoothed_health_index == clean_next.smoothed_health_index
    assert p_next.anomaly_score == clean_next.anomaly_score
    assert orch.step_count == orch_clean.step_count


# ==============================================================================
# Test 4: Duplicate in Middle of Mission Sequence
# ==============================================================================
def test_duplicate_in_middle_of_mission():
    """Validates: T1=1000, T2=1001, T2dup=1001, T3=1002 produces exact state equivalence with T1, T2, T3."""
    orch = SystemPipelineOrchestrator()
    orch.reset(engine_id="ENG_01", mission_id="MIS_01")

    p1 = orch.step(_make_telemetry_sample(1000.0, cht=110.0))
    p2 = orch.step(_make_telemetry_sample(1001.0, cht=115.0))
    p2_dup = orch.step(_make_telemetry_sample(1001.0, cht=115.0))
    p3 = orch.step(_make_telemetry_sample(1002.0, cht=120.0))

    orch_clean = SystemPipelineOrchestrator()
    orch_clean.reset(engine_id="ENG_01", mission_id="MIS_01")
    c1 = orch_clean.step(_make_telemetry_sample(1000.0, cht=110.0))
    c2 = orch_clean.step(_make_telemetry_sample(1001.0, cht=115.0))
    c3 = orch_clean.step(_make_telemetry_sample(1002.0, cht=120.0))

    assert p2_dup.quality_status == "DUPLICATE_TIMESTAMP_REJECTED"
    assert p3.smoothed_health_index == c3.smoothed_health_index
    assert p3.anomaly_score == c3.anomaly_score
    assert p3.predicted_fault_class == c3.predicted_fault_class
    assert orch.step_count == orch_clean.step_count == 3


# ==============================================================================
# Test 5: Duplicate Final Sample
# ==============================================================================
def test_duplicate_final_sample():
    """Mission ending with duplicate observation safely rejects terminal duplicate."""
    orch = SystemPipelineOrchestrator()
    orch.reset(engine_id="ENG_01", mission_id="MIS_01")

    orch.step(_make_telemetry_sample(1000.0))
    orch.step(_make_telemetry_sample(1001.0))
    p_final = orch.step(_make_telemetry_sample(1002.0))
    p_final_dup = orch.step(_make_telemetry_sample(1002.0))

    assert p_final.quality_status != "DUPLICATE_TIMESTAMP_REJECTED"
    assert p_final_dup.quality_status == "DUPLICATE_TIMESTAMP_REJECTED"
    assert orch.step_count == 3
    assert orch.last_timestamp == 1002.0


# ==============================================================================
# Test 6: Missing and Non-Numeric Timestamp Handling
# ==============================================================================
def test_missing_and_non_numeric_timestamps():
    """Missing, None, NaN, and string non-numeric timestamps are rejected without crashing."""
    orch = SystemPipelineOrchestrator()
    orch.reset(engine_id="ENG_01", mission_id="MIS_01")

    # 1. Missing timestamp key
    sample_no_ts = _make_telemetry_sample(1000.0)
    del sample_no_ts["timestamp"]
    p_no_ts = orch.step(sample_no_ts)
    assert p_no_ts.quality_status == "INVALID_TIMESTAMP_REJECTED"
    assert "Missing timestamp" in p_no_ts.summary_explanation

    # 2. None timestamp
    sample_none = _make_telemetry_sample(1000.0)
    sample_none["timestamp"] = None
    p_none = orch.step(sample_none)
    assert p_none.quality_status == "INVALID_TIMESTAMP_REJECTED"

    # 3. NaN timestamp
    sample_nan = _make_telemetry_sample(1000.0)
    sample_nan["timestamp"] = float("nan")
    p_nan = orch.step(sample_nan)
    assert p_nan.quality_status == "INVALID_TIMESTAMP_REJECTED"

    # 4. Non-numeric string timestamp
    sample_str = _make_telemetry_sample(1000.0)
    sample_str["timestamp"] = "not_a_number"
    p_str = orch.step(sample_str)
    assert p_str.quality_status == "INVALID_TIMESTAMP_REJECTED"

    # System should still be uninitialized with zero step count
    assert orch.step_count == 0
    assert orch.last_timestamp is None

    # Subsequent valid observation processes normally
    p_valid = orch.step(_make_telemetry_sample(1000.0))
    assert p_valid.quality_status == "NOMINAL"
    assert orch.step_count == 1
    assert orch.last_timestamp == 1000.0


# ==============================================================================
# Test 7: Out-of-Order Timestamp Monotonicity Preservation
# ==============================================================================
def test_out_of_order_timestamp_preservation():
    """Backwards/decreasing timestamps are rejected with OUT_OF_ORDER_REJECTED and do not corrupt timeline."""
    orch = SystemPipelineOrchestrator()
    orch.reset(engine_id="ENG_01", mission_id="MIS_01")

    p1 = orch.step(_make_telemetry_sample(1000.0))
    p_ooo = orch.step(_make_telemetry_sample(995.0))  # -5s backwards
    p2 = orch.step(_make_telemetry_sample(1001.0))   # Causal forward step

    assert p1.quality_status == "NOMINAL"
    assert p_ooo.quality_status == "OUT_OF_ORDER_REJECTED"
    assert p_ooo.diagnosis_data_quality == "OUT_OF_ORDER"
    assert p2.quality_status == "NOMINAL"
    assert orch.step_count == 2
    assert orch.last_timestamp == 1001.0


# ==============================================================================
# Test 8: Subsystem Level - Causal EWMA Smoother dt <= 0 Robustness
# ==============================================================================
def test_causal_ewma_smoother_non_positive_dt():
    """CausalEWMASmoother preserves filter state when receiving dt == 0 or dt < 0."""
    smoother = CausalEWMASmoother(alpha=0.15)
    h1 = smoother.update(1.0, timestamp=100.0)
    h2 = smoother.update(0.5, timestamp=101.0)

    # Duplicate timestamp with adversarial 0.0 value
    h2_dup = smoother.update(0.0, timestamp=101.0)
    assert h2_dup == h2

    # Out-of-order timestamp
    h2_ooo = smoother.update(0.0, timestamp=99.0)
    assert h2_ooo == h2

    # Subsequent valid step
    h3 = smoother.update(0.5, timestamp=102.0)

    # Reference run without duplicates
    smoother_ref = CausalEWMASmoother(alpha=0.15)
    ref_h1 = smoother_ref.update(1.0, timestamp=100.0)
    ref_h2 = smoother_ref.update(0.5, timestamp=101.0)
    ref_h3 = smoother_ref.update(0.5, timestamp=102.0)

    assert h3 == ref_h3


# ==============================================================================
# Test 9: Subsystem Level - Digital Twin Non-Positive dt Robustness
# ==============================================================================
def test_digital_twin_non_positive_dt():
    """DigitalTwin does not integrate dynamics forward when dt <= 0."""
    twin = DigitalTwin()
    twin.reset()

    # Step 1
    s1 = _make_telemetry_sample(100.0)
    from telemetry.schema import TelemetryRecord
    rec1 = TelemetryRecord(**{k: v for k, v in s1.items() if k in TelemetryRecord.__annotations__})
    state1 = twin.update(rec1)

    # Step 2: Duplicate timestamp
    rec2 = TelemetryRecord(**{k: v for k, v in s1.items() if k in TelemetryRecord.__annotations__})
    state2 = twin.update(rec2)

    # Expected values should remain identical since dt = 0
    assert state2.nominal_estimates["expected_cht"] == state1.nominal_estimates["expected_cht"]
    assert state2.nominal_estimates["expected_egt"] == state1.nominal_estimates["expected_egt"]
    assert state2.nominal_estimates["expected_oil_temp"] == state1.nominal_estimates["expected_oil_temp"]


# ==============================================================================
# Test 10: Dashboard Adapter Payload Rendering
# ==============================================================================
def test_dashboard_adapter_handles_duplicate_rejection_payload():
    """Dashboard adapter formats DUPLICATE_TIMESTAMP_REJECTED payload cleanly as DEGRADED data quality."""
    orch = SystemPipelineOrchestrator()
    orch.reset(engine_id="ENG_01", mission_id="MIS_01")

    orch.step(_make_telemetry_sample(1000.0))
    dup_payload = orch.step(_make_telemetry_sample(1000.0))

    adapter = DashboardAdapter()
    view_model = adapter.adapt(dup_payload)

    assert view_model.data_quality.quality_status == "DUPLICATE_TIMESTAMP_REJECTED"
    assert view_model.overview.data_quality_card.status == StatusLevel.DEGRADED
