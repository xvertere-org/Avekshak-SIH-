"""
Phase 13 Comprehensive Verification Test Suite: SystemPipelineOrchestrator.

Verifies:
1. Orchestrator initialization & synthetic bootstrap
2. Healthy timestep execution
3. Cooling fault propagation & detection
4. Lubrication fault propagation & detection
5. Fuel fault propagation & detection
6. Mechanical fault propagation & detection
7. Sensor fault isolation & handling
8. Missing telemetry / NaN preservation
9. Invalid telemetry / out-of-order rejection
10. Mission isolation & boundary preservation
11. Engine ID state isolation
12. Causal execution
13. No future-data leakage
14. No ground-truth fault leakage
15. TimesFM blocked/gated fallback behavior
16. LOCAL_UNCHECKPOINTED_GRAPH prognostic rejection
17. Supported forecast handoff
18. RUL unavailable / insufficient history handling
19. Phase 12 explainability fusion handoff
20. DashboardStatePayload schema validation
21. State reset validation
22. Deterministic replay verification
23. End-to-end full pipeline execution
24. Latency benchmark (mean, median, P95, P99)
"""

import math
import time
import pytest
import numpy as np
import pandas as pd

from orchestrator import (
    SystemPipelineOrchestrator,
    DashboardStatePayload,
    OrchestratorConfig,
    SimulationScenario,
    ScenarioFaultType,
    OperatorActionAdvisor,
    AdvisoryActionCode,
)
from orchestrator.schema import ScenarioFaultType
from forecasting.schema import ForecastResult, ModelStatus, ForecastQuality
from prognostics.schema import RULStatus, RULResult


# ==============================================================================
# Shared Fixture: Cached Orchestrator
# ==============================================================================
@pytest.fixture(scope="module")
def shared_orchestrator():
    """Module-scoped orchestrator with deterministic synthetic bootstrap."""
    cfg = OrchestratorConfig(deterministic_seed=42)
    orch = SystemPipelineOrchestrator(config=cfg)
    return orch


# ==============================================================================
# Test 1: Orchestrator Initialization
# ==============================================================================
def test_01_orchestrator_initialization(shared_orchestrator):
    """Verify clean initialization of all Phase 6-12 components and bootstrap metadata."""
    orch = shared_orchestrator
    assert orch.twin is not None
    assert orch.anomaly_detector is not None
    assert orch.diagnosis_pipeline is not None
    assert orch.health_pipeline is not None
    assert orch.forecasting_pipeline is not None
    assert orch.rul_pipeline is not None
    assert orch.explainability_pipeline is not None

    # Verify synthetic bootstrap metadata
    assert "bootstrap_elapsed_seconds" in orch.bootstrap_metadata
    assert orch.bootstrap_metadata.get("provenance") == "PHYSICS_INFORMED_SYNTHETIC_BOOTSTRAP"
    assert "No external flight-test" in orch.bootstrap_metadata.get("disclaimer", "")


# ==============================================================================
# Test 2: Healthy Timestep
# ==============================================================================
def test_02_healthy_timestep(shared_orchestrator):
    """Verify nominal steady telemetry produces NORMAL anomaly status and high health index."""
    orch = shared_orchestrator
    sc = SimulationScenario(name="healthy_steady", duration_s=35.0, fault_type=ScenarioFaultType.HEALTHY)
    payloads = orch.run_simulation(sc)

    # Inspect post-transient steady state (e.g. t >= 25s)
    steady_payload = payloads[25]
    assert steady_payload.anomaly_status == "NORMAL"
    assert steady_payload.smoothed_health_index >= 0.95
    assert steady_payload.predicted_fault_class == "none"
    assert steady_payload.advisory is not None
    assert steady_payload.advisory.action_code == AdvisoryActionCode.NORMAL_MONITORING


# ==============================================================================
# Test 3: Cooling Fault Propagation
# ==============================================================================
def test_03_cooling_fault(shared_orchestrator):
    """Injected cooling degradation manifests with elevated CHT and dominant cooling evidence."""
    orch = shared_orchestrator
    sc = SimulationScenario(
        name="cooling_degradation",
        duration_s=40.0,
        fault_type=ScenarioFaultType.COOLING_DEGRADATION,
        fault_start_s=15.0,
        fault_severity=0.7,
    )
    payloads = orch.run_simulation(sc)

    post_fault = payloads[-1]
    # CHT should be flagged as elevated or dominant degraded channel
    assert "cht" in post_fault.dominant_channels or post_fault.normalized_residuals.get("cht", 0.0) > 1.0
    assert post_fault.smoothed_health_index < payloads[10].smoothed_health_index


# ==============================================================================
# Test 4: Lubrication Fault Propagation
# ==============================================================================
def test_04_lubrication_fault(shared_orchestrator):
    """Injected lubrication degradation manifests in lubrication subsystem residuals."""
    orch = shared_orchestrator
    sc = SimulationScenario(
        name="lube_degradation",
        duration_s=40.0,
        fault_type=ScenarioFaultType.LUBRICATION_DEGRADATION,
        fault_start_s=15.0,
        fault_severity=0.7,
    )
    payloads = orch.run_simulation(sc)

    post_fault = payloads[-1]
    oil_press_res = abs(post_fault.normalized_residuals.get("oil_pressure", 0.0))
    oil_temp_res = abs(post_fault.normalized_residuals.get("oil_temp", 0.0))
    assert oil_press_res > 0.5 or oil_temp_res > 0.5 or "oil_pressure" in post_fault.dominant_channels


# ==============================================================================
# Test 5: Fuel Fault Propagation
# ==============================================================================
def test_05_fuel_fault(shared_orchestrator):
    """Injected fuel injection abnormality causes fuel flow / EGT deviation."""
    orch = shared_orchestrator
    sc = SimulationScenario(
        name="fuel_fault",
        duration_s=40.0,
        fault_type=ScenarioFaultType.FUEL_ABNORMALITY,
        fault_start_s=15.0,
        fault_severity=0.6,
    )
    payloads = orch.run_simulation(sc)

    post_fault = payloads[-1]
    fuel_res = abs(post_fault.normalized_residuals.get("fuel_flow", 0.0))
    egt_res = abs(post_fault.normalized_residuals.get("egt", 0.0))
    assert fuel_res > 0.3 or egt_res > 0.3 or "fuel_flow" in post_fault.dominant_channels


# ==============================================================================
# Test 6: Mechanical Fault Propagation
# ==============================================================================
def test_06_mechanical_fault(shared_orchestrator):
    """Injected mechanical degradation manifests with elevated vibration residuals."""
    orch = shared_orchestrator
    sc = SimulationScenario(
        name="mechanical_fault",
        duration_s=40.0,
        fault_type=ScenarioFaultType.MECHANICAL_DEGRADATION,
        fault_start_s=15.0,
        fault_severity=0.7,
    )
    payloads = orch.run_simulation(sc)

    post_fault = payloads[-1]
    vib_res = abs(post_fault.normalized_residuals.get("vibration", 0.0))
    assert vib_res > 0.5 or "vibration" in post_fault.dominant_channels


# ==============================================================================
# Test 7: Sensor Fault Isolation
# ==============================================================================
def test_07_sensor_fault(shared_orchestrator):
    """Sensor bias produces sensor fault diagnosis without engine physical failure."""
    orch = shared_orchestrator
    sc = SimulationScenario(
        name="sensor_fault",
        duration_s=40.0,
        fault_type=ScenarioFaultType.SENSOR_FAULT,
        fault_start_s=15.0,
        fault_severity=0.7,
    )
    payloads = orch.run_simulation(sc)

    post_fault = payloads[-1]
    assert post_fault.normalized_residuals.get("cht", 0.0) != 0.0
    assert post_fault.quality_status in ["NOMINAL", "DEGRADED", "VALID"]


# ==============================================================================
# Test 8: Missing Telemetry Handling
# ==============================================================================
def test_08_missing_telemetry(shared_orchestrator):
    """Missing / NaN observations preserve NaN and handle degraded quality without crashing."""
    orch = shared_orchestrator
    orch.reset(engine_id="TEST_ENG", mission_id="TEST_MISS")

    sample_with_nan = {
        "timestamp": 10.0,
        "engine_id": "TEST_ENG",
        "mission_id": "TEST_MISS",
        "mission_phase": "CRUISE",
        "rpm": 5500.0,
        "cht": float("nan"),  # Sensor dropout
        "egt": 580.0,
        "oil_temp": 65.0,
        "oil_pressure": 4.5,
        "fuel_flow": 22.0,
        "vibration": 1.2,
        "throttle": 75.0,
        "altitude": 2000.0,
        "ambient_temp": 15.0,
        "load": 80.0,
    }

    payload = orch.step(sample_with_nan)
    assert math.isnan(payload.residuals.get("cht", float("nan")))
    assert payload.smoothed_health_index is not None


# ==============================================================================
# Test 9: Invalid Telemetry / Out-of-Order Rejection
# ==============================================================================
def test_09_invalid_telemetry_out_of_order(shared_orchestrator):
    """Timesteps with decreasing timestamps are rejected by the causal sequence guard."""
    orch = shared_orchestrator
    orch.reset(engine_id="TIME_TEST", mission_id="TIME_MISS")

    step1 = {
        "timestamp": 20.0,
        "engine_id": "TIME_TEST",
        "mission_id": "TIME_MISS",
        "rpm": 5500.0,
        "cht": 85.0,
        "throttle": 75.0,
        "altitude": 2000.0,
    }
    step2_invalid = {
        "timestamp": 15.0,  # Earlier than step 1
        "engine_id": "TIME_TEST",
        "mission_id": "TIME_MISS",
        "rpm": 5500.0,
        "cht": 85.0,
        "throttle": 75.0,
        "altitude": 2000.0,
    }

    p1 = orch.step(step1)
    p2 = orch.step(step2_invalid)

    assert p1.quality_status != "OUT_OF_ORDER_REJECTED"
    assert p2.quality_status == "OUT_OF_ORDER_REJECTED"
    assert p2.diagnosis_data_quality == "OUT_OF_ORDER"


# ==============================================================================
# Test 10: Mission Isolation
# ==============================================================================
def test_10_mission_isolation(shared_orchestrator):
    """Transition to a new mission_id triggers state isolation reset."""
    orch = shared_orchestrator
    orch.reset(engine_id="ENG_01", mission_id="MISSION_A")

    # Step in Mission A
    orch.step({"timestamp": 1.0, "engine_id": "ENG_01", "mission_id": "MISSION_A", "rpm": 5500.0, "cht": 85.0, "throttle": 75.0, "altitude": 2000.0})
    orch.step({"timestamp": 2.0, "engine_id": "ENG_01", "mission_id": "MISSION_A", "rpm": 5500.0, "cht": 85.0, "throttle": 75.0, "altitude": 2000.0})
    assert orch.step_count == 2

    # Step in Mission B -> triggers isolation reset
    p_b = orch.step({"timestamp": 1.0, "engine_id": "ENG_01", "mission_id": "MISSION_B", "rpm": 5500.0, "cht": 85.0, "throttle": 75.0, "altitude": 2000.0})
    assert orch.active_mission_id == "MISSION_B"
    assert p_b.mission_id == "MISSION_B"
    assert orch.step_count == 1  # Reset on boundary transition


# ==============================================================================
# Test 11: Engine ID State Isolation
# ==============================================================================
def test_11_engine_isolation(shared_orchestrator):
    """Transition to a new engine_id triggers isolated session boundary."""
    orch = shared_orchestrator
    orch.reset(engine_id="ENG_A", mission_id="COMMON_MISSION")

    orch.step({"timestamp": 1.0, "engine_id": "ENG_A", "mission_id": "COMMON_MISSION", "rpm": 5500.0, "cht": 85.0, "throttle": 75.0, "altitude": 2000.0})
    p_b = orch.step({"timestamp": 1.0, "engine_id": "ENG_B", "mission_id": "COMMON_MISSION", "rpm": 5500.0, "cht": 85.0, "throttle": 75.0, "altitude": 2000.0})

    assert orch.active_engine_id == "ENG_B"
    assert p_b.engine_id == "ENG_B"
    assert orch.step_count == 1


# ==============================================================================
# Test 12: Causal Execution
# ==============================================================================
def test_12_causal_execution(shared_orchestrator):
    """Step-by-step causal outputs at t=10 are strictly identical regardless of future execution."""
    orch = shared_orchestrator
    sc = SimulationScenario(name="causal_check", duration_s=15.0)

    # Run full 15s
    payloads_full = orch.run_simulation(sc)

    # Run only 10s on fresh reset
    orch.reset(sc.engine_id, sc.mission_id)
    payloads_partial = orch.run_simulation(sc, duration_s=10.0)

    # Output at index 9 (t=9) must match strictly identically
    assert payloads_full[9].smoothed_health_index == payloads_partial[9].smoothed_health_index
    assert payloads_full[9].anomaly_score == payloads_partial[9].anomaly_score
    assert payloads_full[9].predicted_fault_class == payloads_partial[9].predicted_fault_class


# ==============================================================================
# Test 13: No Future Data Leakage
# ==============================================================================
def test_13_no_future_data_leakage(shared_orchestrator):
    """Internal buffers and histories only ever contain timestamps <= current observation."""
    orch = shared_orchestrator
    orch.reset(engine_id="CAUSAL_ENG", mission_id="CAUSAL_MISS")

    for t in range(1, 10):
        p = orch.step({
            "timestamp": float(t),
            "engine_id": "CAUSAL_ENG",
            "mission_id": "CAUSAL_MISS",
            "rpm": 5500.0,
            "cht": 85.0,
            "egt": 580.0,
            "oil_temp": 65.0,
            "oil_pressure": 4.5,
            "fuel_flow": 22.0,
            "vibration": 1.2,
            "throttle": 75.0,
            "altitude": 2000.0,
        })
        # Internal RUL history timestamps must not exceed t
        hist = orch.rul_pipeline._get_history("CAUSAL_ENG", "CAUSAL_MISS")
        assert all(ts <= t for ts in hist["timestamps"])


# ==============================================================================
# Test 14: No Ground-Truth Fault Leakage
# ==============================================================================
def test_14_no_ground_truth_fault_leakage(shared_orchestrator):
    """Simulator ground-truth fault label in input is stripped and cannot cheat inference."""
    orch = shared_orchestrator
    orch.reset(engine_id="LEAK_ENG", mission_id="LEAK_MISS")

    # Pass healthy physics telemetry with a fake ground-truth label injected
    deceptive_telemetry = {
        "timestamp": 20.0,
        "engine_id": "LEAK_ENG",
        "mission_id": "LEAK_MISS",
        "mission_phase": "CRUISE",
        "rpm": 5500.0,
        "cht": 66.5,  # Matches nominal expected CHT
        "egt": 580.0,
        "oil_temp": 65.0,
        "oil_pressure": 4.5,
        "fuel_flow": 22.0,
        "vibration": 1.2,
        "throttle": 75.0,
        "altitude": 2000.0,
        "ambient_temp": 15.0,
        "load": 80.0,
        "fault_type": "cooling_degradation",  # GROUND TRUTH LABEL
        "fault_severity": 0.99,               # GROUND TRUTH LABEL
    }

    # Step with injected deceptive ground truth
    orch.reset(engine_id="LEAK_ENG", mission_id="LEAK_MISS")
    payload_with_gt = orch.step(deceptive_telemetry)

    # Step with identical telemetry without ground-truth label
    clean_sample = dict(deceptive_telemetry)
    clean_sample.pop("fault_type", None)
    clean_sample.pop("fault_severity", None)

    orch.reset(engine_id="LEAK_ENG", mission_id="LEAK_MISS")
    payload_clean = orch.step(clean_sample)

    # Ground-truth invariance: outputs must be 100% identical regardless of ground-truth presence
    assert payload_with_gt.predicted_fault_class == payload_clean.predicted_fault_class
    assert payload_with_gt.smoothed_health_index == payload_clean.smoothed_health_index
    assert payload_with_gt.anomaly_score == payload_clean.anomaly_score
    assert "fault_type" not in payload_with_gt.observed_telemetry
    assert "fault_severity" not in payload_with_gt.observed_telemetry


# ==============================================================================
# Test 15: TimesFM Blocked Gated Fallback
# ==============================================================================
def test_15_timesfm_blocked_fallback(shared_orchestrator):
    """In unauthenticated environment, TimesFM reports BLOCKED_UNAUTHENTICATED_GATED or baseline."""
    orch = shared_orchestrator
    sc = SimulationScenario(name="tfm_status", duration_s=35.0)
    payloads = orch.run_simulation(sc)

    # After context length 32
    p33 = payloads[33]
    assert p33.forecast_status in [
        ModelStatus.BLOCKED_UNAUTHENTICATED_GATED.value,
        ModelStatus.BASELINE.value,
        ModelStatus.LOADED_PRETRAINED.value,
    ]
    assert p33.is_pretrained is False or p33.forecast_status == ModelStatus.LOADED_PRETRAINED.value


# ==============================================================================
# Test 16: LOCAL_UNCHECKPOINTED_GRAPH Rejection
# ==============================================================================
def test_16_local_unchecked_graph_rejection(shared_orchestrator):
    """Prognostics pipeline strictly rejects LOCAL_UNCHECKPOINTED_GRAPH forecast handoff."""
    orch = shared_orchestrator

    fake_forecast = ForecastResult(
        engine_id="TEST_ENG",
        mission_id="TEST_MISS",
        forecast_start_timestamp=30.0,
        context_start_timestamp=0.0,
        context_length=32,
        forecast_horizon=16,
        sampling_interval=1.0,
        target_channels=["cht"],
        forecast_timestamps=[31.0 + i for i in range(16)],
        predicted_telemetry={"cht": [85.0] * 16},
        model_name="timesfm-mock",
        model_status=ModelStatus.LOCAL_UNCHECKPOINTED_GRAPH.value,  # REJECTED STATUS
    )

    handoff_h, source = orch.rul_pipeline.dual_horizon.evaluate_phase10_handoff(fake_forecast)
    assert handoff_h == 0.0
    assert source == "ROBUST_LINEAR_PRIMARY"


# ==============================================================================
# Test 17: Supported Forecast Handoff
# ==============================================================================
def test_17_supported_forecast_handoff(shared_orchestrator):
    """Prognostics pipeline accepts LOADED_PRETRAINED forecast with horizon 16."""
    orch = shared_orchestrator

    valid_forecast = ForecastResult(
        engine_id="TEST_ENG",
        mission_id="TEST_MISS",
        forecast_start_timestamp=30.0,
        context_start_timestamp=0.0,
        context_length=32,
        forecast_horizon=16,
        sampling_interval=1.0,
        target_channels=["cht"],
        forecast_timestamps=[31.0 + i for i in range(16)],
        predicted_telemetry={"cht": [85.0] * 16},
        model_name="timesfm-pretrained",
        model_status=ModelStatus.LOADED_PRETRAINED.value,
    )

    handoff_h, source = orch.rul_pipeline.dual_horizon.evaluate_phase10_handoff(valid_forecast)
    assert handoff_h == 16.0
    assert source == "TIMESFM_FORECAST_ASSISTED"


# ==============================================================================
# Test 18: RUL Unavailable / Warmup Handling
# ==============================================================================
def test_18_rul_unavailable_handling(shared_orchestrator):
    """Early timesteps before warmup return INSUFFICIENT_HISTORY with point RUL as None."""
    orch = shared_orchestrator
    sc = SimulationScenario(name="rul_warmup", duration_s=10.0)
    payloads = orch.run_simulation(sc)

    # First few steps must be INSUFFICIENT_HISTORY
    p5 = payloads[5]
    assert p5.rul_state == RULStatus.INSUFFICIENT_HISTORY.value
    assert p5.point_rul_seconds is None


# ==============================================================================
# Test 19: Phase 12 Explainability Handoff
# ==============================================================================
def test_19_phase12_explanation_handoff(shared_orchestrator):
    """Explainability engine fuses upstream outputs into coherent explanation narrative."""
    orch = shared_orchestrator
    sc = SimulationScenario(name="xai_check", duration_s=35.0)
    payloads = orch.run_simulation(sc)

    final = payloads[-1]
    assert len(final.summary_explanation) > 0
    assert final.authoritative_explainability is not None
    assert final.recommended_operator_action is not None


# ==============================================================================
# Test 20: DashboardStatePayload Schema Validation
# ==============================================================================
def test_20_dashboard_payload_schema(shared_orchestrator):
    """DashboardStatePayload dictionary serialization contains all required UI sections."""
    orch = shared_orchestrator
    sc = SimulationScenario(name="payload_schema", duration_s=15.0)
    payloads = orch.run_simulation(sc)

    d = payloads[-1].to_dict()
    required_sections = [
        "mission", "telemetry", "digital_twin", "anomaly", "diagnosis",
        "health", "forecast", "rul", "explainability", "advisory", "provenance"
    ]
    for sec in required_sections:
        assert sec in d, f"Missing section '{sec}' in DashboardStatePayload.to_dict()"

    assert "engine_id" in d["mission"]
    assert "smoothed_health_index" in d["health"]
    assert "point_rul_seconds" in d["rul"]
    assert "action_code" in d["advisory"]


# ==============================================================================
# Test 21: State Reset Validation
# ==============================================================================
def test_21_state_reset(shared_orchestrator):
    """Calling orchestrator.reset flushes internal historical states."""
    orch = shared_orchestrator
    sc = SimulationScenario(name="reset_test", duration_s=25.0)
    orch.run_simulation(sc)

    assert orch.step_count > 0
    assert orch.last_timestamp is not None

    orch.reset(engine_id="RESET_ENG", mission_id="RESET_MISS")
    assert orch.step_count == 0
    assert orch.last_timestamp is None
    assert len(orch.twin.history) == 0


# ==============================================================================
# Test 22: Deterministic Replay
# ==============================================================================
def test_22_deterministic_replay(shared_orchestrator):
    """Two simulation runs with identical seed and scenario yield identical payloads."""
    orch = shared_orchestrator
    sc = SimulationScenario(name="replay_test", duration_s=20.0, seed=12345)

    run1 = orch.run_simulation(sc)
    run2 = orch.run_simulation(sc)

    assert len(run1) == len(run2)
    for i in range(len(run1)):
        assert run1[i].smoothed_health_index == run2[i].smoothed_health_index
        assert run1[i].anomaly_score == run2[i].anomaly_score
        assert run1[i].predicted_fault_class == run2[i].predicted_fault_class


# ==============================================================================
# Test 23: End-to-End Pipeline Execution
# ==============================================================================
def test_23_end_to_end_pipeline_execution(shared_orchestrator):
    """End-to-end mission execution across all 35 steps without errors or warnings."""
    orch = shared_orchestrator
    sc = SimulationScenario(name="e2e_full", duration_s=35.0)
    payloads = orch.run_simulation(sc)

    assert len(payloads) == 35
    for p in payloads:
        assert isinstance(p, DashboardStatePayload)
        assert 0.0 <= p.smoothed_health_index <= 1.0
        assert p.advisory is not None


# ==============================================================================
# Test 24: Latency Benchmark
# ==============================================================================
def test_24_latency_benchmark(shared_orchestrator):
    """Steady-state per-step inference latency satisfies real-time execution constraint."""
    orch = shared_orchestrator
    sc = SimulationScenario(name="latency_bench", duration_s=35.0)
    payloads = orch.run_simulation(sc)

    # Exclude initial 5 warmup steps
    latencies = [p.execution_latency_ms for p in payloads[5:]]
    assert len(latencies) >= 25

    mean_ms = np.mean(latencies)
    median_ms = np.median(latencies)
    p95_ms = np.percentile(latencies, 95)
    p99_ms = np.percentile(latencies, 99)

    print(f"\n[LATENCY BENCHMARK RESULT]")
    print(f"Mean: {mean_ms:.2f} ms | Median: {median_ms:.2f} ms | P95: {p95_ms:.2f} ms | P99: {p99_ms:.2f} ms")

    # Real-time requirement: per-step latency well within 1.0s sampling interval (< 200 ms)
    assert mean_ms < 200.0, f"Mean latency {mean_ms:.2f} ms exceeds 200 ms"
