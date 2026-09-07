"""
Tests for Phase 14 Mission Replay Module.
Verifies determinism, chronological ordering, engine/mission isolation, and error handling.
"""

import pytest
import numpy as np

from orchestrator.schema import (
    DashboardStatePayload,
    OrchestratorConfig,
    SimulationScenario,
    ScenarioFaultType,
)
from orchestrator.pipeline import SystemPipelineOrchestrator
from phase14.schema import MissionReplaySession
from phase14.replay import MissionReplayManager


def _create_mock_payload(
    engine_id: str = "ENG_TEST_01",
    mission_id: str = "MIS_TEST_01",
    timestamp: float = 10.0,
    hi: float = 0.95,
) -> DashboardStatePayload:
    return DashboardStatePayload(
        engine_id=engine_id,
        mission_id=mission_id,
        timestamp=timestamp,
        mission_phase="CRUISE",
        observed_telemetry={"rpm": 4500.0, "cht": 110.0, "egt": 680.0, "oil_temp": 85.0, "oil_pressure": 4.0, "vibration": 0.4, "fuel_flow": 18.0},
        expected_telemetry={"rpm": 4500.0, "cht": 110.0},
        residuals={"rpm": 0.0, "cht": 0.0},
        normalized_residuals={"rpm": 0.0, "cht": 0.0},
        smoothed_health_index=hi,
    )


def test_replay_session_initialization():
    """Verify MissionReplaySession basic initialization and boundary metrics."""
    p1 = _create_mock_payload(timestamp=0.0)
    p2 = _create_mock_payload(timestamp=5.0)
    p3 = _create_mock_payload(timestamp=10.0)
    session = MissionReplaySession(
        engine_id="ENG_TEST_01",
        mission_id="MIS_TEST_01",
        scenario_name="test_run",
        payloads=[p1, p2, p3],
        total_duration_s=10.0,
        total_steps=3,
        dt=5.0,
    )

    assert session.total_steps == 3
    assert session.total_duration_s == 10.0
    assert session.get_step(0).timestamp == 0.0
    assert session.get_step(2).timestamp == 10.0
    # Clamping behavior
    assert session.get_step(-10).timestamp == 0.0
    assert session.get_step(999).timestamp == 10.0


def test_replay_get_history():
    """Verify history extraction up to designated step for timeline visualization."""
    p1 = _create_mock_payload(timestamp=0.0, hi=1.0)
    p2 = _create_mock_payload(timestamp=1.0, hi=0.98)
    p3 = _create_mock_payload(timestamp=2.0, hi=0.95)
    session = MissionReplaySession(
        engine_id="ENG_TEST_01",
        mission_id="MIS_TEST_01",
        scenario_name="test_run",
        payloads=[p1, p2, p3],
        total_duration_s=2.0,
        total_steps=3,
    )

    hist_step1 = session.get_history(1)
    assert hist_step1["timestamps"] == [0.0, 1.0]
    assert hist_step1["health_index"]["hi"] == [1.0, 0.98]
    assert len(hist_step1["cht"]) == 2
    assert hist_step1["cht"][0] == 110.0

    hist_step2 = session.get_history(2)
    assert len(hist_step2["timestamps"]) == 3
    assert hist_step2["health_index"]["hi"] == [1.0, 0.98, 0.95]


def test_replay_chronological_ordering_enforcement():
    """Verify that scrambled or non-monotonic timestamps raise ValueError."""
    mgr = MissionReplayManager()
    p1 = _create_mock_payload(timestamp=5.0)
    p2 = _create_mock_payload(timestamp=3.0)  # Time reversal

    with pytest.raises(ValueError, match="Chronological ordering violation"):
        mgr.create_session_from_payloads([p1, p2])

    p3 = _create_mock_payload(timestamp=5.0)
    p4 = _create_mock_payload(timestamp=5.0)  # Duplicate timestamp
    with pytest.raises(ValueError, match="Chronological ordering violation"):
        mgr.create_session_from_payloads([p3, p4])


def test_replay_engine_mission_isolation():
    """Verify that mixing engines or missions in a single session raises ValueError."""
    mgr = MissionReplayManager()
    p1 = _create_mock_payload(engine_id="ENG_01", mission_id="MIS_01", timestamp=1.0)
    p2 = _create_mock_payload(engine_id="ENG_02", mission_id="MIS_01", timestamp=2.0)

    with pytest.raises(ValueError, match="Payload stream mixes state boundaries"):
        mgr.create_session_from_payloads([p1, p2])

    p3 = _create_mock_payload(engine_id="ENG_01", mission_id="MIS_01", timestamp=1.0)
    p4 = _create_mock_payload(engine_id="ENG_01", mission_id="MIS_02", timestamp=2.0)
    with pytest.raises(ValueError, match="Payload stream mixes state boundaries"):
        mgr.create_session_from_payloads([p3, p4])


def test_replay_empty_payload_rejection():
    """Verify that creating a session from an empty payload list raises ValueError."""
    mgr = MissionReplayManager()
    with pytest.raises(ValueError, match="empty payload list"):
        mgr.create_session_from_payloads([])


def test_replay_manager_lifecycle():
    """Verify session registration, retrieval, listing, and clearing."""
    mgr = MissionReplayManager()
    p1 = _create_mock_payload(engine_id="UAV_ENG_A", mission_id="MIS_01", timestamp=1.0)
    p2 = _create_mock_payload(engine_id="UAV_ENG_A", mission_id="MIS_01", timestamp=2.0)

    s1 = mgr.create_session_from_payloads([p1, p2], scenario_name="patrol_1")
    assert mgr.list_sessions() == ["UAV_ENG_A::MIS_01"]

    retrieved = mgr.get_session("UAV_ENG_A", "MIS_01")
    assert retrieved is not None
    assert retrieved.engine_id == "UAV_ENG_A"
    assert retrieved.total_steps == 2

    # Query non-existent session
    assert mgr.get_session("UNKNOWN", "MIS_01") is None

    mgr.clear()
    assert mgr.list_sessions() == []


def test_replay_deterministic_execution():
    """Verify that recording a mission with matching seed produces deterministic results."""
    cfg = OrchestratorConfig(auto_bootstrap_on_init=True, deterministic_seed=42)
    orch = SystemPipelineOrchestrator(config=cfg)
    sc = SimulationScenario(
        name="det_test",
        duration_s=8.0,
        fault_type=ScenarioFaultType.COOLING_DEGRADATION,
        fault_start_s=4.0,
        fault_severity=0.6,
        seed=101,
        engine_id="ENG_DET",
        mission_id="MIS_DET",
    )

    mgr = MissionReplayManager()
    session_a = mgr.record_mission(orch, sc)
    session_b = mgr.record_mission(orch, sc)

    assert session_a.total_steps == session_b.total_steps
    for idx in range(session_a.total_steps):
        pa = session_a.get_step(idx)
        pb = session_b.get_step(idx)
        assert pa.timestamp == pb.timestamp
        assert pa.anomaly_status == pb.anomaly_status
        assert pa.predicted_fault_class == pb.predicted_fault_class
        assert pa.smoothed_health_index == pb.smoothed_health_index
        assert pa.observed_telemetry["cht"] == pb.observed_telemetry["cht"]
