"""
Test suite verifying telemetry rejection resilience, explainability non-null contracts,
and mission reporting integrity across out-of-order, duplicate, and non-finite observations.
"""

import math
import numpy as np
import pytest

from explainability.schema import EvidenceQuality, EvidenceStatus, ExplainabilityResult
from orchestrator.pipeline import SystemPipelineOrchestrator
from orchestrator.schema import DashboardStatePayload, OrchestratorConfig
from phase14.reporting import MissionReportGenerator
from phase14.replay import MissionReplayManager
from simulator.engine_simulator import EngineSimulator
from simulator.subsystems.mission import MissionProfile, PhaseSegment, FlightPhase


def _build_test_orchestrator() -> SystemPipelineOrchestrator:
    config = OrchestratorConfig(
        default_engine_id="TEST_UAV_01",
        default_mission_id="TEST_MIS_01",
        auto_bootstrap_on_init=True,
        deterministic_seed=42,
    )
    orch = SystemPipelineOrchestrator(config=config)
    orch.reset(engine_id="TEST_UAV_01", mission_id="TEST_MIS_01")
    return orch


def _generate_synthetic_telemetry(num_samples: int = 5, dt: float = 1.0):
    sim = EngineSimulator(seed=42)
    seg = PhaseSegment(
        phase=FlightPhase.CRUISE,
        duration_s=num_samples * dt,
        throttle_start_pct=0.75,
        throttle_end_pct=0.75,
        altitude_start_m=1000.0,
        altitude_end_m=1000.0,
    )
    profile = MissionProfile(mission_id="TEST_MIS_01", segments=[seg])
    records = list(sim.run(profile, dt=dt))
    return records[:num_samples]


def test_out_of_order_final_sample_explainability_resilience():
    """Verify an out-of-order final observation is rejected and never leaves explainability as None."""
    orch = _build_test_orchestrator()
    records = _generate_synthetic_telemetry(num_samples=5)

    payloads = []
    for rec in records:
        payloads.append(orch.step(rec))

    assert len(payloads) == 5
    assert payloads[-1].quality_status == "NOMINAL"
    assert payloads[-1].authoritative_explainability is not None

    # Inject an out-of-order final observation
    bad_dict = records[-1].to_dict()
    bad_dict["timestamp"] = 2.0  # Timestamp earlier than 4.0
    rejected_payload = orch.step(bad_dict)
    payloads.append(rejected_payload)

    # 1. Pipeline payload assertions
    assert rejected_payload.quality_status == "OUT_OF_ORDER_REJECTED"
    assert rejected_payload.authoritative_explainability is not None
    assert rejected_payload.authoritative_explainability.overall_quality == EvidenceQuality.INSUFFICIENT_DATA
    assert rejected_payload.authoritative_explainability.physics_evidence.status == EvidenceStatus.INSUFFICIENT_DATA
    assert "rejected" in rejected_payload.authoritative_explainability.summary_explanation.lower()

    # 2. Mission report generator resilience
    report = MissionReportGenerator.generate_report(payloads)
    assert report.engine_id == records[0].engine_id
    assert report.duration_s == 4.0  # Derived from valid timesteps (0.0 to 4.0)
    assert not math.isnan(report.final_health_index)
    assert report.provenance["total_samples"] == 6
    assert report.provenance["valid_samples"] == 5
    assert report.provenance["trailing_rejected_samples"] == 1
    assert "trailing sample(s) rejected" in report.provenance.get("notice", "")


def test_duplicate_final_sample_resilience():
    """Verify duplicate final observation is rejected without null explainability or reporting crash."""
    orch = _build_test_orchestrator()
    records = _generate_synthetic_telemetry(num_samples=4)

    payloads = [orch.step(rec) for rec in records]
    last_ts = payloads[-1].timestamp

    dup_dict = records[-1].to_dict()
    dup_dict["timestamp"] = last_ts  # Exact duplicate
    dup_payload = orch.step(dup_dict)
    payloads.append(dup_payload)

    assert dup_payload.quality_status == "DUPLICATE_TIMESTAMP_REJECTED"
    assert dup_payload.authoritative_explainability is not None
    assert dup_payload.authoritative_explainability.overall_quality == EvidenceQuality.INSUFFICIENT_DATA

    report = MissionReportGenerator.generate_report(payloads)
    assert report.provenance["trailing_rejected_samples"] == 1
    assert not math.isnan(report.final_health_index)


def test_missing_and_non_finite_final_timestamp_resilience():
    """Verify missing, NaN, and Inf final timestamps are safely rejected and reporting succeeds."""
    orch = _build_test_orchestrator()
    records = _generate_synthetic_telemetry(num_samples=3)
    payloads = [orch.step(rec) for rec in records]

    for bad_ts in (None, float("nan"), float("inf")):
        bad_dict = records[-1].to_dict()
        bad_dict["timestamp"] = bad_ts
        bad_payload = orch.step(bad_dict)

        assert bad_payload.quality_status == "INVALID_TIMESTAMP_REJECTED"
        assert bad_payload.authoritative_explainability is not None
        assert bad_payload.authoritative_explainability.overall_quality == EvidenceQuality.INSUFFICIENT_DATA

        test_stream = payloads + [bad_payload]
        report = MissionReportGenerator.generate_report(test_stream)
        assert not math.isnan(report.final_health_index)
        assert report.provenance["trailing_rejected_samples"] == 1


def test_malformed_final_timestamp_resilience():
    """Verify non-numeric string timestamp is rejected safely."""
    orch = _build_test_orchestrator()
    records = _generate_synthetic_telemetry(num_samples=3)
    payloads = [orch.step(rec) for rec in records]

    bad_dict = records[-1].to_dict()
    bad_dict["timestamp"] = "corrupt_timestamp"
    bad_payload = orch.step(bad_dict)

    assert bad_payload.quality_status == "INVALID_TIMESTAMP_REJECTED"
    assert bad_payload.authoritative_explainability is not None
    assert bad_payload.authoritative_explainability.overall_quality == EvidenceQuality.INSUFFICIENT_DATA

    report = MissionReportGenerator.generate_report(payloads + [bad_payload])
    assert not math.isnan(report.final_health_index)


def test_valid_sample_resuming_after_rejected_sample():
    """Verify valid telemetry arriving after an out-of-order sample continues causal execution."""
    orch = _build_test_orchestrator()
    records = _generate_synthetic_telemetry(num_samples=5)

    payloads = [
        orch.step(records[0]),
        orch.step(records[1]),
        orch.step(records[2]),
    ]

    # Inject out-of-order sample
    bad_dict = records[3].to_dict()
    bad_dict["timestamp"] = 1.0  # Earlier than timestamp 2.0
    bad_payload = orch.step(bad_dict)
    payloads.append(bad_payload)
    assert bad_payload.quality_status == "OUT_OF_ORDER_REJECTED"

    # Resume valid sequence
    valid_resume = orch.step(records[4])
    payloads.append(valid_resume)
    assert valid_resume.quality_status == "NOMINAL"
    assert valid_resume.authoritative_explainability is not None

    report = MissionReportGenerator.generate_report(payloads)
    assert report.provenance["total_samples"] == 5
    assert report.provenance["valid_samples"] == 4
    assert report.provenance["rejected_samples_count"] == 1
    assert report.provenance["trailing_rejected_samples"] == 0


def test_all_rejected_samples_report_generation():
    """Verify report generator handles an edge case sequence of exclusively rejected samples."""
    orch = _build_test_orchestrator()
    records = _generate_synthetic_telemetry(num_samples=3)

    payloads = []
    for rec in records:
        d = rec.to_dict()
        d["timestamp"] = float("nan")
        payloads.append(orch.step(d))

    report = MissionReportGenerator.generate_report(payloads)
    assert report.advisory_assessment == "INSUFFICIENT_DATA"
    assert report.evidence_quality == "INSUFFICIENT_DATA"
    assert math.isnan(report.final_health_index)
    assert report.final_rul_seconds is None


def test_schema_authoritative_explainability_fallback():
    """Verify that DashboardStatePayload guarantees authoritative_explainability is never None."""
    payload = DashboardStatePayload(
        engine_id="TEST_ENG",
        mission_id="TEST_MIS",
        timestamp=100.0,
        mission_phase="CRUISE",
        simulation_mode="SYNTHETIC_SIMULATION",
        quality_status="OUT_OF_ORDER_REJECTED",
        anomaly_status="INSUFFICIENT_DATA",
        anomaly_score=float("nan"),
        predicted_fault_class="none",
        diagnostic_confidence=0.0,
        raw_health_index=float("nan"),
        smoothed_health_index=float("nan"),
        health_state="INSUFFICIENT_DATA",
        degradation_trend="INDETERMINATE",
        rul_state="INSUFFICIENT_DATA",
        _explainability_result=None,  # Intentionally omitted
    )

    expl = payload.authoritative_explainability
    assert expl is not None
    assert isinstance(expl, ExplainabilityResult)
    assert expl.overall_quality == EvidenceQuality.INSUFFICIENT_DATA
    assert expl.physics_evidence.status == EvidenceStatus.INSUFFICIENT_DATA


def test_replay_session_with_trailing_rejected_sample():
    """Verify replay manager accepts payload sequence containing rejected samples."""
    orch = _build_test_orchestrator()
    records = _generate_synthetic_telemetry(num_samples=4)
    payloads = [orch.step(rec) for rec in records]

    # Add duplicate timestamp
    dup_dict = records[-1].to_dict()
    payloads.append(orch.step(dup_dict))

    mgr = MissionReplayManager()
    session = mgr.create_session_from_payloads(payloads=payloads, scenario_name="test_replay")
    assert session.total_steps == 5
    assert session.total_duration_s == 3.0  # 0.0 to 3.0
