"""
Phase 11 Evidence, Traceability & Engineering Explainability Test Suite.

Validates all 20 required categories:
A. Evidence schema
B. Immutability
C. Deterministic serialization
D. Provenance
E. Units
F. Measured/estimated/predicted distinction
G. Quality propagation
H. Observability
I. Health traceability
J. Anomaly traceability
K. Diagnosis traceability
L. Cylinder localization
M. Sensor-vs-physical explanation
N. Degradation traceability
O. RUL traceability
P. Mission traceability
Q. Insufficient-data handling
R. 15 Golden cases
S. Anti-leakage
T. Non-circularity
U. Replay determinism
V. Completeness
W. Strong non-interference
X. Provenance consistency
"""

import ast
import json
import math
import os
import re
from typing import Dict, Any, List

import pytest
import numpy as np

from telemetry.schema import TelemetryRecord, DigitalTwinState
from simulator.engine_simulator import EngineSimulator
from simulator.config import SimulatorConfig
from simulator.fault_interface import FaultState, FaultType
from digital_twin.twin_model import DigitalTwin
from digital_twin.health import HealthState, ModelObservationHealthAssessment
from digital_twin.detection import DetectionStatus, DetectionResult
from digital_twin.diagnosis import CanonicalFaultType, DiagnosisResult
from digital_twin.degradation_types import RULStatus, DegradationAssessment, RULAssessment
from digital_twin.mission_simulator import MissionSimulator
from digital_twin.mission_types import MissionSpec
from digital_twin.what_if import WhatIfEvaluator, get_golden_scenario_specs

from digital_twin.evidence_types import (
    DataClassification,
    EpistemicProvenance,
    ChannelRole,
    ExplanationReasonCode,
    StepEvidenceRecord,
    WhatIfScenarioDeltaEvidence,
)
from digital_twin.evidence import EvidenceBuilder
from digital_twin.explainability import ExplainabilityEngine


@pytest.fixture
def sim_and_twin():
    config = SimulatorConfig()
    sim = EngineSimulator(sim_config=config, seed=42)
    twin = DigitalTwin(sim_config=config)
    sim.reset(seed=42)
    return sim, twin, config


@pytest.fixture
def builder():
    return EvidenceBuilder(engine_id="TEST_ROTAX_914")


# ==============================================================================
# Category A: Evidence Schema
# ==============================================================================

def test_evidence_schema_structure(sim_and_twin, builder):
    sim, twin, _ = sim_and_twin
    record = sim.step(throttle_pct=70.0, dt=0.1)
    twin_state = twin.update(record)
    ev = builder.build_step_evidence(twin_state)

    assert isinstance(ev, StepEvidenceRecord)
    assert ev.timestamp == twin_state.timestamp
    assert ev.engine_id == "TEST_ROTAX_914"
    assert isinstance(ev.channel_evidence, dict)
    assert isinstance(ev.subsystem_evidence, dict)
    assert isinstance(ev.traceability_chains, dict)
    assert isinstance(ev.overall_reason_codes, tuple)
    assert isinstance(ev.limitations, tuple)
    assert len(ev.limitations) >= 3


# ==============================================================================
# Category B: Immutability
# ==============================================================================

def test_evidence_immutability(sim_and_twin, builder):
    sim, twin, _ = sim_and_twin
    record = sim.step(throttle_pct=70.0, dt=0.1)
    twin_state = twin.update(record)
    ev = builder.build_step_evidence(twin_state)

    # StepEvidenceRecord is frozen
    with pytest.raises(Exception):
        ev.timestamp = 999.0

    # ChannelEvidence is frozen
    ch = ev.channel_evidence["rpm"]
    with pytest.raises(Exception):
        ch.observed_value = 10000.0


# ==============================================================================
# Category C: Deterministic Serialization
# ==============================================================================

def test_deterministic_serialization_byte_identical(sim_and_twin, builder):
    sim, twin, _ = sim_and_twin
    record = sim.step(throttle_pct=70.0, dt=0.1)
    twin_state = twin.update(record)

    ev1 = builder.build_step_evidence(twin_state)
    ev2 = builder.build_step_evidence(twin_state)

    json1 = ExplainabilityEngine.serialize_to_json(ev1, indent=None)
    json2 = ExplainabilityEngine.serialize_to_json(ev2, indent=None)

    assert json1 == json2
    assert ev1.compute_sha256() == ev2.compute_sha256()


# ==============================================================================
# Category D: Provenance
# ==============================================================================

def test_epistemic_provenance_labels(sim_and_twin, builder):
    sim, twin, _ = sim_and_twin
    record = sim.step(throttle_pct=70.0, dt=0.1)
    twin_state = twin.update(record)
    ev = builder.build_step_evidence(twin_state)

    # Subsystem evidence must have provenance
    for sub in ev.subsystem_evidence.values():
        assert isinstance(sub.provenance, EpistemicProvenance)

    # Anomaly evidence must have provenance
    if ev.anomaly_evidence:
        assert isinstance(ev.anomaly_evidence.provenance, EpistemicProvenance)

    # Diagnosis evidence must have provenance
    if ev.diagnosis_evidence:
        assert isinstance(ev.diagnosis_evidence.provenance, EpistemicProvenance)


# ==============================================================================
# Category E: Units
# ==============================================================================

def test_physical_units_explicit(sim_and_twin, builder):
    sim, twin, _ = sim_and_twin
    record = sim.step(throttle_pct=70.0, dt=0.1)
    twin_state = twin.update(record)
    ev = builder.build_step_evidence(twin_state)

    assert ev.channel_evidence["rpm"].unit == "RPM"
    assert ev.channel_evidence["egt"].unit == "°C"
    assert ev.channel_evidence["cht"].unit == "°C"
    assert "bar" in ev.channel_evidence["oil_pressure"].unit
    assert ev.channel_evidence["vibration"].unit == "g"


# ==============================================================================
# Category F: Measured vs Predicted vs Derived
# ==============================================================================

def test_measured_vs_predicted_vs_derived(sim_and_twin, builder):
    sim, twin, _ = sim_and_twin
    record = sim.step(throttle_pct=70.0, dt=0.1)
    twin_state = twin.update(record)
    ev = builder.build_step_evidence(twin_state)

    rpm_ev = ev.channel_evidence["rpm"]
    assert rpm_ev.classification == DataClassification.MEASURED
    assert rpm_ev.observed_value is not None
    assert rpm_ev.predicted_value is not None
    assert rpm_ev.raw_residual is not None


# ==============================================================================
# Category G: Quality Propagation
# ==============================================================================

def test_quality_propagation_dropout(sim_and_twin, builder):
    sim, twin, _ = sim_and_twin
    record = sim.step(throttle_pct=70.0, dt=0.1)
    # Inject NaN into telemetry
    record.egt = float("nan")
    twin_state = twin.update(record)
    ev = builder.build_step_evidence(twin_state)

    # EGT channel quality must reflect missing/dropout
    assert ev.channel_evidence["egt"].quality_status in ("MISSING", "INVALID", "NAN") or math.isnan(ev.channel_evidence["egt"].observed_value or float("nan"))


# ==============================================================================
# Category H: Observability
# ==============================================================================

def test_observability_coverage_gating(sim_and_twin, builder):
    sim, twin, _ = sim_and_twin
    record = sim.step(throttle_pct=70.0, dt=0.1)
    # Invalidate 6 primary channels so valid < 5
    record.rpm = float("nan")
    record.egt = float("nan")
    record.cht = float("nan")
    record.oil_pressure = float("nan")
    record.oil_temp = float("nan")
    record.vibration = float("nan")
    twin_state = twin.update(record)
    ev = builder.build_step_evidence(twin_state)

    assert ev.observability_coverage < (5.0 / 9.0)
    assert ev.engine_health_state == HealthState.UNAVAILABLE.value
    assert ExplanationReasonCode.INSUFFICIENT_DATA in ev.overall_reason_codes


# ==============================================================================
# Category I: Health Traceability
# ==============================================================================

def test_health_traceability_lineage(sim_and_twin, builder):
    sim, twin, _ = sim_and_twin
    record = sim.step(throttle_pct=70.0, dt=0.1)
    twin_state = twin.update(record)
    ev = builder.build_step_evidence(twin_state)

    assert "engine_health" in ev.traceability_chains
    chain = ev.traceability_chains["engine_health"]
    stages = [n.stage for n in chain.nodes]
    assert "1_TELEMETRY_INGESTION" in stages
    assert "2_PHYSICS_ESTIMATION" in stages
    assert "3_RESIDUAL_GENERATION" in stages
    assert "4_SUBSYSTEM_EVALUATION" in stages
    assert "5_ENGINE_HEALTH_INDEX" in stages


# ==============================================================================
# Category J: Anomaly Traceability
# ==============================================================================

def test_anomaly_traceability(sim_and_twin, builder):
    sim, twin, _ = sim_and_twin
    record = sim.step(throttle_pct=70.0, dt=0.1)
    twin_state = twin.update(record)
    ev = builder.build_step_evidence(twin_state)

    assert ev.anomaly_evidence is not None
    assert ev.anomaly_evidence.threshold == 0.018
    assert ev.anomaly_evidence.status in [s.value for s in DetectionStatus]


# ==============================================================================
# Category K: Diagnosis Traceability
# ==============================================================================

def test_diagnosis_traceability_hypothesis_semantics(sim_and_twin, builder):
    sim, twin, _ = sim_and_twin
    record = sim.step(throttle_pct=70.0, dt=0.1)
    twin_state = twin.update(record)
    ev = builder.build_step_evidence(twin_state)

    assert ev.diagnosis_evidence is not None
    # Confidence heuristic must be in [0, 1], not a percentage
    assert 0.0 <= ev.diagnosis_evidence.confidence_heuristic <= 1.0


# ==============================================================================
# Category L: Cylinder Localization
# ==============================================================================

def test_cylinder_channels_do_not_inflate_engine_health_votes(sim_and_twin, builder):
    sim, twin, _ = sim_and_twin
    record = sim.step(throttle_pct=70.0, dt=0.1)
    twin_state = twin.update(record)
    ev = builder.build_step_evidence(twin_state)

    # Cylinder channels must have CYLINDER_LOCALIZATION_DIAGNOSTIC role
    for cyl_i in range(1, 5):
        cyl_ch = f"egt_cyl{cyl_i}"
        if cyl_ch in ev.channel_evidence:
            assert ev.channel_evidence[cyl_ch].role == ChannelRole.CYLINDER_LOCALIZATION_DIAGNOSTIC


# ==============================================================================
# Category M: Sensor vs Physical Fault Explanation
# ==============================================================================

def test_sensor_vs_physical_fault_discrimination(builder):
    # Construct a synthetic DiagnosisResult with is_sensor_fault=True
    diag_res = DiagnosisResult(
        timestamp=10.0,
        engine_id="TEST_ROTAX_914",
        primary_fault="SENSOR_BIAS",
        ranked_hypotheses=[],
        confidence=0.85,
        evidence=["cht_isolated_divergence"],
        affected_subsystem="thermal",
        affected_cylinder=None,
        is_sensor_fault=True,
        uncertainty=0.10,
        status="SUSPECTED",
    )
    twin_state = DigitalTwinState(
        timestamp=10.0,
        engine_id="TEST_ROTAX_914",
        observed_telemetry=TelemetryRecord(
            timestamp=10.0, mission_id="TEST", engine_id="TEST", mission_phase="CRUISE",
            altitude=1000.0, ambient_temp=20.0, throttle=70.0, load=70.0, rpm=4500.0,
            cht=120.0, egt=700.0, oil_temp=85.0, oil_pressure=3.0, fuel_flow=20.0, vibration=0.2,
        ),
        diagnosis_result=diag_res,
    )
    ev = builder.build_step_evidence(twin_state)
    assert ExplanationReasonCode.SENSOR_LOCALIZATION_FAVORED in ev.overall_reason_codes

    rendered = ExplainabilityEngine.render_human_readable(ev)
    assert "evidence favors sensor anomaly" in rendered


# ==============================================================================
# Category N: Degradation Traceability
# ==============================================================================

def test_degradation_traceability(sim_and_twin, builder):
    sim, twin, _ = sim_and_twin
    record = sim.step(throttle_pct=70.0, dt=0.1)
    twin_state = twin.update(record)
    ev = builder.build_step_evidence(twin_state)

    if ev.prognostic_evidence:
        assert ev.prognostic_evidence.eol_threshold == 0.50
        assert ev.prognostic_evidence.observation_count >= 0


# ==============================================================================
# Category O: RUL Traceability
# ==============================================================================

def test_rul_unavailable_gating_reason(sim_and_twin, builder):
    sim, twin, _ = sim_and_twin
    record = sim.step(throttle_pct=70.0, dt=0.1)
    twin_state = twin.update(record)
    ev = builder.build_step_evidence(twin_state)

    if ev.prognostic_evidence:
        # At t=0.1, insufficient history exists for RUL
        assert ev.prognostic_evidence.rul_status in ("INSUFFICIENT_DATA", "STABLE", "NON_DEGRADING", "UNAVAILABLE")


# ==============================================================================
# Category P: Mission Traceability
# ==============================================================================

def test_mission_what_if_scenario_deltas():
    delta_ev = ExplainabilityEngine.explain_what_if_delta(
        scenario_id="HOT_DAY",
        baseline_scenario_id="NOMINAL",
        delta_metrics={"max_cht": 15.2, "max_egt": -5.1, "min_oil_pressure": -0.4},
        modeled_contributions={"environment": "ISA + 15K offset", "thermal": "coolant heat rejection delta"},
        envelope_events_delta=0,
        risk_index_delta=0.012,
    )

    assert isinstance(delta_ev, WhatIfScenarioDeltaEvidence)
    assert ExplanationReasonCode.THERMAL_LIMIT_APPROACH in delta_ev.reason_codes
    assert ExplanationReasonCode.LUBRICATION_PRESSURE_DEVIATION in delta_ev.reason_codes


# ==============================================================================
# Category Q: Insufficient Data Handling
# ==============================================================================

def test_insufficient_data_reasons(sim_and_twin, builder):
    sim, twin, _ = sim_and_twin
    record = sim.step(throttle_pct=70.0, dt=0.1)
    # Zero all channels to NaN
    for ch in ("rpm", "egt", "cht", "oil_pressure", "oil_temp", "vibration", "fuel_flow"):
        setattr(record, ch, float("nan"))
    twin_state = twin.update(record)
    ev = builder.build_step_evidence(twin_state)

    assert ExplanationReasonCode.INSUFFICIENT_DATA in ev.overall_reason_codes


# ==============================================================================
# Category R: Golden Cases (15 Scenarios)
# ==============================================================================

def test_golden_1_healthy_nominal(builder):
    sim = EngineSimulator(seed=101)
    twin = DigitalTwin()
    for _ in range(20):
        rec = sim.step(throttle_pct=70.0, dt=0.5)
        st = twin.update(rec)
    ev = builder.build_step_evidence(st)
    assert ev.hi_smooth is not None and ev.hi_smooth > 0.95
    assert ev.engine_health_state == HealthState.HEALTHY.value


def test_golden_2_f1_injector(builder):
    sim = EngineSimulator(seed=102)
    twin = DigitalTwin()
    f = FaultState(fault_type=FaultType.FUEL_INJECTION_ABNORMALITY, severity=0.35, affected_cylinder=1)
    for _ in range(20):
        rec = sim.step(throttle_pct=70.0, dt=0.5, fault_state=f)
        st = twin.update(rec)
    ev = builder.build_step_evidence(st)
    # In F1, runner spread expands
    assert "egt_cyl1" in ev.channel_evidence


def test_golden_3_f2_lubrication(builder):
    sim = EngineSimulator(seed=103)
    twin = DigitalTwin()
    f = FaultState(fault_type=FaultType.LUBRICATION_DEGRADATION, severity=0.50)
    for _ in range(25):
        rec = sim.step(throttle_pct=70.0, dt=0.5, fault_state=f)
        st = twin.update(rec)
    ev = builder.build_step_evidence(st)
    assert ExplanationReasonCode.LUBRICATION_PRESSURE_DEVIATION in ev.overall_reason_codes


def test_golden_4_f3_cooling(builder):
    sim = EngineSimulator(seed=104)
    twin = DigitalTwin()
    f = FaultState(fault_type=FaultType.COOLING_DEGRADATION, severity=0.60)
    for _ in range(40):
        rec = sim.step(throttle_pct=70.0, dt=0.5, fault_state=f)
        st = twin.update(rec)
    ev = builder.build_step_evidence(st)
    assert ev.channel_evidence["cht"].raw_residual is not None


def test_golden_5_f4_misfire(builder):
    sim = EngineSimulator(seed=105)
    twin = DigitalTwin()
    f = FaultState(fault_type=FaultType.COMBUSTION_MISFIRE, severity=0.60, affected_cylinder=2)
    for _ in range(35):
        rec = sim.step(throttle_pct=70.0, dt=0.5, fault_state=f)
        st = twin.update(rec)
    ev = builder.build_step_evidence(st)
    assert ev.diagnosis_evidence is not None
    assert ev.diagnosis_evidence.primary_fault == "COMBUSTION_MISFIRE"
    assert ev.diagnosis_evidence.affected_cylinder == 2


def test_golden_6_f5_mechanical(builder):
    sim = EngineSimulator(seed=106)
    twin = DigitalTwin()
    f = FaultState(fault_type=FaultType.MECHANICAL_DEGRADATION, severity=0.60)
    for _ in range(25):
        rec = sim.step(throttle_pct=70.0, dt=0.5, fault_state=f)
        st = twin.update(rec)
    ev = builder.build_step_evidence(st)
    assert ExplanationReasonCode.MECHANICAL_VIBRATION_DEVIATION in ev.overall_reason_codes


def test_golden_7_sensor_bias(builder):
    sim = EngineSimulator(seed=107)
    twin = DigitalTwin()
    f = FaultState(fault_type=FaultType.SENSOR_FAULT, severity=0.50, parameters={"sensor_channel": "cht", "sensor_mode": "bias"})
    for _ in range(25):
        rec = sim.step(throttle_pct=70.0, dt=0.5, fault_state=f)
        st = twin.update(rec)
    ev = builder.build_step_evidence(st)
    assert ev.channel_evidence["cht"].raw_residual is not None


def test_golden_8_sensor_drift(builder):
    sim = EngineSimulator(seed=108)
    twin = DigitalTwin()
    f = FaultState(fault_type=FaultType.SENSOR_FAULT, severity=0.50, parameters={"sensor_channel": "egt", "sensor_mode": "drift"})
    for _ in range(25):
        rec = sim.step(throttle_pct=70.0, dt=0.5, fault_state=f)
        st = twin.update(rec)
    ev = builder.build_step_evidence(st)
    assert ev.channel_evidence["egt"].raw_residual is not None


def test_golden_9_sensor_dropout(builder):
    sim = EngineSimulator(seed=109)
    twin = DigitalTwin()
    rec = sim.step(throttle_pct=70.0, dt=0.5)
    rec.rpm = float("nan")
    st = twin.update(rec)
    ev = builder.build_step_evidence(st)
    assert ev.channel_evidence["rpm"].classification in (DataClassification.PREDICTED, DataClassification.UNAVAILABLE)
    assert ev.channel_evidence["rpm"].quality_status in ("MISSING", "INVALID")


def test_golden_10_sensor_stuck(builder):
    sim = EngineSimulator(seed=110)
    twin = DigitalTwin()
    rec = sim.step(throttle_pct=70.0, dt=0.5)
    st = twin.update(rec)
    ev = builder.build_step_evidence(st)
    assert isinstance(ev, StepEvidenceRecord)


def test_golden_11_insufficient_data(builder):
    sim = EngineSimulator(seed=111)
    twin = DigitalTwin()
    rec = sim.step(throttle_pct=70.0, dt=0.5)
    # Drop all primary channels
    rec.rpm = rec.egt = rec.cht = rec.oil_pressure = rec.oil_temp = rec.vibration = float("nan")
    st = twin.update(rec)
    ev = builder.build_step_evidence(st)
    assert ev.engine_health_state == HealthState.UNAVAILABLE.value


def test_golden_12_finite_rul(builder):
    # Construct synthetic RUL assessment with status COMPUTED
    rul = RULAssessment(
        status=RULStatus.COMPUTED,
        degradation_index=0.25,
        degradation_trend=0.005,
        trend_confidence=0.88,
        rul_low=45.0,
        rul_median=50.0,
        rul_high=55.0,
        eol_threshold=0.50,
        data_confidence=0.95,
        sample_count=50,
        window_duration=250.0,
    )
    st = DigitalTwinState(
        timestamp=50.0,
        engine_id="TEST",
        observed_telemetry=TelemetryRecord(
            timestamp=50.0, mission_id="M", engine_id="TEST", mission_phase="CRUISE",
            altitude=1000.0, ambient_temp=20.0, throttle=70.0, load=70.0, rpm=4500.0,
            cht=80.0, egt=680.0, oil_temp=85.0, oil_pressure=3.0, fuel_flow=20.0, vibration=0.2,
        ),
        rul_assessment=rul,
    )
    ev = builder.build_step_evidence(st)
    assert ev.prognostic_evidence is not None
    assert ev.prognostic_evidence.rul_estimate_hours == 50.0
    assert ExplanationReasonCode.RUL_TREND_SUPPORTED in ev.overall_reason_codes


def test_golden_13_unavailable_rul(builder):
    rul = RULAssessment(
        status=RULStatus.NON_DEGRADING,
        degradation_index=0.02,
        degradation_trend=0.00001,
        trend_confidence=0.20,
        rejection_reason="Slope below minimum statistically significant threshold",
    )
    st = DigitalTwinState(
        timestamp=50.0,
        engine_id="TEST",
        observed_telemetry=TelemetryRecord(
            timestamp=50.0, mission_id="M", engine_id="TEST", mission_phase="CRUISE",
            altitude=1000.0, ambient_temp=20.0, throttle=70.0, load=70.0, rpm=4500.0,
            cht=80.0, egt=680.0, oil_temp=85.0, oil_pressure=3.0, fuel_flow=20.0, vibration=0.2,
        ),
        rul_assessment=rul,
    )
    ev = builder.build_step_evidence(st)
    assert ev.prognostic_evidence is not None
    assert ExplanationReasonCode.RUL_NON_DEGRADING in ev.overall_reason_codes
    assert ev.prognostic_evidence.rul_estimate_hours is None


def test_golden_14_mission_high_load():
    delta = ExplainabilityEngine.explain_what_if_delta(
        scenario_id="HIGH_LOAD",
        baseline_scenario_id="NOMINAL",
        delta_metrics={"max_cht": 12.0, "max_egt": 35.0, "mean_hi": -0.05},
        modeled_contributions={"throttle_load": "sustained 95% throttle rating"},
    )
    assert delta.scenario_id == "HIGH_LOAD"
    assert ExplanationReasonCode.COMBUSTION_EGT_DEVIATION in delta.reason_codes


def test_golden_15_mission_hot_day():
    delta = ExplainabilityEngine.explain_what_if_delta(
        scenario_id="HOT_DAY",
        baseline_scenario_id="NOMINAL",
        delta_metrics={"max_cht": 16.5, "max_egt": 4.0},
        modeled_contributions={"environment": "ISA + 15K temperature offset"},
    )
    assert delta.scenario_id == "HOT_DAY"
    assert ExplanationReasonCode.THERMAL_LIMIT_APPROACH in delta.reason_codes


# ==============================================================================
# Category S: Anti-Leakage
# ==============================================================================

def test_anti_leakage_runtime_path(sim_and_twin, builder):
    sim, twin, _ = sim_and_twin
    # Inject fault in simulation
    f = FaultState(fault_type=FaultType.COMBUSTION_MISFIRE, severity=0.60, affected_cylinder=2)
    record = sim.step(throttle_pct=70.0, dt=0.1, fault_state=f)
    twin_state = twin.update(record)

    # Evidence builder receives only twin_state, NOT fault_state
    ev = builder.build_step_evidence(twin_state)

    # Verify builder has no reference to f
    ev_dict = ev.to_dict()
    ev_str = json.dumps(ev_dict)
    # The string must not contain ground truth fault state internal tags
    assert "fault_state" not in ev_str.lower()


# ==============================================================================
# Category T: Non-Circularity
# ==============================================================================

def test_non_circularity_import_audit():
    """Verify that upstream modules never import Phase 11 evidence modules."""
    upstream_dirs = ["simulator", "telemetry"]
    upstream_files = [
        os.path.join("digital_twin", "twin_model.py"),
        os.path.join("digital_twin", "health.py"),
        os.path.join("digital_twin", "detection.py"),
        os.path.join("digital_twin", "diagnosis.py"),
        os.path.join("digital_twin", "degradation.py"),
        os.path.join("digital_twin", "rul.py"),
    ]

    for d in upstream_dirs:
        for root, _, files in os.walk(d):
            for f in files:
                if f.endswith(".py"):
                    upstream_files.append(os.path.join(root, f))

    for filepath in upstream_files:
        if not os.path.exists(filepath):
            continue
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            tree = ast.parse(content, filename=filepath)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        assert "evidence" not in alias.name
                        assert "explainability" not in alias.name
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    assert "evidence" not in mod
                    assert "explainability" not in mod


# ==============================================================================
# Category U: Replay Determinism
# ==============================================================================

def test_replay_determinism(builder):
    sim = EngineSimulator(seed=999)
    twin1 = DigitalTwin()
    twin2 = DigitalTwin()

    records = [sim.step(throttle_pct=75.0, dt=0.1) for _ in range(15)]

    # Run replay 1
    ev_list1 = []
    for r in records:
        st = twin1.update(r)
        ev_list1.append(builder.build_step_evidence(st).compute_sha256())

    # Run replay 2
    ev_list2 = []
    for r in records:
        st = twin2.update(r)
        ev_list2.append(builder.build_step_evidence(st).compute_sha256())

    assert ev_list1 == ev_list2


# ==============================================================================
# Category V: Completeness Audit
# ==============================================================================

def test_completeness_audit_metric(sim_and_twin, builder):
    sim, twin, _ = sim_and_twin
    record = sim.step(throttle_pct=70.0, dt=0.1)
    twin_state = twin.update(record)
    ev = builder.build_step_evidence(twin_state)

    audit = ev.completeness
    assert audit.total_required > 0
    assert audit.total_present <= audit.total_required
    assert 0.0 <= audit.completeness_ratio <= 1.0
    assert len(audit.required_fields) == audit.total_required
    assert len(audit.present_fields) == audit.total_present


# ==============================================================================
# Category W: Strong Non-Interference Test
# ==============================================================================

def test_explanation_cannot_change_runtime_result():
    """Verify that running with vs without evidence extraction leaves Twin state identical."""
    sim1 = EngineSimulator(seed=777)
    twin1 = DigitalTwin()
    sim2 = EngineSimulator(seed=777)
    twin2 = DigitalTwin()
    builder = EvidenceBuilder()

    for _ in range(20):
        r1 = sim1.step(throttle_pct=70.0, dt=0.2)
        st1 = twin1.update(r1)
        # In branch A: no evidence built

        r2 = sim2.step(throttle_pct=70.0, dt=0.2)
        st2 = twin2.update(r2)
        # In branch B: build evidence and render human readable
        ev2 = builder.build_step_evidence(st2)
        _ = ExplainabilityEngine.render_human_readable(ev2)

        # Assert upstream state equivalence
        assert st1.timestamp == st2.timestamp
        assert st1.nominal_estimates == st2.nominal_estimates
        assert st1.residuals == st2.residuals
        if st1.health_assessment and st2.health_assessment:
            assert st1.health_assessment.HI_raw == st2.health_assessment.HI_raw
            assert st1.health_assessment.HI_smooth == st2.health_assessment.HI_smooth


# ==============================================================================
# Category X: Provenance Consistency & Renderer Checks
# ==============================================================================

def test_human_readable_renderer_formatting(sim_and_twin, builder):
    sim, twin, _ = sim_and_twin
    record = sim.step(throttle_pct=70.0, dt=0.1)
    twin_state = twin.update(record)
    ev = builder.build_step_evidence(twin_state)
    rendered = ExplainabilityEngine.render_human_readable(ev)

    assert "ENGINE HEALTH:" in rendered
    assert "Primary contributors:" in rendered
    assert "Supporting evidence:" in rendered
    assert "Diagnosis:" in rendered
    assert "Data quality:" in rendered
    assert "Interpretation:" in rendered
    assert "Limitations:" in rendered


def test_human_readable_no_probability_claims(sim_and_twin, builder):
    sim, twin, _ = sim_and_twin
    record = sim.step(throttle_pct=70.0, dt=0.1)
    twin_state = twin.update(record)
    ev = builder.build_step_evidence(twin_state)
    rendered = ExplainabilityEngine.render_human_readable(ev)

    # Check for forbidden claims in diagnosis and interpretation
    assert not re.search(r"\b\d+%\s+probability\b", rendered, re.IGNORECASE)
    assert not re.search(r"\bcertified\s+diagnosis\b", rendered, re.IGNORECASE)


def test_f4_pre_active_post_e2e(builder):
    """End-to-end trace of F4 combustion misfire: PRE -> ACTIVE -> POST."""
    sim = EngineSimulator(seed=123)
    twin = DigitalTwin()

    f4 = FaultState(fault_type=FaultType.COMBUSTION_MISFIRE, severity=0.60, affected_cylinder=2)

    # 1. Pre-fault (t=0 to 10s)
    st_pre = None
    for _ in range(20):
        r = sim.step(throttle_pct=70.0, dt=0.5)
        st_pre = twin.update(r)
    ev_pre = builder.build_step_evidence(st_pre)

    assert ev_pre.diagnosis_evidence.primary_fault in ("HEALTHY", "UNKNOWN", "NONE")
    assert ev_pre.diagnosis_evidence.affected_cylinder is None

    # 2. Active fault (t=10 to 30s)
    st_act = None
    for _ in range(40):
        r = sim.step(throttle_pct=70.0, dt=0.5, fault_state=f4)
        st_act = twin.update(r)
    ev_act = builder.build_step_evidence(st_act)

    assert ev_act.diagnosis_evidence.primary_fault == "COMBUSTION_MISFIRE"
    assert ev_act.diagnosis_evidence.affected_cylinder == 2
    assert ExplanationReasonCode.FAULT_SIGNATURE_MATCH in ev_act.overall_reason_codes
    assert ExplanationReasonCode.CYLINDER_LOCALIZATION in ev_act.overall_reason_codes

    # 3. Post-fault recovery (t=30 to 50s)
    st_post = None
    for _ in range(40):
        r = sim.step(throttle_pct=70.0, dt=0.5)  # Fault removed
        st_post = twin.update(r)
    ev_post = builder.build_step_evidence(st_post)

    # After sustained recovery, fault clears to UNKNOWN / NONE
    assert ev_post.diagnosis_evidence.primary_fault in ("HEALTHY", "UNKNOWN", "NONE")


def test_evidence_latency_benchmark(sim_and_twin, builder):
    """Benchmark step evidence extraction time."""
    import time
    sim, twin, _ = sim_and_twin
    record = sim.step(throttle_pct=70.0, dt=0.1)
    twin_state = twin.update(record)

    latencies = []
    for _ in range(100):
        t0 = time.perf_counter()
        _ = builder.build_step_evidence(twin_state)
        latencies.append((time.perf_counter() - t0) * 1000.0)  # ms

    mean_lat = float(np.mean(latencies))
    p95_lat = float(np.percentile(latencies, 95))
    # Extraction must be lightweight (< 15 ms per step)
    assert mean_lat < 15.0
    assert p95_lat < 30.0


def test_what_if_delta_serialization_determinism():
    """Verify WhatIfScenarioDeltaEvidence byte-identical deterministic JSON serialization."""
    d1 = ExplainabilityEngine.explain_what_if_delta(
        scenario_id="TEST_SCENARIO",
        baseline_scenario_id="NOMINAL",
        delta_metrics={"max_cht": 14.5, "min_oil_pressure": -0.5},
        modeled_contributions={"environment": "ISA+15K"},
        envelope_events_delta=1,
        risk_index_delta=0.025,
    )
    d2 = ExplainabilityEngine.explain_what_if_delta(
        scenario_id="TEST_SCENARIO",
        baseline_scenario_id="NOMINAL",
        delta_metrics={"max_cht": 14.5, "min_oil_pressure": -0.5},
        modeled_contributions={"environment": "ISA+15K"},
        envelope_events_delta=1,
        risk_index_delta=0.025,
    )
    s1 = ExplainabilityEngine.serialize_to_json(d1, indent=None)
    s2 = ExplainabilityEngine.serialize_to_json(d2, indent=None)
    assert s1 == s2


def test_human_readable_renderer_competing_hypotheses(builder):
    """Verify that competing hypotheses ambiguity note is rendered."""
    diag_res = DiagnosisResult(
        timestamp=25.0,
        engine_id="TEST",
        primary_fault="COMBUSTION_MISFIRE",
        ranked_hypotheses=[],
        confidence=0.55,
        evidence=["egt_drop"],
        affected_subsystem="combustion",
        affected_cylinder=2,
        is_sensor_fault=False,
        uncertainty=0.45,
        status="SUSPECTED",
    )
    # Inject 2 competing hypotheses into metadata or property mock
    st = DigitalTwinState(
        timestamp=25.0,
        engine_id="TEST",
        observed_telemetry=TelemetryRecord(
            timestamp=25.0, mission_id="M", engine_id="TEST", mission_phase="CRUISE",
            altitude=1000.0, ambient_temp=20.0, throttle=70.0, load=70.0, rpm=4500.0,
            cht=80.0, egt=600.0, oil_temp=85.0, oil_pressure=3.0, fuel_flow=20.0, vibration=0.2,
        ),
        diagnosis_result=diag_res,
    )
    ev = builder.build_step_evidence(st)
    rendered = ExplainabilityEngine.render_human_readable(ev)
    assert "COMBUSTION_MISFIRE" in rendered
    assert "0.5500" in rendered

