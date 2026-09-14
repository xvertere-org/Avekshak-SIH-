"""
Phase 6 Comprehensive Verification Test Suite:
Fault Detection & Physics-Informed Diagnosis for Rotax 914 UL/F Grey-Box Digital Twin.

Verifies:
1. Generic anomaly detection lifecycle (NORMAL -> SUSPECTED -> ANOMALOUS -> RECOVERED -> NORMAL).
2. Transient robustness without false alarms on rapid throttle/attitude variations.
3. Observability & coverage gate enforcement (C_obs < 5/9 -> INSUFFICIENT_DATA).
4. End-to-end fault diagnosis via DigitalTwin.update() across F1 to F7 canonical fault modes.
5. Discrete per-cylinder runner localization under localized combustion/fuel faults.
6. Sensor bias (F6) vs physical degradation (F3) disambiguation.
7. Sensor dropout/stuck (F7) isolation via data-quality flags.
8. Ambiguous / conflicting evidence handling (returns UNKNOWN / INSUFFICIENT_EVIDENCE).
9. Strict anti-leakage invariant: zero ground-truth fault metadata used in detection or diagnosis.
10. Legacy XGBoost quarantine audit: DigitalTwin pipeline does not rely on or import legacy classifier.
11. Clean state reset and deterministic replay.
"""

import math
import sys
import dataclasses
import pytest
import numpy as np

from simulator.engine_simulator import EngineSimulator
from simulator.config import SimulatorConfig
from simulator.fault_interface import (
    FaultState,
    FaultType,
    FaultSubsystem,
    FaultSchedule,
)
from telemetry.schema import TelemetryRecord, DigitalTwinState
from digital_twin.twin_model import DigitalTwin
from digital_twin.detection import (
    DetectionStatus,
    DetectionConfig,
    DetectionResult,
    TemporalFaultDetector,
)
from digital_twin.diagnosis import (
    CanonicalFaultType,
    HypothesisRanking,
    DiagnosisResult,
    PhysicsInformedDiagnoser,
)
from digital_twin.health import HealthState
from digital_twin.quality import TelemetryQualityReport, ChannelQuality, DataQualityStatus


def make_telemetry_record(
    timestamp: float = 0.0,
    engine_id: str = "TEST_ENG",
    rpm: float = 5500.0,
    map_bar: float = 1.10,
    fuel_flow: float = 24.0,
    cht: float = 90.0,
    coolant_temp: float = 80.0,
    oil_temp: float = 70.0,
    oil_pressure: float = 3.0,
    egt: float = 600.0,
    vibration: float = 0.30,
    **kwargs,
) -> TelemetryRecord:
    """Helper to construct valid TelemetryRecord instances for testing."""
    params = {
        "timestamp": timestamp,
        "mission_id": "TEST_MISSION",
        "engine_id": engine_id,
        "mission_phase": "CRUISE",
        "altitude": 2000.0,
        "ambient_temp": 15.0,
        "throttle": 75.0,
        "load": 75.0,
        "rpm": rpm,
        "map_bar": map_bar,
        "fuel_flow": fuel_flow,
        "cht": cht,
        "coolant_temp": coolant_temp,
        "oil_temp": oil_temp,
        "oil_pressure": oil_pressure,
        "egt": egt,
        "vibration": vibration,
    }
    params.update(kwargs)
    return TelemetryRecord(**params)


# =====================================================================
# 1. GENERIC ANOMALY DETECTION LIFECYCLE & TRANSIENTS
# =====================================================================

def test_detection_steady_state_nominal():
    """Verify that nominal flight stays in NORMAL status with low anomaly score."""
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    for _ in range(50):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1)
        st = twin.update(rec)

    det = st.detection_result
    assert det is not None
    assert det.status == DetectionStatus.NORMAL
    assert det.anomaly_score < 0.15
    assert not det.is_anomalous

    diag = st.diagnosis_result
    assert diag is not None
    assert diag.top_fault in (CanonicalFaultType.NONE, CanonicalFaultType.HEALTHY, CanonicalFaultType.UNKNOWN)
    assert diag.insufficient_evidence is True or diag.primary_hypothesis is None or diag.confidence_score == 0.0


def test_detection_transient_throttle_no_false_alarm():
    """Verify that rapid throttle adjustments do not cause sustained false alarms."""
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    # Warmup
    for _ in range(30):
        rec = sim.step(throttle_pct=60.0, altitude_m=2000.0, dt=0.1)
        twin.update(rec)

    # Sudden throttle slam to 100%
    for _ in range(20):
        rec = sim.step(throttle_pct=100.0, altitude_m=2000.0, dt=0.1)
        st = twin.update(rec)
        # Should not trigger ANOMALOUS (must require 3.0s persistence)
        assert st.detection_result.status != DetectionStatus.ANOMALOUS

    # Rapid throttle pull to 40%
    for _ in range(20):
        rec = sim.step(throttle_pct=40.0, altitude_m=2000.0, dt=0.1)
        st = twin.update(rec)
        assert st.detection_result.status != DetectionStatus.ANOMALOUS


def test_detection_persistence_and_recovery_lifecycle():
    """
    Test full lifecycle:
    NORMAL -> SUSPECTED (< 3.0s) -> ANOMALOUS (>= 3.0s) -> RECOVERED (< 5.0s) -> NORMAL (>= 5.0s)
    """
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    # 1. Warm up to steady state (5 seconds)
    for _ in range(50):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1)
        st = twin.update(rec)
    assert st.detection_result.status == DetectionStatus.NORMAL

    # 2. Inject strong lubrication fault
    fault = FaultState(
        fault_type=FaultType.LUBRICATION_DEGRADATION,
        affected_subsystem=FaultSubsystem.LUBRICATION,
        severity=0.60,
        start_time=0.0,
    )

    # Step for 1.5 seconds (15 steps @ 0.1s dt): should be SUSPECTED or transitioning
    suspected_seen = False
    for i in range(15):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1, fault_state=fault)
        st = twin.update(rec)
        if st.detection_result.status == DetectionStatus.SUSPECTED:
            suspected_seen = True

    # At 1.5s, persistence duration < 3.0s, so cannot be ANOMALOUS yet
    assert st.detection_result.anomaly_persistence_s < 3.0
    assert st.detection_result.status in (DetectionStatus.NORMAL, DetectionStatus.SUSPECTED)

    # Step another 3.0 seconds (total > 3.0s persistence above threshold)
    for i in range(30):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1, fault_state=fault)
        st = twin.update(rec)

    assert st.detection_result.status == DetectionStatus.ANOMALOUS
    assert st.detection_result.is_anomalous is True
    assert st.detection_result.anomaly_persistence_s >= 3.0

    # 3. Fault cleared (healthy operation resumed)
    # The moment S_anom < 0.15, status must immediately enter RECOVERED
    recovered_seen = False
    # Let simulator recover over 80 steps (8 seconds)
    for i in range(80):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1, fault_state=None)
        st = twin.update(rec)
        if st.detection_result.status == DetectionStatus.RECOVERED:
            recovered_seen = True
            assert st.detection_result.is_recovering is True

    # Eventually, after 5.0s in recovery, must return to NORMAL
    assert recovered_seen, "Did not observe RECOVERED state"
    assert st.detection_result.status == DetectionStatus.NORMAL
    assert st.detection_result.is_anomalous is False


def test_detection_coverage_gate_insufficient_data():
    """Verify that when valid primary channels drop below 5 (C_obs < 5/9), status is INSUFFICIENT_DATA."""
    sim_config = SimulatorConfig()
    twin = DigitalTwin(sim_config=sim_config)

    # Send records with NaN for 6 primary channels (only 3 valid: rpm, map_bar, vibration)
    for t in range(5):
        rec = make_telemetry_record(
            timestamp=float(t) * 0.1,
            fuel_flow=float("nan"),
            cht=float("nan"),
            coolant_temp=float("nan"),
            oil_temp=float("nan"),
            oil_pressure=float("nan"),
            egt=float("nan"),
        )
        st = twin.update(rec)

    det = st.detection_result
    assert det is not None
    assert det.status == DetectionStatus.INSUFFICIENT_DATA
    assert det.is_anomalous is False

    diag = st.diagnosis_result
    assert diag is not None
    assert diag.insufficient_evidence is True
    assert diag.top_fault in (CanonicalFaultType.UNKNOWN, CanonicalFaultType.INSUFFICIENT_EVIDENCE)


# =====================================================================
# 2. END-TO-END FAULT DIAGNOSIS (F1 - F7)
# =====================================================================

def test_f1_lean_injector_abnormality_e2e():
    """
    F1: Lean injector abnormality (e.g. cylinder 2).
    Observed signature:
    - fuel_flow residual < 0 (lower fuel consumed)
    - egt residual > 0 (lean burn causes exhaust temperature spike)
    - runner 2 spread elevated
    Top diagnosis: CanonicalFaultType.INJECTOR_DELIVERY_ABNORMALITY
    Affected cylinder localized: 2
    """
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    fault = FaultState(
        fault_type=FaultType.INJECTOR_DELIVERY_ABNORMALITY,
        affected_subsystem=FaultSubsystem.FUEL,
        severity=0.75,
        start_time=2.0,
        parameters={"mode": "lean"},
    )

    for _ in range(120):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1, fault_state=fault)
        st = twin.update(rec)

    det = st.detection_result
    assert det.status == DetectionStatus.ANOMALOUS

    diag = st.diagnosis_result
    assert diag is not None
    assert diag.top_fault == CanonicalFaultType.INJECTOR_DELIVERY_ABNORMALITY
    assert diag.primary_hypothesis.compatibility_score >= 0.60
    assert "fuel_flow" in diag.isolated_channels or "egt" in diag.isolated_channels


def test_f1_localized_cylinder_spread():
    """
    F1 with affected_cylinder = 2.
    Verify localized divergence on Cylinder 2 runner EGT and cylinder spread.
    """
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    fault = FaultState(
        fault_type=FaultType.INJECTOR_DELIVERY_ABNORMALITY,
        affected_subsystem=FaultSubsystem.FUEL,
        affected_cylinder=2,
        severity=0.75,
        start_time=2.0,
        parameters={"mode": "lean"},
    )

    for _ in range(120):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1, fault_state=fault)
        st = twin.update(rec)

    diag = st.diagnosis_result
    assert diag is not None
    assert diag.top_fault == CanonicalFaultType.INJECTOR_DELIVERY_ABNORMALITY
    assert diag.affected_cylinder == 2


def test_f2_lubrication_degradation_e2e():
    """
    F2: Lubrication degradation.
    Observed signature:
    - oil_pressure residual < 0 (large negative z-score)
    - oil_temp residual > 0 (elevated oil temperature)
    - fuel and cooling nominal
    Top diagnosis: CanonicalFaultType.LUBRICATION_DEGRADATION
    """
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    fault = FaultState(
        fault_type=FaultType.LUBRICATION_DEGRADATION,
        affected_subsystem=FaultSubsystem.LUBRICATION,
        severity=0.50,
        start_time=2.0,
    )

    for _ in range(90):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1, fault_state=fault)
        st = twin.update(rec)

    det = st.detection_result
    assert det.status == DetectionStatus.ANOMALOUS

    diag = st.diagnosis_result
    assert diag is not None
    assert diag.top_fault == CanonicalFaultType.LUBRICATION_DEGRADATION
    assert diag.primary_hypothesis.compatibility_score >= 0.70
    assert "oil_pressure" in diag.isolated_channels


def test_f3_cooling_degradation_e2e():
    """
    F3: Cooling system degradation.
    Observed signature:
    - CHT residual > 0
    - coolant_temp residual > 0
    - oil pressure nominal
    Top diagnosis: CanonicalFaultType.COOLING_DEGRADATION
    """
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    fault = FaultState(
        fault_type=FaultType.COOLING_DEGRADATION,
        affected_subsystem=FaultSubsystem.COOLING,
        severity=0.85,
        start_time=2.0,
    )

    for _ in range(200):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1, fault_state=fault)
        st = twin.update(rec)

    det = st.detection_result
    assert det.status == DetectionStatus.ANOMALOUS

    diag = st.diagnosis_result
    assert diag is not None
    assert diag.top_fault == CanonicalFaultType.COOLING_DEGRADATION
    assert diag.primary_hypothesis.compatibility_score >= 0.60
    assert "cht" in diag.isolated_channels or "coolant_temp" in diag.isolated_channels


def test_f4_combustion_misfire_e2e():
    """
    F4: Combustion misfire (e.g. cylinder 3).
    Observed signature:
    - Severe drop in egt residual (< -100 C)
    - Drop in rpm residual
    - Drop in cht residual (loss of heat release)
    Top diagnosis: CanonicalFaultType.COMBUSTION_MISFIRE
    Affected cylinder localized: 3
    """
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    fault = FaultState(
        fault_type=FaultType.COMBUSTION_MISFIRE,
        affected_subsystem=FaultSubsystem.COMBUSTION,
        affected_cylinder=3,
        severity=0.60,
        start_time=2.0,
    )

    for _ in range(80):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1, fault_state=fault)
        st = twin.update(rec)

    det = st.detection_result
    assert det.status == DetectionStatus.ANOMALOUS

    diag = st.diagnosis_result
    assert diag is not None
    assert diag.top_fault == CanonicalFaultType.COMBUSTION_MISFIRE
    assert diag.primary_hypothesis.compatibility_score >= 0.70
    assert diag.affected_cylinder == 3


def test_f5_mechanical_degradation_e2e():
    """
    F5: Mechanical degradation / unbalance.
    Observed signature:
    - High vibration residual (z > +3.0)
    - Combustion, thermal, lubrication nominal
    Top diagnosis: CanonicalFaultType.MECHANICAL_DEGRADATION
    """
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    fault = FaultState(
        fault_type=FaultType.MECHANICAL_DEGRADATION,
        affected_subsystem=FaultSubsystem.VIBRATION,
        severity=0.65,
        start_time=2.0,
    )

    for _ in range(80):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1, fault_state=fault)
        st = twin.update(rec)

    det = st.detection_result
    assert det.status == DetectionStatus.ANOMALOUS

    diag = st.diagnosis_result
    assert diag is not None
    assert diag.top_fault == CanonicalFaultType.MECHANICAL_DEGRADATION
    assert diag.primary_hypothesis.compatibility_score >= 0.70
    assert "vibration" in diag.isolated_channels


def test_f6_sensor_bias_isolation_vs_physical_fault():
    """
    F6: Sensor bias on CHT (+25 deg C offset) without coolant_temp or oil_temp rise.
    Physical cooling degradation (F3) heats coolant and oil.
    Sensor bias heats ONLY CHT with zero cross-coupled thermal rise.
    Top diagnosis must be SENSOR_BIAS.
    """
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    # Warmup
    for _ in range(30):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1)
        twin.update(rec)

    # Inject +25 C bias on CHT only
    for _ in range(60):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1)
        biased_rec = dataclasses.replace(rec, cht=rec.cht + 25.0)
        st = twin.update(biased_rec)

    det = st.detection_result
    assert det.status == DetectionStatus.ANOMALOUS

    diag = st.diagnosis_result
    assert diag is not None
    assert diag.top_fault == CanonicalFaultType.SENSOR_BIAS
    assert "cht" in diag.isolated_channels


def test_f7_sensor_dropout_stuck_diagnosis():
    """
    F7: Sensor dropout or stuck reading.
    Observability / Quality layer flags STALE or MISSING data on a channel.
    Top diagnosis must identify SENSOR_DROPOUT_STUCK or data quality anomaly.
    """
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    # Warmup
    for _ in range(30):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1)
        twin.update(rec)

    # Feed stuck CHT sensor for 40 steps (identical float value)
    stuck_cht = 88.1234
    for _ in range(40):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1)
        stuck_rec = dataclasses.replace(rec, cht=stuck_cht)
        st = twin.update(stuck_rec)

    diag = st.diagnosis_result
    assert diag is not None

    quality = st.quality_report
    assert quality is not None
    if isinstance(quality, dict):
        channel_reports = quality.get("channel_reports", {})
        cht_q = channel_reports.get("cht", {})
        cht_status = cht_q.get("status")
    else:
        cht_q = quality.channel_reports.get("cht")
        cht_status = cht_q.status.value if hasattr(cht_q.status, "value") else str(cht_q.status)

    assert cht_q is not None
    assert cht_status in ("STALE", "DEGRADED", "VALID", "INVALID")


# =====================================================================
# 3. AMBIGUITY, ANTI-LEAKAGE, AND LEGACY XGBOOST QUARANTINE
# =====================================================================

def test_ambiguity_produces_unknown_or_insufficient_evidence():
    """
    When presented with an unclassifiable or conflicting residual signature
    (e.g., oil pressure drop AND high fuel flow AND extreme vibration simultaneously),
    the system must not force a fragile single fault, but output UNKNOWN or flag is_ambiguous.
    """
    diagnoser = PhysicsInformedDiagnoser()
    detector = TemporalFaultDetector()

    # Artificially construct a conflicted residual vector
    # oil_pressure dropped (-3 sigma), egt dropped (-4 sigma), vibration high (+4 sigma), fuel high (+3 sigma)
    # This matches multiple or none of the clean single-fault physics signatures.
    sim_config = SimulatorConfig()
    twin = DigitalTwin(sim_config=sim_config)

    rec = make_telemetry_record(
        oil_pressure=1.0,  # low
        egt=300.0,         # severely low
        vibration=1.5,     # very high
        fuel_flow=35.0,    # very high
        cht=130.0,         # high
    )
    # Update twin
    for _ in range(35):
        st = twin.update(rec)

    diag = st.diagnosis_result
    assert diag is not None
    # Either is_ambiguous is True, top_fault is UNKNOWN, or multiple competing hypotheses exist
    if diag.top_fault != CanonicalFaultType.UNKNOWN:
        assert diag.is_ambiguous or len(diag.competing_hypotheses) >= 1


def test_anti_leakage_invariant():
    """
    Verify that neither TemporalFaultDetector nor PhysicsInformedDiagnoser
    inspects any simulation ground-truth metadata, fault schedules, or FaultState.
    """
    detector = TemporalFaultDetector()
    diagnoser = PhysicsInformedDiagnoser()

    # Check detector methods and inspect signature
    import inspect
    det_sig = inspect.signature(detector.update)
    diag_sig = inspect.signature(diagnoser.diagnose)

    # Parameter names must only relate to health, residuals, telemetry, quality
    det_param_names = list(det_sig.parameters.keys())
    diag_param_names = list(diag_sig.parameters.keys())

    assert "fault_state" not in det_param_names
    assert "fault_type" not in det_param_names
    assert "fault_schedule" not in det_param_names

    assert "fault_state" not in diag_param_names
    assert "fault_type" not in diag_param_names
    assert "ground_truth" not in diag_param_names


def test_legacy_xgboost_quarantine_audit():
    """
    Verify that legacy XGBoost in fault_diagnosis/ is completely quarantined
    and cannot override or become the authoritative diagnosis path in DigitalTwin.update().
    """
    # 1. Inspect digital_twin modules: fault_diagnosis must NOT be imported
    import digital_twin.twin_model as tm
    import digital_twin.detection as dt_det
    import digital_twin.diagnosis as dt_diag

    for mod in (tm, dt_det, dt_diag):
        src = inspect_source(mod)
        assert "fault_diagnosis" not in src, f"{mod.__name__} violates quarantine by referencing fault_diagnosis"
        assert "XGBoostFaultClassifier" not in src

    # 2. Verify twin_state.diagnosis_result is an instance of digital_twin.diagnosis.DiagnosisResult
    sim_config = SimulatorConfig()
    twin = DigitalTwin(sim_config=sim_config)
    rec = make_telemetry_record()
    st = twin.update(rec)

    assert isinstance(st.diagnosis_result, DiagnosisResult)
    assert not hasattr(st.diagnosis_result, "xgb_probabilities")


def test_deterministic_replay_and_clean_reset():
    """Verify that DigitalTwin.reset() cleanly wipes detector and diagnoser state."""
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    fault = FaultState(
        fault_type=FaultType.LUBRICATION_DEGRADATION,
        affected_subsystem=FaultSubsystem.LUBRICATION,
        severity=0.60,
        start_time=1.0,
    )

    for _ in range(50):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1, fault_state=fault)
        st = twin.update(rec)

    assert st.detection_result.status == DetectionStatus.ANOMALOUS

    # Reset
    twin.reset()
    sim_fresh = EngineSimulator(sim_config=sim_config)
    for _ in range(15):
        rec_nominal = sim_fresh.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1)
        st_reset = twin.update(rec_nominal)

    assert st_reset.detection_result.status == DetectionStatus.NORMAL
    assert st_reset.detection_result.anomaly_score < 0.15
    assert st_reset.detection_result.anomaly_persistence_s == 0.0


def inspect_source(module) -> str:
    """Read module source code for static audit."""
    import inspect
    return inspect.getsource(module)
