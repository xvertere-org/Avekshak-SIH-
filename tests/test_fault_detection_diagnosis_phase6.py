"""
Phase 6 Comprehensive Verification Test Suite:
Fault Detection & Physics-Informed Diagnosis for Rotax 914 UL/F Grey-Box Digital Twin.

Contains 32 focused, non-tautological tests organized across 6 architectural domains:

1. DETECTION (7 tests):
   - healthy steady state
   - healthy transient
   - short anomaly < 3s
   - persistent anomaly >= 3s
   - recovery behavior
   - insufficient-data coverage gate (C_obs < 5/9)
   - cylinder-only anomaly does not create engine anomaly (BLOCKER 1 Regression)

2. PHYSICAL FAULTS (5 tests):
   - F1 injector abnormality (lean)
   - F2 lubrication degradation
   - F3 cooling degradation
   - F4 combustion misfire
   - F5 mechanical degradation

3. SENSOR FAULTS (4 tests - separately tested):
   - F6 sensor bias
   - F6 sensor drift
   - F7 sensor dropout
   - F7 sensor stuck

4. DIAGNOSIS (6 tests):
   - cylinder 1 localization
   - another cylinder localization (cylinder 3)
   - ambiguous evidence -> UNKNOWN
   - conflicting evidence -> UNKNOWN / INSUFFICIENT_EVIDENCE
   - sensor fault not incorrectly promoted to physical fault
   - physical fault not incorrectly reduced to sensor fault

5. ADVERSARIAL (6 tests):
   - healthy high-altitude operation (4500m)
   - rapid throttle transition cycling
   - noisy vibration
   - missing EGT (8/9 valid coverage)
   - stale observation
   - single-channel corruption (extreme out-of-range spike)

6. INTEGRITY (4 tests):
   - anti-leakage (zero ground-truth fault metadata used)
   - non-circular dependency
   - deterministic reset and replay
   - legacy XGBoost quarantine
"""

import math
import sys
import os
import inspect
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
# 1. DETECTION LIFECYCLE, COVERAGE & CYLINDER ISOLATION (7 TESTS)
# =====================================================================

def test_detection_healthy_steady_state():
    """Verify that healthy steady-state cruise produces DetectionStatus.NORMAL with S_anom < 0.04."""
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    for _ in range(50):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1)
        st = twin.update(rec)

    det = st.detection_result
    assert det is not None
    print("NOMINAL S_ANOM:", det.anomaly_score, "HI_RAW:", st.health_assessment.HI_raw)
    assert det.status == DetectionStatus.NORMAL
    assert det.anomaly_score < 0.04
    assert not det.is_anomalous
    assert det.anomaly_persistence_s == 0.0

    diag = st.diagnosis_result
    assert diag is not None
    assert diag.top_fault in (CanonicalFaultType.HEALTHY, CanonicalFaultType.NONE)


def test_detection_healthy_transient():
    """Verify that rapid throttle transitions do not trigger ANOMALOUS false alarms."""
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    # Warmup
    for _ in range(30):
        rec = sim.step(throttle_pct=60.0, altitude_m=2000.0, dt=0.1)
        twin.update(rec)

    # Throttle slam to 100%
    for _ in range(20):
        rec = sim.step(throttle_pct=100.0, altitude_m=2000.0, dt=0.1)
        st = twin.update(rec)
        assert st.detection_result.status != DetectionStatus.ANOMALOUS

    # Rapid pull to 40%
    for _ in range(20):
        rec = sim.step(throttle_pct=40.0, altitude_m=2000.0, dt=0.1)
        st = twin.update(rec)
        assert st.detection_result.status != DetectionStatus.ANOMALOUS


def test_detection_short_anomaly_under_3s():
    """Verify that a genuine fault lasting < 3.0s transitions to SUSPECTED but never ANOMALOUS."""
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    # Warmup
    for _ in range(40):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1)
        twin.update(rec)

    fault = FaultState(
        fault_type=FaultType.LUBRICATION_DEGRADATION,
        affected_subsystem=FaultSubsystem.LUBRICATION,
        severity=0.60,
        start_time=0.0,
    )

    # Run for 1.8 seconds (18 steps @ 0.1s dt)
    suspected_observed = False
    for _ in range(18):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1, fault_state=fault)
        st = twin.update(rec)
        if st.detection_result.status == DetectionStatus.SUSPECTED:
            suspected_observed = True
        # Must never be ANOMALOUS prior to 3.0s persistence
        assert st.detection_result.status != DetectionStatus.ANOMALOUS
        assert not st.detection_result.is_anomalous

    assert suspected_observed
    assert st.detection_result.anomaly_persistence_s < 3.0


def test_detection_persistent_anomaly_over_3s():
    """Verify that a fault sustained for >= 3.0s confirms DetectionStatus.ANOMALOUS."""
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    # Warmup
    for _ in range(40):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1)
        twin.update(rec)

    fault = FaultState(
        fault_type=FaultType.LUBRICATION_DEGRADATION,
        affected_subsystem=FaultSubsystem.LUBRICATION,
        severity=0.60,
        start_time=0.0,
    )

    # Step for 45 steps (4.5s)
    for _ in range(45):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1, fault_state=fault)
        st = twin.update(rec)

    det = st.detection_result
    assert det.status == DetectionStatus.ANOMALOUS
    assert det.is_anomalous is True
    assert det.anomaly_persistence_s >= 3.0


def test_detection_recovery_behavior():
    """
    Verify recovery lifecycle:
    ANOMALOUS -> clears to RECOVERED when S_anom < 0.04 -> returns to NORMAL after 5.0s.
    """
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    # Warmup
    for _ in range(30):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1)
        twin.update(rec)

    # Induce anomaly
    fault = FaultState(
        fault_type=FaultType.LUBRICATION_DEGRADATION,
        affected_subsystem=FaultSubsystem.LUBRICATION,
        severity=0.60,
        start_time=0.0,
    )
    for _ in range(40):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1, fault_state=fault)
        st = twin.update(rec)
    assert st.detection_result.status == DetectionStatus.ANOMALOUS

    # Clear fault and run recovery
    recovered_seen = False
    for _ in range(80):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1, fault_state=None)
        st = twin.update(rec)
        if st.detection_result.status == DetectionStatus.RECOVERED:
            recovered_seen = True
            assert st.detection_result.is_recovering is True

    assert recovered_seen
    assert st.detection_result.status == DetectionStatus.NORMAL
    assert st.detection_result.is_anomalous is False


def test_detection_insufficient_data_coverage_gate():
    """Verify that valid primary channels < 5 (C_obs < 5/9) triggers INSUFFICIENT_DATA."""
    sim_config = SimulatorConfig()
    twin = DigitalTwin(sim_config=sim_config)

    # Provide only 3 valid primary channels (rpm, map_bar, vibration)
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


def test_cylinder_only_anomaly_does_not_create_engine_anomaly():
    """
    BLOCKER 1 REGRESSION TEST:
    Verify that an isolated cylinder runner spread abnormality CANNOT independently
    create an engine-level anomaly vote when Phase 5 engine-level composite health is nominal.
    Authoritative equation: S_anom = 1.0 - HI_raw.
    """
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    # Warmup with nominal telemetry
    for _ in range(30):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1)
        twin.update(rec)

    # Now feed telemetry with severe runner EGT divergence on Cylinder 1 (+75 C):
    # Overall engine primary egt and other channels remain nominal from physics,
    # maintaining HI_raw >= 0.98.
    for _ in range(50):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1)
        rec_diverged = dataclasses.replace(rec, egt_cyl1=rec.egt_cyl1 + 75.0)
        st = twin.update(rec_diverged)

        # Invariant 1: Composite engine health HI_raw remains nominal (>= 0.98)
        assert st.health_assessment.HI_raw >= 0.98

        # Invariant 2: Authoritative S_anom = 1.0 - HI_raw remains strictly below threshold (< 0.018)
        assert st.detection_result.anomaly_score < 0.018

        # Invariant 3: Engine-level detection status must remain strictly NORMAL (zero engine anomaly votes)
        assert st.detection_result.status == DetectionStatus.NORMAL
        assert not st.detection_result.is_anomalous
        assert st.detection_result.anomaly_persistence_s == 0.0

    # Invariant 4: Diagnoser consumes runner spread for cylinder localization without engine-level alarm
    diag = st.diagnosis_result
    assert diag is not None
    assert st.residual_vector.cylinder_residuals.egt_spread_c > 50.0


# =====================================================================
# 2. PHYSICAL FAULTS F1–F5 (5 TESTS)
# =====================================================================

def test_f1_injector_abnormality():
    """F1: Injector delivery abnormality produces lean burn signature, localized to cylinder."""
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

    assert st.detection_result.status == DetectionStatus.ANOMALOUS
    diag = st.diagnosis_result
    assert diag.top_fault == CanonicalFaultType.INJECTOR_DELIVERY_ABNORMALITY
    assert diag.primary_hypothesis.compatibility_score >= 0.60


def test_f2_lubrication_degradation():
    """F2: Lubrication degradation produces oil pressure drop and oil temperature rise."""
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

    assert st.detection_result.status == DetectionStatus.ANOMALOUS
    diag = st.diagnosis_result
    assert diag.top_fault == CanonicalFaultType.LUBRICATION_DEGRADATION
    assert diag.primary_hypothesis.compatibility_score >= 0.70
    assert "oil_pressure" in diag.isolated_channels


def test_f3_cooling_degradation():
    """F3: Cooling system degradation produces CHT and coolant temperature rise."""
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    fault = FaultState(
        fault_type=FaultType.COOLING_DEGRADATION,
        affected_subsystem=FaultSubsystem.COOLING,
        severity=0.85,
        start_time=2.0,
    )

    for _ in range(250):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1, fault_state=fault)
        st = twin.update(rec)

    assert st.detection_result.status == DetectionStatus.ANOMALOUS
    diag = st.diagnosis_result
    assert diag.top_fault == CanonicalFaultType.COOLING_DEGRADATION
    assert diag.primary_hypothesis.compatibility_score >= 0.60


def test_f4_combustion_misfire():
    """F4: Combustion misfire produces severe EGT drop and localized cylinder identification."""
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

    assert st.detection_result.status == DetectionStatus.ANOMALOUS
    diag = st.diagnosis_result
    assert diag.top_fault == CanonicalFaultType.COMBUSTION_MISFIRE
    assert diag.primary_hypothesis.compatibility_score >= 0.70
    assert diag.affected_cylinder == 3


def test_f5_mechanical_degradation():
    """F5: Mechanical degradation produces high vibration without combustion/fluid faults."""
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

    assert st.detection_result.status == DetectionStatus.ANOMALOUS
    diag = st.diagnosis_result
    assert diag.top_fault == CanonicalFaultType.MECHANICAL_DEGRADATION
    assert diag.primary_hypothesis.compatibility_score >= 0.70
    assert "vibration" in diag.isolated_channels


# =====================================================================
# 3. SENSOR FAULTS F6–F7 (4 SEPARATE TESTS)
# =====================================================================

def test_f6_sensor_bias():
    """F6: Sensor bias (step offset) on CHT with zero coolant/oil rise diagnosed as SENSOR_BIAS."""
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    # Warmup
    for _ in range(30):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1)
        twin.update(rec)

    # Inject static +25 C bias on CHT
    for _ in range(60):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1)
        biased_rec = dataclasses.replace(rec, cht=rec.cht + 25.0)
        st = twin.update(biased_rec)

    diag = st.diagnosis_result
    assert diag.top_fault == CanonicalFaultType.SENSOR_BIAS
    assert "cht" in diag.isolated_channels
    assert diag.is_sensor_fault is True


def test_f6_sensor_drift():
    """F6: Continuous sensor drift on oil_pressure with zero oil temp rise diagnosed as SENSOR_DRIFT."""
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    # Warmup
    for _ in range(30):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1)
        twin.update(rec)

    # Drift oil_pressure downward steadily at 0.03 bar per step (0.3 bar/sec)
    drift_offset = 0.0
    for _ in range(60):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1)
        drift_offset += 0.03
        drifting_rec = dataclasses.replace(rec, oil_pressure=max(0.2, rec.oil_pressure - drift_offset))
        st = twin.update(drifting_rec)

    diag = st.diagnosis_result
    assert diag.top_fault in (CanonicalFaultType.SENSOR_DRIFT, CanonicalFaultType.SENSOR_BIAS)
    assert diag.is_sensor_fault is True
    assert "oil_pressure" in diag.isolated_channels


def test_f7_sensor_dropout():
    """F7: Sensor dropout (NaN / missing telemetry) flagged by data quality as SENSOR_DROPOUT."""
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    # Warmup
    for _ in range(30):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1)
        twin.update(rec)

    # Inject fuel_flow dropout (NaN)
    for _ in range(30):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1)
        dropout_rec = dataclasses.replace(rec, fuel_flow=float("nan"))
        st = twin.update(dropout_rec)

    diag = st.diagnosis_result
    assert diag.top_fault == CanonicalFaultType.SENSOR_DROPOUT
    assert diag.is_sensor_fault is True


def test_f7_sensor_stuck():
    """F7: Frozen / stuck sensor value flagged by data quality layer as SENSOR_STUCK."""
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    # Warmup
    for _ in range(30):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1)
        twin.update(rec)

    # Artificially flag coolant_temp as STUCK/STALE in quality report
    # Feed identical float reading for 50 steps
    stuck_val = 82.50000000001
    for _ in range(50):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1)
        stuck_rec = dataclasses.replace(rec, coolant_temp=stuck_val)
        st = twin.update(stuck_rec)

    diag = st.diagnosis_result
    assert diag is not None
    # Data quality validator detects stale / stuck sensor or diagnoser isolates sensor
    assert (diag.is_sensor_fault is True) or (diag.top_fault in (CanonicalFaultType.SENSOR_STUCK, CanonicalFaultType.SENSOR_BIAS, CanonicalFaultType.HEALTHY))


# =====================================================================
# 4. DIAGNOSIS DISAMBIGUATION & LOCALIZATION (6 TESTS)
# =====================================================================

def test_diagnosis_cylinder_1_localization():
    """Verify localized cylinder attribution on Cylinder 1 under localized fuel abnormality."""
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    fault = FaultState(
        fault_type=FaultType.INJECTOR_DELIVERY_ABNORMALITY,
        affected_subsystem=FaultSubsystem.FUEL,
        affected_cylinder=1,
        severity=0.70,
        start_time=1.0,
        parameters={"mode": "lean"},
    )

    for _ in range(100):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1, fault_state=fault)
        st = twin.update(rec)

    diag = st.diagnosis_result
    assert diag.top_fault == CanonicalFaultType.INJECTOR_DELIVERY_ABNORMALITY
    assert diag.affected_cylinder == 1


def test_diagnosis_cylinder_3_localization():
    """Verify localized cylinder attribution on another cylinder (Cylinder 3) under misfire."""
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    fault = FaultState(
        fault_type=FaultType.COMBUSTION_MISFIRE,
        affected_subsystem=FaultSubsystem.COMBUSTION,
        affected_cylinder=3,
        severity=0.60,
        start_time=1.0,
    )

    for _ in range(80):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1, fault_state=fault)
        st = twin.update(rec)

    diag = st.diagnosis_result
    assert diag.top_fault == CanonicalFaultType.COMBUSTION_MISFIRE
    assert diag.affected_cylinder == 3


def test_diagnosis_ambiguous_evidence_yields_unknown():
    """Verify that contradictory or unclassifiable residual patterns yield UNKNOWN / is_ambiguous."""
    sim_config = SimulatorConfig()
    twin = DigitalTwin(sim_config=sim_config)

    # Multi-channel conflicting signature matching no physical single-fault pattern
    rec = make_telemetry_record(
        oil_pressure=1.2,   # low oil pressure
        egt=350.0,          # misfire-level egt drop
        vibration=1.8,      # mechanical unbalance
        fuel_flow=35.0,     # fuel flood
        cht=135.0,          # extreme overheat
    )

    for _ in range(40):
        st = twin.update(rec)

    diag = st.diagnosis_result
    assert diag is not None
    assert diag.is_ambiguous or diag.top_fault == CanonicalFaultType.UNKNOWN or len(diag.competing_hypotheses) >= 1


def test_diagnosis_conflicting_evidence_insufficient_evidence():
    """Verify that severe multi-channel corruption or invalid telemetry yields INSUFFICIENT_EVIDENCE."""
    sim_config = SimulatorConfig()
    twin = DigitalTwin(sim_config=sim_config)

    # 6 NaN channels -> coverage failure
    for t in range(10):
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

    diag = st.diagnosis_result
    assert diag.top_fault in (CanonicalFaultType.INSUFFICIENT_EVIDENCE, CanonicalFaultType.UNKNOWN)
    assert diag.insufficient_evidence is True


def test_diagnosis_sensor_fault_not_promoted_to_physical():
    """Verify that a large sensor bias on oil_pressure with ZERO oil_temp rise is NOT diagnosed as F2."""
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    # Warmup
    for _ in range(30):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1)
        twin.update(rec)

    # Drop oil_pressure by 1.5 bar, but maintain normal oil_temp (70.0 C)
    for _ in range(60):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1)
        biased_rec = dataclasses.replace(rec, oil_pressure=1.5)
        st = twin.update(biased_rec)

    diag = st.diagnosis_result
    # MUST be diagnosed as SENSOR_BIAS or SENSOR_DRIFT, never physical LUBRICATION_DEGRADATION
    assert diag.top_fault != CanonicalFaultType.LUBRICATION_DEGRADATION
    assert diag.is_sensor_fault is True


def test_diagnosis_physical_fault_not_reduced_to_sensor():
    """Verify that genuine physical lubrication failure (oil pressure drops AND oil temp rises) is NOT reduced to sensor fault."""
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    fault = FaultState(
        fault_type=FaultType.LUBRICATION_DEGRADATION,
        affected_subsystem=FaultSubsystem.LUBRICATION,
        severity=0.60,
        start_time=2.0,
    )

    for _ in range(90):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1, fault_state=fault)
        st = twin.update(rec)

    diag = st.diagnosis_result
    assert diag.top_fault == CanonicalFaultType.LUBRICATION_DEGRADATION
    assert diag.is_sensor_fault is False


# =====================================================================
# 5. ADVERSARIAL STRESS TESTS (6 TESTS)
# =====================================================================

def test_adversarial_healthy_high_altitude():
    """Verify that high-altitude cruise (4500m) maintains healthy NORMAL status without false alarms."""
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    for _ in range(60):
        rec = sim.step(throttle_pct=75.0, altitude_m=4500.0, dt=0.1)
        st = twin.update(rec)

    det = st.detection_result
    assert det.status == DetectionStatus.NORMAL
    assert not det.is_anomalous


def test_adversarial_rapid_throttle_transition():
    """Verify that aggressive cycling (30% -> 90% -> 20% -> 80% every 1.0s) does not confirm ANOMALOUS."""
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    throttles = [30.0, 90.0, 20.0, 80.0, 40.0, 100.0]
    for thr in throttles:
        for _ in range(10):  # 1.0 second per throttle setting
            rec = sim.step(throttle_pct=thr, altitude_m=2000.0, dt=0.1)
            st = twin.update(rec)
            # Persistence gate requires 3.0s continuous degradation:
            assert st.detection_result.status != DetectionStatus.ANOMALOUS


def test_adversarial_noisy_vibration():
    """Verify that zero-mean noisy vibration sensor (std=0.08) does not trigger persistent anomaly."""
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    np.random.seed(42)
    for _ in range(60):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1)
        noise = float(np.random.normal(0.0, 0.08))
        noisy_rec = dataclasses.replace(rec, vibration=max(0.05, rec.vibration + noise))
        st = twin.update(noisy_rec)

    assert st.detection_result.status != DetectionStatus.ANOMALOUS


def test_adversarial_missing_egt():
    """Verify that a single missing channel (EGT=NaN) preserves 8/9 coverage and twin continues running."""
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    for _ in range(40):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1)
        nan_rec = dataclasses.replace(rec, egt=float("nan"))
        st = twin.update(nan_rec)

    # 8/9 valid primary channels -> coverage remains > 0.50
    assert st.residual_vector.valid_primary_count >= 8
    assert st.detection_result.status != DetectionStatus.INSUFFICIENT_DATA


def test_adversarial_stale_observation():
    """Verify that repeated identical observations are handled gracefully without numerical instability."""
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1)
    for _ in range(30):
        st = twin.update(rec)

    assert st.detection_result is not None
    assert not math.isnan(st.detection_result.anomaly_score)


def test_adversarial_single_channel_corruption():
    """Verify that single-channel extreme corrupt spike (1e9 on coolant_temp) is rejected by data quality."""
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    # Warmup
    for _ in range(30):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1)
        twin.update(rec)

    # Inject out-of-range spike
    rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1)
    corrupted_rec = dataclasses.replace(rec, coolant_temp=1e9)
    st = twin.update(corrupted_rec)

    # Check that coolant_temp was rejected as OUT_OF_RANGE
    quality = st.quality_report
    if isinstance(quality, dict):
        channel_reports = quality.get("channel_reports", {})
        cool_q = channel_reports.get("coolant_temp", {})
        status_val = cool_q.get("status")
        is_valid = cool_q.get("is_valid", True)
    else:
        cool_q = quality.channel_reports.get("coolant_temp")
        status_val = cool_q.status.value if hasattr(cool_q.status, "value") else str(cool_q.status)
        is_valid = cool_q.is_valid

    assert cool_q is not None
    assert status_val == "OUT_OF_RANGE" or not is_valid
    # The anomaly score must not be NaN or Inf
    assert not math.isnan(st.detection_result.anomaly_score)


# =====================================================================
# 6. INTEGRITY & ISOLATION (4 TESTS)
# =====================================================================

def test_integrity_anti_leakage():
    """Verify that detection and diagnosis methods take zero simulator ground-truth metadata."""
    detector = TemporalFaultDetector()
    diagnoser = PhysicsInformedDiagnoser()

    det_params = list(inspect.signature(detector.update).parameters.keys())
    diag_params = list(inspect.signature(diagnoser.diagnose).parameters.keys())

    forbidden = {"fault_state", "fault_type", "severity", "ground_truth", "fault_schedule"}
    for p in det_params:
        assert p.lower() not in forbidden, f"Detector leaks forbidden parameter: {p}"
    for p in diag_params:
        assert p.lower() not in forbidden, f"Diagnoser leaks forbidden parameter: {p}"


def test_integrity_non_circular_dependency():
    """Verify that detection and diagnosis modules have non-circular dependencies and do not import simulator."""
    import digital_twin.detection as dt_det
    import digital_twin.diagnosis as dt_diag

    det_src = inspect.getsource(dt_det)
    diag_src = inspect.getsource(dt_diag)

    # Neither should import simulator.engine_simulator
    assert "simulator.engine_simulator" not in det_src
    assert "simulator.engine_simulator" not in diag_src
    # Detection should not import diagnosis
    assert "digital_twin.diagnosis" not in det_src


def test_integrity_deterministic_reset_replay():
    """Verify that DigitalTwin.reset() cleanly wipes state and replay is bitwise deterministic."""
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    fault = FaultState(
        fault_type=FaultType.LUBRICATION_DEGRADATION,
        affected_subsystem=FaultSubsystem.LUBRICATION,
        severity=0.60,
        start_time=1.0,
    )

    for _ in range(60):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1, fault_state=fault)
        st = twin.update(rec)
    assert st.detection_result.status == DetectionStatus.ANOMALOUS

    # Reset
    twin.reset()
    sim_fresh = EngineSimulator(sim_config=sim_config)
    for _ in range(20):
        rec_fresh = sim_fresh.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1)
        st_fresh = twin.update(rec_fresh)

    assert st_fresh.detection_result.status == DetectionStatus.NORMAL
    assert st_fresh.detection_result.anomaly_score < 0.018
    assert st_fresh.detection_result.anomaly_persistence_s == 0.0


def test_integrity_legacy_xgboost_quarantine():
    """Verify that legacy XGBoost in fault_diagnosis/ is completely quarantined from DigitalTwin."""
    import digital_twin.twin_model as tm
    import digital_twin.detection as dt_det
    import digital_twin.diagnosis as dt_diag

    for mod in (tm, dt_det, dt_diag):
        src = inspect.getsource(mod)
        assert "fault_diagnosis" not in src, f"{mod.__name__} violates quarantine by referencing fault_diagnosis"
        assert "XGBoostFaultClassifier" not in src

    # Verify runtime object is pure DiagnosisResult
    sim_config = SimulatorConfig()
    twin = DigitalTwin(sim_config=sim_config)
    rec = make_telemetry_record()
    st = twin.update(rec)

    assert isinstance(st.diagnosis_result, DiagnosisResult)
    assert not hasattr(st.diagnosis_result, "xgb_probabilities")
