"""
Unit and Integration Tests for Phase 4A Fault & Degradation Interface.
"""

import pytest
from simulator.fault_interface import (
    FaultType,
    FaultSubsystem,
    FaultState,
    FaultSchedule,
)
from simulator.engine_simulator import EngineSimulator
from simulator.subsystems.mission import MissionProfile, PhaseSegment, FlightPhase
from telemetry.schema import TelemetryRecord


def test_valid_fault_state_creation_and_defaults():
    """Verify standard valid FaultState creation and defaults."""
    f = FaultState(
        fault_type=FaultType.COOLING_DEGRADATION,
        severity=0.6,
        start_time=100.0,
        end_time=300.0,
        affected_subsystem=FaultSubsystem.THERMAL,
        parameters={"coolant_leak_rate_pct": 25.0},
    )
    assert f.fault_type == FaultType.COOLING_DEGRADATION
    assert f.severity == 0.6
    assert f.active is True
    assert f.start_time == 100.0
    assert f.end_time == 300.0
    assert f.affected_subsystem == FaultSubsystem.THERMAL
    assert f.parameters["coolant_leak_rate_pct"] == 25.0


def test_severity_range_validation():
    """Verify fault severity is strictly bounded in [0.0, 1.0]."""
    # Valid boundaries
    f_min = FaultState(fault_type=FaultType.LUBRICATION_DEGRADATION, severity=0.0)
    assert f_min.severity == 0.0

    f_max = FaultState(fault_type=FaultType.LUBRICATION_DEGRADATION, severity=1.0)
    assert f_max.severity == 1.0

    # Invalid negative
    with pytest.raises(ValueError, match="normalized in range"):
        FaultState(fault_type=FaultType.LUBRICATION_DEGRADATION, severity=-0.1)

    # Invalid above 1.0
    with pytest.raises(ValueError, match="normalized in range"):
        FaultState(fault_type=FaultType.LUBRICATION_DEGRADATION, severity=1.05)


def test_invalid_fault_type_rejected():
    """Verify invalid fault type strings are rejected."""
    with pytest.raises(ValueError, match="Invalid fault_type"):
        FaultState(fault_type="non_existent_engine_fault", severity=0.5)


def test_invalid_affected_subsystem_rejected():
    """Verify invalid affected subsystem strings are rejected."""
    with pytest.raises(ValueError, match="Invalid affected_subsystem"):
        FaultState(
            fault_type=FaultType.COOLING_DEGRADATION,
            severity=0.5,
            affected_subsystem="quantum_drive",
        )


def test_start_end_timing_validation():
    """Verify start and end timing rules."""
    # Negative start time rejected
    with pytest.raises(ValueError, match="start_time must be non-negative"):
        FaultState(fault_type=FaultType.COOLING_DEGRADATION, severity=0.5, start_time=-10.0)

    # end_time earlier than start_time rejected
    with pytest.raises(ValueError, match="end_time .* cannot be earlier than start_time"):
        FaultState(
            fault_type=FaultType.COOLING_DEGRADATION,
            severity=0.5,
            start_time=100.0,
            end_time=50.0,
        )


def test_timing_activity_and_inactive_fault():
    """Verify fault active window evaluation and inactive flag handling."""
    f = FaultState(
        fault_type=FaultType.FUEL_INJECTION_ABNORMALITY,
        severity=0.7,
        start_time=50.0,
        end_time=150.0,
    )
    # Before start
    assert f.is_active_at(49.9) is False
    assert f.get_effective_severity(49.9) == 0.0

    # During active window
    assert f.is_active_at(50.0) is True
    assert f.is_active_at(100.0) is True
    assert f.is_active_at(150.0) is True
    assert f.get_effective_severity(100.0) == 0.7

    # After end
    assert f.is_active_at(150.1) is False
    assert f.get_effective_severity(150.1) == 0.0

    # Explicitly deactivated fault
    f.active = False
    assert f.is_active_at(100.0) is False
    assert f.get_effective_severity(100.0) == 0.0


def test_zero_severity_produces_no_effect():
    """Verify zero severity fault reports inactive."""
    f = FaultState(
        fault_type=FaultType.COOLING_DEGRADATION,
        severity=0.0,
        start_time=10.0,
        end_time=100.0,
    )
    assert f.is_active_at(50.0) is False
    assert f.get_effective_severity(50.0) == 0.0


def test_dynamic_ramp_duration():
    """Verify effective severity ramping over time."""
    f = FaultState(
        fault_type=FaultType.MECHANICAL_DEGRADATION,
        severity=0.8,
        start_time=100.0,
        end_time=300.0,
        parameters={"ramp_duration": 40.0},
    )
    # At start
    assert pytest.approx(f.get_effective_severity(100.0), abs=1e-4) == 0.0
    # Halfway through ramp
    assert pytest.approx(f.get_effective_severity(120.0), abs=1e-4) == 0.4
    # Full ramp complete
    assert pytest.approx(f.get_effective_severity(140.0), abs=1e-4) == 0.8
    # After ramp complete, before end
    assert pytest.approx(f.get_effective_severity(200.0), abs=1e-4) == 0.8


def test_serialization_roundtrip():
    """Verify to_dict and from_dict roundtrip."""
    f = FaultState(
        fault_type=FaultType.SENSOR_FAULT,
        severity=0.55,
        active=True,
        start_time=20.0,
        end_time=80.0,
        affected_subsystem=FaultSubsystem.SENSOR,
        parameters={"target_channel": "oil_pressure", "bias_offset_bar": 1.2},
    )
    d = f.to_dict()
    reconstructed = FaultState.from_dict(d)

    assert reconstructed.fault_type == f.fault_type
    assert reconstructed.severity == f.severity
    assert reconstructed.affected_subsystem == f.affected_subsystem
    assert reconstructed.parameters == f.parameters


def test_fault_schedule_timeline_resolution():
    """Verify FaultSchedule multi-fault query and resolution."""
    f1 = FaultState(FaultType.COOLING_DEGRADATION, severity=0.4, start_time=10.0, end_time=50.0)
    f2 = FaultState(FaultType.LUBRICATION_DEGRADATION, severity=0.8, start_time=30.0, end_time=80.0)
    schedule = FaultSchedule([f1, f2])

    assert len(schedule) == 2

    # At t = 20s (only f1 active)
    active_20 = schedule.get_active_faults(20.0)
    assert len(active_20) == 1
    assert active_20[0].fault_type == FaultType.COOLING_DEGRADATION
    assert schedule.get_primary_fault(20.0).fault_type == FaultType.COOLING_DEGRADATION

    # At t = 40s (both f1 and f2 active; f2 has higher severity)
    active_40 = schedule.get_active_faults(40.0)
    assert len(active_40) == 2
    primary_40 = schedule.get_primary_fault(40.0)
    assert primary_40.fault_type == FaultType.LUBRICATION_DEGRADATION

    # At t = 60s (only f2 active)
    active_60 = schedule.get_active_faults(60.0)
    assert len(active_60) == 1
    assert active_60[0].fault_type == FaultType.LUBRICATION_DEGRADATION

    # At t = 90s (none active)
    assert len(schedule.get_active_faults(90.0)) == 0
    assert schedule.get_primary_fault(90.0) is None


def test_backward_compatibility_healthy_trajectory_unchanged():
    """
    Verify that providing no fault, fault_state=None, or an inactive fault
    produces identical physical engine telemetry.
    """
    sim_default = EngineSimulator(seed=42)
    sim_none = EngineSimulator(seed=42)
    sim_inactive = EngineSimulator(seed=42)

    seg = PhaseSegment(FlightPhase.CRUISE, 20.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    inactive_fault = FaultState(
        fault_type=FaultType.COOLING_DEGRADATION,
        severity=0.5,
        active=False,
    )

    recs_default = sim_default.run(profile, dt=0.2)
    recs_none = sim_none.run(profile, dt=0.2, fault_schedule=None)
    recs_inactive = sim_inactive.run(profile, dt=0.2, fault_schedule=inactive_fault)

    assert len(recs_default) == len(recs_none) == len(recs_inactive)

    for r_def, r_none, r_inact in zip(recs_default, recs_none, recs_inactive):
        # Physical channels must match identically
        assert r_def.rpm == r_none.rpm == r_inact.rpm
        assert r_def.cht == r_none.cht == r_inact.cht
        assert r_def.egt == r_none.egt == r_inact.egt
        assert r_def.oil_temp == r_none.oil_temp == r_inact.oil_temp
        assert r_def.oil_pressure == r_none.oil_pressure == r_inact.oil_pressure
        assert r_def.fuel_flow == r_none.fuel_flow == r_inact.fuel_flow
        assert r_def.vibration == r_none.vibration == r_inact.vibration
        # Fault tags must be nominal
        assert r_def.fault_type == "none"
        assert r_none.fault_type == "none"
        assert r_inact.fault_type == "none"
        assert r_def.fault_severity == 0.0


def test_active_fault_records_telemetry_tags():
    """
    Verify that when an active fault is scheduled, TelemetryRecord correctly
    reflects fault_type and fault_severity without altering Phase 4A physics.
    """
    sim = EngineSimulator(seed=42)
    seg = PhaseSegment(FlightPhase.CRUISE, 30.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    scheduled_fault = FaultState(
        fault_type=FaultType.COOLING_DEGRADATION,
        severity=0.75,
        start_time=10.0,
        end_time=25.0,
    )

    records = sim.run(profile, dt=0.5, fault_schedule=scheduled_fault)

    # Check timestamps before, during, and after fault
    recs_before = [r for r in records if r.timestamp < 10.0]
    recs_during = [r for r in records if 10.0 <= r.timestamp <= 25.0]
    recs_after = [r for r in records if r.timestamp > 25.0]

    for r in recs_before:
        assert r.fault_type == "none"
        assert r.fault_severity == 0.0

    assert len(recs_during) > 0
    for r in recs_during:
        assert r.fault_type == "cooling_degradation"
        assert r.fault_severity == 0.75

    for r in recs_after:
        assert r.fault_type == "none"
        assert r.fault_severity == 0.0
