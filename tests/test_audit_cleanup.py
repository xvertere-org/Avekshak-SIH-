"""
Pre-Phase 4E Audit Cleanup Tests.

Covers all five audit findings:
- FIX #1: RNG / Reproducibility contract
- FIX #2: MissionConfig fault contract truthfulness
- FIX #3: CHT reference limit consistency
- FIX #4: Telemetry sampling & provenance contract
- FIX #5: Calibration inventory consistency
"""

import json
import math
import numpy as np
import pytest
from pathlib import Path

from simulator.engine_simulator import EngineSimulator
from simulator.config import SimulatorConfig, TierAParameters, TierCParameters
from simulator.subsystems.mission import MissionProfile, PhaseSegment, FlightPhase
from simulator.fault_interface import FaultState, FaultSchedule, FaultType
from telemetry.schema import MissionConfig, TelemetryRecord, FaultCategory


# ---------------------------------------------------------------------------
# FIX #1: RNG / Reproducibility
# ---------------------------------------------------------------------------

def test_fix1_independent_instance_reproducibility():
    """
    FIX #1: Two independent EngineSimulator instances with the same seed
    must produce bitwise identical complete telemetry (all channels).
    """
    profile = MissionProfile(segments=[
        PhaseSegment(FlightPhase.CRUISE, 15.0, 75.0, 75.0, 2000.0, 2000.0),
    ])

    sim1 = EngineSimulator(seed=42)
    sim2 = EngineSimulator(seed=42)

    recs1 = sim1.run(profile, dt=0.5)
    recs2 = sim2.run(profile, dt=0.5)

    assert len(recs1) == len(recs2)

    for r1, r2 in zip(recs1, recs2):
        assert r1.timestamp == r2.timestamp
        assert r1.rpm == r2.rpm
        assert r1.cht == r2.cht
        assert r1.egt == r2.egt
        assert r1.oil_temp == r2.oil_temp
        assert r1.oil_pressure == r2.oil_pressure
        assert r1.fuel_flow == r2.fuel_flow
        assert r1.vibration == r2.vibration
        assert r1.fault_type == r2.fault_type
        assert r1.fault_severity == r2.fault_severity


def test_fix1_different_seed_produces_different_noise():
    """
    FIX #1: Different seeds produce different sensor noise samples.
    """
    profile = MissionProfile(segments=[
        PhaseSegment(FlightPhase.CRUISE, 10.0, 75.0, 75.0, 2000.0, 2000.0),
    ])

    sim_a = EngineSimulator(seed=42)
    sim_b = EngineSimulator(seed=999)

    recs_a = sim_a.run(profile, dt=0.5)
    recs_b = sim_b.run(profile, dt=0.5)

    # At least some vibration samples must differ due to different RNG seeds
    different = any(r1.vibration != r2.vibration for r1, r2 in zip(recs_a, recs_b))
    assert different, "Same vibration with different seeds — sensor noise not seeded correctly"


def test_fix1_streaming_determinism():
    """
    FIX #1: Sequential streaming steps on the same simulator instance
    consume RNG state sequentially and produce deterministic results
    when compared to a fresh instance doing the same steps.
    """
    from simulator.subsystems.mission import MissionStep

    steps = []
    for i in range(20):
        t = (i + 1) * 0.5
        steps.append(MissionStep(
            timestamp_s=t,
            phase=FlightPhase.CRUISE,
            throttle_pct=75.0,
            altitude_m=2000.0,
            airspeed_ms=45.0,
            temp_offset_k=0.0,
            progress_pct=t / 10.0,
        ))

    sim1 = EngineSimulator(seed=77)
    sim1.reset()
    recs1 = [sim1.step(mission_config=s, time_step=0.5) for s in steps]

    sim2 = EngineSimulator(seed=77)
    sim2.reset()
    recs2 = [sim2.step(mission_config=s, time_step=0.5) for s in steps]

    for r1, r2 in zip(recs1, recs2):
        assert r1.rpm == r2.rpm
        assert r1.vibration == r2.vibration
        assert r1.egt == r2.egt
        assert r1.fuel_flow == r2.fuel_flow


# ---------------------------------------------------------------------------
# FIX #2: MissionConfig Fault Contract Truthfulness
# ---------------------------------------------------------------------------

def test_fix2_mission_config_fault_label_not_propagated():
    """
    FIX #2: When MissionConfig specifies a fault but no FaultState is provided,
    the telemetry must say fault_type='none' and fault_severity=0.0.
    Physics must remain healthy.
    """
    faulty_mission = MissionConfig(
        mission_id="TEST_FAULT_LABEL",
        fault_type=FaultCategory.COOLING_DEGRADATION.value,
        fault_severity=0.8,
        throttle=75.0,
        altitude=2000.0,
    )

    sim = EngineSimulator(seed=42)
    rec = sim.step(mission_config=faulty_mission, time_step=1.0)

    # Telemetry must NOT claim a fault that wasn't physically activated
    assert rec.fault_type == "none", (
        f"Telemetry falsely claims fault_type='{rec.fault_type}' when no FaultState was provided"
    )
    assert rec.fault_severity == 0.0, (
        f"Telemetry falsely claims fault_severity={rec.fault_severity} when no FaultState was provided"
    )


def test_fix2_healthy_mission_config_produces_healthy_label():
    """
    FIX #2: Healthy MissionConfig → healthy physics → healthy telemetry labels.
    """
    healthy_mission = MissionConfig(
        mission_id="TEST_HEALTHY",
        fault_type=FaultCategory.NONE.value,
        fault_severity=0.0,
        throttle=75.0,
        altitude=2000.0,
    )

    sim = EngineSimulator(seed=42)
    rec = sim.step(mission_config=healthy_mission, time_step=1.0)

    assert rec.fault_type == "none"
    assert rec.fault_severity == 0.0


def test_fix2_real_fault_state_still_labels_correctly():
    """
    FIX #2: When a real FaultState IS provided, the telemetry fault label
    must correctly reflect the active physical fault. Verify for all 4B-4D types.
    """
    profile = MissionProfile(segments=[
        PhaseSegment(FlightPhase.CRUISE, 30.0, 75.0, 75.0, 2000.0, 2000.0),
    ])

    fault_types_to_test = [
        (FaultType.COOLING_DEGRADATION, "cooling_degradation"),
        (FaultType.LUBRICATION_DEGRADATION, "lubrication_degradation"),
        (FaultType.FUEL_INJECTION_ABNORMALITY, "fuel_injection_abnormality"),
    ]

    for ftype, expected_label in fault_types_to_test:
        fault = FaultState(
            fault_type=ftype,
            severity=0.5,
            start_time=5.0,
            end_time=25.0,
            parameters={"mode": "lean"} if ftype == FaultType.FUEL_INJECTION_ABNORMALITY else {},
        )

        sim = EngineSimulator(seed=42)
        df = sim.run_to_dataframe(profile, dt=0.5, fault_schedule=fault)

        # During fault window: telemetry must claim the active fault
        during = df[(df["timestamp"] >= 5.0) & (df["timestamp"] <= 25.0)]
        assert (during["fault_type"] == expected_label).all(), (
            f"Expected fault_type='{expected_label}' during active {ftype}, "
            f"got: {during['fault_type'].unique()}"
        )
        assert (during["fault_severity"] == 0.5).all()

        # Before fault window: telemetry must say none
        before = df[df["timestamp"] < 5.0]
        assert (before["fault_type"] == "none").all()
        assert (before["fault_severity"] == 0.0).all()


def test_fix2_fault_schedule_multi_fault_labels():
    """
    FIX #2: FaultSchedule with multiple faults produces correct labels for each window.
    """
    profile = MissionProfile(segments=[
        PhaseSegment(FlightPhase.CRUISE, 80.0, 75.0, 75.0, 2000.0, 2000.0),
    ])

    f1 = FaultState(FaultType.COOLING_DEGRADATION, severity=0.4, start_time=10.0, end_time=30.0)
    f2 = FaultState(FaultType.LUBRICATION_DEGRADATION, severity=0.6, start_time=50.0, end_time=70.0)
    sched = FaultSchedule(faults=[f1, f2])

    sim = EngineSimulator(seed=42)
    df = sim.run_to_dataframe(profile, dt=0.5, fault_schedule=sched)

    # During cooling window
    cool_window = df[(df["timestamp"] >= 10.0) & (df["timestamp"] <= 30.0)]
    assert (cool_window["fault_type"] == "cooling_degradation").all()

    # Between faults
    gap = df[(df["timestamp"] > 30.0) & (df["timestamp"] < 50.0)]
    assert (gap["fault_type"] == "none").all()

    # During lubrication window
    lub_window = df[(df["timestamp"] >= 50.0) & (df["timestamp"] <= 70.0)]
    assert (lub_window["fault_type"] == "lubrication_degradation").all()


# ---------------------------------------------------------------------------
# FIX #3: CHT Reference Limit Consistency
# ---------------------------------------------------------------------------

def test_fix3_cht_nominal_less_than_limit():
    """
    FIX #3: Tier A cht_nominal_c must be strictly less than cht_limit_c.
    """
    tier_a = TierAParameters()
    assert tier_a.cht_nominal_c < tier_a.cht_limit_c, (
        f"cht_nominal_c ({tier_a.cht_nominal_c}) must be < cht_limit_c ({tier_a.cht_limit_c})"
    )


def test_fix3_digital_twin_nominal_matches_tier_a():
    """
    FIX #3: Digital Twin nominal CHT must use the Tier A cht_nominal_c value (100°C),
    not exceed the maximum reference limit (150°C).
    """
    from digital_twin.twin_model import DigitalTwin

    twin = DigitalTwin()
    # Create a dummy telemetry record
    rec = TelemetryRecord(
        timestamp=1.0, mission_id="TEST", engine_id="TEST",
        mission_phase="CRUISE", altitude=2000.0, ambient_temp=15.0,
        throttle=75.0, load=70.0, rpm=4300.0, cht=100.0, egt=690.0,
        oil_temp=85.0, oil_pressure=4.2, fuel_flow=12.0, vibration=0.5,
    )
    state = twin.update(rec)

    tier_a = TierAParameters()
    # The nominal CHT used for residual must equal cht_nominal_c
    assert state.nominal_estimates["nominal_cht"] == tier_a.cht_nominal_c
    # And it must be below the maximum limit
    assert state.nominal_estimates["nominal_cht"] < tier_a.cht_limit_c


def test_fix3_default_engine_config_cht_consistent():
    """
    FIX #3: default_engine.json baseline_cht_target_c must not exceed cht_limit_c.
    """
    config_path = Path("configs/default_engine.json")
    with open(config_path) as f:
        cfg = json.load(f)

    baseline_cht = cfg["parameters"]["baseline_cht_target_c"]
    tier_a = TierAParameters()

    assert baseline_cht <= tier_a.cht_limit_c, (
        f"default_engine.json baseline_cht_target_c ({baseline_cht}) exceeds cht_limit_c ({tier_a.cht_limit_c})"
    )
    assert baseline_cht == tier_a.cht_nominal_c, (
        f"default_engine.json baseline_cht_target_c ({baseline_cht}) should match cht_nominal_c ({tier_a.cht_nominal_c})"
    )


# ---------------------------------------------------------------------------
# FIX #4: Telemetry Sampling & Provenance
# ---------------------------------------------------------------------------

def test_fix4_timestamp_monotonicity():
    """
    FIX #4: Generated telemetry timestamps must be strictly monotonically increasing.
    """
    profile = MissionProfile(segments=[
        PhaseSegment(FlightPhase.CRUISE, 10.0, 75.0, 75.0, 2000.0, 2000.0),
    ])

    sim = EngineSimulator(seed=42)
    df = sim.run_to_dataframe(profile, dt=0.1)

    timestamps = df["timestamp"].values
    diffs = np.diff(timestamps)
    assert np.all(diffs > 0), "Timestamps are not strictly monotonically increasing"


def test_fix4_expected_sample_interval():
    """
    FIX #4: At dt=0.1s, the sample interval between consecutive records must be 0.1s.
    """
    profile = MissionProfile(segments=[
        PhaseSegment(FlightPhase.CRUISE, 5.0, 75.0, 75.0, 2000.0, 2000.0),
    ])

    sim = EngineSimulator(seed=42)
    df = sim.run_to_dataframe(profile, dt=0.1)

    timestamps = df["timestamp"].values
    diffs = np.diff(timestamps)
    np.testing.assert_allclose(diffs, 0.1, atol=1e-6)


def test_fix4_provenance_version_matches():
    """
    FIX #4: TelemetryRecord.simulation_version must match SimulatorConfig.provenance_version.
    """
    profile = MissionProfile(segments=[
        PhaseSegment(FlightPhase.CRUISE, 5.0, 75.0, 75.0, 2000.0, 2000.0),
    ])

    sim = EngineSimulator(seed=42)
    recs = sim.run(profile, dt=0.5)

    expected_version = sim.sim_config.provenance_version
    for rec in recs:
        assert rec.simulation_version == expected_version


def test_fix4_telemetry_settings_match_actual():
    """
    FIX #4: configs/telemetry_settings.json metadata must match actual behavior.
    """
    settings_path = Path("configs/telemetry_settings.json")
    with open(settings_path) as f:
        settings = json.load(f)

    cfg = SimulatorConfig()

    # Sampling rate matches 1/dt
    expected_rate = 1.0 / cfg.default_dt
    assert settings["sampling_rate_hz"] == expected_rate, (
        f"telemetry_settings.json sampling_rate_hz ({settings['sampling_rate_hz']}) "
        f"does not match 1/default_dt ({expected_rate})"
    )

    # Provenance version matches
    assert settings["provenance"]["simulation_version"] == cfg.provenance_version


def test_fix4_source_type_is_synthetic():
    """
    FIX #4: Generated telemetry must truthfully declare source_type as 'synthetic'.
    """
    profile = MissionProfile(segments=[
        PhaseSegment(FlightPhase.CRUISE, 5.0, 75.0, 75.0, 2000.0, 2000.0),
    ])

    sim = EngineSimulator(seed=42)
    recs = sim.run(profile, dt=0.5)

    for rec in recs:
        assert rec.source_type == "synthetic"
        assert rec.source == "simulator_v1_physics"


# ---------------------------------------------------------------------------
# FIX #5: Calibration Inventory Consistency
# ---------------------------------------------------------------------------

def test_fix5_inventory_contains_phase4_params():
    """
    FIX #5: The calibration inventory must include all Phase 4B-4D fault parameters.
    """
    from validation.calibration import get_parameter_inventory

    inventory = get_parameter_inventory()
    names = {r.name for r in inventory}

    phase4_params = [
        "k_cooling_max_loss",
        "k_lub_p_loss",
        "k_lub_heat_gain",
        "k_lub_cool_loss",
        "k_fuel_flow_lean",
        "k_fuel_flow_rich",
        "k_comb_loss_lean",
        "k_comb_loss_rich",
        "k_egt_lean_gain_c",
        "k_egt_rich_drop_c",
    ]

    for param in phase4_params:
        assert param in names, f"Phase 4 parameter '{param}' missing from calibration inventory"


def test_fix5_inventory_values_match_config():
    """
    FIX #5: Calibration inventory values must match actual TierCParameters defaults.
    """
    from validation.calibration import get_parameter_inventory

    inventory = get_parameter_inventory()
    inv_map = {r.name: r for r in inventory}
    tier_c = TierCParameters()

    spot_checks = {
        "k_cooling_max_loss": tier_c.k_cooling_max_loss,
        "k_lub_p_loss": tier_c.k_lub_p_loss,
        "k_fuel_flow_lean": tier_c.k_fuel_flow_lean,
        "k_fuel_flow_rich": tier_c.k_fuel_flow_rich,
        "k_egt_lean_gain_c": tier_c.k_egt_lean_gain_c,
        "k_egt_rich_drop_c": tier_c.k_egt_rich_drop_c,
    }

    for name, expected_val in spot_checks.items():
        assert name in inv_map, f"Parameter '{name}' not in inventory"
        assert float(inv_map[name].current_value) == expected_val, (
            f"Inventory value for '{name}' ({inv_map[name].current_value}) "
            f"does not match TierCParameters ({expected_val})"
        )
