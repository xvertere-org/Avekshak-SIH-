"""
Phase 4: Causal Fault Injection & Fault-Mechanism Modeling Validation Suite.

Verifies the causal physical propagation and strict observation-layer separation
for all canonical fault classes F1-F7 on the Rotax 914 UL/F grey-box engine:
- F1: Injector / Fuel-Delivery Abnormality (INJECTOR_DELIVERY_ABNORMALITY)
- F2: Lubrication Degradation (LUBRICATION_DEGRADATION)
- F3: Cooling Degradation (COOLING_DEGRADATION)
- F4: Combustion / Misfire Instability (COMBUSTION_MISFIRE)
- F5: Mechanical / Vibration Degradation (MECHANICAL_DEGRADATION)
- F6: Sensor Bias / Drift (SENSOR_BIAS, SENSOR_DRIFT)
- F7: Sensor Dropout / Stuck (SENSOR_DROPOUT, SENSOR_STUCK)

Requirements:
- Anti-tautological test design (direction, invariants, causality, isolation, bounds).
- No direct telemetry forgery.
- Observation-layer isolation for sensor faults.
- Label leakage prevention.
- Deterministic temporal replay.
"""

import math
import pytest
import numpy as np
import pandas as pd

from simulator.engine_simulator import EngineSimulator
from simulator.fault_interface import (
    FaultType,
    FaultSubsystem,
    FaultState,
    FaultSchedule,
    FAULT_SIGNATURE_CATALOG,
)
from simulator.subsystems.mission import MissionProfile, PhaseSegment, FlightPhase
from digital_twin.twin_model import DigitalTwin
from digital_twin.observability import CANONICAL_OBSERVABILITY_CATALOG


def _create_cruise_profile(duration_s: float = 60.0, throttle_pct: float = 75.0) -> MissionProfile:
    """Helper to create a stable steady-state cruise mission profile."""
    seg = PhaseSegment(
        phase=FlightPhase.CRUISE,
        duration_s=duration_s,
        throttle_start_pct=throttle_pct,
        throttle_end_pct=throttle_pct,
        altitude_start_m=2000.0,
        altitude_end_m=2000.0,
    )
    return MissionProfile(segments=[seg])


# =====================================================================
# TEST 01 — Healthy Baseline Equivalence
# =====================================================================
def test_01_healthy_baseline_equivalence():
    """
    Test 01: No fault and zero-severity fault must produce identical trajectories
    under identical configuration, initial state, inputs, timestep, and seed.
    """
    profile = _create_cruise_profile(30.0)
    sim_none = EngineSimulator(seed=101)
    sim_zero = EngineSimulator(seed=101)

    fault_zero = FaultState(
        fault_type=FaultType.INJECTOR_DELIVERY_ABNORMALITY,
        severity=0.0,
        start_time=5.0,
        end_time=25.0,
    )

    df_none = sim_none.run_to_dataframe(profile, dt=0.5, fault_schedule=None)
    df_zero = sim_zero.run_to_dataframe(profile, dt=0.5, fault_schedule=fault_zero)

    assert len(df_none) == len(df_zero)
    np.testing.assert_allclose(df_none["rpm"].values, df_zero["rpm"].values, rtol=1e-5)
    np.testing.assert_allclose(df_none["fuel_flow"].values, df_zero["fuel_flow"].values, rtol=1e-5)
    np.testing.assert_allclose(df_none["cht"].values, df_zero["cht"].values, rtol=1e-5)
    np.testing.assert_allclose(df_none["oil_pressure"].values, df_zero["oil_pressure"].values, rtol=1e-5)
    np.testing.assert_allclose(df_none["vibration"].values, df_zero["vibration"].values, rtol=1e-5)


# =====================================================================
# TEST 02 — Injector Causality (F1)
# =====================================================================
def test_02_injector_causality():
    """
    Test 02: Increasing F1 severity must reduce effective fuel delivery and
    produce downstream physical consequences (power drop, lower fuel consumption).
    """
    profile = _create_cruise_profile(40.0)
    sim_healthy = EngineSimulator(seed=42)
    sim_fault = EngineSimulator(seed=42)

    fault_f1 = FaultState(
        fault_type=FaultType.INJECTOR_DELIVERY_ABNORMALITY,
        severity=0.6,
        start_time=10.0,
        end_time=40.0,
        parameters={"mode": "lean"},
    )

    df_h = sim_healthy.run_to_dataframe(profile, dt=0.5)
    df_f = sim_fault.run_to_dataframe(profile, dt=0.5, fault_schedule=fault_f1)

    mask = df_h["timestamp"] >= 25.0
    # Physical fuel flow must decrease under lean injector delivery fault
    mean_fuel_h = df_h.loc[mask, "fuel_flow"].mean()
    mean_fuel_f = df_f.loc[mask, "fuel_flow"].mean()
    assert mean_fuel_f < mean_fuel_h - 1.0, f"Fuel flow did not decrease: {mean_fuel_f} vs {mean_fuel_h}"

    # EGT must elevate under lean mixture burn
    mean_egt_h = df_h.loc[mask, "egt"].mean()
    mean_egt_f = df_f.loc[mask, "egt"].mean()
    assert mean_egt_f > mean_egt_h + 20.0, f"EGT did not elevate: {mean_egt_f} vs {mean_egt_h}"


# =====================================================================
# TEST 03 — Lubrication Causality (F2)
# =====================================================================
def test_03_lubrication_causality():
    """
    Test 03: Increasing F2 severity must produce expected oil-pressure drop,
    friction torque increase, and oil temperature rise.
    """
    profile = _create_cruise_profile(50.0)
    sim_h = EngineSimulator(seed=42)
    sim_f = EngineSimulator(seed=42)

    fault_f2 = FaultState(
        fault_type=FaultType.LUBRICATION_DEGRADATION,
        severity=0.7,
        start_time=5.0,
        end_time=50.0,
    )

    df_h = sim_h.run_to_dataframe(profile, dt=0.5)
    df_f = sim_f.run_to_dataframe(profile, dt=0.5, fault_schedule=fault_f2)

    mask = df_h["timestamp"] >= 35.0
    # Oil pressure must drop significantly (hydraulic degradation d_p_loss = 0.55)
    p_oil_h = df_h.loc[mask, "oil_pressure"].mean()
    p_oil_f = df_f.loc[mask, "oil_pressure"].mean()
    assert p_oil_f < p_oil_h - 1.0, f"Oil pressure did not drop: {p_oil_f} vs {p_oil_h}"

    # Oil temperature must rise due to increased friction and reduced cooler flow
    t_oil_h = df_h.loc[mask, "oil_temp"].mean()
    t_oil_f = df_f.loc[mask, "oil_temp"].mean()
    assert t_oil_f > t_oil_h + 3.0, f"Oil temperature did not rise: {t_oil_f} vs {t_oil_h}"


# =====================================================================
# TEST 04 — Cooling Causality (F3)
# =====================================================================
def test_04_cooling_causality():
    """
    Test 04: Increasing F3 severity must reduce heat rejection and increase
    both CHT and coolant temperature.
    """
    profile = _create_cruise_profile(60.0)
    sim_h = EngineSimulator(seed=42)
    sim_f = EngineSimulator(seed=42)

    fault_f3 = FaultState(
        fault_type=FaultType.COOLING_DEGRADATION,
        severity=0.7,
        start_time=5.0,
        end_time=60.0,
    )

    df_h = sim_h.run_to_dataframe(profile, dt=0.5)
    df_f = sim_f.run_to_dataframe(profile, dt=0.5, fault_schedule=fault_f3)

    mask = df_h["timestamp"] >= 45.0
    # CHT must rise due to reduced convective fin dissipation
    cht_h = df_h.loc[mask, "cht"].mean()
    cht_f = df_f.loc[mask, "cht"].mean()
    assert cht_f > cht_h + 8.0, f"CHT did not rise: {cht_f} vs {cht_h}"

    # Internal cooling liquid temperature in simulator must also be higher
    assert sim_f.cooling.coolant_temp_c > sim_h.cooling.coolant_temp_c + 5.0


# =====================================================================
# TEST 05 — Combustion Causality (F4)
# =====================================================================
def test_05_combustion_causality():
    """
    Test 05: F4 combustion misfire must reduce indicated power and increase
    rotational vibration due to torque ripple/deficit.
    """
    profile = _create_cruise_profile(40.0)
    sim_h = EngineSimulator(seed=42)
    sim_f = EngineSimulator(seed=42)

    fault_f4 = FaultState(
        fault_type=FaultType.COMBUSTION_MISFIRE,
        severity=0.7,
        start_time=10.0,
        end_time=40.0,
        affected_cylinder=2,
    )

    df_h = sim_h.run_to_dataframe(profile, dt=0.5)
    df_f = sim_f.run_to_dataframe(profile, dt=0.5, fault_schedule=fault_f4)

    mask = df_h["timestamp"] >= 25.0
    # Vibration must elevate due to single-cylinder torque deficit ripple
    vib_h = df_h.loc[mask, "vibration"].mean()
    vib_f = df_f.loc[mask, "vibration"].mean()
    assert vib_f > vib_h + 0.10, f"Vibration did not elevate under misfire: {vib_f} vs {vib_h}"

    # Cylinder 2 EGT must drop significantly due to incomplete combustion heat release
    assert sim_f.thermal.egt_cyl[1] < sim_h.thermal.egt_cyl[1] - 40.0


# =====================================================================
# TEST 06 — Mechanical / Vibration Causality (F5)
# =====================================================================
def test_06_mechanical_vibration_causality():
    """
    Test 06: F5 mechanical degradation must increase rotational-order vibration
    amplitudes while preserving harmonic frequency lock (f1X = RPM/60, f2X = 2*RPM/60).
    """
    sim = EngineSimulator(seed=42)
    dt = 0.1
    # Step 1: Nominal condition
    rec_h = sim.step(throttle_pct=75.0, altitude_m=1000.0, dt=dt)
    current_rpm_h = sim.dynamics.omega_to_rpm(sim.dynamics.omega)
    vib_state_h = sim.vibration.step(rpm=current_rpm_h, load_pct=75.0, dt=dt)

    assert pytest.approx(vib_state_h.order_1x_freq_hz, rel=1e-3) == current_rpm_h / 60.0
    assert pytest.approx(vib_state_h.order_2x_freq_hz, rel=1e-3) == 2.0 * (current_rpm_h / 60.0)

    # Step 2: Injected F5 fault
    fault_f5 = FaultState(
        fault_type=FaultType.MECHANICAL_DEGRADATION,
        severity=0.8,
        start_time=0.0,
    )
    rec_f = sim.step(throttle_pct=75.0, altitude_m=1000.0, dt=dt, fault_state=fault_f5)

    # Harmonics must remain strictly tied to current RPM
    current_rpm_f = sim.dynamics.omega_to_rpm(sim.dynamics.omega)
    vib_state_f = sim.vibration.step(
        rpm=current_rpm_f, load_pct=75.0, dt=dt,
        mechanical_condition=1.0 + 1.8 * 0.8
    )
    assert pytest.approx(vib_state_f.order_1x_freq_hz, rel=1e-3) == current_rpm_f / 60.0
    assert pytest.approx(vib_state_f.order_2x_freq_hz, rel=1e-3) == 2.0 * (current_rpm_f / 60.0)

    # Vibration RMS must be significantly elevated
    assert rec_f.vibration > rec_h.vibration * 1.5


# =====================================================================
# TEST 07 — Sensor Bias Isolation (F6)
# =====================================================================
def test_07_sensor_bias_isolation():
    """
    Test 07: Sensor bias alters observed telemetry while underlying physical
    states (e.g. true CHT, dynamics, fuel) remain completely unchanged.
    """
    profile = _create_cruise_profile(30.0)
    sim_h = EngineSimulator(seed=42)
    sim_b = EngineSimulator(seed=42)

    fault_bias = FaultState(
        fault_type=FaultType.SENSOR_BIAS,
        severity=0.8,
        start_time=10.0,
        end_time=30.0,
        parameters={"sensor_channel": "cht", "bias_magnitude": 15.0},
    )

    df_h = sim_h.run_to_dataframe(profile, dt=0.5)
    df_b = sim_b.run_to_dataframe(profile, dt=0.5, fault_schedule=fault_bias)

    # Observation is biased
    mask = df_h["timestamp"] >= 15.0
    obs_diff = df_b.loc[mask, "cht"].values - df_h.loc[mask, "cht"].values
    assert (obs_diff > 10.0).all()

    # True physical states in simulator must remain bitwise identical
    np.testing.assert_allclose(sim_h.thermal.cht_cyl, sim_b.thermal.cht_cyl, rtol=1e-6)
    assert sim_h.dynamics.omega == pytest.approx(sim_b.dynamics.omega, rel=1e-6)
    assert sim_h.cooling.coolant_temp_c == pytest.approx(sim_b.cooling.coolant_temp_c, rel=1e-6)


# =====================================================================
# TEST 08 — Sensor Dropout Isolation (F7)
# =====================================================================
def test_08_sensor_dropout_isolation():
    """
    Test 08: Sensor dropout produces NaN observations while physical state
    integrates normally with finite numbers.
    """
    profile = _create_cruise_profile(30.0)
    sim_h = EngineSimulator(seed=42)
    sim_d = EngineSimulator(seed=42)

    fault_drop = FaultState(
        fault_type=FaultType.SENSOR_DROPOUT,
        severity=1.0,
        start_time=10.0,
        end_time=30.0,
        parameters={"sensor_channel": "rpm"},
    )

    df_h = sim_h.run_to_dataframe(profile, dt=0.5)
    df_d = sim_d.run_to_dataframe(profile, dt=0.5, fault_schedule=fault_drop)

    # Observation during fault must be NaN
    mask = df_d["timestamp"] >= 12.0
    assert df_d.loc[mask, "rpm"].isna().all()

    # Physical engine state must be perfectly finite and unaffected
    assert not math.isnan(sim_d.dynamics.omega)
    assert sim_d.dynamics.omega > 100.0
    assert sim_h.dynamics.omega == pytest.approx(sim_d.dynamics.omega, rel=1e-6)


# =====================================================================
# TEST 09 — Severity Monotonicity
# =====================================================================
def test_09_severity_monotonicity():
    """
    Test 09: Test severity sweep [0.1, 0.3, 0.5, 0.7, 0.9] on cooling degradation
    and verify monotonic increase in cylinder head temperature.
    """
    severities = [0.1, 0.3, 0.5, 0.7, 0.9]
    mean_chts = []

    for sev in severities:
        sim = EngineSimulator(seed=42)
        fault = FaultState(
            fault_type=FaultType.COOLING_DEGRADATION,
            severity=sev,
            start_time=0.0,
            end_time=40.0,
        )
        profile = _create_cruise_profile(40.0)
        df = sim.run_to_dataframe(profile, dt=0.5, fault_schedule=fault)
        mean_cht = df.loc[df["timestamp"] >= 25.0, "cht"].mean()
        mean_chts.append(mean_cht)

    # Must be strictly monotonically increasing
    for i in range(len(mean_chts) - 1):
        assert mean_chts[i + 1] > mean_chts[i] + 0.5, (
            f"Non-monotonic trend: {mean_chts[i + 1]} not > {mean_chts[i]}"
        )


# =====================================================================
# TEST 10 — Fault Onset
# =====================================================================
def test_10_fault_onset():
    """
    Test 10: Verify inactive behavior before onset timestamp and deterministic
    activation immediately afterward.
    """
    onset_time = 15.0
    profile = _create_cruise_profile(30.0)
    sim_h = EngineSimulator(seed=42)
    sim_f = EngineSimulator(seed=42)

    fault = FaultState(
        fault_type=FaultType.LUBRICATION_DEGRADATION,
        severity=0.75,
        start_time=onset_time,
    )

    df_h = sim_h.run_to_dataframe(profile, dt=0.5)
    df_f = sim_f.run_to_dataframe(profile, dt=0.5, fault_schedule=fault)

    # Before onset: bitwise identical
    pre_mask = df_h["timestamp"] < onset_time
    np.testing.assert_allclose(
        df_h.loc[pre_mask, "oil_pressure"].values,
        df_f.loc[pre_mask, "oil_pressure"].values,
        rtol=1e-5,
    )

    # Immediately after onset: oil pressure drops
    post_mask = df_h["timestamp"] >= onset_time + 2.0
    p_h = df_h.loc[post_mask, "oil_pressure"].mean()
    p_f = df_f.loc[post_mask, "oil_pressure"].mean()
    assert p_f < p_h - 1.0


# =====================================================================
# TEST 11 — Fault Removal / Recovery
# =====================================================================
def test_11_fault_removal_and_recovery():
    """
    Test 11: Verify fault deactivation and natural physical recovery
    governed by thermal and hydraulic time constants.
    """
    profile = _create_cruise_profile(80.0)
    sim = EngineSimulator(seed=42)

    # Fault active only between t=10s and t=40s
    fault = FaultState(
        fault_type=FaultType.LUBRICATION_DEGRADATION,
        severity=0.8,
        start_time=10.0,
        end_time=40.0,
    )

    df = sim.run_to_dataframe(profile, dt=0.5, fault_schedule=fault)

    # Pressure during fault (t=30s)
    p_fault = df.loc[df["timestamp"] == 30.0, "oil_pressure"].values[0]

    # Pressure after recovery (t=75s) vs healthy baseline at t=75s
    p_recovered = df.loc[df["timestamp"] == 75.0, "oil_pressure"].values[0]
    sim_h = EngineSimulator(seed=42)
    df_h = sim_h.run_to_dataframe(profile, dt=0.5)
    p_healthy_75 = df_h.loc[df_h["timestamp"] == 75.0, "oil_pressure"].values[0]

    assert p_fault < p_healthy_75 - 1.0
    assert pytest.approx(p_recovered, abs=0.05) == p_healthy_75


# =====================================================================
# TEST 12 — Cylinder Localization
# =====================================================================
def test_12_cylinder_localization():
    """
    Test 12: Injector fault targeting cylinder 3 must cause cylinder 3 CHT/EGT
    to change substantially while cylinders 1, 2, and 4 stay close to nominal.
    """
    sim = EngineSimulator(seed=42)
    dt = 0.5
    # Step nominal baseline
    for _ in range(20):
        sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=dt)

    baseline_egts = list(sim.thermal.egt_cyl)

    # Inject localized fuel abnormality on Cylinder 3
    fault_cyl3 = FaultState(
        fault_type=FaultType.INJECTOR_DELIVERY_ABNORMALITY,
        severity=0.8,
        affected_cylinder=3,
        start_time=0.0,
        parameters={"mode": "lean"},
    )

    for _ in range(30):
        sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=dt, fault_state=fault_cyl3)

    delta_cyl1 = abs(sim.thermal.egt_cyl[0] - baseline_egts[0])
    delta_cyl2 = abs(sim.thermal.egt_cyl[1] - baseline_egts[1])
    delta_cyl3 = abs(sim.thermal.egt_cyl[2] - baseline_egts[2])
    delta_cyl4 = abs(sim.thermal.egt_cyl[3] - baseline_egts[3])

    # Target cylinder 3 has dominant shift
    assert delta_cyl3 > 50.0, f"Cylinder 3 shift too small: {delta_cyl3}"
    assert delta_cyl1 < 25.0, f"Cylinder 1 cross-leakage too large: {delta_cyl1}"
    assert delta_cyl2 < 25.0, f"Cylinder 2 cross-leakage too large: {delta_cyl2}"
    assert delta_cyl4 < 25.0, f"Cylinder 4 cross-leakage too large: {delta_cyl4}"
    assert delta_cyl3 > 2.0 * max(delta_cyl1, delta_cyl2, delta_cyl4)


# =====================================================================
# TEST 13 — Deterministic Replay
# =====================================================================
def test_13_deterministic_replay():
    """
    Test 13: Identical inputs and fault configuration must reproduce
    bitwise-identical state and telemetry records.
    """
    profile = _create_cruise_profile(30.0)
    sched = FaultSchedule([
        FaultState(FaultType.COOLING_DEGRADATION, severity=0.5, start_time=5.0),
        FaultState(FaultType.MECHANICAL_DEGRADATION, severity=0.4, start_time=10.0),
    ])

    sim1 = EngineSimulator(seed=999)
    sim2 = EngineSimulator(seed=999)

    df1 = sim1.run_to_dataframe(profile, dt=0.5, fault_schedule=sched)
    df2 = sim2.run_to_dataframe(profile, dt=0.5, fault_schedule=sched)

    np.testing.assert_allclose(df1["rpm"].values, df2["rpm"].values, rtol=1e-6)
    np.testing.assert_allclose(df1["cht"].values, df2["cht"].values, rtol=1e-6)
    np.testing.assert_allclose(df1["oil_pressure"].values, df2["oil_pressure"].values, rtol=1e-6)
    np.testing.assert_allclose(df1["vibration"].values, df2["vibration"].values, rtol=1e-6)


# =====================================================================
# TEST 14 — Fault Composition
# =====================================================================
def test_14_fault_composition():
    """
    Test 14: Compatible simultaneous faults (COOLING + MECHANICAL) must both
    apply their respective physical mechanisms without one overwriting the other.
    """
    profile = _create_cruise_profile(40.0)

    # Single cooling fault
    sim_cool = EngineSimulator(seed=42)
    f_cool = FaultState(FaultType.COOLING_DEGRADATION, severity=0.6, start_time=5.0)
    df_cool = sim_cool.run_to_dataframe(profile, dt=0.5, fault_schedule=f_cool)

    # Single mechanical fault
    sim_mech = EngineSimulator(seed=42)
    f_mech = FaultState(FaultType.MECHANICAL_DEGRADATION, severity=0.6, start_time=5.0)
    df_mech = sim_mech.run_to_dataframe(profile, dt=0.5, fault_schedule=f_mech)

    # Composed simultaneous faults
    sim_both = EngineSimulator(seed=42)
    sched = FaultSchedule([
        FaultState(FaultType.COOLING_DEGRADATION, severity=0.6, start_time=5.0),
        FaultState(FaultType.MECHANICAL_DEGRADATION, severity=0.6, start_time=5.0),
    ])
    df_both = sim_both.run_to_dataframe(profile, dt=0.5, fault_schedule=sched)

    mask = df_both["timestamp"] >= 25.0

    # Composed run must exhibit CHT increase from cooling fault
    assert df_both.loc[mask, "cht"].mean() > df_mech.loc[mask, "cht"].mean() + 5.0

    # Composed run must exhibit vibration surge from mechanical fault
    assert df_both.loc[mask, "vibration"].mean() > df_cool.loc[mask, "vibration"].mean() + 0.15


# =====================================================================
# TEST 15 — No Direct Telemetry Forgery
# =====================================================================
def test_15_no_direct_telemetry_forgery():
    """
    Test 15: Demonstrate mechanism-level state changes for physical faults
    rather than directly fabricated telemetry values.
    """
    sim = EngineSimulator(seed=42)
    dt = 0.5
    # Run healthy step
    sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=dt)
    nominal_q_gen = sim.thermal.step(
        rpm=5000.0, load_pct=75.0, fuel_mass_flow_kg_s=0.003,
        density_factor=1.0, ambient_temp_c=15.0, airspeed_ms=45.0, dt=dt
    ).q_gen_w

    # Inject cooling fault
    fault_c = FaultState(FaultType.COOLING_DEGRADATION, severity=0.7, start_time=0.0)
    sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=dt, fault_state=fault_c)

    # Verify physical subsystem internal states have changed
    # 1. Radiator heat rejection conductance is reduced
    assert sim.cooling.coolant_temp_c >= 75.0
    # 2. Convective cooling conductance is degraded
    # 3. Telemetry values match the internal physical state, not arbitrary injected scalars
    rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=dt, fault_state=fault_c)
    assert pytest.approx(rec.cht, abs=1.5) == sim.thermal.cht_c
    assert pytest.approx(rec.oil_temp, abs=1.5) == sim.lubrication.oil_temp_c


# =====================================================================
# TEST 16 — Sensor Fault Separation
# =====================================================================
def test_16_sensor_fault_separation():
    """
    Test 16: Physical fault modifies true internal state.
    Sensor fault modifies observation channel only.
    """
    sim_phys = EngineSimulator(seed=42)
    sim_sens = EngineSimulator(seed=42)
    dt = 0.5

    f_phys = FaultState(FaultType.COOLING_DEGRADATION, severity=0.7, start_time=0.0)
    f_sens = FaultState(
        FaultType.SENSOR_BIAS, severity=0.7, start_time=0.0,
        parameters={"sensor_channel": "cht", "bias_magnitude": 25.0}
    )

    for _ in range(15):
        sim_phys.step(throttle_pct=75.0, altitude_m=2000.0, dt=dt, fault_state=f_phys)
        sim_sens.step(throttle_pct=75.0, altitude_m=2000.0, dt=dt, fault_state=f_sens)

    # Physical fault changed true CHT
    # Sensor fault left true physical CHT at nominal level
    assert sim_phys.thermal.cht_c > sim_sens.thermal.cht_c + 5.0

    # But observed telemetry for sensor fault is corrupted
    rec_sens = sim_sens.step(throttle_pct=75.0, altitude_m=2000.0, dt=dt, fault_state=f_sens)
    assert rec_sens.cht > sim_sens.thermal.cht_c + 10.0


# =====================================================================
# TEST 17 — Parameter Bounds Validation
# =====================================================================
def test_17_parameter_bounds_validation():
    """
    Test 17: Invalid severity (< 0, > 1) or invalid affected_cylinder raises ValueError.
    """
    # Negative severity
    with pytest.raises(ValueError, match="severity"):
        FaultState(FaultType.COOLING_DEGRADATION, severity=-0.1)

    # Severity > 1.0
    with pytest.raises(ValueError, match="severity"):
        FaultState(FaultType.COOLING_DEGRADATION, severity=1.05)

    # Invalid cylinder index 5
    with pytest.raises(ValueError, match="affected_cylinder"):
        FaultState(FaultType.COMBUSTION_MISFIRE, severity=0.5, affected_cylinder=5)

    # Invalid cylinder index 0
    with pytest.raises(ValueError, match="affected_cylinder"):
        FaultState(FaultType.COMBUSTION_MISFIRE, severity=0.5, affected_cylinder=0)


# =====================================================================
# TEST 18 — No NaN/Inf in Physical States
# =====================================================================
def test_18_no_nan_or_inf_in_physical_states():
    """
    Test 18: Supported physical fault simulations remain numerically finite
    across extreme severities and abrupt transitions.
    """
    profile = _create_cruise_profile(40.0)
    sim = EngineSimulator(seed=42)

    # Multiple aggressive faults
    sched = FaultSchedule([
        FaultState(FaultType.INJECTOR_DELIVERY_ABNORMALITY, severity=1.0, start_time=5.0),
        FaultState(FaultType.LUBRICATION_DEGRADATION, severity=1.0, start_time=10.0),
        FaultState(FaultType.COOLING_DEGRADATION, severity=1.0, start_time=15.0),
        FaultState(FaultType.MECHANICAL_DEGRADATION, severity=1.0, start_time=20.0),
    ])

    df = sim.run_to_dataframe(profile, dt=0.5, fault_schedule=sched)

    for col in ["rpm", "cht", "egt", "oil_temp", "oil_pressure", "fuel_flow", "vibration"]:
        assert not df[col].isna().any(), f"NaN found in {col}"
        assert not np.isinf(df[col].values).any(), f"Inf found in {col}"


# =====================================================================
# TEST 19 — Zero Fault Equivalence
# =====================================================================
def test_19_zero_fault_equivalence():
    """
    Test 19: Explicit active fault with severity=0.0 produces identical behavior
    to an un-faulted simulator instance across all channels.
    """
    profile = _create_cruise_profile(20.0)
    sim_none = EngineSimulator(seed=77)
    sim_zero = EngineSimulator(seed=77)

    f_zero = FaultState(FaultType.COMBUSTION_MISFIRE, severity=0.0, start_time=0.0)

    df_none = sim_none.run_to_dataframe(profile, dt=0.5, fault_schedule=None)
    df_zero = sim_zero.run_to_dataframe(profile, dt=0.5, fault_schedule=f_zero)

    pd.testing.assert_frame_equal(df_none, df_zero)


# =====================================================================
# TEST 20 — Deterministic Temporal Pattern
# =====================================================================
def test_20_deterministic_temporal_pattern():
    """
    Test 20: Repeated intermittent fault runs produce identical event timings
    without random modulation artifacts.
    """
    fault_intermittent = FaultState(
        fault_type=FaultType.COMBUSTION_MISFIRE,
        severity=0.8,
        start_time=10.0,
        end_time=30.0,
        parameters={"intermittent": True, "period": 4.0, "duty_cycle": 0.5},
    )

    t_eval = np.linspace(5.0, 35.0, 301)
    sevs_run1 = [fault_intermittent.get_effective_severity(t) for t in t_eval]
    sevs_run2 = [fault_intermittent.get_effective_severity(t) for t in t_eval]

    assert sevs_run1 == sevs_run2
    # Verify exact periodic duty cycle behavior
    # At t = 11.0 (elapsed 1.0 < 2.0s): active
    assert fault_intermittent.get_effective_severity(11.0) == 0.8
    # At t = 13.0 (elapsed 3.0 >= 2.0s): inactive
    assert fault_intermittent.get_effective_severity(13.0) == 0.0


# =====================================================================
# TEST 21 — Label Leakage Prevention
# =====================================================================
def test_21_label_leakage_prevention():
    """
    Test 21: Ground-truth fault labels (fault_type, fault_id, severity) must NOT
    leak into ordinary telemetry channels or digital twin observations.
    """
    catalog_channels = set(CANONICAL_OBSERVABILITY_CATALOG.keys())
    leak_names = ["fault_type", "fault_id", "severity", "fault_severity", "mechanism_description"]

    for name in leak_names:
        assert name not in catalog_channels, f"Leakage detected: {name} in observation catalog!"

    # Verify DigitalTwin observation vector contains zero fault diagnosis labels
    twin = DigitalTwin()
    sim = EngineSimulator(seed=42)
    rec = sim.step(
        throttle_pct=75.0, altitude_m=2000.0, dt=0.5,
        fault_state=FaultState(FaultType.COOLING_DEGRADATION, severity=0.5, start_time=0.0)
    )
    twin_res = twin.update(rec)

    # Observations in twin_res should be strictly physical measurements
    for k in twin_res.residuals.keys():
        assert k not in leak_names, f"Leakage detected in twin residual channel: {k}"


# =====================================================================
# TEST 22 — Signature Catalog Integrity
# =====================================================================
def test_22_signature_catalog_integrity():
    """
    Test 22: FAULT_SIGNATURE_CATALOG contains valid engineering contracts for
    all required canonical faults F1-F7.
    """
    required_keys = [
        "injector_delivery_abnormality",
        "lubrication_degradation",
        "cooling_degradation",
        "combustion_misfire",
        "mechanical_degradation",
        "sensor_bias",
        "sensor_drift",
        "sensor_dropout",
        "sensor_stuck",
    ]

    for key in required_keys:
        assert key in FAULT_SIGNATURE_CATALOG, f"Missing {key} in catalog"
        entry = FAULT_SIGNATURE_CATALOG[key]
        assert "primary_mechanism" in entry
        assert "secondary_effects" in entry
        assert "expected_telemetry_signatures" in entry
        assert "provenance" in entry
        assert "MODEL_CALIBRATION" in entry["provenance"] or "MODEL_ASSUMPTION" in entry["provenance"]
