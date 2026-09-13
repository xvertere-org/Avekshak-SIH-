"""
Unit and Validation Tests for Phase 4C: Lubrication Degradation Physics.

Covers:
- Test A: Healthy equivalence (no fault vs severity=0)
- Test B: Fault activation (severity > 0 reduces oil pressure and increases oil temp)
- Test C: Oil pressure monotonicity (severity sweep strictly decreases oil pressure)
- Test D: Oil temperature response (severity sweep strictly increases steady-state oil temp)
- Test E: Fault scheduling (start_time / end_time windowing)
- Test F: Natural thermal and hydraulic recovery after fault deactivation
- Test G: Numerical stability (no NaNs, no Infs, bounded values)
- Test H: Subsystem-level hydraulic and thermal parameter verification
"""

import numpy as np
import pytest
from simulator.engine_simulator import EngineSimulator
from simulator.fault_interface import FaultState, FaultType
from simulator.subsystems.mission import MissionProfile, PhaseSegment, FlightPhase
from simulator.subsystems.lubrication import LubricationSystem


def test_a_healthy_equivalence():
    """
    Test A: Verify that a simulation with no fault and a simulation with
    lubrication degradation severity=0.0 produce identical physical trajectories.
    """
    sim_none = EngineSimulator(seed=42)
    sim_zero = EngineSimulator(seed=42)

    seg = PhaseSegment(FlightPhase.CRUISE, 40.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    fault_zero = FaultState(
        fault_type=FaultType.LUBRICATION_DEGRADATION,
        severity=0.0,
        start_time=10.0,
        end_time=30.0,
    )

    df_none = sim_none.run_to_dataframe(profile, dt=0.5, fault_schedule=None)
    df_zero = sim_zero.run_to_dataframe(profile, dt=0.5, fault_schedule=fault_zero)

    assert len(df_none) == len(df_zero)
    np.testing.assert_allclose(df_none["oil_pressure"].values, df_zero["oil_pressure"].values, rtol=1e-5)
    np.testing.assert_allclose(df_none["oil_temp"].values, df_zero["oil_temp"].values, rtol=1e-5)
    np.testing.assert_allclose(df_none["rpm"].values, df_zero["rpm"].values, rtol=1e-5)
    assert (df_zero["fault_type"] == "none").all()
    assert (df_zero["fault_severity"] == 0.0).all()


def test_b_fault_activation():
    """
    Test B: Verify that active lubrication degradation with severity > 0
    measurably reduces oil pressure and increases oil temperature.
    """
    sim_healthy = EngineSimulator(seed=42)
    sim_faulted = EngineSimulator(seed=42)

    seg = PhaseSegment(FlightPhase.CRUISE, 80.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    fault = FaultState(
        fault_type=FaultType.LUBRICATION_DEGRADATION,
        severity=0.6,
        start_time=15.0,
        end_time=80.0,
    )

    df_h = sim_healthy.run_to_dataframe(profile, dt=0.5)
    df_f = sim_faulted.run_to_dataframe(profile, dt=0.5, fault_schedule=fault)

    # Before fault (t = 10s): identical
    p_h_pre = df_h.loc[df_h["timestamp"] == 10.0, "oil_pressure"].values[0]
    p_f_pre = df_f.loc[df_f["timestamp"] == 10.0, "oil_pressure"].values[0]
    assert p_h_pre == pytest.approx(p_f_pre, abs=1e-3)

    # During fault (t = 70s): faulted oil pressure is lower, oil temp is higher
    p_h_during = df_h.loc[df_h["timestamp"] == 70.0, "oil_pressure"].values[0]
    p_f_during = df_f.loc[df_f["timestamp"] == 70.0, "oil_pressure"].values[0]
    oil_h_during = df_h.loc[df_h["timestamp"] == 70.0, "oil_temp"].values[0]
    oil_f_during = df_f.loc[df_f["timestamp"] == 70.0, "oil_temp"].values[0]

    assert p_f_during < p_h_during - 1.0  # Significant pressure drop
    assert oil_f_during > oil_h_during + 5.0  # Noticeable oil temperature rise


def test_c_oil_pressure_monotonicity():
    """
    Test C: Verify that increasing lubrication fault severity monotonically
    decreases steady-state oil pressure.
    """
    severities = [0.0, 0.25, 0.50, 0.75, 1.0]
    pressures = []

    seg = PhaseSegment(FlightPhase.CRUISE, 120.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    for sev in severities:
        sim = EngineSimulator(seed=42)
        fault = FaultState(
            fault_type=FaultType.LUBRICATION_DEGRADATION,
            severity=sev,
            start_time=10.0,
            end_time=120.0,
        )
        df = sim.run_to_dataframe(profile, dt=0.5, fault_schedule=fault)
        pressures.append(float(df["oil_pressure"].iloc[-1]))

    # Verify strict monotonic decrease
    dp = np.diff(pressures)
    assert np.all(dp < 0.0), f"Oil pressure did not strictly decrease: {pressures}"


def test_d_oil_temperature_monotonicity():
    """
    Test D: Verify that increasing lubrication fault severity monotonically
    increases steady-state oil temperature through increased friction and reduced cooler flow.
    """
    severities = [0.0, 0.25, 0.50, 0.75, 1.0]
    oil_temps = []

    seg = PhaseSegment(FlightPhase.CRUISE, 150.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    for sev in severities:
        sim = EngineSimulator(seed=42)
        fault = FaultState(
            fault_type=FaultType.LUBRICATION_DEGRADATION,
            severity=sev,
            start_time=10.0,
            end_time=150.0,
        )
        df = sim.run_to_dataframe(profile, dt=0.5, fault_schedule=fault)
        oil_temps.append(float(df["oil_temp"].iloc[-1]))

    # Verify strict monotonic increase
    dt_oil = np.diff(oil_temps)
    assert np.all(dt_oil > 0.0), f"Oil temp did not strictly increase: {oil_temps}"


def test_e_fault_schedule_gating():
    """
    Test E: Verify that lubrication fault is active only within [start_time, end_time].
    """
    sim = EngineSimulator(seed=42)
    seg = PhaseSegment(FlightPhase.CRUISE, 100.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    fault = FaultState(
        fault_type=FaultType.LUBRICATION_DEGRADATION,
        severity=0.7,
        start_time=25.0,
        end_time=65.0,
    )

    df = sim.run_to_dataframe(profile, dt=0.5, fault_schedule=fault)

    # Before start_time (t < 25)
    pre = df[df["timestamp"] < 25.0]
    assert (pre["fault_type"] == "none").all()
    assert (pre["fault_severity"] == 0.0).all()

    # During fault (25 <= t <= 65)
    during = df[(df["timestamp"] >= 25.0) & (df["timestamp"] <= 65.0)]
    assert (during["fault_type"] == "lubrication_degradation").all()
    assert (during["fault_severity"] == 0.7).all()

    # After end_time (t > 65)
    post = df[df["timestamp"] > 65.0]
    assert (post["fault_type"] == "none").all()
    assert (post["fault_severity"] == 0.0).all()


def test_f_natural_recovery():
    """
    Test F: Verify that after lubrication degradation fault deactivates,
    oil pressure and oil temperature naturally recover toward nominal baseline.
    """
    sim_faulted = EngineSimulator(seed=42)
    sim_healthy = EngineSimulator(seed=42)

    seg = PhaseSegment(FlightPhase.CRUISE, 160.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    # Fault active from t=20s to t=60s
    fault = FaultState(
        fault_type=FaultType.LUBRICATION_DEGRADATION,
        severity=0.65,
        start_time=20.0,
        end_time=60.0,
    )

    df_f = sim_faulted.run_to_dataframe(profile, dt=0.5, fault_schedule=fault)
    df_h = sim_healthy.run_to_dataframe(profile, dt=0.5)

    # Pressure at peak of fault (t = 60s)
    p_fault_t60 = df_f.loc[df_f["timestamp"] == 60.0, "oil_pressure"].values[0]
    # Pressure well after deactivation (t = 150s)
    p_rec_t150 = df_f.loc[df_f["timestamp"] == 150.0, "oil_pressure"].values[0]
    p_nom_t150 = df_h.loc[df_h["timestamp"] == 150.0, "oil_pressure"].values[0]

    # Pressure must recover significantly from degraded trough
    assert p_rec_t150 > p_fault_t60 + 1.0
    # Pressure must recover to within 0.15 bar of nominal
    assert abs(p_rec_t150 - p_nom_t150) < 0.15

    # Oil temperature excess above nominal baseline recovers toward zero
    t_oil_peak = df_f.loc[df_f["timestamp"] == 60.0, "oil_temp"].values[0]
    t_oil_nom_60 = df_h.loc[df_h["timestamp"] == 60.0, "oil_temp"].values[0]
    t_oil_rec = df_f.loc[df_f["timestamp"] == 150.0, "oil_temp"].values[0]
    t_oil_nom = df_h.loc[df_h["timestamp"] == 150.0, "oil_temp"].values[0]

    assert (t_oil_rec - t_oil_nom) < (t_oil_peak - t_oil_nom_60)
    assert abs(t_oil_rec - t_oil_nom) < 3.0


def test_g_numerical_stability():
    """
    Test G: Verify that maximum lubrication degradation produces no NaNs, Infs,
    or negative impossible values across multi-minute simulation.
    """
    sim = EngineSimulator(seed=42)
    seg = PhaseSegment(FlightPhase.CRUISE, 180.0, 100.0, 100.0, 3000.0, 3000.0)
    profile = MissionProfile(segments=[seg])

    max_fault = FaultState(
        fault_type=FaultType.LUBRICATION_DEGRADATION,
        severity=1.0,
        start_time=10.0,
        end_time=180.0,
    )

    df = sim.run_to_dataframe(profile, dt=0.2, fault_schedule=max_fault)

    assert not df.isna().any().any()
    assert not np.isinf(df[["oil_pressure", "oil_temp", "cht", "rpm"]].values).any()
    assert (df["oil_pressure"] >= 0.5).all()
    assert (df["oil_temp"] > 0.0).all()


def test_h_subsystem_hydraulic_scaling():
    """
    Test H: Verify LubricationSystem.step directly scales nominal pressure
    by (1 - k_lub_p_loss * severity).
    """
    sys_nom = LubricationSystem()
    sys_fault = LubricationSystem()

    # Step at fixed temp to isolate hydraulic gain
    state_nom = sys_nom.step(
        rpm=4500.0,
        cht_c=90.0,
        fuel_mass_flow_kg_s=0.0025,
        ambient_temp_c=15.0,
        dt=0.1,
        lubrication_severity=0.0,
    )
    state_fault = sys_fault.step(
        rpm=4500.0,
        cht_c=90.0,
        fuel_mass_flow_kg_s=0.0025,
        ambient_temp_c=15.0,
        dt=0.1,
        lubrication_severity=1.0,
    )

    k_p_loss = sys_nom.tier_c.k_lub_p_loss
    expected_ratio = 1.0 - k_p_loss
    actual_ratio = state_fault.oil_pressure_bar / state_nom.oil_pressure_bar

    assert pytest.approx(actual_ratio, abs=1e-3) == expected_ratio
