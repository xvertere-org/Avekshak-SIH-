"""
Unit and Validation Tests for Phase 4B: Cooling Degradation Physics.

Covers:
- Test A: Healthy equivalence (no fault vs severity=0)
- Test B: Fault activation (severity > 0 elevates CHT)
- Test C: Severity monotonicity (severity sweep strictly increases CHT)
- Test D: Secondary oil temperature response (delayed conduction coupling)
- Test E: Fault scheduling (start_time / end_time windowing)
- Test F: Natural thermal recovery after fault deactivation
- Test G: Numerical stability (no NaNs, no Infs, valid bounds)
- Test H: Subsystem-level thermal conductance verification
"""

import numpy as np
import pytest
from simulator.engine_simulator import EngineSimulator
from simulator.fault_interface import FaultState, FaultType, FaultSchedule
from simulator.subsystems.mission import MissionProfile, PhaseSegment, FlightPhase
from simulator.subsystems.thermal import ThermalSystem


def test_a_healthy_equivalence():
    """
    Test A: Verify that a simulation with no fault and a simulation with
    cooling degradation severity=0.0 produce identical physical trajectories.
    """
    sim_none = EngineSimulator(seed=42)
    sim_zero = EngineSimulator(seed=42)

    seg = PhaseSegment(FlightPhase.CRUISE, 40.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    fault_zero = FaultState(
        fault_type=FaultType.COOLING_DEGRADATION,
        severity=0.0,
        start_time=10.0,
        end_time=30.0,
    )

    df_none = sim_none.run_to_dataframe(profile, dt=0.5, fault_schedule=None)
    df_zero = sim_zero.run_to_dataframe(profile, dt=0.5, fault_schedule=fault_zero)

    assert len(df_none) == len(df_zero)
    # Physical channels must match identically
    np.testing.assert_allclose(df_none["cht"].values, df_zero["cht"].values, rtol=1e-5)
    np.testing.assert_allclose(df_none["oil_temp"].values, df_zero["oil_temp"].values, rtol=1e-5)
    np.testing.assert_allclose(df_none["rpm"].values, df_zero["rpm"].values, rtol=1e-5)
    # Fault tags on severity=0 must be nominal
    assert (df_zero["fault_type"] == "none").all()
    assert (df_zero["fault_severity"] == 0.0).all()


def test_b_fault_activation():
    """
    Test B: Verify that an active cooling degradation fault with severity > 0
    measurably elevates CHT compared to healthy baseline.
    """
    sim_healthy = EngineSimulator(seed=42)
    sim_faulted = EngineSimulator(seed=42)

    seg = PhaseSegment(FlightPhase.CRUISE, 60.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    fault = FaultState(
        fault_type=FaultType.COOLING_DEGRADATION,
        severity=0.5,
        start_time=15.0,
        end_time=60.0,
    )

    df_h = sim_healthy.run_to_dataframe(profile, dt=0.5)
    df_f = sim_faulted.run_to_dataframe(profile, dt=0.5, fault_schedule=fault)

    # Before fault (t = 10s): identical
    cht_h_pre = df_h.loc[df_h["timestamp"] == 10.0, "cht"].values[0]
    cht_f_pre = df_f.loc[df_f["timestamp"] == 10.0, "cht"].values[0]
    assert cht_h_pre == pytest.approx(cht_f_pre, abs=1e-3)

    # During fault (t = 50s): faulted CHT is significantly higher
    cht_h_during = df_h.loc[df_h["timestamp"] == 50.0, "cht"].values[0]
    cht_f_during = df_f.loc[df_f["timestamp"] == 50.0, "cht"].values[0]
    assert cht_f_during > cht_h_during + 15.0  # At least 15°C elevation


def test_c_severity_monotonicity():
    """
    Test C: Verify that increasing cooling fault severity monotonically increases
    steady-state CHT under identical operating conditions.
    """
    severities = [0.0, 0.25, 0.50, 0.75, 1.0]
    final_chts = []

    seg = PhaseSegment(FlightPhase.CRUISE, 120.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    for sev in severities:
        sim = EngineSimulator(seed=42)
        fault = FaultState(
            fault_type=FaultType.COOLING_DEGRADATION,
            severity=sev,
            start_time=10.0,
            end_time=120.0,
        )
        df = sim.run_to_dataframe(profile, dt=0.5, fault_schedule=fault)
        final_chts.append(float(df["cht"].iloc[-1]))

    # Verify strict monotonic increase
    d_cht = np.diff(final_chts)
    assert np.all(d_cht > 0.0), f"CHT did not strictly increase: {final_chts}"


def test_d_secondary_oil_response():
    """
    Test D: Verify that cooling degradation produces a measurable directional
    increase in oil temperature via physical conduction coupling.
    """
    sim_healthy = EngineSimulator(seed=42)
    sim_faulted = EngineSimulator(seed=42)

    seg = PhaseSegment(FlightPhase.CRUISE, 120.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    fault = FaultState(
        fault_type=FaultType.COOLING_DEGRADATION,
        severity=0.75,
        start_time=10.0,
        end_time=120.0,
    )

    df_h = sim_healthy.run_to_dataframe(profile, dt=0.5)
    df_f = sim_faulted.run_to_dataframe(profile, dt=0.5, fault_schedule=fault)

    oil_h_final = float(df_h["oil_temp"].iloc[-1])
    oil_f_final = float(df_f["oil_temp"].iloc[-1])

    # Oil temp must be higher in the faulted case
    assert oil_f_final > oil_h_final + 2.0


def test_e_fault_schedule_gating():
    """
    Test E: Verify that cooling fault is active only within [start_time, end_time].
    """
    sim = EngineSimulator(seed=42)
    seg = PhaseSegment(FlightPhase.CRUISE, 100.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    fault = FaultState(
        fault_type=FaultType.COOLING_DEGRADATION,
        severity=0.7,
        start_time=30.0,
        end_time=70.0,
    )

    df = sim.run_to_dataframe(profile, dt=0.5, fault_schedule=fault)

    # Before start_time (t < 30)
    pre = df[df["timestamp"] < 30.0]
    assert (pre["fault_type"] == "none").all()
    assert (pre["fault_severity"] == 0.0).all()

    # During fault (30 <= t <= 70)
    during = df[(df["timestamp"] >= 30.0) & (df["timestamp"] <= 70.0)]
    assert (during["fault_type"] == "cooling_degradation").all()
    assert (during["fault_severity"] == 0.7).all()

    # After end_time (t > 70)
    post = df[df["timestamp"] > 70.0]
    assert (post["fault_type"] == "none").all()
    assert (post["fault_severity"] == 0.0).all()


def test_f_natural_thermal_recovery():
    """
    Test F: Verify that after cooling degradation fault ends, CHT naturally
    recovers toward the nominal baseline according to thermal dynamics.
    """
    sim_faulted = EngineSimulator(seed=42)
    sim_healthy = EngineSimulator(seed=42)

    seg = PhaseSegment(FlightPhase.CRUISE, 140.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    # Fault active from t=20s to t=60s
    fault = FaultState(
        fault_type=FaultType.COOLING_DEGRADATION,
        severity=0.6,
        start_time=20.0,
        end_time=60.0,
    )

    df_f = sim_faulted.run_to_dataframe(profile, dt=0.5, fault_schedule=fault)
    df_h = sim_healthy.run_to_dataframe(profile, dt=0.5)

    # Peak CHT during fault (at t=60s)
    cht_f_peak = df_f.loc[df_f["timestamp"] == 60.0, "cht"].values[0]
    # CHT well after recovery (at t=130s)
    cht_f_recovered = df_f.loc[df_f["timestamp"] == 130.0, "cht"].values[0]
    cht_h_nominal = df_h.loc[df_h["timestamp"] == 130.0, "cht"].values[0]

    # CHT must have cooled significantly after fault deactivation
    assert cht_f_recovered < cht_f_peak - 15.0
    # And recovered to within 2°C of nominal healthy steady-state
    assert abs(cht_f_recovered - cht_h_nominal) < 2.5


def test_g_numerical_stability():
    """
    Test G: Verify that maximum cooling degradation produces no NaNs, Infs,
    or negative impossible values across multi-minute simulation.
    """
    sim = EngineSimulator(seed=42)
    seg = PhaseSegment(FlightPhase.CRUISE, 180.0, 100.0, 100.0, 3000.0, 3000.0)
    profile = MissionProfile(segments=[seg])

    max_fault = FaultState(
        fault_type=FaultType.COOLING_DEGRADATION,
        severity=1.0,
        start_time=10.0,
        end_time=180.0,
    )

    df = sim.run_to_dataframe(profile, dt=0.2, fault_schedule=max_fault)

    assert not df.isna().any().any()
    assert not np.isinf(df[["cht", "oil_temp", "egt", "oil_pressure", "rpm"]].values).any()
    assert (df["cht"] > 0.0).all()
    assert (df["oil_temp"] > 0.0).all()


def test_h_subsystem_conductance_reduction():
    """
    Test H: Verify ThermalSystem.step directly scales h_cool by k_cooling_max_loss.
    """
    sys_nominal = ThermalSystem()
    sys_faulted = ThermalSystem()

    state_nom = sys_nominal.step(
        rpm=4500.0,
        load_pct=75.0,
        fuel_mass_flow_kg_s=0.0025,
        density_factor=1.0,
        ambient_temp_c=15.0,
        airspeed_ms=45.0,
        dt=0.1,
        cooling_severity=0.0,
    )

    state_fault = sys_faulted.step(
        rpm=4500.0,
        load_pct=75.0,
        fuel_mass_flow_kg_s=0.0025,
        density_factor=1.0,
        ambient_temp_c=15.0,
        airspeed_ms=45.0,
        dt=0.1,
        cooling_severity=1.0,
    )

    # h_cool should be scaled by exactly (1 - k_cooling_max_loss)
    k_loss = sys_nominal.tier_c.k_cooling_max_loss
    expected_ratio = 1.0 - k_loss
    actual_ratio = state_fault.h_cool_w_k / state_nom.h_cool_w_k

    assert pytest.approx(actual_ratio, abs=1e-3) == expected_ratio
