"""
Unit and Validation Tests for Phase 4D: Fuel / Injection Abnormality Physics.

Covers:
- Test A: Healthy equivalence (no fault vs severity=0)
- Test B: Lean activation (fuel flow decreases, EGT increases)
- Test C: Rich activation (fuel flow increases, EGT decreases)
- Test D: Lean EGT direction (severity sweep strictly increases EGT)
- Test E: Rich EGT direction (severity sweep strictly decreases EGT)
- Test F: Fuel-flow direction (lean < healthy < rich)
- Test G: Severity monotonicity (monotonic trends in both modes)
- Test H: Schedule gating (active strictly inside scheduled window)
- Test I: Natural recovery (returns to nominal post-fault)
- Test J: Numerical stability (no NaNs, Infs, bounded states)
- Test K: Direct subsystem scaling on FuelSystem and ThermalSystem
"""

import numpy as np
import pytest
from simulator.engine_simulator import EngineSimulator
from simulator.fault_interface import FaultState, FaultType, FuelMixtureMode
from simulator.subsystems.mission import MissionProfile, PhaseSegment, FlightPhase
from simulator.subsystems.fuel import FuelSystem
from simulator.subsystems.thermal import ThermalSystem


def test_a_healthy_equivalence():
    """
    Test A: Verify that no fault vs fuel abnormality severity=0.0
    produces identical physical trajectories.
    """
    sim_none = EngineSimulator(seed=42)
    sim_zero = EngineSimulator(seed=42)

    seg = PhaseSegment(FlightPhase.CRUISE, 40.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    fault_zero = FaultState(
        fault_type=FaultType.FUEL_INJECTION_ABNORMALITY,
        severity=0.0,
        start_time=10.0,
        end_time=30.0,
        parameters={"mode": "lean"},
    )

    df_none = sim_none.run_to_dataframe(profile, dt=0.5, fault_schedule=None)
    df_zero = sim_zero.run_to_dataframe(profile, dt=0.5, fault_schedule=fault_zero)

    assert len(df_none) == len(df_zero)
    np.testing.assert_allclose(df_none["fuel_flow"].values, df_zero["fuel_flow"].values, rtol=1e-5)
    np.testing.assert_allclose(df_none["egt"].values, df_zero["egt"].values, rtol=1e-5)
    np.testing.assert_allclose(df_none["rpm"].values, df_zero["rpm"].values, rtol=1e-5)
    assert (df_zero["fault_type"] == "none").all()
    assert (df_zero["fault_severity"] == 0.0).all()


def test_b_lean_activation():
    """
    Test B: Verify that active lean abnormality reduces fuel flow and increases EGT.
    """
    sim_healthy = EngineSimulator(seed=42)
    sim_lean = EngineSimulator(seed=42)

    seg = PhaseSegment(FlightPhase.CRUISE, 60.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    fault = FaultState(
        fault_type=FaultType.FUEL_INJECTION_ABNORMALITY,
        severity=0.6,
        start_time=15.0,
        end_time=60.0,
        parameters={"mode": "lean"},
    )

    df_h = sim_healthy.run_to_dataframe(profile, dt=0.5)
    df_l = sim_lean.run_to_dataframe(profile, dt=0.5, fault_schedule=fault)

    # Before fault (t = 10s): identical
    f_h_pre = df_h.loc[df_h["timestamp"] == 10.0, "fuel_flow"].values[0]
    f_l_pre = df_l.loc[df_l["timestamp"] == 10.0, "fuel_flow"].values[0]
    assert f_h_pre == pytest.approx(f_l_pre, abs=1e-3)

    # During fault (t = 50s): lean fuel flow is lower, EGT is higher
    f_h = df_h.loc[df_h["timestamp"] == 50.0, "fuel_flow"].values[0]
    f_l = df_l.loc[df_l["timestamp"] == 50.0, "fuel_flow"].values[0]
    egt_h = df_h.loc[df_h["timestamp"] == 50.0, "egt"].values[0]
    egt_l = df_l.loc[df_l["timestamp"] == 50.0, "egt"].values[0]

    assert f_l < f_h - 1.5  # Noticeable fuel flow drop (L/h)
    assert egt_l > egt_h + 25.0  # Significant EGT elevation (°C)


def test_c_rich_activation():
    """
    Test C: Verify that active rich abnormality increases fuel flow and reduces EGT.
    """
    sim_healthy = EngineSimulator(seed=42)
    sim_rich = EngineSimulator(seed=42)

    seg = PhaseSegment(FlightPhase.CRUISE, 60.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    fault = FaultState(
        fault_type=FaultType.FUEL_INJECTION_ABNORMALITY,
        severity=0.6,
        start_time=15.0,
        end_time=60.0,
        parameters={"mode": FuelMixtureMode.RICH},
    )

    df_h = sim_healthy.run_to_dataframe(profile, dt=0.5)
    df_r = sim_rich.run_to_dataframe(profile, dt=0.5, fault_schedule=fault)

    f_h = df_h.loc[df_h["timestamp"] == 50.0, "fuel_flow"].values[0]
    f_r = df_r.loc[df_r["timestamp"] == 50.0, "fuel_flow"].values[0]
    egt_h = df_h.loc[df_h["timestamp"] == 50.0, "egt"].values[0]
    egt_r = df_r.loc[df_r["timestamp"] == 50.0, "egt"].values[0]

    assert f_r > f_h + 1.2  # Higher fuel flow
    assert egt_r < egt_h - 25.0  # Noticeable EGT drop due to fuel quench


def test_d_lean_egt_direction():
    """
    Test D: Verify that higher lean severity strictly produces higher steady-state EGT.
    """
    severities = [0.0, 0.25, 0.50, 0.75, 1.0]
    egts = []

    seg = PhaseSegment(FlightPhase.CRUISE, 60.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    for sev in severities:
        sim = EngineSimulator(seed=42)
        fault = FaultState(
            fault_type=FaultType.FUEL_INJECTION_ABNORMALITY,
            severity=sev,
            start_time=5.0,
            end_time=60.0,
            parameters={"mode": "lean"},
        )
        df = sim.run_to_dataframe(profile, dt=0.5, fault_schedule=fault)
        egts.append(float(df["egt"].iloc[-1]))

    # Strict monotonic increase in EGT
    d_egt = np.diff(egts)
    assert np.all(d_egt > 0.0), f"Lean EGT did not strictly increase: {egts}"


def test_e_rich_egt_direction():
    """
    Test E: Verify that higher rich severity strictly produces lower steady-state EGT.
    """
    severities = [0.0, 0.25, 0.50, 0.75, 1.0]
    egts = []

    seg = PhaseSegment(FlightPhase.CRUISE, 60.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    for sev in severities:
        sim = EngineSimulator(seed=42)
        fault = FaultState(
            fault_type=FaultType.FUEL_INJECTION_ABNORMALITY,
            severity=sev,
            start_time=5.0,
            end_time=60.0,
            parameters={"mode": "rich"},
        )
        df = sim.run_to_dataframe(profile, dt=0.5, fault_schedule=fault)
        egts.append(float(df["egt"].iloc[-1]))

    # Strict monotonic decrease in EGT
    d_egt = np.diff(egts)
    assert np.all(d_egt < 0.0), f"Rich EGT did not strictly decrease: {egts}"


def test_f_fuel_flow_direction():
    """
    Test F: Verify fuel flow hierarchy: Lean < Nominal < Rich.
    """
    seg = PhaseSegment(FlightPhase.CRUISE, 60.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    sim_h = EngineSimulator(seed=42)
    df_h = sim_h.run_to_dataframe(profile, dt=0.5)

    sim_l = EngineSimulator(seed=42)
    f_l = FaultState(FaultType.FUEL_INJECTION_ABNORMALITY, severity=0.5, start_time=5.0, end_time=60.0, parameters={"mode": "lean"})
    df_l = sim_l.run_to_dataframe(profile, dt=0.5, fault_schedule=f_l)

    sim_r = EngineSimulator(seed=42)
    f_r = FaultState(FaultType.FUEL_INJECTION_ABNORMALITY, severity=0.5, start_time=5.0, end_time=60.0, parameters={"mode": "rich"})
    df_r = sim_r.run_to_dataframe(profile, dt=0.5, fault_schedule=f_r)

    flow_h = df_h["fuel_flow"].iloc[-1]
    flow_l = df_l["fuel_flow"].iloc[-1]
    flow_r = df_r["fuel_flow"].iloc[-1]

    assert flow_l < flow_h < flow_r, f"Expected flow_l < flow_h < flow_r, got {flow_l:.2f}, {flow_h:.2f}, {flow_r:.2f}"


def test_g_severity_monotonicity():
    """
    Test G: Verify monotonic fuel flow response for both modes across severity sweep.
    """
    severities = [0.0, 0.25, 0.50, 0.75, 1.0]
    flows_lean = []
    flows_rich = []

    seg = PhaseSegment(FlightPhase.CRUISE, 60.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    for sev in severities:
        sim_l = EngineSimulator(seed=42)
        fl = FaultState(FaultType.FUEL_INJECTION_ABNORMALITY, severity=sev, start_time=5.0, end_time=60.0, parameters={"mode": "lean"})
        df_l = sim_l.run_to_dataframe(profile, dt=0.5, fault_schedule=fl)
        flows_lean.append(float(df_l["fuel_flow"].iloc[-1]))

        sim_r = EngineSimulator(seed=42)
        fr = FaultState(FaultType.FUEL_INJECTION_ABNORMALITY, severity=sev, start_time=5.0, end_time=60.0, parameters={"mode": "rich"})
        df_r = sim_r.run_to_dataframe(profile, dt=0.5, fault_schedule=fr)
        flows_rich.append(float(df_r["fuel_flow"].iloc[-1]))

    # Lean fuel flow strictly decreases
    assert np.all(np.diff(flows_lean) < 0.0), f"Lean flows not strictly decreasing: {flows_lean}"
    # Rich fuel flow strictly increases
    assert np.all(np.diff(flows_rich) > 0.0), f"Rich flows not strictly increasing: {flows_rich}"


def test_h_schedule_gating():
    """
    Test H: Verify that fault is active strictly inside scheduled window.
    """
    sim = EngineSimulator(seed=42)
    seg = PhaseSegment(FlightPhase.CRUISE, 100.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    fault = FaultState(
        fault_type=FaultType.FUEL_INJECTION_ABNORMALITY,
        severity=0.5,
        start_time=25.0,
        end_time=75.0,
        parameters={"mode": "lean"},
    )

    df = sim.run_to_dataframe(profile, dt=0.5, fault_schedule=fault)

    pre = df[df["timestamp"] < 25.0]
    assert (pre["fault_type"] == "none").all()
    assert (pre["fault_severity"] == 0.0).all()

    during = df[(df["timestamp"] >= 25.0) & (df["timestamp"] <= 75.0)]
    assert (during["fault_type"] == "fuel_injection_abnormality").all()
    assert (during["fault_severity"] == 0.5).all()

    post = df[df["timestamp"] > 75.0]
    assert (post["fault_type"] == "none").all()
    assert (post["fault_severity"] == 0.0).all()


def test_i_natural_recovery():
    """
    Test I: Verify that after fuel abnormality deactivates,
    fuel flow and EGT naturally recover toward nominal baseline.
    """
    sim_faulted = EngineSimulator(seed=42)
    sim_healthy = EngineSimulator(seed=42)

    seg = PhaseSegment(FlightPhase.CRUISE, 120.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    fault = FaultState(
        fault_type=FaultType.FUEL_INJECTION_ABNORMALITY,
        severity=0.6,
        start_time=15.0,
        end_time=50.0,
        parameters={"mode": "lean"},
    )

    df_f = sim_faulted.run_to_dataframe(profile, dt=0.5, fault_schedule=fault)
    df_h = sim_healthy.run_to_dataframe(profile, dt=0.5)

    # At peak of fault (t = 50s): EGT elevated
    egt_peak = df_f.loc[df_f["timestamp"] == 50.0, "egt"].values[0]
    # Well after fault deactivates (t = 110s): EGT recovered
    egt_rec = df_f.loc[df_f["timestamp"] == 110.0, "egt"].values[0]
    egt_nom = df_h.loc[df_h["timestamp"] == 110.0, "egt"].values[0]

    assert egt_rec < egt_peak - 20.0
    assert abs(egt_rec - egt_nom) < 2.0

    # Fuel flow also recovers immediately
    flow_rec = df_f.loc[df_f["timestamp"] == 110.0, "fuel_flow"].values[0]
    flow_nom = df_h.loc[df_h["timestamp"] == 110.0, "fuel_flow"].values[0]
    assert abs(flow_rec - flow_nom) < 0.1


def test_j_numerical_stability():
    """
    Test J: Verify that max lean and max rich produce no NaNs, Infs, or negative flows.
    """
    for mode in ["lean", "rich"]:
        sim = EngineSimulator(seed=42)
        seg = PhaseSegment(FlightPhase.CRUISE, 100.0, 100.0, 100.0, 2500.0, 2500.0)
        profile = MissionProfile(segments=[seg])

        fault = FaultState(
            fault_type=FaultType.FUEL_INJECTION_ABNORMALITY,
            severity=1.0,
            start_time=5.0,
            end_time=100.0,
            parameters={"mode": mode},
        )

        df = sim.run_to_dataframe(profile, dt=0.2, fault_schedule=fault)

        assert not df.isna().any().any()
        assert not np.isinf(df[["fuel_flow", "egt", "rpm", "cht"]].values).any()
        assert (df["fuel_flow"] > 0.0).all()
        assert (df["egt"] > 0.0).all()
        assert (df["rpm"] > 1000.0).all()


def test_k_subsystem_scaling():
    """
    Test K: Verify direct subsystem computations for fuel flow and thermal EGT mixture shifts.
    """
    fuel_sys = FuelSystem()
    state_nom = fuel_sys.compute(power_target_w=45000.0, fuel_severity=0.0)
    state_lean = fuel_sys.compute(power_target_w=45000.0, fuel_severity=1.0, mixture_mode="lean")
    state_rich = fuel_sys.compute(power_target_w=45000.0, fuel_severity=1.0, mixture_mode="rich")

    k_lean = fuel_sys.tier_c.k_fuel_flow_lean
    k_rich = fuel_sys.tier_c.k_fuel_flow_rich

    assert pytest.approx(state_lean.mass_flow_kg_s, rel=1e-3) == state_nom.mass_flow_kg_s * (1.0 - k_lean)
    assert pytest.approx(state_rich.mass_flow_kg_s, rel=1e-3) == state_nom.mass_flow_kg_s * (1.0 + k_rich)

    # Thermal subsystem direct verification
    therm_sys_nom = ThermalSystem()
    therm_sys_lean = ThermalSystem()
    therm_sys_rich = ThermalSystem()

    s_nom = therm_sys_nom.step(4500.0, 75.0, 0.0025, 0.85, 15.0, 45.0, 0.1, fuel_severity=0.0)
    s_lean = therm_sys_lean.step(4500.0, 75.0, 0.0025, 0.85, 15.0, 45.0, 0.1, fuel_severity=1.0, mixture_mode="lean")
    s_rich = therm_sys_rich.step(4500.0, 75.0, 0.0025, 0.85, 15.0, 45.0, 0.1, fuel_severity=1.0, mixture_mode="rich")

    assert s_lean.egt_steady_state_c > s_nom.egt_steady_state_c + 90.0
    assert s_rich.egt_steady_state_c < s_nom.egt_steady_state_c - 75.0
