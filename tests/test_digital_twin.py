"""
Comprehensive Unit and Validation Tests for Phase 6: Digital Twin + Residual Generation.

Covers:
- Test A: Healthy behavior — bounded residuals on nominal telemetry
- Test B: Operating-point sensitivity — throttle, altitude, ambient temp variations
- Test C: Cooling fault — positive CHT residual under cooling degradation
- Test D: Lubrication fault — negative oil pressure residual, positive oil temp residual
- Test E: Fuel abnormality — lean/rich EGT & fuel flow directional residual response
- Test F: Mechanical fault — vibration residual increase under mechanical degradation
- Test G: Sensor bias isolation — residual reflects offset, expected state uncorrupted
- Test H: Sensor dropout isolation — NaN observed yields NaN residual, never zeroed
- Test I: Sensor drift isolation — progressive residual divergence with stable expected state
- Test J: Regression — backward compatibility with Phase 1 DigitalTwin.update()
- Test K: Determinism — identical inputs produce identical expectations and residuals
- Test L: Causality — future telemetry modification does not alter prior residuals
- Test M: Provenance preservation — engine_id, mission_id, source survive in ResidualFrame
- Test N: Multi-mission separation — resetting twin isolates sequential missions
- Test O: Numerical stability — zero unexpected NaNs or Infs across flight profiles
"""

import math
import numpy as np
import pandas as pd
import pytest

from telemetry.schema import TelemetryRecord, DigitalTwinState, EngineConfig
from telemetry.ingestion import CanonicalTelemetryFrame, TelemetryIngestor
from digital_twin.twin_model import DigitalTwin, DigitalTwinModel
from digital_twin.residuals import ResidualGenerator, ResidualFrame, SUPPORTED_RESIDUAL_CHANNELS
from simulator.engine_simulator import EngineSimulator
from simulator.subsystems.mission import MissionProfile, PhaseSegment, FlightPhase
from simulator.fault_interface import (
    FaultState,
    FaultType,
    FaultSubsystem,
    FuelMixtureMode,
)


def _run_simulator_profile(
    duration_s: float = 20.0,
    throttle: float = 75.0,
    altitude: float = 2000.0,
    fault_schedule=None,
    seed: int = 42,
    dt: float = 0.5,
) -> CanonicalTelemetryFrame:
    """Helper to run simulator and return a CanonicalTelemetryFrame."""
    sim = EngineSimulator(seed=seed)
    seg = PhaseSegment(
        phase=FlightPhase.CRUISE,
        duration_s=duration_s,
        throttle_start_pct=throttle,
        throttle_end_pct=throttle,
        altitude_start_m=altitude,
        altitude_end_m=altitude,
    )
    profile = MissionProfile(segments=[seg])
    records = sim.run(profile, dt=dt, fault_schedule=fault_schedule)
    return TelemetryIngestor.ingest(records)


# ────────────────────────────────────────────────────────────────────────────
# Test A: Healthy Behavior
# ────────────────────────────────────────────────────────────────────────────

def test_a_healthy_behavior_bounded_residuals():
    """
    Verify that healthy flight telemetry produces bounded residuals within
    practical engineering tolerances (accounting for sensor noise and model approximation).
    """
    frame = _run_simulator_profile(duration_s=30.0, throttle=75.0, altitude=2000.0)
    twin = DigitalTwin()
    res_frame = twin.process_frame(frame)

    assert isinstance(res_frame, ResidualFrame)
    assert len(res_frame) == len(frame)

    # After initial transient settling (e.g. t >= 5s), residuals should remain tightly bounded
    df = res_frame.to_dataframe()
    steady = df[df["timestamp"] >= 5.0]

    # RPM residual within reasonable tolerance (< 150 RPM)
    assert steady["rpm_residual"].abs().max() < 150.0

    # CHT residual within reasonable tolerance (< 15 °C)
    assert steady["cht_residual"].abs().max() < 15.0

    # EGT residual within reasonable tolerance (< 30 °C)
    assert steady["egt_residual"].abs().max() < 30.0

    # Oil pressure residual within tolerance (< 0.6 bar)
    assert steady["oil_pressure_residual"].abs().max() < 0.6

    # Fuel flow residual within tolerance (< 2.5 L/h)
    assert steady["fuel_flow_residual"].abs().max() < 2.5

    # Vibration residual within tolerance (< 0.25 g)
    assert steady["vibration_residual"].abs().max() < 0.25


# ────────────────────────────────────────────────────────────────────────────
# Test B: Operating-Point Sensitivity
# ────────────────────────────────────────────────────────────────────────────

def test_b_operating_point_sensitivity():
    """
    Verify expected nominal states adjust sensibly across throttle,
    altitude, and ambient temperature variations.
    """
    model = DigitalTwinModel()

    # 1. Throttle sensitivity: 40% vs 90% at sea level
    model.reset()
    low_power = model.step_expected(throttle_pct=40.0, altitude_m=0.0, dt=1.0)
    # Run multiple steps at high power to allow dynamic RPM convergence
    for _ in range(10):
        high_power = model.step_expected(throttle_pct=90.0, altitude_m=0.0, dt=1.0)

    assert high_power["rpm_expected"] > low_power["rpm_expected"]
    assert high_power["fuel_flow_expected"] > low_power["fuel_flow_expected"]
    assert high_power["power_expected_kw"] > low_power["power_expected_kw"]

    # 2. Altitude derating: sea level (0m) vs high altitude (4000m)
    model.reset()
    for _ in range(10):
        sl_state = model.step_expected(throttle_pct=80.0, altitude_m=0.0, dt=1.0)
    model.reset()
    for _ in range(10):
        alt_state = model.step_expected(throttle_pct=80.0, altitude_m=4000.0, dt=1.0)

    # Indicated power and expected RPM derate with altitude
    assert alt_state["power_expected_kw"] < sl_state["power_expected_kw"]
    assert alt_state["rpm_expected"] < sl_state["rpm_expected"]

    # 3. Ambient temperature shift
    model.reset()
    for _ in range(15):
        cold_state = model.step_expected(throttle_pct=75.0, ambient_temp_c=-10.0, dt=1.0)
    model.reset()
    for _ in range(15):
        hot_state = model.step_expected(throttle_pct=75.0, ambient_temp_c=35.0, dt=1.0)

    assert hot_state["cht_expected"] > cold_state["cht_expected"]
    assert hot_state["oil_temp_expected"] > cold_state["oil_temp_expected"]


# ────────────────────────────────────────────────────────────────────────────
# Test C: Cooling Degradation Residual
# ────────────────────────────────────────────────────────────────────────────

def test_c_cooling_fault_positive_cht_residual():
    """
    Verify Phase 4B cooling degradation produces a pronounced positive CHT residual,
    while expected CHT remains nominal.
    """
    fault = FaultState(
        fault_type=FaultType.COOLING_DEGRADATION,
        severity=0.8,
        start_time=5.0,
        end_time=25.0,
        affected_subsystem=FaultSubsystem.THERMAL,
    )
    frame = _run_simulator_profile(duration_s=30.0, fault_schedule=fault)
    twin = DigitalTwin()
    res_frame = twin.process_frame(frame)

    df = res_frame.to_dataframe()

    # Compare residual during fault window (t in [15, 25]) vs before fault (t < 5)
    baseline_cht_res = df[df["timestamp"] < 5.0]["cht_residual"].mean()
    fault_cht_res = df[(df["timestamp"] >= 15.0) & (df["timestamp"] <= 25.0)]["cht_residual"].mean()

    # Cooling fault causes observed CHT to rise, increasing the residual
    assert fault_cht_res > baseline_cht_res + 15.0, (
        f"Expected CHT residual increase during cooling fault, got baseline {baseline_cht_res:.2f}, fault {fault_cht_res:.2f}"
    )


# ────────────────────────────────────────────────────────────────────────────
# Test D: Lubrication Degradation Residuals
# ────────────────────────────────────────────────────────────────────────────

def test_d_lubrication_fault_pressure_and_temp_residuals():
    """
    Verify Phase 4C lubrication degradation causes negative oil pressure residual
    and positive oil temperature residual.
    """
    fault = FaultState(
        fault_type=FaultType.LUBRICATION_DEGRADATION,
        severity=0.7,
        start_time=5.0,
        end_time=25.0,
        affected_subsystem=FaultSubsystem.LUBRICATION,
    )
    frame = _run_simulator_profile(duration_s=30.0, fault_schedule=fault)
    twin = DigitalTwin()
    res_frame = twin.process_frame(frame)

    df = res_frame.to_dataframe()
    fault_window = df[(df["timestamp"] >= 15.0) & (df["timestamp"] <= 25.0)]

    # Oil pressure drops -> negative residual
    oil_p_res = fault_window["oil_pressure_residual"].mean()
    assert oil_p_res < -0.8, f"Expected negative oil pressure residual, got {oil_p_res:.2f}"

    # Oil temp rises -> positive residual
    oil_t_res = fault_window["oil_temp_residual"].mean()
    baseline_oil_t_res = df[df["timestamp"] < 5.0]["oil_temp_residual"].mean()
    assert oil_t_res > baseline_oil_t_res + 3.0


# ────────────────────────────────────────────────────────────────────────────
# Test E: Fuel / Injection Abnormality Residuals
# ────────────────────────────────────────────────────────────────────────────

def test_e_fuel_abnormality_lean_and_rich_direction():
    """
    Verify Phase 4D fuel abnormality produces correct residual directions:
    - Lean: EGT residual > 0, Fuel flow residual < 0
    - Rich: EGT residual < 0, Fuel flow residual > 0
    """
    # 1. Lean injection fault
    lean_fault = FaultState(
        fault_type=FaultType.FUEL_INJECTION_ABNORMALITY,
        severity=0.8,
        start_time=5.0,
        end_time=25.0,
        affected_subsystem=FaultSubsystem.FUEL,
        parameters={"mode": "lean"},
    )
    frame_lean = _run_simulator_profile(duration_s=30.0, fault_schedule=lean_fault)
    twin = DigitalTwin()
    df_lean = twin.process_frame(frame_lean).to_dataframe()
    fault_lean = df_lean[(df_lean["timestamp"] >= 15.0) & (df_lean["timestamp"] <= 25.0)]

    assert fault_lean["egt_residual"].mean() > 30.0, "Lean fault should produce positive EGT residual"
    assert fault_lean["fuel_flow_residual"].mean() < -1.0, "Lean fault should produce negative fuel flow residual"

    # 2. Rich injection fault
    rich_fault = FaultState(
        fault_type=FaultType.FUEL_INJECTION_ABNORMALITY,
        severity=0.8,
        start_time=5.0,
        end_time=25.0,
        affected_subsystem=FaultSubsystem.FUEL,
        parameters={"mode": "rich"},
    )
    frame_rich = _run_simulator_profile(duration_s=30.0, fault_schedule=rich_fault)
    twin.reset()
    df_rich = twin.process_frame(frame_rich).to_dataframe()
    fault_rich = df_rich[(df_rich["timestamp"] >= 15.0) & (df_rich["timestamp"] <= 25.0)]

    assert fault_rich["egt_residual"].mean() < -30.0, "Rich fault should produce negative EGT residual"
    assert fault_rich["fuel_flow_residual"].mean() > 1.0, "Rich fault should produce positive fuel flow residual"


# ────────────────────────────────────────────────────────────────────────────
# Test F: Mechanical Degradation Residual
# ────────────────────────────────────────────────────────────────────────────

def test_f_mechanical_fault_vibration_residual():
    """
    Verify Phase 4E mechanical degradation produces positive vibration residual.
    """
    fault = FaultState(
        fault_type=FaultType.MECHANICAL_DEGRADATION,
        severity=0.8,
        start_time=5.0,
        end_time=25.0,
        affected_subsystem=FaultSubsystem.DYNAMICS,
    )
    frame = _run_simulator_profile(duration_s=30.0, fault_schedule=fault)
    twin = DigitalTwin()
    df = twin.process_frame(frame).to_dataframe()

    baseline_vib_res = df[df["timestamp"] < 5.0]["vibration_residual"].mean()
    fault_vib_res = df[(df["timestamp"] >= 10.0) & (df["timestamp"] <= 20.0)]["vibration_residual"].mean()

    assert fault_vib_res > baseline_vib_res + 0.3, "Mechanical degradation must increase vibration residual"


# ────────────────────────────────────────────────────────────────────────────
# Test G: Sensor Bias Isolation
# ────────────────────────────────────────────────────────────────────────────

def test_g_sensor_bias_isolation():
    """
    CRITICAL: Sensor bias changes observed residual, but MUST NOT alter
    the Digital Twin's expected physical state.
    """
    # Healthy run vs Biased run (100 RPM bias on sensor)
    frame_healthy = _run_simulator_profile(duration_s=20.0)
    bias_fault = FaultState(
        fault_type=FaultType.SENSOR_FAULT,
        severity=1.0,
        start_time=5.0,
        end_time=15.0,
        affected_subsystem=FaultSubsystem.SENSOR,
        parameters={"sensor_channel": "rpm", "sensor_mode": "bias", "bias_magnitude": 100.0},
    )
    frame_biased = _run_simulator_profile(duration_s=20.0, fault_schedule=bias_fault)

    twin = DigitalTwin()
    df_h = twin.process_frame(frame_healthy).to_dataframe()
    twin.reset()
    df_b = twin.process_frame(frame_biased).to_dataframe()

    # 1. Expected RPM must be identical between healthy and biased runs!
    np.testing.assert_allclose(
        df_h["rpm_expected"].values,
        df_b["rpm_expected"].values,
        rtol=1e-5,
        err_msg="Sensor bias corrupted Digital Twin expected state!",
    )

    # 2. Residual during fault window must reflect the +100 RPM bias
    mask = (df_b["timestamp"] >= 6.0) & (df_b["timestamp"] <= 14.0)
    res_diff = df_b.loc[mask, "rpm_residual"].mean() - df_h.loc[mask, "rpm_residual"].mean()
    assert pytest.approx(res_diff, abs=15.0) == 100.0


# ────────────────────────────────────────────────────────────────────────────
# Test H: Sensor Dropout Isolation
# ────────────────────────────────────────────────────────────────────────────

def test_h_sensor_dropout_nan_residual():
    """
    CRITICAL: Observed NaN from sensor dropout must produce NaN residual.
    Must NEVER be converted to 0.0, and expected state remains valid.
    """
    dropout_fault = FaultState(
        fault_type=FaultType.SENSOR_FAULT,
        severity=1.0,
        start_time=5.0,
        end_time=15.0,
        affected_subsystem=FaultSubsystem.SENSOR,
        parameters={"sensor_channel": "oil_pressure", "sensor_mode": "dropout"},
    )
    frame = _run_simulator_profile(duration_s=20.0, fault_schedule=dropout_fault)
    twin = DigitalTwin()
    df = twin.process_frame(frame).to_dataframe()

    mask_dropout = (df["timestamp"] >= 5.0) & (df["timestamp"] <= 15.0)
    for idx, row in df[mask_dropout].iterrows():
        assert math.isnan(row["oil_pressure"]), "Observed must remain NaN"
        assert math.isnan(row["oil_pressure_residual"]), "Residual must remain NaN (never converted to 0)"
        assert not math.isnan(row["oil_pressure_expected"]), "Expected state must remain physically valid"


# ────────────────────────────────────────────────────────────────────────────
# Test I: Sensor Drift Isolation
# ────────────────────────────────────────────────────────────────────────────

def test_i_sensor_drift_isolation():
    """
    Verify sensor drift causes progressively diverging residual while
    expected state remains stable.
    """
    drift_fault = FaultState(
        fault_type=FaultType.SENSOR_FAULT,
        severity=1.0,
        start_time=5.0,
        end_time=20.0,
        affected_subsystem=FaultSubsystem.SENSOR,
        parameters={"sensor_channel": "cht", "sensor_mode": "drift", "drift_rate": 2.0},
    )
    frame = _run_simulator_profile(duration_s=25.0, fault_schedule=drift_fault)
    twin = DigitalTwin()
    df = twin.process_frame(frame).to_dataframe()

    # Drift residual should grow over time between t=6 and t=18
    res_early = df.loc[abs(df["timestamp"] - 7.0) < 0.3, "cht_residual"].mean()
    res_late = df.loc[abs(df["timestamp"] - 17.0) < 0.3, "cht_residual"].mean()
    assert res_late > res_early + 15.0, "Drift residual must grow monotonically with elapsed time"


# ────────────────────────────────────────────────────────────────────────────
# Test J: Regression Compatibility
# ────────────────────────────────────────────────────────────────────────────

def test_j_backward_compatibility_update():
    """Verify Phase 1 twin.update(telemetry) contract remains functional."""
    rec = TelemetryRecord(
        timestamp=1.0,
        mission_id="M1",
        engine_id="E1",
        mission_phase="CRUISE",
        altitude=2000.0,
        ambient_temp=15.0,
        throttle=75.0,
        load=70.0,
        rpm=4000.0,
        cht=100.0,
        egt=650.0,
        oil_temp=85.0,
        oil_pressure=4.2,
        fuel_flow=12.0,
        vibration=0.5,
        source="sim",
        source_type="synthetic",
    )
    twin = DigitalTwin()
    twin_state = twin.update(rec)

    assert isinstance(twin_state, DigitalTwinState)
    assert "cht_residual" in twin_state.residuals
    assert "nominal_cht" in twin_state.nominal_estimates
    assert twin_state.engine_id == "E1"
    assert len(twin.history) == 1


# ────────────────────────────────────────────────────────────────────────────
# Test K: Determinism
# ────────────────────────────────────────────────────────────────────────────

def test_k_determinism():
    """Verify identical inputs produce identical expected trajectories and residuals."""
    frame = _run_simulator_profile(duration_s=10.0)

    twin1 = DigitalTwin()
    res1 = twin1.process_frame(frame).to_dataframe()

    twin2 = DigitalTwin()
    res2 = twin2.process_frame(frame).to_dataframe()

    pd.testing.assert_frame_equal(res1, res2)


# ────────────────────────────────────────────────────────────────────────────
# Test L: Causality
# ────────────────────────────────────────────────────────────────────────────

def test_l_causality():
    """
    Verify changing future observations (t=8) does not alter prior expected
    states or residuals (t <= 5).
    """
    frame_orig = _run_simulator_profile(duration_s=10.0)
    df_mod = frame_orig.to_dataframe().copy()
    # Corrupt future row at index 15
    df_mod.loc[15, "rpm"] = 9999.0
    frame_mod = CanonicalTelemetryFrame(df_mod)

    twin = DigitalTwin()
    res_orig = twin.process_frame(frame_orig).to_dataframe()
    twin.reset()
    res_mod = twin.process_frame(frame_mod).to_dataframe()

    # Rows 0 through 14 must be bitwise identical
    np.testing.assert_allclose(
        res_orig.loc[:14, "rpm_expected"].values,
        res_mod.loc[:14, "rpm_expected"].values,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        res_orig.loc[:14, "rpm_residual"].values,
        res_mod.loc[:14, "rpm_residual"].values,
        rtol=1e-12,
    )


# ────────────────────────────────────────────────────────────────────────────
# Test M: Provenance Preservation
# ────────────────────────────────────────────────────────────────────────────

def test_m_provenance_preservation():
    """Verify engine_id, mission_id, source survive in ResidualFrame."""
    frame = _run_simulator_profile(duration_s=5.0)
    twin = DigitalTwin()
    res_frame = twin.process_frame(frame)
    df = res_frame.to_dataframe()

    assert "engine_id" in df.columns
    assert "mission_id" in df.columns
    assert "source" in df.columns
    assert df["source"].iloc[0] == "simulator_v1_physics"


# ────────────────────────────────────────────────────────────────────────────
# Test N: Multi-Mission Separation
# ────────────────────────────────────────────────────────────────────────────

def test_n_multi_mission_separation():
    """Verify that calling reset() cleanly resets twin state between missions."""
    frame1 = _run_simulator_profile(duration_s=5.0, throttle=50.0)
    frame2 = _run_simulator_profile(duration_s=5.0, throttle=90.0)

    twin = DigitalTwin()
    twin.process_frame(frame1)

    # Calling reset clears internal states
    twin.reset()
    assert twin.model.expected_rpm == twin.sim_config.tier_c.rpm_idle
    assert len(twin.history) == 0

    res2 = twin.process_frame(frame2).to_dataframe()
    assert len(res2) == len(frame2)


# ────────────────────────────────────────────────────────────────────────────
# Test O: Numerical Stability
# ────────────────────────────────────────────────────────────────────────────

def test_o_numerical_stability():
    """Verify zero unexpected NaNs or Infs across nominal flight profile."""
    frame = _run_simulator_profile(duration_s=25.0)
    twin = DigitalTwin()
    res_frame = twin.process_frame(frame)
    df = res_frame.to_dataframe()

    for ch in SUPPORTED_RESIDUAL_CHANNELS:
        exp_col = f"{ch}_expected"
        res_col = f"{ch}_residual"
        assert not df[exp_col].isna().any(), f"Unexpected NaN in {exp_col}"
        assert not np.isinf(df[exp_col]).any(), f"Unexpected Inf in {exp_col}"
        assert not df[res_col].isna().any(), f"Unexpected NaN in {res_col}"
        assert not np.isinf(df[res_col]).any(), f"Unexpected Inf in {res_col}"
