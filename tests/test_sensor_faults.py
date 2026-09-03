"""
Unit and Validation Tests for Phase 4F: Sensor Faults (Observation-Layer).

Covers:
- Test A: Fault interface — sensor fault can be represented and scheduled
- Test B: Severity validation — invalid severity values are rejected
- Test C: Healthy equivalence — severity 0 produces nominal-equivalent observations
- Test D: Bias — observed value shifts according to configured bias
- Test E: Drift — observed error increases with elapsed fault time
- Test F: Stuck sensor — observed remains fixed while physical simulation changes
- Test G: Additional noise — increased variance with deterministic seeded behavior
- Test H: Dropout — affected observation becomes NaN
- Test I: Fault-window isolation — nominal before/after, corrupted during
- Test J: Physical-state invariance — sensor fault does NOT alter physical trajectory
- Test K: Determinism — same seed + config → identical corrupted telemetry
- Test L: Different seeds → different stochastic sensor noise
- Test M: Channel isolation — fault on one channel doesn't affect others
- Test N: Existing fault regression — all 4B–4E faults still work
- Test O: Combined physical + sensor fault coexistence
- Test P: Sequential mission reset — sensor fault state cleared between missions

DISCLAIMER:
All sensor-fault magnitudes are Tier C/D engineering assumptions.
They do NOT represent measured or certified UAV-engine sensor characteristics.
"""

import math
import numpy as np
import pytest
from simulator.engine_simulator import EngineSimulator
from simulator.config import SimulatorConfig
from simulator.fault_interface import (
    FaultState, FaultType, FaultSubsystem, FaultSchedule,
)
from simulator.sensor_faults import SensorFaultMode, SensorChannel, SensorFaultProcessor
from simulator.subsystems.mission import MissionProfile, PhaseSegment, FlightPhase


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────

def _make_cruise_profile(duration_s=60.0):
    """Create a simple cruise-phase mission profile."""
    seg = PhaseSegment(FlightPhase.CRUISE, duration_s, 75.0, 75.0, 2000.0, 2000.0)
    return MissionProfile(segments=[seg])


def _make_sensor_fault(
    channel: str = "rpm",
    mode: str = "bias",
    severity: float = 0.5,
    start_time: float = 10.0,
    end_time: float = 50.0,
    **extra_params,
):
    """Helper to create a sensor FaultState with the given parameters."""
    params = {
        "sensor_channel": channel,
        "sensor_mode": mode,
    }
    params.update(extra_params)
    return FaultState(
        fault_type=FaultType.SENSOR_FAULT,
        severity=severity,
        start_time=start_time,
        end_time=end_time,
        affected_subsystem=FaultSubsystem.SENSOR,
        parameters=params,
    )


# ────────────────────────────────────────────────────────────────────────────
# Test A — Fault interface
# ────────────────────────────────────────────────────────────────────────────

def test_a_fault_interface_representation():
    """Test A: Sensor fault can be represented and scheduled via FaultState/FaultSchedule."""
    fault = _make_sensor_fault(channel="cht", mode="bias", severity=0.7,
                               start_time=5.0, end_time=25.0)

    assert fault.fault_type == FaultType.SENSOR_FAULT
    assert fault.affected_subsystem == FaultSubsystem.SENSOR
    assert fault.severity == 0.7
    assert fault.parameters["sensor_channel"] == "cht"
    assert fault.parameters["sensor_mode"] == "bias"

    # Can be added to a schedule
    schedule = FaultSchedule()
    schedule.add_fault(fault)
    assert len(schedule) == 1

    # is_active_at respects window
    assert not fault.is_active_at(3.0)
    assert fault.is_active_at(10.0)
    assert fault.is_active_at(25.0)
    assert not fault.is_active_at(26.0)


def test_a2_fault_serialization_roundtrip():
    """Test A2: Sensor fault serializes and deserializes correctly."""
    fault = _make_sensor_fault(channel="oil_pressure", mode="drift",
                               severity=0.55, drift_rate=0.1)
    d = fault.to_dict()
    reconstructed = FaultState.from_dict(d)
    assert reconstructed.fault_type == FaultType.SENSOR_FAULT
    assert reconstructed.severity == 0.55
    assert reconstructed.parameters["sensor_channel"] == "oil_pressure"
    assert reconstructed.parameters["sensor_mode"] == "drift"
    assert reconstructed.parameters["drift_rate"] == 0.1


# ────────────────────────────────────────────────────────────────────────────
# Test B — Severity validation
# ────────────────────────────────────────────────────────────────────────────

def test_b_severity_validation_negative():
    """Test B: Negative severity is rejected."""
    with pytest.raises(ValueError, match="severity"):
        _make_sensor_fault(severity=-0.1)


def test_b2_severity_validation_over_one():
    """Test B2: Severity > 1.0 is rejected."""
    with pytest.raises(ValueError, match="severity"):
        _make_sensor_fault(severity=1.5)


def test_b3_severity_boundary_valid():
    """Test B3: Boundary values 0.0 and 1.0 are accepted."""
    f0 = _make_sensor_fault(severity=0.0)
    f1 = _make_sensor_fault(severity=1.0)
    assert f0.severity == 0.0
    assert f1.severity == 1.0


# ────────────────────────────────────────────────────────────────────────────
# Test C — Healthy equivalence (severity 0)
# ────────────────────────────────────────────────────────────────────────────

def test_c_healthy_equivalence_severity_zero():
    """
    Test C: Severity 0 produces nominal-equivalent observations.
    A sensor fault with severity=0.0 must not alter any telemetry values.
    """
    sim_none = EngineSimulator(seed=42)
    sim_zero = EngineSimulator(seed=42)

    profile = _make_cruise_profile(40.0)
    fault_zero = _make_sensor_fault(
        channel="rpm", mode="bias", severity=0.0,
        start_time=5.0, end_time=35.0,
    )

    df_none = sim_none.run_to_dataframe(profile, dt=0.5)
    df_zero = sim_zero.run_to_dataframe(profile, dt=0.5, fault_schedule=fault_zero)

    assert len(df_none) == len(df_zero)
    for col in ["rpm", "cht", "egt", "oil_temp", "oil_pressure", "fuel_flow", "vibration"]:
        np.testing.assert_allclose(
            df_none[col].values, df_zero[col].values, rtol=1e-12,
            err_msg=f"Severity=0 sensor fault altered {col}"
        )


# ────────────────────────────────────────────────────────────────────────────
# Test D — Bias
# ────────────────────────────────────────────────────────────────────────────

def test_d_bias_shifts_observation():
    """
    Test D: Observed value shifts by configured bias magnitude × severity.
    """
    sim_healthy = EngineSimulator(seed=42)
    sim_biased = EngineSimulator(seed=42)

    profile = _make_cruise_profile(60.0)
    bias_mag = 100.0
    severity = 0.5
    fault = _make_sensor_fault(
        channel="rpm", mode="bias", severity=severity,
        start_time=10.0, end_time=50.0,
        bias_magnitude=bias_mag,
    )

    df_h = sim_healthy.run_to_dataframe(profile, dt=0.5)
    df_b = sim_biased.run_to_dataframe(profile, dt=0.5, fault_schedule=fault)

    # During fault window: biased RPM should be ~50 RPM higher
    mask_active = (df_h["timestamp"] >= 10.0) & (df_h["timestamp"] <= 50.0)
    diffs = df_b.loc[mask_active, "rpm"].values - df_h.loc[mask_active, "rpm"].values
    expected_bias = bias_mag * severity  # 50.0
    np.testing.assert_allclose(diffs, expected_bias, atol=1.0,
                               err_msg="Bias did not shift RPM by expected amount")

    # Outside fault window: should be identical
    mask_before = df_h["timestamp"] < 10.0
    if mask_before.any():
        np.testing.assert_allclose(
            df_h.loc[mask_before, "rpm"].values,
            df_b.loc[mask_before, "rpm"].values, rtol=1e-12,
        )


def test_d2_bias_monotonic_with_severity():
    """Test D2: Higher severity → larger measurement bias."""
    biases = []
    for sev in [0.2, 0.5, 0.8, 1.0]:
        sim_h = EngineSimulator(seed=42)
        sim_f = EngineSimulator(seed=42)
        profile = _make_cruise_profile(30.0)
        fault = _make_sensor_fault(
            channel="egt", mode="bias", severity=sev,
            start_time=5.0, end_time=25.0,
        )
        df_h = sim_h.run_to_dataframe(profile, dt=1.0)
        df_f = sim_f.run_to_dataframe(profile, dt=1.0, fault_schedule=fault)
        mask = (df_h["timestamp"] >= 5.0) & (df_h["timestamp"] <= 25.0)
        mean_diff = (df_f.loc[mask, "egt"] - df_h.loc[mask, "egt"]).mean()
        biases.append(mean_diff)

    # Should be strictly increasing
    for i in range(len(biases) - 1):
        assert biases[i + 1] > biases[i], \
            f"Bias not monotonic: sev={[0.2,0.5,0.8,1.0][i+1]} gave {biases[i+1]} <= {biases[i]}"


# ────────────────────────────────────────────────────────────────────────────
# Test E — Drift
# ────────────────────────────────────────────────────────────────────────────

def test_e_drift_increases_with_time():
    """
    Test E: Observed error increases with elapsed fault time.
    """
    sim_h = EngineSimulator(seed=42)
    sim_d = EngineSimulator(seed=42)

    profile = _make_cruise_profile(60.0)
    drift_rate = 2.0  # °C/s
    fault = _make_sensor_fault(
        channel="cht", mode="drift", severity=1.0,
        start_time=10.0, end_time=50.0,
        drift_rate=drift_rate,
    )

    df_h = sim_h.run_to_dataframe(profile, dt=1.0)
    df_d = sim_d.run_to_dataframe(profile, dt=1.0, fault_schedule=fault)

    # At t=20 (elapsed=10s): expected drift = 2.0 * 1.0 * 10 = 20°C
    # At t=40 (elapsed=30s): expected drift = 2.0 * 1.0 * 30 = 60°C
    mask_early = (df_h["timestamp"] >= 19.5) & (df_h["timestamp"] <= 20.5)
    mask_late = (df_h["timestamp"] >= 39.5) & (df_h["timestamp"] <= 40.5)

    diff_early = (df_d.loc[mask_early, "cht"] - df_h.loc[mask_early, "cht"]).mean()
    diff_late = (df_d.loc[mask_late, "cht"] - df_h.loc[mask_late, "cht"]).mean()

    assert diff_late > diff_early, "Drift error should increase with elapsed time"
    # Approximate checks
    np.testing.assert_allclose(diff_early, 20.0, atol=5.0)
    np.testing.assert_allclose(diff_late, 60.0, atol=5.0)


# ────────────────────────────────────────────────────────────────────────────
# Test F — Stuck sensor
# ────────────────────────────────────────────────────────────────────────────

def test_f_stuck_sensor():
    """
    Test F: Observed value remains fixed while physical simulation continues.
    The stuck value is the observed (noisy) value at fault activation, NOT ground truth.
    """
    sim_h = EngineSimulator(seed=42)
    sim_s = EngineSimulator(seed=42)

    # Use a profile with changing throttle to ensure RPM changes
    seg1 = PhaseSegment(FlightPhase.CRUISE, 15.0, 60.0, 60.0, 2000.0, 2000.0)
    seg2 = PhaseSegment(FlightPhase.CRUISE, 30.0, 90.0, 90.0, 2000.0, 2000.0)
    seg3 = PhaseSegment(FlightPhase.CRUISE, 15.0, 60.0, 60.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg1, seg2, seg3])

    fault = _make_sensor_fault(
        channel="rpm", mode="stuck", severity=1.0,
        start_time=15.0, end_time=45.0,
    )

    df_h = sim_h.run_to_dataframe(profile, dt=0.5)
    df_s = sim_s.run_to_dataframe(profile, dt=0.5, fault_schedule=fault)

    # Find stuck value: first reading at or after t=15.0 in faulted run
    mask_stuck_start = df_s["timestamp"] >= 15.0
    stuck_value = df_s.loc[mask_stuck_start, "rpm"].iloc[0]

    # During fault window, all stuck RPM should be the latched value
    mask_fault = (df_s["timestamp"] >= 15.0) & (df_s["timestamp"] <= 45.0)
    stuck_readings = df_s.loc[mask_fault, "rpm"].values
    np.testing.assert_allclose(stuck_readings, stuck_value, rtol=1e-12,
                               err_msg="Stuck sensor values are not constant")

    # Physical RPM should be changing during this window
    healthy_readings = df_h.loc[mask_fault, "rpm"].values
    rpm_range = healthy_readings.max() - healthy_readings.min()
    assert rpm_range > 50.0, f"Physical RPM not changing enough to validate stuck: range={rpm_range}"

    # After fault window, observation should resume tracking physical state
    mask_after = df_s["timestamp"] > 46.0
    if mask_after.any():
        after_stuck = df_s.loc[mask_after, "rpm"].values
        after_healthy = df_h.loc[mask_after, "rpm"].values
        # Should be close to healthy (within sensor noise)
        np.testing.assert_allclose(after_stuck, after_healthy, rtol=1e-4,
                                   err_msg="Stuck sensor did not release after fault window")


# ────────────────────────────────────────────────────────────────────────────
# Test G — Additional noise
# ────────────────────────────────────────────────────────────────────────────

def test_g_noise_increases_variance():
    """
    Test G: Fault-induced noise increases observation variance while
    preserving deterministic seeded behavior.
    """
    sim_h = EngineSimulator(seed=42)
    sim_n = EngineSimulator(seed=42)

    profile = _make_cruise_profile(60.0)
    fault = _make_sensor_fault(
        channel="cht", mode="noise", severity=1.0,
        start_time=5.0, end_time=55.0,
    )

    df_h = sim_h.run_to_dataframe(profile, dt=0.5)
    df_n = sim_n.run_to_dataframe(profile, dt=0.5, fault_schedule=fault)

    # During fault window: variance of faulted should be higher
    mask = (df_h["timestamp"] >= 10.0) & (df_h["timestamp"] <= 50.0)
    var_h = df_h.loc[mask, "cht"].var()
    var_n = df_n.loc[mask, "cht"].var()

    assert var_n > var_h, f"Fault noise did not increase variance: {var_n} <= {var_h}"


# ────────────────────────────────────────────────────────────────────────────
# Test H — Dropout
# ────────────────────────────────────────────────────────────────────────────

def test_h_dropout_produces_nan():
    """
    Test H: Affected observation becomes NaN using float('nan').
    NaN must survive through clamping, rounding, and record construction.
    """
    sim = EngineSimulator(seed=42)
    profile = _make_cruise_profile(40.0)

    fault = _make_sensor_fault(
        channel="oil_pressure", mode="dropout", severity=1.0,
        start_time=10.0, end_time=30.0,
    )

    df = sim.run_to_dataframe(profile, dt=1.0, fault_schedule=fault)

    # During fault window: oil_pressure should be NaN
    mask_fault = (df["timestamp"] >= 10.0) & (df["timestamp"] <= 30.0)
    assert mask_fault.any(), "No records in fault window"
    for val in df.loc[mask_fault, "oil_pressure"]:
        assert math.isnan(val), f"Dropout should produce NaN, got {val}"

    # Before fault: oil_pressure should be valid (not NaN)
    mask_before = df["timestamp"] < 10.0
    if mask_before.any():
        for val in df.loc[mask_before, "oil_pressure"]:
            assert not math.isnan(val), f"Pre-fault oil_pressure should not be NaN"

    # After fault: oil_pressure should recover (not NaN)
    mask_after = df["timestamp"] > 30.0
    if mask_after.any():
        for val in df.loc[mask_after, "oil_pressure"]:
            assert not math.isnan(val), f"Post-fault oil_pressure should not be NaN"


def test_h2_dropout_nan_survives_serialization():
    """
    Test H2: NaN in dropout channel survives to_dict() serialization.
    Do NOT accidentally convert NaN to zero.
    """
    sim = EngineSimulator(seed=42)
    profile = _make_cruise_profile(20.0)

    fault = _make_sensor_fault(
        channel="rpm", mode="dropout", severity=1.0,
        start_time=5.0, end_time=15.0,
    )
    records = sim.run(profile, dt=1.0, fault_schedule=fault)

    for rec in records:
        d = rec.to_dict()
        if 5.0 <= rec.timestamp <= 15.0:
            assert math.isnan(d["rpm"]), \
                f"NaN in dropout RPM did not survive to_dict at t={rec.timestamp}: got {d['rpm']}"


def test_h3_dropout_not_zero():
    """
    Test H3: Dropout must be NaN, NOT zero. Zero is a valid physical value.
    """
    sim = EngineSimulator(seed=42)
    profile = _make_cruise_profile(20.0)

    fault = _make_sensor_fault(
        channel="fuel_flow", mode="dropout", severity=1.0,
        start_time=5.0, end_time=15.0,
    )
    df = sim.run_to_dataframe(profile, dt=1.0, fault_schedule=fault)

    mask = (df["timestamp"] >= 5.0) & (df["timestamp"] <= 15.0)
    for val in df.loc[mask, "fuel_flow"]:
        assert math.isnan(val), f"Dropout fuel_flow should be NaN, not {val}"
        assert val != 0.0, "Dropout must not be zero"


# ────────────────────────────────────────────────────────────────────────────
# Test I — Fault-window isolation
# ────────────────────────────────────────────────────────────────────────────

def test_i_fault_window_isolation():
    """
    Test I: Before fault = nominal, during fault = corrupted, after fault = nominal.
    """
    sim_h = EngineSimulator(seed=42)
    sim_f = EngineSimulator(seed=42)

    profile = _make_cruise_profile(60.0)
    fault = _make_sensor_fault(
        channel="egt", mode="bias", severity=0.8,
        start_time=20.0, end_time=40.0,
        bias_magnitude=50.0,
    )

    df_h = sim_h.run_to_dataframe(profile, dt=0.5)
    df_f = sim_f.run_to_dataframe(profile, dt=0.5, fault_schedule=fault)

    # Before: identical
    mask_before = df_h["timestamp"] < 20.0
    np.testing.assert_allclose(
        df_h.loc[mask_before, "egt"].values,
        df_f.loc[mask_before, "egt"].values, rtol=1e-12,
    )

    # During: biased
    mask_during = (df_h["timestamp"] >= 20.0) & (df_h["timestamp"] <= 40.0)
    diffs_during = df_f.loc[mask_during, "egt"].values - df_h.loc[mask_during, "egt"].values
    expected = 50.0 * 0.8  # = 40.0
    np.testing.assert_allclose(diffs_during, expected, atol=1.0)

    # After: identical (recovered)
    mask_after = df_h["timestamp"] > 40.0
    if mask_after.any():
        np.testing.assert_allclose(
            df_h.loc[mask_after, "egt"].values,
            df_f.loc[mask_after, "egt"].values, rtol=1e-12,
        )


# ────────────────────────────────────────────────────────────────────────────
# Test J — Physical-state invariance (CRITICAL)
# ────────────────────────────────────────────────────────────────────────────

def test_j_physical_state_invariance():
    """
    Test J (CRITICAL): Run identical simulations — one healthy, one with sensor fault.
    Verify that the underlying physical engine evolution is BITWISE IDENTICAL.

    This is the core guarantee: sensor faults NEVER modify physical state.
    """
    sim_h = EngineSimulator(seed=42)
    sim_f = EngineSimulator(seed=42)

    profile = _make_cruise_profile(60.0)
    # Apply a dramatic sensor fault on RPM
    fault = _make_sensor_fault(
        channel="rpm", mode="bias", severity=1.0,
        start_time=10.0, end_time=50.0,
        bias_magnitude=500.0,  # Large bias
    )

    # Run healthy
    sim_h.reset()
    records_h = sim_h.run(profile, dt=0.5)

    # Run with sensor fault
    sim_f.reset()
    records_f = sim_f.run(profile, dt=0.5, fault_schedule=fault)

    assert len(records_h) == len(records_f)

    # Physical state is exposed through metadata (power, torque)
    # and through non-faulted channels.
    # CHT, EGT, oil_temp, oil_pressure, fuel_flow, vibration must be identical
    # because the sensor fault only targets RPM.
    for rh, rf in zip(records_h, records_f):
        assert rh.cht == rf.cht, f"CHT altered at t={rh.timestamp}"
        assert rh.egt == rf.egt, f"EGT altered at t={rh.timestamp}"
        assert rh.oil_temp == rf.oil_temp, f"oil_temp altered at t={rh.timestamp}"
        assert rh.oil_pressure == rf.oil_pressure, f"oil_pressure altered at t={rh.timestamp}"
        assert rh.fuel_flow == rf.fuel_flow, f"fuel_flow altered at t={rh.timestamp}"
        assert rh.vibration == rf.vibration, f"vibration altered at t={rh.timestamp}"
        # Metadata (power, torque) must also be identical
        assert rh.metadata["power_kw"] == rf.metadata["power_kw"], \
            f"power_kw altered at t={rh.timestamp}"
        assert rh.metadata["torque_nm"] == rf.metadata["torque_nm"], \
            f"torque_nm altered at t={rh.timestamp}"


def test_j2_physical_invariance_all_channels():
    """
    Test J2: Sensor faults on ALL channels simultaneously still
    produce identical physical metadata (power, torque, frequencies).
    """
    sim_h = EngineSimulator(seed=42)
    sim_f = EngineSimulator(seed=42)

    profile = _make_cruise_profile(30.0)
    channels = ["rpm", "cht", "egt", "oil_temp", "oil_pressure", "fuel_flow", "vibration"]
    faults = [
        _make_sensor_fault(channel=ch, mode="bias", severity=0.8,
                           start_time=5.0, end_time=25.0, bias_magnitude=100.0)
        for ch in channels
    ]
    schedule = FaultSchedule(faults)

    records_h = sim_h.run(profile, dt=1.0)
    records_f = sim_f.run(profile, dt=1.0, fault_schedule=schedule)

    for rh, rf in zip(records_h, records_f):
        assert rh.metadata["power_kw"] == rf.metadata["power_kw"]
        assert rh.metadata["torque_nm"] == rf.metadata["torque_nm"]
        assert rh.metadata["order_1x_freq_hz"] == rf.metadata["order_1x_freq_hz"]
        assert rh.metadata["order_2x_freq_hz"] == rf.metadata["order_2x_freq_hz"]
        assert rh.metadata["bsfc_g_kwh"] == rf.metadata["bsfc_g_kwh"]


# ────────────────────────────────────────────────────────────────────────────
# Test K — Determinism
# ────────────────────────────────────────────────────────────────────────────

def test_k_determinism():
    """
    Test K: Same seed + same configuration → identical corrupted telemetry.
    """
    fault = _make_sensor_fault(
        channel="cht", mode="noise", severity=0.7,
        start_time=5.0, end_time=25.0,
    )

    results = []
    for _ in range(3):
        sim = EngineSimulator(seed=42)
        df = sim.run_to_dataframe(_make_cruise_profile(30.0), dt=1.0,
                                  fault_schedule=fault)
        results.append(df["cht"].values)

    np.testing.assert_array_equal(results[0], results[1])
    np.testing.assert_array_equal(results[1], results[2])


# ────────────────────────────────────────────────────────────────────────────
# Test L — Different seeds
# ────────────────────────────────────────────────────────────────────────────

def test_l_different_seeds():
    """
    Test L: Different seeds → different stochastic sensor noise.
    """
    fault = _make_sensor_fault(
        channel="cht", mode="noise", severity=1.0,
        start_time=5.0, end_time=25.0,
    )

    sim_a = EngineSimulator(seed=42)
    sim_b = EngineSimulator(seed=99)

    df_a = sim_a.run_to_dataframe(_make_cruise_profile(30.0), dt=1.0,
                                  fault_schedule=fault)
    df_b = sim_b.run_to_dataframe(_make_cruise_profile(30.0), dt=1.0,
                                  fault_schedule=fault)

    mask = (df_a["timestamp"] >= 5.0) & (df_a["timestamp"] <= 25.0)
    vals_a = df_a.loc[mask, "cht"].values
    vals_b = df_b.loc[mask, "cht"].values

    # They should differ (extremely unlikely to be identical)
    assert not np.allclose(vals_a, vals_b, atol=0.01), \
        "Different seeds produced identical noisy telemetry"


# ────────────────────────────────────────────────────────────────────────────
# Test M — Channel isolation
# ────────────────────────────────────────────────────────────────────────────

def test_m_channel_isolation():
    """
    Test M: A sensor fault on one channel must not alter unrelated channels.
    """
    sim_h = EngineSimulator(seed=42)
    sim_f = EngineSimulator(seed=42)

    profile = _make_cruise_profile(40.0)
    # Fault only on vibration
    fault = _make_sensor_fault(
        channel="vibration", mode="bias", severity=1.0,
        start_time=5.0, end_time=35.0,
        bias_magnitude=1.0,
    )

    df_h = sim_h.run_to_dataframe(profile, dt=0.5)
    df_f = sim_f.run_to_dataframe(profile, dt=0.5, fault_schedule=fault)

    # All non-vibration channels must be identical
    for col in ["rpm", "cht", "egt", "oil_temp", "oil_pressure", "fuel_flow"]:
        np.testing.assert_allclose(
            df_h[col].values, df_f[col].values, rtol=1e-12,
            err_msg=f"Vibration fault altered unrelated channel: {col}"
        )

    # Vibration should be different during fault
    mask = (df_h["timestamp"] >= 5.0) & (df_h["timestamp"] <= 35.0)
    assert not np.allclose(
        df_h.loc[mask, "vibration"].values,
        df_f.loc[mask, "vibration"].values, atol=0.01,
    )


# ────────────────────────────────────────────────────────────────────────────
# Test N — Existing fault regression
# ────────────────────────────────────────────────────────────────────────────

def test_n_cooling_degradation_regression():
    """Test N: Cooling degradation still works after Phase 4F changes."""
    sim = EngineSimulator(seed=42)
    profile = _make_cruise_profile(60.0)

    fault = FaultState(
        fault_type=FaultType.COOLING_DEGRADATION,
        severity=0.7,
        start_time=10.0,
        end_time=50.0,
    )
    df_h = EngineSimulator(seed=42).run_to_dataframe(profile, dt=1.0)
    df_f = sim.run_to_dataframe(profile, dt=1.0, fault_schedule=fault)

    # CHT should be higher with cooling degradation
    mask = (df_h["timestamp"] >= 30.0) & (df_h["timestamp"] <= 50.0)
    assert df_f.loc[mask, "cht"].mean() > df_h.loc[mask, "cht"].mean(), \
        "Cooling degradation no longer raises CHT"


def test_n2_lubrication_degradation_regression():
    """Test N2: Lubrication degradation still works after Phase 4F changes."""
    sim = EngineSimulator(seed=42)
    profile = _make_cruise_profile(60.0)

    fault = FaultState(
        fault_type=FaultType.LUBRICATION_DEGRADATION,
        severity=0.7,
        start_time=10.0,
        end_time=50.0,
    )
    df_h = EngineSimulator(seed=42).run_to_dataframe(profile, dt=1.0)
    df_f = sim.run_to_dataframe(profile, dt=1.0, fault_schedule=fault)

    mask = (df_h["timestamp"] >= 30.0) & (df_h["timestamp"] <= 50.0)
    assert df_f.loc[mask, "oil_pressure"].mean() < df_h.loc[mask, "oil_pressure"].mean(), \
        "Lubrication degradation no longer reduces oil pressure"


def test_n3_fuel_injection_regression():
    """Test N3: Fuel injection abnormality still works after Phase 4F changes."""
    sim = EngineSimulator(seed=42)
    profile = _make_cruise_profile(60.0)

    fault = FaultState(
        fault_type=FaultType.FUEL_INJECTION_ABNORMALITY,
        severity=0.7,
        start_time=10.0,
        end_time=50.0,
        parameters={"mode": "lean"},
    )
    df_h = EngineSimulator(seed=42).run_to_dataframe(profile, dt=1.0)
    df_f = sim.run_to_dataframe(profile, dt=1.0, fault_schedule=fault)

    mask = (df_h["timestamp"] >= 30.0) & (df_h["timestamp"] <= 50.0)
    assert df_f.loc[mask, "egt"].mean() > df_h.loc[mask, "egt"].mean(), \
        "Lean fuel abnormality no longer raises EGT"


def test_n4_mechanical_degradation_regression():
    """Test N4: Mechanical degradation still works after Phase 4F changes."""
    sim = EngineSimulator(seed=42)
    profile = _make_cruise_profile(60.0)

    fault = FaultState(
        fault_type=FaultType.MECHANICAL_DEGRADATION,
        severity=0.7,
        start_time=10.0,
        end_time=50.0,
    )
    df_h = EngineSimulator(seed=42).run_to_dataframe(profile, dt=1.0)
    df_f = sim.run_to_dataframe(profile, dt=1.0, fault_schedule=fault)

    mask = (df_h["timestamp"] >= 20.0) & (df_h["timestamp"] <= 50.0)
    assert df_f.loc[mask, "vibration"].mean() > df_h.loc[mask, "vibration"].mean(), \
        "Mechanical degradation no longer increases vibration"


# ────────────────────────────────────────────────────────────────────────────
# Test O — Combined physical + sensor fault
# ────────────────────────────────────────────────────────────────────────────

def test_o_combined_physical_and_sensor_fault():
    """
    Test O: A physical fault and sensor fault can coexist.
    Physical fault → physical state changes.
    Sensor fault → observation changes only.
    The sensor fault must NOT alter physical fault dynamics.
    """
    sim_phys = EngineSimulator(seed=42)
    sim_both = EngineSimulator(seed=42)

    profile = _make_cruise_profile(60.0)

    # Physical fault: cooling degradation
    cooling_fault = FaultState(
        fault_type=FaultType.COOLING_DEGRADATION,
        severity=0.6,
        start_time=10.0,
        end_time=50.0,
    )
    # Sensor fault: RPM bias
    sensor_fault = _make_sensor_fault(
        channel="rpm", mode="bias", severity=0.8,
        start_time=10.0, end_time=50.0,
        bias_magnitude=100.0,
    )

    # Physics-only run
    schedule_phys = FaultSchedule([cooling_fault])
    # Combined run
    schedule_both = FaultSchedule([cooling_fault, sensor_fault])

    df_p = sim_phys.run_to_dataframe(profile, dt=1.0, fault_schedule=schedule_phys)
    df_b = sim_both.run_to_dataframe(profile, dt=1.0, fault_schedule=schedule_both)

    # CHT should be identical (cooling degradation physics unchanged by sensor fault)
    np.testing.assert_allclose(
        df_p["cht"].values, df_b["cht"].values, rtol=1e-12,
        err_msg="Sensor fault altered cooling degradation physics (CHT)"
    )

    # EGT, oil_temp, oil_pressure, fuel_flow, vibration also identical
    for col in ["egt", "oil_temp", "oil_pressure", "fuel_flow", "vibration"]:
        np.testing.assert_allclose(
            df_p[col].values, df_b[col].values, rtol=1e-12,
            err_msg=f"Sensor fault altered physical channel {col}"
        )

    # RPM should differ during fault window (sensor bias)
    mask = (df_p["timestamp"] >= 10.0) & (df_p["timestamp"] <= 50.0)
    rpm_diff = df_b.loc[mask, "rpm"].values - df_p.loc[mask, "rpm"].values
    expected = 100.0 * 0.8  # 80 RPM
    np.testing.assert_allclose(rpm_diff, expected, atol=1.0,
                               err_msg="Sensor bias not applied correctly alongside physical fault")


# ────────────────────────────────────────────────────────────────────────────
# Test P — Sequential mission reset
# ────────────────────────────────────────────────────────────────────────────

def test_p_sequential_mission_reset():
    """
    Test P: Run two missions on the same simulator instance.
    Verify that STUCK/DRIFT sensor-fault state from mission 1
    does NOT leak into mission 2.
    """
    sim = EngineSimulator(seed=42)
    profile = _make_cruise_profile(30.0)

    # Mission 1: stuck sensor on RPM
    stuck_fault = _make_sensor_fault(
        channel="rpm", mode="stuck", severity=1.0,
        start_time=5.0, end_time=25.0,
    )
    df1 = sim.run_to_dataframe(profile, dt=1.0, fault_schedule=stuck_fault)

    # Verify stuck is active in mission 1
    mask1 = (df1["timestamp"] >= 5.0) & (df1["timestamp"] <= 25.0)
    stuck_vals_1 = df1.loc[mask1, "rpm"].values
    assert np.std(stuck_vals_1) < 0.01, "Mission 1 stuck not working"

    # Mission 2: same simulator, NO sensor fault
    # run() calls reset() internally, which must clear sensor fault state
    df2 = sim.run_to_dataframe(profile, dt=1.0, fault_schedule=None)

    # Mission 2 RPM should not be stuck
    mask2 = (df2["timestamp"] >= 5.0) & (df2["timestamp"] <= 25.0)
    rpm_vals_2 = df2.loc[mask2, "rpm"].values

    # RPM should vary (not latched to any prior value)
    # For cruise it may be relatively stable, but certainly not identical to stuck_value
    # The key check is that the processor was reset (no latch leakage)
    assert len(sim.telemetry_gen.sensor_fault_processor._stuck_latches) == 0, \
        "Stuck latches not cleared after reset"


def test_p2_sequential_drift_reset():
    """
    Test P2: Drift sensor fault state is cleared between sequential missions.
    The processor should have no leftover state from a prior mission.
    Verify by running two missions: the second must still exhibit drift
    behavior (not carry forward accumulated drift from mission 1).
    """
    sim = EngineSimulator(seed=42)
    profile = _make_cruise_profile(30.0)

    # Mission 1: drift on CHT
    drift_fault = _make_sensor_fault(
        channel="cht", mode="drift", severity=1.0,
        start_time=5.0, end_time=25.0,
        drift_rate=5.0,
    )
    df1 = sim.run_to_dataframe(profile, dt=1.0, fault_schedule=drift_fault)

    # Mission 2: same drift fault, fresh run
    df2 = sim.run_to_dataframe(profile, dt=1.0, fault_schedule=drift_fault)

    # Key invariant: sensor fault processor state is cleared after reset
    assert len(sim.telemetry_gen.sensor_fault_processor._stuck_latches) == 0, \
        "Sensor fault state not cleared between missions"

    # Both missions should show increasing drift during the fault window
    # (drift = drift_rate * severity * elapsed_from_start_time)
    # At t=10 (elapsed=5s): drift ≈ 25°C, at t=20 (elapsed=15s): drift ≈ 75°C
    mask_early = (df2["timestamp"] >= 9.5) & (df2["timestamp"] <= 10.5)
    mask_late = (df2["timestamp"] >= 19.5) & (df2["timestamp"] <= 20.5)
    if mask_early.any() and mask_late.any():
        cht_early = df2.loc[mask_early, "cht"].mean()
        cht_late = df2.loc[mask_late, "cht"].mean()
        # Drift should cause later values to be higher (drift adds positive offset)
        assert cht_late > cht_early, \
            "Mission 2 drift not behaving correctly — possible state leakage"


# ────────────────────────────────────────────────────────────────────────────
# Test — RNG neutrality (no sensor fault active → zero RNG draws)
# ────────────────────────────────────────────────────────────────────────────

def test_rng_neutral_no_sensor_fault():
    """
    Verify that when no sensor fault is active, the SensorFaultProcessor
    consumes zero RNG draws, so golden healthy telemetry is bitwise unchanged.
    """
    sim_clean = EngineSimulator(seed=42)
    sim_inactive = EngineSimulator(seed=42)

    profile = _make_cruise_profile(30.0)

    # No fault
    df_clean = sim_clean.run_to_dataframe(profile, dt=1.0)

    # Fault exists in schedule but severity=0 (never active)
    inactive_fault = _make_sensor_fault(
        channel="rpm", mode="noise", severity=0.0,
        start_time=5.0, end_time=25.0,
    )
    df_inactive = sim_inactive.run_to_dataframe(profile, dt=1.0,
                                                fault_schedule=inactive_fault)

    # Must be bitwise identical
    for col in ["rpm", "cht", "egt", "oil_temp", "oil_pressure", "fuel_flow", "vibration"]:
        np.testing.assert_array_equal(
            df_clean[col].values, df_inactive[col].values,
            err_msg=f"Inactive sensor fault altered {col} (RNG consumed)"
        )


def test_rng_neutral_future_fault():
    """
    Verify that a sensor fault scheduled in the future (not yet active)
    does not consume RNG draws for the current timestep.
    """
    sim_clean = EngineSimulator(seed=42)
    sim_future = EngineSimulator(seed=42)

    profile = _make_cruise_profile(20.0)

    # Fault starts after the mission ends
    future_fault = _make_sensor_fault(
        channel="rpm", mode="noise", severity=1.0,
        start_time=100.0, end_time=200.0,
    )

    df_clean = sim_clean.run_to_dataframe(profile, dt=1.0)
    df_future = sim_future.run_to_dataframe(profile, dt=1.0,
                                            fault_schedule=future_fault)

    for col in ["rpm", "cht", "egt", "oil_temp", "oil_pressure", "fuel_flow", "vibration"]:
        np.testing.assert_array_equal(
            df_clean[col].values, df_future[col].values,
            err_msg=f"Future sensor fault altered {col}"
        )
