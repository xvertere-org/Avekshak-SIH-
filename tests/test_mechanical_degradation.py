"""
Unit and Validation Tests for Phase 4E: Mechanical Degradation + Vibration Faults.

Covers:
- Test A: Healthy equivalence (no fault vs severity=0 produce identical trajectories)
- Test B: Fault activation (severity > 0 increases vibration RMS)
- Test C: Vibration RMS monotonicity (severity sweep strictly increases vibration RMS)
- Test D: 1x frequency tied to RPM/60
- Test E: 2x frequency tied to 2*RPM/60
- Test F: Fault-window isolation (nominal before, degraded during, recovering after)
- Test G: Schedule gating (fault labels active only in [start_time, end_time])
- Test H: Numerical stability (no NaNs, Infs, bounded at max severity)
- Test I: Seeded determinism (identical seeds produce identical results)
- Test J: No-fault regression (existing nominal trajectory unchanged)
- Test K: Subsystem-level vibration parameter verification
- Test L: Secondary friction effect (slight RPM reduction at max severity)
- Test M: Severity sweep validation table
"""

import math
import numpy as np
import pytest
from simulator.engine_simulator import EngineSimulator
from simulator.config import SimulatorConfig, TierCParameters, TierDParameters
from simulator.fault_interface import FaultState, FaultType
from simulator.subsystems.mission import MissionProfile, PhaseSegment, FlightPhase
from simulator.subsystems.vibration import VibrationSystem


def test_a_healthy_equivalence():
    """
    Test A: Verify that a simulation with no fault and a simulation with
    mechanical degradation severity=0.0 produce identical physical trajectories.
    """
    sim_none = EngineSimulator(seed=42)
    sim_zero = EngineSimulator(seed=42)

    seg = PhaseSegment(FlightPhase.CRUISE, 40.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    fault_zero = FaultState(
        fault_type=FaultType.MECHANICAL_DEGRADATION,
        severity=0.0,
        start_time=10.0,
        end_time=30.0,
    )

    df_none = sim_none.run_to_dataframe(profile, dt=0.5, fault_schedule=None)
    df_zero = sim_zero.run_to_dataframe(profile, dt=0.5, fault_schedule=fault_zero)

    assert len(df_none) == len(df_zero)
    np.testing.assert_allclose(df_none["vibration"].values, df_zero["vibration"].values, rtol=1e-5)
    np.testing.assert_allclose(df_none["rpm"].values, df_zero["rpm"].values, rtol=1e-5)
    np.testing.assert_allclose(df_none["cht"].values, df_zero["cht"].values, rtol=1e-5)
    assert (df_zero["fault_type"] == "none").all()
    assert (df_zero["fault_severity"] == 0.0).all()


def test_b_fault_activation():
    """
    Test B: Verify that active mechanical degradation with severity > 0
    measurably increases vibration RMS.
    """
    sim_healthy = EngineSimulator(seed=42)
    sim_faulted = EngineSimulator(seed=42)

    seg = PhaseSegment(FlightPhase.CRUISE, 80.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    fault = FaultState(
        fault_type=FaultType.MECHANICAL_DEGRADATION,
        severity=0.6,
        start_time=15.0,
        end_time=80.0,
    )

    df_h = sim_healthy.run_to_dataframe(profile, dt=0.5)
    df_f = sim_faulted.run_to_dataframe(profile, dt=0.5, fault_schedule=fault)

    # Before fault (t = 10s): identical vibration
    v_h_pre = df_h.loc[df_h["timestamp"] == 10.0, "vibration"].values[0]
    v_f_pre = df_f.loc[df_f["timestamp"] == 10.0, "vibration"].values[0]
    assert v_h_pre == pytest.approx(v_f_pre, abs=1e-3)

    # During fault (t = 70s): faulted vibration is significantly higher
    v_h_during = df_h.loc[df_h["timestamp"] == 70.0, "vibration"].values[0]
    v_f_during = df_f.loc[df_f["timestamp"] == 70.0, "vibration"].values[0]
    assert v_f_during > v_h_during * 1.3  # At least 30% increase at severity 0.6


def test_c_vibration_rms_monotonicity():
    """
    Test C: Verify that increasing mechanical degradation severity monotonically
    increases vibration RMS across a severity sweep.
    """
    severities = [0.0, 0.25, 0.50, 0.75, 1.0]
    vib_rms_values = []

    seg = PhaseSegment(FlightPhase.CRUISE, 60.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    for sev in severities:
        sim = EngineSimulator(seed=42)
        fault = FaultState(
            fault_type=FaultType.MECHANICAL_DEGRADATION,
            severity=sev,
            start_time=5.0,
            end_time=60.0,
        )
        df = sim.run_to_dataframe(profile, dt=0.5, fault_schedule=fault)
        # Use mean vibration in the last 10s (steady-state window)
        vib_rms_values.append(float(df[df["timestamp"] >= 50.0]["vibration"].mean()))

    # Verify strict monotonic increase
    dv = np.diff(vib_rms_values)
    assert np.all(dv > 0.0), f"Vibration RMS did not strictly increase: {vib_rms_values}"


def test_d_order_1x_frequency():
    """
    Test D: Verify that the 1x rotational component frequency equals RPM/60
    regardless of mechanical degradation severity.
    """
    rpm_test = 4500.0
    expected_f1 = rpm_test / 60.0

    vib_sys = VibrationSystem(rng=np.random.default_rng(42))
    state = vib_sys.step(
        rpm=rpm_test,
        load_pct=75.0,
        dt=0.1,
        mechanical_condition=2.8,  # severity=1.0 condition
        mechanical_noise_factor=3.5,
    )

    assert state.order_1x_freq_hz == pytest.approx(expected_f1, rel=1e-6)


def test_e_order_2x_frequency():
    """
    Test E: Verify that the 2x rotational component frequency equals 2*RPM/60
    regardless of mechanical degradation severity.
    """
    rpm_test = 4500.0
    expected_f2 = 2.0 * rpm_test / 60.0

    vib_sys = VibrationSystem(rng=np.random.default_rng(42))
    state = vib_sys.step(
        rpm=rpm_test,
        load_pct=75.0,
        dt=0.1,
        mechanical_condition=2.8,
        mechanical_noise_factor=3.5,
    )

    assert state.order_2x_freq_hz == pytest.approx(expected_f2, rel=1e-6)


def test_f_fault_window_isolation():
    """
    Test F: Verify fault-window isolation for vibration response:
    - Before start_time -> nominal vibration
    - During fault window -> degraded (elevated) vibration
    - After end_time -> nominal/recovering vibration
    """
    sim_faulted = EngineSimulator(seed=42)
    sim_healthy = EngineSimulator(seed=42)

    seg = PhaseSegment(FlightPhase.CRUISE, 120.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    fault = FaultState(
        fault_type=FaultType.MECHANICAL_DEGRADATION,
        severity=0.8,
        start_time=30.0,
        end_time=70.0,
    )

    df_f = sim_faulted.run_to_dataframe(profile, dt=0.5, fault_schedule=fault)
    df_h = sim_healthy.run_to_dataframe(profile, dt=0.5)

    # Before fault window (t < 30): vibration should be identical to healthy
    pre_f = df_f[df_f["timestamp"] < 30.0]["vibration"].values
    pre_h = df_h[df_h["timestamp"] < 30.0]["vibration"].values
    np.testing.assert_allclose(pre_f, pre_h, rtol=1e-5,
                               err_msg="Vibration before fault window should be identical to healthy")

    # During fault window (30 <= t <= 70): vibration should be elevated
    during_f = df_f[(df_f["timestamp"] >= 30.0) & (df_f["timestamp"] <= 70.0)]["vibration"].mean()
    during_h = df_h[(df_h["timestamp"] >= 30.0) & (df_h["timestamp"] <= 70.0)]["vibration"].mean()
    assert during_f > during_h * 1.3, (
        f"Vibration during fault ({during_f:.4f}) should be significantly higher "
        f"than healthy ({during_h:.4f})"
    )

    # After fault window (t > 70): vibration should recover toward nominal
    # Since vibration has no thermal-like inertia, recovery is immediate
    post_f = df_f[df_f["timestamp"] > 75.0]["vibration"].values
    post_h = df_h[df_h["timestamp"] > 75.0]["vibration"].values
    mean_post_f = np.mean(post_f)
    mean_post_h = np.mean(post_h)
    # Should be within 20% of nominal (RNG state divergence causes some difference)
    assert abs(mean_post_f - mean_post_h) / mean_post_h < 0.20, (
        f"Vibration after fault ({mean_post_f:.4f}) should recover close to nominal ({mean_post_h:.4f})"
    )


def test_g_schedule_gating():
    """
    Test G: Verify that mechanical degradation fault labels are active
    strictly inside scheduled window [start_time, end_time].
    """
    sim = EngineSimulator(seed=42)
    seg = PhaseSegment(FlightPhase.CRUISE, 100.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    fault = FaultState(
        fault_type=FaultType.MECHANICAL_DEGRADATION,
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
    assert (during["fault_type"] == "mechanical_degradation").all()
    assert (during["fault_severity"] == 0.7).all()

    # After end_time (t > 65)
    post = df[df["timestamp"] > 65.0]
    assert (post["fault_type"] == "none").all()
    assert (post["fault_severity"] == 0.0).all()


def test_h_numerical_stability():
    """
    Test H: Verify that maximum mechanical degradation produces no NaNs, Infs,
    or negative impossible values across multi-minute simulation.
    """
    sim = EngineSimulator(seed=42)
    seg = PhaseSegment(FlightPhase.CRUISE, 180.0, 100.0, 100.0, 3000.0, 3000.0)
    profile = MissionProfile(segments=[seg])

    max_fault = FaultState(
        fault_type=FaultType.MECHANICAL_DEGRADATION,
        severity=1.0,
        start_time=10.0,
        end_time=180.0,
    )

    df = sim.run_to_dataframe(profile, dt=0.2, fault_schedule=max_fault)

    assert not df.isna().any().any()
    assert not np.isinf(df[["vibration", "rpm", "cht", "oil_pressure"]].values).any()
    assert (df["vibration"] >= 0.0).all()
    assert (df["rpm"] > 500.0).all()
    assert (df["cht"] > 0.0).all()


def test_i_seeded_determinism():
    """
    Test I: Verify that seeded simulations with mechanical degradation
    produce identical results across repeated runs.
    """
    fault = FaultState(
        fault_type=FaultType.MECHANICAL_DEGRADATION,
        severity=0.5,
        start_time=10.0,
        end_time=40.0,
    )

    seg = PhaseSegment(FlightPhase.CRUISE, 50.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    sim_a = EngineSimulator(seed=42)
    df_a = sim_a.run_to_dataframe(profile, dt=0.5, fault_schedule=fault)

    sim_b = EngineSimulator(seed=42)
    df_b = sim_b.run_to_dataframe(profile, dt=0.5, fault_schedule=fault)

    np.testing.assert_array_equal(df_a["vibration"].values, df_b["vibration"].values)
    np.testing.assert_array_equal(df_a["rpm"].values, df_b["rpm"].values)


def test_j_no_fault_regression():
    """
    Test J: Verify that existing nominal trajectory is unchanged when no
    mechanical fault is active (regression guard for Phase 4E changes).
    """
    sim = EngineSimulator(seed=42)
    seg = PhaseSegment(FlightPhase.CRUISE, 30.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    df = sim.run_to_dataframe(profile, dt=0.5, fault_schedule=None)

    # These are the same checks used in other phase tests — RPM converges, vibration bounded
    final_rpm = df["rpm"].iloc[-1]
    assert 3500.0 < final_rpm < 6000.0

    final_vib = df["vibration"].iloc[-1]
    assert 0.0 < final_vib < 2.0

    # No fault labels
    assert (df["fault_type"] == "none").all()
    assert (df["fault_severity"] == 0.0).all()


def test_k_subsystem_vibration_scaling():
    """
    Test K: Verify VibrationSystem.step directly scales amplitudes
    with mechanical_condition and noise with mechanical_noise_factor.
    """
    tier_c = TierCParameters()
    tier_d = TierDParameters()

    sys_nom = VibrationSystem(tier_c=tier_c, tier_d=tier_d, rng=np.random.default_rng(42))
    sys_deg = VibrationSystem(tier_c=tier_c, tier_d=tier_d, rng=np.random.default_rng(42))

    rpm = 4500.0
    load_pct = 75.0

    state_nom = sys_nom.step(
        rpm=rpm, load_pct=load_pct, dt=0.1,
        mechanical_condition=1.0, mechanical_noise_factor=1.0,
    )
    state_deg = sys_deg.step(
        rpm=rpm, load_pct=load_pct, dt=0.1,
        mechanical_condition=2.8, mechanical_noise_factor=3.5,
    )

    # 1x amplitude should scale by mechanical_condition ratio (2.8 / 1.0)
    expected_amp_ratio = 2.8
    actual_amp_ratio = state_deg.amplitude_1x_g / state_nom.amplitude_1x_g
    assert pytest.approx(actual_amp_ratio, rel=1e-3) == expected_amp_ratio

    # 2x amplitude should also scale by same ratio
    actual_2x_ratio = state_deg.amplitude_2x_g / state_nom.amplitude_2x_g
    assert pytest.approx(actual_2x_ratio, rel=1e-3) == expected_amp_ratio

    # RMS should increase substantially
    assert state_deg.rms_g > state_nom.rms_g * 2.5

    # Frequencies should be identical (deterministic, not affected by condition)
    assert state_deg.order_1x_freq_hz == state_nom.order_1x_freq_hz
    assert state_deg.order_2x_freq_hz == state_nom.order_2x_freq_hz


def test_l_secondary_friction_effect():
    """
    Test L: Verify that maximum mechanical degradation produces a small
    but measurable RPM reduction (secondary friction effect).
    """
    sim_healthy = EngineSimulator(seed=42)
    sim_faulted = EngineSimulator(seed=42)

    seg = PhaseSegment(FlightPhase.CRUISE, 120.0, 75.0, 75.0, 2000.0, 2000.0)
    profile = MissionProfile(segments=[seg])

    fault = FaultState(
        fault_type=FaultType.MECHANICAL_DEGRADATION,
        severity=1.0,
        start_time=10.0,
        end_time=120.0,
    )

    df_h = sim_healthy.run_to_dataframe(profile, dt=0.5)
    df_f = sim_faulted.run_to_dataframe(profile, dt=0.5, fault_schedule=fault)

    rpm_h = df_h[df_h["timestamp"] >= 100.0]["rpm"].mean()
    rpm_f = df_f[df_f["timestamp"] >= 100.0]["rpm"].mean()

    # Friction effect should produce a small but measurable RPM drop (< 5%)
    rpm_drop_pct = (rpm_h - rpm_f) / rpm_h * 100.0
    assert rpm_drop_pct > 0.1, f"Expected measurable RPM drop, got {rpm_drop_pct:.3f}%"
    assert rpm_drop_pct < 5.0, f"RPM drop too large ({rpm_drop_pct:.3f}%), secondary effect should be small"


def test_m_severity_sweep_validation():
    """
    Test M: Comprehensive severity sweep validation table.
    Verify vibration RMS, 1x amplitude, 2x amplitude at representative condition.
    """
    severities = [0.0, 0.25, 0.50, 0.75, 1.0]
    tier_c = TierCParameters()
    tier_d = TierDParameters()

    rpm = 4500.0
    load_pct = 75.0
    load_norm = load_pct / 100.0

    rms_values = []
    amp_1x_values = []
    amp_2x_values = []

    for sev in severities:
        m_cond = 1.0 + tier_c.k_mech_vib_gain * sev
        noise_factor = 1.0 + tier_c.k_mech_noise_gain * sev

        vib_sys = VibrationSystem(tier_c=tier_c, tier_d=tier_d, rng=np.random.default_rng(42))
        state = vib_sys.step(
            rpm=rpm, load_pct=load_pct, dt=0.1,
            mechanical_condition=m_cond, mechanical_noise_factor=noise_factor,
        )

        rms_values.append(state.rms_g)
        amp_1x_values.append(state.amplitude_1x_g)
        amp_2x_values.append(state.amplitude_2x_g)

    # RMS strictly increases with severity
    assert np.all(np.diff(rms_values) > 0.0), f"RMS not monotonic: {rms_values}"

    # 1x amplitude strictly increases with severity
    assert np.all(np.diff(amp_1x_values) > 0.0), f"1x amplitude not monotonic: {amp_1x_values}"

    # 2x amplitude strictly increases with severity
    assert np.all(np.diff(amp_2x_values) > 0.0), f"2x amplitude not monotonic: {amp_2x_values}"

    # Verify expected scaling at severity=1.0
    expected_condition_max = 1.0 + tier_c.k_mech_vib_gain  # 2.8
    amp_1x_base = (tier_c.vib_order1_base_g + tier_c.vib_load_gain * load_norm)
    expected_1x_max = amp_1x_base * expected_condition_max
    assert pytest.approx(amp_1x_values[-1], rel=1e-3) == expected_1x_max

    # Verify frequency is unchanged across all severities
    f1 = rpm / 60.0
    f2 = 2.0 * rpm / 60.0
    for sev in severities:
        m_cond = 1.0 + tier_c.k_mech_vib_gain * sev
        vib_sys = VibrationSystem(tier_c=tier_c, tier_d=tier_d, rng=np.random.default_rng(42))
        state = vib_sys.step(
            rpm=rpm, load_pct=load_pct, dt=0.1,
            mechanical_condition=m_cond,
        )
        assert state.order_1x_freq_hz == pytest.approx(f1, rel=1e-6)
        assert state.order_2x_freq_hz == pytest.approx(f2, rel=1e-6)


def test_severity_validation():
    """Verify FaultState rejects severity outside [0, 1] for MECHANICAL_DEGRADATION."""
    with pytest.raises(ValueError, match="severity"):
        FaultState(
            fault_type=FaultType.MECHANICAL_DEGRADATION,
            severity=1.5,
        )
    with pytest.raises(ValueError, match="severity"):
        FaultState(
            fault_type=FaultType.MECHANICAL_DEGRADATION,
            severity=-0.1,
        )
