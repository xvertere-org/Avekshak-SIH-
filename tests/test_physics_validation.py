"""
Comprehensive Physics Validation Test Suite for Phase 2B Engine Simulator.
"""

import pytest
import math
import numpy as np
import pandas as pd

from simulator.config import SimulatorConfig, TierAParameters, TierCParameters, TierDParameters
from simulator.subsystems.atmosphere import Atmosphere
from simulator.subsystems.mission import MissionProfile, FlightPhase, PhaseSegment
from simulator.subsystems.dynamics import RotationalDynamics
from simulator.subsystems.fuel import FuelSystem
from simulator.subsystems.thermal import ThermalSystem
from simulator.subsystems.lubrication import LubricationSystem
from simulator.subsystems.vibration import VibrationSystem
from simulator.engine_simulator import EngineSimulator
from telemetry.schema import TelemetryRecord, MissionConfig


def test_atmosphere_isa_physics():
    """Verify ISA atmosphere equations and physical relationships."""
    atmo = Atmosphere()
    sl = atmo.compute(0.0)
    assert pytest.approx(sl.temperature_c, abs=0.01) == 15.0
    assert pytest.approx(sl.pressure_pa, abs=5.0) == 101325.0
    assert pytest.approx(sl.density_kg_m3, abs=0.005) == 1.225
    assert pytest.approx(sl.density_factor, abs=0.001) == 1.0

    # Monotonic altitude tests
    h_levels = [0.0, 1000.0, 3000.0, 5000.0, 8000.0]
    densities = [atmo.compute(h).density_kg_m3 for h in h_levels]
    pressures = [atmo.compute(h).pressure_pa for h in h_levels]
    temps = [atmo.compute(h).temperature_k for h in h_levels]

    assert all(densities[i] > densities[i + 1] for i in range(len(densities) - 1))
    assert all(pressures[i] > pressures[i + 1] for i in range(len(pressures) - 1))
    assert all(temps[i] > temps[i + 1] for i in range(len(temps) - 1))


def test_operating_point_power_monotonicity():
    """Verify target power increases monotonically with throttle."""
    dyn = RotationalDynamics()
    p_25 = dyn._torque_derivatives(omega=dyn.rpm_to_omega(4000.0), throttle_pct=25.0, density_factor=1.0)[0]
    p_50 = dyn._torque_derivatives(omega=dyn.rpm_to_omega(4000.0), throttle_pct=50.0, density_factor=1.0)[0]
    p_75 = dyn._torque_derivatives(omega=dyn.rpm_to_omega(4000.0), throttle_pct=75.0, density_factor=1.0)[0]
    p_100 = dyn._torque_derivatives(omega=dyn.rpm_to_omega(4000.0), throttle_pct=100.0, density_factor=1.0)[0]

    assert p_25 < p_50 < p_75 < p_100


def test_fuel_flow_properties():
    """Verify fuel flow increases with power and remains non-negative."""
    fuel_sys = FuelSystem()
    f_idle = fuel_sys.compute(0.0)
    f_cruise = fuel_sys.compute(40000.0)
    f_max = fuel_sys.compute(58000.0)

    assert f_idle.volumetric_flow_l_h > 0.0
    assert f_cruise.volumetric_flow_l_h > f_idle.volumetric_flow_l_h
    assert f_max.volumetric_flow_l_h > f_cruise.volumetric_flow_l_h
    assert f_max.mass_flow_kg_s > 0.0
    assert 200.0 <= f_max.bsfc_g_kwh <= 320.0  # realistic aero-piston BSFC band


def test_thermal_cht_and_egt_dynamics():
    """Verify thermal responses to load, ambient temperature, and convergence."""
    thermal = ThermalSystem()
    dt = 0.1

    # Steady state run at low load
    for _ in range(int(30.0 / dt)):
        state_idle = thermal.step(
            rpm=1400.0,
            load_pct=15.0,
            fuel_mass_flow_kg_s=0.0004,
            density_factor=1.0,
            ambient_temp_c=15.0,
            airspeed_ms=20.0,
            dt=dt,
        )

    # Steady state run at high load
    for _ in range(int(60.0 / dt)):
        state_full = thermal.step(
            rpm=5500.0,
            load_pct=100.0,
            fuel_mass_flow_kg_s=0.0045,
            density_factor=1.0,
            ambient_temp_c=15.0,
            airspeed_ms=45.0,
            dt=dt,
        )

    assert state_full.cht_c > state_idle.cht_c
    assert state_full.egt_c > state_idle.egt_c
    assert state_full.cht_c <= 135.0  # Within nominal design envelope
    assert 600.0 <= state_full.egt_c <= 850.0


def test_oil_pressure_and_temperature_relations():
    """Verify oil pressure increases with RPM and decreases with oil temperature."""
    lub = LubricationSystem()
    dt = 0.1

    # 1. RPM effect at constant oil temperature
    lub.set_oil_temp(80.0)
    state_low_rpm = lub.step(rpm=1800.0, cht_c=100.0, fuel_mass_flow_kg_s=0.001, ambient_temp_c=15.0, dt=dt)
    lub.set_oil_temp(80.0)
    state_high_rpm = lub.step(rpm=5200.0, cht_c=100.0, fuel_mass_flow_kg_s=0.001, ambient_temp_c=15.0, dt=dt)

    assert state_high_rpm.oil_pressure_bar > state_low_rpm.oil_pressure_bar

    # 2. Temperature/viscosity effect at constant RPM
    lub.set_oil_temp(60.0)
    state_cold = lub.step(rpm=4000.0, cht_c=100.0, fuel_mass_flow_kg_s=0.002, ambient_temp_c=15.0, dt=dt)
    lub.set_oil_temp(115.0)
    state_hot = lub.step(rpm=4000.0, cht_c=100.0, fuel_mass_flow_kg_s=0.002, ambient_temp_c=15.0, dt=dt)

    assert state_cold.oil_pressure_bar > state_hot.oil_pressure_bar
    assert state_hot.oil_pressure_bar >= 0.8  # Stays above minimum limit


def test_vibration_synthesis_and_fft_order_peak():
    """
    Verify vibration waveform synthesis and FFT spectral peak at expected rotational orders:
    dominant_frequency ≈ order * (RPM / 60)
    """
    vib_sys = VibrationSystem()
    test_rpm = 3000.0  # 3000 RPM -> 1x = 50 Hz, 2x = 100 Hz
    duration_s = 2.0
    fs = 1000.0  # 1000 Hz sampling rate

    t, signal, f1, f2 = vib_sys.generate_waveform(
        rpm=test_rpm,
        load_pct=75.0,
        duration_s=duration_s,
        sampling_rate_hz=fs,
    )

    assert pytest.approx(f1, abs=0.01) == 50.0
    assert pytest.approx(f2, abs=0.01) == 100.0

    # Compute FFT
    n = len(signal)
    fft_vals = np.abs(np.fft.rfft(signal)) * (2.0 / n)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    # Find top 2 spectral peaks (ignoring DC at 0 Hz)
    pos_mask = freqs > 5.0
    pos_freqs = freqs[pos_mask]
    pos_fft = fft_vals[pos_mask]

    peak_indices = np.argsort(pos_fft)[-2:]
    detected_peak_freqs = sorted(pos_freqs[peak_indices])

    # Check that peaks match 50 Hz (1x) and 100 Hz (2x) within 1 Hz resolution
    assert any(abs(pf - 50.0) < 1.5 for pf in detected_peak_freqs)
    assert any(abs(pf - 100.0) < 1.5 for pf in detected_peak_freqs)


def test_seeded_reproducibility():
    """Verify that identical seed + config produces identical telemetry records."""
    sim1 = EngineSimulator(seed=123)
    sim2 = EngineSimulator(seed=123)
    sim3 = EngineSimulator(seed=999)

    profile = MissionProfile(
        segments=[
            PhaseSegment(
                phase=FlightPhase.CRUISE,
                duration_s=10.0,
                throttle_start_pct=75.0,
                throttle_end_pct=75.0,
                altitude_start_m=2000.0,
                altitude_end_m=2000.0,
            )
        ]
    )

    recs1 = sim1.run(profile, dt=0.2)
    recs2 = sim2.run(profile, dt=0.2)
    recs3 = sim3.run(profile, dt=0.2)

    assert len(recs1) == len(recs2) == len(recs3)

    # Recs 1 and 2 must match exactly
    for r1, r2 in zip(recs1, recs2):
        assert r1.rpm == r2.rpm
        assert r1.cht == r2.cht
        assert r1.oil_pressure == r2.oil_pressure
        assert r1.vibration == r2.vibration

    # Recs 3 with different seed should have different sensor noise samples
    different_samples = any(r1.vibration != r3.vibration for r1, r3 in zip(recs1, recs3))
    assert different_samples


def test_multi_timestep_simulator_stability():
    """Verify simulator operates stably across multiple integration timesteps."""
    timesteps = [0.05, 0.1, 0.2, 0.5]
    for dt in timesteps:
        sim = EngineSimulator(seed=42)
        profile = MissionProfile(
            segments=[
                PhaseSegment(FlightPhase.TAKEOFF, 5.0, 100.0, 100.0, 0.0, 100.0),
                PhaseSegment(FlightPhase.CRUISE, 10.0, 75.0, 75.0, 2000.0, 2000.0),
            ]
        )
        records = sim.run(profile, dt=dt)
        for r in records:
            assert not math.isnan(r.rpm)
            assert not math.isnan(r.cht)
            assert not math.isnan(r.egt)
            assert not math.isnan(r.oil_pressure)
            assert not math.isnan(r.vibration)
            assert 1000.0 <= r.rpm <= 6000.0


def test_batch_and_streaming_consistency():
    """Verify batch run and discrete streaming step use identical underlying physics state updates."""
    profile = MissionProfile(
        segments=[
            PhaseSegment(FlightPhase.TAKEOFF, 5.0, 95.0, 95.0, 0.0, 50.0),
            PhaseSegment(FlightPhase.CLIMB, 5.0, 85.0, 85.0, 50.0, 500.0),
        ]
    )

    # Streaming mode
    sim_stream = EngineSimulator(seed=42)
    stream_records = []
    dt = 0.2
    for step in profile.generate_steps(dt=dt):
        rec = sim_stream.step(mission_input=step, time_step=dt)
        stream_records.append(rec)

    # Batch mode
    sim_batch = EngineSimulator(seed=42)
    batch_records = sim_batch.run(profile, dt=dt)

    assert len(stream_records) == len(batch_records)
    for r_s, r_b in zip(stream_records, batch_records):
        assert r_s.timestamp == r_b.timestamp
        assert r_s.rpm == r_b.rpm
        assert r_s.cht == r_b.cht
        assert r_s.oil_pressure == r_b.oil_pressure
