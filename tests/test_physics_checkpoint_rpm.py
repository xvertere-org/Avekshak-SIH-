"""
Checkpoint verification tests for Atmosphere, Mission, Operating Point, and RPM dynamics.
"""

import pytest
import math
from simulator.config import SimulatorConfig, TierAParameters, TierCParameters, TierDParameters
from simulator.subsystems.atmosphere import Atmosphere
from simulator.subsystems.mission import MissionProfile, FlightPhase
from simulator.subsystems.dynamics import RotationalDynamics


def test_atmosphere_sea_level_and_lapse():
    """Verify sea-level standard ISA values and monotonic decrease with altitude."""
    atmo = Atmosphere()
    sl = atmo.compute(altitude_m=0.0, temp_offset_k=0.0)

    assert pytest.approx(sl.temperature_c, abs=0.1) == 15.0
    assert pytest.approx(sl.pressure_pa, abs=10.0) == 101325.0
    assert pytest.approx(sl.density_kg_m3, abs=0.01) == 1.225
    assert pytest.approx(sl.density_factor, abs=0.001) == 1.0

    # 3000m altitude
    alt3k = atmo.compute(altitude_m=3000.0, temp_offset_k=0.0)
    assert alt3k.pressure_pa < sl.pressure_pa
    assert alt3k.density_kg_m3 < sl.density_kg_m3
    assert alt3k.density_factor < 1.0
    assert alt3k.temperature_c < sl.temperature_c


def test_mission_step_generation():
    """Verify MissionProfile step generation and time progression."""
    profile = MissionProfile()
    dt = 0.5
    steps = list(profile.generate_steps(dt=dt))

    assert len(steps) > 100
    assert steps[0].phase == FlightPhase.TAKEOFF
    assert steps[-1].phase == FlightPhase.LANDING
    assert steps[0].timestamp_s == 0.0
    assert steps[1].timestamp_s == dt


def test_rpm_dynamics_throttle_response_and_convergence():
    """Verify throttle step increases RPM and reaches stable convergence without NaN/Inf."""
    config = SimulatorConfig()
    dynamics = RotationalDynamics(
        tier_a=config.tier_a,
        tier_c=config.tier_c,
        tier_d=config.tier_d,
        initial_rpm=1400.0,
    )

    dt = 0.05
    # Simulate 10 seconds at idle (0% throttle)
    for _ in range(int(10.0 / dt)):
        op = dynamics.step(throttle_pct=0.0, density_factor=1.0, dt=dt)
        assert not math.isnan(op.rpm)
        assert not math.isinf(op.rpm)

    idle_rpm = dynamics.omega_to_rpm(dynamics.omega)
    assert pytest.approx(idle_rpm, abs=150.0) == 1400.0

    # Throttle step to 100%
    for _ in range(int(15.0 / dt)):
        op = dynamics.step(throttle_pct=100.0, density_factor=1.0, dt=dt)
        assert not math.isnan(op.rpm)
        assert not math.isinf(op.rpm)

    full_rpm = dynamics.omega_to_rpm(dynamics.omega)
    assert full_rpm > idle_rpm
    # Full power RPM should reach ~5400 - 5600 RPM
    assert 5200.0 <= full_rpm <= 5700.0


def test_altitude_power_derating():
    """Verify higher altitude (lower density) reduces available power and steady-state RPM."""
    config = SimulatorConfig()
    atmo = Atmosphere(tier_a=config.tier_a)

    sigma_sl = atmo.density_factor(0.0)
    sigma_3k = atmo.density_factor(3000.0)
    assert sigma_3k < sigma_sl

    dyn_sl = RotationalDynamics(config.tier_a, config.tier_c, config.tier_d, initial_rpm=5000.0)
    dyn_3k = RotationalDynamics(config.tier_a, config.tier_c, config.tier_d, initial_rpm=5000.0)

    dt = 0.05
    for _ in range(int(10.0 / dt)):
        op_sl = dyn_sl.step(throttle_pct=90.0, density_factor=sigma_sl, dt=dt)
        op_3k = dyn_3k.step(throttle_pct=90.0, density_factor=sigma_3k, dt=dt)

    assert op_3k.power_target_w < op_sl.power_target_w
    assert op_3k.rpm < op_sl.rpm


def test_multi_timestep_numerical_stability():
    """Verify stability across multiple integration time steps."""
    for dt in [0.01, 0.05, 0.1, 0.2]:
        config = SimulatorConfig()
        dyn = RotationalDynamics(config.tier_a, config.tier_c, config.tier_d, initial_rpm=1400.0)
        for _ in range(int(5.0 / dt)):
            op = dyn.step(throttle_pct=80.0, density_factor=1.0, dt=dt)
            assert not math.isnan(op.rpm)
            assert not math.isinf(op.rpm)
            assert 1000.0 <= op.rpm <= 6000.0
