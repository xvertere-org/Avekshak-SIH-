"""
Rotational Dynamics and Operating Point Model for SIH26054.

Implements engine power target, load torque, friction model,
and 4th-order Runge-Kutta (RK4) numerical integration for crankshaft speed (RPM).
"""

import math
from typing import NamedTuple, Optional
from simulator.config import TierAParameters, TierCParameters, TierDParameters


class OperatingPoint(NamedTuple):
    """Calculated engine operating state at current time step."""
    rpm: float
    omega_rad_s: float
    throttle_pct: float
    density_factor: float
    power_target_w: float
    torque_engine_nm: float
    torque_load_nm: float
    torque_friction_nm: float
    net_torque_nm: float
    domega_dt: float
    engine_load_pct: float


class RotationalDynamics:
    """
    4th-Order Runge-Kutta (RK4) Rotational Speed Simulator.
    """

    def __init__(
        self,
        tier_a: TierAParameters = TierAParameters(),
        tier_c: TierCParameters = TierCParameters(),
        tier_d: TierDParameters = TierDParameters(),
        initial_rpm: Optional[float] = None,
    ):
        self.tier_a = tier_a
        self.tier_c = tier_c
        self.tier_d = tier_d

        # State variable: angular velocity omega in rad/s
        init_rpm = initial_rpm if initial_rpm is not None else self.tier_c.rpm_idle
        self.omega = self.rpm_to_omega(init_rpm)

        # Precalculate idle assist torque to balance idle resistance at calibrated idle RPM
        omega_idle = self.rpm_to_omega(self.tier_c.rpm_idle)
        t_load_idle = self.tier_c.k_load * (omega_idle ** 2)
        t_fric_idle = self.tier_c.k_fric_linear * omega_idle + self.tier_c.torque_fric_static
        self._idle_torque_balance = t_load_idle + t_fric_idle

    @staticmethod
    def rpm_to_omega(rpm: float) -> float:
        """Convert RPM to rad/s."""
        return 2.0 * math.pi * max(0.0, rpm) / 60.0

    @staticmethod
    def omega_to_rpm(omega: float) -> float:
        """Convert rad/s to RPM."""
        return max(0.0, omega) * 60.0 / (2.0 * math.pi)

    def compute_rpm_efficiency(self, rpm: float) -> float:
        """
        Compute normalized RPM efficiency multiplier.
        Tier D polynomial assumption peaking near continuous rated speed (5500 RPM).
        """
        norm_rpm = rpm / self.tier_a.rpm_max_continuous
        a, b, c = self.tier_d.rpm_eff_poly
        eff = a * (norm_rpm ** 2) + b * norm_rpm + c
        return max(0.35, min(1.05, eff))

    def _torque_derivatives(
        self,
        omega: float,
        throttle_pct: float,
        density_factor: float,
        combustion_efficiency_factor: float = 1.0,
        friction_factor: float = 1.0,
    ) -> tuple:
        """
        Calculate instantaneous torques and domega/dt at given state.
        Supports Phase 4D power degradation via combustion_efficiency_factor.
        Supports Phase 4E mechanical degradation via friction_factor.
        """
        rpm = self.omega_to_rpm(omega)
        throttle_norm = max(0.0, min(100.0, throttle_pct)) / 100.0
        eff = self.compute_rpm_efficiency(rpm)

        # Indicated power target: P_max * throttle * density * efficiency * combustion_eff
        comb_factor = max(0.1, float(combustion_efficiency_factor))
        p_combustion = self.tier_a.power_max_continuous_w * throttle_norm * density_factor * eff * comb_factor

        # Indicated engine torque with smooth idle circuit assist at low throttle
        omega_safe = max(omega, 10.0)
        t_combustion = p_combustion / omega_safe
        t_idle_assist = self._idle_torque_balance * max(0.0, (1.0 - throttle_norm * 2.0))
        torque_engine = t_combustion + t_idle_assist

        # Propeller load torque: k_load * omega^2
        torque_load = self.tier_c.k_load * (omega ** 2)

        # Mechanical and pumping friction torque (scaled by friction_factor for Phase 4E)
        torque_friction = (self.tier_c.k_fric_linear * omega + self.tier_c.torque_fric_static) * friction_factor

        # Net acceleration torque
        net_torque = torque_engine - torque_load - torque_friction
        domega_dt = net_torque / self.tier_c.inertia_kg_m2

        return (
            p_combustion,
            torque_engine,
            torque_load,
            torque_friction,
            net_torque,
            domega_dt,
        )

    def _rk4_single_step(
        self,
        dt: float,
        throttle_pct: float,
        density_factor: float,
        combustion_efficiency_factor: float = 1.0,
        friction_factor: float = 1.0,
    ) -> None:
        """Internal single RK4 integration step."""
        omega_0 = self.omega

        # k1
        _, _, _, _, _, k1 = self._torque_derivatives(omega_0, throttle_pct, density_factor, combustion_efficiency_factor, friction_factor)

        # k2
        omega_k2 = max(0.0, omega_0 + 0.5 * dt * k1)
        _, _, _, _, _, k2 = self._torque_derivatives(omega_k2, throttle_pct, density_factor, combustion_efficiency_factor, friction_factor)

        # k3
        omega_k3 = max(0.0, omega_0 + 0.5 * dt * k2)
        _, _, _, _, _, k3 = self._torque_derivatives(omega_k3, throttle_pct, density_factor, combustion_efficiency_factor, friction_factor)

        # k4
        omega_k4 = max(0.0, omega_0 + dt * k3)
        _, _, _, _, _, k4 = self._torque_derivatives(omega_k4, throttle_pct, density_factor, combustion_efficiency_factor, friction_factor)

        # Update state with RK4 weighted average
        self.omega = max(0.0, omega_0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))

    def step(
        self,
        throttle_pct: float,
        density_factor: float,
        dt: float,
        combustion_efficiency_factor: float = 1.0,
        friction_factor: float = 1.0,
    ) -> OperatingPoint:
        """
        Advance rotational dynamics using sub-stepped 4th-Order Runge-Kutta (RK4) integration.
        Sub-stepping guarantees unconditional numerical stability across arbitrary external time steps.

        Args:
            friction_factor: Multiplier for mechanical friction torque.
                1.0 = nominal. Values > 1.0 model increased friction from mechanical
                degradation (Phase 4E). (Tier C/D calibration assumption.)
        """
        max_sub_dt = 0.02
        remaining_time = max(1e-4, float(dt))

        while remaining_time > 1e-6:
            sub_dt = min(remaining_time, max_sub_dt)
            self._rk4_single_step(sub_dt, throttle_pct, density_factor, combustion_efficiency_factor, friction_factor)
            remaining_time -= sub_dt

        rpm = self.omega_to_rpm(self.omega)

        # Compute full operating point diagnostics at updated state
        p_target, t_eng, t_load, t_fric, net_t, domega_dt = self._torque_derivatives(
            self.omega, throttle_pct, density_factor, combustion_efficiency_factor, friction_factor
        )

        # Engine load percentage proxy based on power output relative to density-adjusted max continuous
        max_available_p = max(100.0, self.tier_a.power_max_continuous_w * density_factor)
        load_pct = min(100.0, max(0.0, (p_target / max_available_p) * 100.0))

        return OperatingPoint(
            rpm=rpm,
            omega_rad_s=self.omega,
            throttle_pct=throttle_pct,
            density_factor=density_factor,
            power_target_w=p_target,
            torque_engine_nm=t_eng,
            torque_load_nm=t_load,
            torque_friction_nm=t_fric,
            net_torque_nm=net_t,
            domega_dt=domega_dt,
            engine_load_pct=load_pct,
        )

    def set_rpm(self, rpm: float) -> None:
        """Directly set rotational speed (e.g. for testing or initialization)."""
        self.omega = self.rpm_to_omega(rpm)
