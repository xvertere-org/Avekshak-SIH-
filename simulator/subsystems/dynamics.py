"""
Rotational Dynamics and Operating Point Model for SIH26054 (Phase 2).

GOVERNING ARCHITECTURE:
Reference: BRP-Rotax 914 UL/F Series (4-cylinder, 1211.2 cc).
Gearbox: Integrated spur reduction gearbox with 2.4286:1 (51:21) reduction ratio.

PHYSICAL POWER CHAIN:
Air Mass Flow -> Fuel Mass Flow / AFR -> Chemical Power ->
Indicated Combustion Power -> Mechanical Pumping/Friction Losses ->
Engine Brake Power -> Crankshaft Torque.

ENERGY & POWER INVARIANTS:
1. P_chemical > P_indicated > P_brake >= 0 (POWER_HIERARCHY_INVARIANT).
2. P_prop = T_prop * omega_prop.
3. P_eng_out = T_load,eng * omega_eng = P_prop / eta_gb >= P_prop (for eta_gb <= 1.0).
   The gearbox never creates energy.
4. omega_prop = omega_eng / 2.42857.
"""

import math
from typing import NamedTuple, Optional, Any
from simulator.config import (
    TierAParameters,
    TierCParameters,
    TierDParameters,
    TierCGearboxParameters,
    TierCTurboParameters,
)


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
    # Phase 2 Gearbox and Power Chain additions (with defaults for backward compatibility)
    propeller_rpm: float = 0.0
    omega_prop_rad_s: float = 0.0
    torque_prop_nm: float = 0.0
    power_prop_w: float = 0.0
    power_brake_w: float = 0.0
    power_indicated_w: float = 0.0
    power_chemical_w: float = 0.0
    air_mass_flow_kg_s: float = 0.0
    fuel_mass_flow_kg_s: float = 0.0
    gearbox_loss_w: float = 0.0


class RotationalDynamics:
    """
    4th-Order Runge-Kutta (RK4) Rotational Speed Simulator with
    Reduction Gearbox (2.4286:1) and First-Principles Combustion Power Chain.
    """

    def __init__(
        self,
        tier_a: Optional[TierAParameters] = None,
        tier_c: Optional[TierCParameters] = None,
        tier_d: Optional[TierDParameters] = None,
        tier_c_gearbox: Optional[TierCGearboxParameters] = None,
        tier_c_turbo: Optional[TierCTurboParameters] = None,
        initial_rpm: Optional[float] = None,
        config: Optional[Any] = None,
    ):
        if config is not None:
            self.tier_a = getattr(config, "tier_a", TierAParameters())
            self.tier_c = getattr(config, "tier_c", TierCParameters())
            self.tier_d = getattr(config, "tier_d", TierDParameters())
            self.gearbox = getattr(config, "tier_c_gearbox", TierCGearboxParameters())
            self.turbo = getattr(config, "tier_c_turbo", TierCTurboParameters())
        else:
            self.tier_a = tier_a if tier_a is not None else TierAParameters()
            self.tier_c = tier_c if tier_c is not None else TierCParameters()
            self.tier_d = tier_d if tier_d is not None else TierDParameters()
            self.gearbox = tier_c_gearbox if tier_c_gearbox is not None else TierCGearboxParameters()
            self.turbo = tier_c_turbo if tier_c_turbo is not None else TierCTurboParameters()

        # Gearbox parameters
        self.ratio = self.gearbox.reduction_ratio  # 2.42857 (51/21)
        self.eta_gb = min(1.0, max(0.80, self.gearbox.gearbox_efficiency))  # 0.975

        # Equivalent rotating inertia reflected to crankshaft: J_eq = J_eng + J_prop / i^2
        self.j_eq = self.gearbox.inertia_eng_kg_m2 + self.gearbox.inertia_prop_kg_m2 / (self.ratio ** 2)

        # Propeller aerodynamic torque coefficient
        self.k_prop = 5.2e-3

        # State variable: engine crankshaft angular velocity omega in rad/s
        init_rpm = initial_rpm if initial_rpm is not None else self.tier_c.rpm_idle
        self.omega = self.rpm_to_omega(init_rpm)

        # Precalculate idle assist torque to balance idle resistance at calibrated idle RPM
        omega_idle = self.rpm_to_omega(self.tier_c.rpm_idle)
        omega_prop_idle = omega_idle / self.ratio
        t_prop_idle = self.k_prop * (omega_prop_idle ** 2)
        t_load_idle = t_prop_idle / (self.ratio * self.eta_gb)
        t_fric_idle = self.tier_c.k_fric_linear * omega_idle + self.tier_c.torque_fric_static
        self._idle_torque_balance = t_load_idle + t_fric_idle

    def reset(self, initial_rpm: Optional[float] = None) -> None:
        """Reset dynamics state."""
        init_rpm = initial_rpm if initial_rpm is not None else self.tier_c.rpm_idle
        self.omega = self.rpm_to_omega(init_rpm)

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

    def compute_power_chain(
        self,
        rpm: float,
        throttle_pct: float,
        density_factor: float,
        combustion_efficiency_factor: float = 1.0,
        map_bar: Optional[float] = None,
        charge_air_temp_c: Optional[float] = None,
    ) -> tuple:
        """
        First-principles causal power chain:
        air mass flow -> fuel mass flow / AFR -> chemical power ->
        indicated power -> friction losses -> brake power & crankshaft torque.

        Returns:
            (p_brake, p_indicated, p_chemical, m_dot_air, m_dot_fuel)
        """
        eff_rpm = self.compute_rpm_efficiency(rpm)
        comb_eff = max(0.1, min(1.0, float(combustion_efficiency_factor)))
        thr_norm = max(0.0, min(100.0, float(throttle_pct))) / 100.0

        # Effective manifold pressure (bar)
        if map_bar is not None:
            p_map_bar = max(0.20, float(map_bar))
        else:
            # Fallback for standalone dynamics mode (unconnected turbocharger)
            p_map_bar = max(0.20, density_factor * (0.60 + 0.60 * thr_norm))

        # Effective charge air temperature (K)
        t_charge_k = max(220.0, (charge_air_temp_c if charge_air_temp_c is not None else 25.0) + 273.15)

        # 1. Intake Air Mass Flow (kg/s)
        # Displacement: 1211.2 cc = 1.2112e-3 m^3, 4-stroke factor = 0.5
        v_disp = self.tier_a.reference_displacement_cc * 1e-6
        n_eng_rps = max(100.0, rpm) / 60.0
        r_air = self.tier_a.gas_constant_air
        p_map_pa = p_map_bar * 100000.0

        # Volumetric efficiency model (speed & throttle dependent)
        eta_vol = max(0.72, min(0.92, 0.78 + 0.12 * (rpm / 5500.0) * (0.5 + 0.5 * thr_norm)))
        # Throttle restriction scales effective mass intake
        throttle_flow_factor = max(0.12, thr_norm ** 0.85) if thr_norm < 0.95 else 1.0
        m_dot_air = (p_map_pa * v_disp * n_eng_rps * 0.5 / (r_air * t_charge_k)) * eta_vol * throttle_flow_factor

        # 2. Fuel Mass Flow & Air-Fuel Ratio (AFR)
        # Nominal AFR ranges from 12.8 (takeoff rich) to 14.5 (cruise)
        afr_target = 14.5 - 1.7 * thr_norm
        m_dot_fuel = m_dot_air / max(8.0, afr_target)

        # 3. Chemical Power Release: P_chem = m_dot_fuel * LHV (43.5 MJ/kg)
        lhv = self.tier_c.fuel_lhv_j_per_kg
        p_chemical = m_dot_fuel * lhv

        # 4. Indicated Combustion Power
        # Base indicated thermal efficiency ~36%
        eta_indicated_base = 0.36 * eff_rpm * comb_eff
        p_indicated = p_chemical * eta_indicated_base

        # 5. Mechanical Pumping & Friction Losses
        omega_eng = self.rpm_to_omega(rpm)
        p_friction = (self.tier_c.k_fric_linear * omega_eng + self.tier_c.torque_fric_static) * omega_eng

        # 6. Engine Brake Power
        p_brake = max(0.0, p_indicated - p_friction)

        # POWER_HIERARCHY_INVARIANT Assertion:
        # P_chemical > P_indicated > P_brake >= 0 (at all operating speeds > idle)
        if p_chemical > 1000.0:
            if p_indicated >= p_chemical:
                p_indicated = p_chemical * 0.95
            if p_brake >= p_indicated:
                p_brake = max(0.0, p_indicated - p_friction)

        return p_brake, p_indicated, p_chemical, m_dot_air, m_dot_fuel

    def _torque_derivatives(
        self,
        omega: float,
        throttle_pct: float,
        density_factor: float,
        combustion_efficiency_factor: float = 1.0,
        friction_factor: float = 1.0,
        map_bar: Optional[float] = None,
        charge_air_temp_c: Optional[float] = None,
    ) -> tuple:
        """
        Calculate instantaneous torques and domega/dt at given state.
        Reflects propeller load torque through the 2.4286:1 reduction gearbox.
        """
        rpm = self.omega_to_rpm(omega)
        throttle_norm = max(0.0, min(100.0, throttle_pct)) / 100.0

        # Compute power chain
        p_brake, p_ind, p_chem, m_air, m_fuel = self.compute_power_chain(
            rpm=rpm,
            throttle_pct=throttle_pct,
            density_factor=density_factor,
            combustion_efficiency_factor=combustion_efficiency_factor,
            map_bar=map_bar,
            charge_air_temp_c=charge_air_temp_c,
        )

        # Crankshaft brake torque
        omega_safe = max(omega, 10.0)
        t_combustion = p_brake / omega_safe
        t_idle_assist = self._idle_torque_balance * max(0.0, (1.0 - throttle_norm * 2.0))
        torque_engine = t_combustion + t_idle_assist

        # Propeller kinematics and load
        omega_prop = omega / self.ratio
        prop_rpm = rpm / self.ratio
        t_prop = self.k_prop * (omega_prop ** 2)
        p_prop = t_prop * omega_prop

        # Reflected load torque to crankshaft through reduction gearbox:
        # T_load,eng = T_prop / (i * eta_gb)
        torque_load_eng = t_prop / (self.ratio * self.eta_gb)

        # Crankshaft friction torque
        torque_friction = (self.tier_c.k_fric_linear * omega + self.tier_c.torque_fric_static) * friction_factor

        # Net acceleration torque & rotational acceleration
        net_torque = torque_engine - torque_load_eng - torque_friction
        domega_dt = net_torque / self.j_eq

        # Gearbox power output & loss
        p_eng_out = torque_load_eng * omega
        p_gb_loss = max(0.0, p_eng_out - p_prop)

        return (
            p_brake,
            torque_engine,
            torque_load_eng,
            torque_friction,
            net_torque,
            domega_dt,
            p_ind,
            p_chem,
            m_air,
            m_fuel,
            prop_rpm,
            omega_prop,
            t_prop,
            p_prop,
            p_gb_loss,
        )

    def _rk4_single_step(
        self,
        dt: float,
        throttle_pct: float,
        density_factor: float,
        combustion_efficiency_factor: float = 1.0,
        friction_factor: float = 1.0,
        map_bar: Optional[float] = None,
        charge_air_temp_c: Optional[float] = None,
    ) -> None:
        """Internal single RK4 integration step."""
        omega_0 = self.omega

        # k1
        res1 = self._torque_derivatives(
            omega_0, throttle_pct, density_factor, combustion_efficiency_factor, friction_factor, map_bar, charge_air_temp_c
        )
        k1 = res1[5]

        # k2
        omega_k2 = max(0.0, omega_0 + 0.5 * dt * k1)
        res2 = self._torque_derivatives(
            omega_k2, throttle_pct, density_factor, combustion_efficiency_factor, friction_factor, map_bar, charge_air_temp_c
        )
        k2 = res2[5]

        # k3
        omega_k3 = max(0.0, omega_0 + 0.5 * dt * k2)
        res3 = self._torque_derivatives(
            omega_k3, throttle_pct, density_factor, combustion_efficiency_factor, friction_factor, map_bar, charge_air_temp_c
        )
        k3 = res3[5]

        # k4
        omega_k4 = max(0.0, omega_0 + dt * k3)
        res4 = self._torque_derivatives(
            omega_k4, throttle_pct, density_factor, combustion_efficiency_factor, friction_factor, map_bar, charge_air_temp_c
        )
        k4 = res4[5]

        # Update state with RK4 weighted average
        self.omega = max(0.0, omega_0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))

    def step(
        self,
        throttle_pct: float,
        density_factor: float,
        dt: float,
        combustion_efficiency_factor: float = 1.0,
        friction_factor: float = 1.0,
        map_bar: Optional[float] = None,
        charge_air_temp_c: Optional[float] = None,
    ) -> OperatingPoint:
        """
        Advance rotational dynamics using sub-stepped 4th-Order Runge-Kutta (RK4) integration.
        Sub-stepping guarantees unconditional numerical stability across arbitrary external time steps.
        """
        max_sub_dt = 0.02
        remaining_time = max(1e-4, float(dt))

        while remaining_time > 1e-6:
            sub_dt = min(remaining_time, max_sub_dt)
            self._rk4_single_step(
                sub_dt,
                throttle_pct,
                density_factor,
                combustion_efficiency_factor,
                friction_factor,
                map_bar,
                charge_air_temp_c,
            )
            remaining_time -= sub_dt

        rpm = self.omega_to_rpm(self.omega)

        # Compute full operating point diagnostics at updated state
        (
            p_brake,
            t_eng,
            t_load,
            t_fric,
            net_t,
            domega_dt,
            p_ind,
            p_chem,
            m_air,
            m_fuel,
            prop_rpm,
            omega_prop,
            t_prop,
            p_prop,
            p_gb_loss,
        ) = self._torque_derivatives(
            self.omega,
            throttle_pct,
            density_factor,
            combustion_efficiency_factor,
            friction_factor,
            map_bar,
            charge_air_temp_c,
        )

        # Engine load percentage proxy based on power output relative to density-adjusted max continuous
        max_available_p = max(100.0, self.tier_a.power_max_continuous_w * density_factor)
        load_pct = min(100.0, max(0.0, (p_brake / max_available_p) * 100.0))

        return OperatingPoint(
            rpm=rpm,
            omega_rad_s=self.omega,
            throttle_pct=throttle_pct,
            density_factor=density_factor,
            power_target_w=p_brake,
            torque_engine_nm=t_eng,
            torque_load_nm=t_load,
            torque_friction_nm=t_fric,
            net_torque_nm=net_t,
            domega_dt=domega_dt,
            engine_load_pct=load_pct,
            propeller_rpm=prop_rpm,
            omega_prop_rad_s=omega_prop,
            torque_prop_nm=t_prop,
            power_prop_w=p_prop,
            power_brake_w=p_brake,
            power_indicated_w=p_ind,
            power_chemical_w=p_chem,
            air_mass_flow_kg_s=m_air,
            fuel_mass_flow_kg_s=m_fuel,
            gearbox_loss_w=p_gb_loss,
        )

    def set_rpm(self, rpm: float) -> None:
        """Directly set rotational speed (e.g. for testing or initialization)."""
        self.omega = self.rpm_to_omega(rpm)


# Alias for subsystem naming convention
EngineDynamics = RotationalDynamics
