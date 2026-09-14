"""
Turbocharger Subsystem and Reduced-Order TCU Surrogate for SIH26054.

GOVERNING ARCHITECTURE:
Reference: BRP-Rotax 914 UL/F Engine Series (Configuration 914 F3/F4 [EASA TCDS E.122] and 914 UL3/UL4 [ASTM F2339]).
TCU Version: Standard Electronic Turbo Control Unit P/N 964 872 / 964 874.

SURROGATE DISCLOSURE:
Exhaust manifold pressure P3 = Pamb + Delta_P_exh, turbine expansion approximately to ambient,
and turbine/compressor pressure ratios are modeled quantities (MODEL_ASSUMPTION / MODEL_CALIBRATION)
rather than proprietary OEM compressor/turbine maps. This module implements a physically coupled
reduced-order energy-balance loop and does not claim to reproduce proprietary Rotax / IHI turbocharger maps.
"""

import math
from typing import NamedTuple, Optional
from simulator.config import TierAParameters, TierCTurboParameters


class TurboState(NamedTuple):
    """Instantaneous thermodynamic state of the turbocharger and boost system."""
    map_bar: float                         # Manifold Absolute Pressure (bar)
    map_target_bar: float                  # TCU target MAP for current throttle and altitude (bar)
    pressure_ratio: float                  # Compressor pressure ratio (P2 / P1 >= 1.0)
    compressor_discharge_temp_c: float     # T2 compressor outlet temperature (°C)
    charge_air_temp_c: float               # Airbox / intake charge temperature after cooler (°C)
    wastegate_position: float              # 0.0 = fully closed (max boost), 1.0 = fully open bypass
    turbine_power_w: float                 # Shaft power extracted by turbine (W)
    compressor_power_w: float              # Shaft power absorbed by compressor (W)
    exhaust_manifold_pressure_bar: float   # P3 turbine inlet pressure (bar)


class TCUSurrogate:
    """
    Reduced-order Electronic Turbo Control Unit (TCU) surrogate.
    Implements closed-loop, rate-limited electromechanical servo wastegate displacement
    to maintain target manifold pressure up to the continuous critical altitude (4572 m).
    """

    def __init__(self, config: TierCTurboParameters = TierCTurboParameters()):
        self.config = config
        # Initial wastegate position: 1.0 = fully open (unboosted / cold start)
        self.wastegate_pos: float = 1.0

    def compute_target_map(self, throttle_pct: float, altitude_m: float) -> float:
        """
        Calculate TCU target MAP based on throttle position and flight altitude.
        Selected model targets:
        - Takeoff: 1.350 bar (5-min max @ full throttle)
        - Continuous: 1.200 bar (max continuous @ 50-90% throttle)
        - Partial load: Linear interpolation down to idle intake (~0.60 bar)
        """
        thr = max(0.0, min(100.0, float(throttle_pct)))

        if thr <= 25.0:
            # Idle to low cruise: Naturally aspirated throttle-governed region
            target = 0.60 + (thr / 25.0) * 0.35  # 0.60 to 0.95 bar
        elif thr <= 85.0:
            # Continuous boost range: Regulate to continuous rating (1.200 bar)
            norm = (thr - 25.0) / 60.0
            target = 0.95 + norm * (self.config.tcu_continuous_map_target_bar - 0.95)
        else:
            # Takeoff boost range: 85% to 100% throttle regulates up to 1.350 bar
            norm = (thr - 85.0) / 15.0
            target = self.config.tcu_continuous_map_target_bar + norm * (
                self.config.tcu_takeoff_map_target_bar - self.config.tcu_continuous_map_target_bar
            )

        return float(target)

    def step(self, target_map_bar: float, current_map_bar: float, dt: float) -> float:
        """
        Advance wastegate servo position via rate-limited closed-loop error feedback.
        Error e = target - current.
        If target > current, wastegate closes (w_pos -> 0) to extract more turbine enthalpy.
        If target < current, wastegate opens (w_pos -> 1) to vent exhaust.
        """
        e_map = target_map_bar - current_map_bar
        # Servo displacement rate proportional to MAP error
        # Closing is negative displacement, opening is positive displacement
        desired_rate = -self.config.tcu_kp * e_map
        clamped_rate = max(-self.config.tcu_w_dot_max, min(self.config.tcu_w_dot_max, desired_rate))

        self.wastegate_pos = max(0.0, min(1.0, self.wastegate_pos + clamped_rate * dt))
        return self.wastegate_pos


class TurbineSurrogate:
    """
    Reduced-order radial turbine thermodynamic expansion surrogate.
    Calculates turbine enthalpy extraction and shaft work output.
    """

    def __init__(self, config: TierCTurboParameters = TierCTurboParameters()):
        self.config = config

    def compute_work(
        self,
        m_dot_exh_kg_s: float,
        t_exh_c: float,
        p_amb_pa: float,
        wastegate_pos: float,
    ) -> tuple:
        """
        Compute turbine isentropic expansion enthalpy and actual shaft power output.
        Returns: (w_dot_turb_w, p3_bar)
        """
        t_exh_k = max(273.15, t_exh_c + 273.15)
        p_amb_bar = p_amb_pa / 100000.0

        # Exhaust manifold backpressure surrogate: P3 = P_amb + Delta_P_exh
        # Backpressure scales with non-bypassed exhaust mass flow
        active_fraction = 1.0 - wastegate_pos * 0.82  # Even fully open wastegate has minor exhaust restriction
        m_dot_turb = max(0.0, m_dot_exh_kg_s * active_fraction)

        delta_p_exh = self.config.delta_p_exh_manifold_bar * min(2.0, active_fraction * (t_exh_k / 900.0))
        p3_bar = max(p_amb_bar + 0.05, p_amb_bar + delta_p_exh)

        # Expansion pressure ratio across turbine
        pr_turb = p3_bar / max(0.1, p_amb_bar)

        # Isentropic enthalpy drop: Delta_h_is = c_p * T * [1 - (1/PR)^((gamma-1)/gamma)]
        gamma_e = self.config.gamma_exh
        exp_factor = (gamma_e - 1.0) / gamma_e
        delta_h_is = self.config.c_p_exh * t_exh_k * (1.0 - (1.0 / max(1.001, pr_turb)) ** exp_factor)

        # Actual turbine shaft work: W_dot_turb = m_dot * Delta_h_is * eta_turb
        w_dot_turb = max(0.0, m_dot_turb * delta_h_is * self.config.eta_turb_nominal)
        return w_dot_turb, p3_bar


class CompressorSurrogate:
    """
    Reduced-order centrifugal compressor thermodynamic surrogate.
    Calculates pressure ratio and discharge temperature from shaft work input.
    """

    def __init__(self, config: TierCTurboParameters = TierCTurboParameters()):
        self.config = config

    def compute_compression(
        self,
        w_dot_comp_w: float,
        m_dot_air_kg_s: float,
        p_inlet_pa: float,
        t_inlet_c: float,
    ) -> tuple:
        """
        Compute pressure ratio PR, compressor discharge temp T2, and charge temp T_charge.
        Returns: (pr, t2_c, t_charge_c)
        """
        t1_k = max(200.0, t_inlet_c + 273.15)
        m_air_safe = max(1e-4, m_dot_air_kg_s)

        # Specific work input: w_c = W_dot / m_dot
        w_c = w_dot_comp_w / m_air_safe

        # Pressure ratio from thermodynamic enthalpy rise:
        # PR = [1 + (w_c * eta_comp) / (c_p * T1)]^(gamma / (gamma - 1))
        gamma_a = self.config.gamma_air
        exp_comp = gamma_a / (gamma_a - 1.0)

        pr_term = 1.0 + (w_c * self.config.eta_comp_nominal) / (self.config.c_p_air * t1_k)
        pr = max(1.0, min(3.2, pr_term ** exp_comp))

        # Isentropic discharge temperature: T2s = T1 * PR^((gamma-1)/gamma)
        t2s_k = t1_k * (pr ** ((gamma_a - 1.0) / gamma_a))
        # Actual discharge temperature accounting for isentropic efficiency:
        t2_k = t1_k + (t2s_k - t1_k) / max(0.1, self.config.eta_comp_nominal)
        t2_c = t2_k - 273.15

        # Charge-air cooling across airbox / intercooler
        # T_charge = T1 + (T2 - T1) * (1 - eta_cooler)
        t_charge_k = t1_k + (t2_k - t1_k) * (1.0 - self.config.intercooler_efficiency)
        t_charge_c = t_charge_k - 273.15

        return pr, t2_c, t_charge_c


class TurbochargerSubsystem:
    """
    Complete Turbocharger Subsystem with REDUCED_ORDER_TCU_SURROGATE.
    Couples exhaust enthalpy, turbine work, compressor boost, airbox heat drop,
    and manifold filling/emptying lag dynamics.
    """

    def __init__(
        self,
        tier_a: TierAParameters = TierAParameters(),
        config: TierCTurboParameters = TierCTurboParameters(),
        initial_map_bar: float = 1.013,
    ):
        self.tier_a = tier_a
        self.config = config
        self.tcu = TCUSurrogate(config)
        self.turbine = TurbineSurrogate(config)
        self.compressor = CompressorSurrogate(config)

        # State variables
        self.map_bar: float = initial_map_bar
        self.charge_air_temp_c: float = 25.0
        self.compressor_discharge_temp_c: float = 25.0
        self.pressure_ratio: float = 1.0
        self.wastegate_pos: float = 1.0
        self.turbine_power_w: float = 0.0
        self.compressor_power_w: float = 0.0
        self.p3_bar: float = 1.013

    def step(
        self,
        throttle_pct: float,
        altitude_m: float,
        p_amb_pa: float,
        t_amb_c: float,
        m_dot_air_kg_s: float,
        m_dot_fuel_kg_s: float,
        t_exh_c: float,
        dt: float,
        ram_pressure_recovery_pa: float = 0.0,
    ) -> TurboState:
        """
        Advance turbocharger thermodynamics across timestep dt.

        Preserves the strict physical distinction between:
        1. Turbocharger boost capability (pressure ratio PR >= 1.0)
        2. Throttle downstream restriction
        3. Manifold Absolute Pressure (MAP)
        """
        dt_safe = max(1e-4, float(dt))
        thr = max(0.0, min(100.0, float(throttle_pct)))

        # 1. TCU closed-loop wastegate positioning
        target_map = self.tcu.compute_target_map(thr, altitude_m)
        self.wastegate_pos = self.tcu.step(target_map, self.map_bar, dt_safe)

        # 2. Turbine enthalpy extraction from exhaust gas
        m_dot_exh = m_dot_air_kg_s + m_dot_fuel_kg_s
        w_dot_turb, self.p3_bar = self.turbine.compute_work(
            m_dot_exh_kg_s=m_dot_exh,
            t_exh_c=t_exh_c,
            p_amb_pa=p_amb_pa,
            wastegate_pos=self.wastegate_pos,
        )
        self.turbine_power_w = w_dot_turb

        # 3. Mechanical shaft power transfer to compressor
        w_dot_comp = w_dot_turb * self.config.eta_mech_turbo
        self.compressor_power_w = w_dot_comp

        # 4. Compressor thermodynamic compression
        p_inlet = p_amb_pa + ram_pressure_recovery_pa
        pr, t2_c, t_charge_c = self.compressor.compute_compression(
            w_dot_comp_w=w_dot_comp,
            m_dot_air_kg_s=m_dot_air_kg_s,
            p_inlet_pa=p_inlet,
            t_inlet_c=t_amb_c,
        )
        self.pressure_ratio = pr
        self.compressor_discharge_temp_c = t2_c
        self.charge_air_temp_c = t_charge_c

        # 5. Downstream Intake Manifold Pressure Dynamics
        # Throttle butterfly imposes restriction downstream of the compressor.
        # When throttle is small, downstream target drops below ambient, even while PR >= 1.0!
        p1_bar = p_inlet / 100000.0
        boosted_pressure_bar = p1_bar * self.pressure_ratio

        # Effective downstream target governed by throttle position
        throttle_norm = thr / 100.0
        # Flow conductance characteristic through throttle butterfly
        if throttle_norm < 0.15:
            # Closed / idle throttle creates intake plenum vacuum
            intake_fraction = 0.55 + throttle_norm * 1.5  # ~0.55 to 0.77
        else:
            intake_fraction = 0.77 + (throttle_norm - 0.15) / 0.85 * 0.23  # 0.77 to 1.00

        p_manifold_target = boosted_pressure_bar * intake_fraction

        # Manifold filling/emptying lag ODE: d(MAP)/dt = (P_target - MAP) / tau_map
        # Using exact exponential decay guarantees unconditional numerical stability across arbitrary dt
        tau = max(0.05, self.config.tau_map_s)
        decay = 1.0 - math.exp(-dt_safe / tau)
        self.map_bar = max(0.25, self.map_bar + (p_manifold_target - self.map_bar) * decay)

        return TurboState(
            map_bar=self.map_bar,
            map_target_bar=target_map,
            pressure_ratio=self.pressure_ratio,
            compressor_discharge_temp_c=self.compressor_discharge_temp_c,
            charge_air_temp_c=self.charge_air_temp_c,
            wastegate_position=self.wastegate_pos,
            turbine_power_w=self.turbine_power_w,
            compressor_power_w=self.compressor_power_w,
            exhaust_manifold_pressure_bar=self.p3_bar,
        )

    def set_map(self, map_bar: float) -> None:
        """Directly initialize or set MAP (e.g. for deterministic testing)."""
        self.map_bar = max(0.25, float(map_bar))

    def reset(self, initial_map_bar: float = 1.013) -> None:
        """Reset turbocharger states to initial unboosted condition."""
        self.map_bar = max(0.25, float(initial_map_bar))
        self.charge_air_temp_c = 25.0
        self.compressor_discharge_temp_c = 25.0
        self.pressure_ratio = 1.0
        self.wastegate_pos = 1.0
        self.turbine_power_w = 0.0
        self.compressor_power_w = 0.0
        self.p3_bar = 1.013
        self.tcu.wastegate_pos = 1.0

