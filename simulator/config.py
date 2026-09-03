"""
Configuration and Parameter Tier definitions for SIH26054 Aero Piston Engine Simulator.

DISCLAIMER:
The Rotax 912 ULS is used ONLY as a publicly documented engineering reference anchor.
This configuration does NOT represent the actual MALE-UAV target engine, nor is it a certified OEM model.

Parameter Tiers:
- Tier A: Publicly documented reference constants and published operational limits
- Tier B: Physics-derived relationships and thermodynamic conversions
- Tier C: Calibration parameters (empirically tuned for reduced-order grey-box fidelity)
- Tier D: Engineering assumptions (functional forms, order weightings, idealized mappings)
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass(frozen=True)
class TierAParameters:
    """
    Tier A — Public / Reference Parameters (Rotax 912 ULS Engineering Anchor).
    Units: SI (with standard aviation conversions documented).
    """
    # Rotational limits (crankshaft RPM)
    rpm_max: float = 5800.0                     # Max permissible RPM (5 min limit)
    rpm_max_continuous: float = 5500.0          # Max continuous RPM

    # Power rating (Watts)
    power_max_continuous_w: float = 58000.0     # 58 kW (~77.8 hp) continuous power @ 5500 RPM

    # Operational reference limits
    cht_limit_c: float = 150.0                  # Max CHT limit (°C) (Rotax 912 ULS reference anchor)
    cht_nominal_c: float = 100.0                # Typical nominal CHT (°C)
    oil_temp_min_c: float = 50.0                # Minimum operating oil temp (°C)
    oil_temp_max_c: float = 130.0               # Maximum permissible oil temp (°C)
    oil_temp_nominal_min_c: float = 90.0        # Nominal oil temp lower bound (°C)
    oil_temp_nominal_max_c: float = 110.0       # Nominal oil temp upper bound (°C)
    oil_press_min_bar: float = 0.8              # Minimum oil pressure (bar)
    oil_press_max_bar: float = 7.0              # Maximum oil pressure (bar)
    oil_press_normal_min_bar: float = 2.0       # Normal oil pressure lower bound (bar)
    oil_press_normal_max_bar: float = 5.0       # Normal oil pressure upper bound (bar)

    # International Standard Atmosphere (ISA) Sea-Level Constants
    p0_sea_level: float = 101325.0              # Pa (N/m²)
    t0_sea_level: float = 288.15                # K (15.0 °C)
    rho0_sea_level: float = 1.225               # kg/m³
    lapse_rate: float = 0.0065                  # K/m (troposphere temperature lapse rate)
    gas_constant_air: float = 287.05            # J/(kg·K)
    gravity: float = 9.80665                    # m/s²


@dataclass
class TierCParameters:
    """
    Tier C — Calibration Parameters.
    Tuned for reduced-order grey-box model fidelity.
    """
    # Idle speed calibration
    rpm_idle: float = 1400.0                    # RPM at zero throttle idle

    # Rotational dynamics calibration (crankshaft + propeller equivalent inertia)
    inertia_kg_m2: float = 0.28                 # Effective rotating assembly + prop inertia (kg·m²)
    k_load: float = 3.036e-4                    # Propeller load coefficient: Torque_load = k_load * omega^2
    k_fric_linear: float = 0.015                # Viscous friction coefficient (N·m·s/rad)
    torque_fric_static: float = 3.5             # Pumping/coulomb friction torque (N·m)

    # Fuel flow calibration (Willans-line approximation: Mass_flow = a * P + b)
    a_fuel_kg_per_j: float = 6.8e-8             # Specific fuel consumption slope (kg/J, ~245 g/kWh)
    b_fuel_kg_per_s: float = 0.00032            # Idle fuel consumption rate (kg/s, ~1.15 kg/h)
    fuel_density_kg_per_l: float = 0.72         # Avgas / Mogas fuel density (kg/L)
    k_fuel_flow_lean: float = 0.25              # Fuel mass flow reduction fraction at lean severity=1.0 (Tier C/D)
    k_fuel_flow_rich: float = 0.30              # Fuel mass flow increase fraction at rich severity=1.0 (Tier C/D)
    k_comb_loss_lean: float = 0.08              # Indicated combustion power reduction fraction at lean severity=1.0 (Tier C/D)
    k_comb_loss_rich: float = 0.06              # Indicated combustion power reduction fraction at rich severity=1.0 (Tier C/D)

    # Thermal CHT calibration (lumped thermal capacitance)
    c_th_cht: float = 920.0                     # Lumped CHT thermal capacitance (J/K)
    q_gen_fraction: float = 0.048               # Fraction of fuel energy converted to cylinder head heat (~5%)
    fuel_lhv_j_per_kg: float = 43.5e6           # Fuel lower heating value (J/kg)
    h_cool_base: float = 16.0                   # Base convective cooling conductance (W/K)
    h_cool_rpm: float = 0.012                   # Convective cooling conductance per RPM (W/(K·RPM))
    h_cool_airspeed: float = 0.35               # Convective cooling conductance per m/s airspeed (W/(K·(m/s)))
    k_cooling_max_loss: float = 0.55            # Max cooling conductance loss fraction at severity=1.0 (Tier C/D)

    # Thermal EGT calibration
    t_egt_base_c: float = 520.0                 # Base EGT at idle/low load (°C)
    k_egt_load: float = 210.0                   # EGT increase with load (°C)
    k_egt_rpm: float = 0.018                    # EGT sensitivity to RPM offset (°C/RPM)
    k_egt_density: float = 35.0                 # EGT sensitivity to density factor (°C)
    tau_egt_s: float = 2.5                      # EGT thermocouple dynamic lag time constant (s)
    k_egt_lean_gain_c: float = 95.0             # Steady-state EGT elevation at lean severity=1.0 (°C, Tier C/D)
    k_egt_rich_drop_c: float = 80.0             # Steady-state EGT drop at rich severity=1.0 (°C, Tier C/D)

    # Lubrication / Oil thermal & pressure calibration
    c_oil: float = 1350.0                       # Oil system thermal capacitance (J/K)
    q_oil_fraction: float = 0.016               # Fraction of fuel energy transferred to oil (~1.6%)
    h_oil_cool: float = 20.0                    # Oil radiator cooling conductance (W/K)
    k_oil_cht_couple: float = 2.5               # Thermal conduction coupling between CHT and oil (W/K)
    oil_press_base_bar: float = 1.0             # Base static oil pressure (bar)
    k_oil_p_rpm: float = 0.00085                # Oil pressure gain per RPM (bar/RPM)
    k_oil_p_temp: float = 0.008                 # Oil pressure drop per °C temp rise above nominal (bar/°C)
    k_lub_p_loss: float = 0.55                  # Max hydraulic oil pressure loss fraction at severity=1.0 (Tier C/D)
    k_lub_heat_gain: float = 0.20               # Oil frictional heat generation gain fraction at severity=1.0 (Tier C/D)
    k_lub_cool_loss: float = 0.20               # Oil cooler heat rejection reduction fraction at severity=1.0 (Tier C/D)

    # Vibration synthesis calibration
    vib_order1_base_g: float = 0.32             # 1x rotational order baseline amplitude (g)
    vib_order2_base_g: float = 0.22             # 2x rotational order baseline amplitude (g)
    vib_load_gain: float = 0.45                 # Vibration amplitude scaling with load
    vib_noise_std_g: float = 0.06               # Broadband process vibration noise standard deviation (g)

    # Mechanical degradation calibration (Phase 4E — Tier C/D assumptions, NOT measured engine data)
    k_mech_vib_gain: float = 1.8                # Max 1×/2× amplitude multiplier increase at severity=1.0 (mechanical_condition = 1 + 1.8 = 2.8×)
    k_mech_noise_gain: float = 2.5              # Max broadband noise std multiplier increase at severity=1.0 (noise_std *= 1 + 2.5 = 3.5×)
    k_mech_friction_gain: float = 0.08          # Max friction torque increase fraction at severity=1.0 (secondary effect, 8% max)

    # Sensor measurement noise calibration (standard deviations)
    sensor_noise_rpm: float = 4.0               # RPM
    sensor_noise_cht: float = 0.4               # °C
    sensor_noise_egt: float = 1.8               # °C
    sensor_noise_oil_temp: float = 0.3          # °C
    sensor_noise_oil_press: float = 0.025       # bar
    sensor_noise_fuel_flow: float = 0.15        # L/h
    sensor_noise_vibration: float = 0.015       # g


@dataclass
class TierDParameters:
    """
    Tier D — Engineering Assumptions.
    Structural representations and unverified functional forms.
    """
    # Combustion/Mechanical efficiency curve polynomial coefficients vs normalized RPM (rpm / 5500)
    # Assumes efficiency peak near rated RPM (0.85 to 1.0 normalized)
    rpm_eff_poly: tuple = (-0.75, 1.65, 0.10)  # eff = -0.75*(norm_rpm)^2 + 1.65*(norm_rpm) + 0.10

    # Airspeed proxy relationship from mission phase and RPM
    airspeed_proxy_takeoff_ms: float = 25.0
    airspeed_proxy_climb_ms: float = 35.0
    airspeed_proxy_cruise_ms: float = 48.0
    airspeed_proxy_loiter_ms: float = 38.0
    airspeed_proxy_descent_ms: float = 42.0
    airspeed_proxy_landing_ms: float = 22.0

    # Mechanical condition factor (1.0 = healthy nominal, < 1.0 = degraded, hooked for Phase 4)
    mechanical_condition: float = 1.0


@dataclass
class SimulatorConfig:
    """
    Master configuration combining all parameter tiers for the Engine Simulator.
    """
    tier_a: TierAParameters = field(default_factory=TierAParameters)
    tier_c: TierCParameters = field(default_factory=TierCParameters)
    tier_d: TierDParameters = field(default_factory=TierDParameters)

    # Simulation control settings
    default_dt: float = 0.1                     # Simulation integration time step (s)
    random_seed: int = 42                       # Deterministic random seed for process & sensor noise
    provenance_version: str = "0.2.0-phase2b-physics"

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "tier_a": vars(self.tier_a),
            "tier_c": vars(self.tier_c),
            "tier_d": vars(self.tier_d),
            "default_dt": self.default_dt,
            "random_seed": self.random_seed,
            "provenance_version": self.provenance_version,
        }
