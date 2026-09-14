"""
Configuration and Parameter Tier definitions for SIH26054 Aero Piston Engine Simulator.

REFERENCE ARCHITECTURE:
Authoritative Reference Engine: Rotax 914 UL/F (4-cylinder, 1211.2 cc, turbocharged, 4-stroke boxer).
Authoritative OEM Specification: configs/engine_reference/rotax_914_ul_f.json
Governing Contract: docs/physics_contract.md and configs/physics_contract.json

CURRENT SIMULATOR FIDELITY:
The simulator is a REDUCED-ORDER GREY-BOX PROTOTYPE operating with lumped-parameter 0D/1D physics.
Key physical mechanisms (turbocharger/TCU, 2.43:1 reduction gearbox, 4-cylinder thermal/exhaust network,
and electrical system) are explicitly classified as unmodeled or proxy models in the Physics Contract.

Parameter Tiers:
- Tier A: Authoritative reference constants and published operational limits
- Tier B: Physics-derived relationships and thermodynamic conversions
- Tier C: Calibration parameters (empirically tuned for reduced-order grey-box fidelity)
- Tier D: Engineering assumptions (functional forms, order weightings, idealized mappings)
"""

from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass(frozen=True)
class TierAParameters:
    """
    Tier A — Reference Specifications & Operational Limits.
    Reference Engine Architecture: Rotax 914 UL/F.
    Units: SI (with standard aviation conversions documented).
    """
    # Engine Reference Identity & Architecture
    reference_engine: str = "Rotax 914 UL/F"
    reference_displacement_cc: float = 1211.2    # Bore 79.5 mm, Stroke 61.0 mm, 4 cylinders
    reference_compression_ratio: float = 9.0     # 9.0:1 compression ratio
    reference_bore_mm: float = 79.5              # mm
    reference_stroke_mm: float = 61.0            # mm
    reference_gearbox_ratio: float = 2.4286      # Integrated 2.43:1 (51:21) reduction gearbox
    oem_continuous_power_w: float = 73500.0      # 73.5 kW (100 hp) @ 5500 RPM (continuous)
    oem_takeoff_power_w: float = 84500.0         # 84.5 kW (115 hp) @ 5800 RPM (5-min takeoff limit)

    # Rotational limits (crankshaft RPM)
    rpm_max: float = 5800.0                     # Max permissible RPM (5 min limit)
    rpm_max_continuous: float = 5500.0          # Max continuous RPM

    # Simulator Calibrated Power Ceiling (Watts)
    # NOTE: 58 kW is the calibrated power ceiling of the current naturally aspirated grey-box prototype.
    # It must not be conflated with the Rotax 914 OEM continuous rating (73.5 kW) or takeoff rating (84.5 kW).
    power_max_continuous_w: float = 58000.0     # 58 kW (~77.8 hp) continuous power ceiling @ 5500 RPM

    # Operational reference limits
    cht_limit_c: float = 150.0                  # Operational warning limit (°C) (OEM max continuous: 135 °C)
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

    # Rotational dynamics calibration (crankshaft + propeller equivalent direct 1:1 drive; gearbox ratio 2.43:1 unmodeled)
    inertia_kg_m2: float = 0.28                 # Effective rotating assembly + prop inertia (kg·m²) (1:1 direct drive)
    k_load: float = 3.036e-4                    # Propeller load coefficient: Torque_load = k_load * omega^2 (1:1 direct drive)
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

    # Phase 4F: Sensor fault default magnitudes (Tier C/D engineering assumptions)
    # These are NOT measured or certified UAV-engine sensor characteristics.
    # Bias magnitudes (max offset at severity=1.0)
    sensor_bias_rpm: float = 50.0               # RPM
    sensor_bias_cht: float = 8.0                # °C
    sensor_bias_egt: float = 15.0               # °C
    sensor_bias_oil_temp: float = 6.0           # °C
    sensor_bias_oil_press: float = 0.5          # bar
    sensor_bias_fuel_flow: float = 2.0          # L/h
    sensor_bias_vibration: float = 0.15         # g
    # Drift rates (max rate per second at severity=1.0)
    sensor_drift_rate_rpm: float = 5.0          # RPM/s
    sensor_drift_rate_cht: float = 0.8          # °C/s
    sensor_drift_rate_egt: float = 1.5          # °C/s
    sensor_drift_rate_oil_temp: float = 0.6     # °C/s
    sensor_drift_rate_oil_press: float = 0.05   # bar/s
    sensor_drift_rate_fuel_flow: float = 0.2    # L/h/s
    sensor_drift_rate_vibration: float = 0.015  # g/s
    # Fault-induced noise std (additional noise at severity=1.0)
    sensor_noise_fault_rpm: float = 20.0        # RPM
    sensor_noise_fault_cht: float = 3.0         # °C
    sensor_noise_fault_egt: float = 8.0         # °C
    sensor_noise_fault_oil_temp: float = 2.0    # °C
    sensor_noise_fault_oil_press: float = 0.2   # bar
    sensor_noise_fault_fuel_flow: float = 1.0   # L/h
    sensor_noise_fault_vibration: float = 0.08  # g


@dataclass
class TierCTurboParameters:
    """
    Tier C / Surrogate — Turbocharger & Boost Control Parameters.
    Governs the coupled reduced-order work balance and REDUCED_ORDER_TCU_SURROGATE.
    All parameters are strictly cataloged with explicit provenance.
    """
    tcu_continuous_map_target_bar: float = 1.200
    tcu_takeoff_map_target_bar: float = 1.350
    critical_altitude_m: float = 4572.0
    tau_map_s: float = 0.35
    eta_turb_nominal: float = 0.65
    eta_comp_nominal: float = 0.70
    eta_mech_turbo: float = 0.98
    delta_p_exh_manifold_bar: float = 0.35
    c_p_exh: float = 1150.0
    gamma_exh: float = 1.33
    c_p_air: float = 1005.0
    gamma_air: float = 1.40
    intercooler_efficiency: float = 0.60
    tcu_kp: float = 1.8
    tcu_w_dot_max: float = 1.5

    provenance_metadata: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "tcu_continuous_map_target_bar": {
            "tag": "MODEL_CONFIGURATION_VALUE",
            "source": "Selected simulator operating target under Rotax OM-914 §2.1 and EASA TCDS E.122 upper continuous limit; not universal OEM limit"
        },
        "tcu_takeoff_map_target_bar": {
            "tag": "MODEL_CONFIGURATION_VALUE",
            "source": "Selected simulator operating target under Rotax OM-914 §2.1 and EASA TCDS E.122 upper takeoff limit; not universal OEM limit"
        },
        "critical_altitude_m": {
            "tag": "MODEL_CONFIGURATION_VALUE",
            "source": "Selected simulator operating configuration from Rotax OM-914 §2.1 (15,000 ft / 4572 m continuous rating); EASA TCDS E.122 lists 16,000 ft / 4875 m for TCU v4.3/v4.6. Documented divergence for surrogate configuration"
        },
        "tau_map_s": {
            "tag": "MODEL_CALIBRATION",
            "source": "Intake plenum filling/emptying lag time constant"
        },
        "eta_turb_nominal": {
            "tag": "MODEL_CALIBRATION",
            "source": "Surrogate radial turbine isentropic expansion efficiency"
        },
        "eta_comp_nominal": {
            "tag": "MODEL_CALIBRATION",
            "source": "Surrogate centrifugal compressor isentropic compression efficiency"
        },
        "eta_mech_turbo": {
            "tag": "MODEL_ASSUMPTION",
            "source": "Turbocharger shaft bearing mechanical efficiency"
        },
        "delta_p_exh_manifold_bar": {
            "tag": "MODEL_CALIBRATION",
            "source": "Modeled exhaust manifold backpressure surrogate above ambient"
        },
        "c_p_exh": {
            "tag": "MODEL_ASSUMPTION",
            "source": "Thermodynamic exhaust gas specific heat capacity at constant pressure"
        },
        "gamma_exh": {
            "tag": "MODEL_ASSUMPTION",
            "source": "Thermodynamic exhaust gas ratio of specific heats"
        },
        "c_p_air": {
            "tag": "MODEL_ASSUMPTION",
            "source": "Standard air specific heat capacity at constant pressure"
        },
        "gamma_air": {
            "tag": "MODEL_ASSUMPTION",
            "source": "Standard air ratio of specific heats (diatomic gas)"
        },
        "intercooler_efficiency": {
            "tag": "MODEL_ASSUMPTION",
            "source": "Charge-air cooler thermal effectiveness"
        },
        "tcu_kp": {
            "tag": "MODEL_CALIBRATION",
            "source": "Surrogate TCU proportional servo wastegate displacement gain"
        },
        "tcu_w_dot_max": {
            "tag": "MODEL_CALIBRATION",
            "source": "Surrogate TCU electromechanical servo rate limit (1/s)"
        }
    })


@dataclass
class TierCGearboxParameters:
    """
    Tier C / Reference — Reduction Gearbox & Propeller Dynamics.
    Models the 2.4286:1 (51:21) reduction ratio, drivetrain inertia reflection,
    and mechanical power conservation (P_eng_out >= P_prop).
    """
    reduction_ratio: float = 2.42857
    gearbox_efficiency: float = 0.975
    inertia_prop_kg_m2: float = 0.85
    inertia_eng_kg_m2: float = 0.12
    k_prop: float = 1.8e-3

    provenance_metadata: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "reduction_ratio": {
            "tag": "OEM_REFERENCE_VALUE",
            "source": "BRP-Rotax 914 Series Operators Manual Ed. 4 / Rev. 0, Section 2.1; 51/21 spur gear tooth ratio"
        },
        "gearbox_efficiency": {
            "tag": "MODEL_ASSUMPTION",
            "source": "Spur gearbox mechanical transmission efficiency with dog-clutch damper"
        },
        "inertia_prop_kg_m2": {
            "tag": "MODEL_ASSUMPTION",
            "source": "3-blade constant-pitch / ground-adjustable composite propeller moment of inertia"
        },
        "inertia_eng_kg_m2": {
            "tag": "MODEL_ASSUMPTION",
            "source": "Engine crankshaft, flywheel, and reciprocating mass rotational inertia"
        },
        "k_prop": {
            "tag": "MODEL_CALIBRATION",
            "source": "Propeller aerodynamic torque coefficient calibrated to absorb rated power at rated RPM"
        }
    })


@dataclass
class TierCCylinderParameters:
    """
    Tier C — Four-Cylinder Thermal & Mechanical Architecture.
    Models discrete cylinder heads (1-4-3-2 firing order) and deterministic bank variations.
    """
    num_cylinders: int = 4
    firing_order: str = "1-4-3-2"
    c_th_cylinder: float = 230.0
    bank_variation_factors: tuple = (0.98, 1.00, 1.03, 0.99)

    provenance_metadata: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "num_cylinders": {
            "tag": "OEM_REFERENCE_VALUE",
            "source": "BRP-Rotax 914 Series Operators Manual Ed. 4 / Rev. 0, Section 2.1"
        },
        "firing_order": {
            "tag": "OEM_REFERENCE_VALUE",
            "source": "BRP-Rotax 914 Series Operators Manual Ed. 4 / Rev. 0, Section 2.1"
        },
        "c_th_cylinder": {
            "tag": "MODEL_CALIBRATION",
            "source": "Individual cylinder head thermal capacitance (920 J/K total / 4 cylinders)"
        },
        "bank_variation_factors": {
            "tag": "MODEL_ASSUMPTION",
            "source": "Deterministic boxer engine bank thermal variation factors (front/rear ram air and carburetor distribution)"
        }
    })


@dataclass
class TierCCoolingLoopParameters:
    """
    Tier C — REDUCED_ORDER_COOLING_SURROGATE.
    Lumped liquid cooling loop with thermostat opening and radiator heat dissipation.
    """
    c_coolant_j_per_k: float = 4500.0
    thermostat_temp_c: float = 80.0
    h_rad_base: float = 18.0
    k_rad_airspeed: float = 0.025
    h_head_to_coolant: float = 12.0

    provenance_metadata: Dict[str, Dict[str, str]] = field(default_factory=lambda: {
        "c_coolant_j_per_k": {
            "tag": "MODEL_CALIBRATION",
            "source": "Effective lumped cooling thermal capacitance for cylinder head jackets and head coolant mass (MODEL_CALIBRATION; not full 4 L system inventory)"
        },
        "thermostat_temp_c": {
            "tag": "OEM_REFERENCE_VALUE",
            "source": "BRP-Rotax 914 Series Installation Manual, Section 13.1 nominal thermostat opening threshold"
        },
        "h_rad_base": {
            "tag": "MODEL_CALIBRATION",
            "source": "Base radiator thermal conductance to ambient air"
        },
        "k_rad_airspeed": {
            "tag": "MODEL_CALIBRATION",
            "source": "Radiator convective cooling gain factor per m/s flight airspeed"
        },
        "h_head_to_coolant": {
            "tag": "MODEL_CALIBRATION",
            "source": "Convective heat transfer conductance per cylinder head into liquid jacket"
        }
    })


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
    tier_c_turbo: TierCTurboParameters = field(default_factory=TierCTurboParameters)
    tier_c_gearbox: TierCGearboxParameters = field(default_factory=TierCGearboxParameters)
    tier_c_cylinder: TierCCylinderParameters = field(default_factory=TierCCylinderParameters)
    tier_c_cooling: TierCCoolingLoopParameters = field(default_factory=TierCCoolingLoopParameters)

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
            "tier_c_turbo": {k: v for k, v in vars(self.tier_c_turbo).items() if k != "provenance_metadata"},
            "tier_c_gearbox": {k: v for k, v in vars(self.tier_c_gearbox).items() if k != "provenance_metadata"},
            "tier_c_cylinder": {k: v for k, v in vars(self.tier_c_cylinder).items() if k != "provenance_metadata"},
            "tier_c_cooling": {k: v for k, v in vars(self.tier_c_cooling).items() if k != "provenance_metadata"},
            "default_dt": self.default_dt,
            "random_seed": self.random_seed,
            "provenance_version": self.provenance_version,
        }

