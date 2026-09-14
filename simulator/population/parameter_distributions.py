"""
Parameter Distributions and Sampling Engine for Rotax 914 Synthetic Engine Population.

Governs bounded, physically plausible statistical variations around calibrated baseline
parameters, explicitly tracking provenance, physical units, and engineering rationales.

DISCLAIMER:
All distributions defined herein represent synthetic manufacturing tolerances and model
uncertainty for grey-box simulation. They do NOT represent proprietary OEM fleet statistics.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
import math
import numpy as np


class ProvenanceTag(str, Enum):
    """Authoritative classification tag for model and population parameters."""
    OEM_SUPPORTED = "OEM_SUPPORTED"
    MODEL_CALIBRATION = "MODEL_CALIBRATION"
    ENGINEERING_HEURISTIC = "ENGINEERING_HEURISTIC"
    SYNTHETIC_VARIATION = "SYNTHETIC_VARIATION"


class DistributionType(str, Enum):
    """Statistical distribution family for synthetic parameter sampling."""
    TRUNCATED_NORMAL = "TRUNCATED_NORMAL"
    LOG_NORMAL = "LOG_NORMAL"
    UNIFORM = "UNIFORM"
    FIXED = "FIXED"


@dataclass(frozen=True)
class ParameterDistribution:
    """
    Typed, bounded parameter distribution with explicit provenance and rationale.

    Distinguishes nominal parameter provenance (OEM_SUPPORTED, MODEL_CALIBRATION,
    or ENGINEERING_HEURISTIC) from population variation provenance (SYNTHETIC_VARIATION).
    """
    name: str
    nominal: float
    lower_bound: float
    upper_bound: float
    units: str
    provenance: ProvenanceTag
    rationale: str
    distribution_type: DistributionType = DistributionType.TRUNCATED_NORMAL
    std_dev: float = 0.0
    log_sigma: float = 0.0
    nominal_provenance: Optional[ProvenanceTag] = None
    variation_provenance: ProvenanceTag = ProvenanceTag.SYNTHETIC_VARIATION

    def __post_init__(self):
        if self.nominal_provenance is None:
            object.__setattr__(self, "nominal_provenance", self.provenance)

    def sample(self, rng: np.random.Generator) -> float:
        """
        Sample a scalar parameter value strictly adhering to distribution bounds.

        Rejection sampling ensures accurate truncated density without artificial boundary piling.
        """
        if self.distribution_type == DistributionType.FIXED or self.std_dev <= 0.0:
            return float(self.nominal)

        if self.distribution_type == DistributionType.UNIFORM:
            val = float(rng.uniform(self.lower_bound, self.upper_bound))
            return float(np.clip(val, self.lower_bound, self.upper_bound))

        if self.distribution_type == DistributionType.LOG_NORMAL:
            # Multiplicative positive parameter: x = nominal * exp(N(0, sigma))
            sigma = self.log_sigma if self.log_sigma > 0.0 else (self.std_dev / max(1e-6, self.nominal))
            for _ in range(50):
                mult = math.exp(float(rng.normal(0.0, sigma)))
                candidate = self.nominal * mult
                if self.lower_bound <= candidate <= self.upper_bound:
                    return candidate
            return float(np.clip(candidate, self.lower_bound, self.upper_bound))

        # Default: TRUNCATED_NORMAL
        # Rejection sampling within [lower_bound, upper_bound]
        for _ in range(50):
            candidate = float(rng.normal(self.nominal, self.std_dev))
            if self.lower_bound <= candidate <= self.upper_bound:
                return candidate
        # Fallback clip if 50 rejections exceed bounds (probability < 1e-12 for standard sigmas)
        return float(np.clip(candidate, self.lower_bound, self.upper_bound))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "nominal": self.nominal,
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "units": self.units,
            "provenance": self.provenance.value,
            "nominal_provenance": self.nominal_provenance.value if self.nominal_provenance else self.provenance.value,
            "variation_provenance": self.variation_provenance.value,
            "rationale": self.rationale,
            "distribution_type": self.distribution_type.value,
            "std_dev": self.std_dev,
            "log_sigma": self.log_sigma,
        }



# ==============================================================================
# AUTHORITATIVE CATALOG OF POPULATION PARAMETER DISTRIBUTIONS
# ==============================================================================

POPULATION_DISTRIBUTIONS: Dict[str, ParameterDistribution] = {
    # --- Thermal Subsystem ---
    "c_th_cht": ParameterDistribution(
        name="c_th_cht",
        nominal=920.0,
        lower_bound=800.0,
        upper_bound=1050.0,
        std_dev=45.0,
        units="J/K",
        provenance=ProvenanceTag.MODEL_CALIBRATION,
        rationale="Lumped cylinder head thermal capacitance variation from casting mass and fin surface area tolerances.",
    ),
    "h_cool_base": ParameterDistribution(
        name="h_cool_base",
        nominal=16.0,
        lower_bound=13.5,
        upper_bound=18.5,
        std_dev=1.0,
        units="W/K",
        provenance=ProvenanceTag.MODEL_CALIBRATION,
        rationale="Base convective cooling conductance variation due to cowling baffle fit and aerodynamic ducting tolerances.",
    ),
    "q_gen_fraction": ParameterDistribution(
        name="q_gen_fraction",
        nominal=0.048,
        lower_bound=0.042,
        upper_bound=0.054,
        std_dev=0.002,
        units="dimensionless",
        provenance=ProvenanceTag.MODEL_CALIBRATION,
        rationale="Fraction of fuel combustion energy transferred into cylinder heads.",
    ),
    "t_egt_base_c": ParameterDistribution(
        name="t_egt_base_c",
        nominal=520.0,
        lower_bound=490.0,
        upper_bound=550.0,
        std_dev=12.0,
        units="deg_C",
        provenance=ProvenanceTag.MODEL_CALIBRATION,
        rationale="Base exhaust gas temperature at low load from exhaust manifold fabrication tolerances.",
    ),
    "k_egt_load": ParameterDistribution(
        name="k_egt_load",
        nominal=210.0,
        lower_bound=190.0,
        upper_bound=230.0,
        std_dev=8.0,
        units="deg_C",
        provenance=ProvenanceTag.MODEL_CALIBRATION,
        rationale="Sensitivity of exhaust gas temperature rise per unit engine load.",
    ),
    "tau_egt_s": ParameterDistribution(
        name="tau_egt_s",
        nominal=2.5,
        lower_bound=2.0,
        upper_bound=3.2,
        std_dev=0.22,
        units="s",
        provenance=ProvenanceTag.MODEL_CALIBRATION,
        rationale="Exhaust thermocouple sheath thermal conduction lag time constant.",
    ),
    "c_coolant_j_per_k": ParameterDistribution(
        name="c_coolant_j_per_k",
        nominal=4500.0,
        lower_bound=3800.0,
        upper_bound=5200.0,
        std_dev=250.0,
        units="J/K",
        provenance=ProvenanceTag.MODEL_CALIBRATION,
        rationale="Effective lumped liquid coolant circuit thermal mass variation.",
    ),
    "h_rad_base": ParameterDistribution(
        name="h_rad_base",
        nominal=18.0,
        lower_bound=15.0,
        upper_bound=21.0,
        std_dev=1.2,
        units="W/K",
        provenance=ProvenanceTag.MODEL_CALIBRATION,
        rationale="Radiator core air-side base thermal conductance to ambient.",
    ),

    # --- Lubrication Subsystem ---
    "c_oil": ParameterDistribution(
        name="c_oil",
        nominal=1350.0,
        lower_bound=1150.0,
        upper_bound=1550.0,
        std_dev=75.0,
        units="J/K",
        provenance=ProvenanceTag.MODEL_CALIBRATION,
        rationale="Oil reservoir and circulation passage effective thermal capacitance.",
    ),
    "h_oil_cool": ParameterDistribution(
        name="h_oil_cool",
        nominal=20.0,
        lower_bound=17.0,
        upper_bound=23.0,
        std_dev=1.2,
        units="W/K",
        provenance=ProvenanceTag.MODEL_CALIBRATION,
        rationale="Oil cooler radiator heat dissipation conductance.",
    ),
    "oil_press_base_bar": ParameterDistribution(
        name="oil_press_base_bar",
        nominal=1.0,
        lower_bound=0.85,
        upper_bound=1.15,
        std_dev=0.06,
        units="bar",
        provenance=ProvenanceTag.MODEL_CALIBRATION,
        rationale="Base static oil pressure at idle from pressure relief valve spring preload tolerance.",
    ),
    "k_oil_p_rpm": ParameterDistribution(
        name="k_oil_p_rpm",
        nominal=0.00085,
        lower_bound=0.00075,
        upper_bound=0.00095,
        std_dev=0.00004,
        units="bar/RPM",
        provenance=ProvenanceTag.MODEL_CALIBRATION,
        rationale="Oil pump positive-displacement pressure delivery slope vs engine RPM.",
    ),
    "k_oil_p_temp": ParameterDistribution(
        name="k_oil_p_temp",
        nominal=0.008,
        lower_bound=0.006,
        upper_bound=0.010,
        std_dev=0.0008,
        units="bar/deg_C",
        provenance=ProvenanceTag.MODEL_CALIBRATION,
        rationale="Oil viscosity thinning effect on hydraulic line pressure.",
    ),

    # --- Combustion Subsystem ---
    "a_fuel_kg_per_j": ParameterDistribution(
        name="a_fuel_kg_per_j",
        nominal=6.8e-8,
        lower_bound=6.2e-8,
        upper_bound=7.4e-8,
        std_dev=2.4e-9,
        units="kg/J",
        provenance=ProvenanceTag.MODEL_CALIBRATION,
        rationale="Brake specific fuel consumption slope variation (Willans line parameter).",
    ),
    "b_fuel_kg_per_s": ParameterDistribution(
        name="b_fuel_kg_per_s",
        nominal=0.00032,
        lower_bound=0.00028,
        upper_bound=0.00036,
        std_dev=1.5e-5,
        units="kg/s",
        provenance=ProvenanceTag.MODEL_CALIBRATION,
        rationale="Idle fuel consumption rate from carburetor idle jet metering tolerances.",
    ),
    "combustion_efficiency_multiplier": ParameterDistribution(
        name="combustion_efficiency_multiplier",
        nominal=1.0,
        lower_bound=0.95,
        upper_bound=1.05,
        std_dev=0.02,
        units="dimensionless",
        provenance=ProvenanceTag.SYNTHETIC_VARIATION,
        rationale="Overall engine combustion efficiency variation around baseline nominal.",
    ),

    # --- Turbocharger Subsystem ---
    "tcu_continuous_map_target_bar": ParameterDistribution(
        name="tcu_continuous_map_target_bar",
        nominal=1.200,
        lower_bound=1.170,
        upper_bound=1.230,
        std_dev=0.012,
        units="bar",
        provenance=ProvenanceTag.MODEL_CALIBRATION,
        rationale="Surrogate TCU continuous manifold boost regulation setpoint tolerance.",
    ),
    "tau_map_s": ParameterDistribution(
        name="tau_map_s",
        nominal=0.35,
        lower_bound=0.28,
        upper_bound=0.45,
        std_dev=0.03,
        units="s",
        provenance=ProvenanceTag.MODEL_CALIBRATION,
        rationale="Intake manifold / airbox pneumatic volume filling lag time constant.",
    ),
    "eta_turb_nominal": ParameterDistribution(
        name="eta_turb_nominal",
        nominal=0.65,
        lower_bound=0.60,
        upper_bound=0.70,
        std_dev=0.02,
        units="dimensionless",
        provenance=ProvenanceTag.MODEL_CALIBRATION,
        rationale="Surrogate radial turbine isentropic expansion efficiency tolerance.",
    ),
    "eta_comp_nominal": ParameterDistribution(
        name="eta_comp_nominal",
        nominal=0.70,
        lower_bound=0.65,
        upper_bound=0.75,
        std_dev=0.02,
        units="dimensionless",
        provenance=ProvenanceTag.MODEL_CALIBRATION,
        rationale="Surrogate centrifugal compressor isentropic compression efficiency tolerance.",
    ),
    "intercooler_efficiency": ParameterDistribution(
        name="intercooler_efficiency",
        nominal=0.60,
        lower_bound=0.52,
        upper_bound=0.68,
        std_dev=0.03,
        units="dimensionless",
        provenance=ProvenanceTag.MODEL_CALIBRATION,
        rationale="Charge-air intercooler thermal effectiveness variation.",
    ),

    # --- Mechanical Subsystem ---
    "inertia_kg_m2": ParameterDistribution(
        name="inertia_kg_m2",
        nominal=0.28,
        lower_bound=0.25,
        upper_bound=0.32,
        std_dev=0.012,
        units="kg*m^2",
        provenance=ProvenanceTag.MODEL_CALIBRATION,
        rationale="Rotating assembly plus propeller effective moment of inertia tolerance.",
    ),
    "k_load": ParameterDistribution(
        name="k_load",
        nominal=3.036e-4,
        lower_bound=2.70e-4,
        upper_bound=3.40e-4,
        std_dev=1.4e-5,
        units="N*m/(rad/s)^2",
        provenance=ProvenanceTag.MODEL_CALIBRATION,
        rationale="Propeller aerodynamic absorption load coefficient tolerance.",
    ),
    "k_fric_linear": ParameterDistribution(
        name="k_fric_linear",
        nominal=0.015,
        lower_bound=0.012,
        upper_bound=0.018,
        std_dev=0.0011,
        units="N*m*s/rad",
        provenance=ProvenanceTag.MODEL_CALIBRATION,
        rationale="Viscous bearing friction coefficient variation across engines.",
    ),
    "torque_fric_static": ParameterDistribution(
        name="torque_fric_static",
        nominal=3.5,
        lower_bound=2.8,
        upper_bound=4.2,
        std_dev=0.25,
        units="N*m",
        provenance=ProvenanceTag.MODEL_CALIBRATION,
        rationale="Static friction and ring pumping friction torque variation.",
    ),
    "vib_order1_base_g": ParameterDistribution(
        name="vib_order1_base_g",
        nominal=0.32,
        lower_bound=0.26,
        upper_bound=0.38,
        std_dev=0.024,
        units="g",
        provenance=ProvenanceTag.MODEL_CALIBRATION,
        rationale="1x crankshaft rotational unbalance baseline amplitude variation.",
    ),
    "vib_order2_base_g": ParameterDistribution(
        name="vib_order2_base_g",
        nominal=0.22,
        lower_bound=0.18,
        upper_bound=0.26,
        std_dev=0.016,
        units="g",
        provenance=ProvenanceTag.MODEL_CALIBRATION,
        rationale="2x reciprocating unbalance baseline amplitude variation.",
    ),
    "vib_noise_std_g": ParameterDistribution(
        name="vib_noise_std_g",
        nominal=0.06,
        lower_bound=0.045,
        upper_bound=0.075,
        std_dev=0.006,
        units="g",
        provenance=ProvenanceTag.MODEL_CALIBRATION,
        rationale="Broadband background vibration process noise standard deviation.",
    ),

    # --- Sensor Measurement Noise Standards ---
    "sensor_noise_rpm": ParameterDistribution(
        name="sensor_noise_rpm",
        nominal=4.0,
        lower_bound=2.5,
        upper_bound=6.0,
        std_dev=0.6,
        units="RPM",
        provenance=ProvenanceTag.SYNTHETIC_VARIATION,
        rationale="Crankshaft speed sensor measurement noise std dev variation.",
    ),
    "sensor_noise_cht": ParameterDistribution(
        name="sensor_noise_cht",
        nominal=0.4,
        lower_bound=0.25,
        upper_bound=0.65,
        std_dev=0.07,
        units="deg_C",
        provenance=ProvenanceTag.SYNTHETIC_VARIATION,
        rationale="CHT thermocouple measurement noise std dev variation.",
    ),
    "sensor_noise_egt": ParameterDistribution(
        name="sensor_noise_egt",
        nominal=1.8,
        lower_bound=1.0,
        upper_bound=2.8,
        std_dev=0.32,
        units="deg_C",
        provenance=ProvenanceTag.SYNTHETIC_VARIATION,
        rationale="EGT thermocouple measurement noise std dev variation.",
    ),
    "sensor_noise_oil_temp": ParameterDistribution(
        name="sensor_noise_oil_temp",
        nominal=0.3,
        lower_bound=0.18,
        upper_bound=0.45,
        std_dev=0.05,
        units="deg_C",
        provenance=ProvenanceTag.SYNTHETIC_VARIATION,
        rationale="Oil temperature thermistor measurement noise std dev variation.",
    ),
    "sensor_noise_oil_press": ParameterDistribution(
        name="sensor_noise_oil_press",
        nominal=0.025,
        lower_bound=0.015,
        upper_bound=0.040,
        std_dev=0.004,
        units="bar",
        provenance=ProvenanceTag.SYNTHETIC_VARIATION,
        rationale="Oil pressure piezoresistive sensor measurement noise std dev variation.",
    ),
    "sensor_noise_fuel_flow": ParameterDistribution(
        name="sensor_noise_fuel_flow",
        nominal=0.15,
        lower_bound=0.09,
        upper_bound=0.25,
        std_dev=0.026,
        units="L/h",
        provenance=ProvenanceTag.SYNTHETIC_VARIATION,
        rationale="Turbine fuel flow transducer measurement noise std dev variation.",
    ),
    "sensor_noise_vibration": ParameterDistribution(
        name="sensor_noise_vibration",
        nominal=0.015,
        lower_bound=0.009,
        upper_bound=0.025,
        std_dev=0.0026,
        units="g",
        provenance=ProvenanceTag.SYNTHETIC_VARIATION,
        rationale="Piezoelectric accelerometer measurement noise std dev variation.",
    ),
}
