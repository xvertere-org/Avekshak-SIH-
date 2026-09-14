"""
Engine Profile Representation for Rotax 914 Synthetic Engine Population.

Provides typed, immutable representations of individual synthetic engines,
including subsystem parameter groupings, bounded cylinder-to-cylinder variations,
sensor characteristics, parameter provenance, and translation into SimulatorConfig.

CRITICAL ARCHITECTURAL CONTRACT:
- Certified OEM Tier A constants (Bore, Stroke, Displacement, 4 cylinders, 9:1 CR, Gearbox 2.4286:1)
  remain STRICTLY FIXED.
- Varied parameters are restricted to calibrated grey-box Tier C/D parameters.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Tuple, Optional
import copy

from simulator.config import (
    SimulatorConfig,
    TierAParameters,
    TierCParameters,
    TierDParameters,
    TierCTurboParameters,
    TierCGearboxParameters,
    TierCCylinderParameters,
    TierCCoolingLoopParameters,
)
from simulator.population.parameter_distributions import (
    ProvenanceTag,
    POPULATION_DISTRIBUTIONS,
)


@dataclass(frozen=True)
class ThermalProfile:
    """Thermal subsystem parameters for an individual engine profile."""
    c_th_cht: float = 920.0
    h_cool_base: float = 16.0
    q_gen_fraction: float = 0.048
    t_egt_base_c: float = 520.0
    k_egt_load: float = 210.0
    tau_egt_s: float = 2.5
    c_coolant_j_per_k: float = 4500.0
    h_rad_base: float = 18.0


@dataclass(frozen=True)
class LubricationProfile:
    """Lubrication subsystem parameters for an individual engine profile."""
    c_oil: float = 1350.0
    h_oil_cool: float = 20.0
    oil_press_base_bar: float = 1.0
    k_oil_p_rpm: float = 0.00085
    k_oil_p_temp: float = 0.008


@dataclass(frozen=True)
class TurbochargerProfile:
    """Surrogate turbocharger & TCU parameters for an individual engine profile."""
    tcu_continuous_map_target_bar: float = 1.200
    tau_map_s: float = 0.35
    eta_turb_nominal: float = 0.65
    eta_comp_nominal: float = 0.70
    intercooler_efficiency: float = 0.60


@dataclass(frozen=True)
class CombustionProfile:
    """Combustion and fuel metering parameters for an individual engine profile."""
    a_fuel_kg_per_j: float = 6.8e-8
    b_fuel_kg_per_s: float = 0.00032
    combustion_efficiency_multiplier: float = 1.0


@dataclass(frozen=True)
class MechanicalProfile:
    """Mechanical and vibration parameters for an individual engine profile."""
    rpm_idle: float = 1400.0
    inertia_kg_m2: float = 0.28
    k_load: float = 3.036e-4
    k_fric_linear: float = 0.015
    torque_fric_static: float = 3.5
    vib_order1_base_g: float = 0.32
    vib_order2_base_g: float = 0.22
    vib_noise_std_g: float = 0.06


@dataclass(frozen=True)
class SensorProfile:
    """Sensor observation layer noise and calibration parameters."""
    sensor_noise_rpm: float = 4.0
    sensor_noise_cht: float = 0.4
    sensor_noise_egt: float = 1.8
    sensor_noise_oil_temp: float = 0.3
    sensor_noise_oil_press: float = 0.025
    sensor_noise_fuel_flow: float = 0.15
    sensor_noise_vibration: float = 0.015
    # Persistent baseline calibration bias (small natural zero-offset)
    sensor_bias_rpm: float = 0.0
    sensor_bias_cht: float = 0.0
    sensor_bias_egt: float = 0.0
    sensor_bias_oil_temp: float = 0.0
    sensor_bias_oil_press: float = 0.0
    sensor_bias_fuel_flow: float = 0.0
    sensor_bias_vibration: float = 0.0


@dataclass(frozen=True)
class CylinderProfile:
    """
    Cylinder-to-cylinder variation parameters (cylinders 1 to 4).
    Enforces 4-cylinder architecture, firing order 1-4-3-2, and energy conservation.
    """
    num_cylinders: int = 4
    firing_order: str = "1-4-3-2"
    # Bank/head thermal multipliers across cylinders 1..4 (average strictly 1.0)
    bank_variation_factors: Tuple[float, float, float, float] = (0.98, 1.00, 1.03, 0.99)
    # Relative combustion efficiency multipliers (average strictly 1.0)
    combustion_factors: Tuple[float, float, float, float] = (1.00, 1.00, 1.00, 1.00)


@dataclass(frozen=True)
class EngineProfile:
    """
    Authoritative, immutable profile for a single engine instance in the synthetic population.
    """
    population_id: str
    engine_instance_id: str
    seed: int
    split: str = "train"  # 'train', 'val', or 'test'
    thermal: ThermalProfile = field(default_factory=ThermalProfile)
    lubrication: LubricationProfile = field(default_factory=LubricationProfile)
    turbocharger: TurbochargerProfile = field(default_factory=TurbochargerProfile)
    combustion: CombustionProfile = field(default_factory=CombustionProfile)
    mechanical: MechanicalProfile = field(default_factory=MechanicalProfile)
    sensors: SensorProfile = field(default_factory=SensorProfile)
    cylinders: CylinderProfile = field(default_factory=CylinderProfile)
    provenance_map: Dict[str, str] = field(default_factory=dict)

    def to_simulator_config(self) -> SimulatorConfig:
        """
        Convert this EngineProfile into a complete, executable SimulatorConfig.

        Locks certified Tier A specifications while propagating varied Tier C/D parameters.
        """
        # Tier A: Absolute constants strictly preserved from Rotax 914 reference
        tier_a = TierAParameters()

        # Tier C: Overridden from profile
        tier_c = TierCParameters(
            rpm_idle=self.mechanical.rpm_idle,
            inertia_kg_m2=self.mechanical.inertia_kg_m2,
            k_load=self.mechanical.k_load,
            k_fric_linear=self.mechanical.k_fric_linear,
            torque_fric_static=self.mechanical.torque_fric_static,
            a_fuel_kg_per_j=self.combustion.a_fuel_kg_per_j,
            b_fuel_kg_per_s=self.combustion.b_fuel_kg_per_s,
            c_th_cht=self.thermal.c_th_cht,
            q_gen_fraction=self.thermal.q_gen_fraction,
            h_cool_base=self.thermal.h_cool_base,
            t_egt_base_c=self.thermal.t_egt_base_c,
            k_egt_load=self.thermal.k_egt_load,
            tau_egt_s=self.thermal.tau_egt_s,
            c_oil=self.lubrication.c_oil,
            h_oil_cool=self.lubrication.h_oil_cool,
            oil_press_base_bar=self.lubrication.oil_press_base_bar,
            k_oil_p_rpm=self.lubrication.k_oil_p_rpm,
            k_oil_p_temp=self.lubrication.k_oil_p_temp,
            vib_order1_base_g=self.mechanical.vib_order1_base_g,
            vib_order2_base_g=self.mechanical.vib_order2_base_g,
            vib_noise_std_g=self.mechanical.vib_noise_std_g,
            sensor_noise_rpm=self.sensors.sensor_noise_rpm,
            sensor_noise_cht=self.sensors.sensor_noise_cht,
            sensor_noise_egt=self.sensors.sensor_noise_egt,
            sensor_noise_oil_temp=self.sensors.sensor_noise_oil_temp,
            sensor_noise_oil_press=self.sensors.sensor_noise_oil_press,
            sensor_noise_fuel_flow=self.sensors.sensor_noise_fuel_flow,
            sensor_noise_vibration=self.sensors.sensor_noise_vibration,
        )

        # Tier C Turbo
        tier_c_turbo = TierCTurboParameters(
            tcu_continuous_map_target_bar=self.turbocharger.tcu_continuous_map_target_bar,
            tau_map_s=self.turbocharger.tau_map_s,
            eta_turb_nominal=self.turbocharger.eta_turb_nominal,
            eta_comp_nominal=self.turbocharger.eta_comp_nominal,
            intercooler_efficiency=self.turbocharger.intercooler_efficiency,
        )

        # Tier C Gearbox — 2.42857:1 reduction ratio strictly locked
        tier_c_gearbox = TierCGearboxParameters()

        # Tier C Cylinder — Bank variation factors applied
        tier_c_cylinder = TierCCylinderParameters(
            num_cylinders=self.cylinders.num_cylinders,
            firing_order=self.cylinders.firing_order,
            bank_variation_factors=self.cylinders.bank_variation_factors,
        )

        # Tier C Cooling Loop
        tier_c_cooling = TierCCoolingLoopParameters(
            c_coolant_j_per_k=self.thermal.c_coolant_j_per_k,
            h_rad_base=self.thermal.h_rad_base,
        )

        # Tier D: Baseline engineering assumptions
        tier_d = TierDParameters()

        return SimulatorConfig(
            tier_a=tier_a,
            tier_c=tier_c,
            tier_d=tier_d,
            tier_c_turbo=tier_c_turbo,
            tier_c_gearbox=tier_c_gearbox,
            tier_c_cylinder=tier_c_cylinder,
            tier_c_cooling=tier_c_cooling,
            random_seed=self.seed,
            provenance_version=f"phase7-pop-{self.population_id}",
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert engine profile to serializable dictionary."""
        return {
            "population_id": self.population_id,
            "engine_instance_id": self.engine_instance_id,
            "seed": self.seed,
            "split": self.split,
            "thermal": asdict(self.thermal),
            "lubrication": asdict(self.lubrication),
            "turbocharger": asdict(self.turbocharger),
            "combustion": asdict(self.combustion),
            "mechanical": asdict(self.mechanical),
            "sensors": asdict(self.sensors),
            "cylinders": {
                "num_cylinders": self.cylinders.num_cylinders,
                "firing_order": self.cylinders.firing_order,
                "bank_variation_factors": list(self.cylinders.bank_variation_factors),
                "combustion_factors": list(self.cylinders.combustion_factors),
            },
            "provenance_map": copy.deepcopy(self.provenance_map),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EngineProfile":
        """Reconstruct engine profile from serialized dictionary."""
        cyl_data = d.get("cylinders", {})
        return cls(
            population_id=d["population_id"],
            engine_instance_id=d["engine_instance_id"],
            seed=d["seed"],
            split=d.get("split", "train"),
            thermal=ThermalProfile(**d.get("thermal", {})),
            lubrication=LubricationProfile(**d.get("lubrication", {})),
            turbocharger=TurbochargerProfile(**d.get("turbocharger", {})),
            combustion=CombustionProfile(**d.get("combustion", {})),
            mechanical=MechanicalProfile(**d.get("mechanical", {})),
            sensors=SensorProfile(**d.get("sensors", {})),
            cylinders=CylinderProfile(
                num_cylinders=cyl_data.get("num_cylinders", 4),
                firing_order=cyl_data.get("firing_order", "1-4-3-2"),
                bank_variation_factors=tuple(cyl_data.get("bank_variation_factors", (0.98, 1.0, 1.03, 0.99))),
                combustion_factors=tuple(cyl_data.get("combustion_factors", (1.0, 1.0, 1.0, 1.0))),
            ),
            provenance_map=copy.deepcopy(d.get("provenance_map", {})),
        )
