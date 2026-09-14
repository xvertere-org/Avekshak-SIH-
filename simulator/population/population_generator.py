"""
Population Generator for Rotax 914 Grey-Box Engine Ensemble.

Orchestrates batch and streaming generation of physically constrained synthetic engine profiles,
deterministic train/val/test engine-ID partitioning (preventing row-level leakage),
and correlated physical parameter sampling.

CRITICAL PRINCIPLE:
This is a parameterized simulator ensemble for robustness testing of the Digital Twin,
NOT an ML-falsified dataset generator.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Iterator, Tuple
import math
import numpy as np

from simulator.population.parameter_distributions import (
    ProvenanceTag,
    POPULATION_DISTRIBUTIONS,
)
from simulator.population.engine_profile import (
    EngineProfile,
    ThermalProfile,
    LubricationProfile,
    TurbochargerProfile,
    CombustionProfile,
    MechanicalProfile,
    SensorProfile,
    CylinderProfile,
)
from simulator.population.mission_profiles import (
    CanonicalMission,
    MissionTrajectory,
    build_canonical_trajectory,
)
from simulator.population.validation import (
    validate_engine_profile,
    validate_telemetry_stream,
)
from simulator.engine_simulator import EngineSimulator
from telemetry.schema import TelemetryRecord, EngineConfig


@dataclass
class PopulationConfig:
    """
    Configuration governing the synthetic engine population generator.
    """
    population_id: str = "POP_ROTAX_914_PHASE7"
    num_engines: int = 100
    seed: int = 42
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    enable_correlations: bool = True
    enable_cylinder_variations: bool = True
    enable_sensor_variations: bool = True


class PopulationGenerator:
    """
    Batch and streaming generator of Rotax 914 synthetic engine populations.
    """

    def __init__(self, config: Optional[PopulationConfig] = None):
        self.config = config or PopulationConfig()
        # SeedSequence ensures mathematically independent, reproducible child seeds
        self.seed_seq = np.random.SeedSequence(self.config.seed)

    def _determine_split(self, engine_index: int) -> str:
        """
        Assign an engine strictly to train, val, or test based on engine index.
        Guarantees engine-level isolation: an engine in train never appears in test!
        """
        total = self.config.num_engines
        train_end = int(round(total * self.config.train_ratio))
        val_end = train_end + int(round(total * self.config.val_ratio))

        if engine_index < train_end:
            return "train"
        elif engine_index < val_end:
            return "val"
        else:
            return "test"

    def generate_single_profile(self, engine_index: int) -> EngineProfile:
        """
        Deterministically generate a single, validated EngineProfile.
        """
        engine_id = f"ENG_{self.config.population_id}_{engine_index:04d}"
        split = self._determine_split(engine_index)

        # Derive independent RNG for this specific engine
        engine_seed_seq = self.seed_seq.spawn(self.config.num_engines)[engine_index]
        engine_rng = np.random.default_rng(engine_seed_seq)
        engine_seed = int(engine_rng.integers(0, 2**31 - 1))

        # Sample base parameter values from POPULATION_DISTRIBUTIONS
        sampled_params: Dict[str, float] = {}
        for name, dist in POPULATION_DISTRIBUTIONS.items():
            sampled_params[name] = dist.sample(engine_rng)

        # Apply physically meaningful correlations if enabled
        if self.config.enable_correlations:
            # 1. Thermal mass & liquid coolant mass correlation (r ≈ 0.45)
            # Higher head thermal capacitance slightly correlates with higher radiator/coolant capacity
            z_th = (sampled_params["c_th_cht"] - POPULATION_DISTRIBUTIONS["c_th_cht"].nominal) / POPULATION_DISTRIBUTIONS["c_th_cht"].std_dev
            c_cool_nom = POPULATION_DISTRIBUTIONS["c_coolant_j_per_k"].nominal
            c_cool_std = POPULATION_DISTRIBUTIONS["c_coolant_j_per_k"].std_dev
            correlated_cool = c_cool_nom + c_cool_std * (0.45 * z_th + math.sqrt(1 - 0.45**2) * float(engine_rng.normal()))
            sampled_params["c_coolant_j_per_k"] = float(np.clip(
                correlated_cool,
                POPULATION_DISTRIBUTIONS["c_coolant_j_per_k"].lower_bound,
                POPULATION_DISTRIBUTIONS["c_coolant_j_per_k"].upper_bound,
            ))

            # 2. Static friction & oil heat transfer correlation (r ≈ 0.40)
            z_fric = (sampled_params["torque_fric_static"] - POPULATION_DISTRIBUTIONS["torque_fric_static"].nominal) / POPULATION_DISTRIBUTIONS["torque_fric_static"].std_dev
            h_oil_nom = POPULATION_DISTRIBUTIONS["h_oil_cool"].nominal
            h_oil_std = POPULATION_DISTRIBUTIONS["h_oil_cool"].std_dev
            correlated_h_oil = h_oil_nom + h_oil_std * (0.40 * z_fric + math.sqrt(1 - 0.40**2) * float(engine_rng.normal()))
            sampled_params["h_oil_cool"] = float(np.clip(
                correlated_h_oil,
                POPULATION_DISTRIBUTIONS["h_oil_cool"].lower_bound,
                POPULATION_DISTRIBUTIONS["h_oil_cool"].upper_bound,
            ))

            # 3. Combustion efficiency multiplier & fuel consumption slope (r ≈ -0.35)
            # Higher combustion efficiency slightly correlates with lower BSFC slope
            z_comb = (sampled_params["combustion_efficiency_multiplier"] - POPULATION_DISTRIBUTIONS["combustion_efficiency_multiplier"].nominal) / POPULATION_DISTRIBUTIONS["combustion_efficiency_multiplier"].std_dev
            a_fuel_nom = POPULATION_DISTRIBUTIONS["a_fuel_kg_per_j"].nominal
            a_fuel_std = POPULATION_DISTRIBUTIONS["a_fuel_kg_per_j"].std_dev
            correlated_afuel = a_fuel_nom + a_fuel_std * (-0.35 * z_comb + math.sqrt(1 - 0.35**2) * float(engine_rng.normal()))
            sampled_params["a_fuel_kg_per_j"] = float(np.clip(
                correlated_afuel,
                POPULATION_DISTRIBUTIONS["a_fuel_kg_per_j"].lower_bound,
                POPULATION_DISTRIBUTIONS["a_fuel_kg_per_j"].upper_bound,
            ))

        # Sensor persistent calibration bias (if enabled: zero-mean bounded offset)
        sensor_bias: Dict[str, float] = {
            "sensor_bias_rpm": 0.0,
            "sensor_bias_cht": 0.0,
            "sensor_bias_egt": 0.0,
            "sensor_bias_oil_temp": 0.0,
            "sensor_bias_oil_press": 0.0,
            "sensor_bias_fuel_flow": 0.0,
            "sensor_bias_vibration": 0.0,
        }
        if self.config.enable_sensor_variations:
            sensor_bias["sensor_bias_rpm"] = float(np.clip(engine_rng.normal(0.0, 1.2), -3.0, 3.0))
            sensor_bias["sensor_bias_cht"] = float(np.clip(engine_rng.normal(0.0, 0.15), -0.4, 0.4))
            sensor_bias["sensor_bias_egt"] = float(np.clip(engine_rng.normal(0.0, 0.6), -1.5, 1.5))
            sensor_bias["sensor_bias_oil_temp"] = float(np.clip(engine_rng.normal(0.0, 0.12), -0.3, 0.3))
            sensor_bias["sensor_bias_oil_press"] = float(np.clip(engine_rng.normal(0.0, 0.008), -0.02, 0.02))
            sensor_bias["sensor_bias_fuel_flow"] = float(np.clip(engine_rng.normal(0.0, 0.04), -0.1, 0.1))
            sensor_bias["sensor_bias_vibration"] = float(np.clip(engine_rng.normal(0.0, 0.004), -0.01, 0.01))

        # Cylinder-to-cylinder variations (bounded ±2.5%, normalized to exact sum of 4.0)
        bank_factors = (0.98, 1.00, 1.03, 0.99)
        comb_factors = (1.00, 1.00, 1.00, 1.00)
        if self.config.enable_cylinder_variations:
            # Perturb bank thermal factors
            raw_bank = np.array(bank_factors) + engine_rng.normal(0.0, 0.008, size=4)
            # Normalize so mean is exactly 1.0
            raw_bank = raw_bank - (np.mean(raw_bank) - 1.0)
            raw_bank = np.clip(raw_bank, 0.94, 1.06)
            bank_factors = (float(raw_bank[0]), float(raw_bank[1]), float(raw_bank[2]), float(raw_bank[3]))

            # Perturb combustion factors
            raw_comb = np.array(comb_factors) + engine_rng.normal(0.0, 0.008, size=4)
            raw_comb = raw_comb - (np.mean(raw_comb) - 1.0)
            raw_comb = np.clip(raw_comb, 0.95, 1.05)
            comb_factors = (float(raw_comb[0]), float(raw_comb[1]), float(raw_comb[2]), float(raw_comb[3]))

        # Assemble typed profile
        thermal = ThermalProfile(
            c_th_cht=sampled_params["c_th_cht"],
            h_cool_base=sampled_params["h_cool_base"],
            q_gen_fraction=sampled_params["q_gen_fraction"],
            t_egt_base_c=sampled_params["t_egt_base_c"],
            k_egt_load=sampled_params["k_egt_load"],
            tau_egt_s=sampled_params["tau_egt_s"],
            c_coolant_j_per_k=sampled_params["c_coolant_j_per_k"],
            h_rad_base=sampled_params["h_rad_base"],
        )
        lubrication = LubricationProfile(
            c_oil=sampled_params["c_oil"],
            h_oil_cool=sampled_params["h_oil_cool"],
            oil_press_base_bar=sampled_params["oil_press_base_bar"],
            k_oil_p_rpm=sampled_params["k_oil_p_rpm"],
            k_oil_p_temp=sampled_params["k_oil_p_temp"],
        )
        turbocharger = TurbochargerProfile(
            tcu_continuous_map_target_bar=sampled_params["tcu_continuous_map_target_bar"],
            tau_map_s=sampled_params["tau_map_s"],
            eta_turb_nominal=sampled_params["eta_turb_nominal"],
            eta_comp_nominal=sampled_params["eta_comp_nominal"],
            intercooler_efficiency=sampled_params["intercooler_efficiency"],
        )
        combustion = CombustionProfile(
            a_fuel_kg_per_j=sampled_params["a_fuel_kg_per_j"],
            b_fuel_kg_per_s=sampled_params["b_fuel_kg_per_s"],
            combustion_efficiency_multiplier=sampled_params["combustion_efficiency_multiplier"],
        )
        mechanical = MechanicalProfile(
            rpm_idle=sampled_params["rpm_idle"] if "rpm_idle" in sampled_params else 1400.0,
            inertia_kg_m2=sampled_params["inertia_kg_m2"],
            k_load=sampled_params["k_load"],
            k_fric_linear=sampled_params["k_fric_linear"],
            torque_fric_static=sampled_params["torque_fric_static"],
            vib_order1_base_g=sampled_params["vib_order1_base_g"],
            vib_order2_base_g=sampled_params["vib_order2_base_g"],
            vib_noise_std_g=sampled_params["vib_noise_std_g"],
        )
        sensors = SensorProfile(
            sensor_noise_rpm=sampled_params["sensor_noise_rpm"],
            sensor_noise_cht=sampled_params["sensor_noise_cht"],
            sensor_noise_egt=sampled_params["sensor_noise_egt"],
            sensor_noise_oil_temp=sampled_params["sensor_noise_oil_temp"],
            sensor_noise_oil_press=sampled_params["sensor_noise_oil_press"],
            sensor_noise_fuel_flow=sampled_params["sensor_noise_fuel_flow"],
            sensor_noise_vibration=sampled_params["sensor_noise_vibration"],
            sensor_bias_rpm=sensor_bias["sensor_bias_rpm"],
            sensor_bias_cht=sensor_bias["sensor_bias_cht"],
            sensor_bias_egt=sensor_bias["sensor_bias_egt"],
            sensor_bias_oil_temp=sensor_bias["sensor_bias_oil_temp"],
            sensor_bias_oil_press=sensor_bias["sensor_bias_oil_press"],
            sensor_bias_fuel_flow=sensor_bias["sensor_bias_fuel_flow"],
            sensor_bias_vibration=sensor_bias["sensor_bias_vibration"],
        )
        cylinders = CylinderProfile(
            num_cylinders=4,
            firing_order="1-4-3-2",
            bank_variation_factors=bank_factors,
            combustion_factors=comb_factors,
        )

        provenance_map = {name: dist.provenance.value for name, dist in POPULATION_DISTRIBUTIONS.items()}

        profile = EngineProfile(
            population_id=self.config.population_id,
            engine_instance_id=engine_id,
            seed=engine_seed,
            split=split,
            thermal=thermal,
            lubrication=lubrication,
            turbocharger=turbocharger,
            combustion=combustion,
            mechanical=mechanical,
            sensors=sensors,
            cylinders=cylinders,
            provenance_map=provenance_map,
        )

        # Enforce validation before release
        val_res = validate_engine_profile(profile)
        if not val_res.is_valid:
            raise ValueError(f"Generated profile {engine_id} failed physical validation: {val_res.errors}")

        return profile

    def iter_profiles(self) -> Iterator[EngineProfile]:
        """
        Streaming generator yielding EngineProfile instances one by one.
        Memory-efficient for populations of 1,000+ engines.
        """
        for i in range(self.config.num_engines):
            yield self.generate_single_profile(i)

    def generate_population(self) -> List[EngineProfile]:
        """
        Generate all engine profiles in memory.
        """
        return list(self.iter_profiles())


def simulate_engine_mission(
    profile: EngineProfile,
    mission: CanonicalMission,
    fault_state: Optional[Any] = None,
    dt: float = 0.5,
    duration_scale: float = 1.0,
) -> List[TelemetryRecord]:
    """
    Run an EngineProfile through a CanonicalMission using the authoritative EngineSimulator.

    Maintains clean separation: observable telemetry channels are generated without
    leaking offline ground-truth labels to runtime models.
    """
    sim_config = profile.to_simulator_config()
    sim = EngineSimulator(
        engine_config=EngineConfig(engine_id=profile.engine_instance_id),
        sim_config=sim_config,
        seed=profile.seed,
    )
    sim.reset(seed=profile.seed)

    trajectory = build_canonical_trajectory(
        mission=mission,
        seed=profile.seed,
        dt=dt,
        duration_scale=duration_scale,
    )
    steps = trajectory.generate_steps()

    records: List[TelemetryRecord] = []
    for step in steps:
        rec = sim.step(
            mission_config=step,
            time_step=dt,
            fault_state=fault_state,
        )
        records.append(rec)

    # Validate output stream
    val = validate_telemetry_stream(records)
    if not val.is_valid:
        raise ValueError(f"Simulation of {profile.engine_instance_id} produced invalid telemetry: {val.errors}")

    return records
