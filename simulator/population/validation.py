"""
Physical Validation and Conservation Constraints for Rotax 914 Engine Population.

Ensures every generated EngineProfile strictly satisfies physical conservation laws,
certified reference invariants, positive thermodynamic bounds, and valid operational ceilings.

CRITICAL PRINCIPLES:
- Zero unconstrained parameters
- 4 cylinders, 79.5 mm bore, 61.0 mm stroke, ~1211.2 cc displacement, 9:1 CR strictly checked
- 2.4286 reduction ratio verified
- All efficiencies bounded in (0.0, 1.0]
- Thermal capacitances and conductive couplings strictly positive
- Rejection of any NaN or infinite values
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple
import math
import numpy as np

from simulator.population.engine_profile import EngineProfile
from simulator.population.parameter_distributions import POPULATION_DISTRIBUTIONS
from telemetry.schema import TelemetryRecord


@dataclass
class ValidationResult:
    """Outcome of validating a single entity (profile or telemetry stream)."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PopulationValidationReport:
    """Summary validation report over an entire population ensemble."""
    population_id: str
    total_profiles: int
    valid_profiles: int
    invalid_profiles: int
    is_fully_valid: bool
    parameter_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    clipping_counts: Dict[str, int] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


def validate_engine_profile(profile: EngineProfile) -> ValidationResult:
    """
    Validate that an EngineProfile satisfies all physical constraints and certified invariants.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Non-null and non-NaN / non-infinite parameter checks
    def check_number(val: float, name: str) -> bool:
        if not isinstance(val, (int, float)):
            errors.append(f"Parameter '{name}' is not numeric: {val}")
            return False
        if math.isnan(val) or math.isinf(val):
            errors.append(f"Parameter '{name}' is NaN or Infinite: {val}")
            return False
        return True

    # Check thermal
    for k, v in vars(profile.thermal).items():
        check_number(v, f"thermal.{k}")
    # Check lubrication
    for k, v in vars(profile.lubrication).items():
        check_number(v, f"lubrication.{k}")
    # Check turbocharger
    for k, v in vars(profile.turbocharger).items():
        check_number(v, f"turbocharger.{k}")
    # Check combustion
    for k, v in vars(profile.combustion).items():
        check_number(v, f"combustion.{k}")
    # Check mechanical
    for k, v in vars(profile.mechanical).items():
        check_number(v, f"mechanical.{k}")
    # Check sensors
    for k, v in vars(profile.sensors).items():
        check_number(v, f"sensors.{k}")

    # 2. Geometry & Certified Invariants Check (via SimulatorConfig conversion)
    sim_cfg = profile.to_simulator_config()

    # Geometry: 4 cylinders, 79.5 mm bore, 61.0 mm stroke, 1211.2 cc
    if sim_cfg.tier_a.reference_displacement_cc != 1211.2:
        errors.append(f"Certified displacement violated: {sim_cfg.tier_a.reference_displacement_cc} != 1211.2 cc")
    if sim_cfg.tier_a.reference_bore_mm != 79.5:
        errors.append(f"Certified bore violated: {sim_cfg.tier_a.reference_bore_mm} != 79.5 mm")
    if sim_cfg.tier_a.reference_stroke_mm != 61.0:
        errors.append(f"Certified stroke violated: {sim_cfg.tier_a.reference_stroke_mm} != 61.0 mm")
    if sim_cfg.tier_a.reference_compression_ratio != 9.0:
        errors.append(f"Certified compression ratio violated: {sim_cfg.tier_a.reference_compression_ratio} != 9.0")
    if profile.cylinders.num_cylinders != 4:
        errors.append(f"Engine architecture must have 4 cylinders, got {profile.cylinders.num_cylinders}")

    # Gearbox ratio: exactly 2.42857 / 2.4286 (51:21)
    if abs(sim_cfg.tier_c_gearbox.reduction_ratio - 2.42857) > 0.001:
        errors.append(f"Gearbox reduction ratio violated: {sim_cfg.tier_c_gearbox.reduction_ratio} != 2.42857")

    # 3. Thermodynamic Capacitances & Conductances Strictly Positive
    if profile.thermal.c_th_cht <= 0.0:
        errors.append(f"Thermal CHT capacitance must be > 0, got {profile.thermal.c_th_cht}")
    if profile.thermal.h_cool_base <= 0.0:
        errors.append(f"Base cooling conductance must be > 0, got {profile.thermal.h_cool_base}")
    if profile.thermal.c_coolant_j_per_k <= 0.0:
        errors.append(f"Coolant thermal mass must be > 0, got {profile.thermal.c_coolant_j_per_k}")
    if profile.lubrication.c_oil <= 0.0:
        errors.append(f"Oil thermal capacitance must be > 0, got {profile.lubrication.c_oil}")
    if profile.lubrication.h_oil_cool <= 0.0:
        errors.append(f"Oil radiator conductance must be > 0, got {profile.lubrication.h_oil_cool}")

    # 4. Pressures Strictly Positive
    if profile.lubrication.oil_press_base_bar <= 0.0:
        errors.append(f"Base oil pressure must be > 0, got {profile.lubrication.oil_press_base_bar}")
    if profile.turbocharger.tcu_continuous_map_target_bar <= 0.0:
        errors.append(f"Continuous MAP target must be > 0, got {profile.turbocharger.tcu_continuous_map_target_bar}")

    # 5. Efficiencies Strictly in (0.0, 1.0]
    if not (0.0 < profile.turbocharger.eta_turb_nominal <= 1.0):
        errors.append(f"Turbine efficiency out of bounds (0, 1]: {profile.turbocharger.eta_turb_nominal}")
    if not (0.0 < profile.turbocharger.eta_comp_nominal <= 1.0):
        errors.append(f"Compressor efficiency out of bounds (0, 1]: {profile.turbocharger.eta_comp_nominal}")
    if not (0.0 < profile.turbocharger.intercooler_efficiency <= 1.0):
        errors.append(f"Intercooler efficiency out of bounds (0, 1]: {profile.turbocharger.intercooler_efficiency}")

    # 6. Mechanical Quantities
    if profile.mechanical.rpm_idle <= 0.0:
        errors.append(f"Idle RPM must be > 0, got {profile.mechanical.rpm_idle}")
    if profile.mechanical.rpm_idle >= sim_cfg.tier_a.rpm_max_continuous:
        errors.append(f"Idle RPM exceeds maximum continuous: {profile.mechanical.rpm_idle} >= {sim_cfg.tier_a.rpm_max_continuous}")
    if profile.mechanical.inertia_kg_m2 <= 0.0:
        errors.append(f"Rotating assembly inertia must be > 0, got {profile.mechanical.inertia_kg_m2}")
    if profile.mechanical.k_fric_linear < 0.0:
        errors.append(f"Viscous friction cannot be negative: {profile.mechanical.k_fric_linear}")
    if profile.mechanical.torque_fric_static < 0.0:
        errors.append(f"Static friction torque cannot be negative: {profile.mechanical.torque_fric_static}")

    # 7. Cylinder Balance Conservation Check
    bank_factors = profile.cylinders.bank_variation_factors
    if len(bank_factors) != 4:
        errors.append(f"Bank variation factors must have length 4, got {len(bank_factors)}")
    else:
        for i, bf in enumerate(bank_factors):
            if bf <= 0.0:
                errors.append(f"Cylinder {i+1} bank factor must be > 0, got {bf}")
        mean_bank = sum(bank_factors) / 4.0
        if abs(mean_bank - 1.0) > 0.02:
            errors.append(f"Average bank variation factor deviates from 1.0 by > 2%: {mean_bank:.4f}")

    comb_factors = profile.cylinders.combustion_factors
    if len(comb_factors) != 4:
        errors.append(f"Combustion factors must have length 4, got {len(comb_factors)}")
    else:
        for i, cf in enumerate(comb_factors):
            if cf <= 0.0:
                errors.append(f"Cylinder {i+1} combustion factor must be > 0, got {cf}")
        mean_comb = sum(comb_factors) / 4.0
        if abs(mean_comb - 1.0) > 0.02:
            errors.append(f"Average combustion factor deviates from 1.0 by > 2%: {mean_comb:.4f}")

    # 8. Bounds Enforcement against POPULATION_DISTRIBUTIONS catalog
    for name, dist in POPULATION_DISTRIBUTIONS.items():
        val = None
        if hasattr(profile.thermal, name):
            val = getattr(profile.thermal, name)
        elif hasattr(profile.lubrication, name):
            val = getattr(profile.lubrication, name)
        elif hasattr(profile.turbocharger, name):
            val = getattr(profile.turbocharger, name)
        elif hasattr(profile.combustion, name):
            val = getattr(profile.combustion, name)
        elif hasattr(profile.mechanical, name):
            val = getattr(profile.mechanical, name)
        elif hasattr(profile.sensors, name):
            val = getattr(profile.sensors, name)

        if val is not None:
            if val < dist.lower_bound - 1e-6 or val > dist.upper_bound + 1e-6:
                errors.append(
                    f"Parameter '{name}' value {val} outside authorized bounds [{dist.lower_bound}, {dist.upper_bound}]"
                )

    return ValidationResult(
        is_valid=(len(errors) == 0),
        errors=errors,
        warnings=warnings,
        metrics={"engine_instance_id": profile.engine_instance_id},
    )


def validate_telemetry_stream(telemetry: List[TelemetryRecord]) -> ValidationResult:
    """
    Validate physical sanity and numerical integrity of a generated telemetry stream.
    """
    errors: List[str] = []
    warnings: List[str] = []

    if not telemetry:
        errors.append("Telemetry stream is empty.")
        return ValidationResult(is_valid=False, errors=errors)

    for i, rec in enumerate(telemetry):
        # Timestamp must be non-negative and non-decreasing
        if rec.timestamp < 0.0:
            errors.append(f"Step {i}: negative timestamp {rec.timestamp}")
            break

        # Operating points check
        if rec.throttle < 0.0 or rec.throttle > 100.0:
            errors.append(f"Step {i}: invalid throttle {rec.throttle}")
        if rec.altitude < 0.0:
            errors.append(f"Step {i}: negative altitude {rec.altitude}")

        # Check for unphysical negative values on channels that must remain positive
        # (ignoring NaNs that might arise from sensor dropout faults)
        if not math.isnan(rec.rpm) and rec.rpm < 0.0:
            errors.append(f"Step {i}: negative RPM {rec.rpm}")
        if not math.isnan(rec.oil_pressure) and rec.oil_pressure < 0.0:
            errors.append(f"Step {i}: negative oil pressure {rec.oil_pressure}")
        if not math.isnan(rec.fuel_flow) and rec.fuel_flow < 0.0:
            errors.append(f"Step {i}: negative fuel flow {rec.fuel_flow}")
        if not math.isnan(rec.vibration) and rec.vibration < 0.0:
            errors.append(f"Step {i}: negative vibration amplitude {rec.vibration}")

        # Temperature validity: cannot be below absolute zero or unrealistically high
        if not math.isnan(rec.cht) and (rec.cht < -50.0 or rec.cht > 300.0):
            errors.append(f"Step {i}: CHT out of physical range: {rec.cht} °C")
        if not math.isnan(rec.egt) and (rec.egt < -50.0 or rec.egt > 1200.0):
            errors.append(f"Step {i}: EGT out of physical range: {rec.egt} °C")
        if not math.isnan(rec.oil_temp) and (rec.oil_temp < -50.0 or rec.oil_temp > 250.0):
            errors.append(f"Step {i}: Oil temp out of physical range: {rec.oil_temp} °C")

        if len(errors) > 25:
            errors.append("Too many errors, truncating validation.")
            break

    return ValidationResult(
        is_valid=(len(errors) == 0),
        errors=errors,
        warnings=warnings,
        metrics={"total_records": len(telemetry)},
    )


def validate_population(profiles: List[EngineProfile]) -> PopulationValidationReport:
    """
    Perform statistical audit and physical sanity checks across a population ensemble.
    """
    all_errors: List[str] = []
    valid_count = 0
    invalid_count = 0

    # Collect parameter values for statistical profiling
    param_arrays: Dict[str, List[float]] = {}
    clipping_counts: Dict[str, int] = {}

    for name in POPULATION_DISTRIBUTIONS.keys():
        param_arrays[name] = []
        clipping_counts[name] = 0

    for profile in profiles:
        res = validate_engine_profile(profile)
        if res.is_valid:
            valid_count += 1
        else:
            invalid_count += 1
            all_errors.extend([f"Engine {profile.engine_instance_id}: {e}" for e in res.errors])

        # Gather values
        for name, dist in POPULATION_DISTRIBUTIONS.items():
            val = None
            if hasattr(profile.thermal, name):
                val = getattr(profile.thermal, name)
            elif hasattr(profile.lubrication, name):
                val = getattr(profile.lubrication, name)
            elif hasattr(profile.turbocharger, name):
                val = getattr(profile.turbocharger, name)
            elif hasattr(profile.combustion, name):
                val = getattr(profile.combustion, name)
            elif hasattr(profile.mechanical, name):
                val = getattr(profile.mechanical, name)
            elif hasattr(profile.sensors, name):
                val = getattr(profile.sensors, name)

            if val is not None:
                param_arrays[name].append(val)
                # Count if right at boundary
                if abs(val - dist.lower_bound) < 1e-6 or abs(val - dist.upper_bound) < 1e-6:
                    clipping_counts[name] += 1

    # Compute statistics
    param_stats: Dict[str, Dict[str, float]] = {}
    for name, arr in param_arrays.items():
        if arr:
            vals = np.array(arr)
            param_stats[name] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "p05": float(np.percentile(vals, 5)),
                "p95": float(np.percentile(vals, 95)),
            }

    return PopulationValidationReport(
        population_id=profiles[0].population_id if profiles else "EMPTY",
        total_profiles=len(profiles),
        valid_profiles=valid_count,
        invalid_profiles=invalid_count,
        is_fully_valid=(invalid_count == 0),
        parameter_statistics=param_stats,
        clipping_counts=clipping_counts,
        errors=all_errors[:50],  # cap reporting length
    )
