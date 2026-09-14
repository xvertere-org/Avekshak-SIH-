"""
Physics-Constrained Synthetic Engine Population for Rotax 914 Grey-Box Engine.

Provides:
- Parameter distributions with typed provenance and engineering rationales
- Immutable EngineProfile representing synthetic engine instances
- 8 Canonical Mission trajectories with ISA atmosphere consistency
- Deterministic PopulationGenerator with engine-level train/val/test split
- Physical conservation and sanity validation suites
"""

from simulator.population.parameter_distributions import (
    ProvenanceTag,
    DistributionType,
    ParameterDistribution,
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
from simulator.population.population_generator import (
    PopulationConfig,
    PopulationGenerator,
    simulate_engine_mission,
)
from simulator.population.validation import (
    ValidationResult,
    PopulationValidationReport,
    validate_engine_profile,
    validate_telemetry_stream,
    validate_population,
)

__all__ = [
    # Parameter distributions
    "ProvenanceTag",
    "DistributionType",
    "ParameterDistribution",
    "POPULATION_DISTRIBUTIONS",
    # Engine profiles
    "EngineProfile",
    "ThermalProfile",
    "LubricationProfile",
    "TurbochargerProfile",
    "CombustionProfile",
    "MechanicalProfile",
    "SensorProfile",
    "CylinderProfile",
    # Missions
    "CanonicalMission",
    "MissionTrajectory",
    "build_canonical_trajectory",
    # Generator
    "PopulationConfig",
    "PopulationGenerator",
    "simulate_engine_mission",
    # Validation
    "ValidationResult",
    "PopulationValidationReport",
    "validate_engine_profile",
    "validate_telemetry_stream",
    "validate_population",
]
