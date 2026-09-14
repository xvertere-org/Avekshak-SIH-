"""
Authoritative Observability Taxonomy and Registry for SIH26054 Digital Twin.

Establishes the single source of truth for the observability classification
of every state variable in the Rotax 914 UL/F aero-piston digital twin.

Observability Categories:
1. DIRECTLY_OBSERVED: Directly measured by calibrated on-engine telemetry sensors.
2. INDIRECTLY_OBSERVED: Kinematically or algebraically derived directly from sensor measurements.
3. MODEL_DERIVED: Forward-computed by the physics engine / thermodynamics models.
4. UNOBSERVED: Internal latent / surrogate state without direct measurement.
5. UNAVAILABLE: Unmodeled or uninstrumented subsystem (e.g. electrical).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, List


class ObservabilityType(str, Enum):
    """Observability classification of a physical state variable."""
    DIRECTLY_OBSERVED = "DIRECTLY_OBSERVED"        # Direct physical sensor measurement
    INDIRECTLY_OBSERVED = "INDIRECTLY_OBSERVED"    # Deterministically mapped from sensor via kinematics
    MODEL_DERIVED = "MODEL_DERIVED"                # Physics-computed internal variable
    UNOBSERVED = "UNOBSERVED"                      # Latent / surrogate state without measurement channel
    UNAVAILABLE = "UNAVAILABLE"                    # Uninstrumented / unmodeled subsystem


@dataclass
class ObservabilityEntry:
    """Registry entry detailing the observability metadata for a state variable."""
    name: str
    observability_type: ObservabilityType
    unit: str
    description: str
    source_channel: Optional[str] = None
    derivation_rule: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "observability_type": self.observability_type.value,
            "unit": self.unit,
            "description": self.description,
            "source_channel": self.source_channel,
            "derivation_rule": self.derivation_rule,
        }


# Master catalog of canonical state variables and their observability
CANONICAL_OBSERVABILITY_CATALOG: Dict[str, ObservabilityEntry] = {
    # Directly Observed
    "rpm": ObservabilityEntry(
        name="rpm",
        observability_type=ObservabilityType.DIRECTLY_OBSERVED,
        unit="RPM",
        description="Engine crankshaft rotational speed",
        source_channel="rpm",
    ),
    "cht_cyl1_c": ObservabilityEntry(
        name="cht_cyl1_c",
        observability_type=ObservabilityType.DIRECTLY_OBSERVED,
        unit="°C",
        description="Cylinder 1 cylinder head temperature",
        source_channel="cht_cyl1",
    ),
    "cht_cyl2_c": ObservabilityEntry(
        name="cht_cyl2_c",
        observability_type=ObservabilityType.DIRECTLY_OBSERVED,
        unit="°C",
        description="Cylinder 2 cylinder head temperature",
        source_channel="cht_cyl2",
    ),
    "cht_cyl3_c": ObservabilityEntry(
        name="cht_cyl3_c",
        observability_type=ObservabilityType.DIRECTLY_OBSERVED,
        unit="°C",
        description="Cylinder 3 cylinder head temperature",
        source_channel="cht_cyl3",
    ),
    "cht_cyl4_c": ObservabilityEntry(
        name="cht_cyl4_c",
        observability_type=ObservabilityType.DIRECTLY_OBSERVED,
        unit="°C",
        description="Cylinder 4 cylinder head temperature",
        source_channel="cht_cyl4",
    ),
    "egt_cyl1_c": ObservabilityEntry(
        name="egt_cyl1_c",
        observability_type=ObservabilityType.DIRECTLY_OBSERVED,
        unit="°C",
        description="Cylinder 1 exhaust gas temperature",
        source_channel="egt_cyl1",
    ),
    "egt_cyl2_c": ObservabilityEntry(
        name="egt_cyl2_c",
        observability_type=ObservabilityType.DIRECTLY_OBSERVED,
        unit="°C",
        description="Cylinder 2 exhaust gas temperature",
        source_channel="egt_cyl2",
    ),
    "egt_cyl3_c": ObservabilityEntry(
        name="egt_cyl3_c",
        observability_type=ObservabilityType.DIRECTLY_OBSERVED,
        unit="°C",
        description="Cylinder 3 exhaust gas temperature",
        source_channel="egt_cyl3",
    ),
    "egt_cyl4_c": ObservabilityEntry(
        name="egt_cyl4_c",
        observability_type=ObservabilityType.DIRECTLY_OBSERVED,
        unit="°C",
        description="Cylinder 4 exhaust gas temperature",
        source_channel="egt_cyl4",
    ),
    "oil_pressure_bar": ObservabilityEntry(
        name="oil_pressure_bar",
        observability_type=ObservabilityType.DIRECTLY_OBSERVED,
        unit="bar",
        description="Main oil gallery lubricating pressure",
        source_channel="oil_pressure",
    ),
    "oil_temp_c": ObservabilityEntry(
        name="oil_temp_c",
        observability_type=ObservabilityType.DIRECTLY_OBSERVED,
        unit="°C",
        description="Main engine lubricating oil temperature",
        source_channel="oil_temp",
    ),
    "fuel_flow_l_h": ObservabilityEntry(
        name="fuel_flow_l_h",
        observability_type=ObservabilityType.DIRECTLY_OBSERVED,
        unit="L/h",
        description="Fuel supply volumetric flow rate",
        source_channel="fuel_flow",
    ),
    "coolant_temp_c": ObservabilityEntry(
        name="coolant_temp_c",
        observability_type=ObservabilityType.DIRECTLY_OBSERVED,
        unit="°C",
        description="Liquid cooling system return temperature",
        source_channel="coolant_temp",
    ),
    "vibration_rms_g": ObservabilityEntry(
        name="vibration_rms_g",
        observability_type=ObservabilityType.DIRECTLY_OBSERVED,
        unit="g",
        description="Engine crankcase structural vibration RMS",
        source_channel="vibration",
    ),

    # Indirectly Observed
    "propeller_rpm": ObservabilityEntry(
        name="propeller_rpm",
        observability_type=ObservabilityType.INDIRECTLY_OBSERVED,
        unit="RPM",
        description="Propeller shaft rotational speed",
        source_channel="rpm",
        derivation_rule="propeller_rpm = rpm / 2.42857 (gearbox ratio 51/21)",
    ),

    # Model Derived
    "power_target_w": ObservabilityEntry(
        name="power_target_w",
        observability_type=ObservabilityType.MODEL_DERIVED,
        unit="W",
        description="Target mechanical brake power demand",
        derivation_rule="lookup(throttle_pct, density_factor)",
    ),
    "indicated_power_w": ObservabilityEntry(
        name="indicated_power_w",
        observability_type=ObservabilityType.MODEL_DERIVED,
        unit="W",
        description="Indicated cylinder gas expansion power",
        derivation_rule="power_target_w / (eta_mech * eta_comb)",
    ),
    "engine_torque_nm": ObservabilityEntry(
        name="engine_torque_nm",
        observability_type=ObservabilityType.MODEL_DERIVED,
        unit="N*m",
        description="Net crankshaft drive torque",
        derivation_rule="power_target_w / omega",
    ),
    "propeller_torque_nm": ObservabilityEntry(
        name="propeller_torque_nm",
        observability_type=ObservabilityType.MODEL_DERIVED,
        unit="N*m",
        description="Propeller aerodynamic load torque",
        derivation_rule="k_prop * omega_prop^2",
    ),
    "map_bar": ObservabilityEntry(
        name="map_bar",
        observability_type=ObservabilityType.MODEL_DERIVED,
        unit="bar",
        description="Manifold absolute air pressure",
        derivation_rule="p_amb * pressure_ratio * delta_p_intercooler",
    ),
    "pressure_ratio": ObservabilityEntry(
        name="pressure_ratio",
        observability_type=ObservabilityType.MODEL_DERIVED,
        unit="-",
        description="Turbocharger compressor pressure ratio",
        derivation_rule="compressor_isবentropy_map(m_dot_air, wastegate_pos)",
    ),
    "wastegate_position": ObservabilityEntry(
        name="wastegate_position",
        observability_type=ObservabilityType.MODEL_DERIVED,
        unit="ratio",
        description="TCU wastegate opening duty fraction",
        derivation_rule="tcu_controller(map_target, map_actual)",
    ),
    "q_heads_w": ObservabilityEntry(
        name="q_heads_w",
        observability_type=ObservabilityType.MODEL_DERIVED,
        unit="W",
        description="Total thermal power transferred to cylinder heads",
        derivation_rule="m_dot_fuel * LHV * q_gen_fraction",
    ),
    "q_radiator_w": ObservabilityEntry(
        name="q_radiator_w",
        observability_type=ObservabilityType.MODEL_DERIVED,
        unit="W",
        description="Thermal power dissipated by coolant radiator",
        derivation_rule="h_radiator * (T_coolant - T_amb)",
    ),
    "engine_load_pct": ObservabilityEntry(
        name="engine_load_pct",
        observability_type=ObservabilityType.MODEL_DERIVED,
        unit="%",
        description="Relative engine load fraction",
        derivation_rule="(power_target_w / rated_continuous_power) * 100",
    ),

    # Unobserved / Internal Surrogates
    "turbo_shaft_speed": ObservabilityEntry(
        name="turbo_shaft_speed",
        observability_type=ObservabilityType.UNOBSERVED,
        unit="RPM",
        description="Internal turbocharger spool speed (surrogate)",
        derivation_rule="reduced_order_turbo_spool_ode",
    ),
    "compressor_aerodynamic_efficiency": ObservabilityEntry(
        name="compressor_aerodynamic_efficiency",
        observability_type=ObservabilityType.UNOBSERVED,
        unit="-",
        description="Compressor isentropic aerodynamic efficiency",
        derivation_rule="surrogate_compressor_map",
    ),

    # Unavailable (Electrical)
    "battery_voltage_v": ObservabilityEntry(
        name="battery_voltage_v",
        observability_type=ObservabilityType.UNAVAILABLE,
        unit="V",
        description="Aircraft main bus battery voltage (unmodeled)",
    ),
    "battery_current_a": ObservabilityEntry(
        name="battery_current_a",
        observability_type=ObservabilityType.UNAVAILABLE,
        unit="A",
        description="Alternator / battery electrical current (unmodeled)",
    ),
    "bus_voltage_v": ObservabilityEntry(
        name="bus_voltage_v",
        observability_type=ObservabilityType.UNAVAILABLE,
        unit="V",
        description="Regulated avionics 14V bus voltage (unmodeled)",
    ),
}


class ObservabilityRegistry:
    """
    Singleton / authoritative registry class providing queries and metadata
    for state observability.
    """

    _catalog: Dict[str, ObservabilityEntry] = CANONICAL_OBSERVABILITY_CATALOG

    @classmethod
    def get_entry(cls, var_name: str) -> Optional[ObservabilityEntry]:
        """Look up metadata for a state variable."""
        return cls._catalog.get(var_name)

    @classmethod
    def get_type(cls, var_name: str) -> ObservabilityType:
        """Get the ObservabilityType for a variable; defaults to UNOBSERVED if unknown."""
        entry = cls._catalog.get(var_name)
        if entry is not None:
            return entry.observability_type
        return ObservabilityType.UNOBSERVED

    @classmethod
    def is_directly_observed(cls, var_name: str) -> bool:
        return cls.get_type(var_name) == ObservabilityType.DIRECTLY_OBSERVED

    @classmethod
    def is_observed(cls, var_name: str) -> bool:
        return cls.get_type(var_name) in (
            ObservabilityType.DIRECTLY_OBSERVED,
            ObservabilityType.INDIRECTLY_OBSERVED,
        )

    @classmethod
    def get_observability_coverage(cls) -> float:
        """
        Calculate inherent architectural observability coverage:
        Ratio of directly or indirectly observed state variables to total active modeled state variables
        (excluding unmodeled UNAVAILABLE channels).

        Note: This metric represents the fraction of registered canonical quantities classified as
        observed or indirectly observed within the model catalog (e.g. 9/18 = 0.5000). It is NOT a
        formal nonlinear observability-rank analysis (e.g. Lie derivatives or Gramian rank) and does
        NOT mean the physical engine is "50% observable".
        """
        modeled_entries = [
            e for e in cls._catalog.values()
            if e.observability_type != ObservabilityType.UNAVAILABLE
        ]
        if not modeled_entries:
            return 0.0
        observed_entries = [
            e for e in modeled_entries
            if e.observability_type in (
                ObservabilityType.DIRECTLY_OBSERVED,
                ObservabilityType.INDIRECTLY_OBSERVED,
            )
        ]
        # Return ratio rounded to 4 decimal places
        return round(len(observed_entries) / len(modeled_entries), 4)

    @classmethod
    def list_by_type(cls, obs_type: ObservabilityType) -> List[str]:
        """List variable names matching a given observability type."""
        return [
            name for name, entry in cls._catalog.items()
            if entry.observability_type == obs_type
        ]
