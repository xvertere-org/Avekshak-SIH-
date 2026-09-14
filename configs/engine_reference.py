"""
Authoritative Engine Reference Specification Loader and Typed Contracts.
Anchor: Rotax 914 UL/F Aero Piston Engine.

This module enforces strict parameter provenance, explicit physical units,
and clear demarcation between authoritative reference specifications and
the reduced-order grey-box simulator implementation.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Union


REFERENCE_DIR = Path(__file__).parent / "engine_reference"
DEFAULT_914_SPEC_PATH = REFERENCE_DIR / "rotax_914_ul_f.json"


@dataclass(frozen=True)
class ProvenanceParameter:
    """A numerical or categorical parameter with verifiable source provenance."""
    value: Union[float, int, str, None]
    unit: str
    source: str
    source_type: str
    status: str
    notes: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProvenanceParameter":
        return cls(
            value=data.get("value"),
            unit=data.get("unit", ""),
            source=data.get("source", ""),
            source_type=data.get("source_type", ""),
            status=data.get("status", "UNKNOWN"),
            notes=data.get("notes", "")
        )

    def is_unknown(self) -> bool:
        return self.status == "UNKNOWN" or self.value == "UNKNOWN" or self.value is None


@dataclass
class EngineIdentity:
    manufacturer: str
    model: str
    variants: List[str]
    certification_basis: str
    engine_type: str
    cycle: str
    cylinder_count: int
    cylinder_arrangement: str
    cooling_architecture: str
    cooling_description: str
    aspiration_architecture: str
    fuel_system: str
    ignition_architecture: str
    lubrication_architecture: str
    reduction_gearbox_architecture: str
    engine_applicability: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EngineIdentity":
        return cls(**data)


@dataclass
class EngineGeometry:
    displacement_cc: ProvenanceParameter
    bore_mm: ProvenanceParameter
    stroke_mm: ProvenanceParameter
    compression_ratio: ProvenanceParameter
    firing_order: ProvenanceParameter

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EngineGeometry":
        return cls(
            displacement_cc=ProvenanceParameter.from_dict(data["displacement_cc"]),
            bore_mm=ProvenanceParameter.from_dict(data["bore_mm"]),
            stroke_mm=ProvenanceParameter.from_dict(data["stroke_mm"]),
            compression_ratio=ProvenanceParameter.from_dict(data["compression_ratio"]),
            firing_order=ProvenanceParameter.from_dict(data["firing_order"]),
        )


@dataclass
class OperatingLimits:
    takeoff_power_w: ProvenanceParameter
    continuous_power_w: ProvenanceParameter
    rpm_max_takeoff: ProvenanceParameter
    rpm_max_continuous: ProvenanceParameter
    rpm_idle: ProvenanceParameter
    map_takeoff_bar: ProvenanceParameter
    map_continuous_bar: ProvenanceParameter
    critical_altitude_m: ProvenanceParameter
    cht_limit_c: ProvenanceParameter
    egt_max_takeoff_c: ProvenanceParameter
    egt_max_continuous_c: ProvenanceParameter
    oil_temp_min_c: ProvenanceParameter
    oil_temp_max_c: ProvenanceParameter
    oil_temp_nominal_min_c: ProvenanceParameter
    oil_temp_nominal_max_c: ProvenanceParameter
    oil_press_min_bar: ProvenanceParameter
    oil_press_max_bar: ProvenanceParameter
    oil_press_normal_min_bar: ProvenanceParameter
    oil_press_normal_max_bar: ProvenanceParameter
    fuel_pressure_min_bar: ProvenanceParameter
    fuel_pressure_max_bar: ProvenanceParameter

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OperatingLimits":
        return cls(**{k: ProvenanceParameter.from_dict(v) for k, v in data.items()})


@dataclass
class TurboSystemSpec:
    turbocharger_present: bool
    compressor_present: bool
    turbine_present: bool
    wastegate_present: bool
    wastegate_actuator_type: str
    tcu_present: bool
    tcu_description: str
    map_boost_required: bool
    charge_air_temp_required: bool
    critical_altitude_m: float
    boost_control_concept: str
    compressor_efficiency_map: ProvenanceParameter
    turbine_efficiency_map: ProvenanceParameter
    tcu_control_gains: ProvenanceParameter
    surrogate_disclosure: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TurboSystemSpec":
        return cls(
            turbocharger_present=data["turbocharger_present"],
            compressor_present=data["compressor_present"],
            turbine_present=data["turbine_present"],
            wastegate_present=data["wastegate_present"],
            wastegate_actuator_type=data["wastegate_actuator_type"],
            tcu_present=data["tcu_present"],
            tcu_description=data["tcu_description"],
            map_boost_required=data["map_boost_required"],
            charge_air_temp_required=data["charge_air_temp_required"],
            critical_altitude_m=data["critical_altitude_m"],
            boost_control_concept=data["boost_control_concept"],
            compressor_efficiency_map=ProvenanceParameter.from_dict(data["compressor_efficiency_map"]),
            turbine_efficiency_map=ProvenanceParameter.from_dict(data["turbine_efficiency_map"]),
            tcu_control_gains=ProvenanceParameter.from_dict(data["tcu_control_gains"]),
            surrogate_disclosure=data.get("surrogate_disclosure"),
        )


@dataclass
class DrivetrainSpec:
    engine_crankshaft_configuration: str
    reduction_gearbox_present: bool
    gearbox_type: str
    reduction_ratio: ProvenanceParameter
    propeller_rotation_direction: str
    propeller_speed_takeoff_rpm: ProvenanceParameter
    propeller_speed_continuous_rpm: ProvenanceParameter
    simulator_current_representation: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DrivetrainSpec":
        return cls(
            engine_crankshaft_configuration=data["engine_crankshaft_configuration"],
            reduction_gearbox_present=data["reduction_gearbox_present"],
            gearbox_type=data["gearbox_type"],
            reduction_ratio=ProvenanceParameter.from_dict(data["reduction_ratio"]),
            propeller_rotation_direction=data["propeller_rotation_direction"],
            propeller_speed_takeoff_rpm=ProvenanceParameter.from_dict(data["propeller_speed_takeoff_rpm"]),
            propeller_speed_continuous_rpm=ProvenanceParameter.from_dict(data["propeller_speed_continuous_rpm"]),
            simulator_current_representation=data["simulator_current_representation"],
        )


@dataclass
class Rotax914ReferenceSpec:
    """Root authoritative specification container for the Rotax 914 UL/F."""
    engine_identity: EngineIdentity
    geometry: EngineGeometry
    operating_limits: OperatingLimits
    turbo_system: TurboSystemSpec
    drivetrain: DrivetrainSpec
    telemetry_channel_contract: Dict[str, Dict[str, Any]]
    reference_data: Optional[Dict[str, Any]] = None
    model_configuration: Optional[Dict[str, Any]] = None
    model_acceptance_bands: Optional[Dict[str, Any]] = None
    model_calibration: Optional[Dict[str, Any]] = None
    model_assumption: Optional[Dict[str, Any]] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Rotax914ReferenceSpec":
        return cls(
            engine_identity=EngineIdentity.from_dict(data["engine_identity"]),
            geometry=EngineGeometry.from_dict(data["geometry"]),
            operating_limits=OperatingLimits.from_dict(data["operating_limits"]),
            turbo_system=TurboSystemSpec.from_dict(data["turbo_system"]),
            drivetrain=DrivetrainSpec.from_dict(data["drivetrain"]),
            telemetry_channel_contract=data.get("telemetry_channel_contract", {}),
            reference_data=data.get("reference_data"),
            model_configuration=data.get("model_configuration"),
            model_acceptance_bands=data.get("model_acceptance_bands"),
            model_calibration=data.get("model_calibration"),
            model_assumption=data.get("model_assumption"),
        )


def load_rotax_914_reference(filepath: Union[str, Path] = DEFAULT_914_SPEC_PATH) -> Rotax914ReferenceSpec:
    """Load and validate the authoritative Rotax 914 UL/F reference specification."""
    path = Path(filepath)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return Rotax914ReferenceSpec.from_dict(data)

