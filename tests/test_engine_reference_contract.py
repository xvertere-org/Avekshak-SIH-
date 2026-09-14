"""
Phase 1 Contract Verification Tests: Engine Identity & Physics Contract.

Verifies that the repository adheres to the authoritative Rotax 914 UL/F reference
contract, enforces parameter provenance, prevents false claims about simulator
fidelity, separates reference ratings from grey-box calibration parameters,
and accurately documents missing physics.
"""

import json
from pathlib import Path
import pytest

from configs.engine_reference import load_rotax_914_reference, Rotax914ReferenceSpec
from configs.config_loader import (
    load_engine_reference,
    load_physics_contract,
    load_engine_config,
)
from simulator.config import TierAParameters, TierCParameters, SimulatorConfig
from telemetry.schema import EngineConfig


PROJECT_ROOT = Path(__file__).parent.parent


def test_01_engine_identity_is_rotax_914_ul_f():
    """Requirement 1: Engine identity is exactly Rotax 914 UL/F."""
    spec = load_rotax_914_reference()
    assert isinstance(spec, Rotax914ReferenceSpec)
    assert spec.engine_identity.manufacturer == "BRP-Rotax GmbH & Co KG"
    assert spec.engine_identity.model == "914"
    assert "914 UL" in spec.engine_identity.variants
    assert "914 F" in spec.engine_identity.variants
    assert spec.engine_identity.cycle == "4-stroke Otto cycle"
    assert spec.engine_identity.aspiration_architecture == "turbocharged_with_intercooler_provision"


def test_02_cylinder_count_and_geometry():
    """Requirement 2: Cylinder count is 4 and geometry matches Rotax 914 specifications."""
    spec = load_rotax_914_reference()
    assert spec.engine_identity.cylinder_count == 4
    assert spec.geometry.displacement_cc.value == 1211.2
    assert spec.geometry.displacement_cc.unit == "cm^3"
    assert spec.geometry.bore_mm.value == 79.5
    assert spec.geometry.bore_mm.unit == "mm"
    assert spec.geometry.stroke_mm.value == 61.0
    assert spec.geometry.stroke_mm.unit == "mm"
    assert spec.geometry.compression_ratio.value == 9.0
    assert spec.geometry.firing_order.value == "1-4-3-2"


def test_03_required_parameters_present_or_explicitly_unknown():
    """Requirement 3: Parameters have documented provenance or are explicitly marked UNKNOWN."""
    spec = load_rotax_914_reference()

    # Verified operating limits
    verified_params = [
        spec.operating_limits.takeoff_power_w,
        spec.operating_limits.continuous_power_w,
        spec.operating_limits.rpm_max_takeoff,
        spec.operating_limits.rpm_max_continuous,
        spec.operating_limits.rpm_idle,
        spec.operating_limits.map_takeoff_bar,
        spec.operating_limits.map_continuous_bar,
        spec.operating_limits.critical_altitude_m,
        spec.operating_limits.cht_limit_c,
        spec.operating_limits.oil_temp_min_c,
        spec.operating_limits.oil_temp_max_c,
        spec.operating_limits.oil_press_min_bar,
        spec.operating_limits.oil_press_max_bar,
    ]
    for param in verified_params:
        assert param.status == "AUTHORITATIVE_VERIFIED"
        assert len(param.source) > 0
        assert param.source_type in ["OEM_OPERATORS_MANUAL", "TYPE_CERTIFICATE_DATA_SHEET", "ENGINEERING_DERIVATION"]

    # Proprietary / unverified OEM parameters must be explicitly marked UNKNOWN (never fabricated)
    unknown_params = [
        spec.turbo_system.compressor_efficiency_map,
        spec.turbo_system.turbine_efficiency_map,
        spec.turbo_system.tcu_control_gains,
    ]
    for param in unknown_params:
        assert param.status == "UNKNOWN"
        assert param.value == "UNKNOWN"
        assert "not published" in param.notes.lower()


def test_04_explicit_units_on_all_parameters():
    """Requirement 4: Units are explicit on every engineering parameter."""
    spec = load_rotax_914_reference()
    valid_units = {"cm^3", "mm", "ratio", "sequence", "W", "RPM", "bar", "m", "deg_C", "map_surface", "control_gains"}

    for key, param in vars(spec.geometry).items():
        assert param.unit in valid_units, f"Invalid unit '{param.unit}' for geometry parameter '{key}'"

    for key, param in vars(spec.operating_limits).items():
        assert param.unit in valid_units, f"Invalid unit '{param.unit}' for operating limit '{key}'"


def test_05_no_active_config_silently_identifies_as_912():
    """Requirement 5: No active configuration silently identifies the simulator as 912 UL/ULS."""
    # Check default_engine.json
    eng_json = load_engine_config()
    assert eng_json.reference_engine == "Rotax 914 UL/F"
    assert "912" not in eng_json.model_template_name

    # Check TierAParameters
    tier_a = TierAParameters()
    assert tier_a.reference_engine == "Rotax 914 UL/F"
    assert tier_a.reference_displacement_cc == 1211.2
    assert tier_a.reference_compression_ratio == 9.0


def test_06_turbo_physics_marked_not_implemented():
    """Requirement 6: No configuration claims turbo physics exist if turbo equations are absent."""
    contract = load_physics_contract()
    subsystems = contract["subsystems"]

    # Intake / turbo boost must be explicitly NOT_IMPLEMENTED
    turbo_subsystem = subsystems["subsystem_b_intake_boost"]
    assert turbo_subsystem["status"] == "NOT_IMPLEMENTED"
    assert len(turbo_subsystem["implemented_features"]) == 0
    assert "naturally aspirated" in turbo_subsystem["current_simplification_note"].lower()

    # Multi-cylinder must be explicitly NOT_IMPLEMENTED
    mc_subsystem = subsystems["subsystem_multi_cylinder"]
    assert mc_subsystem["status"] == "NOT_IMPLEMENTED"

    # Electrical must be explicitly NOT_IMPLEMENTED
    elec_subsystem = subsystems["subsystem_i_electrical"]
    assert elec_subsystem["status"] == "NOT_IMPLEMENTED"


def test_07_gearbox_reference_ratio_not_1_to_1():
    """Requirement 7: Gearbox reference ratio is 2.43:1 and distinguished from simulator 1:1 drive."""
    spec = load_rotax_914_reference()
    assert spec.drivetrain.reduction_gearbox_present is True
    # Rotax 914 standard gearbox ratio is 51/21 = 2.4286:1
    assert 2.42 < spec.drivetrain.reduction_ratio.value < 2.44
    assert spec.drivetrain.reduction_ratio.value != 1.0

    # Propeller speed at takeoff is engine_rpm / 2.4286 (~2388 RPM)
    assert spec.drivetrain.propeller_speed_takeoff_rpm.value == 2388.0
    assert spec.drivetrain.propeller_speed_takeoff_rpm.value < spec.operating_limits.rpm_max_takeoff.value

    # Simulator simplification is explicitly labeled
    sim_repr = spec.drivetrain.simulator_current_representation
    assert sim_repr["status"] == "SIMULATOR_SIMPLIFICATION"
    assert "1:1 direct" in sim_repr["mode"]


def test_08_reference_spec_and_simulator_calibration_are_separate():
    """Requirement 8: Reference specifications and simulator calibration parameters are separate."""
    spec = load_rotax_914_reference()
    tier_a = TierAParameters()

    # OEM continuous power is 73.5 kW (100 hp)
    assert spec.operating_limits.continuous_power_w.value == 73500.0
    assert tier_a.oem_continuous_power_w == 73500.0

    # OEM takeoff power is 84.5 kW (115 hp)
    assert spec.operating_limits.takeoff_power_w.value == 84500.0
    assert tier_a.oem_takeoff_power_w == 84500.0

    # Calibrated power ceiling in reduced-order NA prototype is 58.0 kW
    assert tier_a.power_max_continuous_w == 58000.0
    assert tier_a.power_max_continuous_w != tier_a.oem_continuous_power_w

    # Drivetrain: OEM ratio is 2.4286, while Tier C uses lumped inertia direct drive
    tier_c = TierCParameters()
    assert tier_a.reference_gearbox_ratio == 2.4286
    assert tier_c.inertia_kg_m2 == 0.28


def test_09_configuration_loaders_succeed():
    """Requirement 9: Configuration and contract loaders load cleanly without error."""
    ref_dict = load_engine_reference()
    assert "engine_identity" in ref_dict
    assert ref_dict["engine_identity"]["model"] == "914"

    contract_dict = load_physics_contract()
    assert "subsystems" in contract_dict
    assert len(contract_dict["subsystems"]) >= 9

    eng_cfg = load_engine_config()
    assert eng_cfg.num_cylinders == 4
    assert eng_cfg.displacement_cc == 1211.2
    assert eng_cfg.compression_ratio == 9.0
    assert eng_cfg.rated_power_hp == 115.0


def test_10_claim_taxonomy_defined_and_enforced():
    """Requirement 10: Claim taxonomy is strictly defined and followed."""
    contract = load_physics_contract()
    taxonomy = contract["contract_metadata"]["taxonomy_definitions"]

    assert "IMPLEMENTED" in taxonomy
    assert "PARTIALLY_IMPLEMENTED" in taxonomy
    assert "NOT_IMPLEMENTED" in taxonomy

    valid_statuses = {"IMPLEMENTED", "PARTIALLY_IMPLEMENTED", "NOT_IMPLEMENTED"}
    for sub_key, sub_data in contract["subsystems"].items():
        assert sub_data["status"] in valid_statuses, f"Subsystem {sub_key} has invalid status: {sub_data['status']}"
