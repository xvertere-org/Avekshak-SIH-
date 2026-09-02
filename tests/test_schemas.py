"""
Tests for core data schemas, data models, and configurations.
"""

import pytest
from telemetry.schema import (
    MissionConfig,
    EngineConfig,
    TelemetryRecord,
    MissionPhase,
    FaultCategory,
    DigitalTwinState,
    HealthAssessment,
    RULPrediction,
    ExplanationReport,
)
from configs.config_loader import (
    load_mission_config,
    load_engine_config,
    load_telemetry_settings,
)


def test_engine_config_creation_and_defaults():
    """Verify EngineConfig instantiates properly with non-authoritative defaults."""
    cfg = EngineConfig(engine_id="TEST_ENG_01")
    assert cfg.engine_id == "TEST_ENG_01"
    assert cfg.model_template_name == "GENERIC_MALE_UAV_PISTON_4CYL"
    assert cfg.displacement_cc is None
    assert cfg.cooling_type == "air_liquid_hybrid"

    # Dict roundtrip
    d = cfg.to_dict()
    assert isinstance(d, dict)
    reconstructed = EngineConfig.from_dict(d)
    assert reconstructed.engine_id == cfg.engine_id


def test_mission_config_creation():
    """Verify MissionConfig instantiates and supports all required fields."""
    mission = MissionConfig(
        mission_id="MSN_TEST_001",
        engine_id="ENG_001",
        altitude=4500.0,
        ambient_temperature=-5.0,
        duration=7200.0,
        throttle=80.0,
        engine_load=75.0,
        mission_phase=MissionPhase.CLIMB.value,
        fault_type=FaultCategory.COOLING_DEGRADATION.value,
        fault_severity=0.4,
    )
    assert mission.mission_id == "MSN_TEST_001"
    assert mission.altitude == 4500.0
    assert mission.fault_type == "cooling_degradation"
    assert mission.fault_severity == 0.4

    # Dict roundtrip
    d = mission.to_dict()
    reconstructed = MissionConfig.from_dict(d)
    assert reconstructed.duration == 7200.0


def test_telemetry_record_validation_and_provenance():
    """Verify TelemetryRecord validates all required channels and provenance fields."""
    record = TelemetryRecord(
        timestamp=100.0,
        mission_id="MSN_TEST_001",
        engine_id="ENG_001",
        mission_phase="CRUISE",
        altitude=3000.0,
        ambient_temp=15.0,
        throttle=75.0,
        load=70.0,
        rpm=2400.0,
        cht=178.5,
        egt=715.0,
        oil_temp=83.0,
        oil_pressure=4.3,
        fuel_flow=21.8,
        vibration=0.75,
        fault_type="none",
        fault_severity=0.0,
        source="test_simulator",
        source_type="simulated",
        simulation_version="0.1.0-test",
    )
    assert record.rpm == 2400.0
    assert record.source == "test_simulator"
    assert record.source_type == "simulated"
    assert record.simulation_version == "0.1.0-test"

    d = record.to_dict()
    assert "source_type" in d
    assert "simulation_version" in d
    assert "cht" in d
    assert "oil_pressure" in d


def test_config_loader_files():
    """Verify config loader loads default json files into schema objects."""
    mission = load_mission_config()
    assert isinstance(mission, MissionConfig)
    assert mission.mission_id == "MISSION_MALE_UAV_001"

    engine = load_engine_config()
    assert isinstance(engine, EngineConfig)
    assert engine.engine_id == "ENGINE_UAV_01"

    telemetry_settings = load_telemetry_settings()
    assert "sampling_rate_hz" in telemetry_settings
    assert "channels" in telemetry_settings
    assert "cht" in telemetry_settings["channels"]
