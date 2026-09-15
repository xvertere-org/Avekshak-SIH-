"""
Comprehensive Integration Tests for Phase 13 SystemPipelineOrchestrator -> DashboardAdapter.

Verifies end-to-end contract compliance across 10 required operational scenarios:
1. Healthy
2. Cooling degradation
3. Lubrication degradation
4. Fuel abnormality
5. Mechanical degradation
6. Sensor fault
7. Missing telemetry
8. TimesFM gated fallback
9. RUL insufficient-history
10. Mission reset/isolation
"""

import math
import pytest
from typing import List

from orchestrator.pipeline import SystemPipelineOrchestrator
from orchestrator.schema import (
    OrchestratorConfig,
    DashboardStatePayload,
    SimulationScenario,
    ScenarioFaultType,
)
from dashboard.services.adapter import DashboardAdapter
from dashboard.schemas.view_model import StatusLevel, AvailabilityStatus


@pytest.fixture(scope="module")
def orchestrator():
    """Module-level orchestrator with deterministic bootstrap."""
    cfg = OrchestratorConfig(auto_bootstrap_on_init=True, deterministic_seed=42)
    orch = SystemPipelineOrchestrator(config=cfg)
    return orch


@pytest.fixture
def adapter():
    return DashboardAdapter()


# ==============================================================================
# Scenario 1: Healthy Cruise Operation
# ==============================================================================
def test_scenario_01_healthy(orchestrator, adapter):
    """Verify DashboardAdapter properly maps Healthy scenario from SystemPipelineOrchestrator."""
    orchestrator.reset(engine_id="ENG_HLTHY_01", mission_id="MIS_HLTHY_01")
    sc = SimulationScenario(
        engine_id="ENG_HLTHY_01",
        mission_id="MIS_HLTHY_01",
        duration_s=35.0,
        fault_type=ScenarioFaultType.HEALTHY,
    )
    payloads: List[DashboardStatePayload] = orchestrator.run_simulation(sc)
    assert len(payloads) > 0
    last_payload = payloads[-1]

    vm = adapter.adapt(last_payload)

    # Mission Context
    assert vm.engine_id == last_payload.engine_id
    assert vm.mission_id == last_payload.mission_id
    assert vm.simulation_mode == "SYNTHETIC_SIMULATION"
    assert vm.is_synthetic_demo is True

    # Telemetry
    assert len(vm.telemetry.channels) == 7
    for ch_name, ch_model in vm.telemetry.channels.items():
        assert ch_model.observed_value is not None
        assert not math.isnan(ch_model.observed_value)
        assert ch_model.availability == AvailabilityStatus.AVAILABLE

    # Diagnosis & Anomaly
    assert vm.diagnostics.predicted_fault_class.lower() == "none"
    assert vm.diagnostics.anomaly_status in ["NORMAL", "HEALTHY"]

    # Health & Overview
    assert vm.prognostics.smoothed_health_index is not None
    assert vm.prognostics.smoothed_health_index >= 0.80
    assert vm.overview.overall_status in [StatusLevel.HEALTHY, StatusLevel.WARNING]
    assert vm.overview.health_card.status == StatusLevel.HEALTHY


# ==============================================================================
# Scenario 2: Cooling Degradation
# ==============================================================================
def test_scenario_02_cooling_degradation(orchestrator, adapter):
    """Verify DashboardAdapter maps progressive cooling degradation without calculation."""
    orchestrator.reset(engine_id="ENG_COOL_01", mission_id="MIS_COOL_01")
    sc = SimulationScenario(
        engine_id="ENG_COOL_01",
        mission_id="MIS_COOL_01",
        duration_s=35.0,
        fault_type=ScenarioFaultType.COOLING_DEGRADATION,
        fault_start_s=10.0,
        fault_severity=0.8,
    )
    payloads = orchestrator.run_simulation(sc)
    post_fault = payloads[-1]

    vm = adapter.adapt(post_fault)

    assert vm.engine_id == post_fault.engine_id
    assert vm.mission_id == post_fault.mission_id
    # Verify exact match with payload values
    assert vm.telemetry.channels["cht"].observed_value == post_fault.observed_telemetry.get("cht")
    assert vm.diagnostics.predicted_fault_class == post_fault.predicted_fault_class
    assert vm.diagnostics.class_probabilities == post_fault.diagnosis_probabilities
    assert vm.prognostics.smoothed_health_index == post_fault.smoothed_health_index
    assert vm.overview.fault_card.subtext.startswith("Diagnosis Probability:")


# ==============================================================================
# Scenario 3: Lubrication Degradation
# ==============================================================================
def test_scenario_03_lubrication_degradation(orchestrator, adapter):
    """Verify DashboardAdapter maps lubrication degradation scenario."""
    orchestrator.reset(engine_id="ENG_LUB_01", mission_id="MIS_LUB_01")
    sc = SimulationScenario(
        engine_id="ENG_LUB_01",
        mission_id="MIS_LUB_01",
        duration_s=35.0,
        fault_type=ScenarioFaultType.LUBRICATION_DEGRADATION,
        fault_start_s=10.0,
        fault_severity=0.75,
    )
    payloads = orchestrator.run_simulation(sc)
    post_fault = payloads[-1]

    vm = adapter.adapt(post_fault)

    assert vm.telemetry.channels["oil_pressure"].observed_value == post_fault.observed_telemetry.get("oil_pressure")
    assert vm.telemetry.channels["oil_temp"].observed_value == post_fault.observed_telemetry.get("oil_temp")
    assert vm.diagnostics.predicted_fault_class == post_fault.predicted_fault_class


# ==============================================================================
# Scenario 4: Fuel Abnormality
# ==============================================================================
def test_scenario_04_fuel_abnormality(orchestrator, adapter):
    """Verify DashboardAdapter maps fuel flow abnormality scenario."""
    orchestrator.reset(engine_id="ENG_FUEL_01", mission_id="MIS_FUEL_01")
    sc = SimulationScenario(
        engine_id="ENG_FUEL_01",
        mission_id="MIS_FUEL_01",
        duration_s=30.0,
        fault_type=ScenarioFaultType.FUEL_ABNORMALITY,
        fault_start_s=10.0,
        fault_severity=0.7,
    )
    payloads = orchestrator.run_simulation(sc)
    post_fault = payloads[-1]

    vm = adapter.adapt(post_fault)

    assert vm.telemetry.channels["fuel_flow"].observed_value == post_fault.observed_telemetry.get("fuel_flow")
    assert vm.diagnostics.predicted_fault_class == post_fault.predicted_fault_class


# ==============================================================================
# Scenario 5: Mechanical Degradation
# ==============================================================================
def test_scenario_05_mechanical_degradation(orchestrator, adapter):
    """Verify DashboardAdapter maps mechanical degradation and vibration increase."""
    orchestrator.reset(engine_id="ENG_MECH_01", mission_id="MIS_MECH_01")
    sc = SimulationScenario(
        engine_id="ENG_MECH_01",
        mission_id="MIS_MECH_01",
        duration_s=30.0,
        fault_type=ScenarioFaultType.MECHANICAL_DEGRADATION,
        fault_start_s=10.0,
        fault_severity=0.7,
    )
    payloads = orchestrator.run_simulation(sc)
    post_fault = payloads[-1]

    vm = adapter.adapt(post_fault)

    assert vm.telemetry.channels["vibration"].observed_value == post_fault.observed_telemetry.get("vibration")
    assert vm.diagnostics.predicted_fault_class == post_fault.predicted_fault_class


# ==============================================================================
# Scenario 6: Sensor Fault & Isolation
# ==============================================================================
def test_scenario_06_sensor_fault(orchestrator, adapter):
    """Verify DashboardAdapter maps sensor fault isolation without corrupting physics."""
    orchestrator.reset(engine_id="ENG_SENS_01", mission_id="MIS_SENS_01")
    sc = SimulationScenario(
        engine_id="ENG_SENS_01",
        mission_id="MIS_SENS_01",
        duration_s=35.0,
        fault_type=ScenarioFaultType.SENSOR_FAULT,
        fault_start_s=10.0,
        fault_severity=0.8,
    )
    payloads = orchestrator.run_simulation(sc)
    post_fault = payloads[-1]

    vm = adapter.adapt(post_fault)

    assert vm.diagnostics.sensor_fault_indicated == (
        post_fault.predicted_fault_class.lower() == "sensor_fault"
        or len(post_fault.anomaly_contributing_channels) > 0
    )
    assert vm.diagnostics.predicted_fault_class == post_fault.predicted_fault_class
    assert vm.telemetry.channels["cht"].observed_value == post_fault.observed_telemetry.get("cht")

    # Explicit sensor fault payload to verify adapter flag mapping
    mock_sensor_payload = DashboardStatePayload(
        engine_id="ENG_SENS_02",
        mission_id="MIS_SENS_02",
        timestamp=20.0,
        mission_phase="CRUISE",
        observed_telemetry={"cht": 150.0},
        expected_telemetry={"cht": 70.0},
        residuals={"cht": 80.0},
        normalized_residuals={"cht": 8.0},
        anomaly_status="ANOMALY",
        anomaly_score=0.95,
        anomaly_contributing_channels=["cht"],
        predicted_fault_class="sensor_fault",
        diagnosis_probabilities={"sensor_fault": 0.92, "none": 0.08},
        diagnosis_data_quality="VALID",
        raw_health_index=0.85,
        smoothed_health_index=0.88,
        health_state="DEGRADED",
        degradation_rate=0.01,
        degradation_trend="STABLE",
        dominant_channels=["cht"],
        forecast_status="BLOCKED_UNAUTHENTICATED_GATED",
        forecast_source="FALLBACK",
        rul_state="DEGRADED",
        point_rul_seconds=1200.0,
        rul_uncertainty_p05=800.0,
        rul_uncertainty_p95=1600.0,
        limiting_factor="cht",
        summary_explanation="Sensor fault bias detected on CHT channel.",
        advisory=None,
        simulation_mode="SYNTHETIC_SIMULATION",
        scenario_metadata={"injected_fault": "sensor_fault"},
        execution_latency_ms=12.5,
    )
    vm_mock = adapter.adapt(mock_sensor_payload)
    assert vm_mock.diagnostics.sensor_fault_indicated is True
    assert vm_mock.diagnostics.predicted_fault_class == "sensor_fault"


# ==============================================================================
# Scenario 7: Missing Telemetry / NaN Preservation (Non-Fabrication)
# ==============================================================================
def test_scenario_07_missing_telemetry_nan_preservation(orchestrator, adapter):
    """Verify that NaN sensor dropout is preserved as None and never replaced with 0 or nominal."""
    orchestrator.reset(engine_id="ENG_DROP_01", mission_id="MIS_DROP_01")

    telemetry_with_nan = {
        "timestamp": 12.0,
        "engine_id": "ENG_DROP_01",
        "mission_id": "MIS_DROP_01",
        "mission_phase": "CRUISE",
        "rpm": 5400.0,
        "cht": float("nan"),  # CHT sensor dropout
        "egt": 710.0,
        "oil_temp": 88.0,
        "oil_pressure": 4.1,
        "fuel_flow": 16.0,
        "vibration": 0.8,
        "throttle": 75.0,
        "load": 70.0,
        "altitude": 2500.0,
        "ambient_temp": 15.0,
    }

    payload = orchestrator.step(telemetry_with_nan)
    vm = adapter.adapt(payload)

    cht_ch = vm.telemetry.channels["cht"]
    assert cht_ch.observed_value is None
    assert cht_ch.is_missing is True
    assert cht_ch.availability == AvailabilityStatus.UNAVAILABLE
    assert cht_ch.observed_value != 0.0


# ==============================================================================
# Scenario 8: TimesFM Gated Fallback Distinction
# ==============================================================================
def test_scenario_08_timesfm_gated_fallback(orchestrator, adapter):
    """Verify that unauthenticated TimesFM status is cleanly reflected and not presented as deep model."""
    orchestrator.reset(engine_id="ENG_FC_01", mission_id="MIS_FC_01")

    step_data = {
        "timestamp": 5.0,
        "engine_id": "ENG_FC_01",
        "mission_id": "MIS_FC_01",
        "rpm": 5400.0,
        "cht": 105.0,
        "egt": 700.0,
        "oil_temp": 85.0,
        "oil_pressure": 4.0,
        "fuel_flow": 15.0,
        "vibration": 0.7,
    }

    payload = orchestrator.step(step_data)
    vm = adapter.adapt(payload)

    # In local environment without Google credentials, forecast is BLOCKED_UNAUTHENTICATED_GATED or BUFFERING
    assert vm.prognostics.forecast_status in ["BLOCKED_UNAUTHENTICATED_GATED", "BUFFERING"]
    assert vm.prognostics.is_pretrained is False


# ==============================================================================
# Scenario 9: RUL Insufficient History (No Numerical Fabrication)
# ==============================================================================
def test_scenario_09_rul_insufficient_history(orchestrator, adapter):
    """Verify that during warmup, RUL state is INSUFFICIENT_HISTORY and point RUL is NOT 0."""
    orchestrator.reset(engine_id="ENG_RUL_01", mission_id="MIS_RUL_01")

    step_data = {
        "timestamp": 1.0,
        "engine_id": "ENG_RUL_01",
        "mission_id": "MIS_RUL_01",
        "rpm": 5400.0,
        "cht": 100.0,
        "egt": 680.0,
        "oil_temp": 80.0,
        "oil_pressure": 4.2,
        "fuel_flow": 14.5,
        "vibration": 0.6,
    }

    payload = orchestrator.step(step_data)
    vm = adapter.adapt(payload)

    assert vm.prognostics.rul_state in ["INSUFFICIENT_HISTORY", "NOT_DEGRADING", "Unavailable"]
    # Verify card value is NOT fabricated as 0
    assert vm.overview.rul_card.value != "0.0 hrs"
    assert vm.overview.rul_card.value != "0 hrs"


# ==============================================================================
# Scenario 10: Mission Reset & Engine Isolation
# ==============================================================================
def test_scenario_10_mission_reset_isolation(orchestrator, adapter):
    """Verify resetting engine and mission causally isolates state and metadata."""
    orchestrator.reset(engine_id="ENG_ALPHA", mission_id="MIS_ALPHA")
    payload1 = orchestrator.step({
        "timestamp": 10.0,
        "engine_id": "ENG_ALPHA",
        "mission_id": "MIS_ALPHA",
        "rpm": 5300.0,
        "cht": 95.0,
    })
    vm1 = adapter.adapt(payload1)
    assert vm1.engine_id == "ENG_ALPHA"
    assert vm1.mission_id == "MIS_ALPHA"
    assert vm1.timestamp == 10.0

    # Reset with new engine and mission
    orchestrator.reset(engine_id="ENG_BRAVO", mission_id="MIS_BRAVO")
    payload2 = orchestrator.step({
        "timestamp": 1.0,
        "engine_id": "ENG_BRAVO",
        "mission_id": "MIS_BRAVO",
        "rpm": 5400.0,
        "cht": 98.0,
    })
    vm2 = adapter.adapt(payload2)
    assert vm2.engine_id == "ENG_BRAVO"
    assert vm2.mission_id == "MIS_BRAVO"
    assert vm2.timestamp == 1.0
    assert vm2.timestamp < vm1.timestamp  # Reset successfully resets causal clock
