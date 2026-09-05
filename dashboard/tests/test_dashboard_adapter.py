"""
Unit and contract tests for DashboardAdapter in SIH26054 Digital Twin.
"""

import pytest
import math
from dashboard.schemas.contracts import Phase13OutputContract
from dashboard.schemas.view_model import (
    StatusLevel,
    AvailabilityStatus,
    CANONICAL_CHANNELS,
)
from dashboard.services.adapter import DashboardAdapter


def test_contract_from_dict_and_object():
    """Verify Phase13OutputContract deserialization from dict and object."""
    raw_dict = {
        "timestamp": 120.5,
        "engine_id": "ENGINE_MALE_01",
        "mission_id": "MIS_ISR_04",
        "execution_status": "COMPLETED",
        "is_synthetic_demo": False,
        "telemetry": {
            "timestamp": 120.5,
            "rpm": 5450.0,
            "cht": 112.4,
            "egt": 720.0,
            "oil_temp": 92.5,
            "oil_pressure": 3.8,
            "fuel_flow": 16.2,
            "vibration": 0.85,
            "throttle": 75.0,
            "load": 72.0,
            "altitude": 3000.0,
            "ambient_temp": 12.0,
            "mission_phase": "CRUISE",
            "source": "bench_test",
            "source_type": "hardware_loop",
            "simulation_version": "1.0.0",
        },
        "health_index": {
            "smoothed_health_index": 0.94,
            "health_state": "HEALTHY",
            "degradation_rate": -0.0001,
            "degradation_trend": "STABLE",
            "excluded_channels": [],
        },
        "prognostics": {
            "rul_seconds_median": 7200.0,
            "rul_seconds_p05": 6500.0,
            "rul_seconds_p95": 8100.0,
            "status": "ACTIVE_DEGRADATION",
            "limiting_factor": "GLOBAL_HEALTH_INDEX",
            "confidence_score": 0.91,
        },
        "anomaly": {
            "anomaly_status": "NORMAL",
            "anomaly_score": 0.08,
            "contributing_channels": [],
        },
        "fault_diagnosis": {
            "predicted_fault_type": "none",
            "diagnostic_confidence": 0.96,
            "class_probabilities": {"none": 0.96, "cooling_degradation": 0.02},
        },
    }

    contract = Phase13OutputContract.from_dict(raw_dict)
    assert contract.timestamp == 120.5
    assert contract.engine_id == "ENGINE_MALE_01"
    assert contract.mission_id == "MIS_ISR_04"
    assert contract.is_synthetic_demo is False

    # Also test from_object
    contract2 = Phase13OutputContract.from_object(contract)
    assert contract2.timestamp == 120.5


def test_adapter_all_canonical_channels_present():
    """Verify all 7 canonical telemetry channels are mapped into the view model."""
    adapter = DashboardAdapter()
    payload = {
        "timestamp": 50.0,
        "engine_id": "UAV_01",
        "telemetry": {
            "timestamp": 50.0,
            "rpm": 5200.0,
            "cht": 105.0,
            "egt": 680.0,
            "oil_temp": 88.0,
            "oil_pressure": 4.1,
            "fuel_flow": 14.5,
            "vibration": 0.65,
        },
    }

    vm = adapter.adapt(payload)
    assert len(vm.telemetry.channels) == 7
    for ch in CANONICAL_CHANNELS:
        assert ch in vm.telemetry.channels
        ch_model = vm.telemetry.channels[ch]
        assert ch_model.observed_value is not None
        assert ch_model.availability == AvailabilityStatus.AVAILABLE
        assert ch_model.status == StatusLevel.HEALTHY


def test_adapter_sensor_isolation_and_dropout():
    """Verify sensor isolation and NaN dropout status."""
    adapter = DashboardAdapter()
    payload = {
        "timestamp": 85.0,
        "engine_id": "UAV_01",
        "telemetry": {
            "timestamp": 85.0,
            "rpm": 5000.0,
            "cht": float("nan"),  # CHT sensor dropout
            "egt": 650.0,
            "oil_temp": 95.0,
            "oil_pressure": 3.5,
            "fuel_flow": 13.0,
            "vibration": 0.5,
        },
        "health_index": {
            "smoothed_health_index": 0.88,
            "health_state": "HEALTHY",
            "excluded_channels": ["cht"],  # CHT isolated
        },
    }

    vm = adapter.adapt(payload)
    cht_model = vm.telemetry.channels["cht"]
    assert cht_model.observed_value is None
    assert cht_model.is_isolated is True
    assert cht_model.status == StatusLevel.DEGRADED
    assert vm.diagnostics.sensor_fault_indicated is True
    assert "cht" in vm.diagnostics.isolated_channels


def test_adapter_degraded_critical_states():
    """Verify critical status propagation from upstream results."""
    adapter = DashboardAdapter()
    payload = {
        "timestamp": 200.0,
        "engine_id": "UAV_01",
        "health_index": {
            "smoothed_health_index": 0.28,
            "health_state": "CRITICAL",
        },
        "anomaly": {
            "anomaly_status": "ANOMALY",
            "anomaly_score": 0.92,
        },
        "fault_diagnosis": {
            "predicted_fault_type": "cooling_degradation",
            "diagnostic_confidence": 0.89,
            "class_probabilities": {"cooling_degradation": 0.89, "none": 0.05},
        },
        "prognostics": {
            "rul_seconds_median": 0.0,
            "status": "CRITICAL_EOL_REACHED",
            "limiting_factor": "REDLINE_CHT",
        },
    }

    vm = adapter.adapt(payload)
    assert vm.overview.overall_status == StatusLevel.CRITICAL
    assert vm.overview.health_card.status == StatusLevel.CRITICAL
    assert vm.overview.anomaly_card.status == StatusLevel.CRITICAL
    assert vm.overview.fault_card.status == StatusLevel.CRITICAL
    assert vm.overview.rul_card.status == StatusLevel.CRITICAL


def test_adapter_orchestrator_dashboard_state_payload_direct():
    """Verify DashboardAdapter seamlessly adapts Phase 13 DashboardStatePayload object and dict."""
    from orchestrator.pipeline import SystemPipelineOrchestrator, OrchestratorConfig

    config = OrchestratorConfig(auto_bootstrap_on_init=True)
    orch = SystemPipelineOrchestrator(config=config)
    orch.reset(engine_id="TEST_UAV_SYS", mission_id="TEST_MIS_SYS")

    step_data = {
        "timestamp": 10.0,
        "engine_id": "TEST_UAV_SYS",
        "mission_id": "TEST_MIS_SYS",
        "mission_phase": "CRUISE",
        "rpm": 5400.0,
        "cht": 110.0,
        "egt": 710.0,
        "oil_temp": 85.0,
        "oil_pressure": 4.2,
        "fuel_flow": 15.0,
        "vibration": 0.7,
        "throttle": 75.0,
        "load": 70.0,
        "altitude": 2500.0,
        "ambient_temp": 15.0,
    }

    # Step returns a real DashboardStatePayload object
    payload_obj = orch.step(step_data)
    adapter = DashboardAdapter()

    # Test 1: Ingestion of DashboardStatePayload dataclass object
    vm_from_obj = adapter.adapt(payload_obj)
    assert vm_from_obj.engine_id == "TEST_UAV_SYS"
    assert vm_from_obj.mission_id == "TEST_MIS_SYS"
    assert vm_from_obj.timestamp == 10.0
    assert len(vm_from_obj.telemetry.channels) == 7
    assert vm_from_obj.telemetry.channels["rpm"].observed_value == 5400.0
    assert vm_from_obj.telemetry.channels["cht"].observed_value == 110.0
    assert vm_from_obj.overview.health_card.status in [StatusLevel.HEALTHY, StatusLevel.WARNING, StatusLevel.CRITICAL, StatusLevel.UNAVAILABLE]

    # Test 2: Ingestion of payload_obj.to_dict()
    payload_dict = payload_obj.to_dict()
    vm_from_dict = adapter.adapt(payload_dict)
    assert vm_from_dict.engine_id == "TEST_UAV_SYS"
    assert vm_from_dict.mission_id == "TEST_MIS_SYS"
    assert vm_from_dict.timestamp == 10.0
    assert len(vm_from_dict.telemetry.channels) == 7
    assert vm_from_dict.telemetry.channels["rpm"].observed_value == 5400.0
    assert vm_from_dict.telemetry.channels["cht"].observed_value == 110.0

