"""
Component and scenario verification tests for SIH26054 Dashboard.
"""

import pytest
from dashboard.services.adapter import DashboardAdapter
from dashboard.services.demo_provider import DemoScenarioProvider
from dashboard.components.telemetry_charts import build_channel_figure
from dashboard.schemas.view_model import CANONICAL_CHANNELS
from dashboard.app import DashboardInterface


def test_demo_scenario_provider_all_scenarios():
    """Verify all demo scenarios generate schema-compliant contracts and history buffers."""
    scenarios = DemoScenarioProvider.get_available_scenarios()
    assert len(scenarios) == 4

    adapter = DashboardAdapter()
    for sc in scenarios:
        contract, history = DemoScenarioProvider.generate_scenario_payload(sc, step=30)
        assert contract.is_synthetic_demo is True
        assert contract.execution_status == "COMPLETED"

        # Adapt to ViewModel
        vm = adapter.adapt(contract)
        assert vm.overview is not None
        assert vm.telemetry is not None
        assert vm.diagnostics is not None
        assert vm.prognostics is not None
        assert vm.data_quality is not None

        # Verify all 7 channels are present
        for ch in CANONICAL_CHANNELS:
            assert ch in vm.telemetry.channels


def test_build_channel_figure_plotly():
    """Verify Plotly figure generation for telemetry channels."""
    contract, history = DemoScenarioProvider.generate_scenario_payload(
        "2. Thermal Degradation (Cooling Conductance Loss)", step=35
    )
    adapter = DashboardAdapter()
    vm = adapter.adapt(contract)

    for ch in CANONICAL_CHANNELS:
        ch_model = vm.telemetry.channels[ch]
        ch_hist = history.get(ch, {})
        fig = build_channel_figure(
            model=ch_model,
            history_timestamps=ch_hist.get("timestamps"),
            history_observed=ch_hist.get("observed"),
            history_expected=ch_hist.get("expected"),
        )
        assert fig is not None
        assert len(fig.data) >= 1  # Contains observed and/or expected traces


def test_backward_compatibility_dashboard_interface():
    """Verify Phase 1 DashboardInterface continues to return payload expected by main.py."""
    interface = DashboardInterface()

    class MockTelemetry:
        timestamp = 100.0
        engine_id = "TEST_ENG"
        source = "test"
        source_type = "simulated"
        simulation_version = "1.0"
        throttle = 70.0
        load = 65.0
        altitude = 2500.0
        ambient_temp = 15.0
        rpm = 5200.0
        cht = 95.0
        egt = 650.0
        oil_temp = 82.0
        oil_pressure = 4.0
        fuel_flow = 15.0
        vibration = 0.5
        mission_phase = "CRUISE"

    class MockTwinState:
        residuals = {"cht_residual": 0.0}
        nominal_estimates = {"expected_cht": 95.0}

    class MockHealth:
        smoothed_health_index = 0.95
        health_state = "HEALTHY"
        excluded_channels = []

    class MockRUL:
        rul_seconds_median = 36000.0
        status = "NOT_DEGRADING"

    class MockExplanation:
        summary_explanation = "Nominal test"

    payload = interface.render_state(
        telemetry=MockTelemetry(),
        twin_state=MockTwinState(),
        health=MockHealth(),
        rul=MockRUL(),
        explanation=MockExplanation(),
    )

    assert payload["engine_id"] == "TEST_ENG"
    assert payload["timestamp"] == 100.0
    assert payload["mission_phase"] == "CRUISE"
    assert payload["health_index"] == 0.95
    assert payload["estimated_rul_hours"] == 10.0
