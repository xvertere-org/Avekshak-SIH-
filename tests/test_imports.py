"""
Tests verifying clean importability of all project modules and packages.
"""


def test_import_simulator():
    import simulator
    from simulator.base import BaseEngineSimulator
    from simulator.engine_simulator import EngineSimulator
    assert simulator is not None
    assert BaseEngineSimulator is not None
    assert EngineSimulator is not None


def test_import_telemetry():
    import telemetry
    from telemetry.schema import (
        MissionConfig,
        EngineConfig,
        TelemetryRecord,
        DigitalTwinState,
        HealthAssessment,
        RULPrediction,
        ExplanationReport,
    )
    from telemetry.streamer import TelemetryStreamer
    assert telemetry is not None
    assert TelemetryRecord is not None
    assert TelemetryStreamer is not None


def test_import_digital_twin():
    import digital_twin
    from digital_twin.twin_model import DigitalTwin
    assert digital_twin is not None
    assert DigitalTwin is not None


def test_import_forecasting():
    import forecasting
    from forecasting.pipeline import ForecastingPipeline
    assert forecasting is not None
    assert ForecastingPipeline is not None


def test_import_explainability():
    import explainability
    from explainability.shap_explainer import SHAPExplainer
    from explainability.pipeline import ExplainabilityPipeline
    assert explainability is not None
    assert SHAPExplainer is not None
    assert ExplainabilityPipeline is not None


def test_import_dashboard():
    import dashboard
    from dashboard.app import DashboardInterface
    assert dashboard is not None
    assert DashboardInterface is not None


def test_import_configs():
    import configs
    from configs.config_loader import (
        load_mission_config,
        load_engine_config,
        load_telemetry_settings,
    )
    assert configs is not None
    assert load_mission_config is not None
