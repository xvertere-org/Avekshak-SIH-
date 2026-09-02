"""
Tests verifying modular interface connectivity and data flow.
"""

from configs.config_loader import load_mission_config, load_engine_config
from simulator.engine_simulator import EngineSimulator
from telemetry.streamer import TelemetryStreamer
from digital_twin.twin_model import DigitalTwin
from phm.detector import HealthDetector
from forecasting.rul_predictor import RULPredictor
from explainability.explainer import ExplainabilityEngine
from dashboard.app import DashboardInterface
from telemetry.schema import TelemetryRecord, DigitalTwinState, HealthAssessment, RULPrediction, ExplanationReport


def test_simulator_to_telemetry_interface():
    """Verify basic simulator -> telemetry interface call."""
    mission_cfg = load_mission_config()
    engine_cfg = load_engine_config()

    simulator = EngineSimulator(engine_config=engine_cfg)
    record = simulator.step(mission_config=mission_cfg, time_step=1.0)

    assert isinstance(record, TelemetryRecord)
    assert record.engine_id == mission_cfg.engine_id
    assert record.mission_id == mission_cfg.mission_id
    assert record.source == "simulator_v1_stub"
    assert record.source_type == "simulated"

    # Verify streamer accepts and stores the record
    streamer = TelemetryStreamer(buffer_size=10)
    streamer.push(record)
    latest = streamer.get_latest()
    assert latest is not None
    assert latest.timestamp == record.timestamp


def test_end_to_end_interface_flow():
    """
    Verify complete Phase 1 interface connectivity chain:
    MissionConfig -> Simulator -> Telemetry -> Digital Twin -> PHM -> Forecasting/RUL -> Explainability -> Dashboard
    """
    # 1. MissionConfig & EngineConfig
    mission_cfg = load_mission_config()
    engine_cfg = load_engine_config()

    # 2. Simulator Interface
    simulator = EngineSimulator(engine_config=engine_cfg)
    telemetry = simulator.step(mission_config=mission_cfg, time_step=1.0)
    assert isinstance(telemetry, TelemetryRecord)

    # 3. Telemetry Streamer
    streamer = TelemetryStreamer()
    streamer.push(telemetry)
    buffered_telemetry = streamer.get_latest()
    assert buffered_telemetry is not None

    # 4. Digital Twin Interface
    twin = DigitalTwin(engine_config=engine_cfg)
    twin_state = twin.update(buffered_telemetry)
    assert isinstance(twin_state, DigitalTwinState)
    assert "cht_residual" in twin_state.residuals

    # 5. PHM Interface
    phm = HealthDetector()
    health = phm.assess(twin_state)
    assert isinstance(health, HealthAssessment)
    assert 0.0 <= health.health_index <= 1.0

    # 6. Forecasting/RUL Interface
    forecaster = RULPredictor(nominal_life_hours=1500.0)
    rul = forecaster.predict(health)
    assert isinstance(rul, RULPrediction)
    assert rul.estimated_rul_hours > 0

    # 7. Explainability Interface
    xai = ExplainabilityEngine()
    explanation = xai.explain(twin_state, health)
    assert isinstance(explanation, ExplanationReport)
    assert len(explanation.explanation_text) > 0

    # 8. Dashboard Interface
    dashboard = DashboardInterface()
    payload = dashboard.render_state(
        telemetry=buffered_telemetry,
        twin_state=twin_state,
        health=health,
        rul=rul,
        explanation=explanation,
    )
    assert isinstance(payload, dict)
    assert payload["engine_id"] == mission_cfg.engine_id
    assert "provenance" in payload
    assert payload["provenance"]["source_type"] == "simulated"
