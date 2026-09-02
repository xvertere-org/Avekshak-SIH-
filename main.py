"""
NIRVANAA-SIH-SUBMISSION
SIH26054: AI-Enabled Real-Time Digital Twin System for Health Monitoring,
Fault Prediction and Mission Reliability Enhancement of Aero Piston Engines used in MALE UAVs.

Phase 1 Pipeline Interface Orchestrator / Dry-Run Entrypoint.
"""

import argparse
import sys
from pprint import pprint

from configs.config_loader import load_mission_config, load_engine_config
from simulator.engine_simulator import EngineSimulator
from telemetry.streamer import TelemetryStreamer
from digital_twin.twin_model import DigitalTwin
from phm.detector import HealthDetector
from forecasting.rul_predictor import RULPredictor
from explainability.explainer import ExplainabilityEngine
from dashboard.app import DashboardInterface


def run_pipeline(dry_run: bool = True) -> int:
    """
    Executes the modular Phase 1 interface connectivity pipeline.
    Flow: MissionConfig -> Simulator -> Telemetry -> DigitalTwin -> PHM -> Forecasting/RUL -> Explainability -> Dashboard
    """
    print("=" * 70)
    print("SIH26054: Aero Piston Engine Digital Twin - Phase 1 Interface Runner")
    print("=" * 70)

    # 1. Mission & Engine Configuration
    print("\n[1/7] Loading Configurations...")
    mission_cfg = load_mission_config()
    engine_cfg = load_engine_config()
    print(f" -> Mission ID : {mission_cfg.mission_id} ({mission_cfg.mission_phase})")
    print(f" -> Engine ID  : {mission_cfg.engine_id} ({engine_cfg.model_template_name})")

    # 2. Simulator Interface
    print("\n[2/7] Invoking Engine Simulator Interface...")
    simulator = EngineSimulator(engine_config=engine_cfg)
    telemetry_record = simulator.step(mission_config=mission_cfg, time_step=1.0)
    print(f" -> Generated Telemetry Record @ t={telemetry_record.timestamp}s (Source: {telemetry_record.source})")
    print(f"    RPM: {telemetry_record.rpm:.1f} | CHT: {telemetry_record.cht:.1f}°C | EGT: {telemetry_record.egt:.1f}°C | Oil P: {telemetry_record.oil_pressure:.2f} bar")

    # 3. Telemetry Streamer / Buffer
    print("\n[3/7] Ingesting into Telemetry Streamer...")
    streamer = TelemetryStreamer(buffer_size=100)
    streamer.push(telemetry_record)
    buffered = streamer.get_latest()
    print(f" -> Streamer buffer size: {len(streamer.get_buffer())} record(s)")

    # 4. Digital Twin Interface
    print("\n[4/7] Updating Digital Twin State & Residual Engine...")
    digital_twin = DigitalTwin(engine_config=engine_cfg)
    twin_state = digital_twin.update(buffered)
    print(f" -> Digital Twin State estimated (Confidence: {twin_state.state_confidence * 100:.1f}%)")
    print(f"    Residuals: {twin_state.residuals}")

    # 5. PHM Health Assessment Interface
    print("\n[5/7] Running PHM Diagnostic Assessment...")
    phm = HealthDetector()
    health = phm.assess(twin_state)
    print(f" -> Health Index: {health.health_index:.2f} | Anomaly: {health.anomaly_detected} (Score: {health.anomaly_score:.2f})")
    print(f"    Fault Category: '{health.fault_category}'")

    # 6. Forecasting / RUL Interface
    print("\n[6/7] Predicting Remaining Useful Life (RUL)...")
    forecaster = RULPredictor(nominal_life_hours=1500.0)
    rul = forecaster.predict(health)
    print(f" -> Estimated RUL: {rul.estimated_rul_hours:.1f} hrs [Bounds: {rul.confidence_lower_hours:.1f} - {rul.confidence_upper_hours:.1f} hrs]")

    # 7. Explainability & Dashboard Interfaces
    print("\n[7/7] Generating Explanations & Packaging Dashboard State...")
    xai = ExplainabilityEngine()
    explanation = xai.explain(twin_state, health)
    print(f" -> Explanation: {explanation.explanation_text}")

    dashboard = DashboardInterface()
    dashboard_payload = dashboard.render_state(
        telemetry=buffered,
        twin_state=twin_state,
        health=health,
        rul=rul,
        explanation=explanation,
    )

    print("\n" + "-" * 70)
    print("Dashboard Payload Interface Verification:")
    pprint(dashboard_payload)
    print("-" * 70)
    print("\n[SUCCESS] Phase 1 end-to-end interface connectivity verified successfully.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="SIH26054 Aero Piston Engine Digital Twin Interface Runner")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Execute pipeline interface verification dry-run without active ML computation",
    )
    args = parser.parse_args()
    sys.exit(run_pipeline(dry_run=args.dry_run))


if __name__ == "__main__":
    main()
