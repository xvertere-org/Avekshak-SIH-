"""
NIRVANAA-SIH-SUBMISSION
SIH26054: AI-Enabled Real-Time Digital Twin System for Health Monitoring,
Fault Prediction and Mission Reliability Enhancement of Aero Piston Engines used in MALE UAVs.

Phase 13 Unified System Pipeline Production Entrypoint.
Orchestrates Phase 6 (Twin) -> Phase 7 (Anomaly) -> Phase 8 (Diagnosis) ->
Phase 9 (Health) -> Phase 10 (Forecast) -> Phase 11 (RUL) -> Phase 12 (Explainability) ->
DashboardStatePayload.
"""

import argparse
import sys
import time
from pprint import pprint
import numpy as np

from orchestrator import (
    SystemPipelineOrchestrator,
    DashboardStatePayload,
    SimulationScenario,
    ScenarioFaultType,
)


def run_legacy_phase1_pipeline(dry_run: bool = True) -> int:
    """
    Executes the legacy modular Phase 1 interface connectivity pipeline.
    Preserved for historical reference and backward compatibility.
    """
    from configs.config_loader import load_mission_config, load_engine_config
    from simulator.engine_simulator import EngineSimulator
    from telemetry.streamer import TelemetryStreamer
    from digital_twin.twin_model import DigitalTwin
    from phm.detector import HealthDetector
    from forecasting.rul_predictor import RULPredictor
    from explainability.explainer import ExplainabilityEngine
    from dashboard.app import DashboardInterface

    print("=" * 70)
    print("SIH26054: Aero Piston Engine Digital Twin - Legacy Phase 1 Runner")
    print("=" * 70)

    mission_cfg = load_mission_config()
    engine_cfg = load_engine_config()
    simulator = EngineSimulator(engine_config=engine_cfg)
    telemetry_record = simulator.step(mission_config=mission_cfg, time_step=1.0)

    streamer = TelemetryStreamer(buffer_size=100)
    streamer.push(telemetry_record)
    buffered = streamer.get_latest()

    digital_twin = DigitalTwin(engine_config=engine_cfg)
    twin_state = digital_twin.update(buffered)

    phm = HealthDetector()
    health = phm.assess(twin_state)

    forecaster = RULPredictor(nominal_life_hours=1500.0)
    rul = forecaster.predict(health)

    xai = ExplainabilityEngine()
    explanation = xai.explain(twin_state, health)

    dashboard = DashboardInterface()
    dashboard_payload = dashboard.render_state(
        telemetry=buffered,
        twin_state=twin_state,
        health=health,
        rul=rul,
        explanation=explanation,
    )
    pprint(dashboard_payload)
    print("\n[SUCCESS] Legacy Phase 1 interface connectivity verified successfully.")
    return 0


def run_production_pipeline(
    scenario_name: str = "healthy",
    duration_s: float = 35.0,
    fault_start_s: float = 15.0,
    fault_severity: float = 0.6,
    benchmark_mode: bool = False,
) -> int:
    """
    Executes the authoritative Phase 13 production system pipeline.
    Flow: Simulator -> Phase 6 -> Phase 7 -> Phase 8 -> Phase 9 -> Phase 10 -> Phase 11 -> Phase 12 -> Payload
    """
    print("=" * 80)
    print("SIH26054: UNIFIED AERO PISTON ENGINE DIGITAL TWIN & PHM SYSTEM (PHASE 13)")
    print("=" * 80)

    scenario_map = {
        "healthy": ScenarioFaultType.HEALTHY,
        "cooling": ScenarioFaultType.COOLING_DEGRADATION,
        "lubrication": ScenarioFaultType.LUBRICATION_DEGRADATION,
        "fuel": ScenarioFaultType.FUEL_ABNORMALITY,
        "mechanical": ScenarioFaultType.MECHANICAL_DEGRADATION,
        "sensor": ScenarioFaultType.SENSOR_FAULT,
    }

    selected_fault = scenario_map.get(scenario_name.lower(), ScenarioFaultType.HEALTHY)

    scenario = SimulationScenario(
        name=f"sim_{scenario_name}",
        duration_s=duration_s,
        fault_type=selected_fault,
        fault_start_s=fault_start_s,
        fault_severity=fault_severity,
        engine_id="ENG_ROTAX_914F",
        mission_id="MALE_UAV_MISSION_01",
    )

    print(f"\n[1/4] Initializing System Pipeline Orchestrator (Phase 6–12)...")
    init_start = time.perf_counter()
    orchestrator = SystemPipelineOrchestrator()
    init_elapsed = time.perf_counter() - init_start
    print(f" -> Orchestrator initialized in {init_elapsed:.3f} s")
    if orchestrator.bootstrap_metadata:
        print(f" -> Synthetic Calibration Samples: Phase 7={orchestrator.bootstrap_metadata.get('phase7_calibration_samples')}, "
              f"Phase 8={orchestrator.bootstrap_metadata.get('phase8_training_samples')}")

    print(f"\n[2/4] Executing Causal Mission Simulation: '{scenario_name.upper()}' ({duration_s:.0f}s)...")
    payloads = orchestrator.run_simulation(scenario=scenario)
    print(f" -> Processed {len(payloads)} causal timesteps successfully.")

    final_payload = payloads[-1]
    print("\n[3/4] Authoritative System State at t =", f"{final_payload.timestamp:.1f}s:")
    print("-" * 80)
    print(f" MISSION       : Engine={final_payload.engine_id} | Mission={final_payload.mission_id} | Phase={final_payload.mission_phase}")
    print(f" DIGITAL TWIN  : CHT={final_payload.observed_telemetry.get('cht', 0):.1f}°C (Exp: {final_payload.expected_telemetry.get('cht', 0):.1f}°C) | Norm Res={final_payload.normalized_residuals.get('cht', 0):.2f}")
    print(f" ANOMALY (P7)  : Status={final_payload.anomaly_status} | Score={final_payload.anomaly_score:.3f} | Persistence Count={final_payload.persistence_count}")
    print(f" DIAGNOSIS (P8): Predicted={final_payload.predicted_fault_class.upper()} | Confidence={final_payload.diagnostic_confidence * 100:.1f}%")
    print(f" HEALTH (P9)   : Raw HI={final_payload.raw_health_index:.3f} | Smoothed HI={final_payload.smoothed_health_index:.3f} | State={final_payload.health_state} | Trend={final_payload.degradation_trend}")
    if final_payload.dominant_channels:
        print(f"               : Dominant Degraded Channels={final_payload.dominant_channels}")
    print(f" FORECAST (P10): Status={final_payload.forecast_status} | Source={final_payload.forecast_source} | Horizon={final_payload.forecast_horizon} steps")
    print(f" RUL (P11)     : State={final_payload.rul_state} | Median RUL={f'{final_payload.point_rul_seconds:.1f}s' if final_payload.point_rul_seconds else 'N/A'} | Limiting Factor={final_payload.limiting_factor}")
    print(f" EXPLAIN (P12) : Quality={final_payload.authoritative_explainability.overall_quality.value if final_payload.authoritative_explainability else 'N/A'}")
    print(f" ADVISORY (P13): Action={final_payload.advisory.action_code.value if final_payload.advisory else 'N/A'}")
    print(f"               : Recommendation: {final_payload.recommended_operator_action}")
    print("-" * 80)

    if benchmark_mode:
        print("\n[4/4] End-to-End Inference Latency Benchmark (Steady-State):")
        # Measure steady-state timesteps excluding initial warm-up steps
        steady_latencies = [p.execution_latency_ms for p in payloads[5:]]
        if steady_latencies:
            mean_lat = sum(steady_latencies) / len(steady_latencies)
            med_lat = float(sorted(steady_latencies)[len(steady_latencies) // 2])
            p95_lat = float(np.percentile(steady_latencies, 95))
            p99_lat = float(np.percentile(steady_latencies, 99))
            print(f" -> Samples evaluated : {len(steady_latencies)}")
            print(f" -> Mean Latency      : {mean_lat:.2f} ms")
            print(f" -> Median (P50)      : {med_lat:.2f} ms")
            print(f" -> 95th Percentile   : {p95_lat:.2f} ms")
            print(f" -> 99th Percentile   : {p99_lat:.2f} ms")
            print(f" -> Throughput        : {1000.0 / mean_lat:.1f} observations/sec")

    print("\n[SUCCESS] Phase 13 Unified System Pipeline executed successfully.")
    return 0


def run_pipeline(dry_run: bool = False) -> int:
    """
    Default entrypoint for backward compatibility.
    Runs production pipeline by default.
    """
    return run_production_pipeline(scenario_name="healthy", duration_s=35.0)


def main():
    parser = argparse.ArgumentParser(description="SIH26054 Aero Piston Engine Digital Twin - Phase 13 Orchestrator")
    parser.add_argument(
        "--scenario",
        type=str,
        default="healthy",
        choices=["healthy", "cooling", "lubrication", "fuel", "mechanical", "sensor"],
        help="Simulation scenario fault to inject",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=35.0,
        help="Simulation duration in seconds (default: 35.0)",
    )
    parser.add_argument(
        "--fault-start",
        type=float,
        default=15.0,
        help="Timestamp to activate fault in seconds (default: 15.0)",
    )
    parser.add_argument(
        "--severity",
        type=float,
        default=0.6,
        help="Injected fault severity in [0.0, 1.0] (default: 0.6)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=False,
        help="Run end-to-end latency profiling",
    )
    parser.add_argument(
        "--legacy-phase1",
        action="store_true",
        default=False,
        help="Execute legacy Phase 1 dry-run stub instead of Phase 13 production pipeline",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Alias for running dry-run execution",
    )
    args = parser.parse_args()

    if args.legacy_phase1:
        sys.exit(run_legacy_phase1_pipeline(dry_run=True))
    else:
        sys.exit(
            run_production_pipeline(
                scenario_name=args.scenario,
                duration_s=args.duration,
                fault_start_s=args.fault_start,
                fault_severity=args.severity,
                benchmark_mode=args.benchmark,
            )
        )


if __name__ == "__main__":
    main()
