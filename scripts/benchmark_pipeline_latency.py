"""
Authoritative End-to-End Latency Benchmark for SIH26054 Digital Twin Pipeline.

This benchmark instruments and measures the complete execution path:
1. Input / Telemetry preparation & sanitization
2. Physics simulation & residual generation (Digital Twin)
3. Anomaly detection (Hybrid EWMA + dynamic Z-score)
4. ML Baseline / Supervised Fault Diagnosis (XGBoost 16-feature classifier)
5. Health monitoring & degradation tracking (Causal EWMA multi-channel)
6. Deep Sequence Forecasting (TimesFM-3 Foundation Model on CUDA/CPU)
7. Prognostics & RUL with uncertainty calculation (Theil-Sen + P05/P95)
8. Multi-modal Explainability (TreeSHAP attribution & evidence fusion)
9. Operator Advisory Decision Support
10. Payload Assembly
11. Dashboard Adaptation (DashboardAdapter -> ViewModel)
12. Total End-to-End Pipeline Runtime

Evaluates:
- Initialization / Cold Bootstrap latency
- Cold-start Step 0 latency
- Baseline / Buffering mode latency (Steps 1-31, pure ML + physics, TimesFM accumulating context)
- Deep Foundation Model active inference mode latency (Steps 32+, TimesFM running forward pass)
- Overall warm-step statistics (Mean, Median, P95, P99, Min, Max)
- Per-stage latency breakdown
"""

import sys
import os
import time
import json
import statistics
from typing import Dict, List, Any
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from orchestrator.pipeline import SystemPipelineOrchestrator
from orchestrator.schema import OrchestratorConfig
from simulator.engine_simulator import EngineSimulator
from simulator.config import SimulatorConfig
from dashboard.services.adapter import DashboardAdapter


def run_latency_benchmark(total_steps: int = 60, output_json: str = "reports/pipeline_latency_benchmark.json") -> Dict[str, Any]:
    print("=" * 80)
    print("SIH26054 DIGITAL TWIN PIPELINE: AUTHORITATIVE RUNTIME LATENCY BENCHMARK")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # 1. Lifecycle: Model Loading & Initialization Timing
    # --------------------------------------------------------------------------
    print("\n[Stage 0] Measuring Initialization & Bootstrap Lifecycle...")
    t_init_start = time.perf_counter()
    cfg = OrchestratorConfig(auto_bootstrap_on_init=True, deterministic_seed=42)
    orchestrator = SystemPipelineOrchestrator(config=cfg)
    init_latency_s = time.perf_counter() - t_init_start
    print(f"  -> Pipeline Initialization & Bootstrap completed in: {init_latency_s:.3f} s")
    tfm_adapter = orchestrator.forecasting_pipeline.timesfm_adapter
    print(f"  -> Device: TimesFM={tfm_adapter.device}, Status={tfm_adapter.runtime_status}")

    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    adapter = DashboardAdapter()

    # Per-step timing accumulators
    sim_times: List[float] = []
    prep_times: List[float] = []
    physics_times: List[float] = []
    anom_times: List[float] = []
    diag_times: List[float] = []
    health_times: List[float] = []
    forecast_times: List[float] = []
    rul_times: List[float] = []
    exp_times: List[float] = []
    adv_times: List[float] = []
    asm_times: List[float] = []
    adapt_times: List[float] = []
    orchestrator_step_times: List[float] = []
    total_e2e_times: List[float] = []

    print(f"\n[Benchmarking] Executing {total_steps} real streaming flight timesteps...")

    for step_idx in range(total_steps):
        t_step_total_start = time.perf_counter()

        # 1. Simulator observation generation
        t_sim_start = time.perf_counter()
        raw_telemetry = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.5)
        sim_elapsed_ms = (time.perf_counter() - t_sim_start) * 1000.0
        sim_times.append(sim_elapsed_ms)

        # 2. Orchestrator Step (instruments stages 1-10)
        t_orch_start = time.perf_counter()
        payload = orchestrator.step(telemetry=raw_telemetry)
        orch_elapsed_ms = (time.perf_counter() - t_orch_start) * 1000.0
        orchestrator_step_times.append(orch_elapsed_ms)

        # 3. Dashboard Adapter
        t_adapt_start = time.perf_counter()
        vm = adapter.adapt(payload)
        adapt_elapsed_ms = (time.perf_counter() - t_adapt_start) * 1000.0
        adapt_times.append(adapt_elapsed_ms)

        total_elapsed_ms = (time.perf_counter() - t_step_total_start) * 1000.0
        total_e2e_times.append(total_elapsed_ms)

        # Extract per-stage timings from payload
        st = payload.stage_latencies_ms
        prep_times.append(st.get("telemetry_prep_ms", 0.0))
        physics_times.append(st.get("physics_simulation_ms", 0.0))
        anom_times.append(st.get("anomaly_detection_ms", 0.0))
        diag_times.append(st.get("fault_diagnosis_ms", 0.0))
        health_times.append(st.get("health_monitoring_ms", 0.0))
        forecast_times.append(st.get("forecasting_ms", 0.0))
        rul_times.append(st.get("prognostics_rul_ms", 0.0))
        exp_times.append(st.get("explainability_ms", 0.0))
        adv_times.append(st.get("advisory_ms", 0.0))
        asm_times.append(orch_elapsed_ms - sum([
            st.get("telemetry_prep_ms", 0.0),
            st.get("physics_simulation_ms", 0.0),
            st.get("anomaly_detection_ms", 0.0),
            st.get("fault_diagnosis_ms", 0.0),
            st.get("health_monitoring_ms", 0.0),
            st.get("forecasting_ms", 0.0),
            st.get("prognostics_rul_ms", 0.0),
            st.get("explainability_ms", 0.0),
            st.get("advisory_ms", 0.0),
        ]))

    # Compute Statistics
    def stats(arr: List[float]) -> Dict[str, float]:
        if not arr:
            return {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": round(statistics.mean(arr), 3),
            "median": round(statistics.median(arr), 3),
            "p95": round(float(np.percentile(arr, 95)), 3),
            "p99": round(float(np.percentile(arr, 99)), 3),
            "min": round(min(arr), 3),
            "max": round(max(arr), 3),
        }

    cold_start_step0 = {
        "telemetry_prep_ms": round(prep_times[0], 3),
        "physics_simulation_ms": round(physics_times[0], 3),
        "anomaly_detection_ms": round(anom_times[0], 3),
        "fault_diagnosis_ms": round(diag_times[0], 3),
        "health_monitoring_ms": round(health_times[0], 3),
        "forecasting_ms": round(forecast_times[0], 3),
        "prognostics_rul_ms": round(rul_times[0], 3),
        "explainability_ms": round(exp_times[0], 3),
        "advisory_ms": round(adv_times[0], 3),
        "dashboard_adaptation_ms": round(adapt_times[0], 3),
        "total_orchestrator_step_ms": round(orchestrator_step_times[0], 3),
        "total_end_to_end_ms": round(total_e2e_times[0], 3),
    }

    # Warm steps (steps 1+)
    warm_e2e = total_e2e_times[1:]
    warm_orch = orchestrator_step_times[1:]

    # Buffering mode (steps 1 to 31: TimesFM accumulates 32 steps)
    buf_e2e = total_e2e_times[1:32]
    buf_orch = orchestrator_step_times[1:32]

    # Active Deep Model mode (steps 32 to end: TimesFM performs forward pass)
    act_e2e = total_e2e_times[32:]
    act_orch = orchestrator_step_times[32:]

    results = {
        "benchmark_metadata": {
            "total_steps": total_steps,
            "device": str(tfm_adapter.device),
            "timesfm_status": str(tfm_adapter.runtime_status),
            "init_bootstrap_seconds": round(init_latency_s, 3),
        },
        "cold_start_step0": cold_start_step0,
        "warm_all_steps_stats": {
            "orchestrator_step": stats(warm_orch),
            "total_e2e": stats(warm_e2e),
        },
        "buffering_mode_stats_steps_1_to_31": {
            "description": "Pure ML + Physics baseline without TimesFM forward pass (buffering context)",
            "orchestrator_step": stats(buf_orch),
            "total_e2e": stats(buf_e2e),
        },
        "active_foundation_model_stats_steps_32_plus": {
            "description": "Complete pipeline with TimesFM-3 Deep Foundation Model running on CUDA",
            "orchestrator_step": stats(act_orch),
            "total_e2e": stats(act_e2e),
        },
        "per_stage_warm_stats": {
            "1_telemetry_prep": stats(prep_times[1:]),
            "2_physics_twin": stats(physics_times[1:]),
            "3_anomaly_detection": stats(anom_times[1:]),
            "4_fault_diagnosis_xgboost": stats(diag_times[1:]),
            "5_health_monitoring_ewma": stats(health_times[1:]),
            "6_forecasting_buffering": stats(forecast_times[1:32]),
            "6_forecasting_active_timesfm": stats(forecast_times[32:]),
            "7_prognostics_rul_theil_sen": stats(rul_times[1:]),
            "8_explainability_treeshap": stats(exp_times[1:]),
            "9_advisory_support": stats(adv_times[1:]),
            "10_dashboard_adaptation": stats(adapt_times[1:]),
        }
    }

    # Print Report
    print("\n" + "=" * 80)
    print("                     PER-STAGE TIMING BREAKDOWN (WARM STEPS)")
    print("=" * 80)
    print(f"{'Stage Name':<38} | {'Mean (ms)':<10} | {'Median (ms)':<11} | {'P95 (ms)':<10}")
    print("-" * 80)
    for name, s in results["per_stage_warm_stats"].items():
        print(f"{name:<38} | {s['mean']:<10.3f} | {s['median']:<11.3f} | {s['p95']:<10.3f}")

    print("\n" + "=" * 80)
    print("                     END-TO-END PIPELINE LATENCY SUMMARY")
    print("=" * 80)
    print(f"Cold-Start Step 0 E2E Latency: {cold_start_step0['total_end_to_end_ms']:.3f} ms")
    print("\n1. BASELINE / BUFFERING MODE (Steps 1-31, TimesFM buffering, Pure ML + Twin):")
    b_stat = results["buffering_mode_stats_steps_1_to_31"]["total_e2e"]
    print(f"   Mean: {b_stat['mean']:.3f} ms | Median: {b_stat['median']:.3f} ms | P95: {b_stat['p95']:.3f} ms | P99: {b_stat['p99']:.3f} ms")

    if act_e2e:
        print("\n2. ACTIVE DEEP FOUNDATION MODEL MODE (Steps 32+, TimesFM-3 CUDA Forward Pass Active):")
        a_stat = results["active_foundation_model_stats_steps_32_plus"]["total_e2e"]
        print(f"   Mean: {a_stat['mean']:.3f} ms | Median: {a_stat['median']:.3f} ms | P95: {a_stat['p95']:.3f} ms | P99: {a_stat['p99']:.3f} ms")

    print("\n3. OVERALL WARM STEPS (All warm steps combined):")
    w_stat = results["warm_all_steps_stats"]["total_e2e"]
    print(f"   Mean: {w_stat['mean']:.3f} ms | Median: {w_stat['median']:.3f} ms | P95: {w_stat['p95']:.3f} ms | P99: {w_stat['p99']:.3f} ms")
    print("=" * 80)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full benchmark report to {output_json}")

    return results


if __name__ == "__main__":
    steps = 60
    if len(sys.argv) > 1:
        try:
            steps = int(sys.argv[1])
        except ValueError:
            pass
    run_latency_benchmark(total_steps=steps)
