"""
Phase 6 1,000-Step Latency Benchmark:
Evaluates end-to-end DigitalTwin.update() execution speed:
telemetry ingestion -> quality check -> synchronization -> prediction ->
residual generation -> health assessment -> generic anomaly detection -> physics diagnosis.

Measures:
- Mean latency (ms)
- Median latency (ms)
- p95 latency (ms)
- Maximum latency (ms)
- 20 ms (50 Hz) budget compliance
"""

import time
import sys
import os
sys.path.insert(0, os.path.abspath("."))

import statistics
import json
import numpy as np

from simulator.engine_simulator import EngineSimulator
from simulator.config import SimulatorConfig
from digital_twin.twin_model import DigitalTwin


def run_benchmark(steps: int = 1000):
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    # Warmup (50 steps)
    for _ in range(50):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1)
        twin.update(rec)

    latencies_ms = []

    for i in range(steps):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1)
        t_start = time.perf_counter()
        st = twin.update(rec)
        t_elapsed = (time.perf_counter() - t_start) * 1000.0
        latencies_ms.append(t_elapsed)

    mean_lat = statistics.mean(latencies_ms)
    median_lat = statistics.median(latencies_ms)
    p95_lat = float(np.percentile(latencies_ms, 95))
    max_lat = max(latencies_ms)
    min_lat = min(latencies_ms)

    budget_ms = 20.0  # 50 Hz budget
    passed_budget = (p95_lat < budget_ms) and (mean_lat < budget_ms)

    results = {
        "steps": steps,
        "mean_ms": round(mean_lat, 4),
        "median_ms": round(median_lat, 4),
        "p95_ms": round(p95_lat, 4),
        "max_ms": round(max_lat, 4),
        "min_ms": round(min_lat, 4),
        "budget_ms": budget_ms,
        "frequency_hz": round(1000.0 / mean_lat, 1),
        "passed_budget": passed_budget,
    }

    print("=== PHASE 6 1,000-STEP LATENCY BENCHMARK ===")
    print(f"Steps evaluated: {steps}")
    print(f"Mean latency:   {mean_lat:.3f} ms")
    print(f"Median latency: {median_lat:.3f} ms")
    print(f"p95 latency:    {p95_lat:.3f} ms")
    print(f"Max latency:    {max_lat:.3f} ms")
    print(f"Min latency:    {min_lat:.3f} ms")
    print(f"Budget:         {budget_ms:.1f} ms (50 Hz)")
    print(f"Effective Rate: {1000.0 / mean_lat:.1f} Hz")
    print(f"Budget Result:  {'PASS' if passed_budget else 'FAIL'}")

    with open("evidence/phase6_latency_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_benchmark(1000)
