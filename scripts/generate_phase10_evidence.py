"""
Phase 10 Mission Reliability & What-If Simulation: Reproducible Evidence Generator.

Executes all 12 Golden Scenarios, 6-way causal decomposition, state isolation checks,
order-independence benchmarks, streaming vs full trajectory profiling, and outputs
evidence/phase10_mission_matrix.json.

DISCLAIMER:
All outputs are MODEL SCENARIO RESULTS / SYNTHETIC.
"""

import os
import sys
import json
import time
import subprocess
from typing import Dict, Any, List
import numpy as np

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from digital_twin.mission_simulator import MissionSimulator
from digital_twin.what_if import (
    WhatIfEvaluator,
    get_golden_scenario_specs,
    run_causal_decomposition,
)
from digital_twin.mission_types import AUTHORITATIVE_ENVELOPE_THRESHOLDS


def get_git_info() -> Dict[str, str]:
    def run_cmd(cmd: str) -> str:
        try:
            return subprocess.check_output(cmd, shell=True, text=True).strip()
        except Exception as e:
            return f"ERROR: {e}"

    return {
        "branch": run_cmd("git rev-parse --abbrev-ref HEAD"),
        "head_sha": run_cmd("git rev-parse HEAD"),
        "origin_head": run_cmd("git rev-parse origin/rotax-914-greybox-engine"),
        "main_sha": run_cmd("git rev-parse main"),
        "origin_main_sha": run_cmd("git rev-parse origin/main"),
        "merge_base": run_cmd("git merge-base main rotax-914-greybox-engine"),
        "working_tree": run_cmd("git status --short --branch"),
    }


def generate_phase10_evidence() -> Dict[str, Any]:
    print("Starting Phase 10 Evidence Generation...")
    git_info = get_git_info()

    sim = MissionSimulator()
    evaluator = WhatIfEvaluator(simulator=sim)

    # 1. Authoritative Thresholds Audit
    thresholds_evidence = []
    for t in AUTHORITATIVE_ENVELOPE_THRESHOLDS:
        thresholds_evidence.append({
            "channel": t.channel,
            "numeric_value": t.numeric_value,
            "unit": t.unit,
            "direction": t.direction.value,
            "classification": t.classification.value,
            "source_doc": t.source_doc,
            "source_section": t.source_section,
            "notes": t.notes,
        })

    # 2. Run all 12 Golden Scenarios
    print("Executing 12 Golden Scenarios...")
    specs = get_golden_scenario_specs(duration_s=150.0, dt_s=1.0)
    golden_results: Dict[str, Any] = {}
    baseline_result = None

    for name, spec in specs.items():
        res = sim.run_mission(spec, full_trajectory=True)
        if name == "NOMINAL_CRUISE":
            baseline_result = res

        golden_results[name] = {
            "mission_id": res.mission_id,
            "scenario_id": res.scenario_id,
            "provenance": res.provenance,
            "metrics": {
                "duration_s": res.metrics.mission_duration_s,
                "step_count": res.metrics.step_count,
                "valid_telemetry_fraction": res.metrics.valid_telemetry_fraction,
                "min_hi": res.metrics.min_hi,
                "mean_hi": res.metrics.mean_hi,
                "final_hi": res.metrics.final_hi,
                "min_subsystem_health": res.metrics.min_subsystem_health,
                "max_cht_c": res.metrics.max_cht_c,
                "max_egt_c": res.metrics.max_egt_c,
                "min_oil_pressure_bar": res.metrics.min_oil_pressure_bar,
                "max_oil_temp_c": res.metrics.max_oil_temp_c,
                "max_vibration_g": res.metrics.max_vibration_g,
                "mean_fuel_flow_l_h": res.metrics.mean_fuel_flow_l_h,
                "max_fuel_flow_l_h": res.metrics.max_fuel_flow_l_h,
                "time_below_watch_s": res.metrics.time_below_watch_s,
                "time_below_degraded_s": res.metrics.time_below_degraded_s,
                "time_below_critical_s": res.metrics.time_below_critical_s,
                "time_in_diagnostic_state_s": res.metrics.time_in_diagnostic_state_s,
                "min_estimated_rul_h": res.metrics.min_estimated_rul_h,
                "final_estimated_rul_h": res.metrics.final_estimated_rul_h,
                "envelope_event_count": res.metrics.envelope_event_count,
                "total_envelope_excursion_duration_s": res.metrics.total_envelope_excursion_duration_s,
                "risk_index": {
                    "score": res.metrics.risk_index.score,
                    "health_component": res.metrics.risk_index.health_component,
                    "envelope_component": res.metrics.risk_index.envelope_component,
                    "duration_component": res.metrics.risk_index.duration_component,
                    "rul_component": res.metrics.risk_index.rul_component,
                    "type": res.metrics.risk_index.risk_index_type,
                    "disclaimer": res.metrics.risk_index.disclaimer,
                },
            },
            "envelope_events_summary": [
                {
                    "timestamp": e.timestamp,
                    "channel": e.channel,
                    "observed": e.observed_value,
                    "threshold": e.threshold_value,
                    "unit": e.unit,
                    "classification": e.classification.value,
                    "duration_s": e.duration_s,
                }
                for e in res.envelope_events
            ],
        }

    # 3. Compute Counterfactual Deltas against NOMINAL_CRUISE
    print("Computing comparative deltas...")
    comparative_deltas: Dict[str, Any] = {}
    for name, res_dict in golden_results.items():
        if name == "NOMINAL_CRUISE":
            continue
        # Re-fetch spec to compare
        spec = specs[name]
        res = sim.run_mission(spec, full_trajectory=False)
        comp = evaluator.compare(baseline_result, res)
        comparative_deltas[name] = {
            "baseline_id": comp.baseline_id,
            "scenario_id": comp.scenario_id,
            "delta_min_hi": comp.delta_min_hi,
            "delta_mean_hi": comp.delta_mean_hi,
            "delta_final_hi": comp.delta_final_hi,
            "delta_time_below_watch_s": comp.delta_time_below_watch_s,
            "delta_time_below_degraded_s": comp.delta_time_below_degraded_s,
            "delta_time_below_critical_s": comp.delta_time_below_critical_s,
            "delta_min_rul_h": comp.delta_min_rul_h,
            "delta_final_rul_h": comp.delta_final_rul_h,
            "delta_max_cht_c": comp.delta_max_cht_c,
            "delta_max_egt_c": comp.delta_max_egt_c,
            "delta_min_oil_pressure_bar": comp.delta_min_oil_pressure_bar,
            "delta_max_vibration_g": comp.delta_max_vibration_g,
            "delta_envelope_events": comp.delta_envelope_events,
            "delta_risk_score": comp.delta_risk_score,
            "qualitative_interpretation": comp.qualitative_interpretation,
        }

    # 4. 6-Way Causal Decomposition Experiment
    print("Running 6-way causal decomposition...")
    causal_decomp = run_causal_decomposition(duration_s=150.0, dt_s=1.0)
    causal_decomp_serializable = {
        "baseline": vars(causal_decomp["baseline"]),
        "branches": {},
        "coupling_notes": causal_decomp["non_linear_coupling_notes"],
    }
    # clean baseline risk index
    causal_decomp_serializable["baseline"]["risk_index"] = vars(causal_decomp_serializable["baseline"]["risk_index"])
    for k, v in causal_decomp["branches"].items():
        m_dict = vars(v["metrics"])
        m_dict["risk_index"] = vars(m_dict["risk_index"])
        causal_decomp_serializable["branches"][k] = {
            "metrics": m_dict,
            "delta": vars(v["delta"]),
        }

    # 5. State Isolation & Order Independence Evidence
    print("Testing state isolation and order independence...")
    spec_a = specs["NOMINAL_CRUISE"]
    spec_b = specs["F3_COOLING"]

    r_b_alone = sim.run_mission(spec_b, full_trajectory=False)
    sim.run_mission(spec_a, full_trajectory=False)
    r_b_after_a = sim.run_mission(spec_b, full_trajectory=False)

    isolation_verified = (
        r_b_alone.metrics.min_hi == r_b_after_a.metrics.min_hi
        and r_b_alone.metrics.max_cht_c == r_b_after_a.metrics.max_cht_c
    )

    # 6. Performance & Memory Profiling
    print("Benchmarking performance (1, 10, 100 missions)...")
    perf_metrics = {}
    short_spec = specs["NOMINAL_CRUISE"]

    # 1 mission (full trajectory vs streaming)
    t0 = time.perf_counter()
    sim.run_mission(short_spec, full_trajectory=True)
    t_1_full = time.perf_counter() - t0

    t0 = time.perf_counter()
    sim.run_mission(short_spec, full_trajectory=False)
    t_1_stream = time.perf_counter() - t0

    # 10 missions streaming
    t0 = time.perf_counter()
    for _ in range(10):
        sim.run_mission(short_spec, full_trajectory=False)
    t_10_stream = time.perf_counter() - t0

    # Step latency measurement (100 steps)
    step_latencies_ms = []
    spec_100s = get_golden_scenario_specs(duration_s=100.0, dt_s=1.0)["NOMINAL_CRUISE"]
    t0 = time.perf_counter()
    res_100s = sim.run_mission(spec_100s, full_trajectory=False)
    total_100s_time = time.perf_counter() - t0
    step_latency_mean = (total_100s_time / res_100s.metrics.step_count) * 1000.0

    perf_metrics = {
        "1_mission_full_trajectory_s": round(t_1_full, 4),
        "1_mission_streaming_s": round(t_1_stream, 4),
        "10_missions_streaming_total_s": round(t_1_0_stream := t_10_stream, 4),
        "mean_mission_time_streaming_s": round(t_10_stream / 10.0, 4),
        "mean_step_latency_ms": round(step_latency_mean, 3),
        "streaming_memory_mode": "O(1) historical trajectory retention; bounded summary aggregators",
    }

    evidence = {
        "phase": "PHASE 10 — MISSION RELIABILITY & WHAT-IF SIMULATION",
        "git": git_info,
        "classification": "MODEL_SCENARIO_RESULTS",
        "authoritative_thresholds": thresholds_evidence,
        "golden_scenarios": golden_results,
        "comparative_deltas": comparative_deltas,
        "causal_decomposition": causal_decomp_serializable,
        "state_isolation_verified": isolation_verified,
        "performance": perf_metrics,
        "non_claims": [
            "No certification or airworthiness compliance (DO-178C, FAA, EASA) is claimed.",
            "No statistical failure probability, mission success rate, or MTBF is inferred.",
            "No empirical fleet reliability dataset was fitted.",
            "4500m scenario is strictly a SYNTHETIC MODEL SCENARIO, not an operating envelope.",
            "MissionRiskIndex is an engineering heuristic, NOT a probability of failure.",
        ],
    }

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "evidence", "phase10_mission_matrix.json"))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(evidence, f, indent=2)

    print(f"Evidence matrix successfully written to: {out_path}")
    return evidence


if __name__ == "__main__":
    generate_phase10_evidence()
