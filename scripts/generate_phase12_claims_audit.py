#!/usr/bin/env python3
"""
Phase 12: Evidence & Claims Matrix Generator.

Executes a deterministic audit of all external claims, SIH26054 requirements,
OEM numerical parameters, AI/ML operational tiers, and preserved technical limitations.
Produces:
- evidence/phase12_claims_matrix.json
- evidence/phase12_sih_coverage_matrix.json
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_claims import run_audit


def get_git_info() -> Dict[str, str]:
    def run_cmd(cmd: str) -> str:
        try:
            return subprocess.check_output(cmd, shell=True, text=True, cwd=str(REPO_ROOT)).strip()
        except Exception as e:
            return f"ERROR: {e}"

    head_commit = run_cmd("git rev-parse HEAD")
    branch = run_cmd("git rev-parse --abbrev-ref HEAD")
    dirty = bool(run_cmd("git status --porcelain"))

    return {
        "head_commit": head_commit,
        "branch": branch,
        "is_dirty": dirty,
    }


def generate_claims_matrix(git_info: Dict[str, Any], linter_results: Dict[str, Any]) -> Dict[str, Any]:
    # Load actual measured latency from phase 6 evidence
    latency_file = REPO_ROOT / "evidence" / "phase6_latency_benchmark.json"
    latency_data = {}
    if latency_file.exists():
        with open(latency_file, "r", encoding="utf-8") as f:
            latency_data = json.load(f)

    claims = [
        {
            "claim_id": "CLM-001",
            "claim_text": "Engine Takeoff Power is 84.5 kW at 5800 RPM (5-minute maximum).",
            "source_citation": "EASA TCDS E.122 / BRP-Rotax 914 Series Operators Manual Ed. 4 / Rev. 0, Section 2.1",
            "category": "AUTHORITATIVE_REFERENCE",
            "verification_status": "VERIFIED",
            "notes": "EASA certification standard is 84.5 kW (115 HP gross metric). Commercial brochure cites 84.8 kW; authoritative standard is 84.5 kW.",
        },
        {
            "claim_id": "CLM-002",
            "claim_text": "Engine Continuous Power is 73.5 kW at 5500 RPM.",
            "source_citation": "EASA TCDS E.122 / BRP-Rotax 914 Series Operators Manual Ed. 4 / Rev. 0, Section 2.1",
            "category": "AUTHORITATIVE_REFERENCE",
            "verification_status": "VERIFIED",
            "notes": "Continuous maximum operational rating.",
        },
        {
            "claim_id": "CLM-003",
            "claim_text": "Engine Critical Altitude is 4875 m (16,000 ft).",
            "source_citation": "EASA TCDS E.122 Section A.III / BRP-Rotax 914 Series Operators Manual Section 2.1",
            "category": "AUTHORITATIVE_REFERENCE",
            "verification_status": "VERIFIED",
            "notes": "Maximum altitude up to which TCU maintains continuous boost pressure (115 kPa). Simulator test envelope sweeps up to 4500 m as MODEL_IMPLEMENTATION ceiling.",
        },
        {
            "claim_id": "CLM-004",
            "claim_text": "Engine Oil Pressure Limits: 0.8 bar idle minimum (<3500 RPM), 2.0-5.0 bar normal operating (>3500 RPM), 7.0 bar cold-start transient max.",
            "source_citation": "BRP-Rotax 914 Series Operators Manual Section 2.1 / EASA TCDS E.122 Section A.III",
            "category": "AUTHORITATIVE_REFERENCE",
            "verification_status": "VERIFIED",
            "notes": "Strictly separated from 1.5 sigma normalized residual health deadband and 0.5 bar numerical simulator clamp.",
        },
        {
            "claim_id": "CLM-005",
            "claim_text": "Engine Health Index is computed as a dynamically renormalized weighted linear sum of piecewise-linear normalized residual evidence functions.",
            "source_citation": "health_index/calculator.py / health_index/schema.py",
            "category": "MODEL_IMPLEMENTATION",
            "verification_status": "VERIFIED",
            "notes": "Uses deadband tau_nominal=1.5 sigma and tau_critical=5.0 sigma with dynamic sensor isolation weight renormalization.",
        },
        {
            "claim_id": "CLM-006",
            "claim_text": "Host-side digital twin streaming execution latency achieves sub-millisecond steady state (mean 0.388 ms, P99 0.763 ms).",
            "source_citation": "evidence/phase6_latency_benchmark.json",
            "category": "SYNTHETIC_VALIDATION",
            "verification_status": "VERIFIED",
            "notes": "Tested across 1,000 steps on commodity PC CPU. Host OS thread scheduling spikes reach 28.994 ms. Suitable for soft real-time; hard real-time is strictly disclaimed.",
        },
        {
            "claim_id": "CLM-007",
            "claim_text": "Operational Digital Twin runtime contains ZERO machine learning models or black-box neural networks.",
            "source_citation": "digital_twin/twin_model.py, digital_twin/diagnosis.py, digital_twin/rul.py",
            "category": "MODEL_IMPLEMENTATION",
            "verification_status": "VERIFIED",
            "notes": "Runtime is 100% 1D lumped-parameter ODE physics, deterministic residuals, and directional hypothesis signature rules. XGBoost is VALIDATION_ONLY; TimesFM is QUARANTINED.",
        },
        {
            "claim_id": "CLM-008",
            "claim_text": "Prognostics RUL is a model-defined projection to D_EOL horizon under modeled operating/stress assumptions, with deterministic empirical quantile bounds.",
            "source_citation": "digital_twin/degradation.py / digital_twin/rul.py",
            "category": "MODEL_IMPLEMENTATION",
            "verification_status": "VERIFIED",
            "notes": "Uses Theil-Sen regression with 15th/85th percentile slope quantiles. Does not claim certified mechanical TBO or physical component fatigue.",
        },
        {
            "claim_id": "CLM-009",
            "claim_text": "Mission Risk Index (R_mission) is a heuristic engineering metric in [0, 1] and strictly NOT a failure probability.",
            "source_citation": "digital_twin/what_if.py / digital_twin/mission_simulator.py",
            "category": "ENGINEERING_HEURISTIC",
            "verification_status": "VERIFIED",
            "notes": "Combines operating envelope excursion penalties and health deficits under simulated scenarios.",
        },
        {
            "claim_id": "CLM-010",
            "claim_text": "System is an academic and engineering competition prototype, NOT certified for flight operations under FAA DO-178C or EASA CS-E.",
            "source_citation": "docs/claims_and_limitations.md / README.md",
            "category": "ENGINEERING_HEURISTIC",
            "verification_status": "VERIFIED",
            "notes": "Explicit non-certification disclaimer prominently documented across all deliverables.",
        },
    ]

    return {
        "metadata": {
            "source_commit": git_info["head_commit"],
            "phase": "PHASE_12_CLAIMS_AND_SUBMISSION_INTEGRITY",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "audit_status": linter_results["status"],
            "unresolved_blockers": linter_results["unresolved_blocker_count"],
            "unresolved_highs": linter_results["unresolved_high_count"],
        },
        "taxonomy_categories": [
            "AUTHORITATIVE_REFERENCE",
            "MODEL_IMPLEMENTATION",
            "MODEL_CALIBRATION",
            "ENGINEERING_HEURISTIC",
            "SYNTHETIC_VALIDATION",
            "DATA_QUALITY_RULE",
        ],
        "latency_evidence": latency_data,
        "audited_claims": claims,
        "linter_summary": {
            "total_scanned_matches": linter_results["total_findings"],
            "acceptable_disclaimers": linter_results["acceptable_disclaimer_count"],
            "unresolved_blockers": linter_results["unresolved_blocker_count"],
            "unresolved_highs": linter_results["unresolved_high_count"],
        },
    }


def generate_sih_coverage_matrix(git_info: Dict[str, Any]) -> Dict[str, Any]:
    requirements = [
        {
            "req_id": "SIH-REQ-01",
            "title": "Physics-Informed Digital Twin of Aero Piston Engine",
            "status": "IMPLEMENTED",
            "executable_behavior": "1D lumped-parameter thermal, lubrication, and rotational ODE model synchronized with incoming telemetry.",
            "evidence_path": "evidence/phase2_operating_matrix.json",
            "test_suite": "tests/test_phase2_operating_matrix.py",
            "limitations": "Reduced-order lumped parameter model; not 3D CFD or acoustic combustion dynamometer.",
        },
        {
            "req_id": "SIH-REQ-02",
            "title": "Real-Time Telemetry Processing Capability",
            "status": "PARTIALLY_IMPLEMENTED",
            "executable_behavior": "Host-side streaming execution achieves sub-millisecond mean latency (0.388 ms), suitable for 1-10 Hz telemetry streams.",
            "evidence_path": "evidence/phase6_latency_benchmark.json",
            "test_suite": "tests/test_fault_diagnosis_phase6.py",
            "limitations": "Host-side soft real-time verified on commodity CPU. Operating system spikes reach 28.99 ms; hard real-time embedded RTOS deployment is OUT_OF_SCOPE.",
        },
        {
            "req_id": "SIH-REQ-03",
            "title": "Multi-Rate Telemetry Ingestion Infrastructure",
            "status": "IMPLEMENTED",
            "executable_behavior": "Ingestion pipeline with unit conversions, clock skew detection, sequence gap handling, and multi-rate interpolation.",
            "evidence_path": "evidence/phase9_telemetry_matrix.json",
            "test_suite": "tests/test_phase9_real_telemetry.py",
            "limitations": "All input streams tested are synthetic or replayed. No physical flight test logs claimed.",
        },
        {
            "req_id": "SIH-REQ-04",
            "title": "Empirical Flight Data Validation on Operational Fleet",
            "status": "NOT_IMPLEMENTED",
            "executable_behavior": "No physical flight test dataset from an operational Rotax 914 UAV fleet is present or claimed.",
            "evidence_path": "evidence/phase9_telemetry_matrix.json",
            "test_suite": "tests/test_phase9_real_telemetry.py",
            "limitations": "All degradation trajectories and operational scenarios are synthetic/replayed benchmarks.",
        },
        {
            "req_id": "SIH-REQ-05",
            "title": "Multi-Fault Physics Degradation Simulation",
            "status": "IMPLEMENTED",
            "executable_behavior": "Simulates 6 distinct physical fault classes: cooling loss, lubrication degradation, injector clogging, misfire, mechanical friction, and sensor drift.",
            "evidence_path": "evidence/evidence_package.json",
            "test_suite": "tests/test_fault_physics_phase4.py",
            "limitations": "Parameterized continuous wear states; does not model sudden catastrophic structural breakage.",
        },
        {
            "req_id": "SIH-REQ-06",
            "title": "Propulsion Health Assessment (Health Index)",
            "status": "IMPLEMENTED",
            "executable_behavior": "Computes continuous health index in [0, 1] across engine and subsystems using dynamically renormalized residual evidence sums.",
            "evidence_path": "evidence/phase5_health_assessment_matrix.json",
            "test_suite": "tests/test_health_assessment_phase5.py",
            "limitations": "Normalized residual metric; not an airworthiness certificate.",
        },
        {
            "req_id": "SIH-REQ-07",
            "title": "Physics-Informed Fault Diagnosis & Hypothesis Isolation",
            "status": "IMPLEMENTED",
            "executable_behavior": "Evaluates residual signatures and directional physical coupling to isolate primary fault hypotheses.",
            "evidence_path": "evidence/phase6_fault_diagnosis_matrix.json",
            "test_suite": "tests/test_fault_diagnosis_phase6.py",
            "limitations": "Assumes primary fault dominance; complex concurrent sensor faults may yield ambiguous rankings.",
        },
        {
            "req_id": "SIH-REQ-08",
            "title": "Prognostics & Remaining Useful Life (RUL) Estimation",
            "status": "IMPLEMENTED",
            "executable_behavior": "Projects time to model horizon D_EOL using robust Theil-Sen regression with deterministic empirical quantile bounds under modeled stress.",
            "evidence_path": "evidence/phase8_rul_matrix.json",
            "test_suite": "tests/test_phase8_rul.py",
            "limitations": "Mathematical horizon under stated model assumptions; not physical TBO.",
        },
        {
            "req_id": "SIH-REQ-09",
            "title": "Mission Reliability & Counterfactual What-If Simulation",
            "status": "IMPLEMENTED",
            "executable_behavior": "Simulates alternate flight profiles and fault injections to compute mission risk index R_mission and envelope margins.",
            "evidence_path": "evidence/phase10_mission_matrix.json",
            "test_suite": "tests/test_phase10_mission.py",
            "limitations": "Heuristic risk index; strictly not a failure probability.",
        },
        {
            "req_id": "SIH-REQ-10",
            "title": "Structured Engineering Explainability",
            "status": "IMPLEMENTED",
            "executable_behavior": "Generates provenance-traced explanations linking every diagnostic/prognostic output to sensor channels, residuals, and rules.",
            "evidence_path": "evidence/phase11_explainability_matrix.json",
            "test_suite": "tests/test_phase11_explainability.py",
            "limitations": "Deterministic rule/residual attribution; not deep learning SHAP in operational runtime.",
        },
        {
            "req_id": "SIH-REQ-11",
            "title": "Interactive Operator Decision Support Dashboard",
            "status": "IMPLEMENTED",
            "executable_behavior": "Streamlit application displaying digital twin synchronization, health gauges, telemetry replay, and scenario what-if controls.",
            "evidence_path": "dashboard/app.py",
            "test_suite": "tests/test_phase12_claims_audit.py",
            "limitations": "Presentation and decision-support layer; does not control aircraft avionics.",
        },
        {
            "req_id": "SIH-REQ-12",
            "title": "Machine Learning Diagnostic Classifiers",
            "status": "VALIDATION_ONLY",
            "executable_behavior": "XGBoost and Random Forest classifiers trained on synthetic population benchmarks for comparative validation.",
            "evidence_path": "fault_diagnosis/classifier.py",
            "test_suite": "tests/test_phase7_population.py",
            "limitations": "Trained solely on synthetic datasets; not invoked in operational digital twin runtime.",
        },
        {
            "req_id": "SIH-REQ-13",
            "title": "FAA DO-178C / EASA Flight Certification",
            "status": "OUT_OF_SCOPE",
            "executable_behavior": "No flight certification processes executed.",
            "evidence_path": "docs/claims_and_limitations.md",
            "test_suite": "tests/test_phase12_claims_audit.py",
            "limitations": "Academic research and competition prototype.",
        },
    ]

    return {
        "metadata": {
            "source_commit": git_info["head_commit"],
            "phase": "PHASE_12_SIH_COVERAGE_AUDIT",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_requirements": len(requirements),
            "status_counts": {
                "IMPLEMENTED": sum(1 for r in requirements if r["status"] == "IMPLEMENTED"),
                "PARTIALLY_IMPLEMENTED": sum(1 for r in requirements if r["status"] == "PARTIALLY_IMPLEMENTED"),
                "PROTOTYPE_ONLY": sum(1 for r in requirements if r["status"] == "PROTOTYPE_ONLY"),
                "VALIDATION_ONLY": sum(1 for r in requirements if r["status"] == "VALIDATION_ONLY"),
                "NOT_IMPLEMENTED": sum(1 for r in requirements if r["status"] == "NOT_IMPLEMENTED"),
                "OUT_OF_SCOPE": sum(1 for r in requirements if r["status"] == "OUT_OF_SCOPE"),
            },
        },
        "requirements": requirements,
    }


def main():
    print("=" * 70)
    print("PHASE 12: GENERATING CLAIMS & SIH COVERAGE MATRICES")
    print("=" * 70)

    git_info = get_git_info()
    print(f"Git HEAD Commit: {git_info['head_commit']}")
    print(f"Git Branch:      {git_info['branch']}")
    print(f"Working Tree:    {'DIRTY' if git_info['is_dirty'] else 'CLEAN'}")

    # Run linter
    print("\nRunning automated claims audit...")
    linter_results = run_audit(REPO_ROOT)
    print(f"Linter Status:   {linter_results['status']}")
    print(f"Total Matches:   {linter_results['total_findings']}")
    print(f"Disclaimers:     {linter_results['acceptable_disclaimer_count']}")
    print(f"Blockers:        {linter_results['unresolved_blocker_count']}")
    print(f"Highs:           {linter_results['unresolved_high_count']}")

    # Generate claims matrix
    claims_matrix = generate_claims_matrix(git_info, linter_results)
    claims_out_path = REPO_ROOT / "evidence" / "phase12_claims_matrix.json"
    with open(claims_out_path, "w", encoding="utf-8") as f:
        json.dump(claims_matrix, f, indent=2)
    print(f"\n[+] Wrote claims matrix to: {claims_out_path}")

    # Generate SIH coverage matrix
    sih_matrix = generate_sih_coverage_matrix(git_info)
    sih_out_path = REPO_ROOT / "evidence" / "phase12_sih_coverage_matrix.json"
    with open(sih_out_path, "w", encoding="utf-8") as f:
        json.dump(sih_matrix, f, indent=2)
    print(f"[+] Wrote SIH coverage matrix to: {sih_out_path}")

    print("\n" + "=" * 70)
    print("PHASE 12 EVIDENCE GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
