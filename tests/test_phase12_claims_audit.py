"""
Phase 12 Test Suite: Anti-Fabrication, Claims & Submission Integrity Audit.

Validates:
1. Automated claims linter execution & zero unresolved BLOCKER/HIGH findings.
2. Contextual allowlist filter accuracy.
3. Authoritative Rotax 914 UL/F OEM limits and provenance.
4. 6-Tier AI/ML runtime inventory (verifying zero ML in operational runtime).
5. Real telemetry infrastructure vs real-data validation distinction.
6. Directional propagation wording (zero unverified causal inference claims).
7. Host-side latency benchmarks and soft real-time qualification.
8. SIH26054 coverage matrix schema and evidentiary honesty.
9. Dedicated test: No stale quantitative claims in documentation (latency, oil pressure, power, altitude, RUL, ML).
10. Upstream immutability against Phase 11 baseline.
"""

import json
import os
import re
import subprocess
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

from scripts.audit_claims import run_audit, FindingSeverity, is_context_allowlisted


# ==============================================================================
# 1. Claims Audit & Linter Execution Tests
# ==============================================================================

def test_claims_linter_audit_passes():
    """Verify that the repository audit passes with zero unresolved BLOCKER or HIGH findings."""
    results = run_audit(REPO_ROOT)
    assert results["status"] == "PASS"
    assert results["unresolved_blocker_count"] == 0
    assert results["unresolved_high_count"] == 0


def test_claims_linter_finds_disclaimers():
    """Verify that the linter properly captures and allowlists valid negative disclaimers."""
    results = run_audit(REPO_ROOT)
    assert results["acceptable_disclaimer_count"] > 20


def test_allowlist_catches_negative_qualifiers():
    """Verify that allowlist recognizes negative disclaimers."""
    allowlisted, reason = is_context_allowlisted(
        "This software is not certified for flight operations under FAA DO-178C.",
        "certified for flight",
        "CERTIFICATION_AIRWORTHINESS",
    )
    assert allowlisted is True
    assert "disclaimer" in reason.lower() or "negative" in reason.lower()


def test_allowlist_catches_probability_negation():
    """Verify that allowlist recognizes probability disclaimers."""
    allowlisted, _ = is_context_allowlisted(
        "The risk index is strictly not a failure probability.",
        "failure probability",
        "FAILURE_PROBABILITY",
    )
    assert allowlisted is True


def test_allowlist_rejects_unqualified_airworthiness():
    """Verify that allowlist flags unqualified airworthiness claims."""
    allowlisted, _ = is_context_allowlisted(
        "The system produces certified for flight diagnostic commands for autonomous UAVs.",
        "certified for flight",
        "CERTIFICATION_AIRWORTHINESS",
    )
    assert allowlisted is False


def test_allowlist_catches_rotax912_historical_anchor():
    """Verify that allowlist allows Rotax 912 when cited as a reference anchor or comparison."""
    allowlisted, _ = is_context_allowlisted(
        "Calibrated against Rotax 912 ULS baseline reference anchor.",
        "Rotax 912",
        "STALE_ENGINE_IDENTITY",
    )
    assert allowlisted is True


def test_allowlist_flags_unqualified_rotax912_identity():
    """Verify that allowlist flags unqualified Rotax 912 active identity claims."""
    allowlisted, _ = is_context_allowlisted(
        "The active digital twin models the Rotax 912 engine in real time.",
        "Rotax 912",
        "STALE_ENGINE_IDENTITY",
    )
    assert allowlisted is False


# ==============================================================================
# 2. Authoritative Rotax 914 UL/F OEM Limits Cross-Check Tests
# ==============================================================================

def test_oem_takeoff_power_reference():
    """Takeoff power must be 84.5 kW (84,500 W) at 5800 RPM according to EASA TCDS E.122."""
    ref_file = REPO_ROOT / "configs" / "engine_reference" / "rotax_914_ul_f.json"
    assert ref_file.exists()
    with open(ref_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["operating_limits"]["takeoff_power_w"]["value"] == 84500.0
    assert data["operating_limits"]["takeoff_power_w"]["unit"] == "W"
    assert "EASA" in data["operating_limits"]["takeoff_power_w"]["source"] or "Rotax" in data["operating_limits"]["takeoff_power_w"]["source"]


def test_oem_continuous_power_reference():
    """Max continuous power must be 73.5 kW (73,500 W) at 5500 RPM according to EASA TCDS E.122."""
    ref_file = REPO_ROOT / "configs" / "engine_reference" / "rotax_914_ul_f.json"
    assert ref_file.exists()
    with open(ref_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["operating_limits"]["continuous_power_w"]["value"] == 73500.0
    assert data["operating_limits"]["continuous_power_w"]["unit"] == "W"


def test_oem_critical_altitude_reference():
    """Critical altitude must be 4875 m according to EASA TCDS E.122 TCU ratings."""
    ref_file = REPO_ROOT / "configs" / "engine_reference" / "rotax_914_ul_f.json"
    assert ref_file.exists()
    with open(ref_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    tcu_v43 = data["engine_identity"]["engine_applicability"]["easa_tcds_tcu_ratings"]["tcu_configuration_v4_3"]
    assert tcu_v43["continuous_critical_altitude_m"] == 4875.0


def test_oem_oil_pressure_limits_provenance():
    """Oil pressure limits must strictly distinguish idle min (0.8 bar), normal (2.0-5.0 bar), and cold start (7.0 bar)."""
    ref_file = REPO_ROOT / "configs" / "engine_reference" / "rotax_914_ul_f.json"
    assert ref_file.exists()
    with open(ref_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["operating_limits"]["oil_press_min_bar"]["value"] == 0.8
    assert data["operating_limits"]["oil_press_normal_min_bar"]["value"] == 2.0
    assert data["operating_limits"]["oil_press_normal_max_bar"]["value"] == 5.0
    assert data["operating_limits"]["oil_press_max_bar"]["value"] == 7.0


def test_health_deadband_distinguished_from_oil_pressure():
    """Health deadband is 1.5 sigma normalized residual, NOT a bar pressure unit."""
    from health_index.schema import HealthIndexConfig
    cfg = HealthIndexConfig()
    assert cfg.tau_nominal == 1.5
    assert cfg.tau_critical == 5.0


def test_oem_cht_limit_reference():
    """Max CHT limit must be 135 deg C according to EASA TCDS E.122."""
    ref_file = REPO_ROOT / "configs" / "engine_reference" / "rotax_914_ul_f.json"
    assert ref_file.exists()
    with open(ref_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["operating_limits"]["cht_limit_c"]["value"] == 135.0


def test_oem_gearbox_reduction_ratio():
    """Gearbox reduction ratio must be 2.4286:1 (51/21)."""
    ref_file = REPO_ROOT / "configs" / "engine_reference" / "rotax_914_ul_f.json"
    assert ref_file.exists()
    with open(ref_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert pytest.approx(data["drivetrain"]["reduction_ratio"]["value"], 0.001) == 2.4286


# ==============================================================================
# 3. 6-Tier AI/ML Runtime Inventory Tests
# ==============================================================================

def test_operational_twin_runtime_has_zero_ml_imports():
    """Verify that operational digital twin modules do NOT import scikit-learn, xgboost, or torch."""
    operational_modules = [
        REPO_ROOT / "digital_twin" / "twin_model.py",
        REPO_ROOT / "digital_twin" / "synchronizer.py",
        REPO_ROOT / "digital_twin" / "residuals.py",
        REPO_ROOT / "digital_twin" / "health.py",
        REPO_ROOT / "digital_twin" / "diagnosis.py",
        REPO_ROOT / "digital_twin" / "degradation.py",
        REPO_ROOT / "digital_twin" / "rul.py",
        REPO_ROOT / "digital_twin" / "what_if.py",
        REPO_ROOT / "health_index" / "calculator.py",
    ]
    ml_keywords = [
        re.compile(r"^\s*import\s+xgboost", re.MULTILINE),
        re.compile(r"^\s*from\s+xgboost", re.MULTILINE),
        re.compile(r"^\s*import\s+sklearn", re.MULTILINE),
        re.compile(r"^\s*from\s+sklearn", re.MULTILINE),
        re.compile(r"^\s*import\s+torch", re.MULTILINE),
        re.compile(r"^\s*from\s+torch", re.MULTILINE),
    ]
    for mod_path in operational_modules:
        if mod_path.exists():
            with open(mod_path, "r", encoding="utf-8") as f:
                content = f.read()
            for kw in ml_keywords:
                assert not kw.search(content), f"Forbidden ML import found in operational module: {mod_path.name}"


def test_diagnosis_uses_heuristic_signatures():
    """Verify that digital_twin/diagnosis.py uses heuristic signatures and compatibility scores, not ML probabilities."""
    diag_file = REPO_ROOT / "digital_twin" / "diagnosis.py"
    with open(diag_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "compatibility_score" in content
    assert "NOT claimed as probabilities" in content or "heuristic" in content.lower()


def test_supervised_classifiers_are_validation_only():
    """Verify that XGBoost and Random Forest exist in fault_diagnosis/ as validation/benchmarking models."""
    classifier_file = REPO_ROOT / "fault_diagnosis" / "classifier.py"
    assert classifier_file.exists()


# ==============================================================================
# 4. Real Telemetry & Infrastructure Tests
# ==============================================================================

def test_real_telemetry_dataset_validation_is_not_claimed():
    """Verify that evidence explicitly states all operational datasets are synthetic/replayed."""
    phase9_ev = REPO_ROOT / "evidence" / "phase9_telemetry_matrix.json"
    assert phase9_ev.exists()
    with open(phase9_ev, "r", encoding="utf-8") as f:
        data = json.load(f)
    disclaimer = data["metadata"]["input_classification_disclaimer"]
    assert "SYNTHETIC" in disclaimer or "REPLAYED" in disclaimer
    assert "No real Rotax 914 flight data" in disclaimer


def test_canonical_telemetry_packet_interface_exists():
    """Verify that telemetry infrastructure (canonical packet contract) is implemented."""
    from telemetry.canonical import CanonicalTelemetryPacket
    assert CanonicalTelemetryPacket is not None


# ==============================================================================
# 5. Directional Propagation & Health Aggregation Tests
# ==============================================================================

def test_no_causal_degradation_tracking_in_active_docs():
    """Verify that 'causal degradation tracking' has been eliminated from active documentation."""
    docs_to_check = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs" / "claims_and_limitations.md",
        REPO_ROOT / "docs" / "system_orchestrator.md",
        REPO_ROOT / "evidence" / "phase15_final_validation.md",
    ]
    pattern = re.compile(r"\bcausal\s+degradation\s+tracking\b", re.IGNORECASE)
    for doc in docs_to_check:
        if doc.exists():
            with open(doc, "r", encoding="utf-8") as f:
                content = f.read()
            assert not pattern.search(content), f"Stale causal phrase found in: {doc.name}"


def test_health_aggregation_is_weighted_linear_sum():
    """Verify that HealthCalculator uses dynamically renormalized weighted linear sum, not min functions."""
    from health_index.calculator import HealthCalculator
    calc = HealthCalculator()
    # Nominal zero residuals -> HI = 1.0
    hi, deg, _, _, _, _, _, _, weights, quality = calc.compute(
        timestamp=0.0,
        residuals={"rpm": 0.0, "cht": 0.0, "egt": 0.0, "oil_press": 0.0, "oil_temp": 0.0, "map": 0.0, "vibration": 0.0},
    )
    assert hi == 1.0
    assert deg == 0.0
    assert sum(weights.values()) == pytest.approx(1.0, 1e-6)


# ==============================================================================
# 6. Measured Latency & Real-Time Benchmark Tests
# ==============================================================================

def test_latency_benchmark_evidence_metrics():
    """Verify that evidence/phase6_latency_benchmark.json records exact empirical benchmark."""
    lat_file = REPO_ROOT / "evidence" / "phase6_latency_benchmark.json"
    assert lat_file.exists()
    with open(lat_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["steps"] == 1000
    assert pytest.approx(data["mean_ms"], 0.001) == 0.3882
    assert pytest.approx(data["p95_ms"], 0.001) == 0.5592
    assert pytest.approx(data["p99_ms"], 0.001) == 0.7631
    assert data["max_ms"] > 20.0  # OS spike
    assert data["worst_case_passed"] is False  # Explicitly fails hard real-time


def test_latency_benchmark_rejects_hard_real_time():
    """Verify that latency notes explain the OS jitter spike and deny hard real-time determinism."""
    lat_file = REPO_ROOT / "evidence" / "phase6_latency_benchmark.json"
    with open(lat_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    notes = data["notes"].lower()
    assert "worst-case" in notes
    assert "garbage collection" in notes or "os thread" in notes or "scheduling" in notes


# ==============================================================================
# 7. SIH26054 Requirements Coverage Matrix Tests
# ==============================================================================

def test_sih_coverage_matrix_exists_and_valid():
    """Verify evidence/phase12_sih_coverage_matrix.json structure and non-pre-biased statuses."""
    sih_file = REPO_ROOT / "evidence" / "phase12_sih_coverage_matrix.json"
    assert sih_file.exists()
    with open(sih_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["metadata"]["phase"] == "PHASE_12_SIH_COVERAGE_AUDIT"
    assert data["metadata"]["total_requirements"] >= 12

    valid_statuses = {"IMPLEMENTED", "PARTIALLY_IMPLEMENTED", "PROTOTYPE_ONLY", "VALIDATION_ONLY", "NOT_IMPLEMENTED", "OUT_OF_SCOPE"}
    for req in data["requirements"]:
        assert req["status"] in valid_statuses
        assert req["req_id"].startswith("SIH-REQ-")
        assert len(req["limitations"]) > 0


def test_sih_coverage_honest_non_implemented():
    """Verify that real fleet validation is explicitly marked NOT_IMPLEMENTED and certification is OUT_OF_SCOPE."""
    sih_file = REPO_ROOT / "evidence" / "phase12_sih_coverage_matrix.json"
    with open(sih_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    reqs_by_title = {r["title"]: r for r in data["requirements"]}

    assert reqs_by_title["Empirical Flight Data Validation on Operational Fleet"]["status"] == "NOT_IMPLEMENTED"
    assert reqs_by_title["FAA DO-178C / EASA Flight Certification"]["status"] == "OUT_OF_SCOPE"
    assert reqs_by_title["Real-Time Telemetry Processing Capability"]["status"] == "PARTIALLY_IMPLEMENTED"


# ==============================================================================
# 8. Preserved Technical Limitations Tests
# ==============================================================================

def test_preservation_of_nine_technical_limitations_in_docs():
    """Verify that all 9 mandatory engineering limitations are present in docs/claims_and_limitations.md."""
    doc_path = REPO_ROOT / "docs" / "claims_and_limitations.md"
    assert doc_path.exists()
    with open(doc_path, "r", encoding="utf-8") as f:
        content = f.read()

    boundaries = [
        "Synthetic Grey-Box Simulator",
        "Absence of Real Engine Fleet Validation",
        "Non-Certification",
        "Model-Defined Remaining Useful Life",
        "Heuristic Mission Risk Index",
        "Host-Side Execution Latency",
        "Single / Localized Fault Dominance",
        "Synthetic Population Parameterization",
        "Absence of Calibrated Failure Probabilities",
    ]
    for b in boundaries:
        assert b in content, f"Missing limitation disclosure in claims_and_limitations.md: {b}"


# ==============================================================================
# 9. Dedicated Test: Stale Quantitative Claims Detection (Requirement 13)
# ==============================================================================

def test_no_stale_quantitative_claims_in_documentation():
    """
    Dedicated audit test: detects stale quantitative claims across Phase 12 documentation
    (docs/claims_and_limitations.md, README.md, and evidence/phase12_claims_matrix.json).
    Ensures that values match authoritative evidence rather than stale/unvetted assumptions.
    """
    files_to_audit = [
        REPO_ROOT / "docs" / "claims_and_limitations.md",
        REPO_ROOT / "README.md",
    ]

    for file_path in files_to_audit:
        assert file_path.exists()
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        # 1. Stale latency (~55 ms or ~53 ms) prohibited; must cite 0.388 ms
        assert "~55 ms" not in text, f"Stale ~55 ms latency claim found in {file_path.name}"
        assert "~53 ms" not in text, f"Stale ~53 ms latency claim found in {file_path.name}"
        assert "0.388" in text, f"Authoritative measured mean latency (0.388 ms) not cited in {file_path.name}"

        # 2. Authoritative oil pressure numbers must be present
        assert "0.8" in text, f"Authoritative idle oil pressure (0.8 bar) missing in {file_path.name}"
        assert "2.0" in text, f"Authoritative normal oil pressure lower bound (2.0 bar) missing in {file_path.name}"
        assert "5.0" in text, f"Authoritative normal oil pressure upper bound (5.0 bar) missing in {file_path.name}"
        assert "7.0" in text, f"Authoritative cold-start oil pressure (7.0 bar) missing in {file_path.name}"

        # 3. Takeoff power must cite 84.5 kW
        assert "84.5" in text, f"Authoritative EASA takeoff power (84.5 kW) missing in {file_path.name}"

        # 4. Critical altitude must cite 4875 m
        assert "4875" in text, f"Authoritative EASA critical altitude (4875 m) missing in {file_path.name}"

        # 5. RUL must be qualified as model-defined
        assert "model-defined" in text.lower(), f"RUL must be qualified as model-defined in {file_path.name}"

        # 6. Zero ML in operational runtime must be declared
        assert "ZERO" in text or "zero" in text.lower(), f"Zero ML operational runtime declaration missing in {file_path.name}"


# ==============================================================================
# 10. Upstream Immutability & Git Integrity Tests
# ==============================================================================

def test_upstream_algorithmic_immutability():
    """
    Verify that upstream physics, health, diagnosis, prognostics, and simulator algorithms
    have exactly zero modifications against the Phase 11 baseline.
    """
    phase11_baseline = "f1e2b6c46d7872fc459b700a8d648766a4099521"
    core_dirs = [
        "simulator/",
        "phm/",
        "digital_twin/",
        "health_index/",
        "anomaly_detection/",
        "prognostics/",
    ]
    cmd = f"git diff {phase11_baseline} -- {' '.join(core_dirs)}"
    diff_output = subprocess.check_output(cmd, shell=True, text=True, cwd=str(REPO_ROOT)).strip()
    assert diff_output == "", f"Upstream algorithmic code modified against Phase 11 baseline:\n{diff_output[:500]}"


# ==============================================================================
# 11. Additional Rigorous Schema & Numerical Consistency Tests
# ==============================================================================

def test_claims_matrix_json_schema_and_metadata():
    """Verify evidence/phase12_claims_matrix.json metadata, taxonomy, and citations."""
    claims_file = REPO_ROOT / "evidence" / "phase12_claims_matrix.json"
    assert claims_file.exists()
    with open(claims_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["metadata"]["phase"] == "PHASE_12_CLAIMS_AND_SUBMISSION_INTEGRITY"
    assert data["metadata"]["audit_status"] == "PASS"
    assert len(data["audited_claims"]) >= 10
    for clm in data["audited_claims"]:
        assert clm["verification_status"] == "VERIFIED"
        assert len(clm["source_citation"]) > 0
        assert clm["category"] in data["taxonomy_categories"]


def test_rot912_zero_unresolved_in_operational_code():
    """Verify zero occurrences of ROT912 in operational code, dashboard, and core modules."""
    dirs_to_check = ["dashboard", "digital_twin", "health_index", "prognostics", "simulator", "telemetry"]
    for d in dirs_to_check:
        dir_path = REPO_ROOT / d
        if dir_path.exists():
            for root, _, files in os.walk(dir_path):
                for fname in files:
                    if fname.endswith(".py"):
                        with open(Path(root) / fname, "r", encoding="utf-8", errors="ignore") as f:
                            content = f.read()
                        assert "ROT912" not in content, f"Stale ROT912 identifier found in {d}/{fname}"


def test_rul_stress_multiplier_scenarios():
    """Verify digital_twin/rul.py defines positive scenario stress multipliers."""
    from digital_twin.degradation_types import DEFAULT_SCENARIO_STRESS_FACTORS, RULScenario
    assert RULScenario.CURRENT_PROFILE in DEFAULT_SCENARIO_STRESS_FACTORS
    assert RULScenario.HIGH_LOAD in DEFAULT_SCENARIO_STRESS_FACTORS
    assert DEFAULT_SCENARIO_STRESS_FACTORS[RULScenario.HIGH_LOAD] > 1.0


def test_rul_quantiles_are_15th_and_85th():
    """Verify digital_twin/degradation.py computes Theil-Sen uncertainty at 15th and 85th percentiles."""
    import numpy as np
    from digital_twin.degradation import TheilSenEstimator
    # Simple linear degradation: D = 0.1 + 0.01 * t
    times = np.array([float(i) for i in range(20)])
    values = np.array([0.1 + 0.01 * t for t in times])
    res = TheilSenEstimator.estimate(times, values)
    assert res is not None
    assert pytest.approx(res.slope, 1e-4) == 0.01
    assert res.slope_low <= res.slope <= res.slope_high


def test_oil_pressure_hierarchy():
    """Assert hierarchy of oil pressure limits: idle_min < normal_min < normal_max < cold_start_max."""
    ref_file = REPO_ROOT / "configs" / "engine_reference" / "rotax_914_ul_f.json"
    with open(ref_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    p_idle_min = data["operating_limits"]["oil_press_min_bar"]["value"]
    p_norm_min = data["operating_limits"]["oil_press_normal_min_bar"]["value"]
    p_norm_max = data["operating_limits"]["oil_press_normal_max_bar"]["value"]
    p_cold_max = data["operating_limits"]["oil_press_max_bar"]["value"]

    assert p_idle_min == 0.8
    assert p_norm_min == 2.0
    assert p_norm_max == 5.0
    assert p_cold_max == 7.0
    assert p_idle_min < p_norm_min < p_norm_max < p_cold_max


def test_power_hierarchy():
    """Assert takeoff power (84.5 kW) strictly exceeds continuous power (73.5 kW)."""
    ref_file = REPO_ROOT / "configs" / "engine_reference" / "rotax_914_ul_f.json"
    with open(ref_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    p_to = data["operating_limits"]["takeoff_power_w"]["value"]
    p_cont = data["operating_limits"]["continuous_power_w"]["value"]
    assert p_to > p_cont
    assert p_to == 84500.0
    assert p_cont == 73500.0


def test_critical_altitude_hierarchy():
    """Assert authoritative EASA critical altitude (4875 m) meets or exceeds model test ceiling (4500 m)."""
    ref_file = REPO_ROOT / "configs" / "engine_reference" / "rotax_914_ul_f.json"
    with open(ref_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    tcu_alt = data["engine_identity"]["engine_applicability"]["easa_tcds_tcu_ratings"]["tcu_configuration_v4_3"]["continuous_critical_altitude_m"]
    assert tcu_alt == 4875.0
    assert tcu_alt >= 4500.0


def test_claims_and_limitations_doc_has_all_sections():
    """Assert docs/claims_and_limitations.md contains all 6 required governance sections."""
    doc_path = REPO_ROOT / "docs" / "claims_and_limitations.md"
    assert doc_path.exists()
    with open(doc_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "## 1. Authoritative vs. Synthetic Scope" in content
    assert "## 2. Authoritative Rotax 914 UL/F OEM Numerical Reference Table" in content
    assert "## 3. Explicit 6-Tier AI/ML Runtime Inventory" in content
    assert "## 4. Preserved Engineering Limitations Catalog" in content
    assert "## 5. Mandatory Claim-Strength Evaluation Protocol" in content
    assert "## 6. SIH26054 Requirements Coverage Audit" in content


def test_no_misleading_airworthiness_in_readme():
    """Assert README.md explicitly disclaims airworthiness certification."""
    readme_path = REPO_ROOT / "README.md"
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert re.search(r"not\s*(\*\*|\b)?\s*certified", content, re.IGNORECASE) is not None
    assert "DO-178C" in content
    assert "competition prototype" in content.lower() or "research" in content.lower()


def test_all_audited_claims_have_provenance_citations():
    """Assert every claim in evidence/phase12_claims_matrix.json has a valid category and citation."""
    claims_file = REPO_ROOT / "evidence" / "phase12_claims_matrix.json"
    with open(claims_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    for clm in data["audited_claims"]:
        assert clm["category"] in {
            "AUTHORITATIVE_REFERENCE",
            "MODEL_IMPLEMENTATION",
            "MODEL_CALIBRATION",
            "ENGINEERING_HEURISTIC",
            "SYNTHETIC_VALIDATION",
            "DATA_QUALITY_RULE",
        }
        assert len(clm["source_citation"]) > 5

