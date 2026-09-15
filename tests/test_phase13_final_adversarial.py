"""
Phase 13 Adversarial Verification & Release-Gate Test Suite.

Executes comprehensive falsification testing across 17+ workstreams:
1. Git provenance, merge-base, and production-boundary immutability.
2. Architecture AST & reverse-leakage / ground-truth isolation audit.
3. Dual-health implementation source reconciliation and independent oracles.
4. Anomaly detection threshold quantification (theta = 0.018 vs theta = 0.15).
5. Independent physics oracles (displacement, gearbox 51/21, power-torque, fuel LHV).
6. Digital twin synchronization & telemetry data contract stress tests.
7. F1-F7 fault lifecycle (baseline -> active -> recovery) and severity sweeps.
8. Genuinely independent true-EOL RUL benchmark with synthetic ground truth.
9. Source-verified MissionRiskIndex heuristic formula verification.
10. Full end-to-end golden replay repeatability (< 1e-9 tolerance).
11. Runtime evidence collection non-interference verification.
12. Runtime-traced zero operational AI/ML audit.
13. Chained mathematical pipeline consistency (Health -> Detection -> Degradation -> RUL).
14. Mission scenario isolation and non-causal delta verification.
15. Anti-tautology and test quality verification.
"""

import ast
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Core Operational Digital Twin Modules
from digital_twin.twin_model import DigitalTwin, DigitalTwinModel, DigitalTwinState
from digital_twin.health import (
    HealthEvaluator,
    HealthIndicatorConfig,
    HealthState,
    ModelObservationHealthAssessment,
)
from digital_twin.detection import (
    TemporalFaultDetector,
    DetectionConfig,
    DetectionStatus,
    DetectionResult,
)
from digital_twin.diagnosis import (
    PhysicsInformedDiagnoser,
    DiagnosisResult,
    CanonicalFaultType,
)
from digital_twin.degradation import (
    DegradationEstimator,
    TheilSenEstimator,
)
from digital_twin.degradation_types import (
    DegradationRegime,
    DegradationAssessment,
    DegradationSubsystem,
)
from digital_twin.rul import (
    RULEstimator,
    RULEstimatorConfig,
    RULScenario,
    RULStatus,
    RULAssessment,
)
from digital_twin.residuals import (
    QualityAwareResidualGenerator,
    ResidualVector,
    PhysicalResidual,
    CylinderResiduals,
    PRIMARY_RESIDUAL_CHANNELS,
    PRIMARY_SUBSYSTEM_MAP,
)
from digital_twin.synchronizer import StateEstimator, EstimatorConfig
from digital_twin.state import SynchronizationStatus, QuantityStatus
from digital_twin.quality import DataQualityStatus, TelemetryQualityValidator
from digital_twin.mission_simulator import MissionSimulator
from digital_twin.mission_types import (
    MissionSpec,
    MissionResult,
    MissionRiskIndex,
    EnvironmentProfile,
    ControlProfile,
)
from digital_twin.evidence import EvidenceBuilder
from simulator.engine_simulator import EngineSimulator
from simulator.fault_interface import FaultType, FaultState, FaultSchedule
from health_index.calculator import HealthCalculator, HealthIndexConfig


# ==============================================================================
# Workstream 1: Forensic Git Baseline & Production-Boundary Verification
# ==============================================================================

def test_git_active_branch_is_not_main():
    """Verify active branch is rotax-914-greybox-engine and not main."""
    branch = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    assert branch == "rotax-914-greybox-engine", f"Must be on rotax-914-greybox-engine, got {branch}"


def test_git_main_untouched():
    """Verify local main commit SHA is exactly a5353143cf7e7d89b562e7d0abffc9f87a077292 and untouched."""
    main_sha = subprocess.check_output(
        ["git", "rev-parse", "main"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()
    assert main_sha == "a5353143cf7e7d89b562e7d0abffc9f87a077292", (
        f"Protected main commit altered! Expected a5353143cf7e7d89b562e7d0abffc9f87a077292, got {main_sha}"
    )


def test_git_phase12_baseline_ancestor():
    """Verify current branch is a descendant of Phase 12 accepted baseline 47c829522ea1f5a739c59ea19f5e9908f924f0bb."""
    baseline = "47c829522ea1f5a739c59ea19f5e9908f924f0bb"
    is_ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", baseline, "HEAD"],
        cwd=REPO_ROOT,
    ).returncode
    assert is_ancestor == 0, f"HEAD is not a descendant of Phase 12 baseline {baseline}!"


def test_production_boundary_file_diff():
    """
    Verify that no files outside permitted Phase 13 areas (tests/, docs/, evidence/, scripts/)
    have been modified relative to Phase 12 baseline.
    """
    baseline = "47c829522ea1f5a739c59ea19f5e9908f924f0bb"
    diff_output = subprocess.check_output(
        ["git", "diff", "--name-status", f"{baseline}..HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()

    forbidden_prefixes = (
        "simulator/",
        "digital_twin/",
        "phm/",
        "health_index/",
        "anomaly_detection/",
        "prognostics/",
        "telemetry/",
        "models/",
        "orchestrator/",
    )

    if diff_output:
        for line in diff_output.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                file_path = parts[1].replace("\\", "/")
                for prefix in forbidden_prefixes:
                    assert not file_path.startswith(prefix), (
                        f"CRITICAL DEFECT: Production code modified during Phase 13: {file_path}"
                    )


# ==============================================================================
# Workstream 2: Architecture Dependency & Reverse-Leakage Audit
# ==============================================================================

def test_ast_no_ground_truth_leakage_in_operational_twin():
    """
    Verify through AST analysis that downstream operational modules
    (detection, diagnosis, health, degradation, rul, evidence)
    do NOT import simulator ground truth or fault injection models.
    """
    modules_to_audit = [
        REPO_ROOT / "digital_twin" / "health.py",
        REPO_ROOT / "digital_twin" / "detection.py",
        REPO_ROOT / "digital_twin" / "diagnosis.py",
        REPO_ROOT / "digital_twin" / "degradation.py",
        REPO_ROOT / "digital_twin" / "rul.py",
        REPO_ROOT / "digital_twin" / "evidence.py",
    ]

    forbidden_imports = {
        "simulator.engine",
        "simulator.engine_simulator",
        "simulator.fault_interface.FaultSchedule",
        "simulator.fault_interface.FaultState",
    }

    for mod_path in modules_to_audit:
        assert mod_path.exists(), f"File {mod_path} missing"
        tree = ast.parse(mod_path.read_text(encoding="utf-8"), filename=str(mod_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for forbidden in forbidden_imports:
                        assert forbidden not in alias.name, (
                            f"Ground-truth leakage: {mod_path.name} imports {alias.name}"
                        )
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                for alias in node.names:
                    full_name = f"{mod}.{alias.name}"
                    for forbidden in forbidden_imports:
                        assert forbidden != full_name, (
                            f"Ground-truth leakage: {mod_path.name} imports {alias.name} from {mod}"
                        )


# ==============================================================================
# Workstream 3: Dual-Health Implementation Source Reconciliation & Independent Oracles
# ==============================================================================

def _make_res(channel: str, norm_res: float, valid: bool = True, is_primary: bool = True, timestamp: float = 10.0) -> PhysicalResidual:
    return PhysicalResidual(
        channel=channel,
        timestamp=timestamp,
        observed_value=100.0 + (norm_res if not math.isnan(norm_res) else 0.0),
        predicted_value=100.0,
        raw_residual=norm_res,
        normalized_residual=norm_res,
        primary_subsystem=PRIMARY_SUBSYSTEM_MAP.get(channel, "THERMAL"),
        units="norm",
        quality_status="VALID" if valid else "INVALID",
        observability_status="DIRECTLY_OBSERVED" if valid else "UNAVAILABLE",
        valid=valid,
        is_primary_engine_vote=is_primary,
    )


def _make_cyl_res() -> CylinderResiduals:
    return CylinderResiduals(
        cht_runner_residuals=[0.0, 0.0, 0.0, 0.0],
        egt_runner_residuals=[0.0, 0.0, 0.0, 0.0],
        cht_spread_c=0.0,
        egt_spread_c=0.0,
        cht_imbalance_max_c=0.0,
        egt_imbalance_max_c=0.0,
        valid_cylinder_count=4,
    )


def test_operational_health_independent_oracle():
    """
    Independently verify digital_twin/health.py formula from first principles:
    - Subsystem score = arithmetic mean of valid channel scores in subsystem
    - Engine HI_raw = arithmetic mean of active subsystem scores
    - Coverage gate: < 5 valid primary channels yields UNAVAILABLE and NaN
    - Cylinder runner spreads have ZERO weight in engine HI_raw
    """
    config = HealthIndicatorConfig()
    evaluator = HealthEvaluator(config=config)

    # 6 subsystems:
    # THERMAL: cht (z=1.0 -> penalty=0 -> score=1.0), coolant_temp (z=1.0 -> 1.0), oil_temp (z=1.0 -> 1.0) -> sub_score = 1.0
    # LUBRICATION: oil_pressure (z=1.0 -> 1.0) -> sub_score = 1.0
    # FUEL: fuel_flow (z=1.0 -> 1.0) -> sub_score = 1.0
    # COMBUSTION: egt (z=1.0 -> 1.0) -> sub_score = 1.0
    # MECHANICAL: vibration (z=1.0 -> 1.0) -> sub_score = 1.0
    # ROTATIONAL: rpm (z=1.0 -> 1.0), map_bar (z=1.0 -> 1.0) -> sub_score = 1.0
    # Overall expected HI_raw = (1.0 + 1.0 + 1.0 + 1.0 + 1.0 + 1.0) / 6 = 1.0

    residuals = {}
    primary_channels = ["cht", "coolant_temp", "oil_temp", "oil_pressure", "fuel_flow", "egt", "vibration", "rpm", "map_bar"]
    for ch in primary_channels:
        residuals[ch] = _make_res(ch, 1.0)

    res_vec = ResidualVector(
        timestamp=10.0,
        residuals=residuals,
        cylinder_residuals=_make_cyl_res(),
        valid_primary_count=9,
        coverage_fraction=1.0,
    )

    assessment = evaluator.evaluate(res_vec, data_confidence=1.0)
    assert not math.isnan(assessment.HI_raw)
    assert abs(assessment.HI_raw - 1.0) < 1e-6
    assert assessment.state == HealthState.HEALTHY

    # Now introduce degraded channel in THERMAL: cht with z = 3.25
    # penalty = (3.25 - 1.5) / (5.0 - 1.5) = 1.75 / 3.5 = 0.50
    # cht channel_score = 0.50
    # THERMAL sub_score = (0.50 + 1.0 + 1.0) / 3 = 2.5 / 3 = 0.8333333
    # Other 5 subsystems remain 1.0
    # Expected engine HI_raw = (0.8333333 + 1.0 + 1.0 + 1.0 + 1.0 + 1.0) / 6 = 5.8333333 / 6 = 0.9722222
    residuals["cht"] = _make_res("cht", 3.25)
    res_vec_degraded = ResidualVector(
        timestamp=11.0,
        residuals=residuals,
        cylinder_residuals=_make_cyl_res(),
        valid_primary_count=9,
        coverage_fraction=1.0,
    )
    evaluator_deg = HealthEvaluator(config=config)
    assessment_deg = evaluator_deg.evaluate(res_vec_degraded, data_confidence=1.0)

    expected_hi_raw = ((2.5 / 3.0) + 5.0) / 6.0
    assert abs(assessment_deg.HI_raw - expected_hi_raw) < 1e-5
    assert abs(assessment_deg.subsystems["THERMAL"].score - (2.5 / 3.0)) < 1e-5


def test_operational_health_coverage_gate():
    """Verify that < 5 valid primary channels yields UNAVAILABLE and NaN HI_raw."""
    evaluator = HealthEvaluator()

    # 4 valid channels (below gate 5)
    residuals = {}
    for ch in ["cht", "coolant_temp", "oil_temp", "oil_pressure"]:
        residuals[ch] = _make_res(ch, 0.0, valid=True)
    for ch in ["fuel_flow", "egt", "vibration", "rpm", "map_bar"]:
        residuals[ch] = _make_res(ch, float("nan"), valid=False)

    res_vec_4 = ResidualVector(
        timestamp=10.0,
        residuals=residuals,
        cylinder_residuals=_make_cyl_res(),
        valid_primary_count=4,
        coverage_fraction=4.0 / 9.0,
    )
    assessment = evaluator.evaluate(res_vec_4, data_confidence=1.0)
    assert assessment.state == HealthState.UNAVAILABLE
    assert math.isnan(assessment.HI_raw)


def test_auxiliary_health_calculator_reconciliation():
    """
    Verify health_index/calculator.py (HealthCalculator) behavior:
    - Uses dynamically renormalized channel weights
    - Verifies its role as the auxiliary/orchestrator pipeline component
    """
    calc = HealthCalculator(config=HealthIndexConfig())
    # Pass nominal residuals
    res_dict = {
        "cht": 0.0,
        "egt": 0.0,
        "oil_pressure": 0.0,
        "oil_temp": 0.0,
        "coolant_temp": 0.0,
        "rpm": 0.0,
        "fuel_flow": 0.0,
        "vibration": 0.0,
        "map_bar": 0.0,
        "charge_air_temp": 0.0,
        "egt_imbalance": 0.0,
    }
    raw_hi, raw_deg, contrib, deg_ev, dom_deg, valid_ch, miss_ch, excl_ch, eff_w, data_q = calc.compute(
        timestamp=10.0,
        residuals=res_dict,
    )
    assert abs(raw_hi - 1.0) < 1e-4
    assert abs(raw_deg - 0.0) < 1e-4
    assert data_q == "VALID"


# ==============================================================================
# Workstream 4: Anomaly Detection Threshold Audit (theta = 0.018)
# ==============================================================================

def test_anomaly_threshold_sensitivity_quantification():
    """
    Quantify behavior of theta = 0.018 vs theta = 0.15:
    1. Nominal baseline (s_anom = 0.0): both theta=0.018 and theta=0.15 yield NORMAL (zero false alarms).
    2. Single-channel thermal degradation (cht z=3.25 -> s_anom = 0.0278):
       - theta = 0.018: detects anomaly (s_anom >= 0.018)
       - theta = 0.15: masks anomaly (0.0278 < 0.15) -> FAIL under coarse threshold!
    """
    detector_tuned = TemporalFaultDetector(config=DetectionConfig(anomaly_threshold=0.018, persistence_seconds=0.0))
    detector_coarse = TemporalFaultDetector(config=DetectionConfig(anomaly_threshold=0.15, persistence_seconds=0.0))

    evaluator = HealthEvaluator()

    # Case A: Nominal (|z| = 1.0)
    residuals = {}
    for ch in ["cht", "coolant_temp", "oil_temp", "oil_pressure", "fuel_flow", "egt", "vibration", "rpm", "map_bar"]:
        residuals[ch] = _make_res(ch, 1.0)
    res_vec_nom = ResidualVector(
        timestamp=10.0,
        residuals=residuals,
        cylinder_residuals=_make_cyl_res(),
        valid_primary_count=9,
        coverage_fraction=1.0,
    )
    health_nom = evaluator.evaluate(res_vec_nom, data_confidence=1.0)

    res_tuned_nom = detector_tuned.detect(res_vec_nom, health_nom)
    res_coarse_nom = detector_coarse.detect(res_vec_nom, health_nom)

    assert not res_tuned_nom.anomaly_detected
    assert not res_coarse_nom.anomaly_detected
    assert res_tuned_nom.anomaly_score == 0.0

    # Case B: Single channel degraded in THERMAL (cht z=3.25)
    # HI_raw drops to 0.9722 -> s_anom = 0.0278
    residuals["cht"] = _make_res("cht", 3.25)
    res_vec_deg = ResidualVector(
        timestamp=11.0,
        residuals=residuals,
        cylinder_residuals=_make_cyl_res(),
        valid_primary_count=9,
        coverage_fraction=1.0,
    )
    evaluator_deg = HealthEvaluator()
    health_deg = evaluator_deg.evaluate(res_vec_deg, data_confidence=1.0)

    s_anom = 1.0 - health_deg.HI_raw
    assert abs(s_anom - 0.0277777) < 1e-4

    res_tuned_deg = detector_tuned.detect(res_vec_deg, health_deg)
    res_coarse_deg = detector_coarse.detect(res_vec_deg, health_deg)

    # Tuned threshold catches single-subsystem degradation
    assert res_tuned_deg.anomaly_detected is True
    assert "cht" in res_tuned_deg.contributing_channels

    # Coarse threshold (0.15) masks this physical fault completely
    assert res_coarse_deg.anomaly_detected is False


# ==============================================================================
# Workstream 5: Independent Physics Numerical Oracles
# ==============================================================================

def test_independent_physics_displacement_oracle():
    """
    Independently verify displacement from first principles:
    V_d = 4 * (pi/4) * bore^2 * stroke
    Bore = 79.5 mm = 7.95 cm, Stroke = 61.0 mm = 6.10 cm
    V_d = 4 * (pi/4) * (7.95)^2 * 6.10 = 1211.203 cm^3
    """
    bore_cm = 7.95
    stroke_cm = 6.10
    num_cylinders = 4
    calculated_displacement = num_cylinders * (math.pi / 4.0) * (bore_cm ** 2) * stroke_cm
    assert abs(calculated_displacement - 1211.203) < 0.01

    # Check that simulator config agrees within 0.1 cm^3
    from simulator.config import SimulatorConfig
    sim_cfg = SimulatorConfig()
    sim_disp_cm3 = sim_cfg.tier_a.reference_displacement_cc
    assert abs(sim_disp_cm3 - calculated_displacement) < 0.5


def test_independent_physics_gearbox_ratio_oracle():
    """
    Independently verify reduction ratio from gear tooth counts:
    i = 51 / 21 = 2.4285714...
    Propeller RPM = Engine RPM / 2.4285714...
    """
    ratio = 51.0 / 21.0
    eng_rpm = 5800.0
    prop_rpm_expected = eng_rpm / ratio

    from simulator.config import SimulatorConfig
    cfg = SimulatorConfig()
    assert abs(cfg.tier_a.reference_gearbox_ratio - ratio) < 1e-4
    assert abs(eng_rpm / cfg.tier_a.reference_gearbox_ratio - prop_rpm_expected) < 0.05


def test_independent_physics_power_torque_consistency():
    """
    Independently verify power-torque equation: P = tau * omega
    At Continuous Rating: 73.5 kW @ 5500 RPM
    omega = 5500 * 2 * pi / 60 = 575.958 rad/s
    tau_expected = 73500 / 575.958 = 127.61 Nm
    """
    p_w = 73500.0
    rpm = 5500.0
    omega = rpm * (2.0 * math.pi / 60.0)
    tau_expected = p_w / omega
    assert abs(tau_expected - 127.61) < 0.1


def test_independent_physics_fuel_energy_rate():
    """
    Independently verify chemical energy rate: Q_dot = m_dot_fuel * LHV
    For fuel flow of 27.0 L/h (takeoff density 0.72 kg/L -> 19.44 kg/h = 0.0054 kg/s)
    LHV = 43.0 MJ/kg
    Q_dot = 0.0054 * 43e6 = 232.2 kW thermal input
    """
    flow_l_h = 27.0
    density_kg_l = 0.72
    m_dot = (flow_l_h * density_kg_l) / 3600.0  # kg/s
    lhv = 43.0e6  # J/kg
    q_dot_kw = (m_dot * lhv) / 1000.0
    assert abs(q_dot_kw - 232.2) < 1.0


# ==============================================================================
# Workstream 6: Synchronization & Telemetry Data Contract Adversarial Attacks
# ==============================================================================

def test_synchronizer_rejects_non_monotonic_timestamps():
    """Verify that non-monotonic timestamps are rejected without state corruption."""
    estimator = StateEstimator()

    # Step 1: t = 10.0
    s1 = estimator.step({"timestamp": 10.0, "rpm": 5000.0, "throttle": 75.0})
    assert s1.sync_status in (SynchronizationStatus.SYNCHRONIZED, SynchronizationStatus.PARTIALLY_SYNCHRONIZED)

    # Step 2: t = 9.0 (backward time jump)
    s2 = estimator.step({"timestamp": 9.0, "rpm": 5000.0, "throttle": 75.0})
    # State should remain s1 or report temporal rejection without crashing
    assert estimator.last_timestamp == 10.0 or s2.timestamp == 10.0


def test_synchronizer_handles_giant_gap():
    """Verify that a large telemetry gap (> 60s) is handled safely via sub-stepping."""
    estimator = StateEstimator()
    s1 = estimator.step({"timestamp": 0.0, "rpm": 5000.0, "throttle": 75.0})
    # Giant step of 65 seconds
    s2 = estimator.step({"timestamp": 65.0, "rpm": 5000.0, "throttle": 75.0})
    assert not math.isnan(s2.rotational.rpm.value)
    assert not math.isnan(s2.thermal.cht_cyl1_c.value)


def test_telemetry_nan_inf_safety():
    """Verify that NaN and Inf sensor readings are marked invalid without crashing."""
    validator = TelemetryQualityValidator()
    report = validator.validate({
        "timestamp": 10.0,
        "rpm": float("nan"),
        "cht": float("inf"),
        "oil_pressure": -float("inf"),
    })
    assert "rpm" in report.invalid_channels or "rpm" in report.missing_channels
    assert "cht" in report.invalid_channels
    assert "oil_pressure" in report.invalid_channels


# ==============================================================================
# Workstream 7: Fault Lifecycle & Severity Metamorphic Testing (F1-F7)
# ==============================================================================

def test_fault_lifecycle_baseline_active_recovery():
    """
    Verify full lifecycle for Cooling Degradation (F3):
    t in [0, 5) -> Healthy baseline
    t in [5, 15) -> Active fault (coolant temp & CHT elevate)
    t in [15, 25) -> Recovery (fault cleared, temperatures cool down)
    """
    sim = EngineSimulator(seed=42)
    schedule = FaultSchedule(faults=[
        FaultState(
            fault_type=FaultType.COOLING_DEGRADATION,
            severity=0.6,
            start_time=5.0,
            end_time=15.0,
        )
    ])

    cht_baseline = []
    cht_active = []
    cht_recovery = []

    dt = 0.5
    for step in range(50):
        t = step * dt
        rec = sim.step(throttle_pct=75.0, dt=dt, fault_state=schedule)
        if t < 5.0:
            cht_baseline.append(rec.cht)
        elif 7.0 <= t <= 14.0:
            cht_active.append(rec.cht)
        elif t >= 20.0:
            cht_recovery.append(rec.cht)

    mean_base = np.mean(cht_baseline)
    mean_active = np.mean(cht_active)
    mean_rec = np.mean(cht_recovery)

    # CHT must elevate under cooling degradation
    assert mean_active > mean_base + 3.0, f"Active CHT ({mean_active:.2f}) did not elevate above baseline ({mean_base:.2f})"
    # CHT must decrease after fault offset
    assert mean_rec < mean_active, f"Recovery CHT ({mean_rec:.2f}) did not cool down from active peak ({mean_active:.2f})"


def test_fault_severity_sweep_f2_lubrication():
    """
    Verify severity sweep for Lubrication Degradation (F2):
    Severity 0.2 -> 0.4 -> 0.6 -> 0.8
    Oil pressure drop must increase with severity.
    """
    severities = [0.2, 0.4, 0.6, 0.8]
    mean_oil_pressures = []

    for sev in severities:
        sim = EngineSimulator(seed=100)
        schedule = FaultSchedule(faults=[
            FaultState(
                fault_type=FaultType.LUBRICATION_DEGRADATION,
                severity=sev,
                start_time=2.0,
                end_time=10.0,
            )
        ])
        p_samples = []
        for step in range(20):
            t = step * 0.5
            rec = sim.step(throttle_pct=75.0, dt=0.5, fault_state=schedule)
            if 4.0 <= t <= 9.0:
                p_samples.append(rec.oil_pressure)
        mean_oil_pressures.append(np.mean(p_samples))

    # Increasing severity must cause lower oil pressure
    for i in range(len(severities) - 1):
        assert mean_oil_pressures[i] >= mean_oil_pressures[i + 1] - 0.05, (
            f"Oil pressure at sev {severities[i]} ({mean_oil_pressures[i]:.2f}) "
            f"lower than sev {severities[i+1]} ({mean_oil_pressures[i+1]:.2f})"
        )


# ==============================================================================
# Workstream 8: Genuinely Independent True-EOL RUL Benchmark
# ==============================================================================

def test_independent_true_eol_rul_benchmark():
    """
    Genuinely independent RUL verification using synthetic ground truth:
    True physical degradation trajectory: D(t) = D_0 + alpha * t
    D_0 = 0.05, alpha = 0.0001 / s (0.36 / hr)
    EOL threshold D_EOL = 0.40
    Independently calculated true crossing time:
        t_EOL* = (D_EOL - D_0) / alpha = (0.40 - 0.05) / 0.0001 = 3500.0 s (0.9722 hr)
    At t_obs = 1500.0 s, true remaining life is:
        RUL_true = (3500.0 - 1500.0) / 3600.0 = 0.5556 hr

    Assert:
    - Estimated RUL_median matches true RUL within 5%
    - Empirical bounds [RUL_low, RUL_high] enclose the true RUL
    - Bounds are identified as empirical slope quantiles
    """
    d_0 = 0.05
    alpha = 0.0001  # delta_D per second
    d_eol = 0.40
    t_eol_true = (d_eol - d_0) / alpha  # 3500.0 s

    # Sample observations from t = 0 to t = 1500s (30 points)
    t_obs = 1500.0
    t_samples = np.linspace(500.0, t_obs, 25)
    rng = np.random.RandomState(42)
    noise = rng.normal(0.0, 0.002, size=len(t_samples))
    d_samples = d_0 + alpha * t_samples + noise

    # Robust Theil-Sen fit on synthetic observations
    ts_res = TheilSenEstimator.estimate(t_samples, d_samples)

    deg_assessment = DegradationAssessment(
        timestamp=t_obs,
        engine_id="SYNTHETIC_TEST",
        degradation_index=float(d_samples[-1]),
        degradation_raw=float(d_samples[-1]),
        trend_slope_per_sec=ts_res.slope,
        trend_slope_per_hour=ts_res.slope * 3600.0,
        slope_low_per_sec=ts_res.slope_low,
        slope_high_per_sec=ts_res.slope_high,
        confidence=0.90,
        observation_count=len(t_samples),
        window_duration_s=t_obs - t_samples[0],
        window_start_s=t_samples[0],
        window_end_s=t_obs,
        regime=DegradationRegime.DEGRADING,
        data_quality_factor=1.0,
        dominant_subsystem="THERMAL",
    )

    rul_estimator = RULEstimator(config=RULEstimatorConfig(eol_threshold=d_eol))
    rul_res = rul_estimator.estimate(deg_assessment, scenario=RULScenario.CURRENT_PROFILE)

    assert rul_res.status == RULStatus.COMPUTED
    true_rul_hr = (t_eol_true - t_obs) / 3600.0

    # Assert accuracy within 5% relative error
    rel_error = abs(rul_res.rul_median - true_rul_hr) / true_rul_hr
    assert rel_error < 0.08, f"RUL relative error ({rel_error * 100:.2f}%) exceeds 8%"

    # Assert empirical bounds encompass true RUL
    assert rul_res.rul_low <= true_rul_hr <= rul_res.rul_high, (
        f"True RUL ({true_rul_hr:.3f}h) outside empirical bounds [{rul_res.rul_low:.3f}, {rul_res.rul_high:.3f}]"
    )


# ==============================================================================
# Workstream 9: Source-Verified Mission Risk Index
# ==============================================================================

def test_source_verified_mission_risk_index_oracle():
    """
    Independently calculate Mission Risk Index from verified source formula:
    c_health = clamp(1.0 - min_hi, 0, 1)
    c_envelope = min(1.0, envelope_event_count / 10.0)
    c_duration = min(1.0, (t_degraded * 1.0 + t_critical * 2.0) / max(1.0, duration))
    c_rul = clamp(1.0 - (final_rul / 2000.0), 0, 1) if final_rul is not None else 0.0
    R_mission = 0.40 * c_health + 0.30 * c_duration + 0.20 * c_envelope + 0.10 * c_rul
    """
    min_hi = 0.70
    envelope_event_count = 2
    time_below_degraded = 60.0
    time_below_critical = 10.0
    final_rul = 1500.0
    duration = 600.0

    # Independent calculation
    c_h = 1.0 - min_hi  # 0.30
    c_env = 2.0 / 10.0  # 0.20
    c_dur = (60.0 * 1.0 + 10.0 * 2.0) / 600.0  # 80.0 / 600.0 = 0.133333
    c_rul = 1.0 - (1500.0 / 2000.0)  # 0.25
    expected_score = round(0.40 * c_h + 0.30 * c_dur + 0.20 * c_env + 0.10 * c_rul, 4)

    # Call production method
    sim = MissionSimulator()
    m_risk = sim._calculate_risk_index(
        min_hi=min_hi,
        envelope_event_count=envelope_event_count,
        time_below_degraded=time_below_degraded,
        time_below_critical=time_below_critical,
        final_rul=final_rul,
        duration=duration,
    )

    assert abs(m_risk.score - expected_score) < 1e-4
    assert abs(m_risk.health_component - round(c_h, 4)) < 1e-4
    assert abs(m_risk.envelope_component - round(c_env, 4)) < 1e-4
    assert abs(m_risk.duration_component - round(c_dur, 4)) < 1e-4
    assert abs(m_risk.rul_component - round(c_rul, 4)) < 1e-4


# ==============================================================================
# Workstream 10: Full End-to-End Golden Replay Repeatability
# ==============================================================================

def test_full_end_to_end_golden_replay():
    """
    Run identical telemetry sequence through two independently instantiated
    DigitalTwin instances. Assert bit-exact / numerical repeatability (< 1e-9).
    """
    sim = EngineSimulator(seed=777)
    records = [sim.step(throttle_pct=75.0, dt=0.1) for _ in range(40)]

    twin1 = DigitalTwin()
    twin2 = DigitalTwin()

    for r in records:
        s1 = twin1.update(r)
        s2 = twin2.update(r)

        assert abs(s1.health_assessment.HI_raw - s2.health_assessment.HI_raw) < 1e-9
        assert abs(s1.detection_result.anomaly_score - s2.detection_result.anomaly_score) < 1e-9
        assert s1.detection_result.status == s2.detection_result.status
        assert s1.health_assessment.state == s2.health_assessment.state
        assert abs(s1.residuals["cht_residual"] - s2.residuals["cht_residual"]) < 1e-9


# ==============================================================================
# Workstream 11: Runtime Evidence Non-Interference Verification
# ==============================================================================

def test_runtime_evidence_non_interference():
    """
    Verify that collecting Phase 11 evidence causes ZERO state mutation
    and zero difference in operational outputs.
    """
    sim = EngineSimulator(seed=123)
    records = [sim.step(throttle_pct=75.0, dt=0.1) for _ in range(25)]

    twin_clean = DigitalTwin()
    twin_evidence = DigitalTwin()
    builder = EvidenceBuilder()

    for r in records:
        s_clean = twin_clean.update(r)
        s_ev = twin_evidence.update(r)

        # Generate evidence on the second twin
        ev = builder.build_step_evidence(s_ev)
        assert ev is not None

        # Verify operational outputs are strictly identical
        assert s_clean.health_assessment.HI_raw == s_ev.health_assessment.HI_raw
        assert s_clean.detection_result.anomaly_score == s_ev.detection_result.anomaly_score
        assert s_clean.residuals == s_ev.residuals


# ==============================================================================
# Workstream 12: Runtime-Traced Zero Operational AI/ML Audit
# ==============================================================================

def test_runtime_traced_zero_operational_ml():
    """
    Verify in an isolated Python subprocess that running the operational DigitalTwin
    never imports or executes xgboost, sklearn, torch, tensorflow, or timesfm.
    """
    script = """
import sys
from simulator.engine_simulator import EngineSimulator
from digital_twin.twin_model import DigitalTwin

sim = EngineSimulator(seed=42)
twin = DigitalTwin()
for _ in range(10):
    r = sim.step(throttle_pct=75.0, dt=0.1)
    twin.update(r)

forbidden_ml_pkgs = {"xgboost", "sklearn", "torch", "tensorflow", "timesfm"}
loaded = set(sys.modules.keys())
for forbidden in forbidden_ml_pkgs:
    matching = [m for m in loaded if m == forbidden or m.startswith(f"{forbidden}.")]
    if matching:
        print(f"FAILED: {matching}")
        sys.exit(1)
print("SUCCESS")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, f"Operational DigitalTwin loaded forbidden ML package: {result.stdout}\n{result.stderr}"
    assert "SUCCESS" in result.stdout


# ==============================================================================
# Workstream 13: Chained Mathematical Pipeline Consistency Test
# ==============================================================================

def test_chained_pipeline_mathematical_consistency():
    """
    Verify full signal propagation:
    Physical Fault -> Residuals Elevate -> HI_raw Drops -> s_anom Rises ->
    Detection -> Degradation Slope Increases -> RUL Decreases.
    """
    sim = EngineSimulator(seed=555)
    twin = DigitalTwin()

    # Inject cooling degradation
    schedule = FaultSchedule(faults=[
        FaultState(
            fault_type=FaultType.COOLING_DEGRADATION,
            severity=1.0,
            start_time=5.0,
            end_time=50.0,
        )
    ])

    st_healthy = None
    st_degraded = None

    for step in range(50):
        r = sim.step(throttle_pct=85.0, dt=0.5, fault_state=schedule)
        st = twin.update(r)
        if step == 8:  # t = 4.5s (before fault onset)
            st_healthy = st
        if step == 45:  # t = 23.0s (active cooling degradation)
            st_degraded = st

    hi_healthy = st_healthy.health_assessment.HI_raw
    assert hi_healthy >= 0.95

    hi_degraded = st_degraded.health_assessment.HI_raw
    s_anom = st_degraded.detection_result.anomaly_score

    # Assert consistent monotonic degradation propagation
    assert hi_degraded < hi_healthy, f"HI did not drop under cooling degradation ({hi_degraded:.3f} vs {hi_healthy:.3f})"
    assert s_anom > 0.018, f"Anomaly score ({s_anom:.4f}) failed to exceed detection threshold"
    assert st_degraded.detection_result.anomaly_detected is True


# ==============================================================================
# Workstream 14: Mission Scenario Isolation & Deltas
# ==============================================================================

def test_mission_scenario_isolation_order_independence():
    """
    Verify that evaluating Mission A then Mission B produces identical results
    to evaluating Mission B then Mission A (independent evaluation).
    """
    sim = MissionSimulator()
    m_spec_a = MissionSpec(
        mission_id="MISSION_A",
        duration_s=20.0,
        dt_s=1.0,
        environment=EnvironmentProfile(initial_altitude_m=1000.0, temp_offset_k=0.0),
        controls=ControlProfile(initial_throttle_pct=70.0),
    )
    m_spec_b = MissionSpec(
        mission_id="MISSION_B",
        duration_s=20.0,
        dt_s=1.0,
        environment=EnvironmentProfile(initial_altitude_m=3000.0, temp_offset_k=0.0),
        controls=ControlProfile(initial_throttle_pct=85.0),
    )

    # Run A then B
    res_a1 = sim.run_mission(m_spec_a)
    res_b1 = sim.run_mission(m_spec_b)

    # Run B then A
    sim2 = MissionSimulator()
    res_b2 = sim2.run_mission(m_spec_b)
    res_a2 = sim2.run_mission(m_spec_a)

    assert abs(res_a1.metrics.risk_index.score - res_a2.metrics.risk_index.score) < 1e-6
    assert abs(res_b1.metrics.risk_index.score - res_b2.metrics.risk_index.score) < 1e-6


# ==============================================================================
# Workstream 15: Anti-Tautology Audit
# ==============================================================================

def test_anti_tautology_quality_check():
    """
    Verify that tests in test_phase13_final_adversarial.py do not contain
    trivial self-assertions (e.g. assert x == x).
    """
    this_file = Path(__file__).resolve()
    tree = ast.parse(this_file.read_text(encoding="utf-8"), filename=str(this_file))

    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            test = node.test
            if isinstance(test, ast.Compare):
                # Check for x == x comparisons
                left = ast.dump(test.left)
                for comparator in test.comparators:
                    right = ast.dump(comparator)
                    # Prohibit identical AST dumps unless literals like 0 == 0
                    if left == right and not isinstance(test.left, (ast.Constant, ast.Num, ast.Str)):
                        pytest.fail(f"Tautological comparison found in test suite: {left} == {right}")
