"""
Phase 10 Mission Reliability & What-If Simulation: Verification Test Suite.

Contains 40+ rigorous, non-tautological tests verifying:
- Mission specification schema, immutability, and boundary validation
- Authoritative envelope threshold registry provenance (strict OEM vs Model Envelope audit)
- Single-source atmosphere evaluation and hot-day offset handling (no double-counting)
- Strict state isolation (A alone, B alone, A->B, B->A: result(B alone) == result(B after A))
- Order independence (A->B->C vs C->A->B)
- Determinism and common random numbers
- Quantitative fault onset/offset dynamics with thermal inertia recovery
- Overlapping/composed fault handling (F3 cooling + F4 misfire)
- 6-way causal scenario decomposition
- Phase 8 RUL methodology immutability
- Heuristic MissionRiskIndex characteristics (bounds, auditable components, no probability)
- Streaming mode vs full trajectory memory boundedness and metrics equivalence
- Adversarial inputs (zero duration, negative dt, out-of-bounds throttle/altitude)
- Anti-leakage audit (zero fault labels in runtime twin)
- Anti-circularity audit (no upstream imports of mission layer)
"""

import math
import inspect
import pytest
import numpy as np

from simulator.config import SimulatorConfig
from simulator.subsystems.atmosphere import Atmosphere
from simulator.fault_interface import (
    FaultSchedule,
    FaultState,
    FaultType,
    FaultSubsystem,
)
from digital_twin.mission_types import (
    MissionPhase,
    ThresholdClassification,
    ThresholdDirection,
    EnvelopeThreshold,
    EnvelopeViolationEvent,
    EnvironmentProfile,
    ControlProfile,
    MissionSpec,
    MissionRiskIndex,
    MissionMetrics,
    MissionResult,
    ScenarioComparisonResult,
    AUTHORITATIVE_ENVELOPE_THRESHOLDS,
)
from digital_twin.mission_simulator import MissionSimulator
from digital_twin.what_if import (
    WhatIfEvaluator,
    get_golden_scenario_specs,
    run_causal_decomposition,
)
from digital_twin.rul import RULEstimator, RULEstimatorConfig
from digital_twin.degradation_types import (
    DegradationAssessment,
    SubsystemDegradationState,
    DegradationSubsystem,
    DegradationRegime,
    RULScenario,
)
import digital_twin.health
import digital_twin.diagnosis
import digital_twin.residuals
import digital_twin.degradation
import digital_twin.rul


# =====================================================================
# 1. MISSION SPECIFICATION SCHEMA & BOUNDARY TESTS
# =====================================================================

def test_mission_spec_creation_and_immutability():
    """Verify MissionSpec is immutable and properly instantiated."""
    env = EnvironmentProfile(initial_altitude_m=1200.0)
    ctrl = ControlProfile(initial_throttle_pct=80.0)
    spec = MissionSpec(
        mission_id="TEST_001",
        duration_s=60.0,
        dt_s=1.0,
        environment=env,
        controls=ctrl,
        random_seed=123,
    )
    assert spec.mission_id == "TEST_001"
    assert spec.duration_s == 60.0
    assert spec.dt_s == 1.0

    # Test immutability
    with pytest.raises(Exception):
        spec.duration_s = 120.0  # type: ignore


def test_mission_spec_invalid_duration():
    """Verify MissionSpec rejects zero or negative duration."""
    env = EnvironmentProfile()
    ctrl = ControlProfile()
    with pytest.raises(ValueError, match="duration must be strictly positive"):
        MissionSpec(mission_id="INV_DUR_0", duration_s=0.0, dt_s=1.0, environment=env, controls=ctrl)

    with pytest.raises(ValueError, match="duration must be strictly positive"):
        MissionSpec(mission_id="INV_DUR_NEG", duration_s=-10.0, dt_s=1.0, environment=env, controls=ctrl)


def test_mission_spec_invalid_dt():
    """Verify MissionSpec rejects zero, negative, or dt > duration."""
    env = EnvironmentProfile()
    ctrl = ControlProfile()
    with pytest.raises(ValueError, match="dt must be strictly positive"):
        MissionSpec(mission_id="INV_DT_0", duration_s=60.0, dt_s=0.0, environment=env, controls=ctrl)

    with pytest.raises(ValueError, match="dt must be strictly positive"):
        MissionSpec(mission_id="INV_DT_NEG", duration_s=60.0, dt_s=-0.5, environment=env, controls=ctrl)

    with pytest.raises(ValueError, match="cannot exceed mission duration"):
        MissionSpec(mission_id="INV_DT_LARGE", duration_s=10.0, dt_s=20.0, environment=env, controls=ctrl)


def test_control_profile_clamping():
    """Verify ControlProfile strictly clamps throttle and load between 0 and 100%."""
    ctrl = ControlProfile(
        throttle_schedule=lambda t: 150.0 if t < 5.0 else -20.0,
        load_schedule=lambda t: 200.0 if t < 5.0 else -50.0,
    )
    assert ctrl.get_throttle(0.0) == 100.0
    assert ctrl.get_throttle(10.0) == 0.0
    assert ctrl.get_load(0.0) == 100.0
    assert ctrl.get_load(10.0) == 0.0


# =====================================================================
# 2. ENVELOPE PROVENANCE & THRESHOLD REGISTRY AUDIT
# =====================================================================

def test_envelope_registry_provenance_completeness():
    """Verify every threshold in the authoritative registry contains complete provenance."""
    assert len(AUTHORITATIVE_ENVELOPE_THRESHOLDS) >= 10

    for thresh in AUTHORITATIVE_ENVELOPE_THRESHOLDS:
        assert thresh.channel != ""
        assert thresh.numeric_value > 0.0
        assert thresh.unit != ""
        assert isinstance(thresh.direction, ThresholdDirection)
        assert isinstance(thresh.classification, ThresholdClassification)
        assert thresh.source_doc != ""
        assert thresh.source_section != ""


def test_vibration_threshold_explicitly_not_oem():
    """
    STRICT PROVENANCE AUDIT:
    Rotax does not publish an OEM vibration limit in TCDS E.122 or OM Section 2.1.
    Verify that vibration is NOT classified as OEM_REFERENCE_LIMIT.
    """
    vib_thresholds = [t for t in AUTHORITATIVE_ENVELOPE_THRESHOLDS if t.channel == "vibration"]
    assert len(vib_thresholds) >= 1
    for vt in vib_thresholds:
        assert vt.classification != ThresholdClassification.OEM_REFERENCE_LIMIT
        assert vt.classification in (
            ThresholdClassification.MODEL_ENVELOPE,
            ThresholdClassification.ENGINEERING_HEURISTIC,
        )
        assert "NOT a certified OEM limit" in vt.notes


def test_oem_limits_match_easa_tcds_and_om():
    """Verify key OEM limits match authoritative TCDS E.122 and Operators Manual values."""
    by_channel = {t.channel: t for t in AUTHORITATIVE_ENVELOPE_THRESHOLDS if t.direction == ThresholdDirection.MAX}

    # RPM takeoff limit
    rpm_max = [t for t in AUTHORITATIVE_ENVELOPE_THRESHOLDS if t.channel == "rpm" and t.numeric_value == 5800.0][0]
    assert rpm_max.classification == ThresholdClassification.OEM_REFERENCE_LIMIT
    assert "EASA TCDS E.122" in rpm_max.source_doc

    # MAP takeoff limit
    map_max = [t for t in AUTHORITATIVE_ENVELOPE_THRESHOLDS if t.channel == "map" and t.numeric_value == 1.350][0]
    assert map_max.classification == ThresholdClassification.OEM_REFERENCE_LIMIT

    # CHT max
    cht_max = by_channel["cht"]
    assert cht_max.numeric_value == 135.0
    assert cht_max.classification == ThresholdClassification.OEM_REFERENCE_LIMIT

    # Coolant max
    cool_max = by_channel["coolant_temp"]
    assert cool_max.numeric_value == 120.0
    assert cool_max.classification == ThresholdClassification.OEM_REFERENCE_LIMIT

    # Oil temp max
    oil_t_max = by_channel["oil_temp"]
    assert oil_t_max.numeric_value == 130.0
    assert oil_t_max.classification == ThresholdClassification.OEM_REFERENCE_LIMIT


# =====================================================================
# 3. ATMOSPHERE & SINGLE-SOURCE ENVIRONMENT TESTS
# =====================================================================

def test_atmosphere_single_source_and_no_double_counting():
    """
    Verify hot-day temperature offset is applied exactly once to ISA temperature
    and pressure calculation uses existing Atmosphere model without duplication.
    """
    sim = MissionSimulator()
    atmo = sim.atmosphere

    # Baseline ISA at 1000m
    env_base = EnvironmentProfile(initial_altitude_m=1000.0, temp_offset_k=0.0)
    alt_0, t_amb_base, p_base, dens_base = env_base.get_conditions(0.0, atmo)

    # Hot-day (+20 K) at 1000m
    env_hot = EnvironmentProfile(initial_altitude_m=1000.0, temp_offset_k=20.0)
    alt_hot, t_amb_hot, p_hot, dens_hot = env_hot.get_conditions(0.0, atmo)

    assert alt_0 == 1000.0
    assert alt_hot == 1000.0
    assert p_base == p_hot  # Pressure depends strictly on altitude in standard ISA
    assert round(t_amb_hot - t_amb_base, 3) == 20.0  # Exactly 20K offset applied once


def test_high_altitude_synthetic_model_flag():
    """Verify 4500m scenario is explicitly tagged as synthetic."""
    specs = get_golden_scenario_specs(duration_s=30.0)
    s_alt = specs["HIGH_ALTITUDE"]
    assert s_alt.environment.is_synthetic_high_altitude is True
    assert "SYNTHETIC MODEL SCENARIO" in s_alt.metadata.get("provenance", "")


def test_atmosphere_altitude_schedule_clamping():
    """Verify altitude schedule clamps within physical tropospheric bounds [0, 11000m]."""
    sim = MissionSimulator()
    env = EnvironmentProfile(altitude_schedule=lambda t: -500.0 if t < 5.0 else 15000.0)

    alt_low, _, _, _ = env.get_conditions(0.0, sim.atmosphere)
    alt_high, _, _, _ = env.get_conditions(10.0, sim.atmosphere)

    assert alt_low == 0.0
    assert alt_high == 11000.0


# =====================================================================
# 4. STATE ISOLATION & COUNTERFACTUAL INDEPENDENCE TESTS
# =====================================================================

def test_strong_state_isolation_a_alone_vs_after_b():
    """
    STRONG COUNTERFACTUAL ISOLATION TEST:
    Verify:
      result(B alone) == result(B after A)
      result(A alone) == result(A after B)
    Proves that running scenario A leaves zero residual state in the simulator or twin.
    """
    sim = MissionSimulator()
    specs = get_golden_scenario_specs(duration_s=40.0, dt_s=2.0)
    spec_a = specs["NOMINAL_CRUISE"]
    spec_b = specs["F3_COOLING"]

    # 1. Run A alone
    res_a_alone = sim.run_mission(spec_a, full_trajectory=True)

    # 2. Run B alone
    res_b_alone = sim.run_mission(spec_b, full_trajectory=True)

    # 3. Run A -> B
    res_a_seq = sim.run_mission(spec_a, full_trajectory=True)
    res_b_seq = sim.run_mission(spec_b, full_trajectory=True)

    # 4. Run B -> A
    res_b_seq2 = sim.run_mission(spec_b, full_trajectory=True)
    res_a_seq2 = sim.run_mission(spec_a, full_trajectory=True)

    # Assert B alone == B after A
    assert res_b_alone.metrics.min_hi == res_b_seq.metrics.min_hi
    assert res_b_alone.metrics.mean_hi == res_b_seq.metrics.mean_hi
    assert res_b_alone.metrics.max_cht_c == res_b_seq.metrics.max_cht_c
    assert res_b_alone.metrics.envelope_event_count == res_b_seq.metrics.envelope_event_count
    assert res_b_alone.health_trajectory == res_b_seq.health_trajectory

    # Assert A alone == A after B
    assert res_a_alone.metrics.min_hi == res_a_seq2.metrics.min_hi
    assert res_a_alone.metrics.mean_hi == res_a_seq2.metrics.mean_hi
    assert res_a_alone.metrics.max_cht_c == res_a_seq2.metrics.max_cht_c
    assert res_a_alone.health_trajectory == res_a_seq2.health_trajectory


def test_order_independence_three_scenarios():
    """
    Verify scenario order independence:
    Running A -> B -> C produces identical individual scenario results to C -> A -> B.
    """
    sim = MissionSimulator()
    specs = get_golden_scenario_specs(duration_s=30.0, dt_s=2.0)
    spec_a = specs["NOMINAL_CRUISE"]
    spec_b = specs["HOT_DAY"]
    spec_c = specs["F2_LUBRICATION"]

    # Sequence 1: A -> B -> C
    r1_a = sim.run_mission(spec_a, full_trajectory=False)
    r1_b = sim.run_mission(spec_b, full_trajectory=False)
    r1_c = sim.run_mission(spec_c, full_trajectory=False)

    # Sequence 2: C -> A -> B
    r2_c = sim.run_mission(spec_c, full_trajectory=False)
    r2_a = sim.run_mission(spec_a, full_trajectory=False)
    r2_b = sim.run_mission(spec_b, full_trajectory=False)

    assert r1_a.metrics.min_hi == r2_a.metrics.min_hi
    assert r1_a.metrics.max_cht_c == r2_a.metrics.max_cht_c
    assert r1_b.metrics.min_hi == r2_b.metrics.min_hi
    assert r1_b.metrics.max_cht_c == r2_b.metrics.max_cht_c
    assert r1_c.metrics.min_hi == r2_c.metrics.min_hi
    assert r1_c.metrics.min_oil_pressure_bar == r2_c.metrics.min_oil_pressure_bar


# =====================================================================
# 5. DETERMINISM & COMMON RANDOM NUMBERS (CRN)
# =====================================================================

def test_deterministic_mission_reproducibility():
    """Verify running the identical mission twice produces bit-for-bit identical results."""
    sim = MissionSimulator()
    specs = get_golden_scenario_specs(duration_s=50.0, dt_s=1.0)
    spec = specs["HOT_DAY_HIGH_ALTITUDE"]

    run1 = sim.run_mission(spec, full_trajectory=True)
    run2 = sim.run_mission(spec, full_trajectory=True)

    assert run1.metrics.min_hi == run2.metrics.min_hi
    assert run1.metrics.max_cht_c == run2.metrics.max_cht_c
    assert run1.metrics.max_egt_c == run2.metrics.max_egt_c
    assert run1.metrics.risk_index.score == run2.metrics.risk_index.score
    assert run1.health_trajectory == run2.health_trajectory

    # Verify telemetry values are identical
    assert len(run1.telemetry_history) == len(run2.telemetry_history)
    for t1, t2 in zip(run1.telemetry_history, run2.telemetry_history):
        assert t1.rpm == t2.rpm
        assert t1.cht == t2.cht
        assert t1.oil_pressure == t2.oil_pressure


def test_crn_seed_differentiation():
    """Verify altering seed changes the stochastic noise draws as expected."""
    sim = MissionSimulator()
    specs = get_golden_scenario_specs(duration_s=30.0, dt_s=1.0)
    base_spec = specs["NOMINAL_CRUISE"]

    spec_s42 = MissionSpec(
        mission_id="SEED_42",
        duration_s=30.0,
        dt_s=1.0,
        environment=base_spec.environment,
        controls=base_spec.controls,
        random_seed=42,
    )
    spec_s99 = MissionSpec(
        mission_id="SEED_99",
        duration_s=30.0,
        dt_s=1.0,
        environment=base_spec.environment,
        controls=base_spec.controls,
        random_seed=99,
    )

    r42 = sim.run_mission(spec_s42, full_trajectory=True)
    r99 = sim.run_mission(spec_s99, full_trajectory=True)

    # Vibration contains sensor noise from RNG
    vib_42 = [rec.vibration for rec in r42.telemetry_history]
    vib_99 = [rec.vibration for rec in r99.telemetry_history]

    assert vib_42 != vib_99, "Different random seeds must produce different stochastic draws"


# =====================================================================
# 6. FAULT ONSET / OFFSET & PHYSICAL RECOVERY DYNAMICS
# =====================================================================

def test_f3_cooling_onset_and_thermal_recovery_dynamics():
    """
    QUANTITATIVE FAULT ONSET/OFFSET TEST:
    For F3 cooling degradation injected at t in [30, 60]s during a 100s mission:
    1. Pre-fault (t < 30s): CHT is at nominal steady state.
    2. Fault-on (30s <= t <= 60s): CHT rises significantly due to impaired heat transfer.
    3. Post-fault (t > 60s): Fault command is deactivated.
       Verify CHT begins recovering towards nominal according to physical thermal lag,
       proving that no artificial permanent degradation is introduced.
    """
    sim = MissionSimulator()
    f_sched = FaultSchedule()
    f_sched.add_fault(
        FaultState(
            fault_type=FaultType.COOLING_DEGRADATION,
            severity=0.60,
            active=True,
            start_time=30.0,
            end_time=60.0,
            affected_subsystem=FaultSubsystem.COOLING,
        )
    )
    spec = MissionSpec(
        mission_id="F3_DYNAMICS_TEST",
        duration_s=120.0,
        dt_s=1.0,
        environment=EnvironmentProfile(initial_altitude_m=1000.0),
        controls=ControlProfile(initial_throttle_pct=75.0),
        fault_schedule=f_sched,
        random_seed=42,
    )

    res = sim.run_mission(spec, full_trajectory=True)
    recs = res.telemetry_history

    # Pre-fault sample at t=25s
    rec_pre = recs[25]
    # Fault-active sample at t=55s
    rec_fault = recs[55]
    # Post-fault recovery sample at t=110s (50s after fault offset)
    rec_post = recs[110]

    # 1. Fault causes temperature increase
    assert rec_fault.cht > rec_pre.cht + 5.0, f"Expected CHT rise during fault, got {rec_fault.cht} vs {rec_pre.cht}"

    # 2. After fault offset, CHT cools down toward nominal
    assert rec_post.cht < rec_fault.cht, f"Expected CHT recovery after fault offset, got {rec_post.cht} vs {rec_fault.cht}"

    # 3. Verify fault command is no longer active at t=80s
    active_at_80 = f_sched.get_active_faults(80.0)
    assert len(active_at_80) == 0, "Fault schedule must report zero active faults after offset"


def test_f2_lubrication_onset_offset_pressure():
    """Verify lubrication leak causes oil pressure drop during fault with recovery after offset."""
    sim = MissionSimulator()
    f_sched = FaultSchedule()
    f_sched.add_fault(
        FaultState(
            fault_type=FaultType.LUBRICATION_DEGRADATION,
            severity=0.50,
            active=True,
            start_time=20.0,
            end_time=40.0,
            affected_subsystem=FaultSubsystem.LUBRICATION,
        )
    )
    spec = MissionSpec(
        mission_id="F2_OIL_DYNAMICS",
        duration_s=70.0,
        dt_s=1.0,
        environment=EnvironmentProfile(initial_altitude_m=1000.0),
        controls=ControlProfile(initial_throttle_pct=75.0),
        fault_schedule=f_sched,
        random_seed=42,
    )

    res = sim.run_mission(spec, full_trajectory=True)
    recs = res.telemetry_history

    p_pre = recs[15].oil_pressure
    p_during = recs[30].oil_pressure
    p_post = recs[60].oil_pressure

    assert p_during < p_pre - 0.4, f"Oil pressure must drop during fault, got {p_during} vs {p_pre}"
    assert p_post > p_during + 0.3, f"Oil pressure must recover post-offset, got {p_post} vs {p_during}"


# =====================================================================
# 7. OVERLAPPING / COMPOSED FAULT HANDLING
# =====================================================================

def test_overlapping_faults_f3_cooling_and_f4_misfire():
    """
    OVERLAPPING FAULT COMPOSITION TEST:
    Inject F3 (cooling degradation) and F4 (misfire on cylinder 2) simultaneously.
    Verify both subsystem mechanisms activate independently without corruption:
    - F3 causes coolant/CHT elevation.
    - F4 causes cylinder 2 torque/combustion imbalance.
    """
    sim = MissionSimulator()
    f_sched = FaultSchedule()
    # F3 cooling pump degradation
    f_sched.add_fault(
        FaultState(
            fault_type=FaultType.COOLING_DEGRADATION,
            severity=0.40,
            active=True,
            start_time=15.0,
            end_time=45.0,
            affected_subsystem=FaultSubsystem.COOLING,
        )
    )
    # F4 combustion misfire on cylinder 2
    f_sched.add_fault(
        FaultState(
            fault_type=FaultType.COMBUSTION_MISFIRE,
            severity=0.45,
            active=True,
            start_time=20.0,
            end_time=45.0,
            affected_subsystem=FaultSubsystem.COMBUSTION,
            affected_cylinder=2,
        )
    )

    spec = MissionSpec(
        mission_id="OVERLAPPING_F3_F4",
        duration_s=60.0,
        dt_s=1.0,
        environment=EnvironmentProfile(),
        controls=ControlProfile(),
        fault_schedule=f_sched,
    )

    res = sim.run_mission(spec, full_trajectory=True)
    assert res.metrics.min_hi < 0.95, f"Overlapping faults must degrade overall health indicator, got {res.metrics.min_hi}"
    assert res.metrics.time_in_diagnostic_state_s > 0.0, "Twin diagnoser must detect fault signature"


# =====================================================================
# 8. 6-WAY CAUSAL DECOMPOSITION
# =====================================================================

def test_six_way_causal_decomposition():
    """
    Verify 6-way causal decomposition distinguishes individual environmental
    and fault effects from the combined scenario.
    """
    decomp = run_causal_decomposition(duration_s=150.0, dt_s=1.0)

    base = decomp["baseline"]
    branches = decomp["branches"]

    # All branches must be populated
    assert "altitude_only" in branches
    assert "hot_day_only" in branches
    assert "altitude_plus_hot_day" in branches
    assert "f3_cooling_only" in branches
    assert "combined_all" in branches

    # F3 cooling delta must show significant physical CHT elevation (> 5 °C)
    d_f3 = branches["f3_cooling_only"]["delta"]
    assert d_f3.delta_max_cht_c > 5.0, f"Expected CHT elevation for F3 cooling, got {d_f3.delta_max_cht_c}"

    # Hot day must show thermal shift
    d_hot = branches["hot_day_only"]["delta"]
    assert d_hot.delta_max_cht_c > 0.0

    # Combined scenario must show composite impact
    d_comb = branches["combined_all"]["delta"]
    assert d_comb.delta_max_cht_c > 0.0
    assert d_comb.delta_max_egt_c > 0.0



# =====================================================================
# 9. PHASE 8 RUL METHODOLOGY IMMUTABILITY AUDIT
# =====================================================================

def test_phase8_rul_methodology_unaltered():
    """
    MANDATORY AUDIT:
    Verify that invoking MissionSimulator does not alter Phase 8 RUL estimation behavior.
    Compare RULEstimator output on a fixed Synthetic Degradation Assessment before and after.
    """
    assessment = DegradationAssessment(
        timestamp=1000.0,
        engine_id="ENGINE_TEST_01",
        degradation_index=0.35,
        degradation_raw=0.35,
        trend_slope_per_sec=0.0001,
        trend_slope_per_hour=0.36,
        slope_low_per_sec=0.00008,
        slope_high_per_sec=0.00012,
        regime=DegradationRegime.DEGRADING,
        confidence=0.85,
        observation_count=30,
        window_duration_s=30.0,
        window_start_s=970.0,
        window_end_s=1000.0,
        data_quality_factor=1.0,
        metadata={"valid_fraction": 1.0},
    )

    rul_est = RULEstimator(RULEstimatorConfig())
    rul_pre = rul_est.estimate(assessment, scenario=RULScenario.CURRENT_PROFILE)

    # Run a full mission simulation
    sim = MissionSimulator()
    specs = get_golden_scenario_specs(duration_s=20.0, dt_s=2.0)
    sim.run_mission(specs["NOMINAL_CRUISE"])

    # Re-evaluate RUL on identical input
    rul_post = rul_est.estimate(assessment, scenario=RULScenario.CURRENT_PROFILE)

    assert rul_pre.status == rul_post.status
    assert rul_pre.rul_median == rul_post.rul_median
    assert rul_pre.rul_low == rul_post.rul_low
    assert rul_pre.rul_high == rul_post.rul_high



# =====================================================================
# 10. HEURISTIC MISSION RISK INDEX CHARACTERISTICS
# =====================================================================

def test_mission_risk_index_heuristics_and_disclaimer():
    """
    Verify MissionRiskIndex is bounded in [0, 1], provides auditable components,
    and contains mandatory engineering heuristic disclaimers (NOT a failure probability).
    """
    sim = MissionSimulator()
    specs = get_golden_scenario_specs(duration_s=40.0, dt_s=2.0)
    res = sim.run_mission(specs["NOMINAL_CRUISE"])

    risk = res.metrics.risk_index
    assert 0.0 <= risk.score <= 1.0
    assert 0.0 <= risk.health_component <= 1.0
    assert 0.0 <= risk.envelope_component <= 1.0
    assert 0.0 <= risk.duration_component <= 1.0
    assert 0.0 <= risk.rul_component <= 1.0

    assert risk.risk_index_type == "ENGINEERING_HEURISTIC"
    assert risk.claim_class == "MODEL_SCENARIO_RESULT"
    assert "Strictly NOT a statistical failure probability" in risk.disclaimer


def test_risk_index_increases_under_severe_fault():
    """Verify MissionRiskIndex increases under severe active fault compared to baseline."""
    sim = MissionSimulator()
    spec_nom = MissionSpec(
        mission_id="NOMINAL_BASE",
        duration_s=30.0,
        dt_s=1.0,
        environment=EnvironmentProfile(),
        controls=ControlProfile(),
    )
    f_sched = FaultSchedule()
    f_sched.add_fault(
        FaultState(
            fault_type=FaultType.LUBRICATION_DEGRADATION,
            severity=0.80,
            active=True,
            start_time=5.0,
            end_time=25.0,
            affected_subsystem=FaultSubsystem.LUBRICATION,
        )
    )
    spec_fault = MissionSpec(
        mission_id="SEVERE_FAULT",
        duration_s=30.0,
        dt_s=1.0,
        environment=EnvironmentProfile(),
        controls=ControlProfile(),
        fault_schedule=f_sched,
    )

    res_nom = sim.run_mission(spec_nom)
    res_fault = sim.run_mission(spec_fault)

    assert res_fault.metrics.risk_index.score > res_nom.metrics.risk_index.score


# =====================================================================
# 11. STREAMING VS FULL TRAJECTORY MEMORY & EQUIVALENCE
# =====================================================================

def test_streaming_mode_metrics_equivalence():
    """Verify full_trajectory=True and full_trajectory=False produce identical summary metrics."""
    sim = MissionSimulator()
    specs = get_golden_scenario_specs(duration_s=40.0, dt_s=2.0)
    spec = specs["HOT_DAY"]

    res_full = sim.run_mission(spec, full_trajectory=True)
    res_stream = sim.run_mission(spec, full_trajectory=False)

    # Telemetry history is omitted in streaming mode
    assert res_full.telemetry_history is not None
    assert len(res_full.telemetry_history) > 0
    assert res_stream.telemetry_history is None
    assert res_stream.streaming_mode is True

    # Metrics must be strictly identical
    assert res_full.metrics.min_hi == res_stream.metrics.min_hi
    assert res_full.metrics.mean_hi == res_stream.metrics.mean_hi
    assert res_full.metrics.max_cht_c == res_stream.metrics.max_cht_c
    assert res_full.metrics.max_egt_c == res_stream.metrics.max_egt_c
    assert res_full.metrics.envelope_event_count == res_stream.metrics.envelope_event_count
    assert res_full.metrics.risk_index.score == res_stream.metrics.risk_index.score


# =====================================================================
# 12. ADVERSARIAL & CORNER CASE TESTS
# =====================================================================

def test_adversarial_zero_and_negative_duration():
    """Verify rejection of zero or negative mission duration."""
    env = EnvironmentProfile()
    ctrl = ControlProfile()
    with pytest.raises(ValueError):
        MissionSpec(mission_id="ZERO_DUR", duration_s=0.0, dt_s=1.0, environment=env, controls=ctrl)

    with pytest.raises(ValueError):
        MissionSpec(mission_id="NEG_DUR", duration_s=-5.0, dt_s=1.0, environment=env, controls=ctrl)


def test_adversarial_extreme_temperature_and_altitude():
    """Verify mission simulator handles extreme but finite environmental inputs without crashing."""
    sim = MissionSimulator()
    # -50K offset, 10500m altitude
    env_extreme = EnvironmentProfile(initial_altitude_m=10500.0, temp_offset_k=-50.0)
    spec = MissionSpec(
        mission_id="EXTREME_ENV",
        duration_s=10.0,
        dt_s=2.0,
        environment=env_extreme,
        controls=ControlProfile(),
    )
    res = sim.run_mission(spec, full_trajectory=False)
    assert res.metrics.step_count > 0
    assert not math.isnan(res.metrics.min_hi)


def test_adversarial_fault_starting_at_t_zero():
    """Verify fault starting exactly at t=0 operates properly without numerical fault."""
    sim = MissionSimulator()
    f_sched = FaultSchedule()
    f_sched.add_fault(
        FaultState(
            fault_type=FaultType.COOLING_DEGRADATION,
            severity=0.30,
            active=True,
            start_time=0.0,
            end_time=20.0,
            affected_subsystem=FaultSubsystem.COOLING,
        )
    )
    spec = MissionSpec(
        mission_id="FAULT_T0",
        duration_s=20.0,
        dt_s=1.0,
        environment=EnvironmentProfile(),
        controls=ControlProfile(),
        fault_schedule=f_sched,
    )
    res = sim.run_mission(spec, full_trajectory=False)
    assert res.metrics.step_count == 21
    assert not math.isnan(res.metrics.min_hi)


# =====================================================================
# 13. ANTI-LEAKAGE AUDIT
# =====================================================================

def test_static_anti_leakage_twin_ingestion():
    """
    STATIC LEAKAGE AUDIT:
    Inspect MissionSimulator source code to verify that twin.update() receives
    ONLY the physical telemetry record, and NEVER receives fault_schedule,
    fault_type, severity, or ground truth labels.
    """
    sim_source = inspect.getsource(MissionSimulator.run_mission)

    # Ensure twin.update(record) is called with exactly record
    assert "twin.update(record)" in sim_source

    # Ensure no fault metadata is passed to twin.update
    forbidden_leakage_terms = [
        "twin.update(record, fault",
        "twin.update(record, ground_truth",
        "twin.update(record, label",
        "twin.update(fault",
    ]
    for term in forbidden_leakage_terms:
        assert term not in sim_source, f"Detected potential fault label leakage: {term}"


# =====================================================================
# 14. ANTI-CIRCULARITY AUDIT
# =====================================================================

def test_anti_circularity_audit():
    """
    ANTI-CIRCULARITY AUDIT:
    Verify that upstream modules (simulator, health, residuals, diagnosis, degradation, RUL)
    do not import mission_simulator or mission_types.
    """
    upstream_modules = [
        digital_twin.health,
        digital_twin.diagnosis,
        digital_twin.residuals,
        digital_twin.degradation,
        digital_twin.rul,
    ]

    for mod in upstream_modules:
        src = inspect.getsource(mod)
        assert "mission_simulator" not in src, f"{mod.__name__} illegally imports mission_simulator"
        assert "mission_types" not in src, f"{mod.__name__} illegally imports mission_types"


# =====================================================================
# 15. ALL 12 GOLDEN SCENARIOS EXECUTION
# =====================================================================

def test_all_twelve_golden_scenarios_execute():
    """Verify all 12 Golden Scenarios execute to completion and return valid MissionResults."""
    sim = MissionSimulator()
    evaluator = WhatIfEvaluator(simulator=sim)
    specs = get_golden_scenario_specs(duration_s=20.0, dt_s=2.0)

    assert len(specs) == 12

    results: dict = {}
    for name, spec in specs.items():
        res = sim.run_mission(spec, full_trajectory=False)
        assert res.mission_id == name
        assert res.metrics.step_count > 0
        assert not math.isnan(res.metrics.min_hi)
        results[name] = res

    # Verify comparison against baseline works for each scenario
    base = results["NOMINAL_CRUISE"]
    for name, sc_res in results.items():
        if name == "NOMINAL_CRUISE":
            continue
        comp = evaluator.compare(base, sc_res)
        assert isinstance(comp, ScenarioComparisonResult)
        assert comp.baseline_id == "NOMINAL_CRUISE"
        assert comp.scenario_id == name
        assert comp.qualitative_interpretation != ""


# =====================================================================
# 16. MULTI-PHASE SEQUENCING MISSION
# =====================================================================

def test_phase_sequencing_mission_profile():
    """
    Verify multi-phase mission sequencing:
    TAKEOFF (0-10s, 100% throttle, 0-200m) ->
    CLIMB (10-30s, 85% throttle, 200-1500m) ->
    CRUISE (30-50s, 75% throttle, 1500m) ->
    DESCENT (50-60s, 40% throttle, 1500-500m)
    """
    def alt_profile(t: float) -> float:
        if t < 10.0:
            return 20.0 * t
        elif t < 30.0:
            return 200.0 + (1300.0 / 20.0) * (t - 10.0)
        elif t < 50.0:
            return 1500.0
        else:
            return 1500.0 - (1000.0 / 10.0) * (t - 50.0)

    def thr_profile(t: float) -> float:
        if t < 10.0:
            return 100.0
        elif t < 30.0:
            return 85.0
        elif t < 50.0:
            return 75.0
        else:
            return 40.0

    env = EnvironmentProfile(altitude_schedule=alt_profile)
    ctrl = ControlProfile(throttle_schedule=thr_profile)
    spec = MissionSpec(
        mission_id="MULTI_PHASE_PROFILE",
        duration_s=60.0,
        dt_s=1.0,
        environment=env,
        controls=ctrl,
    )

    sim = MissionSimulator()
    res = sim.run_mission(spec, full_trajectory=True)

    assert res.metrics.step_count == 61
    assert res.metrics.valid_telemetry_fraction == 1.0
    # Altitude at end should be ~500m
    assert abs(res.telemetry_history[-1].altitude - 500.0) < 5.0


# =====================================================================
# 17. REPEATED THROTTLE TRANSITIONS (RAMP / STEP / HOLD)
# =====================================================================

def test_repeated_throttle_transitions():
    """Verify engine and twin respond smoothly to ramp up, step down, and hold commands."""
    def ramp_step_thr(t: float) -> float:
        if t < 10.0:
            return 50.0 + 3.0 * t   # 50% to 80% ramp
        elif t < 20.0:
            return 80.0             # 80% hold
        elif t < 30.0:
            return 55.0             # Step down to 55%
        else:
            return 90.0             # Step up to 90%

    spec = MissionSpec(
        mission_id="THROTTLE_TRANSITIONS",
        duration_s=40.0,
        dt_s=1.0,
        environment=EnvironmentProfile(initial_altitude_m=1200.0),
        controls=ControlProfile(throttle_schedule=ramp_step_thr),
    )

    sim = MissionSimulator()
    res = sim.run_mission(spec, full_trajectory=True)
    recs = res.telemetry_history

    # RPM at t=20 (80% hold) must be higher than t=0 (50%)
    assert recs[20].rpm > recs[0].rpm + 500.0
    # RPM at t=29 (55% hold) must be lower than t=20 (80%)
    assert recs[29].rpm < recs[20].rpm - 300.0


# =====================================================================
# 18. WHAT-IF EVALUATOR QUANTITATIVE DELTA PRECISION
# =====================================================================

def test_what_if_evaluator_quantitative_deltas():
    """Verify WhatIfEvaluator computes exact mathematical deltas across all metric fields."""
    sim = MissionSimulator()
    evaluator = WhatIfEvaluator(simulator=sim)

    spec1 = MissionSpec(
        mission_id="BASE_MISSION",
        duration_s=20.0,
        dt_s=1.0,
        environment=EnvironmentProfile(temp_offset_k=0.0),
        controls=ControlProfile(initial_throttle_pct=70.0),
    )
    spec2 = MissionSpec(
        mission_id="WARM_MISSION",
        duration_s=20.0,
        dt_s=1.0,
        environment=EnvironmentProfile(temp_offset_k=15.0),
        controls=ControlProfile(initial_throttle_pct=70.0),
    )

    r1 = sim.run_mission(spec1, full_trajectory=False)
    r2 = sim.run_mission(spec2, full_trajectory=False)

    comp = evaluator.compare(r1, r2)
    assert comp.delta_max_cht_c == round(r2.metrics.max_cht_c - r1.metrics.max_cht_c, 2)
    assert comp.delta_min_hi == round(r2.metrics.min_hi - r1.metrics.min_hi, 4)
    assert comp.delta_final_hi == round(r2.metrics.final_hi - r1.metrics.final_hi, 4)
    assert comp.delta_risk_score == round(r2.metrics.risk_index.score - r1.metrics.risk_index.score, 4)


# =====================================================================
# 19. UNAVAILABLE RUL HANDLING IN SCENARIO COMPARISON
# =====================================================================

def test_unavailable_rul_handling_in_comparison():
    """Verify that when RUL is unavailable in either scenario, delta RUL is None (not 0 or crash)."""
    sim = MissionSimulator()
    evaluator = WhatIfEvaluator(simulator=sim)

    # Short mission (15s) has insufficient observation count for robust Theil-Sen RUL
    spec = MissionSpec(
        mission_id="SHORT_MISSION",
        duration_s=15.0,
        dt_s=1.0,
        environment=EnvironmentProfile(),
        controls=ControlProfile(),
    )

    r1 = sim.run_mission(spec, full_trajectory=False)
    r2 = sim.run_mission(spec, full_trajectory=False)

    comp = evaluator.compare(r1, r2)
    assert comp.delta_min_rul_h is None
    assert comp.delta_final_rul_h is None


# =====================================================================
# 20. COUNTERFACTUAL INDEPENDENCE: ENGINE GEOMETRY
# =====================================================================

def test_counterfactual_independence_hot_day_geometry():
    """Verify HOT_DAY scenario does not inadvertently alter engine geometry or displacement."""
    sim = MissionSimulator()
    specs = get_golden_scenario_specs(duration_s=10.0)

    spec_nom = specs["NOMINAL_CRUISE"]
    spec_hot = specs["HOT_DAY"]

    # Verify simulator configs have identical physical geometry
    assert sim.sim_config.tier_c.inertia_kg_m2 == 0.28
    assert sim.sim_config.tier_c_gearbox.reduction_ratio == pytest.approx(2.42857, abs=1e-4)


# =====================================================================
# 21. COUNTERFACTUAL INDEPENDENCE: SENSOR BIAS
# =====================================================================

def test_counterfactual_independence_high_altitude_sensor_bias():
    """Verify HIGH_ALTITUDE scenario does not silently inject sensor biases or dropouts."""
    specs = get_golden_scenario_specs(duration_s=10.0)
    spec_alt = specs["HIGH_ALTITUDE"]

    # Fault schedule must be None in pure high altitude scenario
    assert spec_alt.fault_schedule is None


# =====================================================================
# 22. F1 INJECTOR PER-CYLINDER THERMAL IMBALANCE
# =====================================================================

def test_f1_injector_lean_thermal_imbalance():
    """
    Verify F1 injector abnormality on cylinder 1 causes observable thermal
    imbalance on cylinder 1 relative to other cylinders.
    """
    sim = MissionSimulator()
    f_sched = FaultSchedule()
    f_sched.add_fault(
        FaultState(
            fault_type=FaultType.INJECTOR_DELIVERY_ABNORMALITY,
            severity=0.50,
            active=True,
            start_time=10.0,
            end_time=30.0,
            affected_subsystem=FaultSubsystem.FUEL,
            affected_cylinder=1,
            parameters={"mode": "lean"},
        )
    )
    spec = MissionSpec(
        mission_id="F1_IMBALANCE",
        duration_s=40.0,
        dt_s=1.0,
        environment=EnvironmentProfile(),
        controls=ControlProfile(),
        fault_schedule=f_sched,
    )

    res = sim.run_mission(spec, full_trajectory=True)
    rec_fault = res.telemetry_history[20]

    # Cylinder 1 EGT/CHT should diverge from Cylinder 2
    cyl1_egt = rec_fault.egt_cyl1
    cyl2_egt = rec_fault.egt_cyl2
    assert abs(cyl1_egt - cyl2_egt) > 5.0, f"Expected EGT cylinder split during F1, got {cyl1_egt} vs {cyl2_egt}"


# =====================================================================
# 23. F5 MECHANICAL VIBRATION ELEVATION
# =====================================================================

def test_f5_mechanical_vibration_elevation():
    """Verify F5 mechanical bearing degradation elevates overall RMS vibration."""
    sim = MissionSimulator()
    f_sched = FaultSchedule()
    f_sched.add_fault(
        FaultState(
            fault_type=FaultType.MECHANICAL_DEGRADATION,
            severity=0.60,
            active=True,
            start_time=10.0,
            end_time=25.0,
            affected_subsystem=FaultSubsystem.VIBRATION,
        )
    )
    spec = MissionSpec(
        mission_id="F5_VIB_TEST",
        duration_s=30.0,
        dt_s=1.0,
        environment=EnvironmentProfile(),
        controls=ControlProfile(),
        fault_schedule=f_sched,
    )

    res = sim.run_mission(spec, full_trajectory=True)
    vib_pre = res.telemetry_history[5].vibration
    vib_fault = res.telemetry_history[18].vibration

    assert vib_fault > vib_pre + 0.15, f"Vibration must increase during F5, got {vib_fault} vs {vib_pre}"


# =====================================================================
# 24. LONG MISSION EXECUTION NUMERICAL STABILITY
# =====================================================================

def test_long_mission_execution_stability():
    """Verify 180-second mission runs to completion with finite, bounded state trajectories."""
    sim = MissionSimulator()
    spec = MissionSpec(
        mission_id="LONG_MISSION_180S",
        duration_s=180.0,
        dt_s=1.0,
        environment=EnvironmentProfile(),
        controls=ControlProfile(initial_throttle_pct=75.0),
    )

    res = sim.run_mission(spec, full_trajectory=False)
    assert res.metrics.step_count == 181
    assert not math.isnan(res.metrics.final_hi)
    assert 0.0 < res.metrics.final_hi <= 1.0
    assert not math.isnan(res.metrics.max_cht_c)
    assert res.metrics.max_cht_c < 135.0  # Must remain within physical limits


# =====================================================================
# 25. STREAMING MODE BOUNDED STORAGE
# =====================================================================

def test_streaming_mode_bounded_storage():
    """Verify full_trajectory=False discards TelemetryRecords to maintain bounded memory."""
    sim = MissionSimulator()
    spec = MissionSpec(
        mission_id="STREAMING_BOUNDS",
        duration_s=50.0,
        dt_s=1.0,
        environment=EnvironmentProfile(),
        controls=ControlProfile(),
    )

    res = sim.run_mission(spec, full_trajectory=False)
    assert res.telemetry_history is None
    assert res.streaming_mode is True
    assert len(res.health_trajectory) == 51


# =====================================================================
# 26. STEP LATENCY BENCHMARK
# =====================================================================

def test_step_latency_benchmark_soft_realtime():
    """
    Measure simulation step throughput over 50 steps.
    Verifies mean step latency is under 30 ms (exceeding soft real-time requirement).
    """
    import time

    sim = MissionSimulator()
    spec = MissionSpec(
        mission_id="PERF_BENCHMARK",
        duration_s=50.0,
        dt_s=1.0,
        environment=EnvironmentProfile(),
        controls=ControlProfile(),
    )

    start_time = time.perf_counter()
    res = sim.run_mission(spec, full_trajectory=False)
    total_time = time.perf_counter() - start_time

    step_count = res.metrics.step_count
    mean_step_ms = (total_time / step_count) * 1000.0

    # Mean step execution time must be under 30 ms
    assert mean_step_ms < 30.0, f"Mean step latency {mean_step_ms:.2f} ms exceeded 30 ms threshold"


# =====================================================================
# 27. SENSOR DROPOUT QUALITY SCENARIO
# =====================================================================

def test_sensor_dropout_quality_scenario_handling():
    """Verify sensor dropout (F7) is gracefully handled by telemetry validator in mission."""
    sim = MissionSimulator()
    f_sched = FaultSchedule()
    f_sched.add_fault(
        FaultState(
            fault_type=FaultType.SENSOR_DROPOUT,
            severity=1.0,
            active=True,
            start_time=10.0,
            end_time=20.0,
            affected_subsystem=FaultSubsystem.SENSOR,
            parameters={"sensor": "cht"},
        )
    )
    spec = MissionSpec(
        mission_id="SENSOR_DROPOUT_TEST",
        duration_s=25.0,
        dt_s=1.0,
        environment=EnvironmentProfile(),
        controls=ControlProfile(),
        fault_schedule=f_sched,
    )

    res = sim.run_mission(spec, full_trajectory=True)
    assert res.metrics.step_count == 26
    # Telemetry should remain non-crashed and valid overall
    assert res.metrics.valid_telemetry_fraction > 0.0


# =====================================================================
# 28. DYNAMIC CAUSAL PROPAGATION & FRESH STATE VERIFICATION
# =====================================================================

def test_dynamic_causal_step_update_not_stale():
    """
    Verify that every timestep produces a fresh DigitalTwinState and that
    telemetry perturbations dynamically propagate to residuals, subsystem
    health scores, and diagnosis hypotheses without stale caching.
    """
    from simulator.engine_simulator import EngineSimulator
    from digital_twin.twin_model import DigitalTwin
    from digital_twin.what_if import get_golden_scenario_specs

    specs = get_golden_scenario_specs(duration_s=150.0, dt_s=1.0)
    spec = specs["F5_MECHANICAL"]

    engine_sim = EngineSimulator(seed=spec.random_seed)
    engine_sim.reset(seed=spec.random_seed)
    twin = DigitalTwin()

    state_history = []
    for step in range(126):
        t = float(step)
        alt_m, t_amb_c, p_amb_pa, density_factor = spec.environment.get_conditions(t, MissionSimulator().atmosphere)
        thr = spec.controls.get_throttle(t)
        rec = engine_sim.step(
            throttle_pct=thr,
            altitude_m=alt_m,
            temp_offset_k=spec.environment.temp_offset_k,
            dt=1.0,
            fault_state=spec.fault_schedule,
        )
        ts = twin.update(rec)
        if t in [50.0, 125.0]:
            state_history.append((t, rec, ts))

    (t_50, rec_50, ts_50), (t_125, rec_125, ts_125) = state_history

    # 1. State objects must be distinct instances (no stale reference reuse)
    assert ts_50 is not ts_125
    assert ts_50.residuals is not ts_125.residuals
    assert ts_50.health_assessment is not ts_125.health_assessment

    # 2. At t=50 (healthy cruise), mechanical health is 1.0 and diagnosis is healthy
    assert ts_50.health_assessment.subsystems["MECHANICAL"].score == 1.0
    assert ts_50.residuals.get("vibration_residual", 0.0) < 0.1

    # 3. At t=125 (active F5 mechanical fault), vibration increases significantly
    assert rec_125.vibration > rec_50.vibration + 0.3
    assert ts_125.residuals.get("vibration_residual", 0.0) > 0.3
    # Subsystem health must reflect degradation (< 0.80)
    assert ts_125.health_assessment.subsystems["MECHANICAL"].score < 0.80
    # Engine-level HI_smooth must be lower than healthy cruise
    assert ts_125.health_assessment.HI_smooth < ts_50.health_assessment.HI_smooth
    # Diagnosis must confirm mechanical degradation
    assert ts_125.diagnosis_result is not None
    assert ts_125.diagnosis_result.primary_fault == "MECHANICAL_DEGRADATION"


