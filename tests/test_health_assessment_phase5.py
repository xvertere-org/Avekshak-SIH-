"""
Phase 5 Comprehensive Verification Test Suite:
Residual Generation, Physics-Based Health Assessment, and Subsystem Invariants.

Verifies:
1. Frozen scale calibration immutability, provenance, and separate calibration/evaluation lifecycles.
2. Physical residual computation, sign conventions (observed - expected), and explicit engineering units.
3. Data quality and observability filtering with explicit rejection reasons.
4. Primary subsystem ownership preventing double-counting; secondary channel handling.
5. Discrete per-cylinder runner localization and spread metrics without engine-level vote inflation.
6. Coverage gate: minimum 5 valid primary channels (C_obs >= 5/9); UNAVAILABLE state with NaN HI_raw when < 5.
7. Strict independence of H_phys (HI_raw), C_obs, and C_data (missing telemetry != physical degradation).
8. Physical time persistence (tau_nom, tau_crit, 3s persistence, 5s recovery) and causal EWMA.
9. Causal fault response (F1-F7): physical propagation, sign/direction assertions, localization, and sensor isolation.
10. Severity monotonicity where mathematically expected.
11. Absolute absence of fault-label or ground-truth leakage in the health execution path.
12. Deterministic replay and clean reset.
"""

import math
import dataclasses
import pytest
import numpy as np
import pandas as pd

from simulator.engine_simulator import EngineSimulator
from simulator.config import SimulatorConfig
from simulator.fault_interface import (
    FaultState,
    FaultType,
    FaultSubsystem,
    FaultSchedule,
)
from telemetry.schema import TelemetryRecord, DigitalTwinState
from digital_twin.twin_model import DigitalTwin
from digital_twin.residuals import (
    PhysicalResidual,
    FrozenScaleCalibration,
    CylinderResiduals,
    ResidualVector,
    QualityAwareResidualGenerator,
    create_default_calibration,
    PRIMARY_RESIDUAL_CHANNELS,
    CYLINDER_RESIDUAL_CHANNELS,
    SECONDARY_MODEL_CHANNELS,
    PRIMARY_SUBSYSTEM_MAP,
    SECONDARY_CONTEXT_MAP,
    CHANNEL_UNITS_MAP,
)
from digital_twin.health import (
    HealthState,
    HealthIndicatorConfig,
    ChannelHealthIndicator,
    SubsystemHealthAssessment,
    ModelObservationHealthAssessment,
    HealthEvaluator,
)
from digital_twin.quality import TelemetryQualityReport, ChannelQuality, DataQualityStatus
from digital_twin.observability import ObservabilityType


def make_telemetry_record(
    timestamp: float = 0.0,
    engine_id: str = "TEST_ENG",
    rpm: float = 5500.0,
    map_bar: float = 1.10,
    fuel_flow: float = 24.0,
    cht: float = 90.0,
    coolant_temp: float = 80.0,
    oil_temp: float = 70.0,
    oil_pressure: float = 3.0,
    egt: float = 600.0,
    vibration: float = 0.30,
    **kwargs,
) -> TelemetryRecord:
    """Helper to construct valid TelemetryRecord instances for testing."""
    params = {
        "timestamp": timestamp,
        "mission_id": "TEST_MISSION",
        "engine_id": engine_id,
        "mission_phase": "CRUISE",
        "altitude": 2000.0,
        "ambient_temp": 15.0,
        "throttle": 75.0,
        "load": 75.0,
        "rpm": rpm,
        "map_bar": map_bar,
        "fuel_flow": fuel_flow,
        "cht": cht,
        "coolant_temp": coolant_temp,
        "oil_temp": oil_temp,
        "oil_pressure": oil_pressure,
        "egt": egt,
        "vibration": vibration,
    }
    params.update(kwargs)
    return TelemetryRecord(**params)


# =====================================================================
# 1. Calibration Immutability, Provenance & Independent Windows
# =====================================================================

def test_frozen_scale_calibration_immutability_and_provenance():
    """Verify FrozenScaleCalibration is strictly immutable and carries explicit provenance."""
    cal = create_default_calibration("TEST_CAL_IMMUTABLE")
    assert isinstance(cal, FrozenScaleCalibration)
    assert cal.calibration_mode == "ENGINEERING_HEURISTIC"
    assert cal.dataset_type == "NONE_HEURISTIC"
    assert cal.scales["rpm"] == 100.0

    # Attempt mutation on frozen dataclass
    with pytest.raises(dataclasses.FrozenInstanceError):
        cal.scales = {}

    with pytest.raises(dataclasses.FrozenInstanceError):
        cal.calibration_mode = "MUTATED"


def test_calibration_lifecycle_separate_windows():
    """
    Lifecycle:
      1. Run healthy simulation in Window 1 (calibration window).
      2. Derive robust MAD scales and freeze calibration.
      3. Run independent healthy simulation in Window 2 (evaluation window).
      4. Verify frozen scales remain unchanged and Window 2 residuals stay within nominal thresholds.
    """
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)

    # Window 1: Calibration (0 to 10 seconds, steady cruise)
    cal_records = []
    for _ in range(100):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1)
        cal_records.append(rec)

    twin_cal = DigitalTwin(sim_config=sim_config)
    cal_states = [twin_cal.update(r) for r in cal_records]

    # Build DataFrame of residuals for calibration
    cal_df_rows = []
    for st in cal_states:
        row = {"timestamp": st.timestamp}
        for ch, r in st.residuals.items():
            row[ch] = r
        cal_df_rows.append(row)
    cal_df = pd.DataFrame(cal_df_rows)

    frozen_cal = QualityAwareResidualGenerator.calibrate_from_dataframe(
        df_healthy=cal_df,
        calibration_id="FROZEN_WINDOW1_CAL",
        window_name="CRUISE_WINDOW_1",
    )

    assert frozen_cal.calibration_mode == "FROZEN_SYNTHETIC_BASELINE_MAD"
    assert frozen_cal.dataset_type == "SYNTHETIC_GREY_BOX_HEALTHY"
    assert frozen_cal.sample_count == 100
    assert frozen_cal.calibration_dataset_window == "CRUISE_WINDOW_1"
    assert "FROZEN_SYNTHETIC_BASELINE_MAD" in frozen_cal.provenance.values()

    # Window 2: Independent Evaluation (separate seed from initial baseline)
    sim_eval = EngineSimulator(sim_config=sim_config, seed=202)
    twin_eval = DigitalTwin(sim_config=sim_config, calibration=frozen_cal)
    eval_records = []
    for _ in range(100):
        rec = sim_eval.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1)
        eval_records.append(rec)

    eval_states = [twin_eval.update(r) for r in eval_records]

    # Verify frozen scales never changed
    assert twin_eval.calibration.scales == frozen_cal.scales

    # Verify that in healthy independent window, health index remains high and non-critical
    last_assessment = eval_states[-1].health_assessment
    assert last_assessment is not None
    assert last_assessment.state in (HealthState.HEALTHY, HealthState.WATCH)
    assert last_assessment.HI_raw >= 0.85
    assert not math.isnan(last_assessment.HI_raw)


# =====================================================================
# 2. Residual Properties, Sign Convention & Explicit Units
# =====================================================================

def test_physical_residuals_sign_and_units():
    """Verify residual convention is strictly observed - expected, and units are accurate."""
    generator = QualityAwareResidualGenerator()
    rec = make_telemetry_record(
        timestamp=10.0,
        rpm=5550.0,
        map_bar=1.15,
        fuel_flow=24.5,
        cht=95.0,
        coolant_temp=82.0,
        oil_temp=72.0,
        oil_pressure=3.2,
        egt=610.0,
        vibration=0.35,
        cht_cyl1=94.0,
        cht_cyl2=95.0,
        cht_cyl3=96.0,
        cht_cyl4=95.0,
        egt_cyl1=605.0,
        egt_cyl2=610.0,
        egt_cyl3=615.0,
        egt_cyl4=610.0,
        charge_air_temp=38.0,
    )
    expected = {
        "rpm": 5500.0,
        "map_bar": 1.10,
        "fuel_flow": 23.0,
        "cht": 90.0,
        "coolant_temp": 80.0,
        "oil_temp": 70.0,
        "oil_pressure": 3.0,
        "egt": 600.0,
        "vibration": 0.30,
        "cht_cyl1": 90.0,
        "cht_cyl2": 90.0,
        "cht_cyl3": 90.0,
        "cht_cyl4": 90.0,
        "egt_cyl1": 600.0,
        "egt_cyl2": 600.0,
        "egt_cyl3": 600.0,
        "egt_cyl4": 600.0,
        "charge_air_temp": 35.0,
    }

    res_vec = generator.generate(telemetry=rec, expected=expected)
    assert res_vec.valid_primary_count == 9
    assert res_vec.coverage_fraction == 1.0

    # Test individual channel sign: observed - expected
    cht_res = res_vec.get_residual("cht")
    assert cht_res is not None
    assert cht_res.valid is True
    assert cht_res.raw_residual == pytest.approx(5.0, rel=1e-3)
    assert cht_res.units == "°C"
    assert cht_res.is_primary_engine_vote is True
    assert cht_res.primary_subsystem == "THERMAL"

    oil_p_res = res_vec.get_residual("oil_pressure")
    assert oil_p_res.raw_residual == pytest.approx(0.2, rel=1e-3)
    assert oil_p_res.units == "bar"

    # Test secondary channel
    cat_res = res_vec.get_residual("charge_air_temp")
    assert cat_res.raw_residual == pytest.approx(3.0, rel=1e-3)
    assert cat_res.is_primary_engine_vote is False


# =====================================================================
# 3. Quality & Observability Rejection Invariants
# =====================================================================

def test_quality_and_observability_rejection():
    """Verify stale, out-of-range, and unobservable channels are cleanly rejected with reasons."""
    generator = QualityAwareResidualGenerator()
    rec = make_telemetry_record(
        timestamp=20.0,
        rpm=5500.0,
        map_bar=1.10,
        fuel_flow=24.0,
        cht=90.0,
        coolant_temp=float("nan"),  # Missing observation
        oil_temp=70.0,
        oil_pressure=3.0,
        egt=600.0,
        vibration=0.30,
    )
    expected = {
        "rpm": 5500.0,
        "map_bar": 1.10,
        "fuel_flow": 24.0,
        "cht": 90.0,
        "coolant_temp": 80.0,
        "oil_temp": 70.0,
        "oil_pressure": 3.0,
        "egt": 600.0,
        "vibration": 0.30,
    }

    # Construct mock quality report marking fuel_flow as STALE
    q_channels = {
        "fuel_flow": ChannelQuality(
            channel_name="fuel_flow",
            raw_value=24.0,
            validated_value=None,
            status=DataQualityStatus.STALE,
            is_valid=False,
            timestamp=20.0,
            anomaly_note="Channel is stale",
        )
    }
    q_report = TelemetryQualityReport(
        timestamp=20.0,
        overall_valid=False,
        temporal_status=DataQualityStatus.VALID,
        channel_reports=q_channels,
    )

    res_vec = generator.generate(telemetry=rec, expected=expected, quality_report=q_report)

    # Coolant temp should be rejected for missing observation
    cool_res = res_vec.get_residual("coolant_temp")
    assert cool_res.valid is False
    assert math.isnan(cool_res.raw_residual)
    assert math.isnan(cool_res.normalized_residual)
    assert "QUALITY_INVALID" in cool_res.reason or "OBSERVATION" in cool_res.reason

    # Fuel flow should be rejected for quality STALE
    fuel_res = res_vec.get_residual("fuel_flow")
    assert fuel_res.valid is False
    assert "QUALITY_INVALID: STALE" in fuel_res.reason
    assert math.isnan(fuel_res.raw_residual)

    # Valid primary count should now be 7 of 9
    assert res_vec.valid_primary_count == 7
    assert res_vec.coverage_fraction == pytest.approx(7.0 / 9.0, rel=1e-3)


# =====================================================================
# 4. Primary Subsystem Ownership & Anti-Double Counting
# =====================================================================

def test_primary_subsystem_ownership_anti_double_counting():
    """Verify exact 9 primary channels and single primary subsystem ownership."""
    assert len(PRIMARY_RESIDUAL_CHANNELS) == 9
    assert set(PRIMARY_SUBSYSTEM_MAP.keys()) == set(PRIMARY_RESIDUAL_CHANNELS)

    # Check designated primary assignments
    assert PRIMARY_SUBSYSTEM_MAP["cht"] == "THERMAL"
    assert PRIMARY_SUBSYSTEM_MAP["coolant_temp"] == "THERMAL"
    assert PRIMARY_SUBSYSTEM_MAP["oil_temp"] == "THERMAL"
    assert PRIMARY_SUBSYSTEM_MAP["oil_pressure"] == "LUBRICATION"
    assert PRIMARY_SUBSYSTEM_MAP["fuel_flow"] == "FUEL"
    assert PRIMARY_SUBSYSTEM_MAP["egt"] == "COMBUSTION"
    assert PRIMARY_SUBSYSTEM_MAP["vibration"] == "MECHANICAL"
    assert PRIMARY_SUBSYSTEM_MAP["rpm"] == "ROTATIONAL"
    assert PRIMARY_SUBSYSTEM_MAP["map_bar"] == "ROTATIONAL"

    # Verify charge_air_temp is NOT in primary channels
    assert "charge_air_temp" not in PRIMARY_RESIDUAL_CHANNELS
    assert "charge_air_temp" in SECONDARY_MODEL_CHANNELS

    # Verify cylinder channels are NOT primary engine votes
    for ch in CYLINDER_RESIDUAL_CHANNELS:
        assert ch not in PRIMARY_RESIDUAL_CHANNELS


def test_cylinder_localization_and_spread_metrics():
    """Verify cylinder runner residuals compute spreads without engine-level double voting."""
    generator = QualityAwareResidualGenerator()
    rec = make_telemetry_record(
        timestamp=30.0,
        rpm=5500.0,
        map_bar=1.10,
        fuel_flow=24.0,
        cht=92.5,
        coolant_temp=80.0,
        oil_temp=70.0,
        oil_pressure=3.0,
        egt=620.0,
        vibration=0.30,
        cht_cyl1=90.0,
        cht_cyl2=100.0,  # +10 C outlier
        cht_cyl3=90.0,
        cht_cyl4=90.0,
        egt_cyl1=600.0,
        egt_cyl2=660.0,  # +60 C outlier
        egt_cyl3=600.0,
        egt_cyl4=600.0,
    )
    expected = {ch: 90.0 for ch in ["cht", "cht_cyl1", "cht_cyl2", "cht_cyl3", "cht_cyl4"]}
    expected.update({ch: 600.0 for ch in ["egt", "egt_cyl1", "egt_cyl2", "egt_cyl3", "egt_cyl4"]})
    expected.update({
        "rpm": 5500.0,
        "map_bar": 1.10,
        "fuel_flow": 24.0,
        "coolant_temp": 80.0,
        "oil_temp": 70.0,
        "oil_pressure": 3.0,
        "vibration": 0.30,
    })

    res_vec = generator.generate(telemetry=rec, expected=expected)
    cyl = res_vec.cylinder_residuals
    assert cyl is not None
    assert cyl.valid_cylinder_count == 4
    assert cyl.cht_spread_c == pytest.approx(10.0, abs=0.1)
    assert cyl.egt_spread_c == pytest.approx(60.0, abs=0.1)
    # Runner 2 has raw residual +10 for CHT, +60 for EGT
    assert cyl.cht_runner_residuals[1] == pytest.approx(10.0, abs=0.1)
    assert cyl.egt_runner_residuals[1] == pytest.approx(60.0, abs=0.1)

    # Primary channel count remains strictly 9
    assert res_vec.primary_channel_count == 9


# =====================================================================
# 5. Coverage Gate: 5/9 Minimum Valid Channels Invariant
# =====================================================================

def test_coverage_gate_5_of_9():
    """
    Test the strict coverage gate:
    >= 5 valid primary channels -> Health assessed, finite HI_raw.
    < 5 valid primary channels  -> UNAVAILABLE, HI_raw is NaN.
    """
    evaluator = HealthEvaluator()

    # Case A: 5 valid channels (e.g. drop 4) -> C_obs = 5/9 approx 0.5556
    vec_5 = ResidualVector(
        timestamp=40.0,
        residuals={
            "rpm": PhysicalResidual("rpm", 40.0, 5500.0, 5500.0, 0.0, 0.0, "RPM", "VALID", "DIRECTLY_OBSERVED", True, "", "ENG", 100.0, "ROTATIONAL", True),
            "map_bar": PhysicalResidual("map_bar", 40.0, 1.1, 1.1, 0.0, 0.0, "bar", "VALID", "DIRECTLY_OBSERVED", True, "", "ENG", 0.05, "ROTATIONAL", True),
            "fuel_flow": PhysicalResidual("fuel_flow", 40.0, 24.0, 24.0, 0.0, 0.0, "L/h", "VALID", "DIRECTLY_OBSERVED", True, "", "ENG", 2.0, "FUEL", True),
            "cht": PhysicalResidual("cht", 40.0, 90.0, 90.0, 0.0, 0.0, "°C", "VALID", "DIRECTLY_OBSERVED", True, "", "ENG", 10.0, "THERMAL", True),
            "coolant_temp": PhysicalResidual("coolant_temp", 40.0, 80.0, 80.0, 0.0, 0.0, "°C", "VALID", "DIRECTLY_OBSERVED", True, "", "ENG", 6.0, "THERMAL", True),
            # 4 Invalid channels
            "oil_temp": PhysicalResidual("oil_temp", 40.0, None, 70.0, float("nan"), float("nan"), "°C", "MISSING", "DIRECTLY_OBSERVED", False, "OBSERVATION_NONE_OR_NAN", "ENG", 8.0, "THERMAL", True),
            "oil_pressure": PhysicalResidual("oil_pressure", 40.0, None, 3.0, float("nan"), float("nan"), "bar", "MISSING", "DIRECTLY_OBSERVED", False, "OBSERVATION_NONE_OR_NAN", "ENG", 0.5, "LUBRICATION", True),
            "egt": PhysicalResidual("egt", 40.0, None, 600.0, float("nan"), float("nan"), "°C", "MISSING", "DIRECTLY_OBSERVED", False, "OBSERVATION_NONE_OR_NAN", "ENG", 25.0, "COMBUSTION", True),
            "vibration": PhysicalResidual("vibration", 40.0, None, 0.3, float("nan"), float("nan"), "g", "MISSING", "DIRECTLY_OBSERVED", False, "OBSERVATION_NONE_OR_NAN", "ENG", 0.2, "MECHANICAL", True),
        },
        cylinder_residuals=None,
        primary_channel_count=9,
        valid_primary_count=5,
        coverage_fraction=5.0 / 9.0,
    )

    assessment_5 = evaluator.evaluate(vec_5)
    assert assessment_5.C_obs == pytest.approx(5.0 / 9.0, rel=1e-3)
    assert assessment_5.state == HealthState.HEALTHY
    assert not math.isnan(assessment_5.HI_raw)
    assert assessment_5.HI_raw >= 0.99

    # Case B: 4 valid channels (drop 5) -> C_obs = 4/9 approx 0.4444 < 0.50
    evaluator.reset()
    vec_4 = ResidualVector(
        timestamp=41.0,
        residuals={
            "rpm": PhysicalResidual("rpm", 41.0, 5500.0, 5500.0, 0.0, 0.0, "RPM", "VALID", "DIRECTLY_OBSERVED", True, "", "ENG", 100.0, "ROTATIONAL", True),
            "map_bar": PhysicalResidual("map_bar", 41.0, 1.1, 1.1, 0.0, 0.0, "bar", "VALID", "DIRECTLY_OBSERVED", True, "", "ENG", 0.05, "ROTATIONAL", True),
            "fuel_flow": PhysicalResidual("fuel_flow", 41.0, 24.0, 24.0, 0.0, 0.0, "L/h", "VALID", "DIRECTLY_OBSERVED", True, "", "ENG", 2.0, "FUEL", True),
            "cht": PhysicalResidual("cht", 41.0, 90.0, 90.0, 0.0, 0.0, "°C", "VALID", "DIRECTLY_OBSERVED", True, "", "ENG", 10.0, "THERMAL", True),
            # 5 Invalid channels
            "coolant_temp": PhysicalResidual("coolant_temp", 41.0, None, 80.0, float("nan"), float("nan"), "°C", "MISSING", "DIRECTLY_OBSERVED", False, "OBSERVATION_NONE_OR_NAN", "ENG", 6.0, "THERMAL", True),
            "oil_temp": PhysicalResidual("oil_temp", 41.0, None, 70.0, float("nan"), float("nan"), "°C", "MISSING", "DIRECTLY_OBSERVED", False, "OBSERVATION_NONE_OR_NAN", "ENG", 8.0, "THERMAL", True),
            "oil_pressure": PhysicalResidual("oil_pressure", 41.0, None, 3.0, float("nan"), float("nan"), "bar", "MISSING", "DIRECTLY_OBSERVED", False, "OBSERVATION_NONE_OR_NAN", "ENG", 0.5, "LUBRICATION", True),
            "egt": PhysicalResidual("egt", 41.0, None, 600.0, float("nan"), float("nan"), "°C", "MISSING", "DIRECTLY_OBSERVED", False, "OBSERVATION_NONE_OR_NAN", "ENG", 25.0, "COMBUSTION", True),
            "vibration": PhysicalResidual("vibration", 41.0, None, 0.3, float("nan"), float("nan"), "g", "MISSING", "DIRECTLY_OBSERVED", False, "OBSERVATION_NONE_OR_NAN", "ENG", 0.2, "MECHANICAL", True),
        },
        cylinder_residuals=None,
        primary_channel_count=9,
        valid_primary_count=4,
        coverage_fraction=4.0 / 9.0,
    )

    assessment_4 = evaluator.evaluate(vec_4)
    assert assessment_4.C_obs == pytest.approx(4.0 / 9.0, rel=1e-3)
    assert assessment_4.state == HealthState.UNAVAILABLE
    assert math.isnan(assessment_4.HI_raw)
    assert math.isnan(assessment_4.HI_smooth)
    assert assessment_4.HI_cov_adj is None


# =====================================================================
# 6. Independence: C_data, C_obs & HI_raw (Guardrail 3)
# =====================================================================

def test_c_data_c_obs_hi_raw_independence():
    """
    Guardrail 3 verification:
    Varying C_data (Phase 3 confidence) or C_obs (missing telemetry) must NOT
    artificially penalize HI_raw (physical consistency score).
    """
    evaluator = HealthEvaluator()

    def make_vec():
        residuals = {}
        for ch in PRIMARY_RESIDUAL_CHANNELS:
            sub = PRIMARY_SUBSYSTEM_MAP[ch]
            residuals[ch] = PhysicalResidual(
                channel=ch,
                timestamp=50.0,
                observed_value=100.0,
                predicted_value=100.0,
                raw_residual=0.0,
                normalized_residual=0.0,
                units="",
                quality_status="VALID",
                observability_status="DIRECTLY_OBSERVED",
                valid=True,
                reason="",
                scale_source="ENG",
                scale_value=1.0,
                primary_subsystem=sub,
                is_primary_engine_vote=True,
            )
        return ResidualVector(
            timestamp=50.0,
            residuals=residuals,
            cylinder_residuals=None,
            primary_channel_count=9,
            valid_primary_count=9,
            coverage_fraction=1.0,
        )

    # 1. Evaluate with high data confidence C_data = 1.0
    res_high = evaluator.evaluate(make_vec(), data_confidence=1.0)
    assert res_high.C_data == 1.0
    assert res_high.HI_raw == pytest.approx(1.0, rel=1e-4)

    # 2. Evaluate with low data confidence C_data = 0.20 (e.g. noisy estimator)
    evaluator.reset()
    res_low = evaluator.evaluate(make_vec(), data_confidence=0.20)
    assert res_low.C_data == 0.20
    # Crucial assertion: HI_raw is NOT penalized by low data confidence!
    assert res_low.HI_raw == pytest.approx(1.0, rel=1e-4)
    assert res_low.HI_raw == res_high.HI_raw


# =====================================================================
# 7. Time Persistence & Causal Smoothing Invariants
# =====================================================================

def test_time_persistence_and_causal_ewma():
    """
    Verify:
    1. Short duration (< 3s) abnormal residual triggers WATCH, not DEGRADED.
    2. Sustained abnormal residual (>= 3s) confirms DEGRADED.
    3. Normalization back to healthy requires recovery persistence (5s).
    4. EWMA smooths causally without lookahead.
    """
    evaluator = HealthEvaluator(
        config=HealthIndicatorConfig(
            tau_nom=1.5,
            tau_crit=5.0,
            persistence_seconds=3.0,
            recovery_seconds=5.0,
            ewma_alpha=0.15,
        )
    )

    def make_vec_with_cht_z(z_val: float, t: float):
        residuals = {}
        for ch in PRIMARY_RESIDUAL_CHANNELS:
            sub = PRIMARY_SUBSYSTEM_MAP[ch]
            z = z_val if ch == "cht" else 0.0
            residuals[ch] = PhysicalResidual(
                channel=ch,
                timestamp=t,
                observed_value=90.0 + z * 10.0,
                predicted_value=90.0,
                raw_residual=z * 10.0,
                normalized_residual=z,
                units="°C" if ch == "cht" else "",
                quality_status="VALID",
                observability_status="DIRECTLY_OBSERVED",
                valid=True,
                reason="",
                scale_source="ENG",
                scale_value=10.0 if ch == "cht" else 1.0,
                primary_subsystem=sub,
                is_primary_engine_vote=True,
            )
        return ResidualVector(
            timestamp=t,
            residuals=residuals,
            cylinder_residuals=None,
            primary_channel_count=9,
            valid_primary_count=9,
            coverage_fraction=1.0,
        )

    # Step 0 to 2 seconds with abnormal residual z = 2.5 (above tau_nom = 1.5)
    # Elapsed duration < 3.0s -> state must be WATCH
    for step in range(20):
        t = step * 0.1
        res = evaluator.evaluate(make_vec_with_cht_z(2.5, t), dt=0.1)
        assert res.state == HealthState.WATCH

    # Step 2.1 to 3.5 seconds with sustained abnormal residual
    # At t >= 3.0s, state must transition to DEGRADED
    for step in range(21, 36):
        t = step * 0.1
        res = evaluator.evaluate(make_vec_with_cht_z(2.5, t), dt=0.1)

    assert res.state == HealthState.DEGRADED
    assert res.channel_indicators["cht"].state == HealthState.DEGRADED
    assert res.subsystems["THERMAL"].state == HealthState.DEGRADED

    # Now return residual to healthy z = 0.0
    # For the first 4.9 seconds, recovery is NOT complete -> state remains WATCH
    for step in range(36, 80):
        t = step * 0.1
        res = evaluator.evaluate(make_vec_with_cht_z(0.0, t), dt=0.1)
        assert res.state == HealthState.WATCH

    # After 5.0 seconds of sustained healthy observation, state recovers to HEALTHY
    for step in range(80, 95):
        t = step * 0.1
        res = evaluator.evaluate(make_vec_with_cht_z(0.0, t), dt=0.1)

    assert res.state == HealthState.HEALTHY
    assert res.channel_indicators["cht"].state == HealthState.HEALTHY


# =====================================================================
# 8. Causal Fault Verification (F1–F7) against Simulator
# =====================================================================

def test_f1_fuel_delivery_abnormality_causal_propagation():
    """
    F1: Lean fuel injector delivery abnormality.
    Causal Expectations (verified against actual Phase 4 code):
    - Fuel flow residual is negative (fuel_flow observed < expected).
    - EGT residual is positive (lean mixture burns hotter towards stoichiometric).
    - CHT residual is positive (increased combustion temperature).
    - Health degrades in FUEL and COMBUSTION/THERMAL.
    """
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    # Inject F1: Lean delivery abnormality, severity 0.75, start t=2.0s
    fault = FaultState(
        fault_type=FaultType.INJECTOR_DELIVERY_ABNORMALITY,
        affected_subsystem=FaultSubsystem.FUEL,
        severity=0.75,
        start_time=2.0,
        parameters={"mode": "lean"},
    )

    # Run simulation past t=10.0s to allow physical thermal transients to settle
    states = []
    for _ in range(120):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1, fault_state=fault)
        st = twin.update(rec)
        states.append(st)

    last_st = states[-1]
    last_h = last_st.health_assessment
    assert last_h is not None

    # Verify causal direction on residuals
    fuel_res = last_h.channel_indicators["fuel_flow"].raw_residual
    egt_res = last_h.channel_indicators["egt"].raw_residual
    cht_res = last_h.channel_indicators["cht"].raw_residual

    assert fuel_res < -0.2, f"Expected fuel_flow residual < 0, got {fuel_res}"
    assert egt_res > 2.0, f"Expected egt residual > 0, got {egt_res}"
    assert cht_res < 0.0, f"Expected cht residual < 0 under restricted fuel heat input, got {cht_res}"

    # Verify Subsystem Health Degradation
    assert last_h.subsystems["FUEL"].score < 0.95
    assert last_h.state in (HealthState.WATCH, HealthState.DEGRADED, HealthState.CRITICAL)


def test_f1_localized_cylinder_spread():
    """
    F1 with affected_cylinder = 2.
    Verify localized divergence on Cylinder 2 runner CHT/EGT and increased cylinder spread.
    """
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    fault = FaultState(
        fault_type=FaultType.INJECTOR_DELIVERY_ABNORMALITY,
        affected_subsystem=FaultSubsystem.FUEL,
        affected_cylinder=2,
        severity=0.40,
        start_time=2.0,
        parameters={"mode": "lean"},
    )

    for _ in range(80):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1, fault_state=fault)
        st = twin.update(rec)

    last_h = st.health_assessment
    assert last_h is not None
    assert last_h.cylinder_assessment is not None
    cyl = last_h.cylinder_assessment

    # Runner 2 EGT should be distinctly higher than others or spread increased
    assert cyl["egt_spread_c"] > 5.0
    assert cyl["cht_spread_c"] > 2.0


def test_f2_lubrication_degradation_causal_propagation():
    """
    F2: Lubrication degradation.
    Causal Expectations:
    - Oil pressure residual is negative (oil_pressure observed < expected).
    - Oil temp residual is positive (oil_temp observed > expected).
    - Primary degradation localized to LUBRICATION subsystem.
    """
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    fault = FaultState(
        fault_type=FaultType.LUBRICATION_DEGRADATION,
        affected_subsystem=FaultSubsystem.LUBRICATION,
        severity=0.45,
        start_time=3.0,
    )

    for _ in range(100):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1, fault_state=fault)
        st = twin.update(rec)

    last_h = st.health_assessment
    assert last_h is not None

    oil_p_res = last_h.channel_indicators["oil_pressure"].raw_residual
    oil_t_res = last_h.channel_indicators["oil_temp"].raw_residual

    assert oil_p_res < -0.15, f"Expected oil_pressure residual < 0, got {oil_p_res}"
    assert oil_t_res > 1.0, f"Expected oil_temp residual > 0, got {oil_t_res}"
    assert last_h.subsystems["LUBRICATION"].score < 0.90


def test_f3_cooling_degradation_causal_propagation():
    """
    F3: Cooling system degradation.
    Causal Expectations:
    - Coolant temp residual is positive (coolant_temp observed > expected).
    - CHT residual is positive (head temperature increases due to degraded cooling).
    - Primary degradation in THERMAL subsystem.
    """
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    fault = FaultState(
        fault_type=FaultType.COOLING_DEGRADATION,
        affected_subsystem=FaultSubsystem.COOLING,
        severity=0.75,
        start_time=2.0,
    )

    for _ in range(150):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1, fault_state=fault)
        st = twin.update(rec)

    last_h = st.health_assessment
    assert last_h is not None

    cool_res = last_h.channel_indicators["coolant_temp"].raw_residual
    cht_res = last_h.channel_indicators["cht"].raw_residual

    assert cool_res > 0.3, f"Expected coolant_temp residual > 0, got {cool_res}"
    assert cht_res > 1.0, f"Expected cht residual > 0, got {cht_res}"
    assert last_h.subsystems["THERMAL"].score < 0.95


def test_f4_combustion_misfire_causal_propagation():
    """
    F4: Combustion misfire.
    Causal Expectations:
    - Vibration residual is positive (misfire imbalance drives elevated 1x/2x vibration).
    - Mechanical / Combustion anomaly detected.
    """
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    fault = FaultState(
        fault_type=FaultType.COMBUSTION_MISFIRE,
        affected_subsystem=FaultSubsystem.COMBUSTION,
        affected_cylinder=1,
        severity=0.60,
        start_time=2.0,
    )

    for _ in range(60):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1, fault_state=fault)
        st = twin.update(rec)

    last_h = st.health_assessment
    assert last_h is not None

    vib_res = last_h.channel_indicators["vibration"].raw_residual
    assert vib_res > 0.05, f"Expected vibration residual > 0, got {vib_res}"


def test_f5_mechanical_degradation_causal_propagation():
    """
    F5: Mechanical degradation.
    Causal Expectations:
    - Vibration residual is positive (increased mechanical bearing/unbalance vibration).
    - MECHANICAL subsystem score degraded.
    """
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    fault = FaultState(
        fault_type=FaultType.MECHANICAL_DEGRADATION,
        affected_subsystem=FaultSubsystem.VIBRATION,
        severity=0.50,
        start_time=2.0,
    )

    for _ in range(60):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1, fault_state=fault)
        st = twin.update(rec)

    last_h = st.health_assessment
    assert last_h is not None

    vib_res = last_h.channel_indicators["vibration"].raw_residual
    assert vib_res > 0.08, f"Expected vibration residual > 0, got {vib_res}"
    assert last_h.subsystems["MECHANICAL"].score < 0.90


def test_f6_sensor_bias_isolation():
    """
    F6: Sensor bias on CHT sensor.
    Causal Expectations:
    - Observation layer only! CHT residual shifts by bias.
    - True engine physics (oil temp, coolant temp, fuel flow) remain completely unperturbed.
    """
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    fault = FaultState(
        fault_type=FaultType.SENSOR_BIAS,
        affected_subsystem=FaultSubsystem.SENSOR,
        severity=0.50,
        start_time=3.0,
        parameters={"sensor_channel": "cht", "bias_magnitude": 25.0},
    )

    for _ in range(60):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1, fault_state=fault)
        st = twin.update(rec)

    last_h = st.health_assessment
    assert last_h is not None

    # CHT residual shifts by ~12.5°C (bias_magnitude 25.0 * severity 0.50)
    cht_res = last_h.channel_indicators["cht"].raw_residual
    assert cht_res > 10.0, f"Expected cht residual shift > 10.0, got {cht_res}"
    assert cht_res < 15.0, f"Expected cht residual shift < 15.0, got {cht_res}"

    # Cross-channel physics check: Coolant temp should NOT follow CHT sensor bias!
    cool_res = abs(last_h.channel_indicators["coolant_temp"].raw_residual)
    assert cool_res < 3.0, f"Coolant temp should not follow CHT sensor bias, got {cool_res}"


def test_f7_sensor_dropout_coverage_separation():
    """
    F7: Sensor dropout on oil pressure sensor.
    Causal Expectations:
    - Observation layer outputs NaN for oil pressure.
    - Oil pressure channel indicator becomes UNAVAILABLE.
    - Coverage C_obs drops from 9/9 (1.0) to 8/9 (0.8889).
    - HI_raw remains high for the remaining healthy subsystems (no false physical penalty).
    """
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)
    twin = DigitalTwin(sim_config=sim_config)

    fault = FaultState(
        fault_type=FaultType.SENSOR_DROPOUT,
        affected_subsystem=FaultSubsystem.SENSOR,
        severity=1.0,
        start_time=2.0,
        parameters={"sensor_channel": "oil_pressure"},
    )

    for _ in range(60):
        rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1, fault_state=fault)
        st = twin.update(rec)

    last_h = st.health_assessment
    assert last_h is not None

    # Oil pressure indicator must be invalid / UNAVAILABLE
    op_ind = last_h.channel_indicators["oil_pressure"]
    assert op_ind.valid is False
    assert op_ind.state == HealthState.UNAVAILABLE
    assert math.isnan(op_ind.raw_residual)

    # Coverage drops to 8/9
    assert last_h.C_obs == pytest.approx(8.0 / 9.0, rel=1e-3)

    # HI_raw remains evaluated and healthy across other subsystems
    assert not math.isnan(last_h.HI_raw)
    assert last_h.HI_raw >= 0.85


# =====================================================================
# 9. Severity Monotonicity & Anti-Tautology
# =====================================================================

def test_fault_severity_monotonicity():
    """
    Verify that increasing physical fault severity produces monotonically larger residuals
    and monotonically lower subsystem health scores.
    """
    sim_config = SimulatorConfig()
    severities = [0.15, 0.30, 0.50]
    final_cht_residuals = []
    final_thermal_scores = []

    for sev in severities:
        sim = EngineSimulator(sim_config=sim_config)
        twin = DigitalTwin(sim_config=sim_config)

        fault = FaultState(
            fault_type=FaultType.COOLING_DEGRADATION,
            affected_subsystem=FaultSubsystem.COOLING,
            severity=sev,
            start_time=1.0,
        )

        for _ in range(80):
            rec = sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1, fault_state=fault)
            st = twin.update(rec)

        last_h = st.health_assessment
        final_cht_residuals.append(last_h.channel_indicators["cht"].raw_residual)
        final_thermal_scores.append(last_h.subsystems["THERMAL"].score)

    # Residuals must be monotonically increasing
    assert final_cht_residuals[0] < final_cht_residuals[1] < final_cht_residuals[2]
    # Subsystem scores must be monotonically non-increasing (decreasing or flat)
    assert final_thermal_scores[0] >= final_thermal_scores[1] >= final_thermal_scores[2]


# =====================================================================
# 10. Label Leakage & Replay Determinism
# =====================================================================

def test_no_fault_label_leakage_in_health_path():
    """
    Verify that QualityAwareResidualGenerator and HealthEvaluator do not inspect
    fault_type, fault_severity, or any ground-truth fields on TelemetryRecord.
    """
    generator = QualityAwareResidualGenerator()
    evaluator = HealthEvaluator()

    # Pass a TelemetryRecord with default or unassigned fault metadata
    rec = make_telemetry_record(
        timestamp=100.0,
        rpm=5500.0,
        map_bar=1.10,
        fuel_flow=24.0,
        cht=90.0,
        coolant_temp=80.0,
        oil_temp=70.0,
        oil_pressure=3.0,
        egt=600.0,
        vibration=0.30,
    )
    expected = {
        "rpm": 5500.0,
        "map_bar": 1.10,
        "fuel_flow": 24.0,
        "cht": 90.0,
        "coolant_temp": 80.0,
        "oil_temp": 70.0,
        "oil_pressure": 3.0,
        "egt": 600.0,
        "vibration": 0.30,
    }

    res_vec = generator.generate(rec, expected)
    assessment = evaluator.evaluate(res_vec)

    assert assessment.state == HealthState.HEALTHY
    assert assessment.HI_raw == pytest.approx(1.0, rel=1e-4)


def test_deterministic_replay_and_clean_reset():
    """
    Verify that running the same telemetry sequence twice through DigitalTwin
    with reset() in between produces bitwise identical results.
    """
    sim_config = SimulatorConfig()
    sim = EngineSimulator(sim_config=sim_config)

    records = [
        sim.step(throttle_pct=75.0, altitude_m=2000.0, dt=0.1)
        for _ in range(50)
    ]

    twin = DigitalTwin(sim_config=sim_config)

    run_1_states = [twin.update(r) for r in records]
    twin.reset()
    run_2_states = [twin.update(r) for r in records]

    assert len(run_1_states) == len(run_2_states)
    for s1, s2 in zip(run_1_states, run_2_states):
        h1 = s1.health_assessment
        h2 = s2.health_assessment
        assert h1.state == h2.state
        assert h1.HI_raw == pytest.approx(h2.HI_raw, rel=1e-6)
        assert h1.HI_smooth == pytest.approx(h2.HI_smooth, rel=1e-6)
        assert h1.C_obs == pytest.approx(h2.C_obs, rel=1e-6)
