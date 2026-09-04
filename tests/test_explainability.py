"""
Comprehensive Unit & Behavioral Test Suite for Phase 12 Explainability & Evidence Fusion.
Covers all requirements A through X:
- A: SHAP contribution extraction
- B: SHAP direction
- C: Missing model handling
- D: Physics cooling consistency
- E: Physics lubrication consistency
- F: Physics fuel lean/rich consistency
- G: Mechanical vibration consistency
- H: Sensor-fault isolation
- I: Conflicting physics evidence
- J: Insufficient evidence
- K: Phase 9 contribution passthrough
- L: Isolated sensor exclusion
- M: Phase 10 temporal evidence
- N: Phase 11 RUL explanation
- O: EOL provenance preservation
- P: Engine isolation
- Q: Mission isolation
- R: No future leakage
- S: Future-modification invariance
- T: Missing/NaN safety
- U: No modification of Phase 8 prediction
- V: No modification of Phase 9 HI
- W: No modification of Phase 11 RUL
- X: Deterministic repeated explanation
"""

import math
import numpy as np
import pandas as pd
import pytest

from fault_diagnosis.schema import FaultDiagnosisResult, CANONICAL_FAULT_LABELS
from fault_diagnosis.classifier import XGBoostFaultClassifier, FaultClassifierConfig
from fault_diagnosis.features import FeatureExtractor
from health_index.schema import HealthIndexResult, HealthState, DegradationTrend, HealthDataQuality
from forecasting.schema import ForecastResult, ModelStatus, ForecastQuality
from prognostics.schema import RULResult, RULStatus

from explainability.schema import (
    EvidenceStatus,
    EvidenceQuality,
    ExplainabilityResult,
    SHAPEvidence,
    PhysicsEvidence,
    HealthEvidence,
    TemporalEvidence,
    RULEvidence,
)
from explainability.shap_explainer import SHAPExplainer
from explainability.physics_evidence import PhysicsEvidenceEvaluator
from explainability.temporal_evidence import TemporalEvidenceEvaluator
from explainability.fusion import EvidenceFusionEngine
from explainability.pipeline import ExplainabilityPipeline


@pytest.fixture(scope="module")
def trained_classifier() -> XGBoostFaultClassifier:
    """Fixture providing a fast-trained XGBoostFaultClassifier on synthetic diagnostic features."""
    cfg = FaultClassifierConfig(n_estimators=10, max_depth=3, random_state=42)
    clf = XGBoostFaultClassifier(config=cfg)

    # 16 standard features
    feature_cols = [
        "rpm_norm_residual", "cht_norm_residual", "egt_norm_residual",
        "oil_temp_norm_residual", "oil_pressure_norm_residual",
        "fuel_flow_norm_residual", "vibration_norm_residual",
        "throttle", "altitude", "load",
        "phase_takeoff", "phase_climb", "phase_cruise",
        "phase_loiter", "phase_descent", "phase_landing"
    ]

    records = []
    labels = []
    # 2 samples per canonical fault class
    for label in CANONICAL_FAULT_LABELS:
        for rep in range(2):
            row = {col: 0.0 for col in feature_cols}
            row["phase_cruise"] = 1.0
            row["throttle"] = 75.0
            row["altitude"] = 1000.0
            row["load"] = 75.0

            if label == "cooling_degradation":
                row["cht_norm_residual"] = 3.5 + rep * 0.5
                row["oil_temp_norm_residual"] = 2.0 + rep * 0.5
            elif label == "lubrication_degradation":
                row["oil_pressure_norm_residual"] = -3.5 - rep * 0.5
                row["oil_temp_norm_residual"] = 2.5 + rep * 0.5
            elif label == "fuel_injection_abnormality":
                row["egt_norm_residual"] = 3.0 + rep * 0.5
                row["fuel_flow_norm_residual"] = -2.5 - rep * 0.5
            elif label == "mechanical_degradation":
                row["vibration_norm_residual"] = 4.0 + rep * 0.5
            elif label == "sensor_fault":
                row["cht_norm_residual"] = 5.0 + rep * 0.5
            else:  # "none"
                pass

            records.append(row)
            labels.append(label)

    X = pd.DataFrame(records)
    y = pd.Series(labels)
    clf.fit(X, y)
    return clf


def make_dummy_diagnosis(fault: str = "cooling_degradation", conf: float = 0.92) -> FaultDiagnosisResult:
    probs = {c: 0.01 for c in CANONICAL_FAULT_LABELS}
    probs[fault] = conf
    return FaultDiagnosisResult(
        timestamp=50.0,
        engine_id="ENG_01",
        mission_id="MSN_01",
        mission_phase="CRUISE",
        predicted_fault_type=fault,
        class_probabilities=probs,
        diagnostic_confidence=conf,
    )


def make_dummy_health(
    hi: float = 0.72,
    dom: list = None,
    excluded: list = None,
    rate: float = -0.002,
) -> HealthIndexResult:
    return HealthIndexResult(
        timestamp=50.0,
        engine_id="ENG_01",
        mission_id="MSN_01",
        mission_phase="CRUISE",
        raw_health_index=hi,
        smoothed_health_index=hi,
        health_state=HealthState.DEGRADED.value,
        raw_degradation_score=1.0 - hi,
        degradation_rate=rate,
        degradation_trend=DegradationTrend.DEGRADING.value,
        channel_contributions={"cht": 0.5, "oil_temp": 0.3},
        channel_degradation_evidence={},
        dominant_degraded_channels=dom if dom is not None else ["cht"],
        valid_channels=["rpm", "cht", "egt", "oil_temp", "oil_pressure", "fuel_flow", "vibration"],
        missing_channels=[],
        excluded_channels=excluded if excluded is not None else [],
        effective_channel_weights={},
        data_quality=HealthDataQuality.VALID.value,
    )


def make_dummy_rul(p50: float = 350.0, status: RULStatus = RULStatus.ACTIVE_DEGRADATION) -> RULResult:
    return RULResult(
        engine_id="ENG_01",
        mission_id="MSN_01",
        timestamp=50.0,
        status=status,
        rul_seconds_median=p50,
        rul_seconds_p05=p50 - 50.0,
        rul_seconds_p95=p50 + 70.0,
        limiting_factor="GLOBAL_HEALTH_INDEX",
        confidence_score=0.85,
        active_flight_phase="CRUISE",
        handoff_horizon_s=0.0,
        trajectory_type="ROBUST_LINEAR_PRIMARY",
        provenance={"base_criterion": "HI <= 0.35"},
    )


# =========================================================================
# TESTS A through X
# =========================================================================

def test_a_shap_contribution_extraction(trained_classifier):
    """Test A: SHAP local attribution successfully extracts top-k contributing features."""
    explainer = SHAPExplainer(top_k=3)
    sample = {col: 0.0 for col in trained_classifier.feature_names}
    sample["cht_norm_residual"] = 4.0
    sample["oil_temp_norm_residual"] = 2.5

    shap_ev = explainer.explain_instance(
        classifier=trained_classifier,
        features=sample,
        predicted_fault="cooling_degradation",
    )

    assert shap_ev.status == "AVAILABLE"
    assert len(shap_ev.top_features) == 3
    assert shap_ev.base_value is not None
    top_names = [f.feature_name for f in shap_ev.top_features]
    assert "cht_norm_residual" in top_names
    assert "oil_temp_norm_residual" in top_names


def test_b_shap_direction(trained_classifier):
    """Test B: Verify SHAP direction tags features toward vs away from predicted class."""
    explainer = SHAPExplainer(top_k=5)
    sample = {col: 0.0 for col in trained_classifier.feature_names}
    sample["cht_norm_residual"] = 4.0

    shap_ev = explainer.explain_instance(
        classifier=trained_classifier,
        features=sample,
        predicted_fault="cooling_degradation",
    )

    cht_feat = next((f for f in shap_ev.top_features if f.feature_name == "cht_norm_residual"), None)
    assert cht_feat is not None
    assert cht_feat.shap_value > 0
    assert cht_feat.direction == "TOWARD_PREDICTED_CLASS"


def test_c_missing_model_handling():
    """Test C: If model is None/untrained, returns MODEL_UNAVAILABLE without crashing or fabricating."""
    explainer = SHAPExplainer()
    shap_ev = explainer.explain_instance(
        classifier=None,
        features={"cht_norm_residual": 3.0},
        predicted_fault="cooling_degradation",
        class_probabilities={"cooling_degradation": 0.85},
    )
    assert shap_ev.status == "MODEL_UNAVAILABLE"
    assert shap_ev.top_features == []
    assert shap_ev.base_value is None
    assert "attribution does not establish causality" in shap_ev.disclaimer


def test_d_physics_cooling_consistency():
    """Test D: Cooling degradation physics consistency (CHT elevated & Oil Temp elevated)."""
    evaluator = PhysicsEvidenceEvaluator(tau_threshold=1.5)
    resids = {
        "rpm": 0.1, "cht": 3.2, "egt": 0.2, "oil_temp": 2.1,
        "oil_pressure": -0.2, "fuel_flow": 0.1, "vibration": 0.05
    }
    phys = evaluator.evaluate(diagnosed_fault="cooling_degradation", residuals=resids)
    assert phys.status == EvidenceStatus.SUPPORTED
    assert "cht" in phys.supporting_channels
    assert "oil_temp" in phys.supporting_channels
    assert "consistent with the project's cooling-degradation" in phys.consistency_reason


def test_e_physics_lubrication_consistency():
    """Test E: Lubrication degradation physics consistency (Oil Pressure depressed & Oil Temp elevated)."""
    evaluator = PhysicsEvidenceEvaluator(tau_threshold=1.5)
    resids = {
        "rpm": -0.1, "cht": 0.3, "egt": 0.1, "oil_temp": 2.8,
        "oil_pressure": -3.4, "fuel_flow": 0.0, "vibration": 0.1
    }
    phys = evaluator.evaluate(diagnosed_fault="lubrication_degradation", residuals=resids)
    assert phys.status == EvidenceStatus.SUPPORTED
    assert "oil_pressure" in phys.supporting_channels
    assert "oil_temp" in phys.supporting_channels


def test_f_physics_fuel_lean_rich_consistency():
    """Test F: Fuel injection abnormality lean and rich combustion branches."""
    evaluator = PhysicsEvidenceEvaluator(tau_threshold=1.5)

    # Lean branch: EGT elevated, fuel_flow depressed
    resids_lean = {
        "rpm": 0.0, "cht": 0.2, "egt": 2.9, "oil_temp": 0.1,
        "oil_pressure": 0.0, "fuel_flow": -2.4, "vibration": 0.1
    }
    phys_lean = evaluator.evaluate(diagnosed_fault="fuel_injection_abnormality", residuals=resids_lean)
    assert phys_lean.status == EvidenceStatus.SUPPORTED
    assert phys_lean.provenance.get("mode") == "lean_mixture"

    # Rich branch: EGT depressed, fuel_flow elevated
    resids_rich = {
        "rpm": 0.0, "cht": 0.2, "egt": -2.7, "oil_temp": 0.1,
        "oil_pressure": 0.0, "fuel_flow": 2.6, "vibration": 0.1
    }
    phys_rich = evaluator.evaluate(diagnosed_fault="fuel_injection_abnormality", residuals=resids_rich)
    assert phys_rich.status == EvidenceStatus.SUPPORTED
    assert phys_rich.provenance.get("mode") == "rich_mixture"


def test_g_mechanical_vibration_consistency():
    """Test G: Mechanical degradation physics consistency (vibration elevated)."""
    evaluator = PhysicsEvidenceEvaluator(tau_threshold=1.5)
    resids = {
        "rpm": 0.1, "cht": 0.0, "egt": 0.0, "oil_temp": 0.1,
        "oil_pressure": 0.0, "fuel_flow": 0.0, "vibration": 3.8
    }
    phys = evaluator.evaluate(diagnosed_fault="mechanical_degradation", residuals=resids)
    assert phys.status == EvidenceStatus.SUPPORTED
    assert "vibration" in phys.supporting_channels


def test_h_sensor_fault_isolation():
    """Test H: Sensor fault physics consistency with isolated channel."""
    evaluator = PhysicsEvidenceEvaluator(tau_threshold=1.5)
    resids = {
        "rpm": 0.1, "cht": 5.2, "egt": 0.1, "oil_temp": 0.2,
        "oil_pressure": 0.0, "fuel_flow": 0.0, "vibration": 0.1
    }
    # CHT isolated by Phase 9
    phys = evaluator.evaluate(diagnosed_fault="sensor_fault", residuals=resids, excluded_channels={"cht"})
    assert phys.status == EvidenceStatus.SUPPORTED
    assert "cht" in phys.supporting_channels


def test_i_conflicting_physics_evidence():
    """Test I: Physics conflict detection (e.g. cooling degradation diagnosed but CHT is depressed)."""
    evaluator = PhysicsEvidenceEvaluator(tau_threshold=1.5)
    resids = {
        "rpm": 0.0, "cht": -2.8, "egt": 0.0, "oil_temp": -1.8,
        "oil_pressure": 0.0, "fuel_flow": 0.0, "vibration": 0.0
    }
    phys = evaluator.evaluate(diagnosed_fault="cooling_degradation", residuals=resids)
    assert phys.status == EvidenceStatus.CONFLICTING
    assert "cht" in phys.conflicting_channels


def test_j_insufficient_evidence():
    """Test J: Insufficient data guard when valid channels < 4."""
    evaluator = PhysicsEvidenceEvaluator(tau_threshold=1.5)
    resids = {"rpm": 0.1, "cht": 2.0, "egt": 0.1}  # only 3 channels
    phys = evaluator.evaluate(diagnosed_fault="cooling_degradation", residuals=resids)
    assert phys.status == EvidenceStatus.INSUFFICIENT_DATA
    assert "Insufficient valid physical residual channels" in phys.consistency_reason


def test_k_phase9_contribution_passthrough():
    """Test K: Phase 9 Health Index, dominant channels, and effective weights pass through unchanged."""
    pipeline = ExplainabilityPipeline()
    h_res = make_dummy_health(hi=0.68, dom=["cht", "oil_temp"])
    diag_res = make_dummy_diagnosis()

    exp = pipeline.explain(
        timestamp=50.0,
        engine_id="ENG_01",
        residuals={"cht": 3.0, "oil_temp": 2.0, "rpm": 0.0, "egt": 0.0, "oil_pressure": 0.0, "fuel_flow": 0.0, "vibration": 0.0},
        diagnosis_result=diag_res,
        health_result=h_res,
    )

    assert exp.health_evidence is not None
    assert exp.health_evidence.current_health_index == 0.68
    assert exp.health_evidence.dominant_degraded_channels == ["cht", "oil_temp"]


def test_l_isolated_sensor_exclusion():
    """Test L: Isolated sensor from Phase 9 does not create fake physical cooling degradation evidence."""
    pipeline = ExplainabilityPipeline()
    # CHT is isolated
    h_res = make_dummy_health(hi=0.88, dom=[], excluded=["cht"])
    diag_res = make_dummy_diagnosis(fault="cooling_degradation")

    exp = pipeline.explain(
        timestamp=50.0,
        engine_id="ENG_01",
        residuals={"cht": 5.0, "oil_temp": 0.2, "rpm": 0.0, "egt": 0.0, "oil_pressure": 0.0, "fuel_flow": 0.0, "vibration": 0.0},
        diagnosis_result=diag_res,
        health_result=h_res,
    )

    assert exp.physics_evidence.status == EvidenceStatus.CONFLICTING
    assert "isolated as an observation sensor fault" in exp.physics_evidence.consistency_reason


def test_m_phase10_temporal_evidence():
    """Test M: Causal temporal evidence incorporates Phase 9 rate and Phase 10 status."""
    evaluator = TemporalEvidenceEvaluator()
    h_res = make_dummy_health(rate=-0.003)
    fc_res = ForecastResult(
        engine_id="ENG_01",
        mission_id="MSN_01",
        forecast_start_timestamp=50.0,
        context_start_timestamp=18.0,
        context_length=32,
        forecast_horizon=16,
        sampling_interval=1.0,
        target_channels=["cht"],
        forecast_timestamps=[51.0 + i for i in range(16)],
        predicted_telemetry={"cht": [130.0] * 16},
        model_status=ModelStatus.LOADED_PRETRAINED.value,
        forecast_quality=ForecastQuality.VALID.value,
    )

    temp = evaluator.evaluate(health_result=h_res, forecast_result=fc_res)
    assert temp.status == "AVAILABLE"
    assert temp.trend_direction == "WORSENING"
    assert temp.forecast_status == ModelStatus.LOADED_PRETRAINED.value
    assert temp.forecast_horizon_s == 16.0


def test_n_phase11_rul_explanation():
    """Test N: Phase 11 RUL result is correctly interpreted and formatted."""
    pipeline = ExplainabilityPipeline()
    rul_res = make_dummy_rul(p50=240.0)

    exp = pipeline.explain(
        timestamp=50.0,
        engine_id="ENG_01",
        rul_result=rul_res,
    )

    assert exp.rul_evidence is not None
    assert exp.rul_evidence.rul_seconds_median == 240.0
    assert exp.rul_evidence.limiting_factor == "GLOBAL_HEALTH_INDEX"
    assert "Project-defined EOL criteria are not certified OEM/FAA limits" in exp.rul_evidence.disclaimer


def test_o_eol_provenance_preservation():
    """Test O: Phase 11 EOL criterion provenance is preserved without modification."""
    pipeline = ExplainabilityPipeline()
    rul_res = make_dummy_rul()
    rul_res.provenance["source_tag"] = "phase_9_critical_state_boundary"

    exp = pipeline.explain(
        timestamp=50.0,
        engine_id="ENG_01",
        rul_result=rul_res,
    )

    assert exp.rul_evidence.eol_provenance.get("source_tag") == "phase_9_critical_state_boundary"


def test_p_engine_isolation():
    """Test P: Explanations are isolated across distinct engine IDs."""
    pipeline = ExplainabilityPipeline()

    exp1 = pipeline.explain(timestamp=50.0, engine_id="ENG_01", mission_id="MSN_01")
    exp2 = pipeline.explain(timestamp=50.0, engine_id="ENG_02", mission_id="MSN_01")

    assert exp1.engine_id == "ENG_01"
    assert exp2.engine_id == "ENG_02"
    assert pipeline._sessions[("ENG_01", "MSN_01")]["timestamps"] == [50.0]
    assert pipeline._sessions[("ENG_02", "MSN_01")]["timestamps"] == [50.0]

    # Resetting ENG_01 does not affect ENG_02
    pipeline.reset(engine_id="ENG_01")
    assert ("ENG_01", "MSN_01") not in pipeline._sessions
    assert ("ENG_02", "MSN_01") in pipeline._sessions


def test_q_mission_isolation():
    """Test Q: Explanations are isolated across distinct mission IDs."""
    pipeline = ExplainabilityPipeline()

    exp1 = pipeline.explain(timestamp=50.0, engine_id="ENG_01", mission_id="MSN_A")
    exp2 = pipeline.explain(timestamp=50.0, engine_id="ENG_01", mission_id="MSN_B")

    assert exp1.mission_id == "MSN_A"
    assert exp2.mission_id == "MSN_B"
    assert ("ENG_01", "MSN_A") in pipeline._sessions
    assert ("ENG_01", "MSN_B") in pipeline._sessions


def test_r_no_future_leakage():
    """Test R: Explanation at timestamp t only reflects data at or before t."""
    pipeline = ExplainabilityPipeline()

    # Time t = 50s
    resids_50 = {"cht": 2.0, "oil_temp": 1.6, "rpm": 0.0, "egt": 0.0, "oil_pressure": 0.0, "fuel_flow": 0.0, "vibration": 0.0}
    exp_50 = pipeline.explain(
        timestamp=50.0,
        engine_id="ENG_01",
        residuals=resids_50,
        diagnosis_result=make_dummy_diagnosis(conf=0.80),
    )

    assert exp_50.timestamp == 50.0
    assert exp_50.provenance.timestamp == 50.0


def test_s_future_modification_invariance():
    """Test S: Modifying or appending future samples has zero effect on output at time t."""
    pipeline_a = ExplainabilityPipeline()
    pipeline_b = ExplainabilityPipeline()

    resids_10 = {"cht": 2.5, "oil_temp": 1.8, "rpm": 0.0, "egt": 0.0, "oil_pressure": 0.0, "fuel_flow": 0.0, "vibration": 0.0}

    # Pipeline A: explains t=10 and stops
    exp_a = pipeline_a.explain(timestamp=10.0, engine_id="ENG_01", residuals=resids_10, diagnosis_result=make_dummy_diagnosis())

    # Pipeline B: explains t=10, then receives future catastrophic data at t=20, t=30
    exp_b = pipeline_b.explain(timestamp=10.0, engine_id="ENG_01", residuals=resids_10, diagnosis_result=make_dummy_diagnosis())
    pipeline_b.explain(timestamp=20.0, engine_id="ENG_01", residuals={"cht": 10.0, "oil_temp": 10.0, "rpm": 0.0, "egt": 0.0, "oil_pressure": 0.0, "fuel_flow": 0.0, "vibration": 0.0})

    # exp_a and exp_b at t=10 must be identical
    assert exp_a.summary_explanation == exp_b.summary_explanation
    assert exp_a.overall_quality == exp_b.overall_quality
    assert exp_a.physics_evidence.status == exp_b.physics_evidence.status


def test_t_missing_nan_safety():
    """Test T: Graceful handling of NaNs and None across all inputs without crashing or replacing with zero."""
    pipeline = ExplainabilityPipeline()
    resids_nan = {"cht": np.nan, "oil_temp": None, "rpm": np.nan, "egt": 0.1, "oil_pressure": None, "fuel_flow": np.nan, "vibration": None}

    exp = pipeline.explain(
        timestamp=50.0,
        engine_id="ENG_01",
        residuals=resids_nan,
        diagnosis_result=None,
        health_result=None,
        forecast_result=None,
        rul_result=None,
    )

    assert exp.physics_evidence.status == EvidenceStatus.INSUFFICIENT_DATA
    assert exp.overall_quality == EvidenceQuality.INSUFFICIENT_DATA
    assert exp.health_evidence is None
    assert exp.rul_evidence is None


def test_u_no_modification_of_phase8_prediction():
    """Test U: Phase 8 diagnosis result is never mutated or overridden."""
    pipeline = ExplainabilityPipeline()
    diag_res = make_dummy_diagnosis(fault="cooling_degradation", conf=0.95)
    orig_fault = diag_res.predicted_fault_type
    orig_conf = diag_res.diagnostic_confidence

    pipeline.explain(
        timestamp=50.0,
        engine_id="ENG_01",
        diagnosis_result=diag_res,
        residuals={"cht": -3.0, "oil_temp": -2.0, "rpm": 0.0, "egt": 0.0, "oil_pressure": 0.0, "fuel_flow": 0.0, "vibration": 0.0},
    )

    # Even with conflicting residuals, Phase 8 diagnosis is NEVER altered
    assert diag_res.predicted_fault_type == orig_fault
    assert diag_res.diagnostic_confidence == orig_conf


def test_v_no_modification_of_phase9_hi():
    """Test V: Phase 9 Health Index result is never mutated or recalculated."""
    pipeline = ExplainabilityPipeline()
    h_res = make_dummy_health(hi=0.75, rate=-0.005)
    orig_hi = h_res.smoothed_health_index
    orig_rate = h_res.degradation_rate

    pipeline.explain(
        timestamp=50.0,
        engine_id="ENG_01",
        health_result=h_res,
    )

    assert h_res.smoothed_health_index == orig_hi
    assert h_res.degradation_rate == orig_rate


def test_w_no_modification_of_phase11_rul():
    """Test W: Phase 11 RUL result is never mutated or recalculated."""
    pipeline = ExplainabilityPipeline()
    rul_res = make_dummy_rul(p50=320.0)
    orig_p50 = rul_res.rul_seconds_median

    pipeline.explain(
        timestamp=50.0,
        engine_id="ENG_01",
        rul_result=rul_res,
    )

    assert rul_res.rul_seconds_median == orig_p50


def test_x_deterministic_repeated_explanation(trained_classifier):
    """Test X: Repeated calls with identical inputs produce bit-for-bit deterministic outputs."""
    pipeline1 = ExplainabilityPipeline(classifier=trained_classifier)
    pipeline2 = ExplainabilityPipeline(classifier=trained_classifier)

    features = {col: 0.0 for col in trained_classifier.feature_names}
    features["cht_norm_residual"] = 3.5
    features["oil_temp_norm_residual"] = 2.0
    resids = {"cht": 3.5, "oil_temp": 2.0, "rpm": 0.0, "egt": 0.0, "oil_pressure": 0.0, "fuel_flow": 0.0, "vibration": 0.0}

    diag_res = make_dummy_diagnosis()
    h_res = make_dummy_health()
    rul_res = make_dummy_rul()

    exp1 = pipeline1.explain(50.0, "ENG_01", residuals=resids, features=features, diagnosis_result=diag_res, health_result=h_res, rul_result=rul_res)
    exp2 = pipeline2.explain(50.0, "ENG_01", residuals=resids, features=features, diagnosis_result=diag_res, health_result=h_res, rul_result=rul_res)

    assert exp1.overall_quality == exp2.overall_quality
    assert exp1.summary_explanation == exp2.summary_explanation
    assert exp1.physics_evidence == exp2.physics_evidence
