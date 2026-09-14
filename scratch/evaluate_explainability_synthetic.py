"""
Synthetic project scenario validation script for Phase 12: Explainability & Evidence Fusion.
Runs explainability across 6 canonical engine scenarios and reports evidence classifications.
"""

import sys
import os
sys.path.insert(0, os.path.abspath("."))

import numpy as np
import pandas as pd
from typing import Dict, List, Any

from simulator.fault_interface import FaultType
from fault_diagnosis.schema import FaultDiagnosisResult, CANONICAL_FAULT_LABELS
from fault_diagnosis.classifier import XGBoostFaultClassifier, FaultClassifierConfig
from health_index.schema import HealthIndexResult, HealthState, DegradationTrend, HealthDataQuality
from prognostics.schema import RULResult, RULStatus

from explainability.schema import EvidenceQuality, EvidenceStatus
from explainability.pipeline import ExplainabilityPipeline


def build_scenario_classifier() -> XGBoostFaultClassifier:
    """Train a fast classifier on synthetic multi-class feature patterns."""
    cfg = FaultClassifierConfig(n_estimators=15, max_depth=3, random_state=42)
    clf = XGBoostFaultClassifier(config=cfg)

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
    for label in CANONICAL_FAULT_LABELS:
        for rep in range(4):
            row = {col: 0.0 for col in feature_cols}
            row["phase_cruise"] = 1.0
            row["throttle"] = 75.0
            row["altitude"] = 1000.0
            row["load"] = 75.0

            if label == "cooling_degradation":
                row["cht_norm_residual"] = 3.5 + rep * 0.2
                row["oil_temp_norm_residual"] = 2.0 + rep * 0.2
            elif label == "lubrication_degradation":
                row["oil_pressure_norm_residual"] = -3.5 - rep * 0.2
                row["oil_temp_norm_residual"] = 2.5 + rep * 0.2
            elif label == "fuel_injection_abnormality":
                row["egt_norm_residual"] = 3.0 + rep * 0.2
                row["fuel_flow_norm_residual"] = -2.5 - rep * 0.2
            elif label == "mechanical_degradation":
                row["vibration_norm_residual"] = 4.0 + rep * 0.2
            elif label == "sensor_fault":
                row["cht_norm_residual"] = 5.5 + rep * 0.2
            else:  # none
                pass

            records.append(row)
            labels.append(label)

    clf.fit(pd.DataFrame(records), pd.Series(labels))
    return clf


def main():
    print("=" * 80)
    print("PHASE 12: EXPLAINABILITY & EVIDENCE FUSION SYNTHETIC SCENARIO VALIDATION")
    print("=" * 80)

    clf = build_scenario_classifier()
    pipeline = ExplainabilityPipeline(classifier=clf)

    scenarios = [
        {
            "name": "Healthy Nominal Cruise",
            "diagnosed": "none",
            "conf": 0.98,
            "residuals": {"rpm": 0.1, "cht": 0.2, "egt": 0.1, "oil_temp": 0.2, "oil_pressure": 0.0, "fuel_flow": 0.1, "vibration": 0.05},
            "hi": 0.96, "dom": [], "excluded": [], "rate": 0.0, "trend": "STABLE",
            "rul": None, "rul_status": RULStatus.NOT_DEGRADING,
        },
        {
            "name": "Cooling Degradation (Thermal Overheat)",
            "diagnosed": "cooling_degradation",
            "conf": 0.94,
            "residuals": {"rpm": 0.1, "cht": 3.8, "egt": 0.2, "oil_temp": 2.4, "oil_pressure": -0.2, "fuel_flow": 0.1, "vibration": 0.1},
            "hi": 0.65, "dom": ["cht", "oil_temp"], "excluded": [], "rate": -0.003, "trend": "DEGRADING",
            "rul": 280.0, "rul_status": RULStatus.ACTIVE_DEGRADATION, "limiting": "GLOBAL_HEALTH_INDEX",
        },
        {
            "name": "Lubrication Degradation (Oil Pressure Loss)",
            "diagnosed": "lubrication_degradation",
            "conf": 0.96,
            "residuals": {"rpm": -0.2, "cht": 0.3, "egt": 0.1, "oil_temp": 2.8, "oil_pressure": -3.8, "fuel_flow": 0.0, "vibration": 0.1},
            "hi": 0.58, "dom": ["oil_pressure", "oil_temp"], "excluded": [], "rate": -0.004, "trend": "DEGRADING",
            "rul": 190.0, "rul_status": RULStatus.ACTIVE_DEGRADATION, "limiting": "REDLINE_OIL_TEMP",
        },
        {
            "name": "Fuel Injection Abnormality (Lean Burn)",
            "diagnosed": "fuel_injection_abnormality",
            "conf": 0.91,
            "residuals": {"rpm": 0.1, "cht": 0.2, "egt": 3.4, "oil_temp": 0.1, "oil_pressure": 0.0, "fuel_flow": -2.8, "vibration": 0.05},
            "hi": 0.72, "dom": ["egt", "fuel_flow"], "excluded": [], "rate": -0.002, "trend": "DEGRADING",
            "rul": 420.0, "rul_status": RULStatus.ACTIVE_DEGRADATION, "limiting": "GLOBAL_HEALTH_INDEX",
        },
        {
            "name": "Mechanical Degradation (Severe Structural Vibration)",
            "diagnosed": "mechanical_degradation",
            "conf": 0.95,
            "residuals": {"rpm": 0.2, "cht": 0.1, "egt": 0.0, "oil_temp": 0.1, "oil_pressure": 0.0, "fuel_flow": 0.0, "vibration": 4.2},
            "hi": 0.62, "dom": ["vibration"], "excluded": [], "rate": -0.003, "trend": "DEGRADING",
            "rul": 150.0, "rul_status": RULStatus.ACTIVE_DEGRADATION, "limiting": "REDLINE_VIBRATION",
        },
        {
            "name": "Sensor Fault (Isolated CHT Thermocouple Bias)",
            "diagnosed": "sensor_fault",
            "conf": 0.89,
            "residuals": {"rpm": 0.1, "cht": 5.5, "egt": 0.1, "oil_temp": 0.1, "oil_pressure": 0.0, "fuel_flow": 0.0, "vibration": 0.05},
            "hi": 0.91, "dom": [], "excluded": ["cht"], "rate": 0.0, "trend": "STABLE",
            "rul": None, "rul_status": RULStatus.DEGRADED_PROGNOSTIC, "limiting": "NONE",
        },
        {
            "name": "Conflicting Case (Cooling Diagnosed but CHT Depressed)",
            "diagnosed": "cooling_degradation",
            "conf": 0.75,
            "residuals": {"rpm": 0.0, "cht": -3.0, "egt": 0.0, "oil_temp": -2.0, "oil_pressure": 0.0, "fuel_flow": 0.0, "vibration": 0.0},
            "hi": 0.88, "dom": [], "excluded": [], "rate": 0.0, "trend": "STABLE",
            "rul": None, "rul_status": RULStatus.NOT_DEGRADING,
        },
        {
            "name": "Insufficient Data Guard (<4 valid channels)",
            "diagnosed": "cooling_degradation",
            "conf": 0.80,
            "residuals": {"rpm": 0.1, "cht": 3.0, "egt": 0.1},
            "hi": float("nan"), "dom": [], "excluded": [], "rate": float("nan"), "trend": "INSUFFICIENT_DATA",
            "rul": None, "rul_status": RULStatus.INSUFFICIENT_DATA,
        },
    ]

    results = []
    print("\n| Scenario | Diagnosed Fault | ML Conf | Physics Status | Overall Quality | Top SHAP Attribution | Consistency Summary |")
    print("|---|---|---|---|---|---|---|")

    for sc in scenarios:
        # Build features
        feats = {col: 0.0 for col in clf.feature_names}
        feats["phase_cruise"] = 1.0
        for ch, val in sc["residuals"].items():
            col = f"{ch}_norm_residual"
            if col in feats:
                feats[col] = val

        diag = FaultDiagnosisResult(
            timestamp=50.0,
            engine_id="ENG_01",
            mission_id="MSN_01",
            mission_phase="CRUISE",
            predicted_fault_type=sc["diagnosed"],
            class_probabilities={sc["diagnosed"]: sc["conf"]},
            diagnostic_confidence=sc["conf"],
        )

        h_res = HealthIndexResult(
            timestamp=50.0,
            engine_id="ENG_01",
            mission_id="MSN_01",
            mission_phase="CRUISE",
            raw_health_index=sc["hi"],
            smoothed_health_index=sc["hi"],
            health_state=HealthState.HEALTHY.value if sc["hi"] >= 0.85 else HealthState.DEGRADED.value,
            raw_degradation_score=1.0 - sc["hi"] if np.isfinite(sc["hi"]) else float("nan"),
            degradation_rate=sc["rate"],
            degradation_trend=sc["trend"],
            channel_contributions={},
            channel_degradation_evidence={},
            dominant_degraded_channels=sc["dom"],
            valid_channels=list(sc["residuals"].keys()),
            missing_channels=[],
            excluded_channels=sc["excluded"],
            effective_channel_weights={},
            data_quality=HealthDataQuality.VALID.value if np.isfinite(sc["hi"]) else HealthDataQuality.INSUFFICIENT_DATA.value,
        )

        r_res = None
        if sc.get("rul") is not None or sc.get("rul_status") is not None:
            r_res = RULResult(
                engine_id="ENG_01",
                mission_id="MSN_01",
                timestamp=50.0,
                status=sc["rul_status"],
                rul_seconds_median=sc.get("rul"),
                rul_seconds_p05=sc["rul"] - 30.0 if sc.get("rul") else None,
                rul_seconds_p95=sc["rul"] + 50.0 if sc.get("rul") else None,
                limiting_factor=sc.get("limiting", "NONE"),
                confidence_score=0.85,
                active_flight_phase="CRUISE",
                handoff_horizon_s=0.0,
                trajectory_type="ROBUST_LINEAR_PRIMARY",
            )

        exp = pipeline.explain(
            timestamp=50.0,
            engine_id="ENG_01",
            mission_id="MSN_01",
            residuals=sc["residuals"],
            features=feats,
            diagnosis_result=diag,
            health_result=h_res,
            rul_result=r_res,
        )

        top_feat = exp.shap_evidence.top_features[0].feature_name if exp.shap_evidence and exp.shap_evidence.top_features else "None"
        summary_one_line = exp.physics_evidence.consistency_reason.replace("\n", " ")[:60] + "..."

        print(f"| {sc['name']} | `{sc['diagnosed']}` | {sc['conf']*100:.0f}% | `{exp.physics_evidence.status.value}` | **`{exp.overall_quality.value}`** | `{top_feat}` | {summary_one_line} |")
        results.append(exp)

    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(f"Total Scenarios Evaluated: {len(scenarios)}")
    print(f"SUPPORTED Cases:           {sum(1 for r in results if r.physics_evidence.status == EvidenceStatus.SUPPORTED)}")
    print(f"CONFLICTING Cases:         {sum(1 for r in results if r.physics_evidence.status == EvidenceStatus.CONFLICTING)}")
    print(f"INSUFFICIENT_DATA Cases:   {sum(1 for r in results if r.physics_evidence.status == EvidenceStatus.INSUFFICIENT_DATA)}")
    print(f"Overall HIGH Quality:      {sum(1 for r in results if r.overall_quality == EvidenceQuality.HIGH)}")
    print(f"Overall LOW Quality:       {sum(1 for r in results if r.overall_quality == EvidenceQuality.LOW)}")
    print(f"Overall INSUFFICIENT_DATA: {sum(1 for r in results if r.overall_quality == EvidenceQuality.INSUFFICIENT_DATA)}")


if __name__ == "__main__":
    main()
