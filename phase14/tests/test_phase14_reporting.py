"""
Tests for Phase 14 Engineering Mission Reporting Module.
Verifies report generation, telemetry statistics, PHM summaries, decision support wording,
and strict safety claim boundaries.
"""

import json
import pytest
import numpy as np

from orchestrator.schema import (
    DashboardStatePayload,
    OperatorAdvisory,
    AdvisoryActionCode,
    OrchestratorConfig,
    SimulationScenario,
    ScenarioFaultType,
)
from orchestrator.pipeline import SystemPipelineOrchestrator
from phase14.reporting import MissionReportGenerator
from phase14.schema import MissionReportSummary


def _build_synthetic_mission_payloads() -> list:
    payloads = []
    for t in range(5):
        adv = OperatorAdvisory(
            action_code=AdvisoryActionCode.NORMAL_MONITORING if t < 3 else AdvisoryActionCode.ADVISORY_CAUTION,
            headline="Nominal Performance" if t < 3 else "Thermal Margin Warning",
            recommended_action="Continue nominal monitoring." if t < 3 else "Reduce engine load to 65% throttle.",
            urgency="LOW" if t < 3 else "MEDIUM",
        )
        p = DashboardStatePayload(
            engine_id="UAV_AERO_01",
            mission_id="MIS_SURV_01",
            timestamp=float(t),
            mission_phase="CRUISE",
            simulation_mode="SYNTHETIC_SIMULATION",
            observed_telemetry={
                "rpm": 4500.0 + t * 50.0,
                "cht": 115.0 + t * 5.0,
                "egt": 680.0 + t * 10.0,
                "oil_temp": 82.0 + t * 2.0,
                "oil_pressure": 4.5 - t * 0.2,
                "vibration": 0.35 + t * 0.05,
                "fuel_flow": 16.0 + t * 1.0,
            },
            anomaly_status="NORMAL" if t < 3 else "WARNING",
            anomaly_contributing_channels=[] if t < 3 else ["cht"],
            predicted_fault_class="none" if t < 3 else "cooling_degradation",
            diagnosis_probabilities={"none": 0.9 if t < 3 else 0.1, "cooling_degradation": 0.1 if t < 3 else 0.88},
            diagnostic_confidence=0.9 if t < 3 else 0.88,
            diagnosis_data_quality="VALID",
            smoothed_health_index=1.0 - t * 0.05,
            degradation_trend="STABLE" if t < 3 else "DEGRADING",
            point_rul_seconds=1200.0 - t * 100.0,
            rul_uncertainty_p05=1000.0 - t * 100.0,
            rul_uncertainty_p95=1400.0 - t * 100.0,
            limiting_factor="THERMAL_MARGIN",
            forecast_status="TIMESFM_ACTIVE",
            forecast_source="TIMESFM_BASELINE",
            advisory=adv,
            fused_evidence={"evidence_quality": "HIGH"},
        )
        payloads.append(p)
    return payloads


def test_mission_report_generation():
    """Verify MissionReportGenerator aggregates telemetry peaks and PHM state."""
    payloads = _build_synthetic_mission_payloads()
    report = MissionReportGenerator.generate_report(payloads, scenario_metadata={"scenario_name": "thermal_stress_eval"})

    assert isinstance(report, MissionReportSummary)
    assert report.engine_id == "UAV_AERO_01"
    assert report.mission_id == "MIS_SURV_01"
    assert report.duration_s == 4.0
    assert report.scenario_name == "thermal_stress_eval"

    # Telemetry peaks
    assert report.rpm_max == 4700.0
    assert report.rpm_min == 4500.0
    assert report.cht_peak == 135.0
    assert report.oil_pressure_min == pytest.approx(3.7, 0.01)
    assert report.vibration_max == pytest.approx(0.55, 0.01)

    # PHM summary
    assert report.anomaly_events_count == 2
    assert "cht" in report.dominant_anomaly_channels
    assert report.final_diagnosis == "cooling_degradation"
    assert report.final_diagnosis_probability == pytest.approx(0.88, 0.01)
    assert report.evidence_quality == "HIGH"
    assert report.final_health_index == pytest.approx(0.80, 0.01)
    assert report.final_rul_seconds == pytest.approx(800.0, 0.01)
    assert report.limiting_factor == "THERMAL_MARGIN"

    # Advisory decision support
    assert report.advisory_assessment == "CAUTION"
    assert report.advisory_urgency == "MEDIUM"


def test_mission_report_serialization():
    """Verify Markdown and JSON export formats."""
    payloads = _build_synthetic_mission_payloads()
    report = MissionReportGenerator.generate_report(payloads)

    # JSON export
    json_str = report.to_json()
    data = json.loads(json_str)
    assert data["mission"]["engine_id"] == "UAV_AERO_01"
    assert data["phm_summary"]["final_diagnosis"] == "cooling_degradation"
    assert data["decision_support"]["advisory_assessment"] == "CAUTION"

    # Markdown export
    md_str = report.to_markdown()
    assert "# MISSION ENGINEERING REPORT" in md_str
    assert "UAV_AERO_01" in md_str
    assert "cooling_degradation" in md_str
    assert "CAUTION" in md_str


def test_safety_claim_boundaries():
    """Verify that reports strictly include decision-support disclaimers and omit certified claims."""
    payloads = _build_synthetic_mission_payloads()
    report = MissionReportGenerator.generate_report(payloads)

    disclaimer = report.disclaimer
    assert "Advisory decision-support summary" in disclaimer
    assert "Not certified airworthiness limits" in disclaimer

    md_str = report.to_markdown()
    # Required decision-support phrases
    assert "Decision Support Boundary Notice" in md_str
    assert "Non-Certification Clause" in md_str

    # Must NOT claim certified safety or flight validation
    lower_md = md_str.lower()
    assert "certified flight clearance" not in lower_md
    assert "oem flight certification" not in lower_md
    assert "guaranteed mission outcome" not in lower_md


def test_empty_payload_error():
    """Verify ValueError is raised on empty payload."""
    with pytest.raises(ValueError, match="empty payload sequence"):
        MissionReportGenerator.generate_report([])
