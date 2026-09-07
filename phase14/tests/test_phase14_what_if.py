"""
Tests for Phase 14 Mission What-If Analysis Module.
Verifies comparative trajectory evaluation, baseline vs what-if delta detection,
zero ML leakage, and strict reuse of the Phase 13 RUL pathway.
"""

import pytest
import numpy as np

from orchestrator.schema import (
    DashboardStatePayload,
    OrchestratorConfig,
    SimulationScenario,
    ScenarioFaultType,
    AdvisoryActionCode,
    OperatorAdvisory,
)
from orchestrator.pipeline import SystemPipelineOrchestrator
from phase14.what_if import WhatIfAnalyzer
from phase14.schema import WhatIfComparisonResult


def test_what_if_comparison_payload_delta():
    """Verify delta computation between two pre-computed payload streams."""
    def make_stream(cht_val: float, hi_val: float, rul_val: float, adv_code: AdvisoryActionCode):
        adv = OperatorAdvisory(
            action_code=adv_code,
            headline="Test",
            recommended_action="Monitor",
            urgency="LOW" if adv_code == AdvisoryActionCode.NORMAL_MONITORING else "HIGH",
        )
        return [
            DashboardStatePayload(
                engine_id="ENG_01",
                mission_id="MIS_01",
                timestamp=0.0,
                mission_phase="CRUISE",
                observed_telemetry={"cht": cht_val, "rpm": 4500.0},
                smoothed_health_index=hi_val,
                point_rul_seconds=rul_val,
                limiting_factor="THERMAL" if cht_val > 130 else "NONE",
                anomaly_status="NORMAL" if adv_code == AdvisoryActionCode.NORMAL_MONITORING else "ANOMALY",
                advisory=adv,
            )
        ]

    b_sc = SimulationScenario(name="baseline", throttle_pct=75.0)
    w_sc = SimulationScenario(name="whatif_high_load", throttle_pct=90.0)

    b_payloads = make_stream(115.0, 0.98, 1200.0, AdvisoryActionCode.NORMAL_MONITORING)
    w_payloads = make_stream(138.0, 0.82, 600.0, AdvisoryActionCode.MAINTENANCE_INSPECTION)

    res = WhatIfAnalyzer.compare_payloads(
        baseline_scenario=b_sc,
        whatif_scenario=w_sc,
        baseline_payloads=b_payloads,
        whatif_payloads=w_payloads,
    )

    assert isinstance(res, WhatIfComparisonResult)
    assert res.baseline_peak_cht == 115.0
    assert res.whatif_peak_cht == 138.0
    assert res.delta_peak_cht == 23.0

    assert res.baseline_health_index == 0.98
    assert res.whatif_health_index == 0.82
    assert res.delta_health_index == pytest.approx(-0.16, 0.001)

    assert res.baseline_rul_seconds == 1200.0
    assert res.whatif_rul_seconds == 600.0
    assert res.delta_rul_seconds == -600.0

    assert res.baseline_advisory_assessment == "GO"
    assert res.whatif_advisory_assessment == "MAINTENANCE"

    # Narrative headline checks
    assert "changes the projected engine health state and advisory assessment" in res.comparison_summary_headline
    assert "Health Index:" in res.simulated_projection_narrative
    assert "Peak CHT:" in res.simulated_projection_narrative
    assert "Advisory Shift:" in res.simulated_projection_narrative


def test_what_if_end_to_end_pipeline_execution():
    """
    Execute end-to-end what-if comparison through Phase 13 orchestrator.
    Verify that mission-condition changes (throttle / duration) yield comparative differences
    without independent RUL recomputation.
    """
    cfg = OrchestratorConfig(auto_bootstrap_on_init=True, deterministic_seed=42)
    orch = SystemPipelineOrchestrator(config=cfg)

    b_sc = SimulationScenario(
        name="baseline_cruise",
        duration_s=10.0,
        throttle_pct=60.0,
        altitude_m=1500.0,
        ambient_temp_c=15.0,
        fault_type=ScenarioFaultType.HEALTHY,
        seed=42,
        engine_id="ENG_WI_BASE",
        mission_id="MIS_WI_BASE",
    )

    w_sc = SimulationScenario(
        name="whatif_hot_high",
        duration_s=10.0,
        throttle_pct=90.0,
        altitude_m=1500.0,
        ambient_temp_c=25.0,
        fault_type=ScenarioFaultType.HEALTHY,
        seed=42,
        engine_id="ENG_WI_WHATIF",
        mission_id="MIS_WI_WHATIF",
    )

    res = WhatIfAnalyzer.run_comparison(orch, b_sc, w_sc)

    assert res.baseline_scenario.throttle_pct == 60.0
    assert res.whatif_scenario.throttle_pct == 90.0

    # Due to higher throttle and warmer ambient temp, what-if CHT and EGT should be higher
    assert res.whatif_peak_cht >= res.baseline_peak_cht
    assert res.telemetry_comparison["egt"]["whatif_max"] > res.telemetry_comparison["egt"]["baseline_max"]
    assert res.telemetry_comparison["fuel_flow"]["whatif_max"] > res.telemetry_comparison["fuel_flow"]["baseline_max"]

    # Ensure scenario metadata isolation across runs
    assert res.baseline_scenario.mission_id == "MIS_WI_BASE"
    assert res.whatif_scenario.mission_id == "MIS_WI_WHATIF"
    assert res.baseline_payloads[-1].scenario_metadata["scenario_name"] == "baseline_cruise"
    assert res.whatif_payloads[-1].scenario_metadata["scenario_name"] == "whatif_hot_high"

    # Verify provenance & disclaimer
    assert "Simulated projection based on comparative synthetic simulation" in res.disclaimer
    assert "rul_pathway" in res.provenance


def test_what_if_simulated_fault_stress_scenario():
    """
    Verify simulated fault-stress scenario comparison (nominal vs injected fault stress).
    Inference operates with zero leakage (ground-truth fault metadata stripped).
    """
    cfg = OrchestratorConfig(auto_bootstrap_on_init=True, deterministic_seed=42)
    orch = SystemPipelineOrchestrator(config=cfg)

    b_sc = SimulationScenario(
        name="nominal_baseline",
        duration_s=10.0,
        fault_type=ScenarioFaultType.HEALTHY,
        seed=55,
        engine_id="ENG_STRESS",
        mission_id="MIS_NOMINAL",
    )

    w_sc = SimulationScenario(
        name="cooling_stress",
        duration_s=10.0,
        fault_type=ScenarioFaultType.COOLING_DEGRADATION,
        fault_start_s=4.0,
        fault_severity=0.7,
        seed=55,
        engine_id="ENG_STRESS",
        mission_id="MIS_FAULT_STRESS",
    )

    res = WhatIfAnalyzer.run_comparison(orch, b_sc, w_sc)

    # Fault-stress run should trigger elevated CHT and potential anomaly
    assert res.whatif_peak_cht > res.baseline_peak_cht
    assert res.telemetry_comparison["cht"]["delta_max"] > 0.0


def test_what_if_empty_payload_rejection():
    """Verify ValueError is raised if either payload stream is empty."""
    sc = SimulationScenario()
    p = [DashboardStatePayload(engine_id="E", mission_id="M", timestamp=0.0, mission_phase="C")]
    with pytest.raises(ValueError, match="must be non-empty"):
        WhatIfAnalyzer.compare_payloads(sc, sc, [], p)
    with pytest.raises(ValueError, match="must be non-empty"):
        WhatIfAnalyzer.compare_payloads(sc, sc, p, [])
