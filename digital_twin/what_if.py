"""
Phase 10 Mission Reliability & What-If Simulation: What-If Evaluator & Golden Scenarios.

Implements:
- WhatIfEvaluator: baseline-vs-scenario comparative delta analysis
- 12 Golden Scenarios catalog
- 6-Way Causal Decomposition experiment (Baseline, Alt, Hot-Day, Alt+Hot-Day, F3, Alt+Hot-Day+F3)

DISCLAIMER:
All outputs are MODEL SCENARIO RESULTS / SYNTHETIC.
These what-if analyses describe behavior within this reduced-order grey-box propulsion model.
They do NOT provide certified flight safety, fleet reliability, or airworthiness certification.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from simulator.fault_interface import FaultSchedule, FaultState, FaultType, FaultSubsystem
from simulator.config import SimulatorConfig
from digital_twin.mission_types import (
    MissionSpec,
    MissionResult,
    MissionMetrics,
    EnvironmentProfile,
    ControlProfile,
    ScenarioComparisonResult,
)
from digital_twin.mission_simulator import MissionSimulator


class WhatIfEvaluator:
    """
    Evaluator for comparing baseline missions against counterfactual what-if branches.
    """

    def __init__(self, simulator: Optional[MissionSimulator] = None):
        self.simulator = simulator or MissionSimulator()

    def compare(
        self,
        baseline: MissionResult,
        scenario: MissionResult,
    ) -> ScenarioComparisonResult:
        """
        Compute quantitative deltas between Baseline and Scenario outcomes.
        """
        b_m = baseline.metrics
        s_m = scenario.metrics

        delta_min_hi = round(s_m.min_hi - b_m.min_hi, 4)
        delta_mean_hi = round(s_m.mean_hi - b_m.mean_hi, 4)
        delta_final_hi = round(s_m.final_hi - b_m.final_hi, 4)

        delta_time_watch = round(s_m.time_below_watch_s - b_m.time_below_watch_s, 2)
        delta_time_degraded = round(s_m.time_below_degraded_s - b_m.time_below_degraded_s, 2)
        delta_time_critical = round(s_m.time_below_critical_s - b_m.time_below_critical_s, 2)

        # Handle RUL deltas safely with explicit None semantics
        delta_min_rul: Optional[float] = None
        if s_m.min_estimated_rul_h is not None and b_m.min_estimated_rul_h is not None:
            delta_min_rul = round(s_m.min_estimated_rul_h - b_m.min_estimated_rul_h, 2)

        delta_final_rul: Optional[float] = None
        if s_m.final_estimated_rul_h is not None and b_m.final_estimated_rul_h is not None:
            delta_final_rul = round(s_m.final_estimated_rul_h - b_m.final_estimated_rul_h, 2)

        delta_max_cht = round(s_m.max_cht_c - b_m.max_cht_c, 2)
        delta_max_egt = round(s_m.max_egt_c - b_m.max_egt_c, 2)
        delta_min_oil_p = round(s_m.min_oil_pressure_bar - b_m.min_oil_pressure_bar, 3)
        delta_max_vib = round(s_m.max_vibration_g - b_m.max_vibration_g, 3)

        delta_envelope = s_m.envelope_event_count - b_m.envelope_event_count
        delta_risk = round(s_m.risk_index.score - b_m.risk_index.score, 4)

        # Build qualitative model-scoped interpretation
        qualitative = self._generate_qualitative_summary(
            scenario_id=scenario.scenario_id,
            d_hi=delta_min_hi,
            d_cht=delta_max_cht,
            d_oil_p=delta_min_oil_p,
            d_vib=delta_max_vib,
            d_env=delta_envelope,
            d_risk=delta_risk,
        )

        return ScenarioComparisonResult(
            baseline_id=baseline.scenario_id,
            scenario_id=scenario.scenario_id,
            delta_min_hi=delta_min_hi,
            delta_mean_hi=delta_mean_hi,
            delta_final_hi=delta_final_hi,
            delta_time_below_watch_s=delta_time_watch,
            delta_time_below_degraded_s=delta_time_degraded,
            delta_time_below_critical_s=delta_time_critical,
            delta_min_rul_h=delta_min_rul,
            delta_final_rul_h=delta_final_rul,
            delta_max_cht_c=delta_max_cht,
            delta_max_egt_c=delta_max_egt,
            delta_min_oil_pressure_bar=delta_min_oil_p,
            delta_max_vibration_g=delta_max_vib,
            delta_envelope_events=delta_envelope,
            delta_risk_score=delta_risk,
            qualitative_interpretation=qualitative,
        )

    def _generate_qualitative_summary(
        self,
        scenario_id: str,
        d_hi: float,
        d_cht: float,
        d_oil_p: float,
        d_vib: float,
        d_env: int,
        d_risk: float,
    ) -> str:
        parts = []
        if d_hi < -0.05:
            parts.append(f"HI degraded by {abs(d_hi):.3f}")
        elif d_hi > 0.05:
            parts.append(f"HI improved by {d_hi:.3f}")

        if abs(d_cht) >= 2.0:
            direction = "increased" if d_cht > 0 else "decreased"
            parts.append(f"CHT {direction} by {abs(d_cht):.1f}°C")

        if d_oil_p <= -0.2:
            parts.append(f"oil pressure dropped by {abs(d_oil_p):.2f} bar")

        if d_vib >= 0.1:
            parts.append(f"vibration elevated by {d_vib:.2f} g")

        if d_env > 0:
            parts.append(f"{d_env} additional envelope excursion events")

        if not parts:
            parts.append("minimal physical deviation from baseline")

        summary = f"Within grey-box simulation, scenario '{scenario_id}' showed: " + "; ".join(parts) + "."
        summary += " Model scenario result; not an empirical fleet reliability finding."
        return summary


def get_golden_scenario_specs(duration_s: float = 300.0, dt_s: float = 1.0) -> Dict[str, MissionSpec]:
    """
    Build the canonical suite of 12 Golden Scenarios.

    Scenarios:
    1. NOMINAL_CRUISE: Standard day (1000m, 75% throttle)
    2. HIGH_ALTITUDE: 4500m synthetic model scenario
    3. HOT_DAY: ISA + 20 K temperature offset (1000m)
    4. HOT_DAY_HIGH_ALTITUDE: ISA + 20 K at 4500m synthetic
    5. HIGH_LOAD: 95% continuous throttle
    6. AGGRESSIVE_THROTTLE: Throttle cycling [60% to 95%]
    7. F1_INJECTOR: Cylinder 1 fuel injector abnormality at t in [100, 200]s
    8. F2_LUBRICATION: Lubrication leak at t in [100, 200]s
    9. F3_COOLING: Coolant pump degradation at t in [100, 200]s
    10. F4_MISFIRE: Cylinder 2 combustion misfire at t in [100, 200]s
    11. F5_MECHANICAL: Mechanical bearing degradation at t in [100, 200]s
    12. COMBINED_ENVIRONMENT_FAULT: 4500m synthetic + Hot Day (+20K) + F3 Cooling fault
    """
    seed = 42

    # 1. NOMINAL_CRUISE
    nominal_env = EnvironmentProfile(initial_altitude_m=1000.0, temp_offset_k=0.0)
    nominal_ctrl = ControlProfile(initial_throttle_pct=75.0)
    s1 = MissionSpec(
        mission_id="NOMINAL_CRUISE",
        duration_s=duration_s,
        dt_s=dt_s,
        environment=nominal_env,
        controls=nominal_ctrl,
        random_seed=seed,
        metadata={"scenario_type": "BASELINE", "provenance": "ISA Troposphere, Healthy Engine"},
    )

    # 2. HIGH_ALTITUDE (SYNTHETIC MODEL SCENARIO)
    high_alt_env = EnvironmentProfile(
        initial_altitude_m=4500.0,
        temp_offset_k=0.0,
        is_synthetic_high_altitude=True,
    )
    s2 = MissionSpec(
        mission_id="HIGH_ALTITUDE",
        duration_s=duration_s,
        dt_s=dt_s,
        environment=high_alt_env,
        controls=nominal_ctrl,
        random_seed=seed,
        metadata={"scenario_type": "ENVIRONMENT", "provenance": "SYNTHETIC MODEL SCENARIO (4500m)"},
    )

    # 3. HOT_DAY
    hot_day_env = EnvironmentProfile(initial_altitude_m=1000.0, temp_offset_k=20.0)
    s3 = MissionSpec(
        mission_id="HOT_DAY",
        duration_s=duration_s,
        dt_s=dt_s,
        environment=hot_day_env,
        controls=nominal_ctrl,
        random_seed=seed,
        metadata={"scenario_type": "ENVIRONMENT", "provenance": "ISA + 20K Temperature Offset"},
    )

    # 4. HOT_DAY_HIGH_ALTITUDE
    hot_alt_env = EnvironmentProfile(
        initial_altitude_m=4500.0,
        temp_offset_k=20.0,
        is_synthetic_high_altitude=True,
    )
    s4 = MissionSpec(
        mission_id="HOT_DAY_HIGH_ALTITUDE",
        duration_s=duration_s,
        dt_s=dt_s,
        environment=hot_alt_env,
        controls=nominal_ctrl,
        random_seed=seed,
        metadata={"scenario_type": "ENVIRONMENT", "provenance": "SYNTHETIC MODEL SCENARIO (4500m, ISA+20K)"},
    )

    # 5. HIGH_LOAD
    high_load_ctrl = ControlProfile(initial_throttle_pct=95.0)
    s5 = MissionSpec(
        mission_id="HIGH_LOAD",
        duration_s=duration_s,
        dt_s=dt_s,
        environment=nominal_env,
        controls=high_load_ctrl,
        random_seed=seed,
        metadata={"scenario_type": "LOAD", "provenance": "Sustained 95% High Throttle"},
    )

    # 6. AGGRESSIVE_THROTTLE
    def aggressive_throttle_sched(t: float) -> float:
        # Cycle between 60% and 95% every 40 seconds
        cycle_pos = (t % 40.0) / 40.0
        return 95.0 if cycle_pos < 0.5 else 60.0

    agg_ctrl = ControlProfile(
        initial_throttle_pct=75.0,
        throttle_schedule=aggressive_throttle_sched,
    )
    s6 = MissionSpec(
        mission_id="AGGRESSIVE_THROTTLE",
        duration_s=duration_s,
        dt_s=dt_s,
        environment=nominal_env,
        controls=agg_ctrl,
        random_seed=seed,
        metadata={"scenario_type": "DYNAMICS", "provenance": "Throttle Cycling 60-95%"},
    )

    # 7. F1_INJECTOR (Injector delivery abnormality on cylinder 1 at t in [100, 200]s)
    f1_sched = FaultSchedule()
    f1_sched.add_fault(
        FaultState(
            fault_type=FaultType.INJECTOR_DELIVERY_ABNORMALITY,
            severity=0.35,
            active=True,
            start_time=100.0,
            end_time=200.0,
            affected_subsystem=FaultSubsystem.FUEL,
            affected_cylinder=1,
            parameters={"mode": "lean"},
        )
    )
    s7 = MissionSpec(
        mission_id="F1_INJECTOR",
        duration_s=duration_s,
        dt_s=dt_s,
        environment=nominal_env,
        controls=nominal_ctrl,
        fault_schedule=f1_sched,
        random_seed=seed,
        metadata={"scenario_type": "FAULT_F1", "provenance": "Injector Abnormality Cyl 1 (t=100-200s)"},
    )

    # 8. F2_LUBRICATION (Lubrication degradation at t in [100, 200]s)
    f2_sched = FaultSchedule()
    f2_sched.add_fault(
        FaultState(
            fault_type=FaultType.LUBRICATION_DEGRADATION,
            severity=0.40,
            active=True,
            start_time=100.0,
            end_time=200.0,
            affected_subsystem=FaultSubsystem.LUBRICATION,
        )
    )
    s8 = MissionSpec(
        mission_id="F2_LUBRICATION",
        duration_s=duration_s,
        dt_s=dt_s,
        environment=nominal_env,
        controls=nominal_ctrl,
        fault_schedule=f2_sched,
        random_seed=seed,
        metadata={"scenario_type": "FAULT_F2", "provenance": "Lubrication Degradation (t=100-200s)"},
    )

    # 9. F3_COOLING (Coolant degradation at t in [100, 200]s)
    f3_sched = FaultSchedule()
    f3_sched.add_fault(
        FaultState(
            fault_type=FaultType.COOLING_DEGRADATION,
            severity=0.50,
            active=True,
            start_time=100.0,
            end_time=200.0,
            affected_subsystem=FaultSubsystem.COOLING,
        )
    )
    s9 = MissionSpec(
        mission_id="F3_COOLING",
        duration_s=duration_s,
        dt_s=dt_s,
        environment=nominal_env,
        controls=nominal_ctrl,
        fault_schedule=f3_sched,
        random_seed=seed,
        metadata={"scenario_type": "FAULT_F3", "provenance": "Cooling Pump Degradation (t=100-200s)"},
    )

    # 10. F4_MISFIRE (Combustion misfire on cylinder 2 at t in [100, 200]s)
    f4_sched = FaultSchedule()
    f4_sched.add_fault(
        FaultState(
            fault_type=FaultType.COMBUSTION_MISFIRE,
            severity=0.40,
            active=True,
            start_time=100.0,
            end_time=200.0,
            affected_subsystem=FaultSubsystem.COMBUSTION,
            affected_cylinder=2,
        )
    )
    s10 = MissionSpec(
        mission_id="F4_MISFIRE",
        duration_s=duration_s,
        dt_s=dt_s,
        environment=nominal_env,
        controls=nominal_ctrl,
        fault_schedule=f4_sched,
        random_seed=seed,
        metadata={"scenario_type": "FAULT_F4", "provenance": "Combustion Misfire Cyl 2 (t=100-200s)"},
    )

    # 11. F5_MECHANICAL (Mechanical bearing degradation at t in [100, 200]s)
    f5_sched = FaultSchedule()
    f5_sched.add_fault(
        FaultState(
            fault_type=FaultType.MECHANICAL_DEGRADATION,
            severity=0.45,
            active=True,
            start_time=100.0,
            end_time=200.0,
            affected_subsystem=FaultSubsystem.VIBRATION,
        )
    )
    s11 = MissionSpec(
        mission_id="F5_MECHANICAL",
        duration_s=duration_s,
        dt_s=dt_s,
        environment=nominal_env,
        controls=nominal_ctrl,
        fault_schedule=f5_sched,
        random_seed=seed,
        metadata={"scenario_type": "FAULT_F5", "provenance": "Mechanical Bearing Degradation (t=100-200s)"},
    )

    # 12. COMBINED_ENVIRONMENT_FAULT (Hot Day + 4500m synthetic + F3 Cooling at t in [100, 200]s)
    f_comb_sched = FaultSchedule()
    f_comb_sched.add_fault(
        FaultState(
            fault_type=FaultType.COOLING_DEGRADATION,
            severity=0.50,
            active=True,
            start_time=100.0,
            end_time=200.0,
            affected_subsystem=FaultSubsystem.COOLING,
        )
    )
    s12 = MissionSpec(
        mission_id="COMBINED_ENVIRONMENT_FAULT",
        duration_s=duration_s,
        dt_s=dt_s,
        environment=hot_alt_env,
        controls=nominal_ctrl,
        fault_schedule=f_comb_sched,
        random_seed=seed,
        metadata={
            "scenario_type": "COMBINED",
            "provenance": "SYNTHETIC MODEL SCENARIO (4500m, ISA+20K, F3 Cooling Fault)",
            "composition_rule": "Orthogonal physical composition: atmosphere affects density/charge air; fault scales coolant heat transfer.",
        },
    )

    return {
        "NOMINAL_CRUISE": s1,
        "HIGH_ALTITUDE": s2,
        "HOT_DAY": s3,
        "HOT_DAY_HIGH_ALTITUDE": s4,
        "HIGH_LOAD": s5,
        "AGGRESSIVE_THROTTLE": s6,
        "F1_INJECTOR": s7,
        "F2_LUBRICATION": s8,
        "F3_COOLING": s9,
        "F4_MISFIRE": s10,
        "F5_MECHANICAL": s11,
        "COMBINED_ENVIRONMENT_FAULT": s12,
    }


def run_causal_decomposition(
    simulator: Optional[MissionSimulator] = None,
    duration_s: float = 300.0,
    dt_s: float = 1.0,
) -> Dict[str, Any]:
    """
    Execute 6-way causal decomposition experiment:
    1. Baseline (1000m, normal ISA)
    2. Altitude only (4500m synthetic)
    3. Hot-Day only (ISA + 20K)
    4. Altitude + Hot-Day (4500m, ISA + 20K)
    5. F3 Cooling only (1000m, normal ISA, F3 fault)
    6. Altitude + Hot-Day + F3 Cooling (combined)

    Returns structured comparison results allowing distinction of environmental vs fault effects.
    """
    sim = simulator or MissionSimulator()
    evaluator = WhatIfEvaluator(simulator=sim)

    specs = get_golden_scenario_specs(duration_s=duration_s, dt_s=dt_s)

    # 1. Baseline
    res_base = sim.run_mission(specs["NOMINAL_CRUISE"], full_trajectory=False)
    # 2. Altitude only
    res_alt = sim.run_mission(specs["HIGH_ALTITUDE"], full_trajectory=False)
    # 3. Hot-Day only
    res_hot = sim.run_mission(specs["HOT_DAY"], full_trajectory=False)
    # 4. Altitude + Hot-Day
    res_hot_alt = sim.run_mission(specs["HOT_DAY_HIGH_ALTITUDE"], full_trajectory=False)
    # 5. F3 only
    res_f3 = sim.run_mission(specs["F3_COOLING"], full_trajectory=False)
    # 6. Combined
    res_comb = sim.run_mission(specs["COMBINED_ENVIRONMENT_FAULT"], full_trajectory=False)

    comp_alt = evaluator.compare(res_base, res_alt)
    comp_hot = evaluator.compare(res_base, res_hot)
    comp_hot_alt = evaluator.compare(res_base, res_hot_alt)
    comp_f3 = evaluator.compare(res_base, res_f3)
    comp_comb = evaluator.compare(res_base, res_comb)

    return {
        "baseline": res_base.metrics,
        "branches": {
            "altitude_only": {"metrics": res_alt.metrics, "delta": comp_alt},
            "hot_day_only": {"metrics": res_hot.metrics, "delta": comp_hot},
            "altitude_plus_hot_day": {"metrics": res_hot_alt.metrics, "delta": comp_hot_alt},
            "f3_cooling_only": {"metrics": res_f3.metrics, "delta": comp_f3},
            "combined_all": {"metrics": res_comb.metrics, "delta": comp_comb},
        },
        "non_linear_coupling_notes": (
            "Thermal response exhibits non-linear grey-box coupling: elevated ambient temperature "
            "reduces heat exchanger delta-T, exacerbating cooling pump degradation non-additively."
        ),
    }
