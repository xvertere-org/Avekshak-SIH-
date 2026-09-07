"""
Mission What-If Analysis Module for Phase 14 — SIH26054 / AVEKSHAK Digital Twin.

Strict architectural rules:
- Projected RUL MUST NOT be independently calculated by Phase 14.
  Both baseline and what-if scenarios execute through the existing authoritative pathway:
  Simulation -> Phase 13 -> Phase 11 RUL (including TimesFM handoff/fallback rules).
- No second health, RUL, or anomaly implementation.
- Clearly separates:
  1. Mission-condition what-if (altitude, ambient temp, throttle/load, duration)
  2. Simulated fault-stress scenario (fault type, severity)
- Clearly labels fault-based comparisons as simulated stress scenarios (inference never receives ground-truth labels).
- Emits explicit simulated projection narratives.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np

from orchestrator.schema import SimulationScenario, DashboardStatePayload, ScenarioFaultType, AdvisoryActionCode
from orchestrator.pipeline import SystemPipelineOrchestrator
from phase14.schema import WhatIfComparisonResult


class WhatIfAnalyzer:
    """
    Executes comparative mission what-if evaluation through the authoritative Phase 13 pipeline.
    Produces strictly simulated projections comparing baseline vs modified scenario trajectories.
    """

    @classmethod
    def run_comparison(
        cls,
        orchestrator: SystemPipelineOrchestrator,
        baseline_scenario: SimulationScenario,
        whatif_scenario: SimulationScenario,
    ) -> WhatIfComparisonResult:
        """
        Execute both baseline and what-if scenarios through the authoritative Phase 13 pipeline,
        preserving mission isolation and extracting non-fabricated trajectory deltas.
        """
        # Execute Baseline run
        baseline_payloads = orchestrator.run_simulation(scenario=baseline_scenario)

        # Execute What-If run (orchestrator.reset is automatically called inside run_simulation)
        whatif_payloads = orchestrator.run_simulation(scenario=whatif_scenario)

        return cls.compare_payloads(
            baseline_scenario=baseline_scenario,
            whatif_scenario=whatif_scenario,
            baseline_payloads=baseline_payloads,
            whatif_payloads=whatif_payloads,
        )

    @classmethod
    def compare_payloads(
        cls,
        baseline_scenario: SimulationScenario,
        whatif_scenario: SimulationScenario,
        baseline_payloads: List[DashboardStatePayload],
        whatif_payloads: List[DashboardStatePayload],
    ) -> WhatIfComparisonResult:
        """
        Compare two already-computed payload streams from Phase 13 and construct WhatIfComparisonResult.
        """
        if not baseline_payloads or not whatif_payloads:
            raise ValueError("Both baseline and what-if payload streams must be non-empty.")

        b_last = baseline_payloads[-1]
        w_last = whatif_payloads[-1]

        # 1. Health Index comparison (from authoritative Phase 9 outputs)
        b_hi = b_last.smoothed_health_index if b_last.smoothed_health_index is not None else float("nan")
        w_hi = w_last.smoothed_health_index if w_last.smoothed_health_index is not None else float("nan")
        delta_hi = (w_hi - b_hi) if (not np.isnan(b_hi) and not np.isnan(w_hi)) else float("nan")

        # 2. Projected RUL comparison (from authoritative Phase 11 / Phase 13 outputs)
        # Strictly extracted from authoritative payload, NEVER independently calculated
        b_rul = b_last.point_rul_seconds
        w_rul = w_last.point_rul_seconds
        delta_rul = (w_rul - b_rul) if (b_rul is not None and w_rul is not None) else None

        # 3. Peak CHT comparison
        b_chts = [p.observed_telemetry.get("cht", float("nan")) for p in baseline_payloads if "cht" in p.observed_telemetry]
        w_chts = [p.observed_telemetry.get("cht", float("nan")) for p in whatif_payloads if "cht" in p.observed_telemetry]
        b_peak_cht = float(np.nanmax(b_chts)) if b_chts else float("nan")
        w_peak_cht = float(np.nanmax(w_chts)) if w_chts else float("nan")
        delta_cht = (w_peak_cht - b_peak_cht) if (not np.isnan(b_peak_cht) and not np.isnan(w_peak_cht)) else float("nan")

        # 4. Advisory comparison
        def map_advisory(payload: DashboardStatePayload) -> str:
            adv = payload.advisory
            if adv is None:
                return "GO"
            code = adv.action_code.value if hasattr(adv.action_code, "value") else str(adv.action_code)
            if code in (AdvisoryActionCode.CRITICAL_ABORT_ACTION.value, AdvisoryActionCode.MAINTENANCE_INSPECTION.value):
                return "MAINTENANCE"
            elif code == AdvisoryActionCode.ADVISORY_CAUTION.value:
                return "CAUTION"
            elif code == AdvisoryActionCode.INSUFFICIENT_DATA.value:
                return "INSUFFICIENT_DATA"
            return "GO"

        b_adv = map_advisory(b_last)
        w_adv = map_advisory(w_last)

        # 5. Limiting factor & Anomaly count
        b_limit = b_last.limiting_factor or "NONE"
        w_limit = w_last.limiting_factor or "NONE"

        b_anom_cnt = sum(1 for p in baseline_payloads if p.anomaly_status in ("WARNING", "ANOMALY"))
        w_anom_cnt = sum(1 for p in whatif_payloads if p.anomaly_status in ("WARNING", "ANOMALY"))

        # 6. Synthesize narrative headline
        headline_parts = []
        if not np.isnan(delta_cht) and abs(delta_cht) >= 2.0:
            if delta_cht > 0:
                headline_parts.append(f"increases simulated thermal stress (+{delta_cht:.1f} °C peak CHT)")
            else:
                headline_parts.append(f"reduces thermal stress ({delta_cht:.1f} °C peak CHT)")

        if not np.isnan(delta_hi) and abs(delta_hi) >= 0.02:
            if delta_hi < 0:
                headline_parts.append(f"accelerates degradation (HI delta {delta_hi:+.3f})")
            else:
                headline_parts.append(f"improves health margin (HI delta {delta_hi:+.3f})")

        if delta_rul is not None and abs(delta_rul) >= 10.0:
            if delta_rul < 0:
                headline_parts.append(f"reduces projected endurance ({delta_rul:.0f} s RUL)")
            else:
                headline_parts.append(f"extends projected endurance (+{delta_rul:.0f} s RUL)")

        if b_adv != w_adv:
            headline_parts.append(f"shifts advisory from {b_adv} to {w_adv}")

        if headline_parts:
            summary_headline = "What-if mission trajectory " + ", and ".join(headline_parts) + "."
        else:
            summary_headline = "What-if mission profile maintains parity with baseline trajectory."

        narrative = (
            f"Simulated projection: Under what-if mission conditions "
            f"(Altitude: {whatif_scenario.altitude_m:.0f} m, OAT: {whatif_scenario.ambient_temp_c:.1f} °C, "
            f"Throttle: {whatif_scenario.throttle_pct:.0f}%), the simulated digital twin reveals "
            f"{'higher' if delta_cht > 0 else 'lower' if delta_cht < 0 else 'equivalent'} operating temperatures "
            f"and an advisory posture of {w_adv}."
        )

        # 7. Detailed Telemetry Extremes Comparison Table
        channels = ["rpm", "cht", "egt", "oil_temp", "oil_pressure", "vibration", "fuel_flow"]
        telemetry_comp: Dict[str, Dict[str, float]] = {}
        for ch in channels:
            b_vals = [p.observed_telemetry.get(ch, float("nan")) for p in baseline_payloads if ch in p.observed_telemetry]
            w_vals = [p.observed_telemetry.get(ch, float("nan")) for p in whatif_payloads if ch in p.observed_telemetry]
            b_max = float(np.nanmax(b_vals)) if b_vals else float("nan")
            w_max = float(np.nanmax(w_vals)) if w_vals else float("nan")
            b_mean = float(np.nanmean(b_vals)) if b_vals else float("nan")
            w_mean = float(np.nanmean(w_vals)) if w_vals else float("nan")
            telemetry_comp[ch] = {
                "baseline_max": round(b_max, 2) if not np.isnan(b_max) else None,
                "whatif_max": round(w_max, 2) if not np.isnan(w_max) else None,
                "delta_max": round(w_max - b_max, 2) if (not np.isnan(b_max) and not np.isnan(w_max)) else None,
                "baseline_mean": round(b_mean, 2) if not np.isnan(b_mean) else None,
                "whatif_mean": round(w_mean, 2) if not np.isnan(w_mean) else None,
            }

        prov = {
            "baseline_seed": baseline_scenario.seed,
            "whatif_seed": whatif_scenario.seed,
            "baseline_engine_id": baseline_scenario.engine_id,
            "whatif_engine_id": whatif_scenario.engine_id,
            "pipeline": "Phase 13 Unified System Pipeline Orchestrator",
            "rul_pathway": "Simulation -> Phase 13 -> Phase 11 RUL (zero second implementation)",
        }

        return WhatIfComparisonResult(
            baseline_scenario=baseline_scenario,
            whatif_scenario=whatif_scenario,
            baseline_payloads=baseline_payloads,
            whatif_payloads=whatif_payloads,
            comparison_summary_headline=summary_headline,
            simulated_projection_narrative=narrative,
            baseline_health_index=b_hi,
            whatif_health_index=w_hi,
            delta_health_index=delta_hi,
            baseline_rul_seconds=b_rul,
            whatif_rul_seconds=w_rul,
            delta_rul_seconds=delta_rul,
            baseline_peak_cht=b_peak_cht,
            whatif_peak_cht=w_peak_cht,
            delta_peak_cht=delta_cht,
            baseline_advisory_assessment=b_adv,
            whatif_advisory_assessment=w_adv,
            baseline_limiting_factor=b_limit,
            whatif_limiting_factor=w_limit,
            baseline_anomaly_count=b_anom_cnt,
            whatif_anomaly_count=w_anom_cnt,
            telemetry_comparison=telemetry_comp,
            provenance=prov,
        )
