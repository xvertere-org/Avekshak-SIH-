"""
Phase 11 Engineering Explainability Engine: Deterministic Plain-Text & JSON Rendering.

Provides auditable, deterministic explanations derived exclusively from
structured evidence (StepEvidenceRecord, WhatIfScenarioDeltaEvidence).

Invariants:
- Downstream-only: extracts and renders existing structured evidence.
- Zero recomputation: never re-evaluates upstream algorithms or thresholds.
- Honest uncertainty: labels scores as engineering heuristics, never probabilities.
- Traceability: surfaces computational dependency traces.
- No ground-truth leakage: explanations derive only from observable evidence.
"""

from __future__ import annotations
import json
import math
from typing import Any, Dict, List, Optional, Tuple, Union

from digital_twin.evidence_types import (
    DataClassification,
    EpistemicProvenance,
    ChannelRole,
    ExplanationReasonCode,
    StepEvidenceRecord,
    WhatIfScenarioDeltaEvidence,
)


class ExplainabilityEngine:
    """
    Deterministic explainability renderer for Digital Twin step evidence and scenario deltas.
    """

    @staticmethod
    def render_human_readable(record: StepEvidenceRecord) -> str:
        """
        Render a deterministic, structured human-readable explanation from StepEvidenceRecord.
        Follows a strict format with sections for Engine Health, Primary Contributors,
        Supporting Evidence, Diagnosis, Data Quality, Interpretation, and Limitations.
        """
        lines: List[str] = []

        # 1. Engine Health
        hi_val_str = f"{record.hi_smooth:.4f}" if record.hi_smooth is not None and not math.isnan(record.hi_smooth) else "UNAVAILABLE"
        lines.append(f"ENGINE HEALTH: {hi_val_str} (State: {record.engine_health_state})")
        lines.append("")

        # 2. Primary Contributors
        lines.append("Primary contributors:")
        if record.subsystem_evidence:
            # Sort subsystems by health score ascending (most degraded first)
            sorted_subs = sorted(
                record.subsystem_evidence.values(),
                key=lambda s: (s.health_score if s.health_score is not None and not math.isnan(s.health_score) else 999.0),
            )
            for sub in sorted_subs:
                score_str = f"{sub.health_score:.4f}" if sub.health_score is not None and not math.isnan(sub.health_score) else "UNAVAILABLE"
                lines.append(f"  - {sub.subsystem.capitalize()} health: {score_str}")
        else:
            lines.append("  - No subsystem assessments available")
        lines.append("")

        # 3. Supporting Evidence
        lines.append("Supporting evidence:")
        ev_items = 0
        # Check notable normalized residuals (|z| >= 1.5)
        for ch_k, ch_ev in sorted(record.channel_evidence.items()):
            if ch_ev.normalized_residual is not None and abs(ch_ev.normalized_residual) >= 1.5:
                raw_str = f"{ch_ev.raw_residual:+.2f} {ch_ev.unit}" if ch_ev.raw_residual is not None else "N/A"
                z_str = f"{ch_ev.normalized_residual:+.2f}"
                lines.append(f"  - {ch_ev.channel.upper()} residual: {raw_str} (normalized z = {z_str})")
                ev_items += 1

        if record.diagnosis_evidence and record.diagnosis_evidence.affected_cylinder is not None:
            lines.append(f"  - Cylinder {record.diagnosis_evidence.affected_cylinder} localized via runner spread")
            ev_items += 1

        if ev_items == 0:
            lines.append("  - All observable channels operate within nominal residual thresholds (|z| < 1.5)")
        lines.append("")

        # 4. Diagnosis
        lines.append("Diagnosis:")
        if record.diagnosis_evidence:
            diag = record.diagnosis_evidence
            lines.append(f"  Primary hypothesis: {diag.primary_fault}")
            lines.append(f"  Diagnostic confidence heuristic: {diag.confidence_heuristic:.4f} (Status: {diag.status})")
            if diag.competing_hypotheses_count > 0:
                lines.append(f"  Ambiguity note: {diag.competing_hypotheses_count} competing hypotheses exceed threshold")
        else:
            lines.append("  Primary hypothesis: UNKNOWN (Diagnostic assessment not active)")
        lines.append("")

        # 5. Data Quality
        lines.append("Data quality:")
        cov_pct = record.observability_coverage * 100.0
        lines.append(f"  Status: {record.data_quality_status} (Observability coverage: {cov_pct:.1f}%)")
        lines.append("")

        # 6. Interpretation (Derived strictly from structured reason codes)
        lines.append("Interpretation:")
        interpretations: List[str] = []
        codes = set(record.overall_reason_codes)

        if ExplanationReasonCode.NOMINAL_OPERATION in codes:
            interpretations.append("The engine is operating within nominal grey-box physical expectations.")
        if ExplanationReasonCode.FAULT_SIGNATURE_MATCH in codes:
            fault_name = record.diagnosis_evidence.primary_fault if record.diagnosis_evidence else "UNKNOWN"
            cyl_info = f" affecting cylinder {record.diagnosis_evidence.affected_cylinder}" if (record.diagnosis_evidence and record.diagnosis_evidence.affected_cylinder) else ""
            interpretations.append(f"The modeled telemetry signature is consistent with {fault_name}{cyl_info}.")
        if ExplanationReasonCode.SENSOR_LOCALIZATION_FAVORED in codes:
            interpretations.append("Divergence is localized to a single sensor channel without multi-channel physical coupling; evidence favors sensor anomaly.")
        if ExplanationReasonCode.INSUFFICIENT_DATA in codes or ExplanationReasonCode.SENSOR_DROPOUT in codes:
            interpretations.append("Telemetry quality or observability coverage is impaired; health outputs are gated to prevent false physical assertions.")
        if ExplanationReasonCode.RUL_TREND_SUPPORTED in codes and record.prognostic_evidence:
            rul_h = record.prognostic_evidence.rul_estimate_hours
            rul_str = f"{rul_h:.1f} hours" if rul_h is not None else "N/A"
            interpretations.append(f"Degradation trend is statistically significant; model-defined RUL projection is {rul_str} to horizon D_EOL.")
        elif ExplanationReasonCode.RUL_NON_DEGRADING in codes:
            interpretations.append("Degradation trend is non-degrading or stable; finite RUL extrapolation is not applicable.")

        if not interpretations:
            interpretations.append("Engine physical metrics reflect current mission operating point.")

        for interp in interpretations:
            lines.append(f"  {interp}")
        lines.append("")

        # 7. Limitations
        lines.append("Limitations:")
        for lim in record.limitations:
            lines.append(f"  - {lim}")

        return "\n".join(lines)

    @staticmethod
    def render_machine_readable(record: StepEvidenceRecord) -> Dict[str, Any]:
        """
        Generate complete, structured machine-readable dictionary representation.
        """
        return record.to_dict()

    @staticmethod
    def render_traceability_summary(record: StepEvidenceRecord) -> Dict[str, List[str]]:
        """
        Format traceability chains into clear, human-readable lineage summaries.
        """
        summaries: Dict[str, List[str]] = {}
        for chain_k, chain in sorted(record.traceability_chains.items()):
            steps: List[str] = []
            for n in chain.nodes:
                steps.append(f"[{n.stage}] {n.source_entity} -> {n.output_entity} ({n.relationship}: {n.value_summary})")
            summaries[chain_k] = steps
        return summaries

    @staticmethod
    def explain_what_if_delta(
        scenario_id: str,
        baseline_scenario_id: str,
        delta_metrics: Dict[str, Optional[float]],
        modeled_contributions: Dict[str, str],
        envelope_events_delta: int = 0,
        risk_index_delta: Optional[float] = None,
    ) -> WhatIfScenarioDeltaEvidence:
        """
        Construct structured What-If delta evidence between baseline and counterfactual scenarios.
        """
        reasons: List[ExplanationReasonCode] = []

        if "max_cht" in delta_metrics and delta_metrics["max_cht"] is not None and abs(delta_metrics["max_cht"]) > 2.0:
            reasons.append(ExplanationReasonCode.THERMAL_LIMIT_APPROACH)
        if "min_oil_pressure" in delta_metrics and delta_metrics["min_oil_pressure"] is not None and abs(delta_metrics["min_oil_pressure"]) > 0.3:
            reasons.append(ExplanationReasonCode.LUBRICATION_PRESSURE_DEVIATION)
        if "max_egt" in delta_metrics and delta_metrics["max_egt"] is not None and abs(delta_metrics["max_egt"]) > 20.0:
            reasons.append(ExplanationReasonCode.COMBUSTION_EGT_DEVIATION)
        if envelope_events_delta > 0:
            reasons.append(ExplanationReasonCode.MISSION_ENVELOPE_EVENT)
        if not reasons:
            reasons.append(ExplanationReasonCode.NOMINAL_OPERATION)

        return WhatIfScenarioDeltaEvidence(
            scenario_id=scenario_id,
            baseline_scenario_id=baseline_scenario_id,
            delta_metrics=delta_metrics,
            modeled_contributions=modeled_contributions,
            envelope_event_count_delta=envelope_events_delta,
            risk_index_delta=risk_index_delta,
            reason_codes=tuple(sorted(set(reasons), key=lambda x: x.value)),
            provenance=EpistemicProvenance.SYNTHETIC_VALIDATION,
        )

    @staticmethod
    def serialize_to_json(record: Union[StepEvidenceRecord, WhatIfScenarioDeltaEvidence], indent: Optional[int] = 2) -> str:
        """
        Deterministic JSON serialization with stable keys and standard float representation.
        """
        return json.dumps(record.to_dict(), sort_keys=True, indent=indent)
