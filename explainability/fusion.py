"""
Evidence fusion and deterministic narrative generator for Phase 12 Explainability.
Combines ML attribution, physical consistency, health degradation, temporal trends, and RUL evidence.
NOTE: Never averages disparate evidence into a synthetic probability; produces categorical evidence quality.
"""

from typing import Optional, Dict, Any, List, Tuple
from explainability.schema import (
    EvidenceQuality,
    EvidenceStatus,
    SHAPEvidence,
    PhysicsEvidence,
    HealthEvidence,
    TemporalEvidence,
    RULEvidence,
)


class EvidenceFusionEngine:
    """
    Fuses multiple independent evidence streams into an authoritative categorical quality assessment
    and generates concise, deterministic, operator-readable explanation narratives.
    """

    def fuse(
        self,
        shap_evidence: Optional[SHAPEvidence],
        physics_evidence: PhysicsEvidence,
        health_evidence: Optional[HealthEvidence],
        temporal_evidence: Optional[TemporalEvidence],
        rul_evidence: Optional[RULEvidence],
    ) -> Tuple[EvidenceQuality, str]:
        """
        Synthesize categorical quality and deterministic narrative summary.

        Returns:
            Tuple of (EvidenceQuality, explanation_summary_string)
        """
        # 1. Determine Categorical Evidence Quality
        quality = self._determine_quality(shap_evidence, physics_evidence, health_evidence)

        # 2. Build Deterministic Explanation Narrative
        narrative = self._build_narrative(
            quality=quality,
            shap=shap_evidence,
            phys=physics_evidence,
            health=health_evidence,
            temp=temporal_evidence,
            rul=rul_evidence,
        )

        return quality, narrative

    def _determine_quality(
        self,
        shap: Optional[SHAPEvidence],
        phys: PhysicsEvidence,
        health: Optional[HealthEvidence],
    ) -> EvidenceQuality:
        # If physics or health data is insufficient
        if phys.status == EvidenceStatus.INSUFFICIENT_DATA:
            return EvidenceQuality.INSUFFICIENT_DATA

        if health is not None and getattr(health, "data_quality", None) == "INSUFFICIENT_DATA":
            return EvidenceQuality.INSUFFICIENT_DATA

        # If physical evidence is in active conflict with diagnosis
        if phys.status == EvidenceStatus.CONFLICTING:
            return EvidenceQuality.LOW

        # If physics is supported
        if phys.status == EvidenceStatus.SUPPORTED:
            # Check for concordance with SHAP attribution and Health evidence
            shap_aligns = False
            if shap is not None and shap.status == "AVAILABLE" and shap.top_features:
                top_names = [f.feature_name for f in shap.top_features[:3]]
                # Check if any supporting channel is in top SHAP features
                for supp in phys.supporting_channels:
                    if any(supp in feat for feat in top_names):
                        shap_aligns = True
                        break

            health_aligns = False
            if health is not None and health.status in {"AVAILABLE", "SENSOR_ISOLATED"}:
                dominant = health.dominant_degraded_channels
                for supp in phys.supporting_channels:
                    if supp in dominant:
                        health_aligns = True
                        break

            if shap_aligns or health_aligns:
                return EvidenceQuality.HIGH
            else:
                return EvidenceQuality.MEDIUM

        # If physics is partially supported
        if phys.status == EvidenceStatus.PARTIALLY_SUPPORTED:
            if shap is not None and shap.status == "AVAILABLE":
                return EvidenceQuality.MEDIUM
            return EvidenceQuality.LOW

        return EvidenceQuality.LOW

    def _build_narrative(
        self,
        quality: EvidenceQuality,
        shap: Optional[SHAPEvidence],
        phys: PhysicsEvidence,
        health: Optional[HealthEvidence],
        temp: Optional[TemporalEvidence],
        rul: Optional[RULEvidence],
    ) -> str:
        lines: List[str] = []

        # Diagnosis Header
        diag_fault = phys.diagnosed_fault.upper()
        lines.append(f"Diagnosis: {diag_fault}")

        # ML Attribution Section
        if shap is not None and shap.status == "AVAILABLE":
            conf_str = f"{shap.diagnostic_confidence * 100:.1f}%" if shap.diagnostic_confidence else "N/A"
            lines.append(f"ML Evidence (Confidence: {conf_str}):")
            if shap.top_features:
                for f in shap.top_features[:3]:
                    val_str = f"{f.feature_value:.2f}" if f.feature_value is not None else "N/A"
                    sign = "+" if f.shap_value > 0 else ""
                    lines.append(f"  - {f.feature_name}: {val_str} (SHAP attribution: {sign}{f.shap_value:.3f}, {f.direction})")
            else:
                lines.append("  - No significant local feature contributions identified.")
        elif shap is not None and shap.status == "MODEL_UNAVAILABLE":
            lines.append("ML Evidence: Phase 8 model uninitialized or unavailable.")
        else:
            lines.append("ML Evidence: Insufficient feature data for attribution.")

        # Physics Evidence Section
        lines.append(f"Physics Evidence: {phys.status.value}")
        if phys.observed_residual_directions:
            obs_items = [f"{ch.upper()} {dir_}" for ch, dir_ in phys.observed_residual_directions.items()]
            lines.append(f"  - Observed Signatures: {', '.join(obs_items)}")
        lines.append(f"  - Consistency Assessment: {phys.consistency_reason}")

        # Health Evidence Section
        if health is not None and health.current_health_index is not None:
            hi_str = f"{health.current_health_index:.3f}"
            state_str = health.health_state or "UNKNOWN"
            lines.append(f"Health Evidence (HI = {hi_str}, State = {state_str}):")
            if health.excluded_channels:
                lines.append(f"  - Isolated Sensor Channels: {', '.join(health.excluded_channels)}")
            if health.dominant_degraded_channels:
                lines.append(f"  - Dominant Degraded Channels: {', '.join(health.dominant_degraded_channels)}")
            else:
                lines.append("  - No dominant degraded channels (system within nominal operating limits).")
        else:
            lines.append("Health Evidence: Insufficient valid channels for health evaluation.")

        # Temporal & RUL Section
        if temp is not None and temp.trend_direction != "INDETERMINATE":
            rate_str = f"{temp.degradation_rate:.5f}/s" if temp.degradation_rate is not None else "N/A"
            lines.append(f"Temporal Evidence: {temp.trend_direction} (Rate: {rate_str}, Trend: {temp.degradation_trend})")

        if rul is not None and rul.rul_status is not None:
            if rul.rul_seconds_median is not None:
                rul_min = rul.rul_seconds_median / 60.0
                lim = rul.limiting_factor or "NONE"
                lines.append(f"Prognostic RUL: {rul.rul_seconds_median:.1f} s (~{rul_min:.1f} min) | Status: {rul.rul_status} | Limiting Factor: {lim}")
            else:
                lines.append(f"Prognostic RUL: Unavailable | Status: {rul.rul_status}")

        # Overall Synthesis Section
        lines.append(f"Overall Evidence Consistency: {quality.value}")
        lines.append(f"Summary: {phys.consistency_reason}")

        return "\n".join(lines)
