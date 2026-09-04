"""
Authoritative Explainability & Evidence Fusion Pipeline for Phase 12.
Coordinates SHAP local attribution, deterministic physics consistency checks, health index passthrough,
temporal evolution tracking, and RUL evidence fusion into a unified ExplainabilityResult.
"""

from typing import Dict, List, Optional, Set, Tuple, Any, Union
import numpy as np
import pandas as pd

from explainability.schema import (
    ExplainabilityResult,
    EvidenceQuality,
    EvidenceStatus,
    EvidenceProvenance,
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

from fault_diagnosis.schema import FaultDiagnosisResult
from fault_diagnosis.classifier import XGBoostFaultClassifier
from health_index.schema import HealthIndexResult
from forecasting.schema import ForecastResult
from prognostics.schema import RULResult


class ExplainabilityPipeline:
    """
    Session-isolated explainability pipeline keyed by (engine_id, mission_id).
    Interprets upstream PHM decisions without altering, overriding, or retraining any model.
    """

    def __init__(
        self,
        classifier: Optional[XGBoostFaultClassifier] = None,
        top_k_shap: int = 5,
        tau_threshold: float = 1.5,
    ):
        self.classifier = classifier
        self.shap_explainer = SHAPExplainer(top_k=top_k_shap)
        self.physics_evaluator = PhysicsEvidenceEvaluator(tau_threshold=tau_threshold)
        self.temporal_evaluator = TemporalEvidenceEvaluator()
        self.fusion_engine = EvidenceFusionEngine()

        # Session history keyed by (engine_id, mission_id)
        self._sessions: Dict[Tuple[str, Optional[str]], Dict[str, Any]] = {}

    def reset(self, engine_id: Optional[str] = None, mission_id: Optional[str] = None) -> None:
        """Reset historical session state for a specific mission or all missions."""
        if engine_id is None:
            self._sessions.clear()
        else:
            keys_to_del = [k for k in self._sessions if k[0] == engine_id and (mission_id is None or k[1] == mission_id)]
            for k in keys_to_del:
                del self._sessions[k]

    def _get_session(self, engine_id: str, mission_id: Optional[str]) -> Dict[str, Any]:
        key = (engine_id, mission_id)
        if key not in self._sessions:
            self._sessions[key] = {
                "timestamps": [],
                "explanations": [],
            }
        return self._sessions[key]

    def explain(
        self,
        timestamp: float,
        engine_id: str,
        mission_id: Optional[str] = None,
        residuals: Optional[Dict[str, Optional[float]]] = None,
        features: Optional[Union[pd.DataFrame, Dict[str, float]]] = None,
        diagnosis_result: Optional[FaultDiagnosisResult] = None,
        health_result: Optional[HealthIndexResult] = None,
        forecast_result: Optional[ForecastResult] = None,
        rul_result: Optional[RULResult] = None,
        classifier: Optional[XGBoostFaultClassifier] = None,
    ) -> ExplainabilityResult:
        """
        Synthesize authoritative explainability result for the current observation point.

        Args:
            timestamp: Current observation timestamp (s)
            engine_id: Unique engine identifier
            mission_id: Optional mission flight identifier
            residuals: Normalized physical residuals from Digital Twin (Phase 6)
            features: Feature vector for Phase 8 classifier (for SHAP)
            diagnosis_result: Upstream Phase 8 diagnosis result
            health_result: Upstream Phase 9 health index result
            forecast_result: Upstream Phase 10 forecasting result
            rul_result: Upstream Phase 11 remaining useful life result
            classifier: Optional classifier override (defaults to self.classifier)

        Returns:
            ExplainabilityResult containing all fused evidence streams
        """
        active_classifier = classifier or self.classifier
        norm_residuals = residuals or {}

        # -------------------------------------------------------------
        # STEP 1: UPSTREAM ATTRIBUTION & CONTEXT HARVESTING
        # -------------------------------------------------------------
        diagnosed_fault = "none"
        class_probs: Dict[str, float] = {}
        diag_conf: Optional[float] = None

        if diagnosis_result is not None:
            diagnosed_fault = getattr(diagnosis_result, "predicted_fault_type", "none")
            class_probs = getattr(diagnosis_result, "class_probabilities", {}) or {}
            diag_conf = getattr(diagnosis_result, "diagnostic_confidence", None)

        excluded_channels: Set[str] = set()
        if health_result is not None:
            excluded_channels = set(getattr(health_result, "excluded_channels", []) or [])

        # -------------------------------------------------------------
        # STEP 2: SHAP LOCAL FEATURE ATTRIBUTION (Phase 8 Interpretation)
        # -------------------------------------------------------------
        shap_evidence: Optional[SHAPEvidence] = None
        if features is not None and active_classifier is not None:
            shap_evidence = self.shap_explainer.explain_instance(
                classifier=active_classifier,
                features=features,
                predicted_fault=diagnosed_fault,
                class_probabilities=class_probs,
            )
        elif diagnosis_result is not None:
            shap_evidence = SHAPEvidence(
                predicted_fault=diagnosed_fault,
                diagnostic_confidence=diag_conf,
                class_probabilities=class_probs,
                top_features=[],
                base_value=None,
                status="MODEL_UNAVAILABLE",
                disclaimer="SHAP feature attribution does not establish causality.",
            )

        # -------------------------------------------------------------
        # STEP 3: DETERMINISTIC PHYSICS CONSISTENCY (Phase 6/8 Integration)
        # -------------------------------------------------------------
        physics_evidence = self.physics_evaluator.evaluate(
            diagnosed_fault=diagnosed_fault,
            residuals=norm_residuals,
            excluded_channels=excluded_channels,
        )

        # -------------------------------------------------------------
        # STEP 4: HEALTH EVIDENCE PASSTHROUGH (Phase 9 Interpretation)
        # -------------------------------------------------------------
        health_evidence: Optional[HealthEvidence] = None
        if health_result is not None:
            hi_val = getattr(health_result, "smoothed_health_index", None)
            hi_state = getattr(health_result, "health_state", None)
            dom_ch = list(getattr(health_result, "dominant_degraded_channels", []) or [])
            contrib_ch = dict(getattr(health_result, "channel_contributions", {}) or {})
            valid_ch = list(getattr(health_result, "valid_channels", []) or [])
            missing_ch = list(getattr(health_result, "missing_channels", []) or [])
            excluded_ch = list(getattr(health_result, "excluded_channels", []) or [])
            eff_w = dict(getattr(health_result, "effective_channel_weights", {}) or {})
            data_q = getattr(health_result, "data_quality", None)

            h_status = "SENSOR_ISOLATED" if len(excluded_ch) > 0 else (
                "AVAILABLE" if (hi_val is not None and np.isfinite(hi_val)) else "INSUFFICIENT_DATA"
            )

            health_evidence = HealthEvidence(
                current_health_index=round(hi_val, 4) if (hi_val is not None and np.isfinite(hi_val)) else None,
                health_state=hi_state,
                dominant_degraded_channels=dom_ch,
                channel_contributions={k: round(float(v), 4) for k, v in contrib_ch.items()},
                valid_channels=valid_ch,
                missing_channels=missing_ch,
                excluded_channels=excluded_ch,
                effective_channel_weights={k: round(float(v), 4) for k, v in eff_w.items()},
                data_quality=data_q,
                status=h_status,
            )

        # -------------------------------------------------------------
        # STEP 5: TEMPORAL EVIDENCE (Phase 9/10 Causal Evolution)
        # -------------------------------------------------------------
        temporal_evidence = self.temporal_evaluator.evaluate(
            health_result=health_result,
            forecast_result=forecast_result,
            rul_result=rul_result,
        )

        # -------------------------------------------------------------
        # STEP 6: RUL EVIDENCE (Phase 11 Interpretation)
        # -------------------------------------------------------------
        rul_evidence: Optional[RULEvidence] = None
        if rul_result is not None:
            p50 = getattr(rul_result, "rul_seconds_median", None)
            p05 = getattr(rul_result, "rul_seconds_p05", None)
            p95 = getattr(rul_result, "rul_seconds_p95", None)
            r_status = getattr(rul_result, "status", None)
            r_status_str = r_status.value if hasattr(r_status, "value") else str(r_status)
            lim_fac = getattr(rul_result, "limiting_factor", None)
            phase = getattr(rul_result, "active_flight_phase", None)
            traj_type = getattr(rul_result, "trajectory_type", None)
            prov = dict(getattr(rul_result, "provenance", {}) or {})

            rul_evidence = RULEvidence(
                rul_seconds_median=round(p50, 1) if (p50 is not None and np.isfinite(p50)) else None,
                rul_seconds_p05=round(p05, 1) if (p05 is not None and np.isfinite(p05)) else None,
                rul_seconds_p95=round(p95, 1) if (p95 is not None and np.isfinite(p95)) else None,
                rul_status=r_status_str,
                limiting_factor=lim_fac,
                active_flight_phase=phase,
                handoff_source=traj_type,
                eol_provenance=prov,
                status="AVAILABLE" if p50 is not None else "NON_DEGRADING",
                disclaimer="Project-defined EOL criteria are not certified OEM/FAA limits.",
            )

        # -------------------------------------------------------------
        # STEP 7: EVIDENCE FUSION & NARRATIVE GENERATION
        # -------------------------------------------------------------
        quality, narrative = self.fusion_engine.fuse(
            shap_evidence=shap_evidence,
            physics_evidence=physics_evidence,
            health_evidence=health_evidence,
            temporal_evidence=temporal_evidence,
            rul_evidence=rul_evidence,
        )

        provenance = EvidenceProvenance(
            engine_id=engine_id,
            mission_id=mission_id,
            timestamp=float(timestamp),
            phase8_present=diagnosis_result is not None,
            phase9_present=health_result is not None,
            phase10_present=forecast_result is not None,
            phase11_present=rul_result is not None,
        )

        result = ExplainabilityResult(
            engine_id=engine_id,
            mission_id=mission_id,
            timestamp=float(timestamp),
            overall_quality=quality,
            summary_explanation=narrative,
            shap_evidence=shap_evidence,
            physics_evidence=physics_evidence,
            health_evidence=health_evidence,
            temporal_evidence=temporal_evidence,
            rul_evidence=rul_evidence,
            provenance=provenance,
        )

        # Update session history
        session = self._get_session(engine_id, mission_id)
        session["timestamps"].append(timestamp)
        session["explanations"].append(result)

        return result
