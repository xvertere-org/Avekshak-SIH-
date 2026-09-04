"""
Evidence fusion engine for Phase 7 Hybrid Anomaly Detection.

Combines evidence from:
1. Instantaneous Residual Threshold Detector
2. EWMA Temporal Detector
3. Persistence Gate
4. Isolation Forest Detector

Yields deterministic anomaly_score, anomaly_status, and contributing_channels.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from anomaly_detection.schema import AnomalyStatus


class EvidenceFusionEngine:
    """
    Deterministic rule-based and weighted evidence fusion engine.
    Combines instantaneous, temporal, persistence, and multivariate evidence.
    """

    def __init__(
        self,
        weight_threshold: float = 0.25,
        weight_ewma: float = 0.35,
        weight_persistence: float = 0.20,
        weight_isolation: float = 0.20,
    ):
        """
        Args:
            weight_threshold: Weight for instantaneous threshold score in fused score.
            weight_ewma: Weight for EWMA temporal score.
            weight_persistence: Weight for persistence score.
            weight_isolation: Weight for Isolation Forest score.
        """
        total = weight_threshold + weight_ewma + weight_persistence + weight_isolation
        self.w_thresh = weight_threshold / total
        self.w_ewma = weight_ewma / total
        self.w_persist = weight_persistence / total
        self.w_iso = weight_isolation / total

    def fuse(
        self,
        is_valid: bool,
        threshold_result: Dict[str, Any],
        ewma_result: Dict[str, Any],
        persistence_result: Dict[str, Any],
        isolation_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Perform deterministic evidence fusion for a single sample.

        Args:
            is_valid: False if row has fewer than min_valid_features.
            threshold_result: Output from ResidualThresholdDetector.evaluate_sample.
            ewma_result: Output from EWMADetector.update_sample.
            persistence_result: Output from PersistenceGate.update.
            isolation_result: Output from IsolationForestDetector.score_sample.

        Returns:
            Dict containing:
                anomaly_score: Continuous fused score in [0.0, 1.0] (or NaN if invalid).
                anomaly_status: AnomalyStatus enum value string.
                contributing_channels: List of normalized channels that crossed thresholds.
        """
        # ======================================================================
        # RULE 1: Insufficient Data / Missingness Guard
        # ======================================================================
        if not is_valid:
            return {
                "anomaly_score": np.nan,
                "anomaly_status": AnomalyStatus.INSUFFICIENT_DATA.value,
                "contributing_channels": [],
            }

        # Sub-scores
        s_thresh = threshold_result.get("threshold_score", 0.0)
        s_ewma = ewma_result.get("ewma_score", 0.0)
        s_persist = persistence_result.get("persistence_score", 0.0)
        s_iso = isolation_result.get("isolation_score", np.nan)

        # Re-weight if Isolation Forest is unfitted or returned NaN
        if np.isnan(s_iso):
            sum_w = self.w_thresh + self.w_ewma + self.w_persist
            fused_score = (
                (self.w_thresh * (s_thresh if not np.isnan(s_thresh) else 0.0)
                 + self.w_ewma * (s_ewma if not np.isnan(s_ewma) else 0.0)
                 + self.w_persist * s_persist) / sum_w
            )
        else:
            fused_score = (
                self.w_thresh * (s_thresh if not np.isnan(s_thresh) else 0.0)
                + self.w_ewma * (s_ewma if not np.isnan(s_ewma) else 0.0)
                + self.w_persist * s_persist
                + self.w_iso * s_iso
            )

        fused_score = float(np.clip(fused_score, 0.0, 1.0))

        # ======================================================================
        # DETERMINISTIC STATUS DECISION TREE
        # ======================================================================
        is_crit = threshold_result.get("is_critical", False)
        is_thresh_anom = threshold_result.get("is_anomaly", False)
        is_thresh_warn = threshold_result.get("is_warning", False)
        is_ewma_anom = ewma_result.get("is_anomaly", False)
        is_ewma_warn = ewma_result.get("is_warning", False)
        is_persist_sat = persistence_result.get("persistence_satisfied", False)
        is_iso_anom = isolation_result.get("is_anomaly", False)
        is_iso_warn = isolation_result.get("is_warning", False)

        crit_ch = threshold_result.get("critical_channels", [])
        anom_ch = threshold_result.get("anomaly_channels", [])
        warn_ch = threshold_result.get("warning_channels", [])
        ewma_anom_ch = ewma_result.get("anomaly_channels", [])
        ewma_warn_ch = ewma_result.get("warning_channels", [])

        # RULE 2: Immediate Critical Override
        # Massive instantaneous deviation flags immediate ANOMALY without waiting for persistence
        if is_crit:
            status = AnomalyStatus.ANOMALY.value
            contributing = sorted(list(set(crit_ch + anom_ch)))
            fused_score = max(fused_score, 0.90)

        # RULE 3: Sustained Anomaly via Persistence Gate
        # Sustained deviation confirmed by consecutive samples + anomaly evidence
        elif is_persist_sat and (is_ewma_anom or is_thresh_anom or is_iso_anom):
            status = AnomalyStatus.ANOMALY.value
            contributing = sorted(list(set(anom_ch + ewma_anom_ch)))
            if not contributing and is_iso_anom:
                # Multivariate anomaly where individual 1D residuals are near threshold
                contributing = sorted(list(set(warn_ch + ewma_warn_ch)))
            fused_score = max(fused_score, 0.75)

        # RULE 4: Warning / Transient Spike
        # Sub-anomaly warning evidence OR unpersisted anomaly spike
        elif (is_thresh_warn or is_thresh_anom or is_ewma_warn or is_ewma_anom or is_iso_warn or is_iso_anom):
            status = AnomalyStatus.WARNING.value
            contributing = sorted(list(set(warn_ch + ewma_warn_ch + anom_ch + ewma_anom_ch)))
            fused_score = min(max(fused_score, 0.35), 0.74)

        # RULE 5: Normal
        else:
            status = AnomalyStatus.NORMAL.value
            contributing = []
            fused_score = min(fused_score, 0.34)

        return {
            "anomaly_score": round(fused_score, 4),
            "anomaly_status": status,
            "contributing_channels": contributing,
        }
