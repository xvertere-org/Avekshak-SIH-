"""
Top-level pipeline orchestrator for Phase 7 Hybrid Anomaly Detection.

Coordinates:
- Residual preprocessing & feature extraction
- Instantaneous threshold detection
- Temporal EWMA smoothing
- Persistence gating
- Multivariate Isolation Forest scoring
- Deterministic evidence fusion
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd

from anomaly_detection.schema import (
    AnomalyStatus,
    AnomalyRecord,
    AnomalyFrame,
    PRIMARY_NORMALIZED_RESIDUAL_CHANNELS,
)
from anomaly_detection.preprocessing import ResidualPreprocessor
from anomaly_detection.detectors import (
    ResidualThresholdDetector,
    EWMADetector,
    PersistenceGate,
    IsolationForestDetector,
)
from anomaly_detection.fusion import EvidenceFusionEngine
from digital_twin.residuals import ResidualFrame


class HybridAnomalyDetector:
    """
    Hybrid Anomaly Detection Pipeline.

    Answers the core question:
    "Is the engine behaving abnormally compared with its nominal behavior?"
    Phase 7 strictly detects anomalies without diagnosing physical fault identities.
    """

    def __init__(
        self,
        min_valid_features: int = 4,
        threshold_warning: float = 1.5,
        threshold_anomaly: float = 3.0,
        threshold_critical: float = 4.5,
        channel_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
        ewma_alpha: float = 0.2,
        ewma_warning: float = 1.2,
        ewma_anomaly: float = 2.2,
        persistence_min_consecutive: int = 3,
        iforest_n_estimators: int = 100,
        iforest_contamination: float = 0.01,
        iforest_random_state: int = 42,
        fusion_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize hybrid anomaly detection pipeline and sub-detectors.
        """
        self.preprocessor = ResidualPreprocessor(
            feature_channels=PRIMARY_NORMALIZED_RESIDUAL_CHANNELS,
            min_valid_features=min_valid_features,
        )

        self.threshold_detector = ResidualThresholdDetector(
            warning_threshold=threshold_warning,
            anomaly_threshold=threshold_anomaly,
            critical_threshold=threshold_critical,
            channel_thresholds=channel_thresholds,
        )

        self.ewma_detector = EWMADetector(
            alpha=ewma_alpha,
            warning_threshold=ewma_warning,
            anomaly_threshold=ewma_anomaly,
            channels=PRIMARY_NORMALIZED_RESIDUAL_CHANNELS,
        )

        self.persistence_gate = PersistenceGate(
            min_consecutive=persistence_min_consecutive,
        )

        self.isolation_forest = IsolationForestDetector(
            n_estimators=iforest_n_estimators,
            contamination=iforest_contamination,
            random_state=iforest_random_state,
            feature_channels=PRIMARY_NORMALIZED_RESIDUAL_CHANNELS,
        )

        weights = fusion_weights or {
            "weight_threshold": 0.25,
            "weight_ewma": 0.35,
            "weight_persistence": 0.20,
            "weight_isolation": 0.20,
        }
        self.fusion_engine = EvidenceFusionEngine(**weights)

    def fit(
        self,
        healthy_residuals: Union[ResidualFrame, pd.DataFrame],
    ) -> "HybridAnomalyDetector":
        """
        Fit the baseline multivariate Isolation Forest detector using healthy residual data only.

        Args:
            healthy_residuals: ResidualFrame or DataFrame of healthy baseline normalized residuals.
        """
        self.isolation_forest.fit(healthy_residuals)
        return self

    def reset(self) -> None:
        """
        Reset internal temporal filter (EWMA) and persistence states.
        """
        self.ewma_detector.reset()
        self.persistence_gate.reset()

    def detect(
        self,
        residuals: Union[ResidualFrame, pd.DataFrame],
        reset_state: bool = True,
    ) -> AnomalyFrame:
        """
        Process a batch or stream of residual telemetry and detect anomalies.

        Args:
            residuals: ResidualFrame or DataFrame of residual data.
            reset_state: Whether to reset EWMA and persistence before processing batch.

        Returns:
            AnomalyFrame containing structured results per sample.
        """
        if reset_state:
            self.reset()

        # 1. Feature extraction and validity checks
        feature_df, valid_mask, missing_per_row = self.preprocessor.extract_features(residuals)
        prov_df = self.preprocessor.extract_provenance(residuals)

        results: List[Dict[str, Any]] = []

        # 2. Sequential sample evaluation
        for idx in feature_df.index:
            row_features = feature_df.loc[idx].to_dict()
            is_valid = bool(valid_mask.loc[idx])
            missing_feats = missing_per_row[len(results)]

            # Instantaneous threshold evaluation
            t_res = self.threshold_detector.evaluate_sample(row_features)

            # Temporal EWMA evaluation
            e_res = self.ewma_detector.update_sample(row_features)

            # Multivariate Isolation Forest scoring
            i_res = self.isolation_forest.score_sample(row_features, is_valid=is_valid)

            # Persistence input: abnormal condition from instantaneous, temporal, or multivariate evidence
            is_abnormal = (
                t_res.get("is_anomaly", False)
                or e_res.get("is_anomaly", False)
                or t_res.get("is_critical", False)
                or i_res.get("is_anomaly", False)
            )
            p_res = self.persistence_gate.update(is_abnormal=is_abnormal, is_valid=is_valid)

            # Deterministic evidence fusion
            f_res = self.fusion_engine.fuse(
                is_valid=is_valid,
                threshold_result=t_res,
                ewma_result=e_res,
                persistence_result=p_res,
                isolation_result=i_res,
            )

            # Compile result record
            record = {
                "timestamp": prov_df.at[idx, "timestamp"],
                "engine_id": prov_df.at[idx, "engine_id"],
                "mission_id": prov_df.at[idx, "mission_id"],
                "mission_phase": prov_df.at[idx, "mission_phase"],
                "source": prov_df.at[idx, "source"],
                "source_type": prov_df.at[idx, "source_type"],
                "anomaly_score": f_res["anomaly_score"],
                "anomaly_status": f_res["anomaly_status"],
                "threshold_score": t_res["threshold_score"],
                "ewma_score": e_res["ewma_score"],
                "persistence_score": p_res["persistence_score"],
                "isolation_score": i_res["isolation_score"],
                "persistence_count": p_res["persistence_count"],
                "contributing_channels": f_res["contributing_channels"],
                "quality_status": prov_df.at[idx, "quality_status"],
                "missing_features": missing_feats,
            }
            results.append(record)

        result_df = pd.DataFrame(results, index=feature_df.index)

        meta = {
            "min_valid_features": self.preprocessor.min_valid_features,
            "threshold_warning": self.threshold_detector.warning_threshold,
            "threshold_anomaly": self.threshold_detector.anomaly_threshold,
            "threshold_critical": self.threshold_detector.critical_threshold,
            "ewma_alpha": self.ewma_detector.alpha,
            "persistence_min_consecutive": self.persistence_gate.min_consecutive,
            "isolation_forest_fitted": self.isolation_forest.is_fitted,
        }

        return AnomalyFrame(result_df, metadata=meta)
