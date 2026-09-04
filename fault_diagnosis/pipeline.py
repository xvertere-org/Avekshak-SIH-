"""
Phase 8 Fault Diagnosis Pipeline.

End-to-end integration wrapper that combines:
- FeatureExtractor (16-feature schema)
- XGBoostFaultClassifier (6-class supervised multiclass model)
- Optional Phase 7 gating / provenance context
- Data quality checking (insufficient residual validation)
- Structured FaultDiagnosisResult generation
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd

from fault_diagnosis.schema import (
    FaultDiagnosisResult,
    DiagnosisDataQuality,
    CANONICAL_FAULT_LABELS,
)
from fault_diagnosis.features import (
    FeatureExtractor,
    DIAGNOSTIC_FEATURES,
)
from fault_diagnosis.classifier import XGBoostFaultClassifier


class FaultDiagnosisPipeline:
    """
    Inference pipeline for Phase 8 Fault Diagnosis.

    Operates on ResidualFrame data (single record dict or batch DataFrame).
    Produces structured FaultDiagnosisResult objects with full provenance,
    class probabilities, diagnostic confidence, and data quality status.

    Phase 7 Integration:
    - Fully independent of Phase 7: works when Phase 7 is not present.
    - If Phase 7 anomaly_status is provided:
        * 'NORMAL': can optionally bypass classifier and report 'none'
        * 'WARNING' or 'ANOMALY': runs full classifier
        * 'INSUFFICIENT_DATA': flags data quality accordingly
    """

    def __init__(
        self,
        classifier: XGBoostFaultClassifier,
        feature_extractor: FeatureExtractor,
        enable_phase7_gating: bool = True,
    ):
        """
        Initialize pipeline with trained classifier and fitted feature extractor.

        Args:
            classifier: Trained XGBoostFaultClassifier.
            feature_extractor: Fitted FeatureExtractor instance.
            enable_phase7_gating: If True and anomaly_status == 'NORMAL',
                                 report 'none' without running model.
        """
        self.classifier = classifier
        self.feature_extractor = feature_extractor
        self.enable_phase7_gating = enable_phase7_gating

        # Cache global model feature importance
        if classifier.is_trained:
            self._cached_feature_importance = classifier.get_feature_importance()
        else:
            self._cached_feature_importance = {}

    def diagnose_sample(
        self,
        sample: Dict[str, Any],
        anomaly_status: Optional[str] = None,
        anomaly_score: Optional[float] = None,
    ) -> FaultDiagnosisResult:
        """
        Diagnose a single telemetry/residual sample.

        Args:
            sample: Dict containing residual channels and operating context.
            anomaly_status: Optional Phase 7 status ('NORMAL', 'WARNING', 'ANOMALY', etc.)
            anomaly_score: Optional Phase 7 anomaly score.

        Returns:
            FaultDiagnosisResult dataclass instance.
        """
        df = pd.DataFrame([sample])
        results = self.diagnose_batch(df, anomaly_statuses=[anomaly_status] if anomaly_status else None,
                                      anomaly_scores=[anomaly_score] if anomaly_score is not None else None)
        return results[0]

    def diagnose_batch(
        self,
        df: pd.DataFrame,
        anomaly_statuses: Optional[List[Optional[str]]] = None,
        anomaly_scores: Optional[List[Optional[float]]] = None,
    ) -> List[FaultDiagnosisResult]:
        """
        Diagnose a batch of samples in a DataFrame.

        Args:
            df: DataFrame containing ResidualFrame columns and operating context.
            anomaly_statuses: Optional list of Phase 7 anomaly statuses per row.
            anomaly_scores: Optional list of Phase 7 anomaly scores per row.

        Returns:
            List of FaultDiagnosisResult instances.
        """
        n_samples = len(df)
        if n_samples == 0:
            return []

        # Extract features (16 columns guaranteed)
        X = self.feature_extractor.transform(df)

        # Evaluate data quality: count valid (non-NaN) core diagnostic residuals
        core_cols = [c for c in DIAGNOSTIC_FEATURES if c in df.columns]
        if core_cols:
            valid_core_counts = df[core_cols].notna().sum(axis=1).values
        else:
            valid_core_counts = np.zeros(n_samples, dtype=int)

        # Check Phase 7 context columns if embedded in df
        if anomaly_statuses is None:
            if "anomaly_status" in df.columns:
                anomaly_statuses = df["anomaly_status"].tolist()
            else:
                anomaly_statuses = [None] * n_samples

        if anomaly_scores is None:
            if "anomaly_score" in df.columns:
                anomaly_scores = [float(s) if pd.notna(s) else None for s in df["anomaly_score"]]
            else:
                anomaly_scores = [None] * n_samples

        # Model feature importance (global)
        model_importance = dict(self._cached_feature_importance)

        # Get classifier predictions & probabilities for all rows
        if self.classifier.is_trained:
            prob_matrix = self.classifier.predict_proba(X)
            preds = self.classifier.predict(X)
            classes = self.classifier.classes
            probs = [
                {classes[j]: float(prob_matrix[i, j]) for j in range(len(classes))}
                for i in range(n_samples)
            ]
        else:
            # Fallback for untrained model
            probs = [{cls: 1.0 / len(CANONICAL_FAULT_LABELS) for cls in CANONICAL_FAULT_LABELS} for _ in range(n_samples)]
            preds = ["none" for _ in range(n_samples)]

        results: List[FaultDiagnosisResult] = []

        for i in range(n_samples):
            row = df.iloc[i]
            timestamp = float(row.get("timestamp", 0.0))
            engine_id = str(row.get("engine_id", "ENG_001"))
            mission_id = str(row.get("mission_id", row.get("mission_run_id", "MISSION_001")))
            mission_phase = str(row.get("mission_phase", "UNKNOWN"))

            a_status = anomaly_statuses[i] if anomaly_statuses else None
            a_score = anomaly_scores[i] if anomaly_scores else None

            # Check insufficient data threshold: < 4 valid core residuals
            is_insufficient = (valid_core_counts[i] < 4)

            if is_insufficient:
                data_quality = DiagnosisDataQuality.INSUFFICIENT_DATA.value
                pred_fault = "none"  # Or keep uncommitted
                # Uniform or zeroed confidence
                prob_dict = {cls: 1.0 / len(CANONICAL_FAULT_LABELS) for cls in CANONICAL_FAULT_LABELS}
                conf = 0.0
            elif self.enable_phase7_gating and a_status == "NORMAL":
                # Phase 7 gating: NORMAL flight reported as healthy
                data_quality = DiagnosisDataQuality.VALID.value
                pred_fault = "none"
                prob_dict = {cls: (1.0 if cls == "none" else 0.0) for cls in CANONICAL_FAULT_LABELS}
                conf = 1.0
            else:
                data_quality = DiagnosisDataQuality.VALID.value
                pred_fault = str(preds[i])
                prob_dict = dict(probs[i])
                conf = float(max(prob_dict.values()))

            res = FaultDiagnosisResult(
                timestamp=timestamp,
                engine_id=engine_id,
                mission_id=mission_id,
                mission_phase=mission_phase,
                predicted_fault_type=pred_fault,
                class_probabilities=prob_dict,
                diagnostic_confidence=conf,
                data_quality=data_quality,
                model_feature_importance=model_importance,
                anomaly_status=a_status,
                anomaly_score=a_score,
            )
            results.append(res)

        return results
