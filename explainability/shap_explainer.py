"""
Local feature attribution explainer for Phase 8 XGBoost fault diagnosis using TreeSHAP.
NOTE: SHAP feature attribution does NOT establish causality.
"""

from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import xgboost as xgb

from explainability.schema import SHAPEvidence, SHAPFeatureContribution
from fault_diagnosis.classifier import XGBoostFaultClassifier


class SHAPExplainer:
    """
    Computes per-sample TreeSHAP feature attribution for Phase 8 XGBoost fault classifier.
    Interprets existing models without retraining or modifying model weights.
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def explain_instance(
        self,
        classifier: Optional[XGBoostFaultClassifier],
        features: Union[pd.DataFrame, Dict[str, float], np.ndarray],
        predicted_fault: Optional[str] = None,
        class_probabilities: Optional[Dict[str, float]] = None,
    ) -> SHAPEvidence:
        """
        Generate local SHAP feature attribution for a single observation instance.

        Args:
            classifier: Trained Phase 8 XGBoostFaultClassifier instance (or None)
            features: Feature vector matching the classifier's feature_names schema
            predicted_fault: Optional pre-computed diagnosis label from Phase 8
            class_probabilities: Optional pre-computed class probabilities from Phase 8

        Returns:
            SHAPEvidence containing top-k contributing features and base value
        """
        disclaimer = "SHAP feature attribution does not establish causality."

        # Guard: Check model availability
        if classifier is None or not getattr(classifier, "is_trained", False):
            return SHAPEvidence(
                predicted_fault=predicted_fault,
                diagnostic_confidence=max(class_probabilities.values()) if class_probabilities else None,
                class_probabilities=class_probabilities or {},
                top_features=[],
                base_value=None,
                status="MODEL_UNAVAILABLE",
                disclaimer=disclaimer,
            )

        feature_names = classifier.feature_names
        classes = classifier.classes

        # Format input into a 1-row DataFrame
        if isinstance(features, dict):
            row_df = pd.DataFrame([{col: features.get(col, np.nan) for col in feature_names}])
        elif isinstance(features, pd.DataFrame):
            row_df = features[feature_names].iloc[[0]].copy() if len(features) > 0 else pd.DataFrame(columns=feature_names)
        elif isinstance(features, np.ndarray):
            row_df = pd.DataFrame([features.flatten()[:len(feature_names)]], columns=feature_names[:features.size])
        else:
            return SHAPEvidence(
                predicted_fault=predicted_fault,
                diagnostic_confidence=None,
                class_probabilities={},
                top_features=[],
                base_value=None,
                status="INSUFFICIENT_DATA",
                disclaimer=disclaimer,
            )

        if len(row_df) == 0:
            return SHAPEvidence(
                predicted_fault=predicted_fault,
                diagnostic_confidence=None,
                class_probabilities={},
                top_features=[],
                base_value=None,
                status="INSUFFICIENT_DATA",
                disclaimer=disclaimer,
            )

        # Check for sufficient valid features (if all are NaN, return INSUFFICIENT_DATA)
        if row_df.isna().all().all():
            return SHAPEvidence(
                predicted_fault=predicted_fault,
                diagnostic_confidence=None,
                class_probabilities={},
                top_features=[],
                base_value=None,
                status="INSUFFICIENT_DATA",
                disclaimer=disclaimer,
            )

        try:
            # Extract underlying XGBoost booster
            booster = classifier._model.get_booster()
            dmat = xgb.DMatrix(row_df, feature_names=feature_names)

            # Compute TreeSHAP contributions natively via XGBoost C++ core
            # Shape for multiclass: (1, n_classes, n_features + 1)
            # Last column index (-1) is the base value / expected margin
            contribs = booster.predict(dmat, pred_contribs=True)

            # Determine target class for attribution
            if predicted_fault is not None and predicted_fault in classes:
                target_idx = classes.index(predicted_fault)
            else:
                probs_raw = classifier.predict_proba(row_df)
                if isinstance(probs_raw, np.ndarray):
                    probs = {classes[c]: float(probs_raw[0, c]) for c in range(len(classes))}
                elif hasattr(probs_raw, "iloc"):
                    probs = probs_raw.iloc[0].to_dict()
                else:
                    probs = dict(probs_raw)
                target_idx = int(np.argmax([probs[c] for c in classes]))
                predicted_fault = classes[target_idx]

            class_contribs = contribs[0, target_idx, :-1]
            base_value = float(contribs[0, target_idx, -1])

            # Calculate probabilities if not provided
            if class_probabilities is None:
                probs_raw = classifier.predict_proba(row_df)
                if isinstance(probs_raw, np.ndarray):
                    class_probabilities = {classes[c]: float(probs_raw[0, c]) for c in range(len(classes))}
                elif hasattr(probs_raw, "iloc"):
                    class_probabilities = probs_raw.iloc[0].to_dict()
                else:
                    class_probabilities = dict(probs_raw)

            diag_conf = float(class_probabilities.get(predicted_fault, 0.0))

            # Rank features by absolute SHAP attribution magnitude
            abs_contribs = np.abs(class_contribs)
            total_abs = float(np.sum(abs_contribs))
            ranked_indices = np.argsort(-abs_contribs)

            top_k_indices = ranked_indices[:self.top_k]
            top_features: List[SHAPFeatureContribution] = []

            for idx in top_k_indices:
                val = row_df.iloc[0, idx]
                shap_val = float(class_contribs[idx])
                feat_val = float(val) if pd.notna(val) else None
                direction = "TOWARD_PREDICTED_CLASS" if shap_val > 0 else "AWAY_FROM_PREDICTED_CLASS"
                rel_weight = round(abs(shap_val) / total_abs, 4) if total_abs > 1e-8 else 0.0

                top_features.append(
                    SHAPFeatureContribution(
                        feature_name=feature_names[idx],
                        feature_value=feat_val,
                        shap_value=round(shap_val, 4),
                        direction=direction,
                        relative_weight=rel_weight,
                    )
                )

            return SHAPEvidence(
                predicted_fault=predicted_fault,
                diagnostic_confidence=round(diag_conf, 4),
                class_probabilities={k: round(float(v), 4) for k, v in class_probabilities.items()},
                top_features=top_features,
                base_value=round(base_value, 4),
                status="AVAILABLE",
                disclaimer=disclaimer,
            )

        except Exception as e:
            return SHAPEvidence(
                predicted_fault=predicted_fault,
                diagnostic_confidence=max(class_probabilities.values()) if class_probabilities else None,
                class_probabilities=class_probabilities or {},
                top_features=[],
                base_value=None,
                status="INSUFFICIENT_DATA",
                disclaimer=disclaimer,
            )
