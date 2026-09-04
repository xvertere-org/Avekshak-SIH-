"""
XGBoost multiclass fault classifier for Phase 8.

Provides a supervised 6-class fault diagnosis model with:
- Deterministic training (fixed random_state)
- NaN-safe prediction (tree_method="hist")
- Class probability output (multi:softprob)
- Global model feature importance (gain-based, NOT per-sample)
- Model persistence (save/load)

Label encoding uses deterministic alphabetical ordering of CANONICAL_FAULT_LABELS.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
import numpy as np
import pandas as pd
import json
import os

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

from fault_diagnosis.schema import CANONICAL_FAULT_LABELS


@dataclass
class FaultClassifierConfig:
    """
    XGBoost classifier configuration.

    Defensible baseline — no arbitrary hyperparameter tuning.
    All parameters are exposed for transparency.
    """
    n_estimators: int = 200
    max_depth: int = 6
    learning_rate: float = 0.1
    random_state: int = 42
    objective: str = "multi:softprob"
    num_class: int = 6
    eval_metric: str = "mlogloss"
    tree_method: str = "hist"    # handles NaN natively
    verbosity: int = 0

    def to_xgb_params(self) -> Dict[str, Any]:
        """Convert to XGBoost parameter dict."""
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "random_state": self.random_state,
            "objective": self.objective,
            "num_class": self.num_class,
            "eval_metric": self.eval_metric,
            "tree_method": self.tree_method,
            "verbosity": self.verbosity,
        }


class XGBoostFaultClassifier:
    """
    Supervised multiclass XGBoost fault classifier.

    Independently usable without Phase 7:
        classifier.predict(features)
        classifier.predict_proba(features)

    The classifier does NOT compute anomaly detection. Phase 7 anomaly_status
    and anomaly_score are never calculated or used by this class.
    """

    def __init__(self, config: Optional[FaultClassifierConfig] = None):
        self.config = config or FaultClassifierConfig()

        # Label encoder with deterministic alphabetical ordering
        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(CANONICAL_FAULT_LABELS)

        self._model: Optional[XGBClassifier] = None
        self._feature_names: List[str] = []
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def classes(self) -> List[str]:
        """Canonical fault class labels in encoding order."""
        return list(self._label_encoder.classes_)

    @property
    def feature_names(self) -> List[str]:
        """List of feature column names the model was trained on."""
        return list(self._feature_names)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[np.ndarray] = None,
    ) -> "XGBoostFaultClassifier":
        """
        Train the XGBoost classifier.

        Args:
            X: Feature matrix (from FeatureExtractor.transform()).
            y: Fault type labels (string values from FaultType).
            sample_weight: Optional per-sample weights for class balancing.
                          Applied ONLY during training.

        Returns:
            self
        """
        # Encode labels to integers
        y_encoded = self._label_encoder.transform(y.values)

        self._feature_names = list(X.columns)

        # Build XGBoost model
        self._model = XGBClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            random_state=self.config.random_state,
            objective=self.config.objective,
            num_class=self.config.num_class,
            eval_metric=self.config.eval_metric,
            tree_method=self.config.tree_method,
            verbosity=self.config.verbosity,
            use_label_encoder=False,
        )

        self._model.fit(
            X.values,
            y_encoded,
            sample_weight=sample_weight,
        )
        self._is_trained = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict fault types.

        Args:
            X: Feature matrix (same schema as training features).

        Returns:
            Array of predicted fault type strings.
        """
        if not self._is_trained:
            raise RuntimeError("Classifier must be trained before calling predict().")

        y_pred_encoded = self._model.predict(X.values)
        return self._label_encoder.inverse_transform(y_pred_encoded.astype(int))

    def predict_proba(
        self,
        X: pd.DataFrame,
        return_dict: bool = False,
    ) -> Union[np.ndarray, List[Dict[str, float]]]:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix.
            return_dict: If True, returns List[Dict[class_name, probability]].
                         If False (default), returns np.ndarray of shape (n_samples, 6).

        Returns:
            Array of shape (n_samples, 6) or List of dicts with class probabilities.
            Columns correspond to self.classes (alphabetical ordering).

        Note:
            Probabilities are model-estimated likelihoods, NOT certainties.
        """
        if not self._is_trained:
            raise RuntimeError("Classifier must be trained before calling predict_proba().")

        raw_probs = self._model.predict_proba(X.values)
        if return_dict:
            classes = self.classes
            return [
                {classes[j]: float(row[j]) for j in range(len(classes))}
                for row in raw_probs
            ]
        return raw_probs

    def predict_proba_dict(self, X: pd.DataFrame) -> List[Dict[str, float]]:
        """Convenience method returning class probability dictionaries per sample."""
        return self.predict_proba(X, return_dict=True)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get global model-level feature importance (gain-based).

        This is global importance across ALL predictions, NOT per-sample
        causal attribution. A feature being important means it contributed
        to model decisions generally, not that it physically caused any fault.

        Per-sample SHAP attribution is deferred to the Explainability phase.

        Returns:
            Dict mapping feature name -> importance score.
        """
        if not self._is_trained:
            raise RuntimeError("Classifier must be trained before getting feature importance.")

        importance_vals = self._model.feature_importances_
        importance_dict = {}
        for i, name in enumerate(self._feature_names):
            if i < len(importance_vals):
                importance_dict[name] = float(importance_vals[i])
            else:
                importance_dict[name] = 0.0

        return importance_dict

    def save_model(self, path: str) -> None:
        """
        Save the trained model and metadata to disk.

        Args:
            path: Directory path to save model files.
        """
        if not self._is_trained:
            raise RuntimeError("Classifier must be trained before saving.")

        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, "xgboost_fault_classifier.json")
        self._model.save_model(model_path)

        meta = {
            "feature_names": self._feature_names,
            "classes": list(self._label_encoder.classes_),
            "config": self.config.to_xgb_params(),
        }
        meta_path = os.path.join(path, "classifier_metadata.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    def load_model(self, path: str) -> "XGBoostFaultClassifier":
        """
        Load a trained model from disk.

        Args:
            path: Directory path containing saved model files.

        Returns:
            self
        """
        model_path = os.path.join(path, "xgboost_fault_classifier.json")
        meta_path = os.path.join(path, "classifier_metadata.json")

        with open(meta_path, "r") as f:
            meta = json.load(f)

        self._feature_names = meta["feature_names"]
        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(meta["classes"])

        self._model = XGBClassifier()
        self._model.load_model(model_path)
        self._is_trained = True

        return self
