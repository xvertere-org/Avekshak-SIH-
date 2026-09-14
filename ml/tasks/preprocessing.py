"""
Leakage-Safe Preprocessor for Phase 2 Grey-Box ML Baselines.

Guarantees:
- Scalers and imputers are fitted STRICTLY on training partitions.
- Zero feature statistics from validation or test sets contaminate the transformation.
- Supports serialization and provenance reporting of normalization statistics.
"""

from typing import List, Optional, Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


class LeakageSafePreprocessor:
    """
    Standardizes numeric features using training-only statistics.
    Prevents mean/variance leakage from test splits.
    """

    def __init__(self, with_imputer: bool = True):
        self.with_imputer = with_imputer
        self.imputer = SimpleImputer(strategy="median") if with_imputer else None
        self.scaler = StandardScaler()
        self.feature_names_: List[str] = []
        self.fitted_: bool = False
        self.means_: Optional[np.ndarray] = None
        self.scales_: Optional[np.ndarray] = None

    def fit(self, X_train: Union[pd.DataFrame, np.ndarray], feature_names: Optional[List[str]] = None) -> "LeakageSafePreprocessor":
        """
        Fit imputer and scaler strictly on training samples.
        """
        if isinstance(X_train, pd.DataFrame):
            self.feature_names_ = feature_names or list(X_train.columns)
            arr = X_train[self.feature_names_].to_numpy(dtype=float)
        else:
            arr = np.asarray(X_train, dtype=float)
            self.feature_names_ = feature_names or [f"feature_{i}" for i in range(arr.shape[1])]

        if self.imputer is not None:
            arr = self.imputer.fit_transform(arr)

        self.scaler.fit(arr)
        self.means_ = self.scaler.mean_
        self.scales_ = self.scaler.scale_
        self.fitted_ = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Transform a feature matrix using strictly training-fitted statistics.
        """
        if not self.fitted_:
            raise RuntimeError("Preprocessor must be fitted on training data before calling transform().")

        if isinstance(X, pd.DataFrame):
            arr = X[self.feature_names_].to_numpy(dtype=float)
        else:
            arr = np.asarray(X, dtype=float)

        if self.imputer is not None:
            arr = self.imputer.transform(arr)

        return self.scaler.transform(arr)

    def fit_transform(self, X_train: Union[pd.DataFrame, np.ndarray], feature_names: Optional[List[str]] = None) -> np.ndarray:
        """Fit on training data and return transformed training array."""
        return self.fit(X_train, feature_names=feature_names).transform(X_train)

    def get_statistics(self) -> Dict[str, Any]:
        """Return fitted training statistics for audit and reporting."""
        if not self.fitted_:
            return {}
        return {
            "features": self.feature_names_,
            "means": self.means_.tolist() if self.means_ is not None else [],
            "scales": self.scales_.tolist() if self.scales_ is not None else [],
        }
