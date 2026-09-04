"""
Independent anomaly detectors for Phase 7 Hybrid Anomaly Detection.

Contains:
1. ResidualThresholdDetector: Instantaneous per-channel thresholding.
2. EWMADetector: Temporal exponential smoothing.
3. PersistenceGate: Gated consecutive sample confirmation.
4. IsolationForestDetector: Unsupervised multivariate anomaly model fitted on healthy baseline.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from anomaly_detection.schema import PRIMARY_NORMALIZED_RESIDUAL_CHANNELS
from digital_twin.residuals import ResidualFrame


# ==============================================================================
# 1. RESIDUAL THRESHOLD DETECTOR
# ==============================================================================

class ResidualThresholdDetector:
    """
    Instantaneous threshold detector operating on normalized residuals.
    Detects large deviations from nominal expectations per channel.
    """

    def __init__(
        self,
        warning_threshold: float = 1.5,
        anomaly_threshold: float = 3.0,
        critical_threshold: float = 4.5,
        channel_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
    ):
        """
        Args:
            warning_threshold: Global default warning threshold (|norm_residual|).
            anomaly_threshold: Global default anomaly threshold (|norm_residual|).
            critical_threshold: Global default critical threshold (|norm_residual|) for immediate override.
            channel_thresholds: Optional per-channel overrides, e.g.:
                {"vibration_norm_residual": {"warning": 1.8, "anomaly": 3.2, "critical": 5.0}}
        """
        self.warning_threshold = float(warning_threshold)
        self.anomaly_threshold = float(anomaly_threshold)
        self.critical_threshold = float(critical_threshold)
        self.channel_thresholds = channel_thresholds or {}

    def get_thresholds(self, channel: str) -> Tuple[float, float, float]:
        """Get (warning, anomaly, critical) thresholds for a specific channel."""
        overrides = self.channel_thresholds.get(channel, {})
        w = overrides.get("warning", self.warning_threshold)
        a = overrides.get("anomaly", self.anomaly_threshold)
        c = overrides.get("critical", self.critical_threshold)
        return w, a, c

    def evaluate_sample(
        self,
        features: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Evaluate instantaneous threshold crossings for a single sample dictionary.

        Args:
            features: Dictionary mapping normalized residual channel to value (or NaN).

        Returns:
            Dict containing:
                threshold_score: Continuous score in [0.0, 1.0].
                warning_channels: Channels exceeding warning threshold.
                anomaly_channels: Channels exceeding anomaly threshold.
                critical_channels: Channels exceeding critical threshold.
                is_warning: bool.
                is_anomaly: bool.
                is_critical: bool.
        """
        warning_ch = []
        anomaly_ch = []
        critical_ch = []
        max_ratio = 0.0
        has_valid = False

        for ch, val in features.items():
            if val is None or np.isnan(val):
                continue
            has_valid = True
            abs_val = abs(float(val))
            w, a, c = self.get_thresholds(ch)

            ratio = abs_val / max(c, 1e-6)
            if ratio > max_ratio:
                max_ratio = ratio

            if abs_val >= c:
                critical_ch.append(ch)
                anomaly_ch.append(ch)
                warning_ch.append(ch)
            elif abs_val >= a:
                anomaly_ch.append(ch)
                warning_ch.append(ch)
            elif abs_val >= w:
                warning_ch.append(ch)

        if not has_valid:
            return {
                "threshold_score": np.nan,
                "warning_channels": [],
                "anomaly_channels": [],
                "critical_channels": [],
                "is_warning": False,
                "is_anomaly": False,
                "is_critical": False,
            }

        # Continuous threshold score scaled to [0.0, 1.0]
        score = float(np.clip(max_ratio, 0.0, 1.0))

        return {
            "threshold_score": score,
            "warning_channels": warning_ch,
            "anomaly_channels": anomaly_ch,
            "critical_channels": critical_ch,
            "is_warning": len(warning_ch) > 0,
            "is_anomaly": len(anomaly_ch) > 0,
            "is_critical": len(critical_ch) > 0,
        }


# ==============================================================================
# 2. EWMA TEMPORAL DETECTOR
# ==============================================================================

class EWMADetector:
    """
    Exponentially Weighted Moving Average (EWMA) temporal detector.
    Applies EWMA independently per normalized residual channel to smooth noise
    and detect sustained temporal drift.
    """

    def __init__(
        self,
        alpha: float = 0.2,
        warning_threshold: float = 1.2,
        anomaly_threshold: float = 2.2,
        channels: Optional[List[str]] = None,
    ):
        """
        Args:
            alpha: Smoothing parameter in (0, 1]. Lower = more smoothing.
            warning_threshold: Threshold on smoothed normalized residual for warning.
            anomaly_threshold: Threshold on smoothed normalized residual for anomaly.
            channels: List of channels to track.
        """
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"EWMA alpha must be in (0, 1], got {alpha}")
        self.alpha = float(alpha)
        self.warning_threshold = float(warning_threshold)
        self.anomaly_threshold = float(anomaly_threshold)
        self.channels = list(channels or PRIMARY_NORMALIZED_RESIDUAL_CHANNELS)
        self.state: Dict[str, float] = {}

    def reset(self) -> None:
        """Reset internal EWMA filter states."""
        self.state.clear()

    def update_sample(
        self,
        features: Dict[str, float],
    ) -> Dict[str, Any]:
        """
        Update EWMA filters with an incoming sample and evaluate smoothed deviations.

        Args:
            features: Dictionary mapping normalized residual channel to value (or NaN).

        Returns:
            Dict containing:
                ewma_score: Continuous score in [0.0, 1.0].
                smoothed_values: Current smoothed values per channel.
                warning_channels: Channels exceeding EWMA warning threshold.
                anomaly_channels: Channels exceeding EWMA anomaly threshold.
                is_warning: bool.
                is_anomaly: bool.
        """
        smoothed = {}
        warning_ch = []
        anomaly_ch = []
        max_ratio = 0.0
        valid_count = 0

        for ch in self.channels:
            val = features.get(ch, np.nan)
            if val is None or np.isnan(val):
                # Preserve existing smoothed state if available, but do not update with NaN
                # NEVER convert NaN to 0
                if ch in self.state:
                    smoothed[ch] = self.state[ch]
                else:
                    smoothed[ch] = np.nan
                continue

            valid_count += 1
            float_val = float(val)

            if ch not in self.state or np.isnan(self.state[ch]):
                new_state = float_val
            else:
                new_state = self.alpha * float_val + (1.0 - self.alpha) * self.state[ch]

            self.state[ch] = new_state
            smoothed[ch] = new_state

            abs_s = abs(new_state)
            ratio = abs_s / max(self.anomaly_threshold * 1.5, 1e-6)
            if ratio > max_ratio:
                max_ratio = ratio

            if abs_s >= self.anomaly_threshold:
                anomaly_ch.append(ch)
                warning_ch.append(ch)
            elif abs_s >= self.warning_threshold:
                warning_ch.append(ch)

        if valid_count == 0 and not self.state:
            return {
                "ewma_score": np.nan,
                "smoothed_values": smoothed,
                "warning_channels": [],
                "anomaly_channels": [],
                "is_warning": False,
                "is_anomaly": False,
            }

        score = float(np.clip(max_ratio, 0.0, 1.0))
        return {
            "ewma_score": score,
            "smoothed_values": smoothed,
            "warning_channels": warning_ch,
            "anomaly_channels": anomaly_ch,
            "is_warning": len(warning_ch) > 0,
            "is_anomaly": len(anomaly_ch) > 0,
        }


# ==============================================================================
# 3. PERSISTENCE GATE
# ==============================================================================

class PersistenceGate:
    """
    Persistence gate tracking consecutive abnormal samples.
    Distinguishes isolated noise spikes from sustained abnormal behavior.
    """

    def __init__(
        self,
        min_consecutive: int = 3,
    ):
        """
        Args:
            min_consecutive: Number of consecutive abnormal samples required
                             to satisfy sustained anomaly condition.
        """
        if min_consecutive < 1:
            raise ValueError(f"min_consecutive must be >= 1, got {min_consecutive}")
        self.min_consecutive = int(min_consecutive)
        self.counter: int = 0

    def reset(self) -> None:
        """Reset the consecutive abnormal counter to 0."""
        self.counter = 0

    def update(
        self,
        is_abnormal: bool,
        is_valid: bool = True,
    ) -> Dict[str, Any]:
        """
        Update the persistence counter.

        STRICT RULES:
        - Missing samples or INSUFFICIENT_DATA samples (is_valid=False) must NOT increment the counter.
        - Merely elevated but sub-abnormal samples must NOT increment the counter.
        - If normal and valid, counter resets to 0.

        Args:
            is_abnormal: True if instantaneous or EWMA evidence indicates abnormal deviation.
            is_valid: True if sample contains sufficient valid features.

        Returns:
            Dict containing:
                persistence_count: Current integer count.
                persistence_threshold: min_consecutive.
                persistence_satisfied: bool (count >= min_consecutive).
                persistence_score: float in [0.0, 1.0].
        """
        if not is_valid:
            # Insufficient or missing data: do not increment. Reset to avoid carrying over invalid state.
            self.counter = 0
        elif is_abnormal:
            self.counter += 1
        else:
            self.counter = 0

        satisfied = self.counter >= self.min_consecutive
        score = float(np.clip(self.counter / float(self.min_consecutive), 0.0, 1.0))

        return {
            "persistence_count": self.counter,
            "persistence_threshold": self.min_consecutive,
            "persistence_satisfied": satisfied,
            "persistence_score": score,
        }


# ==============================================================================
# 4. ISOLATION FOREST DETECTOR
# ==============================================================================

class IsolationForestDetector:
    """
    Multivariate unsupervised anomaly detector based on Isolation Forest.
    Fitted STRICTLY on healthy baseline normalized residuals.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.01,
        random_state: int = 42,
        warning_score_threshold: float = 0.5,
        anomaly_score_threshold: float = 0.7,
        feature_channels: Optional[List[str]] = None,
    ):
        """
        Args:
            n_estimators: Number of trees in isolation forest.
            contamination: Expected anomaly fraction in baseline (typically low, ~0.01).
            random_state: Fixed seed ensuring determinism.
            warning_score_threshold: Score threshold for warning (default 0.5).
            anomaly_score_threshold: Score threshold for anomaly (default 0.7).
            feature_channels: List of normalized residual features to evaluate.
        """
        self.n_estimators = int(n_estimators)
        self.contamination = float(contamination)
        self.random_state = int(random_state)
        self.warning_score_threshold = float(warning_score_threshold)
        self.anomaly_score_threshold = float(anomaly_score_threshold)
        self.feature_channels = list(feature_channels or PRIMARY_NORMALIZED_RESIDUAL_CHANNELS)

        self.model: Optional[IsolationForest] = None
        self.feature_medians: Dict[str, float] = {}
        self.is_fitted: bool = False

    def fit(
        self,
        healthy_residuals: Union[ResidualFrame, pd.DataFrame],
    ) -> "IsolationForestDetector":
        """
        Fit the Isolation Forest exclusively on healthy baseline normalized residuals.

        Args:
            healthy_residuals: ResidualFrame or DataFrame of healthy baseline telemetry.

        Raises:
            ValueError if insufficient healthy samples or features are present.
        """
        if isinstance(healthy_residuals, ResidualFrame):
            df = healthy_residuals.to_dataframe()
        else:
            df = healthy_residuals.copy()

        # Strict check: If fault_type column exists, assert only healthy data is used
        if "fault_type" in df.columns:
            non_healthy = df[df["fault_type"].astype(str).str.lower() != "none"]
            if len(non_healthy) > 0:
                raise ValueError(
                    f"Fault leakage detected! Isolation Forest must be fitted ONLY on healthy baseline "
                    f"data, but found {len(non_healthy)} fault-injected rows in training dataset."
                )

        # Extract primary normalized residual channels
        X = df[[c for c in self.feature_channels if c in df.columns]].copy()
        if len(X.columns) < len(self.feature_channels):
            raise ValueError(
                f"Missing required normalized residual channels. Expected {self.feature_channels}, "
                f"found {list(X.columns)}"
            )

        # Drop any rows with NaN in healthy baseline to fit clean baseline distribution
        X_clean = X.dropna()
        if len(X_clean) < 20:
            raise ValueError(
                f"Insufficient valid healthy baseline samples to train Isolation Forest. "
                f"Required >= 20, got {len(X_clean)}"
            )

        # Store channel medians from healthy training set for robust partial imputation
        # (Used ONLY if row meets min_valid_features but has 1 or 2 missing features, NEVER zero!)
        for ch in self.feature_channels:
            self.feature_medians[ch] = float(X_clean[ch].median())

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=1,
        )
        self.model.fit(X_clean[self.feature_channels].values)
        self.is_fitted = True
        return self

    def score_sample(
        self,
        features: Dict[str, float],
        is_valid: bool = True,
    ) -> Dict[str, Any]:
        """
        Score a single sample using the fitted Isolation Forest.

        Mapping from raw decision_function(X) to [0.0, 1.0]:
        score = 1.0 / (1.0 + exp(12.0 * df))
        - df > 0 (inlier)  -> score < 0.5 (nominal)
        - df = 0 (boundary)-> score = 0.5 (warning threshold)
        - df < 0 (outlier) -> score > 0.5 (abnormal)
        - df < -0.07       -> score > 0.7 (anomaly threshold)

        Args:
            features: Dictionary mapping normalized residual channels to values.
            is_valid: Boolean indicating whether sample meets minimum-valid-features.

        Returns:
            Dict containing:
                isolation_score: Float in [0.0, 1.0] (or NaN if invalid/unfitted).
                is_warning: bool.
                is_anomaly: bool.
        """
        if not self.is_fitted or self.model is None or not is_valid:
            return {
                "isolation_score": np.nan,
                "is_warning": False,
                "is_anomaly": False,
            }

        # Build feature vector; replace missing with healthy baseline median (NOT zero)
        x_vec = []
        for ch in self.feature_channels:
            val = features.get(ch, np.nan)
            if val is None or np.isnan(val):
                val = self.feature_medians.get(ch, 0.0)
            x_vec.append(float(val))

        X = np.array(x_vec).reshape(1, -1)
        df_val = float(self.model.decision_function(X)[0])

        # Deterministic sigmoid mapping
        score = float(np.clip(1.0 / (1.0 + np.exp(12.0 * df_val)), 0.0, 1.0))
        is_warn = score >= self.warning_score_threshold
        is_anom = score >= self.anomaly_score_threshold

        return {
            "isolation_score": round(score, 4),
            "is_warning": is_warn,
            "is_anomaly": is_anom,
        }
