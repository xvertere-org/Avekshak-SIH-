"""
Feature extraction and data splitting for Phase 8 Fault Diagnosis.

Responsibilities:
- Extract 16 classifier features from ResidualFrame data
  (7 normalized residuals + 3 continuous context + 6 one-hot mission_phase)
- One-hot encode mission_phase with deterministic handling of unseen phases
- Enforce leakage prevention (no fault_type, fault_severity, sensor metadata in features)
- Evaluate per-sample data quality (sufficient valid residuals)
- Provide grouped stratified splitting by mission_run_id

Missing data strategy:
- XGBoost tree_method="hist" handles NaN natively — no imputation
- NaN is NEVER converted to 0
- Samples with <4/7 core residuals valid → INSUFFICIENT_DATA
"""

from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
import pandas as pd

from fault_diagnosis.schema import DiagnosisDataQuality


# ── Feature definitions ──────────────────────────────────────────────────────

DIAGNOSTIC_FEATURES: List[str] = [
    "rpm_norm_residual",
    "cht_norm_residual",
    "egt_norm_residual",
    "oil_temp_norm_residual",
    "oil_pressure_norm_residual",
    "fuel_flow_norm_residual",
    "vibration_norm_residual",
]

CONTEXT_CONTINUOUS_FEATURES: List[str] = [
    "throttle",
    "altitude",
    "load",
]

# Minimum valid (non-NaN) core residuals to produce a diagnosis
MIN_VALID_RESIDUALS: int = 4

# Expected total feature count after one-hot phase encoding: 7 + 3 + 6 = 16
EXPECTED_FEATURE_COUNT: int = 16

# Columns that must NEVER appear as classifier features (leakage prevention)
FORBIDDEN_FEATURE_COLUMNS: Set[str] = {
    "fault_type",
    "fault_severity",
    "generation_severity",
    "generation_sensor_channel",
    "generation_sensor_mode",
    "mission_run_id",
    "sensor_channel",
    "sensor_mode",
}


from simulator.subsystems.mission import FlightPhase

# 6 Canonical mission phases from simulator
CANONICAL_MISSION_PHASES: List[str] = [
    FlightPhase.TAKEOFF.value,
    FlightPhase.CLIMB.value,
    FlightPhase.CRUISE.value,
    FlightPhase.LOITER.value,
    FlightPhase.DESCENT.value,
    FlightPhase.LANDING.value,
]


class FeatureExtractor:
    """
    Extracts and validates classifier features from ResidualFrame data.

    Usage:
        extractor = FeatureExtractor()
        extractor.fit(train_df)          # learn one-hot categories from training data
        X_train = extractor.transform(train_df)
        X_test = extractor.transform(test_df)   # same schema guaranteed

    The fit() method defines the final 16-column schema.
    The transform() method conforms to it unconditionally.
    """

    def __init__(self, min_valid_residuals: int = MIN_VALID_RESIDUALS):
        self.min_valid_residuals = min_valid_residuals
        self._fitted = False
        self._phase_categories: List[str] = list(CANONICAL_MISSION_PHASES)
        self._feature_columns: List[str] = (
            list(DIAGNOSTIC_FEATURES)
            + list(CONTEXT_CONTINUOUS_FEATURES)
            + [f"phase_{cat}" for cat in CANONICAL_MISSION_PHASES]
        )

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def feature_columns(self) -> List[str]:
        """The final ordered list of feature column names after fitting."""
        if not self._fitted:
            raise RuntimeError("FeatureExtractor must be fitted before accessing feature_columns.")
        return list(self._feature_columns)

    @property
    def feature_names_(self) -> List[str]:
        """Alias for feature_columns."""
        return self.feature_columns

    @property
    def feature_count(self) -> int:
        """Total number of features in the output schema."""
        if not self._fitted:
            raise RuntimeError("FeatureExtractor must be fitted before accessing feature_count.")
        return len(self._feature_columns)

    def fit(self, df: pd.DataFrame) -> "FeatureExtractor":
        """
        Learn one-hot encoding categories from training data.

        Only the training split should be passed here. Validation/test data
        must NOT influence the feature schema.

        Args:
            df: Training DataFrame containing a 'mission_phase' column.

        Returns:
            self
        """
        self._phase_categories = list(CANONICAL_MISSION_PHASES)
        self._feature_columns = (
            list(DIAGNOSTIC_FEATURES)
            + list(CONTEXT_CONTINUOUS_FEATURES)
            + [f"phase_{cat}" for cat in self._phase_categories]
        )
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features conforming to the fitted schema.

        Unseen mission_phase values during inference: all one-hot phase columns
        are set to 0 (no phase active). The method never crashes, never adds
        new columns, and never alters the fitted schema.

        Args:
            df: DataFrame with ResidualFrame columns.

        Returns:
            DataFrame with exactly the fitted feature columns.
            NaN residuals are preserved (XGBoost handles them natively).
        """
        if not self._fitted:
            raise RuntimeError("FeatureExtractor must be fitted before calling transform().")

        result = pd.DataFrame(index=df.index)

        # 1. Core diagnostic features (7 normalized residuals, NaN preserved)
        for col in DIAGNOSTIC_FEATURES:
            if col in df.columns:
                result[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
            else:
                result[col] = np.nan

        # 2. Context continuous features
        for col in CONTEXT_CONTINUOUS_FEATURES:
            if col in df.columns:
                result[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
            else:
                result[col] = np.nan

        # 3. One-hot encode mission_phase
        for cat in self._phase_categories:
            col_name = f"phase_{cat}"
            if "mission_phase" in df.columns:
                result[col_name] = (df["mission_phase"].astype(str) == cat).astype(int)
            else:
                result[col_name] = 0

        # Verify schema matches exactly
        result = result[self._feature_columns]

        # Verify no leakage columns
        for col in result.columns:
            if col in FORBIDDEN_FEATURE_COLUMNS:
                raise ValueError(
                    f"LEAKAGE DETECTED: forbidden column '{col}' found in feature matrix."
                )

        return result

    def check_data_quality(self, features: pd.DataFrame) -> pd.Series:
        """
        Evaluate per-sample data quality based on valid residual count.

        Args:
            features: Transformed feature DataFrame.

        Returns:
            Series of DiagnosisDataQuality values.
        """
        residual_cols = [c for c in DIAGNOSTIC_FEATURES if c in features.columns]
        valid_counts = features[residual_cols].notna().sum(axis=1)
        quality = pd.Series(
            DiagnosisDataQuality.VALID.value,
            index=features.index,
            dtype=str,
        )
        quality[valid_counts < self.min_valid_residuals] = (
            DiagnosisDataQuality.INSUFFICIENT_DATA.value
        )
        return quality


# ── Grouped Stratified Splitting ─────────────────────────────────────────────


def grouped_stratified_split(
    df: pd.DataFrame,
    group_col: str = "mission_run_id",
    stratify_col: str = "fault_type",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset so that:
    1. No mission_run_id appears in more than one split.
    2. Each split maintains approximate fault_type class balance.
    3. Every class with >=3 runs has at least 1 run in each split.

    Algorithm:
    - For each fault_type class independently:
      - Collect all unique mission_run_ids.
      - Deterministically shuffle with fixed seed.
      - Allocate 70% train, 15% val, 15% test (at run level).
      - Guarantee at least 1 run per split when class has >=3 runs.

    Args:
        df: Dataset with group_col and stratify_col.
        group_col: Column identifying independent mission runs.
        stratify_col: Column identifying fault class for stratification.
        train_ratio: Fraction of runs for training.
        val_ratio: Fraction of runs for validation.
        test_ratio: Fraction of runs for test.
        random_state: Fixed seed for reproducibility.

    Returns:
        (train_df, val_df, test_df) tuple of DataFrames.
    """
    rng = np.random.RandomState(random_state)

    # Get one label per group
    group_labels = df.groupby(group_col)[stratify_col].first().reset_index()

    train_groups: List[str] = []
    val_groups: List[str] = []
    test_groups: List[str] = []

    for class_label in sorted(group_labels[stratify_col].unique()):
        class_groups = group_labels[
            group_labels[stratify_col] == class_label
        ][group_col].tolist()
        rng.shuffle(class_groups)

        n = len(class_groups)
        if n >= 3:
            # Guarantee at least 1 in each split
            n_train = max(1, int(round(n * train_ratio)))
            n_val = max(1, int(round(n * val_ratio)))
            n_test = max(1, n - n_train - n_val)

            # Adjust if allocation exceeds total
            if n_train + n_val + n_test > n:
                n_train = n - n_val - n_test
                if n_train < 1:
                    n_train = 1
                    n_val = max(1, (n - 1) // 2)
                    n_test = n - n_train - n_val
        elif n == 2:
            # Can only guarantee 2 splits; put 1 in train, 1 in test
            n_train = 1
            n_val = 0
            n_test = 1
        else:
            # Only 1 run: goes to train
            n_train = 1
            n_val = 0
            n_test = 0

        train_groups.extend(class_groups[:n_train])
        val_groups.extend(class_groups[n_train:n_train + n_val])
        test_groups.extend(class_groups[n_train + n_val:])

    # Verify no overlap
    train_set = set(train_groups)
    val_set = set(val_groups)
    test_set = set(test_groups)
    assert train_set.isdisjoint(val_set), "Train/val overlap detected!"
    assert train_set.isdisjoint(test_set), "Train/test overlap detected!"
    assert val_set.isdisjoint(test_set), "Val/test overlap detected!"

    train_df = df[df[group_col].isin(train_set)].copy()
    val_df = df[df[group_col].isin(val_set)].copy()
    test_df = df[df[group_col].isin(test_set)].copy()

    return train_df, val_df, test_df


def compute_sample_weights(
    y_train: pd.Series,
) -> np.ndarray:
    """
    Compute sample weights inversely proportional to class frequency.

    weight_for_class_c = total_samples / (num_classes * samples_in_class_c)

    Applied ONLY during training. Validation/test sets are never weighted.

    Args:
        y_train: Training labels (fault_type strings).

    Returns:
        Array of per-sample weights.
    """
    class_counts = y_train.value_counts()
    total = len(y_train)
    num_classes = len(class_counts)

    weight_map = {}
    for cls, count in class_counts.items():
        weight_map[cls] = total / (num_classes * count)

    weights = y_train.map(weight_map).values.astype(float)
    return weights


# Aliases
split_by_mission_run = grouped_stratified_split
compute_class_weights = compute_sample_weights


def get_split_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_col: str = "mission_run_id",
    stratify_col: str = "fault_type",
) -> Dict[str, Any]:
    """
    Report per-class run counts and sample counts for each split.
    Warns when a class has too few independent runs for robust evaluation.
    """
    summary: Dict[str, Any] = {"splits": {}, "warnings": []}

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_info: Dict[str, Dict[str, int]] = {}
        for cls in sorted(
            set(train_df[stratify_col].unique())
            | set(val_df[stratify_col].unique())
            | set(test_df[stratify_col].unique())
        ):
            cls_data = split_df[split_df[stratify_col] == cls]
            runs = cls_data[group_col].nunique()
            samples = len(cls_data)
            split_info[cls] = {"runs": runs, "samples": samples}

            # Warn about small test/val run counts
            if split_name in ("val", "test") and 0 < runs <= 1:
                summary["warnings"].append(
                    f"{split_name}: class '{cls}' has only {runs} independent run(s). "
                    f"Evaluation metrics for this class are based on a single trajectory "
                    f"and should be interpreted with caution."
                )
            elif split_name in ("val", "test") and runs == 0:
                summary["warnings"].append(
                    f"{split_name}: class '{cls}' has 0 runs. "
                    f"This class cannot be evaluated in this split."
                )

        summary["splits"][split_name] = split_info

    return summary
