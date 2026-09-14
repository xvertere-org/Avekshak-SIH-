"""
Leakage-Safe Group-Aware Data Splitting Strategy.

Implements rigorous group-level partitioning to ensure zero data leakage:
- Units, bearings, batteries, and source files never span multiple partitions.
- CWRU is guarded against conventional train/val/test splits due to the 3-file limitation.
- Deterministic seeding ensures exact reproducibility.
- Verifies and reports class distribution and group assignments.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
import pandas as pd

from ml.dataset_registry import DatasetRegistry, GroupColumnNotFoundError
from ml.validation import validate_no_leakage, DataLeakageError


class InsufficientGroupsError(Exception):
    """Raised when a dataset does not possess enough independent groups for the requested split."""
    pass


@dataclass
class SplitResult:
    """Detailed summary of a leakage-safe group split."""
    dataset_name: str
    group_column: str
    num_total_groups: int
    num_train_groups: int
    num_val_groups: int
    num_test_groups: int
    num_train_rows: int
    num_val_rows: int
    num_test_rows: int
    train_groups: List[Any]
    val_groups: List[Any]
    test_groups: List[Any]
    train_class_distribution: Dict[str, int]
    test_class_distribution: Dict[str, int]
    val_class_distribution: Dict[str, int]
    leakage_detected: bool
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GroupSplitter:
    """
    Partitions datasets by groups (e.g. unit_number, bearing_id, battery_id, source_file)
    to guarantee zero leakage across training and evaluation sets.
    """

    def __init__(self, registry: Optional[DatasetRegistry] = None):
        self.registry = registry or DatasetRegistry()

    def split(
        self,
        df: pd.DataFrame,
        dataset_id: str,
        group_column: Optional[str] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        label_column: Optional[str] = None,
        seed: int = 42,
        allow_cwru_demo: bool = False,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitResult]:
        """
        Perform a group-aware, leakage-safe split into (train_df, val_df, test_df).

        Guarantees:
        - train_groups ∩ val_groups = ∅
        - train_groups ∩ test_groups = ∅
        - val_groups ∩ test_groups = ∅
        """
        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-5:
            raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

        # Resolve grouping column
        if group_column is None:
            group_column = self.registry.resolve_grouping_column(dataset_id, list(df.columns))

        if group_column not in df.columns:
            raise GroupColumnNotFoundError(
                f"Resolved grouping column '{group_column}' does not exist in DataFrame columns: {list(df.columns)}"
            )

        unique_groups = sorted(df[group_column].dropna().unique())
        n_groups = len(unique_groups)

        # Explicit CWRU limitation handling
        if dataset_id.lower() == "cwru":
            if not allow_cwru_demo:
                raise InsufficientGroupsError(
                    "CWRU limitation: The current processed CWRU dataset contains only 3 independent source files "
                    f"({unique_groups}). Standard random train/val/test splitting is invalid and will create "
                    "severe distribution distortion. Use primarily for feature validation and controlled demonstrations, "
                    "or pass `allow_cwru_demo=True` to run an explicit 2-train / 1-test controlled demonstration."
                )
            # Controlled demonstration split: 2 groups train, 0 val, 1 test
            rng = np.random.default_rng(seed)
            shuffled_groups = rng.permutation(unique_groups).tolist()
            train_groups = shuffled_groups[:2]
            val_groups = []
            test_groups = shuffled_groups[2:]
            notes = "CWRU controlled demonstration split (2 train files, 1 test file; 0 val)."
        else:
            # General group split
            min_required_groups = 2 if val_ratio == 0 else 3
            if n_groups < min_required_groups:
                raise InsufficientGroupsError(
                    f"Dataset '{dataset_id}' only has {n_groups} independent groups in '{group_column}'. "
                    f"At least {min_required_groups} independent groups are required for the requested partition."
                )

            rng = np.random.default_rng(seed)

            # Check if group-stratified splitting is possible
            can_stratify = False
            group_label_map = None
            if label_column and label_column in df.columns:
                group_label_map = df.groupby(group_column)[label_column].agg(lambda s: s.mode()[0])
                label_counts = group_label_map.value_counts()
                if (label_counts >= 2).all() and len(label_counts) > 1:
                    can_stratify = True

            if can_stratify and group_label_map is not None:
                train_groups = []
                val_groups = []
                test_groups = []
                for lbl in sorted(group_label_map.unique()):
                    lbl_groups = rng.permutation(group_label_map[group_label_map == lbl].index.tolist()).tolist()
                    n_g = len(lbl_groups)
                    n_tr = max(1, int(round(train_ratio * n_g)))
                    n_va = int(round(val_ratio * n_g)) if val_ratio > 0 else 0
                    n_te = n_g - n_tr - n_va
                    if test_ratio > 0 and n_te == 0 and n_tr > 1:
                        n_tr -= 1
                        n_te = 1
                    train_groups.extend(lbl_groups[:n_tr])
                    val_groups.extend(lbl_groups[n_tr:n_tr + n_va])
                    test_groups.extend(lbl_groups[n_tr + n_va:])
                train_groups = sorted(train_groups)
                val_groups = sorted(val_groups)
                test_groups = sorted(test_groups)
                notes = "Stratified leakage-safe group partition."
            else:
                shuffled_groups = rng.permutation(unique_groups).tolist()
                n_train = max(1, int(round(train_ratio * n_groups)))
                n_val = int(round(val_ratio * n_groups)) if val_ratio > 0 else 0
                n_test = n_groups - n_train - n_val

                # Ensure at least 1 test group if test_ratio > 0
                if test_ratio > 0 and n_test == 0:
                    if n_train > 1:
                        n_train -= 1
                        n_test = 1
                    elif n_val > 1:
                        n_val -= 1
                        n_test = 1

                train_groups = sorted(shuffled_groups[:n_train])
                val_groups = sorted(shuffled_groups[n_train:n_train + n_val])
                test_groups = sorted(shuffled_groups[n_train + n_val:])
                notes = "Standard leakage-safe group partition."

        # Fail-fast verification of disjoint sets
        validate_no_leakage(train_groups, test_groups, val_groups, group_col_name=group_column)

        # Slice DataFrames
        train_df = df[df[group_column].isin(train_groups)].copy()
        val_df = df[df[group_column].isin(val_groups)].copy()
        test_df = df[df[group_column].isin(test_groups)].copy()

        # Class distribution reporting
        train_dist = {}
        val_dist = {}
        test_dist = {}

        if label_column and label_column in df.columns:
            train_dist = {str(k): int(v) for k, v in train_df[label_column].value_counts().items()}
            val_dist = {str(k): int(v) for k, v in val_df[label_column].value_counts().items()}
            test_dist = {str(k): int(v) for k, v in test_df[label_column].value_counts().items()}

        result = SplitResult(
            dataset_name=dataset_id,
            group_column=group_column,
            num_total_groups=n_groups,
            num_train_groups=len(train_groups),
            num_val_groups=len(val_groups),
            num_test_groups=len(test_groups),
            num_train_rows=len(train_df),
            num_val_rows=len(val_df),
            num_test_rows=len(test_df),
            train_groups=train_groups,
            val_groups=val_groups,
            test_groups=test_groups,
            train_class_distribution=train_dist,
            test_class_distribution=test_dist,
            val_class_distribution=val_dist,
            leakage_detected=False,
            notes=notes,
        )

        return train_df, val_df, test_df, result
