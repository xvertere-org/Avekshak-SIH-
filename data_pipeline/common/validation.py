"""
Data validation utilities for the pipeline.

Provides checks for schema conformance, missing values, duplicates,
label validity, and unsafe column name detection.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Set
from .schemas import UNSAFE_CMAPSS_COLUMN_NAMES


def check_no_unsafe_cmapss_columns(df: pd.DataFrame, dataset_name: str = "cmapss") -> List[str]:
    """Return list of unsafe column names found in a C-MAPSS DataFrame."""
    violations = []
    for col in df.columns:
        if col.lower() in UNSAFE_CMAPSS_COLUMN_NAMES:
            violations.append(f"UNSAFE column '{col}' found in {dataset_name} output")
    return violations


def check_missing_values(df: pd.DataFrame) -> dict:
    """Return per-column missing value counts."""
    missing = df.isnull().sum()
    return {col: int(cnt) for col, cnt in missing.items() if cnt > 0}


def check_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> int:
    """Return number of duplicate rows."""
    return int(df.duplicated(subset=subset).sum())


def check_empty_dataframe(df: pd.DataFrame) -> bool:
    """Return True if DataFrame is empty."""
    return len(df) == 0


def check_rul_validity(rul_values: pd.Series) -> List[str]:
    """Validate RUL values: should be non-negative and finite."""
    issues = []
    if rul_values.isnull().any():
        issues.append(f"RUL contains {int(rul_values.isnull().sum())} null values")
    if (rul_values < 0).any():
        issues.append(f"RUL contains {int((rul_values < 0).sum())} negative values")
    if not np.isfinite(rul_values.dropna()).all():
        issues.append("RUL contains non-finite values")
    return issues


def check_no_train_test_leakage(
    train_ids: Set[str], test_ids: Set[str], id_name: str = "unit"
) -> List[str]:
    """Check that train and test sets have no overlapping IDs."""
    overlap = train_ids & test_ids
    if overlap:
        return [f"Train/test leakage: {id_name} IDs {overlap} appear in both sets"]
    return []


def check_timestamp_ordering(df: pd.DataFrame, ts_col: str = "timestamp") -> List[str]:
    """Check that timestamps are monotonically non-decreasing within groups."""
    issues = []
    if ts_col not in df.columns:
        issues.append(f"Timestamp column '{ts_col}' not found")
        return issues
    if not df[ts_col].is_monotonic_increasing:
        # Check per-group if there's a group column
        issues.append(f"Timestamps in '{ts_col}' are not globally monotonic")
    return issues


def validate_schema(df: pd.DataFrame, required_columns: List[str]) -> List[str]:
    """Check that all required columns are present."""
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        return [f"Missing required columns: {missing}"]
    return []
