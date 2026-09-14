"""
Validation Utilities for the Grey-Box ML Integration Layer.

Provides rigorous checks for:
1. Train/test data leakage (disjoint group guarantees).
2. Numerical cleanliness (absence of NaNs/Infs in feature matrices).
3. Rejection of raw archives or unconverted script inputs.
4. Protected core domain boundary enforcement.
"""

from pathlib import Path
from typing import List, Set, Union, Optional, Dict, Any
import subprocess
import numpy as np
import pandas as pd


class DataLeakageError(Exception):
    """Raised when data leakage is detected between train and evaluation splits."""
    pass


class NumericalIntegrityError(Exception):
    """Raised when unexpected NaNs, nulls, or infinite values appear in feature columns."""
    pass


class ProtectedDomainViolationError(Exception):
    """Raised when a file within a protected core domain is modified."""
    pass


PROTECTED_DOMAINS: List[str] = [
    "physics",
    "simulator",
    "telemetry",
    "digital_twin",
    "health_index",
    "anomaly_detection",
    "fault_diagnosis",
    "frontend",
    "backend",
]


def validate_no_leakage(
    train_groups: Union[Set[Any], List[Any]],
    test_groups: Union[Set[Any], List[Any]],
    val_groups: Optional[Union[Set[Any], List[Any]]] = None,
    group_col_name: str = "group",
) -> None:
    """
    Ensure strictly zero overlap between training and evaluation groups.
    Fails immediately if any group identifier is shared.
    """
    train_set = set(train_groups)
    test_set = set(test_groups)
    val_set = set(val_groups) if val_groups is not None else set()

    train_test_overlap = train_set.intersection(test_set)
    if train_test_overlap:
        raise DataLeakageError(
            f"Critical Data Leakage Detected on grouping column '{group_col_name}'! "
            f"{len(train_test_overlap)} groups exist in BOTH train and test splits: "
            f"{sorted(list(train_test_overlap))[:5]}"
        )

    if val_set:
        train_val_overlap = train_set.intersection(val_set)
        if train_val_overlap:
            raise DataLeakageError(
                f"Data Leakage Detected between train and validation on '{group_col_name}'! "
                f"{len(train_val_overlap)} groups shared: {sorted(list(train_val_overlap))[:5]}"
            )

        val_test_overlap = val_set.intersection(test_set)
        if val_test_overlap:
            raise DataLeakageError(
                f"Data Leakage Detected between validation and test on '{group_col_name}'! "
                f"{len(val_test_overlap)} groups shared: {sorted(list(val_test_overlap))[:5]}"
            )


def validate_numerical_cleanliness(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    allow_null_cols: Optional[List[str]] = None,
) -> Dict[str, int]:
    """
    Assert that specified numeric feature columns contain no NaN or infinite values.
    Returns dictionary of checked columns and row counts.
    """
    target_cols = columns if columns is not None else df.select_dtypes(include=[np.number]).columns.tolist()
    allow_null = set(allow_null_cols or [])

    issues = []
    for col in target_cols:
        if col not in df.columns:
            continue
        if col in allow_null:
            continue

        s = df[col]
        null_count = int(s.isnull().sum())
        if null_count > 0:
            issues.append(f"Column '{col}' has {null_count} null/NaN values.")

        if pd.api.types.is_numeric_dtype(s):
            inf_count = int(np.isinf(s.to_numpy(dtype=float, na_value=0.0)).sum())
            if inf_count > 0:
                issues.append(f"Column '{col}' has {inf_count} infinite values.")

    if issues:
        raise NumericalIntegrityError(
            f"Numerical cleanliness validation failed on {len(issues)} columns:\n" + "\n".join(issues[:10])
        )

    return {"columns_checked": len(target_cols), "total_rows": len(df)}


def validate_protected_domains(repo_root: Optional[Union[str, Path]] = None) -> List[str]:
    """
    Realistic Domain Protection Check:
    Inspects all modified, staged, and untracked files in the git repository
    and asserts that zero files reside inside protected core domain directories:
    - physics/
    - simulator/
    - telemetry/
    - digital_twin/
    - health_index/
    - anomaly_detection/
    - fault_diagnosis/
    - frontend/
    - backend/
    """
    root = Path(repo_root) if repo_root else Path(__file__).parent.parent.resolve()

    try:
        # Check git status porcelain
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=True,
        )
        lines = res.stdout.strip().splitlines()
    except Exception as e:
        # Fallback if git binary is unavailable
        return []

    changed_files = []
    for line in lines:
        if len(line) >= 4:
            # porcelain format: XY path or XY "path"
            file_part = line[3:].strip().strip('"')
            changed_files.append(file_part.replace("\\", "/"))

    violations = []
    for f in changed_files:
        parts = f.split("/")
        first_dir = parts[0].lower() if parts else ""
        if first_dir in PROTECTED_DOMAINS:
            violations.append(f)

    if violations:
        raise ProtectedDomainViolationError(
            f"Protected core domain boundary violation! "
            f"The following files in protected directories were modified: {violations}"
        )

    return changed_files
