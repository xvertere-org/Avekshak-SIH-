"""
NUST preprocessing — quality checks and feature normalization.

RESTRICTION: Do NOT invent CHT, EGT, oil_pressure, or other channels
that are not actually measured in this dataset. Original column names preserved.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def check_data_quality(df: pd.DataFrame) -> Dict:
    """
    Run quality checks on a loaded NUST CSV DataFrame.

    Returns a dict of quality metrics.
    """
    quality = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "missing_values": {},
        "duplicate_rows": 0,
        "malformed_rows": 0,
        "constant_columns": [],
    }

    # Missing values
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        if null_count > 0:
            quality["missing_values"][col] = null_count

    # Duplicates
    quality["duplicate_rows"] = int(df.duplicated().sum())

    # Constant columns (zero variance)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].nunique() <= 1:
            quality["constant_columns"].append(col)

    return quality


def identify_operating_region(df: pd.DataFrame) -> str:
    """
    Attempt to classify the operating region as steady-state or transient.

    Uses the coefficient of variation of 'Demand 1' and 'Control 1'.
    """
    for col in ["Demand 1", "Control 1"]:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) > 0 and series.mean() != 0:
                cv = series.std() / abs(series.mean())
                if cv > 0.1:
                    return "transient"
    return "steady_state"
