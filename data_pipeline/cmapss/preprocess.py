"""
C-MAPSS preprocessing — RUL label generation and sensor filtering.

Uses the standard piece-wise linear RUL convention:
  - In training: RUL = max_cycle_for_unit - current_cycle
  - Optionally capped at a maximum RUL threshold (e.g., 125 cycles)
"""

import pandas as pd
import numpy as np
from typing import Optional, List
from .loader import load_train, load_test, load_rul, list_available_subsets

# Sensors known to be nearly constant across operating conditions in FD001/FD003
# (op_setting_3 ≈ 100, cmapss_s_1, cmapss_s_5, cmapss_s_10, cmapss_s_16, cmapss_s_18, cmapss_s_19)
NEAR_CONSTANT_SENSORS_FD001 = [
    "cmapss_s_1", "cmapss_s_5", "cmapss_s_10",
    "cmapss_s_16", "cmapss_s_18", "cmapss_s_19",
]


def generate_train_rul(df: pd.DataFrame, max_rul: Optional[int] = 125) -> pd.DataFrame:
    """
    Generate RUL labels for training data.

    For each unit, RUL = max_cycle - current_cycle, optionally capped.
    """
    df = df.copy()
    max_cycles = df.groupby("unit_number")["time_cycles"].max().reset_index()
    max_cycles.columns = ["unit_number", "max_cycle"]
    df = df.merge(max_cycles, on="unit_number")
    df["rul_label"] = df["max_cycle"] - df["time_cycles"]
    if max_rul is not None:
        df["rul_label"] = df["rul_label"].clip(upper=max_rul)
    df.drop(columns=["max_cycle"], inplace=True)
    return df


def generate_test_rul(df: pd.DataFrame, rul_truth: pd.Series) -> pd.DataFrame:
    """
    Generate RUL labels for test data using ground-truth RUL values.

    The ground truth gives the RUL at the last cycle of each test unit.
    """
    df = df.copy()
    units = df["unit_number"].unique()
    if len(units) != len(rul_truth):
        raise ValueError(
            f"Number of test units ({len(units)}) != RUL truth length ({len(rul_truth)})"
        )

    rul_map = dict(zip(sorted(units), rul_truth.values))
    max_cycles = df.groupby("unit_number")["time_cycles"].max().to_dict()

    rul_labels = []
    for _, row in df.iterrows():
        u = row["unit_number"]
        remaining_at_end = rul_map[u]
        cycles_left_in_data = max_cycles[u] - row["time_cycles"]
        rul_labels.append(remaining_at_end + cycles_left_in_data)
    df["rul_label"] = rul_labels
    return df


def remove_constant_sensors(
    df: pd.DataFrame,
    sensors_to_remove: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Remove documented near-constant sensors."""
    if sensors_to_remove is None:
        sensors_to_remove = NEAR_CONSTANT_SENSORS_FD001
    cols_to_drop = [c for c in sensors_to_remove if c in df.columns]
    return df.drop(columns=cols_to_drop)


def preprocess_subset(
    subset: str = "FD001",
    data_dir: Optional[str] = None,
    max_rul: int = 125,
    remove_constant: bool = True,
) -> dict:
    """
    Full preprocessing pipeline for a single C-MAPSS subset.

    Returns dict with 'train' and 'test' DataFrames.
    """
    train_df = load_train(subset, data_dir)
    test_df = load_test(subset, data_dir)
    rul_truth = load_rul(subset, data_dir)

    train_df = generate_train_rul(train_df, max_rul=max_rul)
    test_df = generate_test_rul(test_df, rul_truth)

    if remove_constant and subset in ["FD001", "FD003"]:
        train_df = remove_constant_sensors(train_df)
        test_df = remove_constant_sensors(test_df)

    return {"train": train_df, "test": test_df, "subset": subset}
