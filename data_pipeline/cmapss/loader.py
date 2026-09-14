"""
C-MAPSS dataset loader.

Loads the NASA C-MAPSS turbofan degradation simulation datasets (FD001-FD004).
Data is space-delimited text with 26 columns (no header).
"""

import os
import pandas as pd
from typing import Optional

# Standard C-MAPSS column names — using cmapss_ prefix for ALL sensor columns
CMAPSS_COLUMNS = [
    "unit_number",
    "time_cycles",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
] + [f"cmapss_s_{i}" for i in range(1, 22)]


def get_cmapss_data_dir() -> str:
    """Return the path to the extracted CMAPSSData directory."""
    base = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "raw", "cmapss",
        "6.+Turbofan+Engine+Degradation+Simulation+Data+Set",
        "6. Turbofan Engine Degradation Simulation Data Set",
        "CMAPSSData",
    )
    return os.path.normpath(base)


def load_train(subset: str = "FD001", data_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Load a C-MAPSS training file.

    Args:
        subset: One of FD001, FD002, FD003, FD004.
        data_dir: Override data directory path.

    Returns:
        DataFrame with cmapss_s_N column naming (never cht/egt/etc).
    """
    if data_dir is None:
        data_dir = get_cmapss_data_dir()
    filepath = os.path.join(data_dir, f"train_{subset}.txt")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"C-MAPSS train file not found: {filepath}")
    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=CMAPSS_COLUMNS)
    df["subset"] = subset
    df["split"] = "train"
    return df


def load_test(subset: str = "FD001", data_dir: Optional[str] = None) -> pd.DataFrame:
    """Load a C-MAPSS test file."""
    if data_dir is None:
        data_dir = get_cmapss_data_dir()
    filepath = os.path.join(data_dir, f"test_{subset}.txt")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"C-MAPSS test file not found: {filepath}")
    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=CMAPSS_COLUMNS)
    df["subset"] = subset
    df["split"] = "test"
    return df


def load_rul(subset: str = "FD001", data_dir: Optional[str] = None) -> pd.Series:
    """Load C-MAPSS RUL ground truth for the test set."""
    if data_dir is None:
        data_dir = get_cmapss_data_dir()
    filepath = os.path.join(data_dir, f"RUL_{subset}.txt")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"C-MAPSS RUL file not found: {filepath}")
    rul = pd.read_csv(filepath, header=None, names=["rul"])
    return rul["rul"]


def list_available_subsets(data_dir: Optional[str] = None) -> list:
    """List which FD00x subsets are available on disk."""
    if data_dir is None:
        data_dir = get_cmapss_data_dir()
    available = []
    for subset in ["FD001", "FD002", "FD003", "FD004"]:
        if os.path.exists(os.path.join(data_dir, f"train_{subset}.txt")):
            available.append(subset)
    return available
