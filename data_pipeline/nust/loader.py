"""
NUST IC-Engine Journal Bearing dataset loader.

CSV structure (20 columns):
  Time, Demand 1, Control 1, Output Drive 1,
  Channel 1, Channel 2, Channel 3, Channel 4,
  Channel 1 Kurtosis, Channel 2 Kurtosis, Channel 3 Kurtosis, Channel 4 Kurtosis,
  Rear Input 1..8

Directory structure encodes metadata:
  {Healthy|Faulty} Bearings Dataset / {1000|1500|2000} RPM / {0|50|100}% Humidity / {temp} deg Celsius /

File types: CSV (134), DOCX (201), RTF (346)
Only CSV files are machine-readable and processed here.
"""

import os
import glob
import re
import pandas as pd
from typing import Optional, List, Dict, Tuple

NUST_CSV_COLUMNS = [
    "Time", "Demand 1", "Control 1", "Output Drive 1",
    "Channel 1", "Channel 2", "Channel 3", "Channel 4",
    "Channel 1 Kurtosis", "Channel 2 Kurtosis", "Channel 3 Kurtosis", "Channel 4 Kurtosis",
    "Rear Input 1", "Rear Input 2", "Rear Input 3", "Rear Input 4",
    "Rear Input 5", "Rear Input 6", "Rear Input 7", "Rear Input 8",
]


def get_nust_data_dir() -> str:
    """Return path to NUST raw data directory."""
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "nust")
    )


def list_csv_files(data_dir: Optional[str] = None) -> List[str]:
    """List all CSV files in the NUST dataset tree."""
    if data_dir is None:
        data_dir = get_nust_data_dir()
    return sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))


def parse_path_metadata(filepath: str) -> dict:
    """
    Extract metadata from the NUST file path.

    Example path: .../Faulty Bearings Dataset/1000 RPM/0% Humidity/-10 deg Celsius/file.csv
    """
    parts = filepath.replace("\\", "/").split("/")
    metadata = {
        "bearing_condition": None,
        "rpm": None,
        "humidity": None,
        "temperature_celsius": None,
        "dataset_variant": None,
    }

    for part in parts:
        if "Healthy" in part:
            metadata["bearing_condition"] = "healthy"
        elif "Faulty" in part:
            metadata["bearing_condition"] = "faulty"

        rpm_match = re.search(r"(\d+)\s*RPM", part)
        if rpm_match:
            metadata["rpm"] = int(rpm_match.group(1))

        hum_match = re.search(r"(\d+)%\s*Humidity", part)
        if hum_match:
            metadata["humidity"] = int(hum_match.group(1))

        temp_match = re.search(r"(-?\d+)\s*(?:deg|°)\s*[Cc]", part)
        if temp_match:
            metadata["temperature_celsius"] = int(temp_match.group(1))

        if "Dataset 2" in part or "Dataset  2" in part:
            metadata["dataset_variant"] = "v2"
        elif "Additional" in part:
            metadata["dataset_variant"] = "additional"

    return metadata


def load_csv_file(filepath: str) -> Tuple[pd.DataFrame, dict]:
    """
    Load a single NUST CSV file and its path-derived metadata.

    Handles both quoted and unquoted CSV variants found in the dataset.
    """
    metadata = parse_path_metadata(filepath)

    try:
        df = pd.read_csv(filepath)
        # Verify expected columns
        if len(df.columns) == len(NUST_CSV_COLUMNS):
            df.columns = NUST_CSV_COLUMNS
    except Exception as e:
        return pd.DataFrame(), {"error": str(e), **metadata}

    metadata["source_file"] = os.path.basename(filepath)
    metadata["row_count"] = len(df)
    metadata["column_count"] = len(df.columns)

    return df, metadata
