"""
FEMTO/PRONOSTIA bearing degradation dataset loader.

Data structure:
  FEMTOBearingDataSet/
    Training_set/Learning_set/Bearing{X}_{Y}/acc_NNNNN.csv
    Test_set/Test_set/Bearing{X}_{Y}/acc_NNNNN.csv

Each CSV has columns: hour, minute, second, microsecond, horiz_accel, vert_accel
Sampling rate: 25.6 kHz (sampled every 10 seconds for 0.1s = 2560 samples per file)

Operating conditions by bearing prefix:
  Bearing1_*: 1800 rpm, 4000 N
  Bearing2_*: 1650 rpm, 4200 N
  Bearing3_*: 1500 rpm, 5000 N
"""

import os
import glob
import re
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple


FEMTO_OPERATING_CONDITIONS = {
    "1": {"rpm": 1800, "load_N": 4000},
    "2": {"rpm": 1650, "load_N": 4200},
    "3": {"rpm": 1500, "load_N": 5000},
}

FEMTO_CSV_COLUMNS = [
    "hour", "minute", "second", "microsecond", "horiz_accel", "vert_accel"
]


def get_femto_data_dir() -> str:
    """Return path to extracted FEMTO data."""
    return os.path.normpath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "raw", "femto",
            "10.+FEMTO+Bearing", "10. FEMTO Bearing", "FEMTOBearingDataSet",
        )
    )


def list_bearings(split: str = "train", data_dir: Optional[str] = None) -> List[str]:
    """List available bearing directories for the given split."""
    if data_dir is None:
        data_dir = get_femto_data_dir()

    if split == "train":
        search_dir = os.path.join(data_dir, "Training_set", "Learning_set")
    else:
        search_dir = os.path.join(data_dir, "Test_set", "Test_set")

    if not os.path.exists(search_dir):
        return []

    bearings = sorted([
        d for d in os.listdir(search_dir)
        if os.path.isdir(os.path.join(search_dir, d)) and d.startswith("Bearing")
    ])
    return bearings


def parse_bearing_id(bearing_name: str) -> Tuple[str, str]:
    """Parse 'Bearing1_2' into condition group '1' and run '2'."""
    match = re.match(r"Bearing(\d+)_(\d+)", bearing_name)
    if match:
        return match.group(1), match.group(2)
    return "unknown", "unknown"


def load_bearing_files(
    bearing_name: str,
    split: str = "train",
    data_dir: Optional[str] = None,
    max_files: Optional[int] = None,
) -> List[pd.DataFrame]:
    """
    Load all CSV files for a single bearing, ordered by file sequence number.

    Returns list of DataFrames (one per acquisition), sorted by file number.
    """
    if data_dir is None:
        data_dir = get_femto_data_dir()

    if split == "train":
        bearing_dir = os.path.join(data_dir, "Training_set", "Learning_set", bearing_name)
    else:
        bearing_dir = os.path.join(data_dir, "Test_set", "Test_set", bearing_name)

    if not os.path.exists(bearing_dir):
        return []

    csv_files = sorted(glob.glob(os.path.join(bearing_dir, "acc_*.csv")))
    if max_files:
        csv_files = csv_files[:max_files]

    frames = []
    for fp in csv_files:
        try:
            df = pd.read_csv(fp, header=None, names=FEMTO_CSV_COLUMNS)
            df["source_file"] = os.path.basename(fp)
            # Extract sequence number from filename
            seq_match = re.search(r"acc_(\d+)", os.path.basename(fp))
            df["file_sequence"] = int(seq_match.group(1)) if seq_match else 0
            frames.append(df)
        except Exception:
            continue

    return frames


def get_bearing_file_count(
    bearing_name: str, split: str = "train", data_dir: Optional[str] = None
) -> int:
    """Count CSV files for a bearing without loading them."""
    if data_dir is None:
        data_dir = get_femto_data_dir()
    if split == "train":
        bearing_dir = os.path.join(data_dir, "Training_set", "Learning_set", bearing_name)
    else:
        bearing_dir = os.path.join(data_dir, "Test_set", "Test_set", bearing_name)
    if not os.path.exists(bearing_dir):
        return 0
    return len(glob.glob(os.path.join(bearing_dir, "acc_*.csv")))
