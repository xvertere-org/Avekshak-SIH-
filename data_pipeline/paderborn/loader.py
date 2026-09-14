"""
Paderborn University Bearing Dataset loader.

Loads vibration signal arrays, operating condition metadata,
and catalogs documentation logs (PDFs) from extracted files.

Bearing conditions:
  K0xx  = Healthy baseline bearings (e.g. K001)
  KAxx  = Outer race damage bearings (e.g. KA01)
  KBxx  = Combined damage bearings
  KIxx  = Inner race damage bearings (e.g. KI04)

Vibration signal is sampled at native 64 kHz (piezoelectric accelerometer).
RESTRICTION: Vibration features must NOT be mapped to engine telemetry channels.
"""

import os
import glob
import re
from typing import Optional, List, Dict, Tuple, Any
import numpy as np


BEARING_FAULT_MAP = {
    "K0": "healthy",
    "KA": "outer_race_fault",
    "KB": "combined_fault",
    "KI": "inner_race_fault",
}

OPERATING_CONDITIONS = {
    "N09_M07_F10": {"speed_rpm": 900, "torque_nm": 0.7, "radial_force_n": 1000},
    "N15_M01_F10": {"speed_rpm": 1500, "torque_nm": 0.1, "radial_force_n": 1000},
    "N15_M07_F04": {"speed_rpm": 1500, "torque_nm": 0.7, "radial_force_n": 400},
    "N15_M07_F10": {"speed_rpm": 1500, "torque_nm": 0.7, "radial_force_n": 1000},
}


def get_paderborn_extracted_dir() -> str:
    """
    Return path to extracted Paderborn data.
    Checks data/processed/paderborn/extracted first, then _extracted.
    """
    p1 = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "paderborn", "extracted")
    )
    if os.path.exists(p1):
        return p1
    p2 = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "paderborn", "_extracted")
    )
    if os.path.exists(p2):
        return p2
    return p1


def discover_paderborn_files(extract_dir: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """
    Recursively discover all valid .mat files and catalog PDF log files.
    Returns (mat_files, pdf_files).
    """
    if extract_dir is None:
        extract_dir = get_paderborn_extracted_dir()

    mat_files = sorted(glob.glob(os.path.join(extract_dir, "**", "*.mat"), recursive=True))
    pdf_files = sorted(glob.glob(os.path.join(extract_dir, "**", "*.pdf"), recursive=True))
    return mat_files, pdf_files


def parse_filename_metadata(filepath: str) -> Dict[str, Any]:
    """
    Parse filename into structured metadata:
    e.g. N09_M07_F10_K001_1.mat
    """
    fn = os.path.basename(filepath).replace(".mat", "")
    parts = fn.split("_")
    
    # Defaults
    op_cond = "UNKNOWN"
    bearing_id = "UNKNOWN"
    repeat_idx = 1

    if len(parts) >= 5:
        op_cond = f"{parts[0]}_{parts[1]}_{parts[2]}"
        bearing_id = parts[3]
        try:
            repeat_idx = int(parts[4])
        except ValueError:
            repeat_idx = 1
    elif len(parts) >= 4:
        op_cond = f"{parts[0]}_{parts[1]}_{parts[2]}"
        bearing_id = parts[3]

    prefix = bearing_id[:2]
    fault_label = BEARING_FAULT_MAP.get(prefix, "unknown")
    op_params = OPERATING_CONDITIONS.get(op_cond, {"speed_rpm": np.nan, "torque_nm": np.nan, "radial_force_n": np.nan})

    return {
        "source_file": os.path.basename(filepath),
        "measurement_id": fn,
        "bearing_id": bearing_id,
        "repeat_index": repeat_idx,
        "operating_condition": op_cond,
        "fault_label": fault_label,
        "speed_rpm": op_params["speed_rpm"],
        "torque_nm": op_params["torque_nm"],
        "radial_force_n": op_params["radial_force_n"],
        "sampling_frequency": 64000.0,
    }


def load_paderborn_mat(filepath: str) -> Dict:
    """Load MATLAB .mat file safely."""
    try:
        from scipy.io import loadmat
    except ImportError:
        raise ImportError("scipy is required: pip install scipy")

    return loadmat(filepath)


def extract_vibration_signal(mat_data: Dict) -> Tuple[Optional[np.ndarray], float]:
    """
    Extract 64 kHz vibration signal array and sampling rate from loaded mat dictionary.
    Returns (signal_1d_array, sampling_rate_hz).
    """
    # Look for the main struct key (non-dunder)
    main_key = None
    for k in mat_data:
        if not k.startswith("__"):
            main_key = k
            break

    if main_key is None:
        return None, 64000.0

    top_val = mat_data[main_key]
    if not hasattr(top_val, "dtype") or not top_val.dtype.names:
        return None, 64000.0

    struct = top_val[0, 0]
    if "Y" not in struct.dtype.names:
        return None, 64000.0

    y_arr = struct["Y"]
    sampling_rate = 64000.0

    # Search for vibration channel
    for i in range(y_arr.shape[1]):
        y_elem = y_arr[0, i]
        ch_name = ""
        if "Name" in y_elem.dtype.names and len(y_elem["Name"]) > 0:
            ch_name = str(y_elem["Name"][0]).strip().lower()

        if "vibration" in ch_name or ch_name == "vibration_1":
            if "Data" in y_elem.dtype.names:
                data = y_elem["Data"]
                flat_data = np.asarray(data, dtype=np.float64).flatten()
                return flat_data, sampling_rate

    # Fallback to the largest array in Y if vibration_1 wasn't found by name
    max_len = 0
    best_data = None
    for i in range(y_arr.shape[1]):
        y_elem = y_arr[0, i]
        if "Data" in y_elem.dtype.names:
            d = np.asarray(y_elem["Data"], dtype=np.float64).flatten()
            if len(d) > max_len:
                max_len = len(d)
                best_data = d

    return best_data, sampling_rate
