"""
CWRU Bearing Data Center loader and auditor.

Loads and audits MATLAB .mat and other raw files from the CWRU bearing dataset.
Provides recursive file discovery, MATLAB key inspection, accelerometer channel
detection (DE, FE, BA), RPM extraction, sensor location mapping, and fail-safe error handling.

File naming convention:
  - File number encodes the experiment / fault condition
  - DE_time = Drive End accelerometer (12kHz or 48kHz)
  - FE_time = Fan End accelerometer (12kHz)
  - BA_time = Base accelerometer (12kHz)
  - RPM = Shaft rotational speed
"""

import os
import glob
import logging
from typing import Dict, Optional, List, Any
import numpy as np

logger = logging.getLogger(__name__)

# Known CWRU metadata registry for drive-end 12k experiments and normal baselines
CWRU_FILE_METADATA = {
    "97.mat": {
        "fault_type": "normal",
        "fault_size_mils": 0,
        "motor_load_hp": 0,
        "rpm": 1796,
        "sampling_rate_hz": 12000.0,
        "fault_location": "none",
        "description": "Normal baseline data (12k fan/drive end)",
    },
    "105.mat": {
        "fault_type": "inner_race",
        "fault_size_mils": 7,
        "motor_load_hp": 0,
        "rpm": 1797,
        "sampling_rate_hz": 12000.0,
        "fault_location": "drive_end",
        "description": "Drive end 12k bearing fault: Inner race 0.007 inches",
    },
    "130.mat": {
        "fault_type": "outer_race",
        "fault_size_mils": 7,
        "motor_load_hp": 0,
        "rpm": 1796,
        "sampling_rate_hz": 12000.0,
        "fault_location": "drive_end_centered_6_oclock",
        "description": "Drive end 12k bearing fault: Outer race centered 0.007 inches",
    },
}

SUPPORTED_EXTENSIONS = {".mat", ".csv", ".txt"}


def get_cwru_data_dir() -> str:
    """Return absolute path to CWRU raw data directory."""
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "cwru")
    )


def list_available_files(data_dir: Optional[str] = None) -> List[str]:
    """Recursively list all supported raw files in the CWRU directory."""
    if data_dir is None:
        data_dir = get_cwru_data_dir()
    if not os.path.exists(data_dir):
        return []

    discovered = []
    for root, _, files in os.walk(data_dir):
        for f in sorted(files):
            ext = os.path.splitext(f)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                discovered.append(os.path.normpath(os.path.join(root, f)))
    return sorted(discovered)


def parse_matlab_keys(mat_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inspect MATLAB keys and categorize into vibration signals, RPM, and metadata.

    Returns dict with:
      - 'signals': dict of {channel_name: {array, sensor_location, length}}
      - 'rpm': float or None
      - 'matlab_keys': list of all non-dunder keys
      - 'skipped_keys': list of skipped keys with reason
    """
    signals = {}
    skipped_keys = []
    rpm_val = None
    all_keys = [k for k in mat_dict.keys() if not k.startswith("__")]

    for key in all_keys:
        val = mat_dict[key]

        # Check for RPM key
        if "rpm" in key.lower():
            if isinstance(val, np.ndarray) and val.size > 0:
                try:
                    rpm_val = float(val.flat[0])
                except Exception as e:
                    skipped_keys.append({"key": key, "reason": f"Failed to parse RPM: {e}"})
            continue

        # Check for numeric vibration arrays
        if not isinstance(val, np.ndarray):
            skipped_keys.append({"key": key, "reason": "Non-ndarray type"})
            continue

        if not np.issubdtype(val.dtype, np.number):
            skipped_keys.append({"key": key, "reason": f"Non-numeric dtype: {val.dtype}"})
            continue

        arr = val.flatten()
        if len(arr) == 0:
            skipped_keys.append({"key": key, "reason": "Empty array"})
            continue

        if not np.all(np.isfinite(arr)):
            # Check if there are non-finite values
            finite_mask = np.isfinite(arr)
            if np.sum(finite_mask) == 0:
                skipped_keys.append({"key": key, "reason": "All values are non-finite (NaN/Inf)"})
                continue
            # Clean array
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        # Detect channel location
        location = "unknown"
        k_upper = key.upper()
        if "_DE_" in k_upper or k_upper.endswith("DE_TIME") or "DE_TIME" in k_upper:
            location = "drive_end"
        elif "_FE_" in k_upper or k_upper.endswith("FE_TIME") or "FE_TIME" in k_upper:
            location = "fan_end"
        elif "_BA_" in k_upper or k_upper.endswith("BA_TIME") or "BA_TIME" in k_upper:
            location = "base"

        signals[key] = {
            "signal": arr,
            "sensor_location": location,
            "length": len(arr),
            "dtype": str(val.dtype),
        }

    return {
        "signals": signals,
        "rpm": rpm_val,
        "matlab_keys": all_keys,
        "skipped_keys": skipped_keys,
    }


def audit_raw_cwru_files(data_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Recursively scan data/raw/cwru/ and record complete audit records for all files.

    Records:
      - source_path, filename, file_size_bytes
      - MATLAB keys, if applicable
      - signal channels found
      - sampling frequency
      - RPM
      - load condition
      - fault type and fault size
      - sensor location
      - file parsing status (SUCCESS, SKIPPED, FAILED)
      - reason for skipping unsupported or corrupt files
    """
    if data_dir is None:
        data_dir = get_cwru_data_dir()

    audit_records = []

    if not os.path.exists(data_dir):
        logger.warning("CWRU directory does not exist: %s", data_dir)
        return audit_records

    # Recursive scan of all files
    all_filepaths = []
    for root, _, files in os.walk(data_dir):
        for f in sorted(files):
            all_filepaths.append(os.path.normpath(os.path.join(root, f)))

    for fp in all_filepaths:
        fn = os.path.basename(fp)
        ext = os.path.splitext(fn)[1].lower()
        size_bytes = os.path.getsize(fp) if os.path.exists(fp) else 0

        record = {
            "source_path": fp,
            "filename": fn,
            "extension": ext,
            "file_size_bytes": size_bytes,
            "matlab_keys": [],
            "signal_channels": [],
            "sensor_locations": [],
            "sampling_frequency": None,
            "rpm": None,
            "load_condition": None,
            "fault_type": None,
            "fault_size": None,
            "parsing_status": "PENDING",
            "skip_reason": None,
        }

        # Check if extension is supported
        if ext not in SUPPORTED_EXTENSIONS:
            record["parsing_status"] = "SKIPPED"
            record["skip_reason"] = f"Unsupported file extension: {ext}"
            audit_records.append(record)
            continue

        if size_bytes == 0:
            record["parsing_status"] = "SKIPPED"
            record["skip_reason"] = "Empty file (0 bytes)"
            audit_records.append(record)
            continue

        if ext == ".mat":
            try:
                from scipy.io import loadmat
                mat_data = loadmat(fp)
                parsed = parse_matlab_keys(mat_data)

                record["matlab_keys"] = parsed["matlab_keys"]
                signals = parsed["signals"]

                if not signals:
                    record["parsing_status"] = "SKIPPED"
                    record["skip_reason"] = "No valid numeric vibration channels found in MAT file"
                    audit_records.append(record)
                    continue

                record["signal_channels"] = list(signals.keys())
                record["sensor_locations"] = [signals[k]["sensor_location"] for k in signals]

                # Match known metadata or default
                meta = CWRU_FILE_METADATA.get(fn, {})
                rpm = parsed["rpm"] if parsed["rpm"] is not None else meta.get("rpm", None)
                record["rpm"] = rpm
                record["sampling_frequency"] = meta.get("sampling_rate_hz", 12000.0)
                record["load_condition"] = f"{meta.get('motor_load_hp', 0)}HP" if "motor_load_hp" in meta else "UNKNOWN"
                record["fault_type"] = meta.get("fault_type", "unknown")
                record["fault_size"] = meta.get("fault_size_mils", None)
                record["parsing_status"] = "SUCCESS"

            except Exception as e:
                record["parsing_status"] = "FAILED"
                record["skip_reason"] = f"MATLAB parsing error: {type(e).__name__}: {str(e)}"

        elif ext in [".csv", ".txt"]:
            # Check if any CSV/TXT files exist and handle gracefully
            record["parsing_status"] = "SKIPPED"
            record["skip_reason"] = f"Non-MATLAB raw file ({ext}); CWRU vibration dataset uses .mat format"

        audit_records.append(record)

    return audit_records


def load_mat_file(filepath: str) -> Dict[str, np.ndarray]:
    """
    Load a CWRU .mat file and extract vibration channels.

    Returns dict of channel_name -> 1D numpy array.
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        raise ImportError("scipy is required to load CWRU .mat files: pip install scipy")

    try:
        mat_data = loadmat(filepath)
    except Exception as e:
        logger.error("Failed to load MATLAB file %s: %s", filepath, e)
        return {}

    parsed = parse_matlab_keys(mat_data)
    return {k: v["signal"] for k, v in parsed["signals"].items()}


def load_all(data_dir: Optional[str] = None) -> Dict[str, dict]:
    """
    Load all available CWRU files with metadata and channels.

    Returns dict keyed by filename with:
      - 'filepath': str
      - 'channels': Dict[str, np.ndarray]
      - 'metadata': dict
      - 'sensor_locations': Dict[str, str]
    """
    audits = audit_raw_cwru_files(data_dir)
    results = {}

    for aud in audits:
        if aud["parsing_status"] != "SUCCESS":
            continue

        fn = aud["filename"]
        fp = aud["source_path"]

        try:
            from scipy.io import loadmat
            mat_data = loadmat(fp)
            parsed = parse_matlab_keys(mat_data)
            channels = {k: v["signal"] for k, v in parsed["signals"].items()}
            locations = {k: v["sensor_location"] for k, v in parsed["signals"].items()}
        except Exception as e:
            logger.error("Error loading channels from %s: %s", fp, e)
            continue

        meta = CWRU_FILE_METADATA.get(fn, {}).copy()
        # Merge parsed RPM
        if parsed.get("rpm") is not None:
            meta["rpm"] = parsed["rpm"]
        if "sampling_rate_hz" not in meta:
            meta["sampling_rate_hz"] = 12000.0

        results[fn] = {
            "filepath": fp,
            "channels": channels,
            "sensor_locations": locations,
            "metadata": meta,
            "audit": aud,
        }

    return results
