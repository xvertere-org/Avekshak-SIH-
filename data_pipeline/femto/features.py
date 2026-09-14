"""
FEMTO vibration feature extraction and RUL label generation.

Features are extracted per-file (each file = one 0.1s acquisition at 25.6 kHz).
RUL is generated from run-to-failure trajectories: RUL = total_files - current_file_index.
"""

import numpy as np
import pandas as pd
from typing import List, Optional


def extract_acquisition_features(df: pd.DataFrame) -> dict:
    """
    Extract statistical features from a single FEMTO acquisition file.

    Each file contains ~2560 samples of horizontal and vertical acceleration.
    """
    features = {}
    for channel in ["horiz_accel", "vert_accel"]:
        if channel not in df.columns:
            continue
        signal = df[channel].values.astype(float)

        rms = np.sqrt(np.mean(signal ** 2))
        std = np.std(signal)
        peak = np.max(np.abs(signal))
        peak_to_peak = np.max(signal) - np.min(signal)
        crest_factor = peak / rms if rms > 0 else 0.0
        mean_val = np.mean(signal)
        n = len(signal)

        if std > 0:
            kurtosis = np.mean(((signal - mean_val) / std) ** 4) - 3.0
            skewness = np.mean(((signal - mean_val) / std) ** 3)
        else:
            kurtosis = 0.0
            skewness = 0.0

        energy = np.sum(signal ** 2)

        prefix = "h" if channel == "horiz_accel" else "v"
        features[f"{prefix}_rms"] = rms
        features[f"{prefix}_std"] = std
        features[f"{prefix}_peak"] = peak
        features[f"{prefix}_peak_to_peak"] = peak_to_peak
        features[f"{prefix}_crest_factor"] = crest_factor
        features[f"{prefix}_kurtosis"] = kurtosis
        features[f"{prefix}_skewness"] = skewness
        features[f"{prefix}_energy"] = energy

    return features


def build_degradation_trajectory(
    file_frames: List[pd.DataFrame],
    bearing_name: str,
    total_files: int,
    is_run_to_failure: bool = True,
) -> pd.DataFrame:
    """
    Build a degradation feature trajectory from a sequence of acquisition files.

    For training bearings (run-to-failure), RUL = total_files - current_index - 1.
    For test bearings, RUL is not generated (set to null).
    """
    records = []
    for idx, frame in enumerate(file_frames):
        features = extract_acquisition_features(frame)
        features["bearing_id"] = bearing_name
        features["file_index"] = idx
        features["total_files"] = total_files

        if is_run_to_failure:
            features["rul_label"] = total_files - idx - 1
        else:
            features["rul_label"] = None

        if "source_file" in frame.columns:
            features["source_file"] = frame["source_file"].iloc[0]
        if "file_sequence" in frame.columns:
            features["file_sequence"] = frame["file_sequence"].iloc[0]

        records.append(features)

    return pd.DataFrame(records)
