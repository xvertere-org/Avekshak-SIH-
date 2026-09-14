"""
FEMTO adapter — processes all bearings and saves degradation trajectories.

Split: by bearing ID. Train bearings stay in train, test bearings in test.
No random splitting of windows from the same bearing across partitions.
"""

import os
import pandas as pd
from typing import Optional
from .loader import list_bearings, load_bearing_files, get_bearing_file_count, parse_bearing_id, FEMTO_OPERATING_CONDITIONS
from .features import build_degradation_trajectory


def process_and_save(
    output_dir: str = "data/processed/femto",
    data_dir: Optional[str] = None,
    max_files_per_bearing: Optional[int] = None,
) -> dict:
    """
    Process all FEMTO bearings and save degradation trajectories.

    Uses a sampling strategy for large bearings if max_files_per_bearing is set.
    """
    os.makedirs(output_dir, exist_ok=True)
    summary = {
        "train_bearings": [],
        "test_bearings": [],
        "total_train_rows": 0,
        "total_test_rows": 0,
        "files_loaded": 0,
        "output_files": [],
        "skipped": [],
    }

    for split in ["train", "test"]:
        bearings = list_bearings(split, data_dir)
        all_features = []

        for bearing_name in bearings:
            file_count = get_bearing_file_count(bearing_name, split, data_dir)
            if file_count == 0:
                summary["skipped"].append({"bearing": bearing_name, "reason": "no files"})
                continue

            # Load files (with optional limit for performance)
            frames = load_bearing_files(bearing_name, split, data_dir, max_files_per_bearing)
            if not frames:
                summary["skipped"].append({"bearing": bearing_name, "reason": "load failed"})
                continue

            # Build trajectory
            is_rtf = (split == "train")  # Training = run-to-failure
            trajectory = build_degradation_trajectory(
                frames, bearing_name, total_files=file_count, is_run_to_failure=is_rtf,
            )

            # Add metadata
            cond_group, run_id = parse_bearing_id(bearing_name)
            cond = FEMTO_OPERATING_CONDITIONS.get(cond_group, {})
            trajectory["dataset_name"] = "femto"
            trajectory["split"] = split
            trajectory["source_bearing_id"] = bearing_name
            trajectory["source_run_id"] = run_id
            trajectory["operating_condition"] = f"{cond.get('rpm', '?')}rpm_{cond.get('load_N', '?')}N"
            trajectory["original_sampling_rate"] = 25600.0
            trajectory["preprocessing_version"] = "0.1.0"

            all_features.append(trajectory)
            summary["files_loaded"] += len(frames)

            if split == "train":
                summary["train_bearings"].append({
                    "bearing": bearing_name, "files": file_count,
                    "features_rows": len(trajectory),
                })
            else:
                summary["test_bearings"].append({
                    "bearing": bearing_name, "files": file_count,
                    "features_rows": len(trajectory),
                })

        if all_features:
            combined = pd.concat(all_features, ignore_index=True)
            filename = f"femto_{split}_features.parquet"
            filepath = os.path.join(output_dir, filename)
            try:
                combined.to_parquet(filepath, index=False)
            except Exception:
                filename = f"femto_{split}_features.csv"
                filepath = os.path.join(output_dir, filename)
                combined.to_csv(filepath, index=False)

            row_count = len(combined)
            summary["output_files"].append({"file": filename, "rows": row_count})
            if split == "train":
                summary["total_train_rows"] = row_count
            else:
                summary["total_test_rows"] = row_count

    return summary
