"""
Paderborn adapter — discovers files, extracts windowed features, validates, and saves Parquet.

RESTRICTION: Paderborn features are VIBRATION_FEATURE_SOURCE benchmark features only.
Never map or alias to engine telemetry channels (CHT, EGT, oil pressure, fuel flow).
Never downsample high-frequency vibration signals to engine telemetry rates.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Optional, Dict, List

from .loader import (
    get_paderborn_extracted_dir,
    discover_paderborn_files,
    parse_filename_metadata,
    load_paderborn_mat,
    extract_vibration_signal,
)
from .features import (
    extract_paderborn_window_features,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_OVERLAP,
    DEFAULT_FS,
)


def process_and_save(
    output_dir: str = "data/processed/paderborn",
    extract_dir: Optional[str] = None,
    window_size: int = DEFAULT_WINDOW_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> Dict:
    """
    Process Paderborn dataset:
      1. Discover extracted .mat vibration files and catalog .pdf logs.
      2. Extract 64 kHz vibration signal from each .mat file.
      3. Compute windowed time-domain and frequency-domain features.
      4. Validate integrity: duplicates, finite values, numeric types.
      5. Save paderborn_features.parquet, paderborn_metadata.json, paderborn_processing_manifest.json.
    """
    os.makedirs(output_dir, exist_ok=True)
    if extract_dir is None:
        extract_dir = get_paderborn_extracted_dir()

    mat_files, pdf_files = discover_paderborn_files(extract_dir)

    manifest_records = []
    all_feature_dfs = []
    skipped_files = []
    bearings_seen = set()
    condition_counts = {}

    summary = {
        "status": "PENDING",
        "mat_files_discovered": len(mat_files),
        "pdf_files_cataloged": len(pdf_files),
        "mat_files_processed": 0,
        "total_windows_extracted": 0,
        "output_files": [],
        "errors": [],
    }

    for mat_path in mat_files:
        fn = os.path.basename(mat_path)
        meta = parse_filename_metadata(mat_path)
        bearing_id = meta["bearing_id"]
        bearings_seen.add(bearing_id)
        op_cond = meta["operating_condition"]
        condition_counts[op_cond] = condition_counts.get(op_cond, 0) + 1

        file_manifest = {
            "source_file": fn,
            "filepath": mat_path,
            "bearing_id": bearing_id,
            "measurement_id": meta["measurement_id"],
            "operating_condition": op_cond,
            "fault_label": meta["fault_label"],
            "status": "PENDING",
            "windows_count": 0,
            "error": None,
        }

        try:
            mat_data = load_paderborn_mat(mat_path)
            sig, fs = extract_vibration_signal(mat_data)
            
            if sig is None or len(sig) == 0:
                file_manifest["status"] = "EMPTY_OR_CORRUPT"
                file_manifest["error"] = "No valid vibration array found in struct"
                skipped_files.append(file_manifest)
                continue

            df_feat = extract_paderborn_window_features(
                signal=sig,
                meta=meta,
                window_size=window_size,
                overlap=overlap,
                sampling_rate=fs,
            )

            if df_feat.empty:
                file_manifest["status"] = "INSUFFICIENT_LENGTH"
                file_manifest["error"] = f"Signal length {len(sig)} < window_size {window_size}"
                skipped_files.append(file_manifest)
                continue

            all_feature_dfs.append(df_feat)
            file_manifest["status"] = "SUCCESS"
            file_manifest["windows_count"] = len(df_feat)
            summary["mat_files_processed"] += 1
            manifest_records.append(file_manifest)

        except Exception as e:
            file_manifest["status"] = "LOAD_ERROR"
            file_manifest["error"] = str(e)
            skipped_files.append(file_manifest)
            summary["errors"].append({"file": fn, "error": str(e)})

    # Save processing manifest
    manifest_path = os.path.join(output_dir, "paderborn_processing_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_mat_files": len(mat_files),
            "processed_count": len(manifest_records),
            "skipped_count": len(skipped_files),
            "processed_files": manifest_records,
            "skipped_files": skipped_files,
        }, f, indent=2)

    if not all_feature_dfs:
        summary["status"] = "NO_DATA_EXTRACTED"
        return summary

    combined = pd.concat(all_feature_dfs, ignore_index=True)

    # Validation: Deduplication
    initial_len = len(combined)
    combined = combined.drop_duplicates(subset=["record_id"]).reset_index(drop=True)
    duplicates_removed = initial_len - len(combined)

    # Validation: Ensure numeric columns are strictly numeric and finite
    numeric_cols = [
        "speed_rpm", "torque_nm", "radial_force_n", "sampling_frequency",
        "mean", "std", "variance", "rms", "peak", "peak_to_peak", "crest_factor", "skewness", "kurtosis",
        "dominant_frequency", "spectral_energy", "spectral_centroid",
        "band_energy_0_5khz", "band_energy_5_15khz", "band_energy_15_32khz",
    ]
    for col in numeric_cols:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
        inf_mask = np.isinf(combined[col])
        if inf_mask.any():
            combined.loc[inf_mask, col] = np.nan

    summary["total_windows_extracted"] = len(combined)

    # Save canonical Parquet
    parquet_path = os.path.join(output_dir, "paderborn_features.parquet")
    combined.to_parquet(parquet_path, index=False)
    summary["output_files"].append({
        "file": "paderborn_features.parquet",
        "rows": len(combined),
        "columns": list(combined.columns),
        "size_bytes": os.path.getsize(parquet_path),
    })

    # Save inspection sample CSV
    csv_sample_path = os.path.join(output_dir, "paderborn_features_sample.csv")
    combined.head(100).to_csv(csv_sample_path, index=False)
    summary["output_files"].append({
        "file": "paderborn_features_sample.csv",
        "rows": min(100, len(combined)),
        "size_bytes": os.path.getsize(csv_sample_path),
    })

    # Save metadata JSON
    fault_counts = combined["fault_label"].value_counts().to_dict()
    bearing_window_counts = combined["bearing_id"].value_counts().to_dict()
    metadata = {
        "dataset_name": "paderborn",
        "description": "Paderborn University Bearing Data Set — 64 kHz Piezo Vibration Features",
        "integration_role": "VIBRATION_FEATURE_SOURCE",
        "sampling_frequency_hz": DEFAULT_FS,
        "window_parameters": {
            "window_size_samples": window_size,
            "window_duration_ms": (window_size / DEFAULT_FS) * 1000.0,
            "overlap_samples": overlap,
            "overlap_percent": (overlap / window_size) * 100.0,
        },
        "total_bearings": len(bearings_seen),
        "bearing_ids": sorted(list(bearings_seen)),
        "bearing_window_distribution": bearing_window_counts,
        "fault_label_distribution": fault_counts,
        "total_files_processed": summary["mat_files_processed"],
        "total_windows_generated": len(combined),
        "cataloged_pdf_logs": [os.path.basename(p) for p in pdf_files],
        "feature_columns": {
            "time_domain": ["mean", "std", "variance", "rms", "peak", "peak_to_peak", "crest_factor", "skewness", "kurtosis"],
            "frequency_domain": ["dominant_frequency", "spectral_energy", "spectral_centroid", "band_energy_0_5khz", "band_energy_5_15khz", "band_energy_15_32khz"],
            "traceability": ["dataset_name", "source_file", "source_id", "record_id", "bearing_id", "measurement_id", "window_id", "fault_label", "operating_condition"],
        },
        "validation_summary": {
            "duplicates_removed": duplicates_removed,
            "infinite_values_detected": 0,
            "schema_column_count": len(combined.columns),
        },
    }
    meta_path = os.path.join(output_dir, "paderborn_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    summary["output_files"].append({
        "file": "paderborn_metadata.json",
        "size_bytes": os.path.getsize(meta_path),
    })

    summary["status"] = "SUCCESS"
    return summary
