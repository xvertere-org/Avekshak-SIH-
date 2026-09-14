"""
CWRU adapter — audits, processes, and extracts features for the CWRU dataset.

Split Strategy & Data Leakage Prevention:
  - Splitting individual windows from the same file across train and test causes severe data leakage.
  - Furthermore, with the available 3 files (97.mat: Normal, 105.mat: Inner Race, 130.mat: Outer Race),
    each file represents a unique fault class. Splitting across files would leave entire fault classes
    absent from either train or test.
  - In accordance with Section 5 of the specification, the canonical output is a single, thoroughly
    validated feature table (cwru_features.parquet) partitioned logically by 'source_file' and 'bearing_id'.
    The limitation of this 3-file subset is explicitly documented in cwru_metadata.json and manifests.

Outputs generated:
  - data/processed/cwru/cwru_features.parquet
  - data/processed/cwru/cwru_metadata.json
  - data/processed/cwru/cwru_processing_manifest.json
  - data/processed/cwru/cwru_features_sample.csv
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
import pandas as pd

from .loader import audit_raw_cwru_files, load_all, get_cwru_data_dir
from .features import extract_features_for_file, DEFAULT_WINDOW_SIZE, DEFAULT_OVERLAP, DEFAULT_FS
from ..common.provenance import compute_file_checksum

logger = logging.getLogger(__name__)


def process_and_save(
    output_dir: str = "data/processed/cwru",
    data_dir: Optional[str] = None,
    window_size: int = DEFAULT_WINDOW_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> Dict[str, Any]:
    """
    Process all available CWRU files, extract features, and save canonical outputs.

    Returns structured summary dictionary.
    """
    if data_dir is None:
        data_dir = get_cwru_data_dir()

    os.makedirs(output_dir, exist_ok=True)

    # 1. Audit raw files
    audits = audit_raw_cwru_files(data_dir)
    total_raw_discovered = len(audits)
    supported_files = [a for a in audits if a["extension"] in [".mat", ".csv", ".txt"]]
    files_to_process = [a for a in audits if a["parsing_status"] == "SUCCESS"]
    files_skipped = [a for a in audits if a["parsing_status"] != "SUCCESS"]

    loaded_data = load_all(data_dir)

    all_features = []
    processed_files_manifest = []
    skipped_files_manifest = []

    for aud in files_skipped:
        skipped_files_manifest.append({
            "source_path": aud["source_path"],
            "filename": aud["filename"],
            "reason": aud.get("skip_reason", "Parsing skipped"),
        })

    for filename, file_data in loaded_data.items():
        channels = file_data["channels"]
        metadata = file_data["metadata"]
        filepath = file_data["filepath"]
        locations = file_data.get("sensor_locations", {})

        if not channels:
            skipped_files_manifest.append({
                "source_path": filepath,
                "filename": filename,
                "reason": "No valid vibration channels found",
            })
            continue

        features = extract_features_for_file(
            channels=channels,
            file_metadata=metadata,
            window_size=window_size,
            overlap=overlap,
            source_file=filename,
            sensor_locations=locations,
        )

        if features.empty:
            skipped_files_manifest.append({
                "source_path": filepath,
                "filename": filename,
                "reason": f"Signal length shorter than window_size ({window_size})",
            })
            continue

        # Add provenance checksum and versions
        try:
            checksum = compute_file_checksum(filepath)
        except Exception:
            checksum = None

        features["source_checksum"] = checksum
        features["preprocessing_version"] = "0.2.0"
        features["feature_version"] = "0.2.0"

        all_features.append(features)

        processed_files_manifest.append({
            "source_path": filepath,
            "filename": filename,
            "file_size_bytes": os.path.getsize(filepath) if os.path.exists(filepath) else 0,
            "checksum_sha256": checksum,
            "channels_extracted": list(channels.keys()),
            "sensor_locations": list(locations.values()),
            "sampling_frequency_hz": metadata.get("sampling_rate_hz", DEFAULT_FS),
            "rpm": metadata.get("rpm"),
            "fault_type": metadata.get("fault_type"),
            "fault_size_mils": metadata.get("fault_size_mils"),
            "motor_load_hp": metadata.get("motor_load_hp"),
            "windows_generated": len(features),
        })

    if not all_features:
        logger.warning("No features extracted from CWRU files.")
        return {
            "status": "FAILED",
            "reason": "Zero features extracted",
            "total_raw_files_discovered": total_raw_discovered,
            "files_processed": 0,
            "files_skipped": skipped_files_manifest,
        }

    combined = pd.concat(all_features, ignore_index=True)

    # Save canonical parquet
    parquet_path = os.path.join(output_dir, "cwru_features.parquet")
    combined.to_parquet(parquet_path, index=False)

    # Save small inspection sample CSV (50 rows)
    sample_csv_path = os.path.join(output_dir, "cwru_features_sample.csv")
    sample_df = combined.head(50)
    sample_df.to_csv(sample_csv_path, index=False)

    # Build metadata document
    metadata_doc = {
        "dataset_name": "cwru",
        "description": "Case Western Reserve University Bearing Data Center — Vibration Feature Dataset",
        "provenance": {
            "source_repository": "Case Western Reserve University Bearing Data Center",
            "sampling_frequency_hz": DEFAULT_FS,
            "testbed": "Reliance Electric 2-hp induction motor",
            "bearing_specifications": {
                "drive_end": "SKF 6205-2RS deep groove ball bearing",
                "fan_end": "SKF 6203-2RS deep groove ball bearing",
            },
        },
        "window_parameters": {
            "window_size_samples": int(window_size),
            "window_duration_ms": round(float(window_size) / DEFAULT_FS * 1000.0, 2),
            "overlap_samples": int(overlap),
            "overlap_percent": round(float(overlap) / float(window_size) * 100.0, 1),
        },
        "feature_definitions": {
            "time_domain_14_metrics": [
                "mean", "std", "variance", "RMS", "minimum", "maximum",
                "peak_to_peak", "absolute_mean", "skewness", "kurtosis",
                "crest_factor", "shape_factor", "impulse_factor", "clearance_factor"
            ],
            "compatibility_aliases": {
                "rms": "Alias for RMS",
                "peak": "Maximum absolute amplitude np.max(np.abs(window))",
                "energy": "Sum of squared window values np.sum(window**2)"
            },
            "frequency_domain_metrics": [
                "dominant_frequency", "spectral_energy", "spectral_centroid", "frequency_band_energy"
            ],
            "frequency_subbands": {
                "band_energy_0_1500hz": "[0, 1500 Hz)",
                "band_energy_1500_3000hz": "[1500, 3000 Hz)",
                "band_energy_3000_4500hz": "[3000, 4500 Hz)",
                "band_energy_4500_6000hz": "[4500, 6000 Hz] (Nyquist limit)"
            },
        },
        "statistics": {
            "total_windows": int(len(combined)),
            "total_features": len(combined.columns),
            "fault_distribution": combined["fault_label"].value_counts().to_dict(),
            "sensor_location_distribution": combined["sensor_location"].value_counts().to_dict(),
            "source_file_distribution": combined["source_file"].value_counts().to_dict(),
            "operating_conditions": {
                "motor_load_hp": [int(x) for x in combined["motor_load_hp"].unique() if pd.notna(x)],
                "speeds_rpm": [float(x) for x in combined["RPM"].unique() if pd.notna(x)],
            },
        },
        "splitting_strategy": {
            "recommendation": "Partition by 'source_file' or 'bearing_id'. Do not randomly split windows.",
            "subset_limitations": (
                "The available raw CWRU subset contains 3 files representing distinct fault conditions "
                "(97.mat=normal, 105.mat=inner race 7mil, 130.mat=outer race 7mil). Because each fault "
                "class exists in only one file, train/test splitting across files would cause zero-shot "
                "evaluation or class omission. Therefore, a single unified, validated feature table "
                "is published for baseline methodology verification."
            ),
        },
        "generated_at": datetime.now().isoformat(),
        "preprocessing_version": "0.2.0",
    }

    metadata_path = os.path.join(output_dir, "cwru_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata_doc, f, indent=2)

    # Build processing manifest
    manifest_doc = {
        "dataset_name": "cwru",
        "generated_at": datetime.now().isoformat(),
        "total_raw_files_discovered": total_raw_discovered,
        "supported_files_discovered": len(supported_files),
        "files_successfully_parsed": len(processed_files_manifest),
        "files_skipped_count": len(skipped_files_manifest),
        "processed_files": processed_files_manifest,
        "skipped_files": skipped_files_manifest,
        "outputs": {
            "parquet_file": os.path.basename(parquet_path),
            "parquet_path": parquet_path,
            "parquet_rows": int(len(combined)),
            "parquet_columns": list(combined.columns),
            "sample_csv": os.path.basename(sample_csv_path),
            "metadata_json": os.path.basename(metadata_path),
        },
    }

    manifest_path = os.path.join(output_dir, "cwru_processing_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_doc, f, indent=2)

    summary = {
        "dataset_name": "cwru",
        "total_raw_files_discovered": total_raw_discovered,
        "supported_files_discovered": len(supported_files),
        "files_processed": len(processed_files_manifest),
        "files_skipped": skipped_files_manifest,
        "total_rows": int(len(combined)),
        "columns": list(combined.columns),
        "output_files": [
            {"file": os.path.basename(parquet_path), "rows": int(len(combined))},
            {"file": os.path.basename(sample_csv_path), "rows": int(len(sample_df))},
            {"file": os.path.basename(metadata_path), "type": "json_metadata"},
            {"file": os.path.basename(manifest_path), "type": "json_manifest"},
        ],
        "fault_distribution": combined["fault_label"].value_counts().to_dict(),
        "sensor_distribution": combined["sensor_location"].value_counts().to_dict(),
        "sampling_frequencies": [float(x) for x in combined["sampling_frequency"].unique()],
        "operating_conditions": {
            "loads": list(combined["load_condition"].unique()),
            "rpms": [float(x) for x in combined["RPM"].unique() if pd.notna(x)],
        },
        "output_parquet_path": parquet_path,
        "limitations": metadata_doc["splitting_strategy"]["subset_limitations"],
    }

    return summary
