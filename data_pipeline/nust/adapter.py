"""
NUST adapter — processes all CSV files and saves with original column names.

RESTRICTION: Only creates explicit variable mappings that are physically defensible.
Does NOT rename channels to CHT, EGT, oil_temp, oil_pressure, or fuel_flow.
"""

import os
import pandas as pd
from typing import Optional
from .loader import list_csv_files, load_csv_file
from .preprocess import check_data_quality, identify_operating_region
from ..common.provenance import compute_file_checksum


def process_and_save(
    output_dir: str = "data/processed/nust",
    data_dir: Optional[str] = None,
) -> dict:
    """Process all NUST CSV files and save combined output."""
    os.makedirs(output_dir, exist_ok=True)
    csv_files = list_csv_files(data_dir)

    summary = {
        "files_processed": 0,
        "files_skipped": [],
        "total_rows": 0,
        "output_files": [],
        "quality_issues": [],
        "operating_regions": {"steady_state": 0, "transient": 0},
        "conditions": {"healthy": 0, "faulty": 0},
    }

    all_records = []

    for filepath in csv_files:
        df, metadata = load_csv_file(filepath)

        if df.empty:
            summary["files_skipped"].append({
                "file": os.path.basename(filepath),
                "reason": metadata.get("error", "empty file"),
            })
            continue

        # Quality check
        quality = check_data_quality(df)
        if quality["duplicate_rows"] > 0 or quality["missing_values"]:
            summary["quality_issues"].append({
                "file": os.path.basename(filepath),
                "duplicates": quality["duplicate_rows"],
                "missing": quality["missing_values"],
            })

        # Operating region
        region = identify_operating_region(df)
        summary["operating_regions"][region] = summary["operating_regions"].get(region, 0) + 1

        # Condition tracking
        cond = metadata.get("bearing_condition", "unknown")
        if cond in summary["conditions"]:
            summary["conditions"][cond] += 1

        # Add metadata columns — using ORIGINAL column names
        df["dataset_name"] = "nust"
        df["source_file"] = os.path.basename(filepath)
        df["bearing_condition"] = metadata.get("bearing_condition")
        df["rpm"] = metadata.get("rpm")
        df["humidity_pct"] = metadata.get("humidity")
        df["temperature_celsius"] = metadata.get("temperature_celsius")
        df["dataset_variant"] = metadata.get("dataset_variant")
        df["operating_region"] = region
        df["preprocessing_version"] = "0.1.0"

        try:
            df["source_checksum"] = compute_file_checksum(filepath)
        except Exception:
            df["source_checksum"] = None

        all_records.append(df)
        summary["files_processed"] += 1

    if all_records:
        combined = pd.concat(all_records, ignore_index=True)
        summary["total_rows"] = len(combined)

        filename = "nust_processed.parquet"
        filepath_out = os.path.join(output_dir, filename)
        try:
            combined.to_parquet(filepath_out, index=False)
        except Exception:
            filename = "nust_processed.csv"
            filepath_out = os.path.join(output_dir, filename)
            combined.to_csv(filepath_out, index=False)

        summary["output_files"].append({"file": filename, "rows": len(combined)})
        summary["columns"] = list(combined.columns)

    return summary
