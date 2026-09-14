"""
C-MAPSS adapter — produces processed outputs in benchmark-only namespace.

SAFETY: This adapter never outputs columns named cht, egt, oil_temp,
oil_pressure, or fuel_flow. All sensor columns use cmapss_s_N naming.
"""

import os
import pandas as pd
from typing import Optional
from .preprocess import preprocess_subset
from .loader import list_available_subsets
from ..common.schemas import UNSAFE_CMAPSS_COLUMN_NAMES, ProcessedRecordMetadata
from ..common.provenance import compute_file_checksum


def _validate_no_unsafe_columns(df: pd.DataFrame) -> None:
    """Raise if any unsafe aero-piston column names appear."""
    unsafe = set(df.columns) & UNSAFE_CMAPSS_COLUMN_NAMES
    if unsafe:
        raise ValueError(
            f"UNSAFE C-MAPSS columns detected: {unsafe}. "
            "C-MAPSS sensors must use cmapss_s_N naming."
        )


def process_and_save(
    output_dir: str = "data/processed/cmapss",
    data_dir: Optional[str] = None,
    max_rul: int = 125,
) -> dict:
    """
    Process all available C-MAPSS subsets and save to output directory.

    Returns summary dict with file counts and row counts.
    """
    os.makedirs(output_dir, exist_ok=True)
    available = list_available_subsets(data_dir)
    summary = {"subsets_processed": [], "total_train_rows": 0, "total_test_rows": 0, "files": []}

    for subset in available:
        result = preprocess_subset(subset, data_dir, max_rul=max_rul)

        for split_name in ["train", "test"]:
            df = result[split_name]

            # SAFETY CHECK: ensure no unsafe column names
            _validate_no_unsafe_columns(df)

            # Add metadata columns
            df["dataset_name"] = "cmapss"
            df["preprocessing_version"] = "0.1.0"

            filename = f"cmapss_{subset}_{split_name}.parquet"
            filepath = os.path.join(output_dir, filename)

            try:
                df.to_parquet(filepath, index=False)
            except Exception:
                # Fallback to CSV if parquet not available
                filename = f"cmapss_{subset}_{split_name}.csv"
                filepath = os.path.join(output_dir, filename)
                df.to_csv(filepath, index=False)

            summary["files"].append({
                "file": filename,
                "rows": len(df),
                "columns": list(df.columns),
                "subset": subset,
                "split": split_name,
            })

            if split_name == "train":
                summary["total_train_rows"] += len(df)
            else:
                summary["total_test_rows"] += len(df)

        summary["subsets_processed"].append(subset)

    return summary
