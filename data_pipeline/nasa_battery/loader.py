"""
NASA Battery dataset loader.

The dataset consists of ZIP archives containing MATLAB .mat files
with charge/discharge/impedance cycle data for NASA Ames PCoE batteries.

RESTRICTION: Battery variables must NOT be mapped to engine telemetry channels.
"""

import os
import zipfile
import glob
import json
from typing import Optional, List, Dict, Tuple


def get_nasa_battery_raw_dir() -> str:
    """Return path to NASA Battery raw directory."""
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "nasa_battery")
    )


def get_battery_extracted_dir() -> str:
    """
    Return path containing the archived ZIP files.
    Checks data/processed/nasa_battery/extracted first, then _extracted.
    """
    p1 = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "nasa_battery", "extracted")
    )
    if os.path.exists(p1):
        return p1
    p2 = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "nasa_battery", "_extracted", "5. Battery Data Set")
    )
    if os.path.exists(p2):
        return p2
    return p1


def get_safe_staging_dir() -> str:
    """Return directory for safely extracted .mat files."""
    return os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "nasa_battery", "extracted_mat")
    )


def extract_battery_archives(
    source_dir: Optional[str] = None,
    staging_dir: Optional[str] = None,
) -> Tuple[List[Dict], Dict]:
    """
    Safely extract ZIP files from source_dir into staging_dir without overwriting unrelated files.
    Returns (manifest_entries, overall_summary).
    """
    if source_dir is None:
        source_dir = get_battery_extracted_dir()
    if staging_dir is None:
        staging_dir = get_safe_staging_dir()

    os.makedirs(staging_dir, exist_ok=True)

    zip_files = sorted(glob.glob(os.path.join(source_dir, "*.zip")))
    manifest_entries = []
    total_files_extracted = 0

    for zpath in zip_files:
        archive_name = os.path.basename(zpath)
        entry = {
            "archive_name": archive_name,
            "archive_path": zpath,
            "extracted_file_count": 0,
            "extracted_file_paths": [],
            "extraction_status": "PENDING",
            "errors": [],
        }

        try:
            with zipfile.ZipFile(zpath, "r") as zf:
                for member in zf.infolist():
                    # Security check: avoid directory traversal
                    norm_name = os.path.normpath(member.filename)
                    if norm_name.startswith("..") or os.path.isabs(norm_name):
                        entry["errors"].append(f"Skipped unsafe path: {member.filename}")
                        continue

                    if member.is_dir():
                        continue

                    # Safe target extraction path
                    target_file = os.path.join(staging_dir, os.path.basename(member.filename))
                    
                    # Extract only if not already extracted or has different size
                    should_extract = True
                    if os.path.exists(target_file):
                        if os.path.getsize(target_file) == member.file_size:
                            should_extract = False

                    if should_extract:
                        with zf.open(member) as src, open(target_file, "wb") as dst:
                            dst.write(src.read())

                    entry["extracted_file_paths"].append(target_file)
                    entry["extracted_file_count"] += 1
                    total_files_extracted += 1

            entry["extraction_status"] = "SUCCESS"
        except Exception as e:
            entry["extraction_status"] = "FAILED"
            entry["errors"].append(str(e))

        manifest_entries.append(entry)

    summary = {
        "archives_processed": len(zip_files),
        "total_extracted_files": total_files_extracted,
        "staging_dir": staging_dir,
    }

    return manifest_entries, summary


def discover_battery_files(staging_dir: Optional[str] = None) -> Tuple[List[str], List[str]]:
    """
    Discover all supported .mat battery files and any non-mat files (e.g. README.txt).
    Returns (mat_files, non_mat_files).
    """
    if staging_dir is None:
        staging_dir = get_safe_staging_dir()

    mat_files = sorted(glob.glob(os.path.join(staging_dir, "*.mat")))
    all_files = sorted(glob.glob(os.path.join(staging_dir, "*.*")))
    non_mat_files = [f for f in all_files if not f.endswith(".mat")]

    return mat_files, non_mat_files


def load_battery_mat(filepath: str) -> Dict:
    """
    Load a NASA battery .mat file.
    Returns loaded mat object dictionary.
    """
    try:
        from scipy.io import loadmat
    except ImportError:
        raise ImportError("scipy is required: pip install scipy")

    return loadmat(filepath, squeeze_me=True, struct_as_record=False)
