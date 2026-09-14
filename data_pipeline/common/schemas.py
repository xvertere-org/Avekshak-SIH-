"""
Common metadata schemas for processed dataset records.

Every processed record carries provenance metadata so that outputs
can be traced back to their source files and processing configuration.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
import json


# Unsafe column names that must NEVER appear in C-MAPSS processed outputs
UNSAFE_CMAPSS_COLUMN_NAMES = frozenset({
    "cht", "egt", "oil_temp", "oil_pressure", "fuel_flow",
})

# Integration role categories
INTEGRATION_ROLES = {
    "AERO_PISTON_DIRECT",
    "AERO_PISTON_SUPPORTING",
    "VIBRATION_FEATURE_SOURCE",
    "RUL_METHODOLOGY_BENCHMARK",
    "DOMAIN_TRANSFER_EXPERIMENT",
}

# Compatibility categories
COMPATIBILITY_CATEGORIES = {
    "DIRECTLY_USABLE",
    "USABLE_AFTER_DOMAIN_SPECIFIC_PREPROCESSING",
    "METHODOLOGY_BENCHMARK_ONLY",
    "UNAVAILABLE",
    "UNSAFE_FOR_DIRECT_INTEGRATION",
}


@dataclass
class ProcessedRecordMetadata:
    """Metadata attached to every processed record / window."""

    dataset_name: str
    source_file: str
    source_checksum: Optional[str] = None
    source_unit_id: Optional[str] = None
    source_bearing_id: Optional[str] = None
    source_run_id: Optional[str] = None
    source_channel: Optional[str] = None
    original_sampling_rate: Optional[float] = None
    window_start: Optional[int] = None
    window_end: Optional[int] = None
    window_length: Optional[int] = None
    operating_condition: Optional[str] = None
    fault_label: Optional[str] = None
    degradation_label: Optional[str] = None
    rul_label: Optional[float] = None
    feature_version: str = "0.1.0"
    preprocessing_version: str = "0.1.0"
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


@dataclass
class DatasetAuditRecord:
    """Structured audit result for a single dataset."""

    dataset_name: str
    availability_status: str  # PRESENT, ARCHIVED, PARTIAL, UNAVAILABLE
    local_path: str
    file_count: int
    total_size_bytes: int
    file_formats: list
    dataset_purpose: str
    signal_channel_inventory: list
    sampling_frequency: Optional[str] = None
    labels_and_targets: Optional[str] = None
    operating_conditions: Optional[str] = None
    missing_values: Optional[str] = None
    corrupted_or_incomplete_files: Optional[str] = None
    duplicate_files: Optional[str] = None
    data_leakage_risks: Optional[str] = None
    suitable_project_use: Optional[str] = None
    unsuitable_project_use: Optional[str] = None
    required_preprocessing: Optional[str] = None
    compatibility_with_aero_piston: Optional[str] = None
    final_recommendation: Optional[str] = None
    integration_role: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)
