"""
Dataset-Specific ML Adapters for Phase 2 Grey-Box Baselines.

Inspects actual Parquet schemas at runtime, dynamically resolves targets and grouping keys,
and rigorously filters out identifiers, target proxies, metadata, and future-derived columns.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import numpy as np
import pandas as pd

from ml.dataset_registry import DatasetRegistry, DatasetUnavailableError, GroupColumnNotFoundError
from ml.data_loader import DataLoader
from ml.tasks.exceptions import (
    TargetNotFoundError,
    FeatureSelectionError,
    HealthyReferenceNotFoundError,
    IncompatibleDomainError,
)

# Standard metadata and identifier columns to exclude from feature matrices
METADATA_AND_ID_COLUMNS: Set[str] = frozenset({
    "dataset_name", "record_id", "source_file", "source_id", "window_id",
    "measurement_id", "source_channel", "source_checksum", "preprocessing_version",
    "feature_version", "split", "subset", "file_index", "total_files",
    "file_sequence", "source_bearing_id", "source_run_id", "dataset_variant",
    "window_start", "window_end", "window_length", "sensor_location",
    "sampling_frequency", "original_sampling_rate", "operating_region",
})

# Known target columns and their potential proxies / future leakage sources
TARGET_AND_PROXY_COLUMNS: Set[str] = frozenset({
    "fault_label", "fault_size", "fault_size_mils", "bearing_condition",
    "rul_label", "soh", "soh_clipped", "capacity_loss", "discharge_capacity",
    "reference_capacity", "re_electrolyte_resistance", "rct_charge_transfer_resistance",
})


@dataclass
class AdapterConfig:
    """Configuration for dataset-specific adapter resolution."""
    dataset_id: str
    target_candidate_columns: List[str]
    healthy_label_candidate: Optional[str] = None
    time_column: Optional[str] = None
    excluded_features: List[str] = field(default_factory=list)


DATASET_ADAPTER_REGISTRY: Dict[str, AdapterConfig] = {
    "cwru": AdapterConfig(
        dataset_id="cwru",
        target_candidate_columns=["fault_label"],
        healthy_label_candidate="normal",
        time_column="window_id",
        excluded_features=["RPM", "motor_load_hp"],
    ),
    "paderborn": AdapterConfig(
        dataset_id="paderborn",
        target_candidate_columns=["fault_label"],
        healthy_label_candidate="healthy",
        time_column="window_id",
        excluded_features=["speed_rpm", "torque_nm", "radial_force_n", "window_size", "overlap"],
    ),
    "nust": AdapterConfig(
        dataset_id="nust",
        target_candidate_columns=["bearing_condition"],
        healthy_label_candidate="healthy",
        time_column="Time",
        excluded_features=["rpm", "humidity_pct", "temperature_celsius"],
    ),
    "femto": AdapterConfig(
        dataset_id="femto",
        target_candidate_columns=["rul_label"],
        healthy_label_candidate=None,  # Run-to-failure; no explicit healthy label
        time_column="file_index",
        excluded_features=[],
    ),
    "nasa_battery": AdapterConfig(
        dataset_id="nasa_battery",
        target_candidate_columns=["soh", "capacity_loss"],
        healthy_label_candidate=None,
        time_column="cycle",
        excluded_features=["duration_s"],
    ),
    "cmapss": AdapterConfig(
        dataset_id="cmapss",
        target_candidate_columns=["rul_label"],
        healthy_label_candidate=None,
        time_column="time_cycles",
        excluded_features=["op_setting_1", "op_setting_2", "op_setting_3"],
    ),
}


class DatasetAdapter:
    """
    Adapter responsible for loading a dataset and producing a clean,
    leakage-safe feature matrix and validated target vector.
    """

    def __init__(
        self,
        dataset_id: str,
        loader: Optional[DataLoader] = None,
        registry: Optional[DatasetRegistry] = None,
    ):
        self.dataset_id = dataset_id.lower().strip()
        self.registry = registry or DatasetRegistry()
        self.loader = loader or DataLoader(registry=self.registry)

        if not self.registry.is_available(self.dataset_id):
            meta = self.registry.get(self.dataset_id)
            raise DatasetUnavailableError(
                f"Dataset '{self.dataset_id}' is unavailable. Limitations: {meta.limitations}"
            )

        if self.dataset_id not in DATASET_ADAPTER_REGISTRY:
            raise KeyError(f"No AdapterConfig registered for dataset '{self.dataset_id}'.")

        self.config = DATASET_ADAPTER_REGISTRY[self.dataset_id]

    def load_and_adapt(
        self,
        target_column: Optional[str] = None,
        subset: Optional[str] = None,
        split: Optional[str] = None,
        file_path: Optional[Path] = None,
    ) -> Tuple[pd.DataFrame, str, str, List[str]]:
        """
        Load dataset, validate and resolve target, grouping key, and clean feature list.

        Returns:
            df (pd.DataFrame): Raw loaded DataFrame.
            resolved_target (str): Actual validated target column present in schema.
            resolved_group (str): Actual grouping column present in schema.
            feature_columns (List[str]): List of strictly numeric, non-leakage feature column names.
        """
        df = self.loader.load_dataset(
            self.dataset_id,
            subset=subset,
            split=split,
            file_path=file_path,
        )

        existing_cols = list(df.columns)

        # 1. Resolve Target Column with Runtime Validation
        resolved_target = None
        if target_column:
            if target_column not in df.columns:
                raise TargetNotFoundError(
                    f"Requested target '{target_column}' does not exist in schema for '{self.dataset_id}'. "
                    f"Existing columns: {existing_cols}"
                )
            resolved_target = target_column
        else:
            for cand in self.config.target_candidate_columns:
                if cand in df.columns:
                    resolved_target = cand
                    break

        if not resolved_target:
            raise TargetNotFoundError(
                f"None of the candidate target columns {self.config.target_candidate_columns} "
                f"were found in dataset '{self.dataset_id}'. Available columns: {existing_cols}"
            )

        # 2. Resolve Grouping Column Dynamically
        resolved_group = self.registry.resolve_grouping_column(self.dataset_id, existing_cols)

        # 3. Derive Clean Numeric Feature Columns
        excluded = (
            set(METADATA_AND_ID_COLUMNS)
            | set(TARGET_AND_PROXY_COLUMNS)
            | set(self.config.excluded_features)
            | {resolved_target, resolved_group}
        )
        if self.config.time_column:
            excluded.add(self.config.time_column)

        candidate_features = [
            c for c in existing_cols
            if c not in excluded
            and pd.api.types.is_numeric_dtype(df[c])
        ]

        if not candidate_features:
            raise FeatureSelectionError(
                f"No valid numeric feature columns identified for '{self.dataset_id}' after excluding {excluded}."
            )

        return df, resolved_target, resolved_group, candidate_features

    def resolve_healthy_label(self, df: pd.DataFrame, target_column: str) -> str:
        """
        Resolve and verify the healthy/normal class label in the target column.
        Raises HealthyReferenceNotFoundError if not found or unconfigured.
        """
        candidate = self.config.healthy_label_candidate
        if not candidate:
            raise HealthyReferenceNotFoundError(
                f"Dataset '{self.dataset_id}' has no configured healthy/normal reference class."
            )

        unique_labels = set(df[target_column].dropna().unique())
        if candidate not in unique_labels:
            raise HealthyReferenceNotFoundError(
                f"Configured healthy reference '{candidate}' was not found in target '{target_column}'. "
                f"Actual labels present: {unique_labels}"
            )

        return candidate
