"""
Leakage-Safe Processed Data Loader for ML Integration.

Responsible for:
1. Loading processed Parquet files from data/processed/.
2. Safe rejection of raw archives (.zip, .mat, .txt, .csv) as canonical ML inputs.
3. Schema validation and protection against Rotax channel collisions.
4. Preserving dataset-specific metadata and column identities.
5. Incompatible dataset combination prevention.
6. Generating comprehensive, verified dataset summaries.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import glob
import json
import numpy as np
import pandas as pd

from ml.dataset_registry import (
    DatasetRegistry,
    DatasetMetadata,
    DatasetUnavailableError,
    IncompatibleDatasetError,
    GroupColumnNotFoundError,
)


class InvalidFileFormatError(Exception):
    """Raised when an attempt is made to load a raw, non-canonical ML file format."""
    pass


class SchemaValidationError(Exception):
    """Raised when the loaded dataset fails required schema validations."""
    pass


@dataclass
class DatasetSummary:
    """Structured summary of a loaded processed dataset with actual verified counts."""
    dataset_name: str
    file_path: str
    num_rows: int
    num_features: int
    available_labels: List[str]
    grouping_column: str
    target_task: List[str]
    missing_value_count: int
    infinite_value_count: int
    dataset_limitations: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DataLoader:
    """
    Modular loader for processed machine-learning datasets.
    """

    ALLOWED_EXTENSIONS = frozenset({".parquet"})
    DISALLOWED_RAW_EXTENSIONS = frozenset({".zip", ".mat", ".txt", ".csv", ".rar", ".7z", ".tar", ".gz"})

    def __init__(self, project_root: Optional[Union[str, Path]] = None, registry: Optional[DatasetRegistry] = None):
        if project_root is None:
            self.project_root = Path(__file__).parent.parent.resolve()
        else:
            self.project_root = Path(project_root).resolve()

        self.registry = registry or DatasetRegistry()

    def _resolve_path(self, relative_path: str) -> Path:
        """Resolve a path relative to the project root."""
        return (self.project_root / relative_path).resolve()

    def validate_file_path(self, file_path: Union[str, Path]) -> Path:
        """
        Validate that the target file exists, is a canonical Parquet file,
        and is NOT a disallowed raw archive or script.
        """
        p = Path(file_path)
        if not p.is_absolute():
            p = self._resolve_path(str(file_path))

        suffix = p.suffix.lower()
        if suffix in self.DISALLOWED_RAW_EXTENSIONS:
            raise InvalidFileFormatError(
                f"Cannot load raw file '{p.name}' as canonical ML input. "
                f"ML models must only ingest canonical processed Parquet files from data/processed/. "
                f"Raw formats ({suffix}) are disallowed."
            )

        if suffix not in self.ALLOWED_EXTENSIONS:
            raise InvalidFileFormatError(
                f"Unsupported file format '{suffix}' for '{p.name}'. Only Parquet files are permitted."
            )

        if not p.exists():
            raise FileNotFoundError(f"Processed file not found at: {p}")

        return p

    def discover_files(self, dataset_id: str, subset: Optional[str] = None, split: Optional[str] = None) -> List[Path]:
        """
        Discover existing processed parquet files for a given dataset ID.
        """
        meta = self.registry.get(dataset_id)
        if not meta.is_available():
            raise DatasetUnavailableError(
                f"Dataset '{dataset_id}' is marked as UNAVAILABLE. "
                f"Reasons/Limitations: {'; '.join(meta.limitations)}"
            )

        base_dir = self._resolve_path(meta.processed_path)
        if not base_dir.exists():
            raise FileNotFoundError(
                f"Processed directory for '{dataset_id}' does not exist at {base_dir}."
            )

        pattern = meta.file_pattern
        all_files = sorted(base_dir.glob(pattern))

        if not all_files:
            raise FileNotFoundError(
                f"No processed files matching '{pattern}' found in {base_dir} for dataset '{dataset_id}'."
            )

        filtered_files = all_files
        if subset:
            filtered_files = [f for f in filtered_files if subset.lower() in f.name.lower()]
        if split:
            filtered_files = [f for f in filtered_files if split.lower() in f.name.lower()]

        if not filtered_files:
            raise FileNotFoundError(
                f"No processed files found for '{dataset_id}' with subset='{subset}' and split='{split}'."
            )

        return filtered_files

    def load_dataset(
        self,
        dataset_id: str,
        file_path: Optional[Union[str, Path]] = None,
        subset: Optional[str] = None,
        split: Optional[str] = None,
        required_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Load a processed dataset as a DataFrame.

        Validates:
        1. Non-raw file format.
        2. Channel collision with Rotax engine channels.
        3. Required columns presence.
        4. Preserves original column names and types.
        """
        meta = self.registry.get(dataset_id)

        if file_path is not None:
            path = self.validate_file_path(file_path)
        else:
            discovered = self.discover_files(dataset_id, subset=subset, split=split)
            path = discovered[0]

        df = pd.read_parquet(path)

        # 1. Rotax channel collision check
        collisions = self.registry.check_rotax_channel_collision(list(df.columns), dataset_id)
        if collisions:
            raise SchemaValidationError(
                f"Dataset '{dataset_id}' in file '{path.name}' contains protected Rotax engine channels: {collisions}. "
                f"Non-engine benchmark datasets must never shadow core piston engine channels."
            )

        # 2. Required columns check
        if required_columns:
            missing = [c for c in required_columns if c not in df.columns]
            if missing:
                raise SchemaValidationError(
                    f"Dataset '{dataset_id}' file '{path.name}' is missing required columns: {missing}"
                )

        return df

    def combine_compatible_datasets(
        self,
        dataset_ids: List[str],
        subsets: Optional[Dict[str, str]] = None,
        splits: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Combine multiple datasets ONLY if they are explicitly compatible.
        Strictly prevents combining disparate domains (e.g. C-MAPSS with FEMTO).
        """
        if not dataset_ids:
            raise ValueError("No dataset IDs provided to combine.")

        # Check compatibility via registry
        self.registry.validate_compatibility(dataset_ids)

        subsets = subsets or {}
        splits = splits or {}

        dfs = []
        reference_columns = None

        for d_id in dataset_ids:
            df = self.load_dataset(
                d_id,
                subset=subsets.get(d_id),
                split=splits.get(d_id),
            )
            if reference_columns is None:
                reference_columns = set(df.columns)
            else:
                # Require common subset of feature columns
                common = reference_columns.intersection(set(df.columns))
                if len(common) < 3:
                    raise IncompatibleDatasetError(
                        f"Datasets '{dataset_ids[0]}' and '{d_id}' have incompatible schemas (common columns: {list(common)})."
                    )

            dfs.append(df)

        return pd.concat(dfs, axis=0, ignore_index=True)

    def summarize_dataset(
        self,
        dataset_id: str,
        file_path: Optional[Union[str, Path]] = None,
        subset: Optional[str] = None,
        split: Optional[str] = None,
    ) -> DatasetSummary:
        """
        Generate a structured summary of actual observed values directly from the processed file.
        Does NOT fabricate or assume any metrics.
        """
        meta = self.registry.get(dataset_id)
        if not meta.is_available():
            return DatasetSummary(
                dataset_name=meta.name,
                file_path=meta.processed_path,
                num_rows=0,
                num_features=0,
                available_labels=[],
                grouping_column="NONE",
                target_task=meta.target_tasks,
                missing_value_count=0,
                infinite_value_count=0,
                dataset_limitations=meta.limitations,
            )

        if file_path is not None:
            path = self.validate_file_path(file_path)
        else:
            path = self.discover_files(dataset_id, subset=subset, split=split)[0]

        df = pd.read_parquet(path)
        existing_cols = list(df.columns)

        # Dynamically resolve grouping column
        try:
            grouping_col = self.registry.resolve_grouping_column(dataset_id, existing_cols)
        except GroupColumnNotFoundError:
            grouping_col = "NOT_FOUND"

        # Identify available target labels
        available_labels = [c for c in meta.target_columns if c in df.columns]

        # Calculate actual counts
        num_rows = int(len(df))
        num_features = int(len(existing_cols))
        missing_count = int(df.isnull().sum().sum())

        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            inf_count = int(np.isinf(numeric_df.to_numpy()).sum())
        else:
            inf_count = 0

        # Preserve relative path for report portability
        try:
            rel_path = str(path.relative_to(self.project_root)).replace("\\", "/")
        except ValueError:
            rel_path = str(path).replace("\\", "/")

        return DatasetSummary(
            dataset_name=meta.name,
            file_path=rel_path,
            num_rows=num_rows,
            num_features=num_features,
            available_labels=available_labels,
            grouping_column=grouping_col,
            target_task=meta.target_tasks,
            missing_value_count=missing_count,
            infinite_value_count=inf_count,
            dataset_limitations=meta.limitations,
        )
