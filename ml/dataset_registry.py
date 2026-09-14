"""
Dataset Registry for SIH26054 Aero-Piston Engine Grey-Box Digital Twin.

Documents the purpose, limitations, integration roles, and grouping candidate
columns of all benchmark and supporting datasets.

Strictly preserves physics segregation: benchmark and component datasets
(C-MAPSS, NASA Battery, CWRU, FEMTO, Paderborn, NUST) are NEVER treated
as direct Rotax 914 engine telemetry.
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import json
import os


class DatasetUnavailableError(Exception):
    """Raised when an operation is attempted on an unavailable dataset."""
    pass


class IncompatibleDatasetError(Exception):
    """Raised when incompatible datasets are attempted to be merged or concatenated."""
    pass


class GroupColumnNotFoundError(Exception):
    """Raised when no valid candidate grouping column is found in the dataset schema."""
    pass


@dataclass
class DatasetMetadata:
    """Metadata describing a dataset's purpose, integration role, and constraints."""
    dataset_id: str
    name: str
    display_name: str
    availability: str  # "AVAILABLE" or "UNAVAILABLE"
    integration_role: str
    target_tasks: List[str]
    processed_path: str
    file_pattern: str
    candidate_grouping_columns: List[str]
    target_columns: List[str]
    feature_types: List[str]
    is_direct_rotax_telemetry: bool = False
    limitations: List[str] = field(default_factory=list)

    def is_available(self) -> bool:
        return self.availability.upper() == "AVAILABLE"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DatasetRegistry:
    """
    Central registry for dataset metadata, constraints, and task definitions.
    """

    DEFAULT_CONFIG_PATH = Path(__file__).parent / "configs" / "dataset_config.json"

    # Core engine channels that must NEVER be mapped from benchmark/bearing datasets
    PROTECTED_ROTAX_CHANNELS: Set[str] = frozenset({
        "cht", "egt", "oil_temp", "oil_pressure", "fuel_flow", "map_bar",
        "propeller_rpm", "engine_rpm", "coolant_temp", "tcu_wastegate_position",
        "cht_cyl1", "cht_cyl2", "cht_cyl3", "cht_cyl4",
        "egt_cyl1", "egt_cyl2", "egt_cyl3", "egt_cyl4",
    })

    def __init__(self, config_path: Optional[str or Path] = None):
        self.config_path = Path(config_path) if config_path else self.DEFAULT_CONFIG_PATH
        self._datasets: Dict[str, DatasetMetadata] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load dataset definitions from the JSON configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Dataset configuration not found at {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for key, entry in data.get("datasets", {}).items():
            self._datasets[key] = DatasetMetadata(
                dataset_id=key,
                name=entry.get("name", key),
                display_name=entry.get("display_name", key),
                availability=entry.get("availability", "UNAVAILABLE"),
                integration_role=entry.get("integration_role", "UNKNOWN"),
                target_tasks=entry.get("target_tasks", []),
                processed_path=entry.get("processed_path", ""),
                file_pattern=entry.get("file_pattern", "*.parquet"),
                candidate_grouping_columns=entry.get("candidate_grouping_columns", []),
                target_columns=entry.get("target_columns", []),
                feature_types=entry.get("feature_types", []),
                is_direct_rotax_telemetry=entry.get("is_direct_rotax_telemetry", False),
                limitations=entry.get("limitations", []),
            )

    def get(self, dataset_id: str) -> DatasetMetadata:
        """Retrieve metadata for a dataset by ID."""
        key = dataset_id.lower().strip()
        if key not in self._datasets:
            raise KeyError(
                f"Dataset '{dataset_id}' not found in registry. Registered datasets: {list(self._datasets.keys())}"
            )
        return self._datasets[key]

    def list_datasets(self) -> List[str]:
        """Return a list of all registered dataset IDs."""
        return list(self._datasets.keys())

    def list_available_datasets(self) -> List[str]:
        """Return a list of available dataset IDs."""
        return [k for k, v in self._datasets.items() if v.is_available()]

    def is_available(self, dataset_id: str) -> bool:
        """Check if a dataset is marked as available."""
        try:
            return self.get(dataset_id).is_available()
        except KeyError:
            return False

    def resolve_grouping_column(self, dataset_id: str, existing_columns: List[str]) -> str:
        """
        Dynamically determine the actual grouping column present in a DataFrame
        from the registered candidate grouping columns.

        Does NOT assume or hardcode a single column name.
        """
        meta = self.get(dataset_id)
        if not meta.candidate_grouping_columns:
            raise GroupColumnNotFoundError(
                f"Dataset '{dataset_id}' has no registered candidate grouping columns."
            )

        for candidate in meta.candidate_grouping_columns:
            if candidate in existing_columns:
                return candidate

        raise GroupColumnNotFoundError(
            f"None of the candidate grouping columns {meta.candidate_grouping_columns} "
            f"for dataset '{dataset_id}' were found in actual columns: {existing_columns}"
        )

    def check_rotax_channel_collision(self, columns: List[str], dataset_id: str) -> List[str]:
        """
        Verify that no non-Rotax dataset attempts to define or shadow protected Rotax engine channels.
        """
        meta = self.get(dataset_id)
        if meta.is_direct_rotax_telemetry:
            return []

        collisions = [col for col in columns if col.lower() in self.PROTECTED_ROTAX_CHANNELS]
        return collisions

    def validate_compatibility(self, dataset_ids: List[str]) -> None:
        """
        Validate whether multiple datasets can be loaded together.
        Strictly forbids combining incompatible physical systems (e.g. turbofans and bearings).
        """
        if len(dataset_ids) <= 1:
            return

        roles = set()
        for d_id in dataset_ids:
            meta = self.get(d_id)
            if not meta.is_available():
                raise DatasetUnavailableError(
                    f"Dataset '{d_id}' is unavailable and cannot be combined."
                )
            roles.add(meta.integration_role)

        if len(roles) > 1:
            raise IncompatibleDatasetError(
                f"Cannot combine datasets across incompatible integration roles: {roles}. "
                f"Datasets must share identical physical domain and task scope."
            )
