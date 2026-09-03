"""
Telemetry Ingestion and Canonical Schema for SIH26054 Aero Piston Engine Digital Twin.

Transforms raw telemetry records, streams, DataFrames, and external inputs into a
standardized, typed CanonicalTelemetryFrame while preserving provenance and metadata.

Design Rules:
- Explicit required vs. optional channels: not all channels are present in every dataset
  (e.g., external substitute vibration benchmarks or phase-dependent telemetry).
- Zero silent metadata loss: all auxiliary fields and provenance tags are preserved.
- Non-destructive: raw records remain unchanged.
"""

from typing import List, Dict, Any, Optional, Union, Iterator, Set
import pandas as pd
import numpy as np

from telemetry.schema import TelemetryRecord, FaultCategory


# Canonical channel definitions
REQUIRED_COLUMNS: Set[str] = {
    "timestamp",
    "engine_id",
    "mission_id",
    "source",
    "source_type",
}

OPTIONAL_PHYSICAL_CHANNELS: Set[str] = {
    "altitude",
    "ambient_temp",
    "throttle",
    "load",
    "rpm",
    "cht",
    "egt",
    "oil_temp",
    "oil_pressure",
    "fuel_flow",
    "vibration",
}

OPTIONAL_METADATA_COLUMNS: Set[str] = {
    "mission_phase",
    "fault_type",
    "fault_severity",
    "simulation_version",
    "metadata",
}

ALL_CANONICAL_COLUMNS: Set[str] = REQUIRED_COLUMNS | OPTIONAL_PHYSICAL_CHANNELS | OPTIONAL_METADATA_COLUMNS

# Standard column ordering for presentation
CANONICAL_COLUMN_ORDER: List[str] = [
    "timestamp",
    "engine_id",
    "mission_id",
    "mission_phase",
    "altitude",
    "ambient_temp",
    "throttle",
    "load",
    "rpm",
    "cht",
    "egt",
    "oil_temp",
    "oil_pressure",
    "fuel_flow",
    "vibration",
    "fault_type",
    "fault_severity",
    "source",
    "source_type",
    "simulation_version",
    "metadata",
]


class CanonicalTelemetryFrame:
    """
    Standardized tabular container for aero engine telemetry.
    Wraps a pandas DataFrame with strict column-level schema awareness,
    provenance retention, and helper accessors.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize CanonicalTelemetryFrame.

        Args:
            data: DataFrame containing canonical telemetry data.
            metadata: Optional dataset-level metadata or provenance dictionary.
        """
        self._data = data.copy()
        self._metadata = dict(metadata) if metadata is not None else {}

    @property
    def data(self) -> pd.DataFrame:
        """Access the underlying pandas DataFrame."""
        return self._data

    @property
    def metadata(self) -> Dict[str, Any]:
        """Dataset-level metadata."""
        return self._metadata

    @property
    def channels_present(self) -> Set[str]:
        """Set of canonical columns actually present in this frame."""
        return set(self._data.columns).intersection(ALL_CANONICAL_COLUMNS)

    @property
    def physical_channels_present(self) -> Set[str]:
        """Set of physical engine telemetry channels present in this frame."""
        return set(self._data.columns).intersection(OPTIONAL_PHYSICAL_CHANNELS)

    @property
    def missing_canonical_channels(self) -> Set[str]:
        """Canonical channels that are not present in this frame."""
        return ALL_CANONICAL_COLUMNS - set(self._data.columns)

    @property
    def provenance(self) -> Dict[str, Any]:
        """Summary of provenance attributes across the frame."""
        sources = list(self._data["source"].unique()) if "source" in self._data.columns else []
        source_types = list(self._data["source_type"].unique()) if "source_type" in self._data.columns else []
        sim_vers = list(self._data["simulation_version"].unique()) if "simulation_version" in self._data.columns else []
        return {
            "source": sources[0] if len(sources) == 1 else sources,
            "source_type": source_types[0] if len(source_types) == 1 else source_types,
            "simulation_version": sim_vers[0] if len(sim_vers) == 1 else sim_vers,
            "row_count": len(self._data),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the underlying DataFrame."""
        return self._data.copy()

    def to_records(self) -> List[Dict[str, Any]]:
        """Return frame rows as a list of dictionaries."""
        return self._data.to_dict(orient="records")

    def copy(self) -> "CanonicalTelemetryFrame":
        """Return a deep copy of the frame."""
        return CanonicalTelemetryFrame(data=self._data.copy(), metadata=dict(self._metadata))

    def filter_by_phase(self, phase: str) -> "CanonicalTelemetryFrame":
        """Filter rows by mission phase (if present)."""
        if "mission_phase" not in self._data.columns:
            return self.copy()
        filtered = self._data[self._data["mission_phase"].str.upper() == phase.upper()].copy()
        return CanonicalTelemetryFrame(filtered, metadata=dict(self._metadata))

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key: Any) -> Any:
        return self._data[key]

    def __repr__(self) -> str:
        rows = len(self._data)
        channels = len(self.physical_channels_present)
        src = self.provenance.get("source", "unknown")
        return f"<CanonicalTelemetryFrame: {rows} samples, {channels} physical channels, source='{src}'>"


class TelemetryIngestor:
    """
    Ingestion factory converting diverse telemetry sources into CanonicalTelemetryFrame.

    Supports:
    - Single TelemetryRecord
    - List or Iterator of TelemetryRecord
    - TelemetryStreamer instances
    - pandas DataFrames
    - List of dictionaries
    """

    @classmethod
    def ingest(
        cls,
        input_data: Union[
            TelemetryRecord,
            List[TelemetryRecord],
            Iterator[TelemetryRecord],
            pd.DataFrame,
            List[Dict[str, Any]],
            Any,
        ],
        default_source: str = "simulator_v1_physics",
        default_source_type: str = "synthetic",
        default_engine_id: str = "ENGINE_UAV_01",
        default_mission_id: str = "MISSION_001",
        strict: bool = True,
    ) -> CanonicalTelemetryFrame:
        """
        Ingest telemetry data and return a CanonicalTelemetryFrame.

        Args:
            input_data: Telemetry data in any supported format.
            default_source: Fallback source identifier if missing.
            default_source_type: Fallback source type if missing.
            default_engine_id: Fallback engine identifier if missing.
            default_mission_id: Fallback mission identifier if missing.
            strict: If True, raises ValueError on missing REQUIRED_COLUMNS.

        Returns:
            CanonicalTelemetryFrame instance.
        """
        records: List[Dict[str, Any]] = []

        # Handle TelemetryStreamer instance
        if hasattr(input_data, "get_buffer"):
            input_data = input_data.get_buffer()

        # Handle single TelemetryRecord
        if isinstance(input_data, TelemetryRecord):
            records = [input_data.to_dict()]
        # Handle list/iterable
        elif isinstance(input_data, (list, tuple)):
            if len(input_data) == 0:
                df = pd.DataFrame(columns=list(REQUIRED_COLUMNS))
                return CanonicalTelemetryFrame(df)
            first = input_data[0]
            if isinstance(first, TelemetryRecord):
                records = [r.to_dict() for r in input_data]
            elif isinstance(first, dict):
                records = [dict(d) for d in input_data]
            else:
                raise TypeError(f"Unsupported sequence item type: {type(first)}")
        # Handle iterator/generator
        elif hasattr(input_data, "__iter__") and not isinstance(input_data, pd.DataFrame):
            for item in input_data:
                if isinstance(item, TelemetryRecord):
                    records.append(item.to_dict())
                elif isinstance(item, dict):
                    records.append(dict(item))
                else:
                    raise TypeError(f"Unsupported iterator item type: {type(item)}")
        # Handle pandas DataFrame
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
            return cls._normalize_dataframe(
                df,
                default_source=default_source,
                default_source_type=default_source_type,
                default_engine_id=default_engine_id,
                default_mission_id=default_mission_id,
                strict=strict,
            )
        else:
            raise TypeError(f"Unsupported input type for TelemetryIngestor: {type(input_data)}")

        df = pd.DataFrame(records)
        return cls._normalize_dataframe(
            df,
            default_source=default_source,
            default_source_type=default_source_type,
            default_engine_id=default_engine_id,
            default_mission_id=default_mission_id,
            strict=strict,
        )

    @classmethod
    def _normalize_dataframe(
        cls,
        df: pd.DataFrame,
        default_source: str,
        default_source_type: str,
        default_engine_id: str,
        default_mission_id: str,
        strict: bool,
    ) -> CanonicalTelemetryFrame:
        """Apply canonical typing, defaults, and ordering to DataFrame."""
        df = df.copy()

        # Assign defaults for missing required identity columns if not in strict mode or if None
        if "source" not in df.columns or df["source"].isnull().all():
            df["source"] = default_source
        else:
            df["source"] = df["source"].fillna(default_source)

        if "source_type" not in df.columns or df["source_type"].isnull().all():
            df["source_type"] = default_source_type
        else:
            df["source_type"] = df["source_type"].fillna(default_source_type)

        if "engine_id" not in df.columns or df["engine_id"].isnull().all():
            df["engine_id"] = default_engine_id
        else:
            df["engine_id"] = df["engine_id"].fillna(default_engine_id)

        if "mission_id" not in df.columns or df["mission_id"].isnull().all():
            df["mission_id"] = default_mission_id
        else:
            df["mission_id"] = df["mission_id"].fillna(default_mission_id)

        # Enforce required columns check
        missing_req = REQUIRED_COLUMNS - set(df.columns)
        if missing_req and strict:
            raise ValueError(f"Telemetry ingestion failed: missing required column(s): {sorted(list(missing_req))}")

        # Ensure timestamp is numeric
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

        # Cast physical channels to numeric float where present (preserving NaN for dropout)
        for col in OPTIONAL_PHYSICAL_CHANNELS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

        if "fault_severity" in df.columns:
            df["fault_severity"] = pd.to_numeric(df["fault_severity"], errors="coerce").fillna(0.0).astype(float)

        if "fault_type" in df.columns:
            df["fault_type"] = df["fault_type"].fillna(FaultCategory.NONE.value).astype(str)

        # Order columns cleanly: present canonical columns first, then auxiliary
        present_ordered = [c for c in CANONICAL_COLUMN_ORDER if c in df.columns]
        extra_cols = [c for c in df.columns if c not in CANONICAL_COLUMN_ORDER]
        final_cols = present_ordered + extra_cols
        df = df[final_cols]

        return CanonicalTelemetryFrame(df)
