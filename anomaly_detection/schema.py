"""
Core data schemas and contracts for Phase 7 Hybrid Anomaly Detection.

Defines the output data structures, status enums, and primary feature channels.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd


class AnomalyStatus(str, Enum):
    """
    Standard anomaly state classifications for Phase 7.
    """
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    ANOMALY = "ANOMALY"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


# The canonical 7 normalized residual feature names consumed by Phase 7 detectors
PRIMARY_NORMALIZED_RESIDUAL_CHANNELS: List[str] = [
    "rpm_norm_residual",
    "cht_norm_residual",
    "egt_norm_residual",
    "oil_temp_norm_residual",
    "oil_pressure_norm_residual",
    "fuel_flow_norm_residual",
    "vibration_norm_residual",
]

# Raw residual channel equivalents (for reporting / reference, not for ML features)
RAW_RESIDUAL_CHANNELS: List[str] = [
    "rpm_residual",
    "cht_residual",
    "egt_residual",
    "oil_temp_residual",
    "oil_pressure_residual",
    "fuel_flow_residual",
    "vibration_residual",
]


@dataclass
class AnomalyRecord:
    """
    Structured anomaly assessment record for an individual telemetry sample.
    """
    timestamp: float
    engine_id: str
    mission_id: str
    mission_phase: str
    source: str
    source_type: str

    anomaly_score: Optional[float]
    anomaly_status: str

    threshold_score: Optional[float] = None
    ewma_score: Optional[float] = None
    persistence_score: Optional[float] = None
    isolation_score: Optional[float] = None

    persistence_count: int = 0
    contributing_channels: List[str] = field(default_factory=list)

    quality_status: Optional[str] = None
    missing_features: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnomalyRecord":
        return cls(**data)


class AnomalyFrame:
    """
    Tabular container for Phase 7 anomaly detection results.
    Wraps a pandas DataFrame with schema accessors, filtering, and summary helpers.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self._data = data.copy()
        self._metadata = dict(metadata) if metadata is not None else {}

    @property
    def data(self) -> pd.DataFrame:
        """Direct access to underlying DataFrame."""
        return self._data

    @property
    def metadata(self) -> Dict[str, Any]:
        """Configuration and run metadata."""
        return self._metadata

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the underlying DataFrame."""
        return self._data.copy()

    def to_records(self) -> List[Dict[str, Any]]:
        """Return frame rows as a list of dictionaries."""
        return self._data.to_dict(orient="records")

    def anomalies(self) -> pd.DataFrame:
        """Filter rows classified as ANOMALY."""
        if "anomaly_status" not in self._data.columns:
            return pd.DataFrame()
        return self._data[self._data["anomaly_status"] == AnomalyStatus.ANOMALY.value].copy()

    def warnings(self) -> pd.DataFrame:
        """Filter rows classified as WARNING."""
        if "anomaly_status" not in self._data.columns:
            return pd.DataFrame()
        return self._data[self._data["anomaly_status"] == AnomalyStatus.WARNING.value].copy()

    def insufficient_data(self) -> pd.DataFrame:
        """Filter rows classified as INSUFFICIENT_DATA."""
        if "anomaly_status" not in self._data.columns:
            return pd.DataFrame()
        return self._data[self._data["anomaly_status"] == AnomalyStatus.INSUFFICIENT_DATA.value].copy()

    def normal(self) -> pd.DataFrame:
        """Filter rows classified as NORMAL."""
        if "anomaly_status" not in self._data.columns:
            return pd.DataFrame()
        return self._data[self._data["anomaly_status"] == AnomalyStatus.NORMAL.value].copy()

    def summary(self) -> Dict[str, Any]:
        """Compute high-level anomaly statistics."""
        total = len(self._data)
        if total == 0:
            return {
                "total_samples": 0,
                "normal_count": 0,
                "warning_count": 0,
                "anomaly_count": 0,
                "insufficient_data_count": 0,
                "anomaly_rate": 0.0,
            }
        counts = self._data["anomaly_status"].value_counts().to_dict()
        return {
            "total_samples": total,
            "normal_count": counts.get(AnomalyStatus.NORMAL.value, 0),
            "warning_count": counts.get(AnomalyStatus.WARNING.value, 0),
            "anomaly_count": counts.get(AnomalyStatus.ANOMALY.value, 0),
            "insufficient_data_count": counts.get(AnomalyStatus.INSUFFICIENT_DATA.value, 0),
            "anomaly_rate": round(counts.get(AnomalyStatus.ANOMALY.value, 0) / total, 4),
        }

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key: Any) -> Any:
        return self._data[key]

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"<AnomalyFrame: {s['total_samples']} samples | "
            f"NORMAL: {s['normal_count']} | WARNING: {s['warning_count']} | "
            f"ANOMALY: {s['anomaly_count']} | INSUFFICIENT_DATA: {s['insufficient_data_count']}>"
        )
