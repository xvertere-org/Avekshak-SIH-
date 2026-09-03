"""
External Dataset Adapters for SIH26054 Digital Twin.

Provides lightweight ingestion boundaries for public benchmark and substitute datasets:
- GenericCSVAdapter: Configurable column mapping for arbitrary CSV telemetry
- VibrationBenchmarkAdapter: Normalizes bearing/gearbox vibration benchmarks (CWRU/Paderborn surrogate)
- CMAPSSBenchmarkAdapter: Normalizes NASA C-MAPSS turbofan degradation datasets

DISCLAIMER:
No single public dataset contains a complete aero-piston telemetry stack.
External datasets are strictly methodology and component substitutes.
All adapters enforce explicit provenance tags (source_type='external_benchmark' or
'substitute_dataset') and NEVER label substitute datasets as aero-piston engine data.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np

from telemetry.ingestion import CanonicalTelemetryFrame, TelemetryIngestor


class ExternalDatasetAdapter(ABC):
    """
    Abstract contract for converting external datasets into CanonicalTelemetryFrame.
    Enforces explicit provenance and substitute metadata tags.
    """

    @abstractmethod
    def adapt(
        self,
        source_data: Union[str, Path, pd.DataFrame],
        **kwargs,
    ) -> CanonicalTelemetryFrame:
        """
        Convert external dataset into a CanonicalTelemetryFrame.

        Args:
            source_data: File path (str/Path) or raw pandas DataFrame.
            **kwargs: Adapter-specific parameters.

        Returns:
            CanonicalTelemetryFrame instance.
        """
        pass


class GenericCSVAdapter(ExternalDatasetAdapter):
    """
    Adapter for converting arbitrary tabular CSV telemetry using a configurable column mapping.
    """

    def __init__(
        self,
        column_mapping: Optional[Dict[str, str]] = None,
        source_name: str = "generic_external_csv",
        source_type: str = "external_benchmark",
        default_engine_id: str = "EXT_ENGINE_01",
        default_mission_id: str = "EXT_MISSION_01",
    ):
        self.column_mapping = column_mapping or {}
        self.source_name = source_name
        self.source_type = source_type
        self.default_engine_id = default_engine_id
        self.default_mission_id = default_mission_id

    def adapt(
        self,
        source_data: Union[str, Path, pd.DataFrame],
        **kwargs,
    ) -> CanonicalTelemetryFrame:
        if isinstance(source_data, (str, Path)):
            df = pd.read_csv(source_data)
        elif isinstance(source_data, pd.DataFrame):
            df = source_data.copy()
        else:
            raise TypeError(f"Unsupported source_data type: {type(source_data)}")

        # Rename columns according to mapping
        if self.column_mapping:
            df = df.rename(columns=self.column_mapping)

        # Ensure timestamp exists
        if "timestamp" not in df.columns:
            # If no timestamp, generate sequential index as proxy
            df["timestamp"] = np.arange(len(df), dtype=float)

        return TelemetryIngestor.ingest(
            df,
            default_source=self.source_name,
            default_source_type=self.source_type,
            default_engine_id=self.default_engine_id,
            default_mission_id=self.default_mission_id,
            strict=False,
        )


class VibrationBenchmarkAdapter(ExternalDatasetAdapter):
    """
    Adapter for vibration and bearing fault benchmark datasets (e.g. CWRU, Paderborn, FEMTO).
    Preserves raw vibration signals and RPM where available.
    """

    def __init__(
        self,
        dataset_name: str = "cwru_bearing_surrogate",
        dt_s: float = 0.001,
        column_mapping: Optional[Dict[str, str]] = None,
    ):
        self.dataset_name = dataset_name
        self.dt_s = dt_s
        self.column_mapping = column_mapping or {
            "DE_time": "vibration",
            "FE_time": "vibration_fan_end",
            "RPM": "rpm",
        }

    def adapt(
        self,
        source_data: Union[str, Path, pd.DataFrame],
        engine_id: str = "BENCH_VIB_01",
        mission_id: str = "BENCH_TEST_01",
        **kwargs,
    ) -> CanonicalTelemetryFrame:
        if isinstance(source_data, (str, Path)):
            df = pd.read_csv(source_data)
        elif isinstance(source_data, pd.DataFrame):
            df = source_data.copy()
        else:
            raise TypeError(f"Unsupported source_data type: {type(source_data)}")

        df = df.rename(columns=self.column_mapping)

        # Synthesize timestamp if not present
        if "timestamp" not in df.columns:
            df["timestamp"] = np.arange(len(df), dtype=float) * self.dt_s

        frame = TelemetryIngestor.ingest(
            df,
            default_source=f"substitute_{self.dataset_name}",
            default_source_type="substitute_dataset",
            default_engine_id=engine_id,
            default_mission_id=mission_id,
            strict=False,
        )

        # Attach surrogate disclaimer metadata
        frame.metadata.update({
            "substitute_role": "bearing_mechanical_vibration_methodology",
            "disclaimer": "Component substitute dataset, NOT certified aero-piston telemetry.",
            "benchmark_dataset": self.dataset_name,
        })
        return frame


class CMAPSSBenchmarkAdapter(ExternalDatasetAdapter):
    """
    Adapter for NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) datasets.
    Maps turbofan run-to-failure cycles into canonical schema for RUL methodology benchmarking.
    """

    CMAPSS_COLUMNS = [
        "unit_number",
        "time_cycles",
        "op_setting_1",
        "op_setting_2",
        "op_setting_3",
    ] + [f"s_{i}" for i in range(1, 22)]

    def __init__(
        self,
        subset_id: str = "FD001",
        cycle_dt_s: float = 3600.0,
    ):
        self.subset_id = subset_id
        self.cycle_dt_s = cycle_dt_s

    def adapt(
        self,
        source_data: Union[str, Path, pd.DataFrame],
        **kwargs,
    ) -> CanonicalTelemetryFrame:
        if isinstance(source_data, (str, Path)):
            df = pd.read_csv(
                source_data,
                sep=r"\s+",
                header=None,
                names=self.CMAPSS_COLUMNS,
            )
        elif isinstance(source_data, pd.DataFrame):
            df = source_data.copy()
            if len(df.columns) == len(self.CMAPSS_COLUMNS):
                df.columns = self.CMAPSS_COLUMNS
        else:
            raise TypeError(f"Unsupported source_data type: {type(source_data)}")

        # Construct canonical representation
        # Map unit_number -> engine_id, time_cycles -> timestamp
        canonical_df = pd.DataFrame()
        canonical_df["engine_id"] = "CMAPSS_UNIT_" + df["unit_number"].astype(str)
        canonical_df["mission_id"] = f"CMAPSS_{self.subset_id}"
        canonical_df["timestamp"] = df["time_cycles"].astype(float) * self.cycle_dt_s

        # Map relevant operational settings & proxy sensors
        # s_2: LPC outlet temp -> proxy cht
        # s_3: HPC outlet temp -> proxy egt
        # s_4: LPT outlet temp -> proxy oil_temp
        # s_7: HPC outlet pressure -> proxy oil_pressure
        # s_9: Core speed -> proxy rpm
        # s_11: Static pressure at HPC outlet -> proxy load
        # s_12: Ratio of fuel flow to Ps30 -> proxy fuel_flow
        if "s_9" in df.columns:
            canonical_df["rpm"] = df["s_9"].astype(float)
        if "s_2" in df.columns:
            canonical_df["cht"] = df["s_2"].astype(float)
        if "s_3" in df.columns:
            canonical_df["egt"] = df["s_3"].astype(float)
        if "s_4" in df.columns:
            canonical_df["oil_temp"] = df["s_4"].astype(float)
        if "s_7" in df.columns:
            canonical_df["oil_pressure"] = df["s_7"].astype(float)
        if "s_12" in df.columns:
            canonical_df["fuel_flow"] = df["s_12"].astype(float)

        canonical_df["source"] = f"nasa_cmapss_{self.subset_id.lower()}"
        canonical_df["source_type"] = "external_benchmark"

        frame = TelemetryIngestor.ingest(
            canonical_df,
            default_source=f"nasa_cmapss_{self.subset_id.lower()}",
            default_source_type="external_benchmark",
            strict=False,
        )

        frame.metadata.update({
            "substitute_role": "turbofan_rul_prognostic_methodology",
            "disclaimer": "RUL methodology benchmark, NOT aero-piston telemetry.",
            "subset": self.subset_id,
        })
        return frame
