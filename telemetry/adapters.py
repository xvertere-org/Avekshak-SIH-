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
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import json
import pandas as pd
import numpy as np

from telemetry.ingestion import CanonicalTelemetryFrame, TelemetryIngestor
from telemetry.canonical import (
    CanonicalTelemetryPacket,
    CanonicalMeasurement,
    SourceType,
    CalibrationMetadata,
)
from telemetry.validator import BoundaryValidator


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


class JSONReplayAdapter:
    """
    Ingestion adapter for external versioned JSON / NDJSON telemetry interchange formats.
    Enforces boundary validation and explicit provenance (SourceType.REPLAY / SIMULATOR).
    """

    def __init__(
        self,
        schema_version: str = "1.0",
        source_id: str = "json_external_replay",
        source_type: SourceType = SourceType.REPLAY,
        validator: Optional[BoundaryValidator] = None,
    ):
        self.schema_version = schema_version
        self.source_id = source_id
        self.source_type = source_type
        self.validator = validator or BoundaryValidator()

    def parse_record(self, raw_record: Dict[str, Any], ingest_time: Optional[float] = None) -> Optional[CanonicalTelemetryPacket]:
        """Parse a single JSON telemetry record."""
        packet, report = self.validator.validate_packet(
            raw_record,
            ingest_time=ingest_time,
            source_id=self.source_id,
            source_type=self.source_type,
        )
        if packet and report.is_acceptable:
            return packet
        return None

    def adapt_file(self, file_path: Union[str, Path]) -> List[CanonicalTelemetryPacket]:
        """Parse an entire JSON or NDJSON file into a list of CanonicalTelemetryPackets."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")

        packets: List[CanonicalTelemetryPacket] = []
        with open(p, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Check if NDJSON (lines) or JSON (array/object)
        if content.startswith("["):
            data = json.loads(content)
            for item in data:
                if isinstance(item, dict):
                    pkt = self.parse_record(item)
                    if pkt:
                        packets.append(pkt)
        elif content.startswith("{") and "\n{" not in content:
            data = json.loads(content)
            if "records" in data:
                for item in data["records"]:
                    pkt = self.parse_record(item)
                    if pkt:
                        packets.append(pkt)
            else:
                pkt = self.parse_record(data)
                if pkt:
                    packets.append(pkt)
        else:
            # Assume NDJSON
            for line in content.splitlines():
                l = line.strip()
                if l and not l.startswith("#"):
                    item = json.loads(l)
                    if isinstance(item, dict):
                        pkt = self.parse_record(item)
                        if pkt:
                            packets.append(pkt)

        return packets


class CSVReplayAdapter(ExternalDatasetAdapter):
    """
    Adapter for converting external CSV telemetry with explicit column, unit,
    and sensor ID mappings into canonical representations.
    """

    def __init__(
        self,
        column_mapping: Optional[Dict[str, str]] = None,
        unit_mapping: Optional[Dict[str, str]] = None,
        sensor_id_mapping: Optional[Dict[str, str]] = None,
        calibration_configs: Optional[Dict[str, CalibrationMetadata]] = None,
        source_id: str = "csv_external_replay",
        source_type: SourceType = SourceType.REPLAY,
        default_engine_id: str = "ENGINE_UAV_01",
        default_mission_id: str = "MISSION_EXT_01",
        validator: Optional[BoundaryValidator] = None,
    ):
        self.column_mapping = column_mapping or {}
        self.unit_mapping = unit_mapping or {}
        self.sensor_id_mapping = sensor_id_mapping or {}
        self.calibration_configs = calibration_configs or {}
        self.source_id = source_id
        self.source_type = source_type
        self.default_engine_id = default_engine_id
        self.default_mission_id = default_mission_id
        self.validator = validator or BoundaryValidator()

    def adapt_packets(self, source_data: Union[str, Path, pd.DataFrame]) -> List[CanonicalTelemetryPacket]:
        """Convert CSV data into a list of validated CanonicalTelemetryPackets."""
        if isinstance(source_data, (str, Path)):
            df = pd.read_csv(source_data)
        elif isinstance(source_data, pd.DataFrame):
            df = source_data.copy()
        else:
            raise TypeError(f"Unsupported source_data type: {type(source_data)}")

        packets: List[CanonicalTelemetryPacket] = []

        # Ensure timestamp
        ts_col = self.column_mapping.get("timestamp", "timestamp")
        if ts_col not in df.columns:
            df[ts_col] = np.arange(len(df), dtype=float)

        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            raw_ts = row_dict.get(ts_col, idx)

            channel_dict: Dict[str, Any] = {}
            for col_name, val in row_dict.items():
                if col_name == ts_col:
                    continue

                # Map column name to canonical channel name
                canon_ch = self.column_mapping.get(col_name, col_name)
                raw_unit = self.unit_mapping.get(col_name, self.unit_mapping.get(canon_ch))
                sensor_id = self.sensor_id_mapping.get(col_name)
                cal_meta = self.calibration_configs.get(canon_ch, self.calibration_configs.get(col_name))

                ch_payload = {
                    "value": val,
                    "unit": raw_unit,
                    "sensor_id": sensor_id,
                }
                if cal_meta:
                    ch_payload["calibration"] = cal_meta.to_dict()

                channel_dict[canon_ch] = ch_payload

            record_payload = {
                "timestamp": raw_ts,
                "engine_id": row_dict.get("engine_id", self.default_engine_id),
                "mission_id": row_dict.get("mission_id", self.default_mission_id),
                "mission_phase": row_dict.get("mission_phase", "CRUISE"),
                "channels": channel_dict,
            }

            pkt, report = self.validator.validate_packet(
                record_payload,
                source_id=self.source_id,
                source_type=self.source_type,
            )
            if pkt and report.is_acceptable:
                packets.append(pkt)

        return packets

    def adapt(
        self,
        source_data: Union[str, Path, pd.DataFrame],
        **kwargs,
    ) -> CanonicalTelemetryFrame:
        """Standard adapt method returning CanonicalTelemetryFrame."""
        packets = self.adapt_packets(source_data)
        records = [p.to_telemetry_record() for p in packets]
        df = pd.DataFrame([r.to_dict() for r in records])
        return TelemetryIngestor.ingest(
            df,
            default_source=self.source_id,
            default_source_type=self.source_type.value.lower(),
            default_engine_id=self.default_engine_id,
            default_mission_id=self.default_mission_id,
            strict=False,
        )

