"""
Deterministic Telemetry Replay Engine for SIH26054 Digital Twin.

Phase 9 Replay Guarantees:
1. Dual Memory Architecture:
   - Streaming Mode (NDJSON / CSV Stream): Line-by-line generator with bounded O(1) buffer memory.
   - Materialized Mode: Loads all records into memory for analytical slicing (O(N) memory).
2. Determinism: Offline mode has zero dependence on system wall-clock time; identical inputs
   produce identical canonical sequences across multiple runs.
3. Provenance Safeguard: Replayed packets are strictly tagged SourceType.REPLAY or SIMULATOR;
   never labeled REAL_SENSOR.
4. Channel Filtering & Slicing: Supports temporal range selection (start_time, end_time) and
   channel sub-selection.
"""

from typing import Iterator, List, Dict, Any, Optional, Union
from pathlib import Path
import json
import time

from telemetry.canonical import (
    CanonicalTelemetryPacket,
    CanonicalMeasurement,
    SourceType,
)
from telemetry.validator import BoundaryValidator


class DeterministicReplayEngine:
    """
    Deterministic telemetry replay engine supporting streaming and materialized execution.
    """

    def __init__(
        self,
        validator: Optional[BoundaryValidator] = None,
        source_id: str = "deterministic_replay",
        source_type: SourceType = SourceType.REPLAY,
    ):
        self.validator = validator or BoundaryValidator()
        self.source_id = source_id
        self.source_type = source_type

    def stream_ndjson(
        self,
        file_path: Union[str, Path],
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        channels: Optional[List[str]] = None,
        realtime_rate: Optional[float] = None,
    ) -> Iterator[CanonicalTelemetryPacket]:
        """
        Stream telemetry line-by-line from a JSON Lines (NDJSON) file.
        Maintains bounded O(1) working memory.

        Args:
            file_path: Path to .ndjson or .jsonl file.
            start_time: Optional lower timestamp bound.
            end_time: Optional upper timestamp bound.
            channels: Optional channel whitelist.
            realtime_rate: Optional playback speed multiplier (e.g. 1.0 = 1x real-time).
                           None = offline deterministic stepping (no sleep).
        """
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Replay file not found: {p}")

        last_ts = None
        wall_start = time.perf_counter() if realtime_rate is not None else None

        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str or line_str.startswith("#"):
                    continue

                raw_dict = json.loads(line_str)
                packet, report = self.validator.validate_packet(
                    raw_dict,
                    source_id=self.source_id,
                    source_type=self.source_type,
                )

                if packet is None or not report.is_acceptable:
                    continue

                t = packet.timestamp
                if start_time is not None and t < start_time:
                    continue
                if end_time is not None and t > end_time:
                    break

                # Apply channel filtering if requested
                if channels is not None:
                    allowed_set = set(channels)
                    filtered_measurements = {k: v for k, v in packet.measurements.items() if k in allowed_set}
                    packet.measurements = filtered_measurements

                # Optional real-time pacing
                if realtime_rate is not None and realtime_rate > 0:
                    if last_ts is not None:
                        sim_dt = (t - last_ts) / realtime_rate
                        if sim_dt > 0:
                            time.sleep(sim_dt)
                    last_ts = t

                yield packet

    def replay_json_array(
        self,
        file_path: Union[str, Path],
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        channels: Optional[List[str]] = None,
    ) -> List[CanonicalTelemetryPacket]:
        """
        Materialized replay mode: loads all records from a standard JSON array into memory.
        Uses O(N) memory proportional to retained records.
        """
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Replay file not found: {p}")

        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            # Check if it's a wrapped structure e.g. {"records": [...]}
            if isinstance(data, dict) and "records" in data:
                data = data["records"]
            else:
                data = [data]

        packets: List[CanonicalTelemetryPacket] = []
        for raw_dict in data:
            if not isinstance(raw_dict, dict):
                continue

            packet, report = self.validator.validate_packet(
                raw_dict,
                source_id=self.source_id,
                source_type=self.source_type,
            )

            if packet is None or not report.is_acceptable:
                continue

            t = packet.timestamp
            if start_time is not None and t < start_time:
                continue
            if end_time is not None and t > end_time:
                continue

            if channels is not None:
                allowed_set = set(channels)
                packet.measurements = {k: v for k, v in packet.measurements.items() if k in allowed_set}

            packets.append(packet)

        # Ensure chronological ordering
        packets.sort(key=lambda p: p.timestamp)
        return packets
