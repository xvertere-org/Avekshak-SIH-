"""
Boundary, Sequence, Clock, and Quality Validator for SIH26054 Digital Twin.

Phase 9 Ingestion Guard:
1. Defensive Boundary Validation: Malformed external data, wrong types, NaN, and +/-Inf
   are stopped at the boundary before reaching engine physics calculations.
2. Clock & Latency Model: Evaluates source_timestamp vs ingest_timestamp. Negative latency
   is explicitly flagged as CLOCK_SKEW without conflating clock diagnostics with engine physics.
3. Sequence Number Auditing: Detects duplicate, skipped, and out-of-order sequence numbers.
   Wraparound is strictly evaluated only if sequence_modulus is explicitly configured.
4. Non-Destructive Outlier Policy: Raw observations remain immutable. Boundary sanitization
   records failure reasons without silently clipping bad data into plausible values.
5. Limit Provenance: Distinguishes physical/thermodynamic impossibility from instrument bounds
   and operational warning envelopes. No sensor limits are invented.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, Tuple, List, Union
import math

from digital_twin.quality import DataQualityStatus, PHYSICAL_INSTRUMENT_LIMITS
from digital_twin.state import QuantityStatus
from telemetry.canonical import (
    CanonicalMeasurement,
    CanonicalTelemetryPacket,
    SourceType,
    CalibrationMetadata,
)
from telemetry.units import convert_unit, CANONICAL_UNITS, UnitConversionError


class ClockStatus(str, Enum):
    """Timestamp and clock relationship evaluation."""
    VALID = "VALID"
    CLOCK_SKEW = "CLOCK_SKEW"                  # Ingest timestamp precedes source timestamp
    CLOCK_UNAVAILABLE = "CLOCK_UNAVAILABLE"    # Source or ingest timestamp missing
    EXCESSIVE_LATENCY = "EXCESSIVE_LATENCY"    # Latency exceeds latency threshold


class SequenceStatus(str, Enum):
    """Sequence number validation classification."""
    NORMAL = "NORMAL"
    DUPLICATE = "DUPLICATE"
    SKIPPED = "SKIPPED"
    OUT_OF_ORDER = "OUT_OF_ORDER"
    WRAPAROUND = "WRAPAROUND"
    CONFIGURATION_UNKNOWN = "CONFIGURATION_UNKNOWN"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass
class ValidationReport:
    """Detailed audit report from the boundary validation layer."""
    is_acceptable: bool
    clock_status: ClockStatus
    sequence_status: SequenceStatus
    latency: Optional[float] = None
    dropped_channels: List[str] = field(default_factory=list)
    quarantined_reasons: Dict[str, str] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_acceptable": self.is_acceptable,
            "clock_status": self.clock_status.value,
            "sequence_status": self.sequence_status.value,
            "latency": self.latency,
            "dropped_channels": self.dropped_channels,
            "quarantined_reasons": self.quarantined_reasons,
            "notes": self.notes,
        }


class BoundaryValidator:
    """
    Defensive input gatekeeper for raw and semi-structured external telemetry feeds.
    Maintains temporal sequence state, clock skew monitoring, and sequence number auditing.
    """

    def __init__(
        self,
        tau_stale: float = 2.0,
        dt_horizon: float = 1.0,
        max_latency_sec: float = 5.0,
        sequence_modulus: Optional[int] = None,
        custom_instrument_limits: Optional[Dict[str, Tuple[float, float]]] = None,
        strict_channel_check: bool = False,
    ):
        self.tau_stale = float(tau_stale)
        self.dt_horizon = float(dt_horizon)
        self.max_latency_sec = float(max_latency_sec)
        self.sequence_modulus = sequence_modulus
        self.instrument_limits = custom_instrument_limits or dict(PHYSICAL_INSTRUMENT_LIMITS)
        self.strict_channel_check = strict_channel_check

        # State tracking
        self.last_source_timestamp: Optional[float] = None
        self.last_ingest_timestamp: Optional[float] = None
        self.last_sequence_num: Optional[int] = None
        self.packet_count: int = 0

    def reset(self) -> None:
        """Reset sequence and temporal tracking state."""
        self.last_source_timestamp = None
        self.last_ingest_timestamp = None
        self.last_sequence_num = None
        self.packet_count = 0

    def validate_clock(
        self,
        source_ts: Optional[float],
        ingest_ts: Optional[float],
        ref_clock_ts: Optional[float] = None,
    ) -> Tuple[ClockStatus, Optional[float], Optional[str]]:
        """
        Evaluate timestamp relationships and clock skew.
        Negative latency (ingest < source) indicates clock skew / unsynchronized clocks.
        """
        if source_ts is None or ingest_ts is None:
            return ClockStatus.CLOCK_UNAVAILABLE, None, "Missing source or ingestion timestamp"

        if math.isnan(source_ts) or math.isnan(ingest_ts) or math.isinf(source_ts) or math.isinf(ingest_ts):
            return ClockStatus.CLOCK_UNAVAILABLE, None, "Non-finite timestamp"

        latency = float(ingest_ts) - float(source_ts)

        # Detect clock skew
        if latency < -0.05: # Allow small 50ms tolerance for network time jitter if applicable
            return ClockStatus.CLOCK_SKEW, latency, f"Negative latency detected: {latency:.4f}s (clock skew)"

        if latency > self.max_latency_sec:
            return ClockStatus.EXCESSIVE_LATENCY, latency, f"Latency {latency:.4f}s exceeds threshold {self.max_latency_sec}s"

        return ClockStatus.VALID, latency, None

    def validate_sequence(self, seq_num: Optional[int]) -> Tuple[SequenceStatus, Optional[str]]:
        """
        Audit sequence numbers.
        Wraparound is strictly evaluated only if sequence_modulus is configured.
        """
        if seq_num is None:
            return SequenceStatus.UNAVAILABLE, None

        if not isinstance(seq_num, int):
            return SequenceStatus.UNAVAILABLE, f"Sequence number '{seq_num}' is not an integer"

        if self.last_sequence_num is None:
            self.last_sequence_num = seq_num
            return SequenceStatus.NORMAL, None

        diff = seq_num - self.last_sequence_num

        if diff == 1:
            self.last_sequence_num = seq_num
            return SequenceStatus.NORMAL, None

        if diff == 0:
            return SequenceStatus.DUPLICATE, f"Duplicate sequence number: {seq_num}"

        if diff > 1:
            note = f"Skipped sequence jump: {self.last_sequence_num} -> {seq_num} (gap: {diff - 1})"
            self.last_sequence_num = seq_num
            return SequenceStatus.SKIPPED, note

        # diff < 0 (Sequence decreased)
        if self.sequence_modulus is not None:
            # Check if this matches a legitimate modulo wraparound (e.g. from 65535 to 0)
            expected_wrap_diff = (seq_num + self.sequence_modulus) - self.last_sequence_num
            if expected_wrap_diff == 1:
                self.last_sequence_num = seq_num
                return SequenceStatus.WRAPAROUND, f"Sequence counter wrapped around modulus {self.sequence_modulus}"
            elif expected_wrap_diff > 1:
                self.last_sequence_num = seq_num
                return SequenceStatus.SKIPPED, f"Wrapped sequence jump across modulus {self.sequence_modulus}"

        # If modulus is not configured, do NOT infer wraparound
        self.last_sequence_num = seq_num
        return SequenceStatus.CONFIGURATION_UNKNOWN, f"Sequence decreased ({diff}) with unknown counter modulus"

    def validate_channel_value(
        self,
        channel_name: str,
        raw_val: Any,
        raw_unit: Optional[str] = None,
        target_canonical_unit: Optional[str] = None,
    ) -> Tuple[Optional[float], DataQualityStatus, str]:
        """
        Validate single channel raw observation:
        - Check for None / missing.
        - Check for parseable numeric type.
        - Check for NaN / +/-Inf.
        - Perform unit conversion if units are specified.
        - Validate against physical instrument limits without modifying raw value.
        """
        if raw_val is None:
            return None, DataQualityStatus.MISSING, "Missing value"

        # Attempt float conversion
        try:
            val_float = float(raw_val)
        except (ValueError, TypeError):
            return None, DataQualityStatus.INVALID, f"Non-numeric value: '{raw_val}'"

        # Non-finite check
        if math.isnan(val_float) or math.isinf(val_float):
            return None, DataQualityStatus.NON_FINITE, f"Non-finite floating point: {val_float}"

        # Unit conversion
        conv_val = val_float
        if raw_unit is not None and target_canonical_unit is not None and raw_unit != target_canonical_unit:
            try:
                conv_val = convert_unit(val_float, from_unit=raw_unit, to_unit=target_canonical_unit)
            except UnitConversionError as e:
                return None, DataQualityStatus.INVALID, f"Unit conversion failed: {str(e)}"

        # Absolute thermodynamic impossibilities
        if channel_name in ("rpm", "engine_rpm") and conv_val < 0.0:
            return conv_val, DataQualityStatus.OUT_OF_RANGE, f"Negative RPM ({conv_val}) is physically impossible"

        if channel_name in ("oil_pressure", "map_bar", "map") and conv_val < 0.0:
            return conv_val, DataQualityStatus.OUT_OF_RANGE, f"Negative absolute/gage pressure ({conv_val}) is impossible"

        if "temp" in channel_name or channel_name in ("cht", "egt") or "cht_" in channel_name or "egt_" in channel_name:
            if conv_val < -273.15:
                return conv_val, DataQualityStatus.OUT_OF_RANGE, f"Temperature below absolute zero ({conv_val} degC)"

        # Check against physical instrument limits (if known)
        lim = self.instrument_limits.get(channel_name)
        if lim is not None:
            lo, hi = lim
            if conv_val < lo or conv_val > hi:
                return conv_val, DataQualityStatus.OUT_OF_RANGE, f"Value {conv_val} outside instrument bounds [{lo}, {hi}]"

        return conv_val, DataQualityStatus.VALID, ""

    def validate_packet(
        self,
        raw_dict: Dict[str, Any],
        ingest_time: Optional[float] = None,
        source_id: str = "feed_01",
        source_type: SourceType = SourceType.SIMULATOR,
    ) -> Tuple[Optional[CanonicalTelemetryPacket], ValidationReport]:
        """
        Validate an external telemetry record dictionary and produce a CanonicalTelemetryPacket.
        Preserves raw values and quarantined reasons.
        """
        report = ValidationReport(
            is_acceptable=True,
            clock_status=ClockStatus.VALID,
            sequence_status=SequenceStatus.NORMAL,
        )

        # 1. Extract and validate timestamp
        raw_ts = raw_dict.get("timestamp")
        if raw_ts is None:
            report.is_acceptable = False
            report.quarantined_reasons["timestamp"] = "Mandatory timestamp field is missing"
            return None, report

        try:
            source_ts = float(raw_ts)
        except (ValueError, TypeError):
            report.is_acceptable = False
            report.quarantined_reasons["timestamp"] = f"Invalid timestamp format: '{raw_ts}'"
            return None, report

        if math.isnan(source_ts) or math.isinf(source_ts):
            report.is_acceptable = False
            report.quarantined_reasons["timestamp"] = f"Non-finite timestamp: {source_ts}"
            return None, report

        # 2. Clock & Latency Evaluation
        actual_ingest_time = ingest_time if ingest_time is not None else source_ts
        clock_status, lat, clock_note = self.validate_clock(source_ts, actual_ingest_time)
        report.clock_status = clock_status
        report.latency = lat
        if clock_note:
            report.notes.append(clock_note)

        # If negative latency (clock skew) is severe, note it in report
        if clock_status == ClockStatus.CLOCK_SKEW:
            report.notes.append("CLOCK_SKEW_DETECTED: Ingestion timestamp precedes source timestamp")

        # 3. Sequence Number Validation
        raw_seq = raw_dict.get("sequence", raw_dict.get("sequence_num"))
        seq_status, seq_note = self.validate_sequence(raw_seq)
        report.sequence_status = seq_status
        if seq_note:
            report.notes.append(seq_note)

        # 4. Temporal Progression Check
        if self.last_source_timestamp is not None:
            dt = source_ts - self.last_source_timestamp
            if dt == 0.0:
                report.notes.append("DUPLICATE_SOURCE_TIMESTAMP")
            elif dt < 0.0:
                report.notes.append("NON_MONOTONIC_SOURCE_TIMESTAMP")
        self.last_source_timestamp = source_ts
        self.last_ingest_timestamp = actual_ingest_time

        # 5. Extract and Validate Channels
        measurements: Dict[str, CanonicalMeasurement] = {}

        # Channels can be passed either as a nested "channels" dict or top-level keys
        channel_data = raw_dict.get("channels", raw_dict)

        for ch_key, ch_info in channel_data.items():
            if ch_key in ("timestamp", "sequence", "sequence_num", "source", "source_type", "channels", "metadata", "engine_id", "mission_id", "mission_phase"):
                continue

            # Support both scalar values and structured {value: ..., unit: ...} dicts
            raw_val = ch_info.get("value") if isinstance(ch_info, dict) else ch_info
            raw_unit = ch_info.get("unit") if isinstance(ch_info, dict) else None
            sensor_id = ch_info.get("sensor_id") if isinstance(ch_info, dict) else None
            cal_info = ch_info.get("calibration") if isinstance(ch_info, dict) else None

            # Determine expected canonical unit
            # Default lookup based on channel standard
            target_unit = None
            if ch_key in ("rpm", "engine_rpm", "propeller_rpm"):
                target_unit = "RPM"
            elif ch_key in ("map", "map_bar", "oil_pressure"):
                target_unit = "bar"
            elif "temp" in ch_key or ch_key in ("cht", "egt") or "cht_" in ch_key or "egt_" in ch_key:
                target_unit = "degC"
            elif ch_key == "fuel_flow":
                target_unit = "L/h"
            elif ch_key == "vibration":
                target_unit = "g"
            elif ch_key == "altitude":
                target_unit = "m"
            elif ch_key in ("throttle", "load"):
                target_unit = "%"

            # Parse calibration metadata if provided
            cal_meta = None
            if isinstance(cal_info, dict):
                cal_meta = CalibrationMetadata(
                    scale=float(cal_info.get("scale", 1.0)),
                    offset=float(cal_info.get("offset", 0.0)),
                    calibration_version=str(cal_info.get("version", "1.0")),
                    calibration_source=str(cal_info.get("source", "external")),
                    is_active=bool(cal_info.get("is_active", False)),
                )
                if cal_meta.is_active and isinstance(raw_val, (int, float)):
                    raw_val = cal_meta.apply(raw_val)

            canon_val, quality, note = self.validate_channel_value(
                channel_name=ch_key,
                raw_val=raw_val,
                raw_unit=raw_unit,
                target_canonical_unit=target_unit,
            )

            if quality != DataQualityStatus.VALID:
                report.dropped_channels.append(ch_key)
                report.quarantined_reasons[ch_key] = note

            measurements[ch_key] = CanonicalMeasurement(
                channel_name=ch_key,
                value=canon_val if quality == DataQualityStatus.VALID else None,
                unit=target_unit or (raw_unit or "unknown"),
                raw_value=raw_val,
                raw_unit=raw_unit,
                timestamp=source_ts,
                source_timestamp=source_ts,
                ingest_timestamp=actual_ingest_time,
                latency=lat,
                sequence_num=raw_seq if isinstance(raw_seq, int) else None,
                source_id=source_id,
                source_type=source_type,
                quality=quality,
                status=QuantityStatus.MEASURED if quality == DataQualityStatus.VALID else QuantityStatus.UNAVAILABLE,
                sensor_id=sensor_id,
                calibration=cal_meta,
                notes=note,
            )

        self.packet_count += 1

        packet = CanonicalTelemetryPacket(
            timestamp=source_ts,
            source_timestamp=source_ts,
            ingest_timestamp=actual_ingest_time,
            latency=lat,
            sequence_num=raw_seq if isinstance(raw_seq, int) else None,
            source_id=source_id,
            source_type=source_type,
            schema_version="1.0",
            engine_id=raw_dict.get("engine_id", "ENGINE_UAV_01"),
            mission_id=raw_dict.get("mission_id", "MISSION_01"),
            mission_phase=raw_dict.get("mission_phase", "CRUISE"),
            measurements=measurements,
            metadata=dict(raw_dict.get("metadata", {})),
        )

        return packet, report
