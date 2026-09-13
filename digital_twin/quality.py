"""
Telemetry Data Quality and Sensor Validity Layer for SIH26054.

Provides deterministic data quality validation, physical limit checks, and
strict temporal sequence auditing for incoming aero-piston telemetry.

Key Architectural Guarantees:
1. DataQualityStatus enum is strictly distinct from state QuantityStatus and twin health.
2. Physical abnormality != sensor invalidity (e.g. elevated CHT within instrument bounds is VALID).
3. Temporal validation cleanly separates DUPLICATE_TIMESTAMP, NON_MONOTONIC_TIMESTAMP,
   and FUTURE_TIMESTAMP from physical value OUT_OF_RANGE.
4. Identical values with advancing timestamps are recognized as valid steady-state telemetry.
5. Telemetry packet dropouts exceeding tau_stale trigger STALE status.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, List, Tuple
import math


class DataQualityStatus(str, Enum):
    """Data quality and validity status of a sensor channel or telemetry frame."""
    VALID = "VALID"                                      # Measurement within physical instrument limits and healthy sequence
    MISSING = "MISSING"                                  # Channel missing from packet or explicitly None
    NON_FINITE = "NON_FINITE"                            # NaN, +Inf, or -Inf value
    OUT_OF_RANGE = "OUT_OF_RANGE"                        # Value violates instrument physical limits
    STALE = "STALE"                                      # Telemetry packet interval exceeds staleness timeout
    DUPLICATE_TIMESTAMP = "DUPLICATE_TIMESTAMP"          # Packet timestamp equals preceding packet (rejected)
    NON_MONOTONIC_TIMESTAMP = "NON_MONOTONIC_TIMESTAMP"  # Packet timestamp precedes preceding packet (rejected)
    FUTURE_TIMESTAMP = "FUTURE_TIMESTAMP"                # Packet timestamp exceeds allowed forward horizon (rejected)
    INVALID = "INVALID"                                  # Malformed structure or type conversion failure
    UNAVAILABLE = "UNAVAILABLE"                          # Sensor channel physically uninstrumented / unmodeled


@dataclass
class ChannelQuality:
    """Quality and validity audit for an individual sensor channel."""
    channel_name: str
    raw_value: Any
    validated_value: Optional[float]
    status: DataQualityStatus
    is_valid: bool
    timestamp: float
    anomaly_note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel_name": self.channel_name,
            "raw_value": self.raw_value,
            "validated_value": round(self.validated_value, 4) if self.validated_value is not None else None,
            "status": self.status.value,
            "is_valid": self.is_valid,
            "timestamp": self.timestamp,
            "anomaly_note": self.anomaly_note,
        }


@dataclass
class TelemetryQualityReport:
    """Comprehensive data quality report for an incoming telemetry packet."""
    timestamp: float
    overall_valid: bool
    temporal_status: DataQualityStatus
    channel_reports: Dict[str, ChannelQuality] = field(default_factory=dict)
    valid_channels: List[str] = field(default_factory=list)
    invalid_channels: List[str] = field(default_factory=list)

    @property
    def num_valid(self) -> int:
        return len(self.valid_channels)

    @property
    def num_expected(self) -> int:
        return len(self.channel_reports)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "overall_valid": self.overall_valid,
            "temporal_status": self.temporal_status.value,
            "num_valid": self.num_valid,
            "num_expected": self.num_expected,
            "valid_channels": self.valid_channels,
            "invalid_channels": self.invalid_channels,
            "channel_reports": {k: v.to_dict() for k, v in self.channel_reports.items()},
        }


# Authoritative physical sensor instrument limits (physical measurement range)
# Violations here are OUT_OF_RANGE. Plausible physical abnormalities within these bounds are VALID.
PHYSICAL_INSTRUMENT_LIMITS: Dict[str, Tuple[float, float]] = {
    "rpm": (0.0, 7500.0),                  # Rotax 914 max speed limit ~5800 RPM; 7500 is physical sensor bound
    "cht": (-50.0, 300.0),                # Cylinder head temp thermocouple bounds [°C]
    "cht_cyl1": (-50.0, 300.0),
    "cht_cyl2": (-50.0, 300.0),
    "cht_cyl3": (-50.0, 300.0),
    "cht_cyl4": (-50.0, 300.0),
    "egt": (0.0, 1200.0),                 # Exhaust gas temp thermocouple bounds [°C]
    "egt_cyl1": (0.0, 1200.0),
    "egt_cyl2": (0.0, 1200.0),
    "egt_cyl3": (0.0, 1200.0),
    "egt_cyl4": (0.0, 1200.0),
    "oil_pressure": (0.0, 15.0),           # Oil pressure transducer bounds [bar]
    "oil_temp": (-50.0, 200.0),           # Oil temperature sensor bounds [°C]
    "coolant_temp": (-50.0, 200.0),       # Coolant temperature sensor bounds [°C]
    "fuel_flow": (0.0, 100.0),            # Fuel flow meter bounds [L/h]
    "vibration": (0.0, 50.0),             # Accelerometer full-scale range [g]
    "map": (0.1, 3.0),                    # Manifold pressure sensor bounds [bar]
    "throttle": (0.0, 100.0),             # Throttle position sensor [%]
    "altitude": (-1000.0, 15000.0),       # Barometric / GPS altitude [m]
    "ambient_temp": (-70.0, 70.0),        # Ambient air temp [°C]
}


class TelemetryQualityValidator:
    """
    Deterministic validator for telemetry packets.
    Maintains temporal sequence state and validates channels against instrument bounds.
    """

    def __init__(
        self,
        tau_stale: float = 2.0,
        dt_horizon: float = 1.0,
        limits: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self.tau_stale = float(tau_stale)
        self.dt_horizon = float(dt_horizon)
        self.limits = limits or PHYSICAL_INSTRUMENT_LIMITS

        self.last_timestamp: Optional[float] = None
        self.last_valid_timestamp: Optional[float] = None

    def reset(self) -> None:
        """Reset temporal tracking state."""
        self.last_timestamp = None
        self.last_valid_timestamp = None

    def validate(
        self,
        telemetry: Any,
        sim_time: Optional[float] = None,
    ) -> TelemetryQualityReport:
        """
        Validate incoming telemetry packet.

        Args:
            telemetry: Dict or object with telemetry channel attributes / keys.
            sim_time: Current simulation or reference clock time for horizon check.

        Returns:
            TelemetryQualityReport detailing temporal and per-channel quality.
        """
        # Extract packet timestamp
        if hasattr(telemetry, "timestamp"):
            t_pkt = float(telemetry.timestamp)
        elif isinstance(telemetry, dict) and "timestamp" in telemetry:
            t_pkt = float(telemetry["timestamp"])
        else:
            t_pkt = 0.0

        current_reference_time = sim_time if sim_time is not None else (
            self.last_timestamp if self.last_timestamp is not None else t_pkt
        )

        # 1. Temporal Sequence Check
        temporal_status = DataQualityStatus.VALID
        temporal_error_note = ""

        if self.last_timestamp is not None:
            if t_pkt == self.last_timestamp:
                temporal_status = DataQualityStatus.DUPLICATE_TIMESTAMP
                temporal_error_note = f"Duplicate timestamp received: t = {t_pkt:.4f} s"
            elif t_pkt < self.last_timestamp:
                temporal_status = DataQualityStatus.NON_MONOTONIC_TIMESTAMP
                temporal_error_note = (
                    f"Non-monotonic timestamp jump backward: t = {t_pkt:.4f} s < "
                    f"last = {self.last_timestamp:.4f} s"
                )
            elif (t_pkt - self.last_timestamp) > self.tau_stale:
                temporal_status = DataQualityStatus.STALE
                temporal_error_note = (
                    f"Observation interval {t_pkt - self.last_timestamp:.4f} s exceeds "
                    f"stale timeout {self.tau_stale:.4f} s"
                )

        # Future horizon check: applicable only if sim_time is provided and a baseline timestamp exists
        if sim_time is not None and self.last_timestamp is not None and t_pkt > (sim_time + self.dt_horizon):
            temporal_status = DataQualityStatus.FUTURE_TIMESTAMP
            temporal_error_note = (
                f"Timestamp t = {t_pkt:.4f} s exceeds allowed horizon "
                f"sim_time + {self.dt_horizon:.4f} s = {sim_time + self.dt_horizon:.4f} s"
            )

        # 2. Per-Channel Validation
        channel_reports: Dict[str, ChannelQuality] = {}
        valid_channels: List[str] = []
        invalid_channels: List[str] = []

        # Standard baseline observation channels
        channels_to_evaluate = [
            "rpm", "cht", "egt", "oil_pressure", "oil_temp",
            "fuel_flow", "vibration",
        ]
        # Include per-cylinder channels only if explicitly provided in packet
        for i in range(1, 5):
            for prefix in ["cht_cyl", "egt_cyl"]:
                key = f"{prefix}{i}"
                has_key = False
                if isinstance(telemetry, dict) and key in telemetry and telemetry[key] is not None:
                    has_key = True
                elif hasattr(telemetry, key) and getattr(telemetry, key, None) is not None:
                    has_key = True
                if has_key and key not in channels_to_evaluate:
                    channels_to_evaluate.append(key)

        # If temporal status is rejected (duplicate, non-monotonic, future), reject whole frame
        is_packet_rejected = temporal_status in (
            DataQualityStatus.DUPLICATE_TIMESTAMP,
            DataQualityStatus.NON_MONOTONIC_TIMESTAMP,
            DataQualityStatus.FUTURE_TIMESTAMP,
        )

        for ch in channels_to_evaluate:
            raw_val = None
            if hasattr(telemetry, ch):
                raw_val = getattr(telemetry, ch)
            elif isinstance(telemetry, dict) and ch in telemetry:
                raw_val = telemetry[ch]

            # Validate individual channel
            ch_quality = self._validate_channel(
                channel_name=ch,
                raw_val=raw_val,
                timestamp=t_pkt,
                packet_temporal_status=temporal_status,
                packet_rejected=is_packet_rejected,
            )
            channel_reports[ch] = ch_quality

            if ch_quality.is_valid:
                valid_channels.append(ch)
            else:
                invalid_channels.append(ch)

        # Overall validity requires: not rejected temporally and at least one valid channel
        overall_valid = (not is_packet_rejected) and (len(valid_channels) > 0)

        # Update temporal tracking only if timestamp was non-rejected or advancing
        if not is_packet_rejected:
            self.last_timestamp = t_pkt
            if overall_valid:
                self.last_valid_timestamp = t_pkt

        return TelemetryQualityReport(
            timestamp=t_pkt,
            overall_valid=overall_valid,
            temporal_status=temporal_status,
            channel_reports=channel_reports,
            valid_channels=valid_channels,
            invalid_channels=invalid_channels,
        )

    def _validate_channel(
        self,
        channel_name: str,
        raw_val: Any,
        timestamp: float,
        packet_temporal_status: DataQualityStatus,
        packet_rejected: bool,
    ) -> ChannelQuality:
        """Validate an individual channel value."""
        if packet_rejected:
            return ChannelQuality(
                channel_name=channel_name,
                raw_value=raw_val,
                validated_value=None,
                status=packet_temporal_status,
                is_valid=False,
                timestamp=timestamp,
                anomaly_note=f"Packet rejected due to temporal anomaly: {packet_temporal_status.value}",
            )

        if raw_val is None:
            return ChannelQuality(
                channel_name=channel_name,
                raw_value=raw_val,
                validated_value=None,
                status=DataQualityStatus.MISSING,
                is_valid=False,
                timestamp=timestamp,
                anomaly_note="Channel value missing from packet",
            )

        try:
            val_float = float(raw_val)
        except (ValueError, TypeError):
            return ChannelQuality(
                channel_name=channel_name,
                raw_value=raw_val,
                validated_value=None,
                status=DataQualityStatus.INVALID,
                is_valid=False,
                timestamp=timestamp,
                anomaly_note="Non-numeric value conversion failure",
            )

        if math.isnan(val_float) or math.isinf(val_float):
            return ChannelQuality(
                channel_name=channel_name,
                raw_value=raw_val,
                validated_value=None,
                status=DataQualityStatus.NON_FINITE,
                is_valid=False,
                timestamp=timestamp,
                anomaly_note="Non-finite NaN or Inf detected",
            )

        # Check physical instrument limits
        # Find limit key (e.g. "cht_cyl1" falls back to "cht" if not explicitly listed)
        lim_key = channel_name
        if lim_key not in self.limits:
            if channel_name.startswith("cht_cyl"):
                lim_key = "cht"
            elif channel_name.startswith("egt_cyl"):
                lim_key = "egt"

        if lim_key in self.limits:
            low, high = self.limits[lim_key]
            if val_float < low or val_float > high:
                return ChannelQuality(
                    channel_name=channel_name,
                    raw_value=raw_val,
                    validated_value=None,
                    status=DataQualityStatus.OUT_OF_RANGE,
                    is_valid=False,
                    timestamp=timestamp,
                    anomaly_note=f"Value {val_float:.2f} violates instrument range [{low}, {high}]",
                )

        # Check if packet was stale
        if packet_temporal_status == DataQualityStatus.STALE:
            return ChannelQuality(
                channel_name=channel_name,
                raw_value=raw_val,
                validated_value=val_float,
                status=DataQualityStatus.STALE,
                is_valid=False,
                timestamp=timestamp,
                anomaly_note="Channel received after staleness timeout",
            )

        # Legitimate valid channel measurement
        return ChannelQuality(
            channel_name=channel_name,
            raw_value=raw_val,
            validated_value=val_float,
            status=DataQualityStatus.VALID,
            is_valid=True,
            timestamp=timestamp,
            anomaly_note="",
        )
