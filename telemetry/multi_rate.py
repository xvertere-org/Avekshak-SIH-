"""
Multi-Rate Telemetry Synchronization and Sample Alignment for SIH26054.

Architectural Guarantees:
1. Multi-Rate Handling: Accommodates asynchronous channels arriving at distinct rates
   (e.g., high-rate RPM/vibration at 50-100 Hz, thermal at 5 Hz, environment at 1 Hz).
2. Epistemic Integrity for Held Values: A held-last-value sample is NEVER labeled MEASURED
   at the alignment timestamp. It is explicitly marked QuantityStatus.ESTIMATED with
   sub-status 'HELD_LAST_VALUE'.
3. Maximum Sample Age (tau_stale): Reuses Phase 3 tau_stale baseline (2.0s). Samples
   older than tau_stale evaluate strictly to STALE / UNAVAILABLE.
4. Interpolation Bounding: Interpolation is strictly bounded by max_interpolation_gap and
   forbidden across invalid samples or fault boundaries. Interpolated points are marked
   QuantityStatus.ESTIMATED with notes 'INTERPOLATED'.
5. Bounded Memory: Channel history buffers retain only a rolling window, preventing
   unbounded memory growth.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
import math

from digital_twin.quality import DataQualityStatus
from digital_twin.state import QuantityStatus
from telemetry.canonical import CanonicalMeasurement, CanonicalTelemetryPacket, SourceType


@dataclass
class ChannelSample:
    """Historical sample for a single channel in the multi-rate buffer."""
    timestamp: float
    value: Optional[float]
    unit: str
    quality: DataQualityStatus
    status: QuantityStatus
    source_id: str
    source_type: SourceType
    notes: str = ""


class MultiRateBuffer:
    """
    Rolling multi-rate channel buffer providing sample alignment and bounded interpolation.
    """

    def __init__(
        self,
        tau_stale: float = 2.0,
        max_interpolation_gap: float = 1.0,
        channel_stale_limits: Optional[Dict[str, float]] = None,
        max_buffer_len: int = 100,
    ):
        self.default_tau_stale = float(tau_stale)
        self.max_interpolation_gap = float(max_interpolation_gap)
        self.channel_stale_limits = channel_stale_limits or {
            "rpm": 0.5,
            "vibration": 0.5,
            "map_bar": 1.0,
            "oil_pressure": 2.0,
            "cht": 3.0,
            "egt": 2.0,
            "oil_temp": 5.0,
            "coolant_temp": 5.0,
            "altitude": 5.0,
            "ambient_temp": 5.0,
        }
        self.max_buffer_len = int(max_buffer_len)
        self._buffers: Dict[str, deque] = {}

    def reset(self) -> None:
        """Clear all buffered channel histories."""
        self._buffers.clear()

    def get_tau_stale(self, channel_name: str) -> float:
        """Get stale timeout for a specific channel."""
        return self.channel_stale_limits.get(channel_name, self.default_tau_stale)

    def push_measurement(self, measurement: CanonicalMeasurement) -> None:
        """Add a canonical measurement to the rolling channel buffer, including dropouts."""
        ch = measurement.channel_name
        if ch not in self._buffers:
            self._buffers[ch] = deque(maxlen=self.max_buffer_len)

        val = float(measurement.value) if measurement.value is not None and not math.isnan(measurement.value) else None

        sample = ChannelSample(
            timestamp=float(measurement.timestamp),
            value=val,
            unit=measurement.unit,
            quality=measurement.quality,
            status=measurement.status,
            source_id=measurement.source_id,
            source_type=measurement.source_type,
            notes=measurement.notes,
        )
        self._buffers[ch].append(sample)

    def push_packet(self, packet: CanonicalTelemetryPacket) -> None:
        """Push all measurements in a packet into the buffer."""
        for m in packet.measurements.values():
            self.push_measurement(m)

    def get_aligned_measurement(
        self,
        channel_name: str,
        target_timestamp: float,
        allow_interpolation: bool = False,
    ) -> CanonicalMeasurement:
        """
        Extract an aligned measurement for channel at target_timestamp.

        Alignment Policy:
        - If an exact sample exists at target_timestamp, return it directly.
        - If allow_interpolation is True and target_timestamp is bounded by two strictly VALID
          samples within max_interpolation_gap with NO intervening invalid/dropout samples,
          compute linear interpolation (marked ESTIMATED / INTERPOLATED).
        - Otherwise, use hold-last-value from latest preceding sample:
          - If preceding sample is not VALID or value is None: return unavailable/invalid.
          - If age <= tau_stale: return sample marked ESTIMATED with notes='HELD_LAST_VALUE'.
          - If age > tau_stale: return sample marked STALE / UNAVAILABLE.
        """
        tau_stale = self.get_tau_stale(channel_name)
        buf = self._buffers.get(channel_name)

        if not buf:
            return CanonicalMeasurement(
                channel_name=channel_name,
                value=None,
                unit="unknown",
                timestamp=target_timestamp,
                quality=DataQualityStatus.UNAVAILABLE,
                status=QuantityStatus.UNAVAILABLE,
                notes="NO_HISTORY_AVAILABLE",
            )

        # Look for exact match or immediate bounding samples
        exact_sample = None
        prev_sample = None
        next_sample = None

        for s in buf:
            if abs(s.timestamp - target_timestamp) < 1e-6:
                exact_sample = s
                break
            elif s.timestamp < target_timestamp:
                if prev_sample is None or s.timestamp > prev_sample.timestamp:
                    prev_sample = s
            elif s.timestamp > target_timestamp:
                if next_sample is None or s.timestamp < next_sample.timestamp:
                    next_sample = s

        # 1. Exact match
        if exact_sample is not None:
            return CanonicalMeasurement(
                channel_name=channel_name,
                value=exact_sample.value,
                unit=exact_sample.unit,
                timestamp=target_timestamp,
                source_timestamp=exact_sample.timestamp,
                quality=exact_sample.quality,
                status=exact_sample.status,
                source_id=exact_sample.source_id,
                source_type=exact_sample.source_type,
                notes=exact_sample.notes,
            )

        # 2. Linear Interpolation (if enabled and bounded)
        # Strictly forbidden across invalid samples, NaNs, dropouts, or gaps > max_interpolation_gap
        if allow_interpolation and prev_sample is not None and next_sample is not None:
            gap = next_sample.timestamp - prev_sample.timestamp
            # Ensure both bounding samples are strictly VALID with finite values
            can_interpolate = (
                gap <= self.max_interpolation_gap
                and prev_sample.quality == DataQualityStatus.VALID
                and next_sample.quality == DataQualityStatus.VALID
                and prev_sample.value is not None
                and next_sample.value is not None
            )

            # Also check that no intervening samples between prev and next were invalid/dropped
            if can_interpolate:
                for s in buf:
                    if prev_sample.timestamp < s.timestamp < next_sample.timestamp:
                        if s.quality != DataQualityStatus.VALID or s.value is None:
                            can_interpolate = False
                            break

            if can_interpolate:
                fraction = (target_timestamp - prev_sample.timestamp) / gap
                interp_val = prev_sample.value + fraction * (next_sample.value - prev_sample.value)
                return CanonicalMeasurement(
                    channel_name=channel_name,
                    value=interp_val,
                    unit=prev_sample.unit,
                    timestamp=target_timestamp,
                    source_timestamp=prev_sample.timestamp,
                    quality=DataQualityStatus.VALID,
                    status=QuantityStatus.ESTIMATED,
                    source_id=prev_sample.source_id,
                    source_type=prev_sample.source_type,
                    notes=f"INTERPOLATED across gap {gap:.4f}s",
                )

        # 3. Hold-Last-Value from preceding sample
        if prev_sample is not None:
            # If preceding sample itself was invalid or missing, do NOT hold a bogus value
            if prev_sample.quality != DataQualityStatus.VALID or prev_sample.value is None:
                return CanonicalMeasurement(
                    channel_name=channel_name,
                    value=None,
                    unit=prev_sample.unit,
                    timestamp=target_timestamp,
                    source_timestamp=prev_sample.timestamp,
                    quality=prev_sample.quality,
                    status=QuantityStatus.UNAVAILABLE,
                    source_id=prev_sample.source_id,
                    source_type=prev_sample.source_type,
                    notes=f"HOLD_PRECEDING_INVALID ({prev_sample.quality.value})",
                )

            age = target_timestamp - prev_sample.timestamp
            if age <= tau_stale:
                return CanonicalMeasurement(
                    channel_name=channel_name,
                    value=prev_sample.value,
                    unit=prev_sample.unit,
                    timestamp=target_timestamp,
                    source_timestamp=prev_sample.timestamp,
                    quality=DataQualityStatus.VALID,
                    status=QuantityStatus.ESTIMATED,
                    source_id=prev_sample.source_id,
                    source_type=prev_sample.source_type,
                    notes=f"HELD_LAST_VALUE (age={age:.4f}s <= tau_stale={tau_stale}s)",
                )
            else:
                return CanonicalMeasurement(
                    channel_name=channel_name,
                    value=None,
                    unit=prev_sample.unit,
                    timestamp=target_timestamp,
                    source_timestamp=prev_sample.timestamp,
                    quality=DataQualityStatus.STALE,
                    status=QuantityStatus.UNAVAILABLE,
                    source_id=prev_sample.source_id,
                    source_type=prev_sample.source_type,
                    notes=f"STALE (age={age:.4f}s > tau_stale={tau_stale}s)",
                )


        # No preceding sample available
        return CanonicalMeasurement(
            channel_name=channel_name,
            value=None,
            unit="unknown",
            timestamp=target_timestamp,
            quality=DataQualityStatus.UNAVAILABLE,
            status=QuantityStatus.UNAVAILABLE,
            notes="PRE_FIRST_SAMPLE",
        )

    def align_packet(
        self,
        target_timestamp: float,
        channel_names: List[str],
        allow_interpolation: bool = False,
        source_id: str = "multi_rate_feed",
        source_type: SourceType = SourceType.REPLAY,
    ) -> CanonicalTelemetryPacket:
        """
        Synthesize a synchronized CanonicalTelemetryPacket at target_timestamp
        containing all requested channels aligned from the multi-rate buffer.
        """
        measurements: Dict[str, CanonicalMeasurement] = {}
        for ch in channel_names:
            measurements[ch] = self.get_aligned_measurement(
                channel_name=ch,
                target_timestamp=target_timestamp,
                allow_interpolation=allow_interpolation,
            )

        return CanonicalTelemetryPacket(
            timestamp=target_timestamp,
            source_timestamp=target_timestamp,
            ingest_timestamp=target_timestamp,
            latency=0.0,
            source_id=source_id,
            source_type=source_type,
            measurements=measurements,
        )
