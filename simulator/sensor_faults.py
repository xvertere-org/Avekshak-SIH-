"""
Sensor Fault Processor for SIH26054 Aero Piston Engine Simulator.

Implements observation-layer sensor faults that corrupt telemetry readings
WITHOUT modifying any physical engine state. This is Phase 4F of the fault
injection framework.

Supported fault modes:
- BIAS:    Constant measurement offset
- DRIFT:   Time-dependent measurement error
- STUCK:   Sensor latches at observed value at fault activation
- NOISE:   Additional fault-induced sensor noise
- DROPOUT: Observation becomes unavailable (NaN)

DISCLAIMER:
All sensor-fault magnitudes are Tier C/D calibration or engineering assumptions.
They do NOT represent measured or certified UAV-engine sensor characteristics.
"""

import math
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple


class SensorFaultMode(str, Enum):
    """Sensor fault mode categories."""
    BIAS = "bias"
    DRIFT = "drift"
    STUCK = "stuck"
    NOISE = "noise"
    DROPOUT = "dropout"


class SensorChannel(str, Enum):
    """Telemetry channels that can be affected by sensor faults."""
    RPM = "rpm"
    CHT = "cht"
    EGT = "egt"
    OIL_TEMP = "oil_temp"
    OIL_PRESSURE = "oil_pressure"
    FUEL_FLOW = "fuel_flow"
    VIBRATION = "vibration"


# Default per-channel fault magnitudes (Tier C/D engineering assumptions)
# These are the maximum magnitudes at severity=1.0.
# Keys map SensorChannel -> default magnitude for each fault mode.
_DEFAULT_BIAS_MAGNITUDES: Dict[str, float] = {
    SensorChannel.RPM.value: 50.0,            # RPM
    SensorChannel.CHT.value: 8.0,             # °C
    SensorChannel.EGT.value: 15.0,            # °C
    SensorChannel.OIL_TEMP.value: 6.0,        # °C
    SensorChannel.OIL_PRESSURE.value: 0.5,    # bar
    SensorChannel.FUEL_FLOW.value: 2.0,       # L/h
    SensorChannel.VIBRATION.value: 0.15,      # g
}

_DEFAULT_DRIFT_RATES: Dict[str, float] = {
    SensorChannel.RPM.value: 5.0,             # RPM/s
    SensorChannel.CHT.value: 0.8,             # °C/s
    SensorChannel.EGT.value: 1.5,             # °C/s
    SensorChannel.OIL_TEMP.value: 0.6,        # °C/s
    SensorChannel.OIL_PRESSURE.value: 0.05,   # bar/s
    SensorChannel.FUEL_FLOW.value: 0.2,       # L/h/s
    SensorChannel.VIBRATION.value: 0.015,     # g/s
}

_DEFAULT_NOISE_STDS: Dict[str, float] = {
    SensorChannel.RPM.value: 20.0,            # RPM
    SensorChannel.CHT.value: 3.0,             # °C
    SensorChannel.EGT.value: 8.0,             # °C
    SensorChannel.OIL_TEMP.value: 2.0,        # °C
    SensorChannel.OIL_PRESSURE.value: 0.2,    # bar
    SensorChannel.FUEL_FLOW.value: 1.0,       # L/h
    SensorChannel.VIBRATION.value: 0.08,      # g
}


class SensorFaultProcessor:
    """
    Stateful processor that applies sensor faults to telemetry channel values.

    Operates purely at the observation layer — never modifies physical engine state.

    Fault application order when multiple faults target the same channel:
    1. DROPOUT takes highest precedence — if any active dropout fault exists,
       the channel becomes NaN immediately (no further processing).
    2. STUCK takes next precedence — if any active stuck fault exists,
       the channel value is replaced with the latched observation value.
    3. BIAS, DRIFT, NOISE compose additively in FaultSchedule insertion order.

    RNG-neutrality: When no sensor fault is active at a given timestamp,
    this processor consumes zero RNG draws, preserving bitwise-identical
    golden healthy telemetry.
    """

    def __init__(
        self,
        rng: Optional[np.random.Generator] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the sensor fault processor.

        Args:
            rng: Seeded NumPy random generator. If None, created from seed.
            seed: Random seed (used only if rng is None).
        """
        if rng is not None:
            self.rng = rng
        else:
            self.rng = np.random.default_rng(seed if seed is not None else 42)

        # Stateful tracking for STUCK faults: {fault_id -> latched_value}
        # fault_id is (id(fault_state), channel) to handle multiple stuck faults
        self._stuck_latches: Dict[Tuple, float] = {}

    def reset(self) -> None:
        """
        Clear all stateful sensor-fault tracking (STUCK latches, etc.).

        Must be called when the simulator resets or starts a new mission
        to prevent sensor fault state from leaking across missions.
        """
        self._stuck_latches.clear()

    def apply(
        self,
        channel_values: Dict[str, float],
        fault_states: List[Any],
        timestamp: float,
    ) -> Dict[str, float]:
        """
        Apply all active sensor faults to channel values.

        Args:
            channel_values: Dict mapping channel name -> observed value
                            (after normal sensor noise has been applied).
            fault_states: List of FaultState objects with fault_type == SENSOR_FAULT.
            timestamp: Current simulation timestamp in seconds.

        Returns:
            Dict mapping channel name -> fault-corrupted observed value.
            Values may be float('nan') for DROPOUT faults.
        """
        if not fault_states:
            return channel_values

        result = dict(channel_values)

        # Collect active sensor faults grouped by channel, preserving insertion order
        channel_faults: Dict[str, List[Any]] = {}
        for fs in fault_states:
            if not fs.is_active_at(timestamp):
                continue
            if fs.get_effective_severity(timestamp) <= 0.0:
                continue

            channel = self._get_channel(fs)
            if channel is None or channel not in result:
                continue

            if channel not in channel_faults:
                channel_faults[channel] = []
            channel_faults[channel].append(fs)

        # Apply faults per channel
        for channel, faults in channel_faults.items():
            result[channel] = self._apply_channel_faults(
                channel, result[channel], faults, timestamp
            )

        return result

    def _get_channel(self, fault_state: Any) -> Optional[str]:
        """Extract the sensor channel from a FaultState's parameters."""
        params = fault_state.parameters if fault_state.parameters else {}
        channel = params.get("sensor_channel", None)
        if channel is None:
            return None
        # Normalize to string value
        if hasattr(channel, "value"):
            channel = channel.value
        return str(channel).lower()

    def _get_mode(self, fault_state: Any) -> Optional[SensorFaultMode]:
        """Extract the sensor fault mode from a FaultState's parameters."""
        params = fault_state.parameters if fault_state.parameters else {}
        mode = params.get("sensor_mode", None)
        if mode is None:
            return None
        if isinstance(mode, SensorFaultMode):
            return mode
        try:
            return SensorFaultMode(str(mode).lower())
        except ValueError:
            return None

    def _apply_channel_faults(
        self,
        channel: str,
        value: float,
        faults: List[Any],
        timestamp: float,
    ) -> float:
        """
        Apply multiple faults to a single channel, respecting precedence rules.

        Precedence (highest first):
        1. DROPOUT → NaN (short-circuit)
        2. STUCK → latch value (short-circuit)
        3. BIAS / DRIFT / NOISE → additive composition
        """
        # Check for DROPOUT first (highest precedence)
        for fs in faults:
            mode = self._get_mode(fs)
            if mode == SensorFaultMode.DROPOUT:
                severity = fs.get_effective_severity(timestamp)
                if severity > 0.0:
                    return float('nan')

        # Check for STUCK (second precedence)
        for fs in faults:
            mode = self._get_mode(fs)
            if mode == SensorFaultMode.STUCK:
                severity = fs.get_effective_severity(timestamp)
                if severity > 0.0:
                    latch_key = (id(fs), channel)
                    if latch_key not in self._stuck_latches:
                        # Latch the OBSERVED value (after normal sensor noise)
                        self._stuck_latches[latch_key] = value
                    return self._stuck_latches[latch_key]

        # Apply additive faults (BIAS, DRIFT, NOISE) in insertion order
        for fs in faults:
            mode = self._get_mode(fs)
            severity = fs.get_effective_severity(timestamp)
            params = fs.parameters if fs.parameters else {}

            if mode == SensorFaultMode.BIAS:
                magnitude = params.get("bias_magnitude",
                                       _DEFAULT_BIAS_MAGNITUDES.get(channel, 0.0))
                value += float(magnitude) * severity

            elif mode == SensorFaultMode.DRIFT:
                drift_rate = params.get("drift_rate",
                                        _DEFAULT_DRIFT_RATES.get(channel, 0.0))
                elapsed = timestamp - fs.start_time
                value += float(drift_rate) * severity * elapsed

            elif mode == SensorFaultMode.NOISE:
                noise_std = params.get("noise_std",
                                       _DEFAULT_NOISE_STDS.get(channel, 0.0))
                noise_val = float(self.rng.normal(0.0, float(noise_std) * severity))
                value += noise_val

        return value
