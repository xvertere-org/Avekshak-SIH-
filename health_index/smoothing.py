"""
Causal Exponentially Weighted Moving Average (EWMA) smoothing for Health Index.
"""

from typing import Dict, Optional, Tuple
import math


def _make_key(engine_id: str, mission_id: Optional[str]) -> Tuple[str, str]:
    return (str(engine_id), str(mission_id) if mission_id is not None else "")


class CausalEWMASmoother:
    """
    Strictly causal EWMA filter for Health Index.

    At any time t, smoothed_health_index(t) depends strictly on observations
    at or before time t. No future samples or centered windows are ever accessed.

    Formula:
        HI_smooth(t) = alpha * HI_raw(t) + (1 - alpha) * HI_smooth(t_prev)
    """

    def __init__(self, alpha: float = 0.15, max_timestamp_gap_s: float = 5.0):
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"EWMA alpha must be in (0.0, 1.0], got {alpha}")
        self.alpha = alpha
        self.max_timestamp_gap_s = max_timestamp_gap_s
        # Map (engine_id, mission_id) -> previous smoothed value
        self._states: Dict[Tuple[str, str], float] = {}
        self._last_timestamps: Dict[Tuple[str, str], float] = {}

    def reset(
        self,
        engine_id: Optional[str] = None,
        mission_id: Optional[str] = None,
    ) -> None:
        """Reset internal filter state."""
        if engine_id is not None and mission_id is not None:
            key = _make_key(engine_id, mission_id)
            self._states.pop(key, None)
            self._last_timestamps.pop(key, None)
        elif engine_id is not None:
            keys_to_remove = [k for k in self._states if k[0] == str(engine_id)]
            for k in keys_to_remove:
                self._states.pop(k, None)
                self._last_timestamps.pop(k, None)
        else:
            self._states.clear()
            self._last_timestamps.clear()

    def update(
        self,
        raw_hi: float,
        engine_id: str = "ENG_001",
        mission_id: Optional[str] = None,
        timestamp: Optional[float] = None,
    ) -> float:
        """
        Update smoother with new raw health index observation.

        Args:
            raw_hi: Instantaneous raw health index in [0.0, 1.0], or NaN.
            engine_id: Identifier for the engine state.
            mission_id: Optional identifier for mission state isolation.
            timestamp: Optional telemetry observation timestamp for gap detection.

        Returns:
            Smoothed health index in [0.0, 1.0], or NaN if raw_hi is NaN.
        """
        if math.isnan(raw_hi):
            return float("nan")

        key = _make_key(engine_id, mission_id)

        # Gap & Monotonicity detection
        if timestamp is not None and key in self._last_timestamps:
            last_t = self._last_timestamps[key]
            gap = timestamp - last_t
            if gap <= 0.0:
                # Duplicate (gap == 0) or out-of-order (gap < 0) observation:
                # Do NOT advance EWMA filter state or corrupt timestamp tracking.
                prev = self._states.get(key, None)
                return prev if prev is not None else raw_hi
            if gap > self.max_timestamp_gap_s:
                # Telemetry continuity broken by large gap: reset state
                self._states.pop(key, None)
            self._last_timestamps[key] = timestamp
        elif timestamp is not None:
            self._last_timestamps[key] = timestamp

        prev = self._states.get(key, None)
        if prev is None or math.isnan(prev):
            # Initial condition: exact instantaneous value
            smoothed = raw_hi
        else:
            smoothed = self.alpha * raw_hi + (1.0 - self.alpha) * prev

        # Ensure bounded [0.0, 1.0]
        smoothed = float(min(1.0, max(0.0, smoothed)))
        self._states[key] = smoothed
        return smoothed
