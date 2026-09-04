"""
Causal degradation rate calculation, trend tracking, and health state categorization.
"""

from typing import Dict, List, Tuple, Optional
import math
import numpy as np

from health_index.schema import (
    HealthIndexConfig,
    HealthState,
    DegradationTrend,
)


def _make_key(engine_id: str, mission_id: Optional[str]) -> Tuple[str, str]:
    return (str(engine_id), str(mission_id) if mission_id is not None else "")


class DegradationTracker:
    """
    Tracks causal degradation trajectory, rate of health change, and trend classification.
    """

    def __init__(self, config: Optional[HealthIndexConfig] = None):
        self.config = config or HealthIndexConfig()
        # Per (engine_id, mission_id) history of (timestamp, smoothed_hi)
        self._history: Dict[Tuple[str, str], List[Tuple[float, float]]] = {}
        self._last_timestamp: Dict[Tuple[str, str], float] = {}

    def reset(
        self,
        engine_id: Optional[str] = None,
        mission_id: Optional[str] = None,
    ) -> None:
        """Reset historical tracking state."""
        if engine_id is not None and mission_id is not None:
            key = _make_key(engine_id, mission_id)
            self._history.pop(key, None)
            self._last_timestamp.pop(key, None)
        elif engine_id is not None:
            keys_to_remove = [k for k in self._history if k[0] == str(engine_id)]
            for k in keys_to_remove:
                self._history.pop(k, None)
                self._last_timestamp.pop(k, None)
        else:
            self._history.clear()
            self._last_timestamp.clear()

    def update_and_evaluate(
        self,
        timestamp: float,
        smoothed_hi: float,
        engine_id: str = "ENG_001",
        mission_id: Optional[str] = None,
    ) -> Tuple[float, str, str]:
        """
        Update tracker with new timestamped smoothed health index and compute rate/trend/state.

        Args:
            timestamp: Telemetry observation time in seconds.
            smoothed_hi: Causally smoothed health index, or NaN.
            engine_id: Engine identifier.
            mission_id: Optional mission identifier for mission state isolation.

        Returns:
            (degradation_rate, degradation_trend, health_state)
        """
        key = _make_key(engine_id, mission_id)

        # 1. Health state classification
        if math.isnan(smoothed_hi):
            health_state = HealthState.INSUFFICIENT_DATA.value
            return float("nan"), DegradationTrend.INSUFFICIENT_DATA.value, health_state

        if smoothed_hi >= self.config.healthy_threshold:
            health_state = HealthState.HEALTHY.value
        elif smoothed_hi >= self.config.degraded_threshold:
            health_state = HealthState.DEGRADED.value
        elif smoothed_hi >= self.config.severely_degraded_threshold:
            health_state = HealthState.SEVERELY_DEGRADED.value
        else:
            health_state = HealthState.CRITICAL.value

        # 2. Timestamp Robustness Checks
        last_t = self._last_timestamp.get(key, None)

        if last_t is not None:
            dt_step = timestamp - last_t

            # Case A: Duplicate or Out-of-Order timestamp (dt <= 0)
            if dt_step <= 0.0:
                # Deterministic invalid-observation policy:
                # Do NOT corrupt causal chronological history with backwards or zero-time data.
                # Rate cannot be computed over non-positive elapsed time -> NaN.
                return float("nan"), DegradationTrend.INSUFFICIENT_DATA.value, health_state

            # Case B: Large timestamp gap exceeding max_timestamp_gap_s
            if dt_step > self.config.max_timestamp_gap_s:
                # Break degradation history: start new continuous history epoch
                self._history[key] = [(timestamp, smoothed_hi)]
                self._last_timestamp[key] = timestamp
                return float("nan"), DegradationTrend.INSUFFICIENT_HISTORY.value, health_state

        # 3. Record valid causal sample
        if key not in self._history:
            self._history[key] = []

        hist = self._history[key]
        hist.append((timestamp, smoothed_hi))
        self._last_timestamp[key] = timestamp

        # Prune old history exceeding 2.5x rate_horizon to maintain memory efficiency
        horizon = self.config.rate_horizon_s
        cutoff = timestamp - 2.5 * horizon
        while len(hist) > 1 and hist[0][0] < cutoff:
            hist.pop(0)

        # 4. Evaluate initial-history condition
        t_start = hist[0][0]
        if (timestamp - t_start) < horizon:
            # Not enough elapsed history to compute a statistically meaningful rate
            return float("nan"), DegradationTrend.INSUFFICIENT_HISTORY.value, health_state

        # 5. Find latest sample at or before (timestamp - horizon)
        target_t = timestamp - horizon
        ref_sample = None
        for t_past, hi_past in reversed(hist):
            if t_past <= target_t:
                ref_sample = (t_past, hi_past)
                break

        if ref_sample is None:
            ref_sample = hist[0]

        t_ref, hi_ref = ref_sample
        dt = timestamp - t_ref
        if dt <= 0.0:
            return 0.0, DegradationTrend.STABLE.value, health_state

        # Degradation rate = change in health per second
        rate = float((smoothed_hi - hi_ref) / dt)

        # 6. Trend classification based on rate thresholds
        if rate <= self.config.rapid_degrade_rate:
            trend = DegradationTrend.RAPIDLY_DEGRADING.value
        elif rate <= self.config.degrade_rate:
            trend = DegradationTrend.DEGRADING.value
        elif rate >= self.config.improving_rate:
            trend = DegradationTrend.IMPROVING.value
        else:
            trend = DegradationTrend.STABLE.value

        return rate, trend, health_state
