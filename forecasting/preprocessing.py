"""
Strictly causal data preprocessing and context extraction for Phase 10 forecasting.
"""

import math
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from forecasting.schema import (
    DEFAULT_FORECAST_CHANNELS,
    ForecastingConfig,
    ForecastQuality,
)


def _make_key(engine_id: str, mission_id: Optional[str]) -> Tuple[str, str]:
    return (str(engine_id), str(mission_id) if mission_id is not None else "")


class CausalTelemetryBuffer:
    """
    Chronological ring-buffer maintaining historical telemetry context per (engine_id, mission_id).

    Guarantees:
    - Strictly causal: only observations up to current time t enter context.
    - Rejects non-positive elapsed time (dt <= 0).
    - Invalidates history when timestamp gap > max_timestamp_gap_s.
    - Zero mission leakage: resets on mission change or explicit reset.
    - Causal NaN handling: internal NaNs are forward-filled; leading NaNs yield INSUFFICIENT_CONTEXT.
    - Missing values are NEVER blindly converted to zero.
    """

    def __init__(self, config: Optional[ForecastingConfig] = None):
        self.config = config or ForecastingConfig()
        # Keyed by (engine_id, mission_id) -> list of record dicts
        self._history: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    def reset(self, engine_id: Optional[str] = None, mission_id: Optional[str] = None) -> None:
        """Reset internal history for a specific mission or all missions."""
        if engine_id is None and mission_id is None:
            self._history.clear()
        elif engine_id is not None and mission_id is not None:
            key = _make_key(engine_id, mission_id)
            self._history.pop(key, None)
        elif engine_id is not None:
            keys_to_del = [k for k in self._history if k[0] == str(engine_id)]
            for k in keys_to_del:
                self._history.pop(k, None)

    def add_observation(
        self,
        record: Dict[str, Any],
        engine_id: str = "ENG_001",
        mission_id: Optional[str] = None,
    ) -> bool:
        """
        Add a single telemetry observation to the causal history buffer.

        Returns:
            True if observation was accepted into causal history, False if rejected (dt <= 0).
        """
        key = _make_key(engine_id, mission_id)
        if key not in self._history:
            self._history[key] = []

        history = self._history[key]
        timestamp = float(record.get("timestamp", 0.0))

        if history:
            last_t = float(history[-1].get("timestamp", 0.0))
            dt = timestamp - last_t

            # Reject duplicate or out-of-order timestamps
            if dt <= 0.0:
                return False

            # Break history on large gap
            if dt > self.config.max_timestamp_gap_s:
                history.clear()

        # Clean record: extract target channels as floats
        clean_record: Dict[str, Any] = {"timestamp": timestamp}
        for ch in self.config.target_channels:
            val = record.get(ch)
            if val is not None and pd.notna(val):
                clean_record[ch] = float(val)
            else:
                clean_record[ch] = float("nan")

        history.append(clean_record)

        # Cap history length to avoid unbounded memory growth
        max_keep = max(self.config.context_length * 4, 256)
        if len(history) > max_keep:
            del history[:-max_keep]

        return True

    def get_context(
        self,
        engine_id: str = "ENG_001",
        mission_id: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], str, Dict[str, Any]]:
        """
        Extract the most recent context window for forecasting.

        Returns:
            (context_2d, timestamps_1d, quality, metadata):
            - context_2d: shape (num_channels, context_length) float32, or None if insufficient
            - timestamps_1d: shape (context_length,) float64, or None
            - quality: ForecastQuality enum value
            - metadata: Dict describing provenance and imputation details
        """
        key = _make_key(engine_id, mission_id)
        history = self._history.get(key, [])
        ctx_len = self.config.context_length

        if len(history) < ctx_len:
            return (
                None,
                None,
                ForecastQuality.INSUFFICIENT_CONTEXT.value,
                {"reason": f"History length {len(history)} < required context {ctx_len}"},
            )

        # Take the most recent ctx_len samples
        window = history[-ctx_len:]
        timestamps = np.array([row["timestamp"] for row in window], dtype=np.float64)

        # Audit latest observation: if all target channels are missing (blackout), do not forecast
        latest_obs = window[-1]
        valid_latest = sum(
            1 for ch in self.config.target_channels
            if ch in latest_obs and latest_obs[ch] is not None and not math.isnan(float(latest_obs[ch]))
        )
        if valid_latest == 0:
            return (
                None,
                None,
                ForecastQuality.INSUFFICIENT_CONTEXT.value,
                {"reason": "Sensor blackout at forecast origin: all target channels are missing in latest observation."},
            )

        # Audit channels and perform strictly causal forward-fill
        imputed_channels: List[str] = []
        channel_arrays: List[np.ndarray] = []

        for ch in self.config.target_channels:
            raw_vals = [row[ch] for row in window]

            # Check for leading NaN -> no backward fill permitted!
            if math.isnan(raw_vals[0]):
                return (
                    None,
                    None,
                    ForecastQuality.INSUFFICIENT_CONTEXT.value,
                    {"reason": f"Channel '{ch}' has leading NaN in context window; backward fill disallowed."},
                )

            # Causal forward-fill internal NaNs
            clean_vals = list(raw_vals)
            channel_was_imputed = False
            for k in range(1, len(clean_vals)):
                if math.isnan(clean_vals[k]):
                    clean_vals[k] = clean_vals[k - 1]
                    channel_was_imputed = True

            if channel_was_imputed:
                imputed_channels.append(ch)

            channel_arrays.append(np.array(clean_vals, dtype=np.float32))

        context_2d = np.stack(channel_arrays, axis=0)  # Shape (num_channels, ctx_len)
        quality = (
            ForecastQuality.IMPUTED.value if imputed_channels else ForecastQuality.VALID.value
        )
        metadata = {
            "context_start_timestamp": float(timestamps[0]),
            "context_end_timestamp": float(timestamps[-1]),
            "imputed_channels": imputed_channels,
            "engine_id": engine_id,
            "mission_id": mission_id,
        }

        return context_2d, timestamps, quality, metadata
