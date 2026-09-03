"""
Telemetry Preprocessing, Synchronization, and Resampling for SIH26054.

Provides:
- TelemetrySynchronizer: Multi-channel time-grid validation and alignment
- TelemetryResampler: Causal, deterministic resampling with signal-specific aggregation:
    * Vibration: Scalar Root-Mean-Square (RMS) aggregation: sqrt(mean(v^2))
    * Continuous thermal/fluid channels: Causal mean or last-sample interpolation
    * Categorical channels: Mode or forward-fill
    * Sensor dropout / NaN: Missingness preserved (never converted to 0)
    * Strict causality: Window [t - dt, t] stamped at t (zero future data leakage)
- TelemetryCleaner: Explicit, non-destructive cleaning and traceable imputation
"""

import math
from typing import Dict, Any, List, Optional, Union, Callable
import numpy as np
import pandas as pd

from telemetry.ingestion import (
    CanonicalTelemetryFrame,
    OPTIONAL_PHYSICAL_CHANNELS,
    REQUIRED_COLUMNS,
)


class TelemetrySynchronizer:
    """
    Validates and enforces common time grid synchronization across telemetry channels.
    """

    @classmethod
    def check_alignment(
        cls,
        frame: CanonicalTelemetryFrame,
        expected_dt_s: float = 0.1,
        tolerance_s: float = 0.005,
    ) -> Dict[str, Any]:
        """
        Check whether timestamps in frame adhere to a regular, aligned grid.

        Returns:
            Dict containing pass/fail, max_jitter_s, and sampling stats.
        """
        df = frame.to_dataframe()
        if "timestamp" not in df.columns or len(df) < 2:
            return {"aligned": True, "max_jitter_s": 0.0, "mean_dt_s": expected_dt_s}

        ts = df["timestamp"].to_numpy(dtype=float)
        diffs = np.diff(ts)
        jitter = np.abs(diffs - expected_dt_s)
        max_jitter = float(np.max(jitter)) if len(jitter) > 0 else 0.0
        is_aligned = max_jitter <= tolerance_s

        return {
            "aligned": is_aligned,
            "max_jitter_s": round(max_jitter, 6),
            "mean_dt_s": round(float(np.mean(diffs)), 6),
            "std_dt_s": round(float(np.std(diffs)), 6),
        }

    @classmethod
    def align_to_grid(
        cls,
        frame: CanonicalTelemetryFrame,
        grid_dt_s: float = 0.1,
        method: str = "nearest",
    ) -> CanonicalTelemetryFrame:
        """
        Align asynchronously sampled records to a uniform time grid.

        Args:
            frame: CanonicalTelemetryFrame to align.
            grid_dt_s: Uniform grid interval in seconds.
            method: 'nearest' or 'forward_fill'.

        Returns:
            Synchronized CanonicalTelemetryFrame.
        """
        df = frame.to_dataframe()
        if "timestamp" not in df.columns or len(df) < 2:
            return frame.copy()

        t_min = df["timestamp"].min()
        t_max = df["timestamp"].max()
        target_grid = np.arange(t_min, t_max + grid_dt_s * 0.5, grid_dt_s)

        df_aligned = pd.DataFrame({"timestamp": target_grid})
        df_sorted = df.sort_values("timestamp")

        if method == "nearest":
            merged = pd.merge_asof(
                df_aligned,
                df_sorted,
                on="timestamp",
                direction="nearest",
                tolerance=grid_dt_s * 0.6,
            )
        else:
            merged = pd.merge_asof(
                df_aligned,
                df_sorted,
                on="timestamp",
                direction="backward",
            )

        return CanonicalTelemetryFrame(merged, metadata=frame.metadata)


class TelemetryResampler:
    """
    Causal, deterministic telemetry resampling with channel-specific aggregation rules.
    """

    def __init__(
        self,
        target_dt_s: float = 1.0,
    ):
        """
        Initialize TelemetryResampler.

        Args:
            target_dt_s: Target resampled interval in seconds (e.g. 1.0s for 1 Hz from 10 Hz).
        """
        if target_dt_s <= 0:
            raise ValueError(f"target_dt_s must be positive, got {target_dt_s}")
        self.target_dt_s = target_dt_s

    def resample(
        self,
        frame: CanonicalTelemetryFrame,
    ) -> CanonicalTelemetryFrame:
        """
        Resample frame to target_dt_s strictly using backward-looking causal windows.

        Aggregations:
        - vibration: Scalar RMS aggregation: sqrt(mean(v^2))
        - continuous physical channels: Mean over window
        - categorical / string channels: Last observation in window
        - fault_severity: Max in window
        - NaN / sensor dropout: Preserves NaN if entire window is missing
        """
        df = frame.to_dataframe()
        if "timestamp" not in df.columns or len(df) == 0:
            return frame.copy()

        df_sorted = df.sort_values("timestamp").copy()
        t_start = df_sorted["timestamp"].iloc[0]
        t_end = df_sorted["timestamp"].iloc[-1]

        if t_end <= t_start:
            return frame.copy()

        # Define causal grid points stamped at the END of each window [t - target_dt_s, t]
        grid_stamps = np.arange(t_start + self.target_dt_s, t_end + self.target_dt_s * 0.5, self.target_dt_s)
        if len(grid_stamps) == 0:
            # Short sequence smaller than target_dt_s: single window ending at t_end
            grid_stamps = np.array([t_end])

        resampled_rows: List[Dict[str, Any]] = []

        for idx, t_stamp in enumerate(grid_stamps):
            w_start = t_stamp - self.target_dt_s
            # Strictly causal window: include left boundary for initial window to capture t_start
            if idx == 0:
                window = df_sorted[(df_sorted["timestamp"] >= w_start) & (df_sorted["timestamp"] <= t_stamp)]
            else:
                window = df_sorted[(df_sorted["timestamp"] > w_start) & (df_sorted["timestamp"] <= t_stamp)]

            if len(window) == 0:
                # If window is empty, fallback to the latest available sample <= t_stamp
                prior = df_sorted[df_sorted["timestamp"] <= t_stamp]
                if len(prior) == 0:
                    continue
                row_dict = prior.iloc[-1].to_dict()
                row_dict["timestamp"] = round(float(t_stamp), 4)
                resampled_rows.append(row_dict)
                continue

            row_dict: Dict[str, Any] = {"timestamp": round(float(t_stamp), 4)}

            for col in df_sorted.columns:
                if col == "timestamp":
                    continue

                vals = window[col]

                # 1. Vibration: Scalar RMS aggregation sqrt(mean(v^2))
                if col == "vibration":
                    num_vals = vals.to_numpy(dtype=float)
                    valid_num = num_vals[~np.isnan(num_vals)]
                    if len(valid_num) > 0:
                        rms_val = float(np.sqrt(np.mean(valid_num ** 2)))
                        row_dict[col] = round(rms_val, 4)
                    else:
                        row_dict[col] = float("nan")

                # 2. Continuous physical channels: mean
                elif col in OPTIONAL_PHYSICAL_CHANNELS or col in ["altitude", "ambient_temp", "throttle", "load"]:
                    num_vals = vals.to_numpy(dtype=float)
                    valid_num = num_vals[~np.isnan(num_vals)]
                    if len(valid_num) > 0:
                        row_dict[col] = round(float(np.mean(valid_num)), 4)
                    else:
                        row_dict[col] = float("nan")

                # 3. Fault severity: max in window
                elif col == "fault_severity":
                    num_vals = vals.to_numpy(dtype=float)
                    valid_num = num_vals[~np.isnan(num_vals)]
                    row_dict[col] = round(float(np.max(valid_num)), 4) if len(valid_num) > 0 else 0.0

                # 4. Metadata dictionary: take last non-empty
                elif col == "metadata":
                    last_meta = None
                    for m in vals:
                        if isinstance(m, dict) and len(m) > 0:
                            last_meta = m
                    row_dict[col] = last_meta if last_meta is not None else {}

                # 5. Categorical/identity columns: last observation in window
                else:
                    row_dict[col] = vals.iloc[-1]

            resampled_rows.append(row_dict)

        resampled_df = pd.DataFrame(resampled_rows)
        return CanonicalTelemetryFrame(resampled_df, metadata=frame.metadata)


class TelemetryCleaner:
    """
    Explicit, non-destructive cleaning and traceable imputation utilities.
    """

    @classmethod
    def deduplicate(
        cls,
        frame: CanonicalTelemetryFrame,
        keep: str = "first",
    ) -> CanonicalTelemetryFrame:
        """
        Deduplicate telemetry records by (engine_id, mission_id, timestamp).
        """
        df = frame.to_dataframe()
        subset = [c for c in ["engine_id", "mission_id", "timestamp"] if c in df.columns]
        cleaned = df.drop_duplicates(subset=subset, keep=keep).copy()
        return CanonicalTelemetryFrame(cleaned, metadata=frame.metadata)

    @classmethod
    def drop_invalid(
        cls,
        frame: CanonicalTelemetryFrame,
        status_column: str = "quality_status",
    ) -> CanonicalTelemetryFrame:
        """
        Drop rows tagged as INVALID (preserving WARNING and MISSING).
        """
        df = frame.to_dataframe()
        if status_column not in df.columns:
            return frame.copy()
        filtered = df[df[status_column] != "invalid"].copy()
        return CanonicalTelemetryFrame(filtered, metadata=frame.metadata)

    @classmethod
    def impute_missing(
        cls,
        frame: CanonicalTelemetryFrame,
        method: str = "forward_fill",
        limit: int = 5,
        channels: Optional[List[str]] = None,
    ) -> CanonicalTelemetryFrame:
        """
        Traceable imputation for missing values (e.g. brief sensor dropout).
        Annotates an 'imputed_mask' column so downstream models are aware of imputed values.

        Args:
            frame: CanonicalTelemetryFrame with potential NaNs.
            method: 'forward_fill' or 'linear'.
            limit: Maximum consecutive missing samples to impute.
            channels: Specific channels to impute (defaults to physical channels).
        """
        df = frame.to_dataframe()
        target_channels = channels or [c for c in OPTIONAL_PHYSICAL_CHANNELS if c in df.columns]

        was_nan = df[target_channels].isna().any(axis=1)
        imputed_df = df.copy()

        for ch in target_channels:
            if ch not in imputed_df.columns:
                continue
            if method == "forward_fill":
                imputed_df[ch] = imputed_df[ch].ffill(limit=limit)
            elif method == "linear":
                imputed_df[ch] = imputed_df[ch].interpolate(method="linear", limit=limit)

        now_nan = imputed_df[target_channels].isna().any(axis=1)
        # Any row that had NaNs that are now filled is marked imputed
        imputed_df["imputed_mask"] = was_nan & (~now_nan)

        return CanonicalTelemetryFrame(imputed_df, metadata=frame.metadata)
