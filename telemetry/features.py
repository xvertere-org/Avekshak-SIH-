"""
Feature Engineering and Vibration Analytics for SIH26054 Aero Piston Engine Digital Twin.

Transforms canonical telemetry into ML-ready feature matrices with strict causality:
- Raw signal preservation (temperatures, pressures, rpm, fuel flow, vibration)
- Causal backward-looking temporal features (rolling mean, std, slopes, rates of change)
- Thermal & dynamic stability metrics (temperature trends, RPM coefficient of variation)
- Vibration analytics:
    * Statistical metrics (RMS, peak, crest factor)
    * Analytical 1x/2x crankshaft order frequencies (RPM / 60, 2 * RPM / 60)
    * Low-rate scalar spectral process energy (explicitly disclaimed as scalar process
      spectrum, NOT raw high-frequency kHz waveform FFT)
- LeakageSafeSplitter: Grouped and temporal partitioning with zero future-data leakage
"""

import math
from typing import Dict, Any, List, Optional, Tuple, Set, Union
import numpy as np
import pandas as pd

from telemetry.ingestion import CanonicalTelemetryFrame, OPTIONAL_PHYSICAL_CHANNELS


class TelemetryFeatureExtractor:
    """
    Extracts ML-ready causal temporal and vibration features from CanonicalTelemetryFrame.
    Guarantees strict backward-looking causality (closed='right') to prevent future leakage.
    """

    def __init__(
        self,
        rolling_windows: Optional[List[int]] = None,
        include_rates: bool = True,
        include_stability: bool = True,
        include_vibration_analytics: bool = True,
        min_periods: int = 1,
    ):
        """
        Initialize TelemetryFeatureExtractor.

        Args:
            rolling_windows: List of window sample sizes for rolling statistics (default: [5, 10]).
            include_rates: Whether to compute first differences and slopes.
            include_stability: Whether to compute RPM and temperature stability metrics.
            include_vibration_analytics: Whether to compute vibration statistics and analytical orders.
            min_periods: Minimum observations required in window (default 1 for causal startup).
        """
        self.rolling_windows = rolling_windows or [5, 10]
        self.include_rates = include_rates
        self.include_stability = include_stability
        self.include_vibration_analytics = include_vibration_analytics
        self.min_periods = min_periods

    def extract_features(
        self,
        frame: CanonicalTelemetryFrame,
    ) -> pd.DataFrame:
        """
        Extract all features from CanonicalTelemetryFrame into a unified DataFrame.
        """
        df = frame.to_dataframe().copy()
        if len(df) == 0:
            return pd.DataFrame()

        # Sort by timestamp for causal rolling computations
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)

        features = pd.DataFrame(index=df.index)

        # 1. Identity & Provenance columns (unmodified)
        id_cols = [c for c in ["timestamp", "engine_id", "mission_id", "mission_phase", "fault_type", "fault_severity", "source", "source_type"] if c in df.columns]
        for col in id_cols:
            features[col] = df[col]

        # 2. Raw Physical Signals (preserved)
        phys_present = [c for c in OPTIONAL_PHYSICAL_CHANNELS if c in df.columns]
        for col in phys_present:
            features[col] = df[col]

        # Auxiliary flight conditions
        for aux in ["altitude", "ambient_temp", "throttle", "load"]:
            if aux in df.columns and aux not in features.columns:
                features[aux] = df[aux]

        # Calculate time difference dt for slope estimations
        if "timestamp" in df.columns and len(df) > 1:
            dt_series = df["timestamp"].diff().replace(0, np.nan).fillna(0.1)
        else:
            dt_series = pd.Series(0.1, index=df.index)

        # 3. Causal Rolling Temporal Features
        key_channels = [c for c in ["rpm", "cht", "egt", "oil_temp", "oil_pressure", "fuel_flow", "vibration"] if c in df.columns]

        for w in self.rolling_windows:
            for ch in key_channels:
                s = df[ch]
                features[f"{ch}_roll_mean_w{w}"] = s.rolling(window=w, min_periods=self.min_periods, closed="right").mean()
                features[f"{ch}_roll_std_w{w}"] = s.rolling(window=w, min_periods=self.min_periods, closed="right").std().fillna(0.0)

        # 4. Rates of Change & Trends
        if self.include_rates:
            for ch in key_channels:
                diff = df[ch].diff().fillna(0.0)
                features[f"{ch}_rate_of_change"] = diff / dt_series

            # Specific thermal and pressure gradients
            if "cht" in df.columns:
                features["cht_trend_c_per_s"] = df["cht"].diff().fillna(0.0) / dt_series
            if "egt" in df.columns:
                features["egt_trend_c_per_s"] = df["egt"].diff().fillna(0.0) / dt_series
            if "oil_pressure" in df.columns:
                features["oil_press_gradient_bar_per_s"] = df["oil_pressure"].diff().fillna(0.0) / dt_series

        # 5. Stability Metrics
        if self.include_stability:
            if "rpm" in df.columns:
                w_stab = self.rolling_windows[0] if self.rolling_windows else 5
                rpm_mean = df["rpm"].rolling(window=w_stab, min_periods=self.min_periods, closed="right").mean()
                rpm_std = df["rpm"].rolling(window=w_stab, min_periods=self.min_periods, closed="right").std().fillna(0.0)
                # RPM coefficient of variation (std / mean)
                features["rpm_stability_cov"] = (rpm_std / rpm_mean.replace(0, np.nan)).fillna(0.0)

            # Mission phase persistence duration (seconds spent in current phase)
            if "mission_phase" in df.columns and "timestamp" in df.columns:
                phase_series = df["mission_phase"].astype(str)
                phase_changed = phase_series != phase_series.shift(1)
                phase_group = phase_changed.cumsum()
                t_start_per_phase = df.groupby(phase_group)["timestamp"].transform("min")
                features["phase_persistence_duration_s"] = (df["timestamp"] - t_start_per_phase).round(2)

        # 6. Vibration Features
        if self.include_vibration_analytics and "vibration" in df.columns:
            vib_s = df["vibration"]
            w_vib = self.rolling_windows[-1] if self.rolling_windows else 10

            # Scalar statistical features over causal window
            roll_mean_sq = (vib_s ** 2).rolling(window=w_vib, min_periods=self.min_periods, closed="right").mean()
            features["vibration_roll_rms"] = np.sqrt(roll_mean_sq).round(4)
            features["vibration_roll_peak"] = vib_s.abs().rolling(window=w_vib, min_periods=self.min_periods, closed="right").max().round(4)
            features["vibration_roll_std"] = vib_s.rolling(window=w_vib, min_periods=self.min_periods, closed="right").std().fillna(0.0).round(4)

            # Crest factor: peak / RMS
            eps = 1e-6
            features["vibration_crest_factor"] = (
                features["vibration_roll_peak"] / (features["vibration_roll_rms"] + eps)
            ).round(3)

            # Analytical crankshaft harmonic order frequencies from instantaneous RPM
            if "rpm" in df.columns:
                # 1x order: RPM / 60 (crankshaft rotational frequency in Hz)
                features["vib_order_1x_freq_hz"] = (df["rpm"] / 60.0).round(2)
                # 2x order: 2 * RPM / 60 (combustion fundamental order for 4-cylinder 4-stroke engine)
                features["vib_order_2x_freq_hz"] = (2.0 * df["rpm"] / 60.0).round(2)

            # Scalar process energy (variance of scalar vibration in causal window)
            vib_variance = vib_s.rolling(window=w_vib, min_periods=self.min_periods, closed="right").var().fillna(0.0)
            features["vib_spectral_energy_proxy"] = vib_variance.round(5)

        return features


class LeakageSafeSplitter:
    """
    Guarantees strict train / validation / test isolation without future-data leakage.
    Enforces grouped splitting by mission_id or engine_id, or strictly causal temporal cutoffs.
    """

    @classmethod
    def split_by_mission(
        cls,
        data: pd.DataFrame,
        train_missions: List[str],
        test_missions: List[str],
        val_missions: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Split dataset by distinct mission identifiers.
        Ensures adjacent or continuous flights are not randomly split across sets.
        """
        if "mission_id" not in data.columns:
            raise ValueError("Dataframe must contain 'mission_id' column for mission-based splitting.")

        train_df = data[data["mission_id"].isin(train_missions)].copy()
        test_df = data[data["mission_id"].isin(test_missions)].copy()

        result = {"train": train_df, "test": test_df}
        if val_missions is not None:
            val_df = data[data["mission_id"].isin(val_missions)].copy()
            result["val"] = val_df

        cls.assert_no_mission_leakage(result)
        return result

    @classmethod
    def temporal_split(
        cls,
        data: pd.DataFrame,
        split_timestamp_s: float,
        buffer_window_s: float = 0.0,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split a single continuous trajectory at a specific temporal cutoff.

        Args:
            data: DataFrame containing 'timestamp'.
            split_timestamp_s: Cutoff time. Training data <= split_timestamp_s.
            buffer_window_s: Optional dead-band buffer to prevent rolling window leakage.

        Returns:
            Tuple of (train_df, test_df).
        """
        if "timestamp" not in data.columns:
            raise ValueError("Dataframe must contain 'timestamp' column for temporal splitting.")

        train_df = data[data["timestamp"] <= split_timestamp_s].copy()
        test_df = data[data["timestamp"] > (split_timestamp_s + buffer_window_s)].copy()
        return train_df, test_df

    @classmethod
    def assert_no_mission_leakage(cls, splits: Dict[str, pd.DataFrame]) -> None:
        """Verify zero overlap in mission_ids across splits."""
        seen: Dict[str, str] = {}
        for split_name, df in splits.items():
            if "mission_id" not in df.columns:
                continue
            for m_id in df["mission_id"].unique():
                if m_id in seen:
                    raise AssertionError(
                        f"Data leakage detected! Mission '{m_id}' appears in both '{seen[m_id]}' and '{split_name}' splits."
                    )
                seen[m_id] = split_name
