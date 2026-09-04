"""
Deterministic reference baselines for Phase 10 telemetry trajectory forecasting:
1. PersistenceForecaster (Last Value)
2. CausalEWMAForecaster (Causal Trend Extrapolator using actual timestamps)
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from forecasting.schema import (
    DEFAULT_FORECAST_CHANNELS,
    PHYSICAL_LOWER_BOUNDS,
    ForecastingConfig,
)


class PersistenceForecaster:
    """
    Persistence / Last-Value Baseline:
    y_hat(t + k) = y(t) for all k in [1, horizon].
    """

    def __init__(self, name: str = "persistence"):
        self.name = name

    def predict(
        self,
        context_2d: np.ndarray,
        horizon: int,
        target_channels: Optional[List[str]] = None,
        timestamps: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate persistence forecast from context.

        Args:
            context_2d: Array of shape (num_channels, context_len)
            horizon: Future forecast steps
            target_channels: Optional list of channel names matching rows of context_2d
            timestamps: Optional array of timestamps (unused in persistence)

        Returns:
            Dict mapping channel_name -> np.ndarray of shape (horizon,)
        """
        channels = target_channels or DEFAULT_FORECAST_CHANNELS
        num_channels, _ = context_2d.shape

        forecasts: Dict[str, np.ndarray] = {}
        for i in range(num_channels):
            ch = channels[i] if i < len(channels) else f"channel_{i}"
            last_val = float(context_2d[i, -1])
            forecasts[ch] = np.full((horizon,), last_val, dtype=np.float32)

        return forecasts


class CausalEWMAForecaster:
    """
    Causal EWMA & Trend Extrapolator Baseline:
    y_hat(t + k) = EWMA(t) + beta * (t_future_k - t_last)

    Causal slope beta is estimated via Ordinary Least Squares (OLS) over the context
    using actual elapsed timestamps.

    Robustness Guarantees:
    - Uses actual timestamps, not sample indices.
    - Zero/near-zero denominator in OLS (e.g. constant timestamps) deterministically sets beta = 0.0.
    - Zero NaNs, zero ZeroDivisionError exceptions.
    - Configurable physical non-negative clamping with is_clamped tracking.
    """

    def __init__(
        self,
        alpha: float = 0.15,
        name: str = "causal_ewma",
        clamp_to_physical_limits: bool = True,
    ):
        self.name = name
        self.alpha = alpha
        self.clamp_to_physical_limits = clamp_to_physical_limits

    def predict(
        self,
        context_2d: np.ndarray,
        horizon: int,
        target_channels: Optional[List[str]] = None,
        timestamps: Optional[np.ndarray] = None,
        sampling_interval_s: float = 1.0,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, bool]]:
        """
        Generate causal trend extrapolation forecast using actual timestamps.

        Args:
            context_2d: Array of shape (num_channels, context_len)
            horizon: Future forecast steps
            target_channels: Optional list of channel names matching rows of context_2d
            timestamps: Array of shape (context_len,) with actual historical timestamps
            sampling_interval_s: Nominal interval for future projection steps

        Returns:
            (forecasts_dict, clamping_flags_dict):
            - forecasts_dict: channel_name -> np.ndarray of shape (horizon,)
            - clamping_flags_dict: channel_name -> bool indicating if physical clamping was applied
        """
        channels = target_channels or DEFAULT_FORECAST_CHANNELS
        num_channels, context_len = context_2d.shape

        # Build or use actual historical timestamps
        if timestamps is not None and len(timestamps) == context_len:
            t_hist = np.array(timestamps, dtype=np.float64)
        else:
            t_hist = np.arange(context_len, dtype=np.float64) * sampling_interval_s

        last_t = t_hist[-1]
        t_mean = float(np.mean(t_hist))
        t_diff = t_hist - t_mean
        t_var = float(np.sum(t_diff ** 2))

        # Future timestamps for projection
        future_dt = np.arange(1, horizon + 1, dtype=np.float64) * sampling_interval_s

        forecasts: Dict[str, np.ndarray] = {}
        clamping_flags: Dict[str, bool] = {}

        for i in range(num_channels):
            ch = channels[i] if i < len(channels) else f"channel_{i}"
            y_hist = context_2d[i].astype(np.float64)

            # 1. Causal EWMA computation over historical window
            ewma_val = float(y_hist[0])
            for val in y_hist[1:]:
                ewma_val = self.alpha * float(val) + (1.0 - self.alpha) * ewma_val

            # 2. Causal Slope beta via OLS using actual timestamps
            # If denominator is near-zero (e.g. constant timestamps), slope is deterministically 0.0
            if t_var > 1e-9:
                y_mean = float(np.mean(y_hist))
                cov_ty = float(np.sum(t_diff * (y_hist - y_mean)))
                beta = cov_ty / t_var
            else:
                beta = 0.0

            # 3. Future extrapolation
            raw_pred = ewma_val + beta * future_dt

            # 4. Physical sanity clamping (e.g. RPM >= 0)
            was_clamped = False
            if self.clamp_to_physical_limits and ch in PHYSICAL_LOWER_BOUNDS:
                lower_limit = PHYSICAL_LOWER_BOUNDS[ch]
                if np.any(raw_pred < lower_limit):
                    raw_pred = np.maximum(raw_pred, lower_limit)
                    was_clamped = True

            forecasts[ch] = raw_pred.astype(np.float32)
            clamping_flags[ch] = was_clamped

        return forecasts, clamping_flags
