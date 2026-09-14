"""
Trajectory modeling and dual-horizon synthesis for Phase 11.
Combines Phase 10 short-horizon forecasts with robust Theil-Sen extended wear extrapolation.
Enforces Phase 10 handoff safety (rejects uncheckpointed local graph).
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from forecasting.schema import ForecastResult, ModelStatus
from prognostics.schema import RULConfig


class TheilSenExtrapolator:
    """
    Robust causal trend estimator using Theil-Sen median pairwise slope regression.
    Enforces actual timestamps, gap breaking, and minimum sample counts.
    """

    def __init__(self, window_s: float = 60.0, min_samples: int = 5, max_gap_s: float = 5.0):
        self.window_s = window_s
        self.min_samples = min_samples
        self.max_gap_s = max_gap_s

    def estimate_slope(
        self,
        timestamps: np.ndarray,
        health_values: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Estimate robust slope dHI/dt and its standard error over recent history.

        Args:
            timestamps: 1D array of actual telemetry timestamps (s)
            health_values: 1D array of smoothed Health Index values

        Returns:
            Tuple of (median_slope: float, slope_std_error: float).
            Returns (NaN, NaN) if points are insufficient or uncalibrated.
        """
        if len(timestamps) < self.min_samples or len(health_values) < self.min_samples:
            return float("nan"), float("nan")

        ts = np.asarray(timestamps, dtype=np.float64)
        hi = np.asarray(health_values, dtype=np.float64)

        # Filter out NaNs and non-finite values
        valid = np.isfinite(ts) & np.isfinite(hi)
        ts = ts[valid]
        hi = hi[valid]

        if len(ts) < self.min_samples:
            return float("nan"), float("nan")

        # Keep only the last window_s seconds
        t_current = ts[-1]
        t_cutoff = t_current - self.window_s
        in_window = ts >= t_cutoff
        ts = ts[in_window]
        hi = hi[in_window]

        if len(ts) < self.min_samples:
            return float("nan"), float("nan")

        # Check for timestamp gaps > max_gap_s; break history after last gap
        gaps = np.diff(ts)
        large_gaps = np.where(gaps > self.max_gap_s)[0]
        if len(large_gaps) > 0:
            last_break_idx = large_gaps[-1] + 1
            ts = ts[last_break_idx:]
            hi = hi[last_break_idx:]

        if len(ts) < self.min_samples:
            return float("nan"), float("nan")

        # Compute all pairwise slopes (i < j)
        pairwise_slopes = []
        n = len(ts)
        for i in range(n - 1):
            dt = ts[i + 1:] - ts[i]
            dhi = hi[i + 1:] - hi[i]
            # Avoid division by zero on identical timestamps
            valid_dt = dt > 1e-4
            if np.any(valid_dt):
                slopes = dhi[valid_dt] / dt[valid_dt]
                pairwise_slopes.extend(slopes)

        min_pairs = max(1, self.min_samples * (self.min_samples - 1) // 2) if self.min_samples > 2 else 1
        if len(pairwise_slopes) < min_pairs:
            return float("nan"), float("nan")

        slopes_arr = np.sort(np.asarray(pairwise_slopes, dtype=np.float64))
        median_slope = round(float(np.median(slopes_arr)), 7)

        N = len(slopes_arr)
        n_eff = (1.0 + np.sqrt(1.0 + 8.0 * N)) / 2.0
        # Sen (1968) exact Kendall rank variance accounting for pairwise dependencies
        # V = (1 / 18) * [n_eff * (n_eff - 1) * (2*n_eff + 5) - sum_t t * (t - 1) * (2t + 5)]
        _, tie_counts = np.unique(hi, return_counts=True)
        tie_sum = np.sum(tie_counts * (tie_counts - 1) * (2 * tie_counts + 5))
        var_k = (n_eff * (n_eff - 1.0) * (2.0 * n_eff + 5.0) - tie_sum) / 18.0
        sigma_k = np.sqrt(max(0.0, var_k))

        # Standard error estimation via 1-sigma rank inversion (z = 1.0)
        z_1sig = 1.0
        margin_1sig = z_1sig * sigma_k
        m1_1sig = max(0, int(np.floor((N - margin_1sig) / 2.0)))
        m2_1sig = min(N - 1, int(np.ceil((N + margin_1sig) / 2.0)))

        rank_diff = slopes_arr[m2_1sig] - slopes_arr[m1_1sig]
        if rank_diff > 0.0:
            slope_se = round(float(rank_diff / (2.0 * z_1sig)), 7)
        else:
            # When slopes are identical, fall back to robust residual dispersion
            alpha_res = float(np.median(hi - median_slope * ts))
            residuals = hi - (alpha_res + median_slope * ts)
            mad_resid = float(np.median(np.abs(residuals - np.median(residuals))))
            s_tt = float(np.sum((ts - np.mean(ts)) ** 2))
            if s_tt > 1e-6 and mad_resid > 0.0:
                resid_se = np.sqrt(np.pi / 3.0) * (1.4826 * mad_resid) / np.sqrt(s_tt)
                slope_se = round(float(resid_se), 7)
            else:
                slope_se = 1e-6

        return median_slope, max(1e-6, slope_se)

    def estimate_slope_interval(
        self,
        timestamps: np.ndarray,
        health_values: np.ndarray,
        alpha: float = 0.90,
    ) -> Tuple[float, float, float, float]:
        """
        Estimate robust Theil-Sen slope, empirical confidence bounds, and standard error.
        Derived from Sen (1968) nonparametric rank inversion over pairwise slope combinations.

        Args:
            timestamps: 1D array of actual telemetry timestamps (s)
            health_values: 1D array of smoothed Health Index values
            alpha: Nominal confidence level (default 0.90 for P05 / P95 bounds)

        Returns:
            Tuple of (median_slope: float, p05_slope: float, p95_slope: float, slope_se: float).
            Returns (NaN, NaN, NaN, NaN) if points are insufficient.
        """
        if len(timestamps) < self.min_samples or len(health_values) < self.min_samples:
            return float("nan"), float("nan"), float("nan"), float("nan")

        ts = np.asarray(timestamps, dtype=np.float64)
        hi = np.asarray(health_values, dtype=np.float64)

        valid = np.isfinite(ts) & np.isfinite(hi)
        ts = ts[valid]
        hi = hi[valid]

        if len(ts) < self.min_samples:
            return float("nan"), float("nan"), float("nan"), float("nan")

        t_current = ts[-1]
        t_cutoff = t_current - self.window_s
        in_window = ts >= t_cutoff
        ts = ts[in_window]
        hi = hi[in_window]

        if len(ts) < self.min_samples:
            return float("nan"), float("nan"), float("nan"), float("nan")

        gaps = np.diff(ts)
        large_gaps = np.where(gaps > self.max_gap_s)[0]
        if len(large_gaps) > 0:
            last_break_idx = large_gaps[-1] + 1
            ts = ts[last_break_idx:]
            hi = hi[last_break_idx:]

        if len(ts) < self.min_samples:
            return float("nan"), float("nan"), float("nan"), float("nan")

        pairwise_slopes = []
        n = len(ts)
        for i in range(n - 1):
            dt = ts[i + 1:] - ts[i]
            dhi = hi[i + 1:] - hi[i]
            valid_dt = dt > 1e-4
            if np.any(valid_dt):
                slopes = dhi[valid_dt] / dt[valid_dt]
                pairwise_slopes.extend(slopes)

        min_pairs = max(1, self.min_samples * (self.min_samples - 1) // 2) if self.min_samples > 2 else 1
        if len(pairwise_slopes) < min_pairs:
            return float("nan"), float("nan"), float("nan"), float("nan")

        slopes_arr = np.sort(np.asarray(pairwise_slopes, dtype=np.float64))
        median_slope = round(float(np.median(slopes_arr)), 7)

        N = len(slopes_arr)
        n_eff = (1.0 + np.sqrt(1.0 + 8.0 * N)) / 2.0
        _, tie_counts = np.unique(hi, return_counts=True)
        tie_sum = np.sum(tie_counts * (tie_counts - 1) * (2 * tie_counts + 5))
        var_k = (n_eff * (n_eff - 1.0) * (2.0 * n_eff + 5.0) - tie_sum) / 18.0
        sigma_k = np.sqrt(max(0.0, var_k))

        # Nominal confidence bounds (alpha=0.90 -> z_alpha = 1.6448536 for P05 / P95)
        # or general z = sqrt(2) * erfinv(alpha)
        from scipy.special import erfinv
        z_alpha = float(np.sqrt(2.0) * erfinv(alpha)) if 0.0 < alpha < 1.0 else 1.6448536
        margin_alpha = z_alpha * sigma_k
        m1_alpha = max(0, int(np.floor((N - margin_alpha) / 2.0)))
        m2_alpha = min(N - 1, int(np.ceil((N + margin_alpha) / 2.0)))
        p05_slope = round(float(slopes_arr[m1_alpha]), 7)
        p95_slope = round(float(slopes_arr[m2_alpha]), 7)

        # 1-sigma standard error
        z_1sig = 1.0
        margin_1sig = z_1sig * sigma_k
        m1_1sig = max(0, int(np.floor((N - margin_1sig) / 2.0)))
        m2_1sig = min(N - 1, int(np.ceil((N + margin_1sig) / 2.0)))

        rank_diff = slopes_arr[m2_1sig] - slopes_arr[m1_1sig]
        if rank_diff > 0.0:
            slope_se = round(float(rank_diff / (2.0 * z_1sig)), 7)
        else:
            alpha_res = float(np.median(hi - median_slope * ts))
            residuals = hi - (alpha_res + median_slope * ts)
            mad_resid = float(np.median(np.abs(residuals - np.median(residuals))))
            s_tt = float(np.sum((ts - np.mean(ts)) ** 2))
            if s_tt > 1e-6 and mad_resid > 0.0:
                resid_se = np.sqrt(np.pi / 3.0) * (1.4826 * mad_resid) / np.sqrt(s_tt)
                slope_se = round(float(resid_se), 7)
            else:
                slope_se = 1e-6

        return median_slope, p05_slope, p95_slope, max(1e-6, slope_se)


class DualHorizonSynthesizer:
    """
    Synthesizes short-horizon forecasts and extended-horizon wear projections.
    Enforces strict handoff safety: rejects LOCAL_UNCHECKPOINTED_GRAPH.
    """

    def __init__(self, config: Optional[RULConfig] = None):
        self.config = config or RULConfig()
        self.extrapolator = TheilSenExtrapolator(
            window_s=self.config.theil_sen_window_s,
            min_samples=self.config.theil_sen_min_samples,
            max_gap_s=self.config.theil_sen_max_gap_s,
        )

    def evaluate_phase10_handoff(
        self,
        forecast: Optional[ForecastResult],
        allow_baseline: bool = True,
    ) -> Tuple[float, str]:
        """
        Evaluate Phase 10 forecast usability and determine short-horizon length H.

        Rules:
        - LOADED_PRETRAINED + horizon in {16, 32}: usable forecast (H = 16 or 32)
        - LOCAL_UNCHECKPOINTED_GRAPH: REJECTED (H = 0)
        - BLOCKED_UNAUTHENTICATED_GATED: REJECTED (H = 0)
        - BASELINE: usable ONLY if allow_baseline is True, labeled BASELINE_EWMA (H = 16 or 32)
        """
        if forecast is None:
            return 0.0, "ROBUST_LINEAR_PRIMARY"

        status = forecast.model_status
        horizon = float(forecast.forecast_horizon)

        if status == ModelStatus.LOADED_PRETRAINED and horizon in {16.0, 32.0}:
            return horizon, "TIMESFM_FORECAST_ASSISTED"

        if status == ModelStatus.BASELINE and allow_baseline and horizon in {16.0, 32.0}:
            return horizon, "BASELINE_EWMA_ASSISTED"

        # Explicitly reject local uncheckpointed graph and gated status from influencing RUL
        return 0.0, "ROBUST_LINEAR_PRIMARY"

    def synthesize_trajectory(
        self,
        current_time: float,
        current_hi: float,
        history_timestamps: np.ndarray,
        history_hi: np.ndarray,
        forecast: Optional[ForecastResult] = None,
        max_projection_s: float = 86400.0,
    ) -> Dict[str, Any]:
        """
        Construct dual-horizon projected trajectory.

        Returns:
            Dict containing handoff_horizon, anchor_time, anchor_hi, theil_sen_slope,
            slope_se, trajectory_type, and analytical EOL crossing times.
        """
        # 1. Check Phase 10 handoff safety
        H, traj_type = self.evaluate_phase10_handoff(forecast)

        # 2. Estimate robust historical slope via Theil-Sen
        slope, slope_se = self.extrapolator.estimate_slope(history_timestamps, history_hi)

        # 3. Determine Anchor Point at t + H
        if H > 0.0 and forecast is not None and forecast.predicted_telemetry:
            anchor_t = current_time + H
            # In forecast-assisted mode, calculate projected HI at t + H
            # If Phase 10 provides projected health, use it; otherwise project using slope
            proj_hi = getattr(forecast, "projected_health_trajectory", None) or getattr(forecast, "projected_health_index", None)
            if proj_hi and len(proj_hi) > 0 and np.isfinite(proj_hi[-1]):
                candidate_anchor = float(proj_hi[-1])
                # Physical consistency guard: if engine is actively degrading (slope < -0.0005),
                # anchor HI cannot spontaneously jump higher than current_hi unless recovering
                if slope < -0.0005 and candidate_anchor > current_hi:
                    anchor_hi = max(0.0, min(1.0, current_hi + slope * H))
                else:
                    anchor_hi = max(0.0, min(1.0, candidate_anchor))
            else:
                anchor_hi = max(0.0, min(1.0, current_hi + (slope if np.isfinite(slope) else 0.0) * H))
        else:
            anchor_t = current_time
            anchor_hi = max(0.0, min(1.0, current_hi))
            H = 0.0

        return {
            "handoff_horizon_s": H,
            "anchor_time": anchor_t,
            "anchor_hi": anchor_hi,
            "theil_sen_slope": slope,
            "slope_std_error": slope_se,
            "trajectory_type": traj_type,
        }

    def solve_linear_crossing(
        self,
        anchor_time: float,
        anchor_hi: float,
        slope: float,
        target_hi: float = 0.35,
        current_time: float = 0.0,
        tolerance: float = 1e-5,
    ) -> Optional[float]:
        """
        Analytically solve for time-to-crossing:
        anchor_hi + slope * (t_cross - anchor_time) = target_hi
        """
        if not np.isfinite(slope) or slope >= -1e-5:
            return None

        # slope is negative
        delta_hi = (target_hi + tolerance) - anchor_hi
        if delta_hi >= 0.0:
            # Already at or below target within numerical tolerance
            return max(0.0, anchor_time - current_time)

        tau_from_anchor = (target_hi - anchor_hi) / slope  # (negative / negative = positive)
        t_cross = anchor_time + tau_from_anchor
        return max(0.0, t_cross - current_time)
