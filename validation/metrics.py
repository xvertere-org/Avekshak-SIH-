"""
Validation Metrics and Evaluation Utilities for SIH26054 Engine Simulator.

Provides quantitative evaluators for:
- Physical bounds and operating envelope verification
- Monotonicity checks with noise tolerance
- Transient settling time and dynamic response hierarchy
- Channel-specific steady-state stability analysis
- Cross-channel coherence and correlation
- Vibration harmonic order peak tracking via FFT
"""

import math
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import pandas as pd


def check_bounds(
    values: Union[List[float], np.ndarray, pd.Series],
    lower_valid: float,
    upper_valid: float,
    lower_warning: Optional[float] = None,
    upper_warning: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Check if a series of values stays within physical validity and operational warning envelopes.

    Returns:
        Dict with min, max, validity pass/fail, and warning pass/fail.
    """
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {"pass_validity": True, "pass_warning": True, "count": 0}

    min_val = float(np.min(arr))
    max_val = float(np.max(arr))

    has_nan = bool(np.isnan(arr).any())
    has_inf = bool(np.isinf(arr).any())

    valid_violations = int(np.sum((arr < lower_valid) | (arr > upper_valid)))
    pass_validity = (valid_violations == 0) and not has_nan and not has_inf

    warn_violations = 0
    pass_warning = True
    if lower_warning is not None and upper_warning is not None:
        warn_violations = int(np.sum((arr < lower_warning) | (arr > upper_warning)))
        pass_warning = (warn_violations == 0)

    return {
        "pass_validity": pass_validity,
        "pass_warning": pass_warning,
        "min": min_val,
        "max": max_val,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "valid_violations": valid_violations,
        "warning_violations": warn_violations,
        "lower_valid": lower_valid,
        "upper_valid": upper_valid,
        "lower_warning": lower_warning,
        "upper_warning": upper_warning,
    }


def check_monotonicity(
    x: Union[List[float], np.ndarray],
    y: Union[List[float], np.ndarray],
    expected_direction: str = "increasing",
    tolerance: float = 1e-4,
) -> Dict[str, Any]:
    """
    Verify directional monotonicity between independent variable x and response y.

    Args:
        x: Independent variable array (e.g. throttle, altitude, RPM)
        y: Dependent response array (e.g. power, density, pressure)
        expected_direction: 'increasing' (dy/dx >= 0) or 'decreasing' (dy/dx <= 0)
        tolerance: Allowed noise slack for near-flat slopes
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    # Sort by x
    sort_idx = np.argsort(x_arr)
    x_sorted = x_arr[sort_idx]
    y_sorted = y_arr[sort_idx]

    dy = np.diff(y_sorted)

    if expected_direction == "increasing":
        violations = int(np.sum(dy < -abs(tolerance)))
    elif expected_direction == "decreasing":
        violations = int(np.sum(dy > abs(tolerance)))
    else:
        raise ValueError(f"Unknown expected_direction: {expected_direction}")

    pass_monotonic = (violations == 0)
    monotonic_ratio = 1.0 - (violations / max(1, len(dy)))

    return {
        "pass_monotonic": pass_monotonic,
        "violations": violations,
        "total_intervals": len(dy),
        "monotonic_ratio": monotonic_ratio,
        "expected_direction": expected_direction,
    }


def compute_settling_time(
    time_series: Union[List[float], np.ndarray],
    value_series: Union[List[float], np.ndarray],
    step_time: float,
    band_pct: float = 0.02,
) -> float:
    """
    Compute 2% settling time following a step change.
    Settling time is the elapsed time from step_time until the signal permanently
    enters and remains within ±(band_pct * delta) of its final steady-state value.
    """
    t = np.asarray(time_series, dtype=float)
    v = np.asarray(value_series, dtype=float)

    mask_after = t >= step_time
    if not np.any(mask_after):
        return 0.0

    t_post = t[mask_after]
    v_post = v[mask_after]

    final_val = float(np.mean(v_post[-max(5, int(len(v_post) * 0.1)):]))
    init_val = float(v_post[0])
    delta = abs(final_val - init_val)

    if delta < 1e-4:
        return 0.0

    band = max(abs(band_pct * delta), 0.01)

    # Search backwards for the last time the signal was outside the band
    outside_indices = np.where(np.abs(v_post - final_val) > band)[0]
    if len(outside_indices) == 0:
        return 0.0

    last_outside_idx = outside_indices[-1]
    if last_outside_idx + 1 < len(t_post):
        settle_time = float(t_post[last_outside_idx + 1] - step_time)
    else:
        settle_time = float(t_post[-1] - step_time)

    return max(0.0, settle_time)


def compute_channel_stability(
    values: Union[List[float], np.ndarray],
    dt: float = 0.1,
    window_fraction: float = 0.25,
) -> Dict[str, float]:
    """
    Compute steady-state stability metrics over the trailing window of a simulation:
    - mean
    - standard deviation (std)
    - coefficient of variation (std / |mean|)
    - linear drift rate (slope in units per second)
    """
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "cov": 0.0, "drift_rate": 0.0}

    window_size = max(5, int(len(arr) * window_fraction))
    window = arr[-window_size:]

    mean_val = float(np.mean(window))
    std_val = float(np.std(window))
    cov = float(std_val / abs(mean_val)) if abs(mean_val) > 1e-5 else 0.0

    # Fit linear slope over time
    t_win = np.arange(window_size) * dt
    if window_size > 1:
        slope, _ = np.polyfit(t_win, window, 1)
        drift_rate = float(slope)
    else:
        drift_rate = 0.0

    return {
        "mean": mean_val,
        "std": std_val,
        "cov": cov,
        "drift_rate": drift_rate,
        "window_samples": window_size,
    }


def compute_cross_channel_correlations(df: pd.DataFrame, pairs: List[Tuple[str, str]]) -> Dict[str, float]:
    """
    Compute Pearson correlation coefficients between designated channel pairs.
    """
    correlations = {}
    for ch1, ch2 in pairs:
        key = f"{ch1}_vs_{ch2}"
        if ch1 in df.columns and ch2 in df.columns:
            corr = float(df[ch1].corr(df[ch2]))
            correlations[key] = corr if not math.isnan(corr) else 0.0
        else:
            correlations[key] = 0.0
    return correlations


def compute_vibration_order_metrics(
    signal: np.ndarray,
    fs: float,
    rpm: float,
    expected_orders: Tuple[float, ...] = (1.0, 2.0),
) -> Dict[str, Any]:
    """
    Compute FFT spectrum and evaluate harmonic order peak frequencies and relative errors.
    """
    n = len(signal)
    fft_vals = np.abs(np.fft.rfft(signal)) * (2.0 / n)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    # Exclude DC offset (< 5 Hz)
    mask = freqs >= 5.0
    valid_freqs = freqs[mask]
    valid_fft = fft_vals[mask]

    f_rot = max(0.0, rpm) / 60.0
    order_results = {}

    for order in expected_orders:
        expected_f = order * f_rot
        # Search in a ±20% window around expected order frequency
        f_min = expected_f * 0.8
        f_max = expected_f * 1.2
        win_mask = (valid_freqs >= f_min) & (valid_freqs <= f_max)

        if np.any(win_mask):
            sub_freqs = valid_freqs[win_mask]
            sub_fft = valid_fft[win_mask]
            peak_idx = int(np.argmax(sub_fft))
            detected_f = float(sub_freqs[peak_idx])
            peak_amp = float(sub_fft[peak_idx])
            freq_error = abs(detected_f - expected_f)
            freq_error_pct = (freq_error / expected_f) * 100.0 if expected_f > 0 else 0.0
        else:
            detected_f = 0.0
            peak_amp = 0.0
            freq_error = expected_f
            freq_error_pct = 100.0

        order_results[f"order_{order}x"] = {
            "expected_hz": round(expected_f, 2),
            "detected_hz": round(detected_f, 2),
            "amplitude_g": round(peak_amp, 4),
            "error_hz": round(freq_error, 2),
            "error_pct": round(freq_error_pct, 2),
            "pass": freq_error_pct < 4.0,  # within 4% of target harmonic frequency
        }

    rms_val = float(np.sqrt(np.mean(signal ** 2)))
    dominant_peak_idx = int(np.argmax(valid_fft))
    dominant_f = float(valid_freqs[dominant_peak_idx])

    return {
        "f_rot_hz": f_rot,
        "dominant_freq_hz": dominant_f,
        "rms_g": rms_val,
        "orders": order_results,
    }
