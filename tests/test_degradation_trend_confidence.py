"""
Mathematical verification and regression suite for degradation trend slope and confidence bounds.
Verifies Sen (1968) exact Kendall rank variance and rank inversion across small, normal,
constant, zero, noisy, rapidly changing, and insufficient telemetry samples.
"""

import math
import numpy as np
import pytest

from prognostics.trajectory import TheilSenExtrapolator
from prognostics.uncertainty import MonteCarloTrajectoryPropagator


# =========================================================================
# 1. SMALL SAMPLE TESTS (2, 3, 5 POINTS)
# =========================================================================

def test_small_sample_two_points():
    """Verify 2-point behavior with explicit min_samples=2 and default min_samples=5."""
    # With default min_samples=5, 2 points must return NaN
    extrapolator_default = TheilSenExtrapolator(min_samples=5)
    ts = np.array([0.0, 1.0])
    hi = np.array([1.0, 0.98])
    slope_def, se_def = extrapolator_default.estimate_slope(ts, hi)
    assert np.isnan(slope_def) and np.isnan(se_def)

    # With min_samples=2, exactly 1 pairwise slope S_01 = -0.02 is formed
    extrapolator_2 = TheilSenExtrapolator(min_samples=2)
    slope_2, se_2 = extrapolator_2.estimate_slope(ts, hi)
    assert math.isclose(slope_2, -0.02, abs_tol=1e-6)
    # Degrees of freedom is 0, so SE should cleanly clamp to floor 1e-6
    assert math.isclose(se_2, 1e-6, abs_tol=1e-7)


def test_small_sample_three_points():
    """Verify 3-point dataset with hand-calculated exact Sen rank statistics."""
    # Data: t = [0.0, 1.0, 2.0], hi = [1.00, 0.98, 0.95]
    # S_01 = (0.98 - 1.00) / 1.0 = -0.02
    # S_12 = (0.95 - 0.98) / 1.0 = -0.03
    # S_02 = (0.95 - 1.00) / 2.0 = -0.025
    # Sorted slopes: [-0.030, -0.025, -0.020]
    # Median = -0.025
    # Kendall V = 3 * 2 * 11 / 18 = 3.6667, sqrt(V) = 1.9149
    # z=1.0: margin = 1.9149 -> M1 = floor((3 - 1.9149)/2) = 0, M2 = ceil((3 + 1.9149)/2) = 2
    # SE = (S_2 - S_0) / 2.0 = (-0.02 - (-0.03)) / 2.0 = 0.0050
    extrapolator_3 = TheilSenExtrapolator(min_samples=3)
    ts = np.array([0.0, 1.0, 2.0])
    hi = np.array([1.00, 0.98, 0.95])
    slope, se = extrapolator_3.estimate_slope(ts, hi)

    assert math.isclose(slope, -0.025, abs_tol=1e-6)
    assert math.isclose(se, 0.0050, abs_tol=1e-4)

    med, p05, p95, se_int = extrapolator_3.estimate_slope_interval(ts, hi, alpha=0.90)
    assert math.isclose(med, -0.025, abs_tol=1e-6)
    assert p05 <= med <= p95


def test_small_sample_five_points():
    """Verify 5-point dataset against independently calculated Sen statistics."""
    # Data: t = [0, 1, 2, 3, 4], hi = [1.00, 0.98, 0.95, 0.93, 0.90]
    # N = 10 slopes
    # Median slope = -0.025
    # Kendall V = 5 * 4 * 15 / 18 = 300 / 18 = 16.6667, sqrt(V) = 4.0825
    # z=1.0: margin = 4.0825 -> M1 = floor((10 - 4.0825)/2) = 2, M2 = ceil((10 + 4.0825)/2) = 8
    # S = [-0.030, -0.030, -0.0266667, -0.025, -0.025, -0.025, -0.025, -0.0233333, -0.020, -0.020]
    # S[2] = -0.0266667, S[8] = -0.0200
    # Expected SE = (-0.020 - (-0.0266667)) / 2.0 = 0.0033333
    extrapolator_5 = TheilSenExtrapolator(min_samples=5)
    ts = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    hi = np.array([1.00, 0.98, 0.95, 0.93, 0.90])
    slope, se = extrapolator_5.estimate_slope(ts, hi)

    assert math.isclose(slope, -0.025, abs_tol=1e-5)
    assert math.isclose(se, 0.0033333, rel_tol=1e-2)

    med, p05, p95, se_int = extrapolator_5.estimate_slope_interval(ts, hi, alpha=0.90)
    assert math.isclose(med, -0.025, abs_tol=1e-5)
    assert p05 <= med <= p95


# =========================================================================
# 2. NORMAL SAMPLE TESTS (10, 20, 60 POINTS)
# =========================================================================

def test_normal_sample_ten_points():
    """Verify 10-point dataset preserves monotonicity and expected noise scaling."""
    ts = np.arange(10, dtype=np.float64)
    np.random.seed(123)
    hi = 1.0 - 0.002 * ts + np.array([0.001, -0.002, 0.0015, -0.001, 0.0005, -0.0015, 0.002, -0.0005, 0.001, -0.001])

    extrapolator = TheilSenExtrapolator(min_samples=5)
    slope, se = extrapolator.estimate_slope(ts, hi)

    assert math.isclose(slope, -0.002, abs_tol=5e-4)
    # Expected SE from Sen rank inversion is ~ 0.000208
    assert 0.00010 < se < 0.00040

    med, p05, p95, se_int = extrapolator.estimate_slope_interval(ts, hi, alpha=0.90)
    assert p05 <= med <= p95
    assert math.isclose(se, se_int, abs_tol=1e-7)


def test_normal_sample_twenty_and_sixty_points():
    """Verify SE shrinks proportionally to O(n^-1.5) over expanding duration."""
    extrapolator = TheilSenExtrapolator(min_samples=5, window_s=100.0)

    # 20 points
    t20 = np.arange(20, dtype=np.float64)
    np.random.seed(42)
    noise20 = np.random.normal(0.0, 0.002, size=20)
    hi20 = 1.0 - 0.0015 * t20 + noise20
    slope20, se20 = extrapolator.estimate_slope(t20, hi20)

    # 60 points with identical noise scale
    t60 = np.arange(60, dtype=np.float64)
    np.random.seed(42)
    noise60 = np.random.normal(0.0, 0.002, size=60)
    hi60 = 1.0 - 0.0015 * t60 + noise60
    slope60, se60 = extrapolator.estimate_slope(t60, hi60)

    assert math.isclose(slope20, -0.0015, abs_tol=5e-4)
    assert math.isclose(slope60, -0.0015, abs_tol=5e-4)
    # Uncertainty must strictly decrease with longer observation window
    assert se60 < se20
    # Expected scaling factor: (60/20)^1.5 = 3^1.5 = ~5.19
    scaling_ratio = se20 / se60
    assert 3.0 < scaling_ratio < 7.0


# =========================================================================
# 3. CONSTANT & ZERO SLOPE TESTS
# =========================================================================

def test_constant_slope():
    """Verify perfect linear degradation collapses standard error cleanly to floor."""
    extrapolator = TheilSenExtrapolator(min_samples=5)
    ts = np.linspace(0.0, 50.0, 25)
    hi = 1.0 - 0.004 * ts  # slope exactly -0.004 s^-1
    slope, se = extrapolator.estimate_slope(ts, hi)

    assert math.isclose(slope, -0.004, abs_tol=1e-7)
    assert math.isclose(se, 1e-6, abs_tol=1e-7)

    med, p05, p95, _ = extrapolator.estimate_slope_interval(ts, hi, alpha=0.90)
    assert math.isclose(p05, -0.004, abs_tol=1e-6)
    assert math.isclose(p95, -0.004, abs_tol=1e-6)


def test_zero_slope():
    """Verify steady-state telemetry yields zero slope and minimal SE floor."""
    extrapolator = TheilSenExtrapolator(min_samples=5)
    ts = np.linspace(100.0, 150.0, 30)
    hi = np.full(30, 0.92)  # constant HI
    slope, se = extrapolator.estimate_slope(ts, hi)

    assert math.isclose(slope, 0.0, abs_tol=1e-7)
    assert math.isclose(se, 1e-6, abs_tol=1e-7)


# =========================================================================
# 4. NOISY & RAPIDLY CHANGING SLOPE TESTS
# =========================================================================

def test_noisy_slope_robustness():
    """Verify heavy outliers do not distort the median slope, and SE reflects true dispersion."""
    extrapolator = TheilSenExtrapolator(min_samples=5)
    ts = np.arange(30, dtype=np.float64)

    # 1. Continuous Gaussian noise plus severe intermittent outliers
    np.random.seed(42)
    hi_noisy = 1.0 - 0.002 * ts + np.random.normal(0.0, 0.003, size=30)
    hi_noisy[5] += 0.15
    hi_noisy[15] -= 0.20
    hi_noisy[25] += 0.18

    slope, se = extrapolator.estimate_slope(ts, hi_noisy)
    # Theil-Sen breakdown point is ~29%, so it must remain close to -0.002
    assert math.isclose(slope, -0.002, abs_tol=5e-4)
    assert 1e-5 < se < 1e-2

    # 2. Pure collinear line with 3 isolated impulse spikes (27/30 points exactly collinear)
    hi_impulse = 1.0 - 0.002 * ts
    hi_impulse[5] += 0.15
    hi_impulse[15] -= 0.20
    hi_impulse[25] += 0.18
    slope_imp, se_imp = extrapolator.estimate_slope(ts, hi_impulse)
    assert math.isclose(slope_imp, -0.002, abs_tol=1e-6)
    # With 80% identical slopes, dispersion collapses to floor
    assert math.isclose(se_imp, 1e-6, abs_tol=1e-7)


def test_rapidly_changing_slope():
    """Verify accelerating degradation increases standard error appropriately."""
    extrapolator = TheilSenExtrapolator(min_samples=5)
    ts = np.linspace(0.0, 40.0, 41)
    # Quadratic wear: dHI/dt = -0.0002 * t
    hi_quadratic = 1.0 - 0.0001 * (ts ** 2)

    slope, se = extrapolator.estimate_slope(ts, hi_quadratic)
    assert slope < 0.0
    # Dispersion across slopes must be substantially higher than constant slope floor
    assert se > 0.0001

    med, p05, p95, _ = extrapolator.estimate_slope_interval(ts, hi_quadratic, alpha=0.90)
    assert p05 < med < p95


# =========================================================================
# 5. INSUFFICIENT DATA & ADVERSARIAL CASES
# =========================================================================

def test_insufficient_data_fewer_than_min_samples():
    """Verify fewer points than min_samples cleanly returns NaN."""
    extrapolator = TheilSenExtrapolator(min_samples=5)
    ts = np.array([1.0, 2.0, 3.0, 4.0])
    hi = np.array([0.9, 0.89, 0.88, 0.87])
    slope, se = extrapolator.estimate_slope(ts, hi)
    assert np.isnan(slope) and np.isnan(se)


def test_insufficient_data_excessive_gap():
    """Verify history breaks upon timestamp gap exceeding max_gap_s."""
    extrapolator = TheilSenExtrapolator(min_samples=5, max_gap_s=5.0)
    # 4 points, gap of 10s, then 3 points -> after gap, only 3 points remain (< 5)
    ts = np.array([1.0, 2.0, 3.0, 4.0, 14.0, 15.0, 16.0])
    hi = np.array([0.9, 0.89, 0.88, 0.87, 0.86, 0.85, 0.84])
    slope, se = extrapolator.estimate_slope(ts, hi)
    assert np.isnan(slope) and np.isnan(se)


def test_adversarial_nan_and_inf_handling():
    """Verify arrays containing NaNs or Infs are filtered before sample check."""
    extrapolator = TheilSenExtrapolator(min_samples=5)
    ts = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0])
    hi = np.array([0.90, np.inf, 0.88, 0.87, 0.86, 0.85])
    slope, se = extrapolator.estimate_slope(ts, hi)
    # Valid points: (1, 0.90), (4, 0.87), (5, 0.86), (6, 0.85) -> only 4 valid (< 5)
    assert np.isnan(slope) and np.isnan(se)


# =========================================================================
# 6. MONTE CARLO INTEGRATION & P05 / P50 / P95 ORDERING
# =========================================================================

def test_monte_carlo_propagation_with_exact_se():
    """Verify standard error directly generates valid P05 <= P50 <= P95 RUL distribution."""
    extrapolator = TheilSenExtrapolator(min_samples=5)
    ts = np.linspace(0.0, 30.0, 31)
    np.random.seed(99)
    hi = 0.80 - 0.001 * ts + np.random.normal(0, 0.001, 31)
    slope, se = extrapolator.estimate_slope(ts, hi)

    propagator = MonteCarloTrajectoryPropagator(seed=42)
    p50, p05, p95, samples = propagator.propagate_linear_trajectory(
        anchor_time=30.0,
        anchor_hi=hi[-1],
        median_slope=slope,
        slope_se=se,
        current_time=30.0,
        m_samples=500,
    )

    assert p50 is not None and p05 is not None and p95 is not None
    assert p05 <= p50 <= p95
    # Theoretical RUL crossing: (0.77 - 0.35) / 0.001 = ~420 s
    assert 300.0 < p50 < 550.0
    assert len(samples) == 500
    assert np.all(samples >= 0.0)
