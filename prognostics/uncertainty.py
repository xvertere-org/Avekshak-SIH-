"""
Monte Carlo trajectory propagation and uncertainty quantification for Phase 11.

NOTE ON PROGNOSTIC UNCERTAINTY:
The Monte Carlo distributions implemented herein represent physics-informed engineering
uncertainty assumptions for synthetic-data prognostics, NOT statistically calibrated
real-engine confidence distributions. They capture three primary uncertainty components:
1. Current health state estimation uncertainty (delta_HI ~ N(0, sigma_hi^2))
2. Degradation rate / trend slope estimation uncertainty (delta_beta ~ N(0, SE(beta)^2))
3. Functional failure threshold tolerance (HI_EOL ~ U(0.33, 0.37))
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

from prognostics.schema import RULConfig


class MonteCarloTrajectoryPropagator:
    """
    Propagates degradation trajectories across stochastic realizations to quantify RUL uncertainty.
    Produces median point estimates and P05/P95 empirical confidence bounds.
    """

    def __init__(self, config: Optional[RULConfig] = None, seed: Optional[int] = 42):
        self.config = config or RULConfig()
        self.rng = np.random.default_rng(seed)

    def propagate_linear_trajectory(
        self,
        anchor_time: float,
        anchor_hi: float,
        median_slope: float,
        slope_se: float,
        current_time: float = 0.0,
        m_samples: Optional[int] = None,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], np.ndarray]:
        """
        Propagate Monte Carlo trajectories for linear degradation to determine RUL distribution.

        Args:
            anchor_time: Timestamp where extended projection starts (s)
            anchor_hi: Health index value at anchor timestamp
            median_slope: Robust Theil-Sen degradation slope (s^-1)
            slope_se: Standard error of degradation slope (s^-1)
            current_time: Current simulation timestamp (s)
            m_samples: Number of stochastic realizations (defaults to config.mc_samples = 500)

        Returns:
            Tuple of (p50_median, p05_lower, p95_upper, samples_array)
        """
        if not np.isfinite(median_slope) or median_slope >= -1e-5:
            return None, None, None, np.array([], dtype=np.float64)

        M = m_samples if m_samples is not None else self.config.mc_samples

        # 1. Perturb anchor HI: delta_hi ~ N(0, sigma_hi^2)
        hi_noise = self.rng.normal(0.0, self.config.assumed_residual_std, size=M)
        sampled_anchor_hi = anchor_hi + hi_noise

        # 2. Perturb slope: delta_beta ~ N(0, SE(beta)^2)
        # Ensure standard error is non-zero
        se = max(slope_se, abs(median_slope) * self.config.assumed_slope_rel_std)
        sampled_slopes = self.rng.normal(median_slope, se, size=M)
        # Degradation slope must remain negative to reach EOL
        sampled_slopes = np.minimum(sampled_slopes, -1e-6)

        # 3. Perturb EOL threshold: HI_EOL ~ U(base - width, base + width)
        base_eol = self.config.eol_criteria.hi_eol.threshold_value
        half_w = self.config.assumed_threshold_half_width
        sampled_eol = self.rng.uniform(base_eol - half_w, base_eol + half_w, size=M)

        # 4. Analytical solving for each realization:
        # sampled_anchor_hi + sampled_slope * (t_cross - anchor_time) = sampled_eol
        delta_hi = sampled_eol - sampled_anchor_hi
        
        # Where already below EOL, RUL from anchor is 0
        already_breached = delta_hi >= 0.0
        
        tau_from_anchor = np.zeros(M, dtype=np.float64)
        not_breached = ~already_breached
        tau_from_anchor[not_breached] = delta_hi[not_breached] / sampled_slopes[not_breached]
        
        # Total crossing time t_cross = anchor_time + tau_from_anchor
        # RUL from current_time = t_cross - current_time
        rul_samples = (anchor_time - current_time) + tau_from_anchor
        rul_samples = np.maximum(0.0, rul_samples)

        # Compute empirical percentiles
        p05 = float(np.percentile(rul_samples, 5.0))
        p50 = float(np.percentile(rul_samples, 50.0))
        p95 = float(np.percentile(rul_samples, 95.0))

        return p50, p05, p95, rul_samples

    def check_mc_convergence(
        self,
        anchor_time: float,
        anchor_hi: float,
        median_slope: float,
        slope_se: float,
        current_time: float = 0.0,
        sample_sizes: Tuple[int, ...] = (100, 500, 1000),
    ) -> Dict[str, Any]:
        """
        Verify convergence of Monte Carlo estimation across sample sizes {100, 500, 1000}.
        """
        results = {}
        for M in sample_sizes:
            p50, p05, p95, _ = self.propagate_linear_trajectory(
                anchor_time=anchor_time,
                anchor_hi=anchor_hi,
                median_slope=median_slope,
                slope_se=slope_se,
                current_time=current_time,
                m_samples=M,
            )
            results[f"M_{M}"] = {
                "p50": p50,
                "p05": p05,
                "p95": p95,
                "mpiw": (p95 - p05) if (p95 is not None and p05 is not None) else None,
            }
        return results
