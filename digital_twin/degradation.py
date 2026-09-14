"""
Phase 8 Degradation Estimator and Robust Theil-Sen Trend Extraction.

Implements:
- Non-parametric Theil-Sen robust slope estimator with 29.3% breakdown point
- Uncertainty interval extraction (slope_low, slope_median, slope_high)
- Rolling historical degradation state tracking across engine subsystems
- Data quality penalty integration from Phase 3 & Phase 5
- Explicit regime classification (STABLE, DEGRADING, RAPID_DEGRADATION, INSUFFICIENT_DATA)

SCIENTIFIC DISCLAIMER:
These models estimate mathematical degradation trends on digital twin residuals.
They DO NOT imply certified Rotax 914 failure physics or flightworthiness certification.
"""

from collections import deque
from dataclasses import dataclass, field
import math
from typing import Any, Deque, Dict, List, Optional, Tuple
import numpy as np

from digital_twin.degradation_types import (
    DegradationSubsystem,
    DegradationRegime,
    SubsystemDegradationState,
    DegradationAssessment,
    ProvenanceTag,
)
from digital_twin.health import ModelObservationHealthAssessment


@dataclass
class TheilSenResult:
    """
    Result of Theil-Sen robust linear regression.
    """
    slope: float                        # Median pairwise slope
    intercept: float                    # Median intercept
    slope_low: float                    # Lower percentile bound (e.g. 15th percentile)
    slope_high: float                   # Upper percentile bound (e.g. 85th percentile)
    concordance: float                  # Fraction of pairs with same sign as median [0.0, 1.0]
    pair_count: int                     # Number of evaluated point pairs
    sample_count: int                   # Number of input points


class TheilSenEstimator:
    """
    Non-parametric robust slope estimator.
    Computes the median of all pairwise slopes:
        s_ij = (y_j - y_i) / (x_j - x_i)  for x_j > x_i
    Robust against isolated outliers (up to 29.3% breakdown point).
    """

    @staticmethod
    def estimate(
        x: np.ndarray,
        y: np.ndarray,
        max_pairs: int = 4000,
    ) -> TheilSenResult:
        """
        Estimate robust slope, intercept, and uncertainty bounds.

        Args:
            x: Monotonically increasing independent variable (timestamps, seconds).
            y: Dependent variable (degradation indicator).
            max_pairs: Upper bound on evaluated pairs for real-time bounding.

        Returns:
            TheilSenResult with median slope, bounds, and concordance.
        """
        n = len(x)
        if n < 2:
            return TheilSenResult(
                slope=0.0,
                intercept=float(y[0]) if n == 1 else 0.0,
                slope_low=0.0,
                slope_high=0.0,
                concordance=0.0,
                pair_count=0,
                sample_count=n,
            )

        original_n = n
        # Deterministic uniform decimation if history exceeds 50 samples
        # Preserves endpoint fidelity and slope precision while bounding pair count to <= 1225
        if n > 50:
            stride = int(math.ceil(n / 50))
            indices = list(range(0, n - 1, stride))
            if indices[-1] != n - 1:
                indices.append(n - 1)
            x = x[indices]
            y = y[indices]
            n = len(x)

        # Generate pairwise slopes: for j > i, (x_j - x_i) and (y_j - y_i)
        dx = x[np.newaxis, :] - x[:, np.newaxis]
        dy = y[np.newaxis, :] - y[:, np.newaxis]

        # Upper triangular indices where j > i
        i_idx, j_idx = np.triu_indices(n, k=1)
        valid_dx = dx[i_idx, j_idx]
        valid_dy = dy[i_idx, j_idx]

        # Filter strictly positive dx to avoid division by zero
        pos_mask = valid_dx > 1e-6
        valid_dx = valid_dx[pos_mask]
        valid_dy = valid_dy[pos_mask]

        if len(valid_dx) == 0:
            return TheilSenResult(
                slope=0.0,
                intercept=float(np.median(y)),
                slope_low=0.0,
                slope_high=0.0,
                concordance=0.0,
                pair_count=0,
                sample_count=n,
            )

        slopes = valid_dy / valid_dx

        # If too many pairs, deterministically subsample to preserve execution speed
        if len(slopes) > max_pairs:
            stride = int(math.ceil(len(slopes) / max_pairs))
            slopes = slopes[::stride]

        slopes.sort()
        median_slope = float(np.median(slopes))
        intercept = float(np.median(y - median_slope * x))

        # Quantile uncertainty bounds (15th and 85th percentiles)
        slope_low = float(np.percentile(slopes, 15.0))
        slope_high = float(np.percentile(slopes, 85.0))

        # Concordance: fraction of pairwise slopes with identical sign to median
        if abs(median_slope) < 1e-9:
            concordance = 0.5
        else:
            sign_match = (slopes > 0) if median_slope > 0 else (slopes < 0)
            concordance = float(np.mean(sign_match))

        return TheilSenResult(
            slope=median_slope,
            intercept=intercept,
            slope_low=slope_low,
            slope_high=slope_high,
            concordance=concordance,
            pair_count=len(slopes),
            sample_count=original_n,
        )


@dataclass
class DegradationEstimatorConfig:
    """
    Configuration and heuristic thresholds for degradation estimation.
    All parameters carry explicit provenance tags.
    """
    window_duration_s: float = 300.0                # Rolling historical window duration (seconds)
    min_observations: int = 10                      # Minimum sample count before reporting trend
    min_window_duration_s: float = 15.0             # Minimum elapsed time required (seconds)
    min_valid_data_fraction: float = 0.80           # Gating data quality threshold
    min_confidence_threshold: float = 0.30          # Minimum confidence to accept trend
    stable_slope_threshold_per_hour: float = 0.02   # |dD/dh| < 0.02 is STABLE
    rapid_slope_threshold_per_hour: float = 0.20    # dD/dh >= 0.20 is RAPID_DEGRADATION
    eol_degradation_threshold: float = 0.50         # Model horizon D_EOL (D = 1 - HI = 0.50)
    provenance: str = ProvenanceTag.ENGINEERING_HEURISTIC.value


class DegradationEstimator:
    """
    Estimates engine and subsystem degradation states from historical Phase 5 health assessments.
    Strictly causal: uses only past and present observations (t <= current_time).
    """

    def __init__(self, config: Optional[DegradationEstimatorConfig] = None):
        self.config = config or DegradationEstimatorConfig()
        # History queue: stores tuples of (timestamp, D_val, D_raw, C_data, C_obs, is_valid, sub_D_dict)
        self._history: Deque[Tuple[float, float, float, float, float, bool, Dict[str, float]]] = deque()
        self._last_timestamp: float = -1.0

    def reset(self) -> None:
        """Clear all historical state buffer."""
        self._history.clear()
        self._last_timestamp = -1.0

    def estimate(
        self,
        health_assessment: ModelObservationHealthAssessment,
        dt: Optional[float] = None,
    ) -> DegradationAssessment:
        """
        Ingest current health assessment and estimate degradation state and trends.

        Args:
            health_assessment: Authoritative Phase 5 health assessment.
            dt: Optional elapsed time step.

        Returns:
            DegradationAssessment with robust trends, subsystem breakdowns, and regime.
        """
        t = health_assessment.timestamp
        self._last_timestamp = t

        # Extract normalized health-derived degradation indicator: D(t) = 1.0 - HI_smooth
        # Clamp to [0.0, 1.0]
        if math.isnan(health_assessment.HI_smooth):
            d_val = 1.0  # Unavailable / critical
            is_valid = False
        else:
            d_val = max(0.0, min(1.0, 1.0 - health_assessment.HI_smooth))
            is_valid = True

        if math.isnan(health_assessment.HI_raw):
            d_raw = 1.0
        else:
            d_raw = max(0.0, min(1.0, 1.0 - health_assessment.HI_raw))

        c_data = health_assessment.C_data
        c_obs = health_assessment.C_obs

        # Compute subsystem degradation indicators: D_sub = 1.0 - score
        sub_d: Dict[str, float] = {}
        for sub_name, sub_ass in health_assessment.subsystems.items():
            if math.isnan(sub_ass.score):
                sub_d[sub_name] = 1.0
            else:
                sub_d[sub_name] = max(0.0, min(1.0, 1.0 - sub_ass.score))

        # Check for specific cooling channel indicator
        coolant_ind = health_assessment.channel_indicators.get("coolant_temp")
        if coolant_ind is not None and coolant_ind.valid:
            sub_d["COOLING"] = max(0.0, min(1.0, 1.0 - coolant_ind.channel_score))
        elif "THERMAL" in sub_d:
            sub_d["COOLING"] = sub_d["THERMAL"]

        # Append to historical rolling queue
        self._history.append((t, d_val, d_raw, c_data, c_obs, is_valid, sub_d))

        # Evict samples older than window_duration_s
        cutoff = t - self.config.window_duration_s
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        # Extract vectors for trend estimation
        times = np.array([record[0] for record in self._history], dtype=float)
        d_vals = np.array([record[1] for record in self._history], dtype=float)
        val_flags = np.array([record[5] for record in self._history], dtype=bool)

        sample_count = len(times)
        window_duration = float(times[-1] - times[0]) if sample_count > 1 else 0.0
        window_start = float(times[0]) if sample_count > 0 else t
        window_end = float(times[-1]) if sample_count > 0 else t

        valid_fraction = float(np.mean(val_flags)) if sample_count > 0 else 0.0
        avg_c_data = float(np.mean([r[3] for r in self._history])) if sample_count > 0 else 0.0

        # Minimum data gating checks
        insufficient_data = (
            sample_count < self.config.min_observations
            or window_duration < self.config.min_window_duration_s
            or valid_fraction < self.config.min_valid_data_fraction
        )

        if insufficient_data:
            # Report insufficient data with zero trend
            regime = DegradationRegime.INSUFFICIENT_DATA
            theil_sen = TheilSenEstimator.estimate(times, d_vals)
            slope_sec = 0.0
            slope_hour = 0.0
            slope_low = 0.0
            slope_high = 0.0
            confidence = round(max(0.0, min(1.0, valid_fraction * avg_c_data * 0.2)), 4)
        else:
            # Perform robust Theil-Sen regression
            theil_sen = TheilSenEstimator.estimate(times, d_vals)
            slope_sec = theil_sen.slope
            slope_hour = slope_sec * 3600.0
            slope_low = theil_sen.slope_low
            slope_high = theil_sen.slope_high

            # Trend confidence modulated by concordance, sample coverage, and sensor quality
            base_confidence = theil_sen.concordance * min(1.0, sample_count / 30.0)
            confidence = float(np.clip(base_confidence * avg_c_data * valid_fraction, 0.0, 1.0))

            # Regime classification based on robust slope per hour
            if slope_hour >= self.config.rapid_slope_threshold_per_hour:
                regime = DegradationRegime.RAPID_DEGRADATION
            elif slope_hour >= self.config.stable_slope_threshold_per_hour:
                regime = DegradationRegime.DEGRADING
            else:
                regime = DegradationRegime.STABLE

        # Map to typed SubsystemDegradationState
        subsystem_states = self._build_subsystem_states(
            times=times,
            window_start=window_start,
            window_end=window_end,
            sample_count=sample_count,
            avg_c_data=avg_c_data,
            health_assessment=health_assessment,
        )

        # Identify dominant degradation subsystem
        dominant_sub = None
        max_d_sub = -1.0
        for s_name, s_state in subsystem_states.items():
            if s_state.degradation_index > max_d_sub:
                max_d_sub = s_state.degradation_index
                dominant_sub = s_name

        return DegradationAssessment(
            timestamp=t,
            engine_id=health_assessment.engine_id,
            degradation_index=d_val,
            degradation_raw=d_raw,
            trend_slope_per_sec=slope_sec,
            trend_slope_per_hour=slope_hour,
            slope_low_per_sec=slope_low,
            slope_high_per_sec=slope_high,
            regime=regime,
            confidence=confidence,
            observation_count=sample_count,
            window_duration_s=window_duration,
            window_start_s=window_start,
            window_end_s=window_end,
            data_quality_factor=avg_c_data,
            subsystems=subsystem_states,
            dominant_subsystem=dominant_sub,
            provenance=self.config.provenance,
            metadata={
                "valid_fraction": valid_fraction,
                "theil_sen_pair_count": theil_sen.pair_count,
                "theil_sen_concordance": theil_sen.concordance,
            },
        )

    def _build_subsystem_states(
        self,
        times: np.ndarray,
        window_start: float,
        window_end: float,
        sample_count: int,
        avg_c_data: float,
        health_assessment: ModelObservationHealthAssessment,
    ) -> Dict[str, SubsystemDegradationState]:
        """
        Build typed SubsystemDegradationState models for all 6 core degradation dimensions.
        """
        # Mapping between DegradationSubsystem and Phase 5 subsystem / channels
        sub_channel_map = {
            DegradationSubsystem.THERMAL_DEGRADATION: ("THERMAL", ["cht", "oil_temp"]),
            DegradationSubsystem.LUBRICATION_DEGRADATION: ("LUBRICATION", ["oil_pressure"]),
            DegradationSubsystem.COMBUSTION_DEGRADATION: ("COMBUSTION", ["egt"]),
            DegradationSubsystem.MECHANICAL_DEGRADATION: ("MECHANICAL", ["vibration"]),
            DegradationSubsystem.FUEL_SYSTEM_DEGRADATION: ("FUEL", ["fuel_flow"]),
            DegradationSubsystem.COOLING_DEGRADATION: ("COOLING", ["coolant_temp"]),
        }

        states: Dict[str, SubsystemDegradationState] = {}

        for deg_sub, (p5_name, channels) in sub_channel_map.items():
            # Extract historical trajectory for this subsystem
            sub_y = []
            for record in self._history:
                sub_dict = record[6]
                sub_val = sub_dict.get(p5_name, 0.0)
                sub_y.append(sub_val)

            sub_arr = np.array(sub_y, dtype=float)
            curr_d = float(sub_arr[-1]) if len(sub_arr) > 0 else 0.0

            if sample_count >= self.config.min_observations and (window_end - window_start) >= self.config.min_window_duration_s:
                ts_res = TheilSenEstimator.estimate(times, sub_arr)
                sub_slope_sec = ts_res.slope
                sub_slope_hour = sub_slope_sec * 3600.0
                if sub_slope_hour >= self.config.rapid_slope_threshold_per_hour:
                    sub_regime = DegradationRegime.RAPID_DEGRADATION
                elif sub_slope_hour >= self.config.stable_slope_threshold_per_hour:
                    sub_regime = DegradationRegime.DEGRADING
                else:
                    sub_regime = DegradationRegime.STABLE
                sub_conf = float(np.clip(ts_res.concordance * avg_c_data, 0.0, 1.0))
            else:
                sub_slope_sec = 0.0
                sub_slope_hour = 0.0
                sub_regime = DegradationRegime.INSUFFICIENT_DATA
                sub_conf = 0.1

            states[deg_sub.value] = SubsystemDegradationState(
                subsystem=deg_sub,
                degradation_index=curr_d,
                trend_per_second=sub_slope_sec,
                trend_per_hour=sub_slope_hour,
                regime=sub_regime,
                confidence=sub_conf,
                observation_count=sample_count,
                window_start=window_start,
                window_end=window_end,
                data_quality=avg_c_data,
                contributing_channels=channels,
            )

        return states
