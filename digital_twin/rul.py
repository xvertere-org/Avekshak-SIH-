"""
Phase 8 Remaining Useful Life (RUL) Estimator.

Implements:
- Uncertainty-aware prognostic extrapolation to model-defined EOL horizon D_EOL
- Upper/lower uncertainty propagation (RUL_low, RUL_median, RUL_high)
- Operational scenario projections (CURRENT_PROFILE, NORMAL_MISSION, HIGH_ALTITUDE, HOT_DAY, HIGH_LOAD)
- Minimum evidence gating and explicit non-degrading states (no fake infinite numbers)
- Structured, non-LLM explainability directly from degradation evidence
- Zero ground-truth label leakage

SCIENTIFIC DISCLAIMER:
Projections are mathematical horizons under stated grey-box digital twin assumptions.
They DO NOT represent certified engine-life or airworthiness predictions.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional
import numpy as np

from digital_twin.degradation_types import (
    DegradationAssessment,
    DegradationRegime,
    RULAssessment,
    RULScenario,
    RULStatus,
    ProvenanceTag,
    DEFAULT_SCENARIO_STRESS_FACTORS,
)


@dataclass
class RULEstimatorConfig:
    """
    Configuration parameters for Remaining Useful Life estimation.
    All parameters carry explicit provenance tags.
    """
    eol_threshold: float = 0.50                 # Model horizon D_EOL (D = 1 - HI = 0.50)
    min_observations: int = 10                  # Minimum observations gate
    min_window_duration_s: float = 15.0         # Minimum elapsed history gate (seconds)
    min_confidence: float = 0.25                # Minimum trend confidence to project RUL
    min_valid_data_fraction: float = 0.70       # Data quality gate
    stable_slope_threshold_per_hour: float = 0.02  # |dD/dh| < 0.02 is STABLE
    min_positive_slope_per_hour: float = 0.02   # Slopes below this are non-degrading or stable
    provenance: str = ProvenanceTag.ENGINEERING_HEURISTIC.value
    assumptions: str = (
        "Linear extrapolation of robust Theil-Sen degradation slope to model-defined D_EOL horizon. "
        "Not an OEM maintenance limit or airworthiness prediction."
    )


class RULEstimator:
    """
    Uncertainty-aware Remaining Useful Life (RUL) estimator.
    Consumes DegradationAssessment and projects operational horizons without label leakage.
    """

    def __init__(self, config: Optional[RULEstimatorConfig] = None):
        self.config = config or RULEstimatorConfig()
        self.scenario_stress_factors = dict(DEFAULT_SCENARIO_STRESS_FACTORS)

    def estimate(
        self,
        degradation_assessment: DegradationAssessment,
        scenario: RULScenario = RULScenario.CURRENT_PROFILE,
        operating_context: Optional[Dict[str, Any]] = None,
    ) -> RULAssessment:
        """
        Estimate Remaining Useful Life under the specified operating scenario.

        Args:
            degradation_assessment: Degradation state and trend from DegradationEstimator.
            scenario: Simulated future operating scenario profile.
            operating_context: Optional environmental/flight context (e.g. altitude, ambient_temp).

        Returns:
            RULAssessment with central estimate, uncertainty bounds, and evidence breakdown.
        """
        deg = degradation_assessment
        d_curr = deg.degradation_index
        slope_sec = deg.trend_slope_per_sec
        slope_hour = deg.trend_slope_per_hour
        eol_d = self.config.eol_threshold
        sample_count = deg.observation_count
        window_dur = deg.window_duration_s
        c_data = deg.data_quality_factor
        conf = deg.confidence

        stress_multiplier = self.scenario_stress_factors.get(scenario, 1.00)

        # 1. Gate: Sensor/Data Quality Degraded
        valid_frac = deg.metadata.get("valid_fraction", 1.0)
        if c_data < 0.35 or valid_frac < self.config.min_valid_data_fraction:
            return self._build_result(
                status=RULStatus.DATA_QUALITY_DEGRADED,
                deg=deg,
                scenario=scenario,
                rul_low=None,
                rul_median=None,
                rul_high=None,
                rejection_reason=(
                    f"Sensor data quality degraded: C_data={c_data:.2f} < 0.35 or "
                    f"valid_fraction={valid_frac:.2f} < {self.config.min_valid_data_fraction:.2f}."
                ),
            )

        # 2. Gate: Insufficient Historical Evidence
        if sample_count < self.config.min_observations or window_dur < self.config.min_window_duration_s:
            return self._build_result(
                status=RULStatus.INSUFFICIENT_DATA,
                deg=deg,
                scenario=scenario,
                rul_low=None,
                rul_median=None,
                rul_high=None,
                rejection_reason=(
                    f"Insufficient history: {sample_count}/{self.config.min_observations} samples, "
                    f"{window_dur:.1f}/{self.config.min_window_duration_s:.1f}s window."
                ),
            )

        # 3. Gate: Already Beyond Model Horizon
        if d_curr >= eol_d:
            return self._build_result(
                status=RULStatus.ALREADY_BEYOND_MODEL_HORIZON,
                deg=deg,
                scenario=scenario,
                rul_low=0.0,
                rul_median=0.0,
                rul_high=0.0,
                rejection_reason=f"Current degradation index D(t)={d_curr:.3f} exceeds D_EOL={eol_d:.2f}.",
            )

        # 4. Gate: Significant Negative Slope (Active Recovery / Clearing Disturbance)
        if slope_hour <= -self.config.stable_slope_threshold_per_hour:
            return self._build_result(
                status=RULStatus.NON_DEGRADING,
                deg=deg,
                scenario=scenario,
                rul_low=None,
                rul_median=None,
                rul_high=None,
                rejection_reason=f"Degradation slope is negative ({slope_hour:.4f}/hr), indicating recovery.",
            )

        # 5. Gate: Stable Baseline Engine (Regime STABLE or slope within stable threshold)
        if deg.regime == DegradationRegime.STABLE or abs(slope_hour) < self.config.stable_slope_threshold_per_hour:
            return self._build_result(
                status=RULStatus.STABLE,
                deg=deg,
                scenario=scenario,
                rul_low=None,
                rul_median=None,
                rul_high=None,
                rejection_reason=f"Degradation slope ({slope_hour:.4f}/hr) is within stable regime.",
            )

        # 6. Gate: Low Trend Confidence
        if conf < self.config.min_confidence:
            return self._build_result(
                status=RULStatus.INSUFFICIENT_DATA,
                deg=deg,
                scenario=scenario,
                rul_low=None,
                rul_median=None,
                rul_high=None,
                rejection_reason=f"Trend confidence ({conf:.3f}) below acceptance gate ({self.config.min_confidence:.2f}).",
            )

        # 7. Compute RUL under scenario stress
        effective_slope_sec = slope_sec * stress_multiplier
        remaining_headroom = max(0.0, eol_d - d_curr)

        # Median estimate
        rul_median_sec = remaining_headroom / effective_slope_sec
        rul_median_hours = rul_median_sec / 3600.0

        # Pessimistic lower bound (using upper quantile slope)
        slope_high_eff = deg.slope_high_per_sec * stress_multiplier
        if slope_high_eff > 0:
            rul_low_hours = min(rul_median_hours, (remaining_headroom / slope_high_eff) / 3600.0)
        else:
            rul_low_hours = rul_median_hours

        # Optimistic upper bound (using lower quantile slope)
        slope_low_eff = deg.slope_low_per_sec * stress_multiplier
        if slope_low_eff > 0:
            rul_high_hours = max(rul_median_hours, (remaining_headroom / slope_low_eff) / 3600.0)
        else:
            rul_high_hours = None  # Unbounded upper horizon under model assumptions

        # Generate scenario comparisons
        scenario_projections = self._project_all_scenarios(remaining_headroom, slope_sec)

        # Generate structured explainability evidence
        explanation = self._build_explanation(deg, scenario, stress_multiplier, rul_median_hours)

        return RULAssessment(
            status=RULStatus.COMPUTED,
            degradation_index=d_curr,
            degradation_trend=slope_hour,
            trend_unit="delta_D_per_hour",
            trend_confidence=conf,
            rul_low=rul_low_hours,
            rul_median=rul_median_hours,
            rul_high=rul_high_hours,
            rul_unit="hours",
            eol_threshold=eol_d,
            data_confidence=c_data,
            sample_count=sample_count,
            window_duration=window_dur,
            scenario=scenario.value,
            assumptions=self.config.assumptions,
            provenance=self.config.provenance,
            explanation_evidence=explanation,
            scenario_projections=scenario_projections,
            rejection_reason=None,
            metadata={
                "stress_multiplier": stress_multiplier,
                "effective_slope_per_hour": effective_slope_sec * 3600.0,
                "remaining_headroom_D": remaining_headroom,
            },
        )

    def _project_all_scenarios(
        self,
        remaining_headroom: float,
        base_slope_sec: float,
    ) -> Dict[str, Dict[str, Optional[float]]]:
        """
        Compute deterministic projections across all supported scenarios.
        """
        projections: Dict[str, Dict[str, Optional[float]]] = {}
        for scen, mult in self.scenario_stress_factors.items():
            scen_slope_sec = base_slope_sec * mult
            if scen_slope_sec > 1e-9:
                scen_rul = (remaining_headroom / scen_slope_sec) / 3600.0
            else:
                scen_rul = None

            projections[scen.value] = {
                "rul_median_hours": round(scen_rul, 3) if scen_rul is not None else None,
                "stress_multiplier": mult,
                "projected_slope_per_hour": round(scen_slope_sec * 3600.0, 4),
            }
        return projections

    def _build_explanation(
        self,
        deg: DegradationAssessment,
        scenario: RULScenario,
        stress_multiplier: float,
        rul_hours: float,
    ) -> List[Dict[str, Any]]:
        """
        Build deterministic structured evidence explaining the RUL projection.
        """
        factors: List[Dict[str, Any]] = []

        # 1. Overall trend contribution
        factors.append({
            "category": "TREND",
            "description": f"Robust Theil-Sen degradation slope is {deg.trend_slope_per_hour:.4f}/hour (regime: {deg.regime.value}).",
            "severity": "RAPID" if deg.trend_slope_per_hour >= 0.20 else "MODERATE",
            "contribution_metric": round(deg.trend_slope_per_hour, 4),
        })

        # 2. Subsystem contributions (sorted by highest degradation)
        sorted_subs = sorted(
            deg.subsystems.values(),
            key=lambda s: s.degradation_index,
            reverse=True,
        )
        for sub_state in sorted_subs[:3]:
            if sub_state.degradation_index > 0.05 or sub_state.trend_per_hour > 0.01:
                factors.append({
                    "category": "SUBSYSTEM",
                    "subsystem": sub_state.subsystem.value,
                    "description": (
                        f"Subsystem {sub_state.subsystem.value} degradation={sub_state.degradation_index:.3f}, "
                        f"slope={sub_state.trend_per_hour:.4f}/hr (regime: {sub_state.regime.value})."
                    ),
                    "channels": sub_state.contributing_channels,
                    "contribution_metric": round(sub_state.degradation_index, 4),
                })

        # 3. Scenario impact
        if stress_multiplier != 1.0:
            factors.append({
                "category": "SCENARIO",
                "scenario": scenario.value,
                "description": f"Scenario '{scenario.value}' applies {stress_multiplier:.2f}x stress acceleration.",
                "stress_multiplier": stress_multiplier,
            })

        # 4. Data quality factor
        if deg.data_quality_factor < 0.90:
            factors.append({
                "category": "DATA_QUALITY",
                "description": f"Sensor data quality factor is {deg.data_quality_factor:.2f}, reducing overall confidence.",
                "quality_factor": round(deg.data_quality_factor, 3),
            })

        return factors

    def _build_result(
        self,
        status: RULStatus,
        deg: DegradationAssessment,
        scenario: RULScenario,
        rul_low: Optional[float],
        rul_median: Optional[float],
        rul_high: Optional[float],
        rejection_reason: str,
    ) -> RULAssessment:
        """
        Construct a non-computed or gated RULAssessment.
        """
        return RULAssessment(
            status=status,
            degradation_index=deg.degradation_index,
            degradation_trend=deg.trend_slope_per_hour,
            trend_unit="delta_D_per_hour",
            trend_confidence=deg.confidence,
            rul_low=rul_low,
            rul_median=rul_median,
            rul_high=rul_high,
            rul_unit="hours",
            eol_threshold=self.config.eol_threshold,
            data_confidence=deg.data_quality_factor,
            sample_count=deg.observation_count,
            window_duration=deg.window_duration_s,
            scenario=scenario.value,
            assumptions=self.config.assumptions,
            provenance=self.config.provenance,
            explanation_evidence=[{"category": "STATUS", "description": rejection_reason}],
            scenario_projections={},
            rejection_reason=rejection_reason,
            metadata={"rejection_reason": rejection_reason},
        )
