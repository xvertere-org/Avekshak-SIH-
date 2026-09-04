"""
Temporal evidence extraction for Phase 12 Explainability.
Evaluates causal trend evolution, degradation rate velocity, and trajectory direction.
NOTE: Operates under strict causality; never inspects future observations or future ground truth.
"""

from typing import Optional, Dict, Any
import numpy as np

from explainability.schema import TemporalEvidence
from health_index.schema import HealthIndexResult
from forecasting.schema import ForecastResult
from prognostics.schema import RULResult


class TemporalEvidenceEvaluator:
    """
    Evaluates temporal evolution and directionality of engine degradation strictly at or before current time.
    """

    def __init__(self, rate_threshold: float = 0.0005):
        self.rate_threshold = rate_threshold

    def evaluate(
        self,
        health_result: Optional[HealthIndexResult],
        forecast_result: Optional[ForecastResult] = None,
        rul_result: Optional[RULResult] = None,
    ) -> TemporalEvidence:
        """
        Synthesize causal temporal evidence from Phase 9, 10, and 11 outputs.
        """
        if health_result is None:
            return TemporalEvidence(
                degradation_rate=None,
                degradation_trend=None,
                trend_direction="INDETERMINATE",
                forecast_status=None,
                forecast_horizon_s=None,
                status="UNAVAILABLE",
            )

        deg_rate = getattr(health_result, "degradation_rate", None)
        deg_trend = getattr(health_result, "degradation_trend", None)

        # Classify trend direction
        if deg_rate is None or not np.isfinite(deg_rate):
            direction = "INDETERMINATE"
        elif deg_rate < -self.rate_threshold:
            direction = "WORSENING"
        elif deg_rate > self.rate_threshold:
            direction = "IMPROVING"
        else:
            direction = "STABLE"

        # Forecast context from Phase 10
        fc_status = None
        fc_horizon = None
        if forecast_result is not None:
            fc_status = getattr(forecast_result, "model_status", None)
            fc_horizon = float(getattr(forecast_result, "forecast_horizon", 0.0))

        status = "AVAILABLE" if (deg_rate is not None and np.isfinite(deg_rate)) else "INSUFFICIENT_HISTORY"

        return TemporalEvidence(
            degradation_rate=round(deg_rate, 6) if (deg_rate is not None and np.isfinite(deg_rate)) else None,
            degradation_trend=deg_trend,
            trend_direction=direction,
            forecast_status=fc_status,
            forecast_horizon_s=fc_horizon,
            status=status,
        )
