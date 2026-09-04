"""
Phase 10: TimesFM-3 Future Telemetry Forecasting.

Answers:
"Given the engine behavior observed so far, what is likely to happen next?"
"""

from forecasting.schema import (
    DEFAULT_FORECAST_CHANNELS,
    DEFAULT_NRMSE_DENOMINATORS,
    PHYSICAL_LOWER_BOUNDS,
    ForecastQuality,
    ModelStatus,
    ForecastingConfig,
    ForecastResult,
)
from forecasting.preprocessing import CausalTelemetryBuffer
from forecasting.baselines import PersistenceForecaster, CausalEWMAForecaster
from forecasting.models import TimesFM3ModelAdapter, ForecastModel
from forecasting.pipeline import ForecastingPipeline
from forecasting.evaluator import (
    ForecastingEvaluator,
    compute_forecast_metrics,
    split_missions_grouped,
)

__all__ = [
    "DEFAULT_FORECAST_CHANNELS",
    "DEFAULT_NRMSE_DENOMINATORS",
    "PHYSICAL_LOWER_BOUNDS",
    "ForecastQuality",
    "ModelStatus",
    "ForecastingConfig",
    "ForecastResult",
    "CausalTelemetryBuffer",
    "PersistenceForecaster",
    "CausalEWMAForecaster",
    "TimesFM3ModelAdapter",
    "ForecastModel",
    "ForecastingPipeline",
    "ForecastingEvaluator",
    "compute_forecast_metrics",
    "split_missions_grouped",
]
