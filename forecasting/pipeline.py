"""
End-to-end forecasting pipeline for Phase 10: TimesFM-3 Future Telemetry Forecasting.
"""

from typing import Dict, List, Optional, Any, Union
import math
import numpy as np
import pandas as pd

from forecasting.schema import (
    DEFAULT_FORECAST_CHANNELS,
    DEFAULT_NRMSE_DENOMINATORS,
    DEFAULT_NOMINAL_EXPECTED,
    ForecastingConfig,
    ForecastResult,
    ForecastQuality,
    ModelStatus,
)
from forecasting.preprocessing import CausalTelemetryBuffer
from forecasting.baselines import PersistenceForecaster, CausalEWMAForecaster
from forecasting.models import TimesFM3ModelAdapter


class ForecastingPipeline:
    """
    Inference pipeline for Phase 10 Telemetry Forecasting.

    Features:
    - Strictly causal context extraction.
    - Full (engine_id, mission_id) mission state isolation.
    - Evaluates Persistence and Causal EWMA baselines side-by-side with TimesFM-3.
    - Optional derived projected Health Index via Phase 9 calculator.
    - Phase 7 and Phase 8 independence: operates completely on raw telemetry observations.
    """

    def __init__(
        self,
        config: Optional[ForecastingConfig] = None,
        use_multivariate: bool = True,
        force_local_graph: bool = False,
    ):
        self.config = config or ForecastingConfig()
        self.buffer = CausalTelemetryBuffer(self.config)
        self.persistence = PersistenceForecaster()
        self.causal_ewma = CausalEWMAForecaster(
            clamp_to_physical_limits=self.config.clamp_to_physical_limits
        )
        self.timesfm_adapter = TimesFM3ModelAdapter(
            config=self.config,
            use_multivariate=use_multivariate,
            force_local_graph=force_local_graph,
        )

        # Optional Phase 9 Health Calculator for derived projected health trajectory
        self._health_calculator = None
        if self.config.enable_projected_health:
            try:
                from health_index.calculator import HealthCalculator
                from health_index.schema import HealthIndexConfig
                self._health_calculator = HealthCalculator(HealthIndexConfig())
            except ImportError:
                self._health_calculator = None

    def reset(self, engine_id: Optional[str] = None, mission_id: Optional[str] = None) -> None:
        """Reset internal history and state for a specific mission or all missions."""
        self.buffer.reset(engine_id=engine_id, mission_id=mission_id)

    def process_sample(
        self,
        record: Dict[str, Any],
        optional_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[ForecastResult]:
        """
        Process a single telemetry record and generate a forecast if sufficient context exists.

        Args:
            record: Dict of telemetry measurements with 'timestamp'
            optional_context: Optional metadata from upstream phases (diagnosis/provenance)

        Returns:
            ForecastResult if context is sufficient, or None if context is accumulating.
        """
        engine_id = str(record.get("engine_id", "ENG_001"))
        raw_mission = record.get("mission_id") or record.get("mission_run_id")
        mission_id = str(raw_mission) if (raw_mission is not None and pd.notna(raw_mission)) else None

        accepted = self.buffer.add_observation(record, engine_id=engine_id, mission_id=mission_id)
        if not accepted:
            return None

        context_2d, timestamps, quality, meta = self.buffer.get_context(
            engine_id=engine_id, mission_id=mission_id
        )

        if context_2d is None or timestamps is None:
            return None

        current_t = float(timestamps[-1])
        context_start_t = float(timestamps[0])
        horizon = self.config.forecast_horizon
        dt = self.config.sampling_interval_s

        forecast_timestamps = [current_t + (k + 1) * dt for k in range(horizon)]

        # 1. Compute Persistence Baseline
        persist_preds = self.persistence.predict(
            context_2d=context_2d,
            horizon=horizon,
            target_channels=self.config.target_channels,
            timestamps=timestamps,
        )

        # 2. Compute Causal EWMA Baseline
        ewma_preds, clamping_flags = self.causal_ewma.predict(
            context_2d=context_2d,
            horizon=horizon,
            target_channels=self.config.target_channels,
            timestamps=timestamps,
            sampling_interval_s=dt,
        )

        # 3. Generate Primary Forecast via TimesFM-3 (or fallback to Causal EWMA if model blocked)
        model_name = self.config.model_name
        model_status = self.timesfm_adapter.runtime_status
        predicted_telemetry: Dict[str, List[float]] = {}
        lower_bounds: Optional[Dict[str, List[float]]] = None
        upper_bounds: Optional[Dict[str, List[float]]] = None

        if self.timesfm_adapter.is_available():
            try:
                tfm_preds, tfm_q = self.timesfm_adapter.predict(
                    context_2d=context_2d,
                    horizon=horizon,
                    target_channels=self.config.target_channels,
                    timestamps=timestamps,
                    return_quantiles=True,
                )
                for ch in self.config.target_channels:
                    predicted_telemetry[ch] = [float(v) for v in tfm_preds[ch]]
                    if tfm_q is not None and ch in tfm_q:
                        q_arr = tfm_q[ch]
                        lower_bounds = lower_bounds or {}
                        upper_bounds = upper_bounds or {}
                        lower_bounds[ch] = [float(q_arr[k, 0]) for k in range(horizon)]
                        upper_bounds[ch] = [float(q_arr[k, -1]) for k in range(horizon)]
            except Exception as e:
                # If execution fails, fall back gracefully to Causal EWMA without crashing
                model_status = f"ERROR: {e}"
                for ch in self.config.target_channels:
                    predicted_telemetry[ch] = [float(v) for v in ewma_preds[ch]]
        else:
            # When model checkpoint is gated/unauthenticated, primary output reflects baseline
            for ch in self.config.target_channels:
                predicted_telemetry[ch] = [float(v) for v in ewma_preds[ch]]

        # 4. Baseline comparison container
        baseline_comp = {
            "persistence": {ch: [float(v) for v in persist_preds[ch]] for ch in persist_preds},
            "causal_ewma": {ch: [float(v) for v in ewma_preds[ch]] for ch in ewma_preds},
            "clamping_flags": clamping_flags,
        }

        # 5. Derived Projected Health Trajectory
        projected_health: Optional[List[float]] = None
        if self.config.enable_projected_health:
            projected_health = self._compute_projected_health_trajectory(
                predicted_telemetry=predicted_telemetry,
                forecast_timestamps=forecast_timestamps,
                optional_context=optional_context,
                engine_id=engine_id,
                mission_id=mission_id,
            )

        provenance = {
            "imputed_channels": meta.get("imputed_channels", []),
            "status_detail": self.timesfm_adapter.status_detail,
            "optional_context": optional_context or {},
        }

        return ForecastResult(
            engine_id=engine_id,
            mission_id=mission_id,
            forecast_start_timestamp=current_t,
            context_start_timestamp=context_start_t,
            context_length=self.config.context_length,
            forecast_horizon=horizon,
            sampling_interval=dt,
            target_channels=list(self.config.target_channels),
            forecast_timestamps=forecast_timestamps,
            predicted_telemetry=predicted_telemetry,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            projected_health_trajectory=projected_health,
            forecast_quality=quality,
            model_name=model_name,
            model_status=model_status,
            baseline_comparison=baseline_comp,
            provenance=provenance,
        )

    def process_dataframe(
        self,
        df: pd.DataFrame,
        optional_context: Optional[Union[List[Optional[Dict[str, Any]]], pd.DataFrame]] = None,
    ) -> List[ForecastResult]:
        """
        Process a batch dataframe in chronological order, isolating distinct missions.
        """
        results: List[ForecastResult] = []
        n = len(df)
        if n == 0:
            return results

        contexts = [None] * n
        if optional_context is not None:
            if isinstance(optional_context, pd.DataFrame):
                contexts = [row.to_dict() for _, row in optional_context.iterrows()]
            elif isinstance(optional_context, list):
                contexts = list(optional_context)

        # Detect mission transitions to preserve mission boundaries
        current_mission_key = None

        for i in range(n):
            row = df.iloc[i].to_dict()
            eng = str(row.get("engine_id", "ENG_001"))
            raw_m = row.get("mission_id") or row.get("mission_run_id")
            miss = str(raw_m) if (raw_m is not None and pd.notna(raw_m)) else None
            key = (eng, miss)

            if current_mission_key is not None and key != current_mission_key:
                # Mission boundary transition -> reset buffer for preceding mission
                self.reset(engine_id=current_mission_key[0], mission_id=current_mission_key[1])
            current_mission_key = key

            ctx = contexts[i] if i < len(contexts) else None
            res = self.process_sample(row, optional_context=ctx)
            if res is not None:
                results.append(res)

        return results

    def _compute_projected_health_trajectory(
        self,
        predicted_telemetry: Dict[str, List[float]],
        forecast_timestamps: List[float],
        optional_context: Optional[Dict[str, Any]] = None,
        engine_id: str = "ENG_001",
        mission_id: Optional[str] = None,
    ) -> List[float]:
        """
        Mathematically derive projected Health Index trajectory from forecasted sensor values
        using the canonical Phase 9 HealthCalculator and Digital Twin residual scales.
        """
        horizon = len(forecast_timestamps)
        if horizon == 0 or not predicted_telemetry:
            return []

        # Ensure HealthCalculator is available
        if self._health_calculator is None:
            try:
                from health_index.calculator import HealthCalculator
                from health_index.schema import HealthIndexConfig
                self._health_calculator = HealthCalculator(HealthIndexConfig())
            except ImportError:
                return []

        ctx = optional_context or {}
        expected_dict = ctx.get("expected_telemetry") or ctx.get("expected_dict") or DEFAULT_NOMINAL_EXPECTED

        # Residual scale factors (matching Digital Twin Phase 6)
        try:
            from digital_twin.residuals import DEFAULT_RESIDUAL_SCALES
            scale_factors = DEFAULT_RESIDUAL_SCALES
        except ImportError:
            scale_factors = {
                "rpm": 100.0,
                "cht": 10.0,
                "egt": 20.0,
                "oil_temp": 10.0,
                "oil_pressure": 0.5,
                "fuel_flow": 2.0,
                "vibration": 0.2,
            }

        isolated_channels = set(ctx.get("isolated_channels") or [])
        fault_type = ctx.get("fault_type") or ctx.get("predicted_fault_type")
        diag_conf = ctx.get("diagnostic_confidence")
        suspect_ch = ctx.get("suspect_channel")

        # Causal continuity: initialize EWMA from current smoothed health index if available
        current_hi = ctx.get("current_health_index")
        smoothed_val: Optional[float] = None
        if current_hi is not None and isinstance(current_hi, (int, float)) and not math.isnan(current_hi):
            smoothed_val = max(0.0, min(1.0, float(current_hi)))

        alpha = getattr(self._health_calculator.config, "ewma_alpha", 0.15)

        projected_health: List[float] = []
        for k in range(horizon):
            step_residuals: Dict[str, float] = {}
            for ch in self._health_calculator.config.channel_weights:
                if ch in isolated_channels:
                    step_residuals[ch] = float("nan")
                elif ch in predicted_telemetry and k < len(predicted_telemetry[ch]):
                    val = predicted_telemetry[ch][k]
                    if val is None or not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
                        step_residuals[ch] = float("nan")
                    else:
                        exp_val = expected_dict.get(ch, DEFAULT_NOMINAL_EXPECTED.get(ch, 0.0))
                        scale = max(1e-6, scale_factors.get(ch, 1.0))
                        step_residuals[ch] = float((val - exp_val) / scale)
                else:
                    step_residuals[ch] = float("nan")

            raw_hi, _, _, _, _, _, _, _, _, _ = self._health_calculator.compute(
                timestamp=forecast_timestamps[k],
                residuals=step_residuals,
                upstream_fault_type=fault_type,
                upstream_confidence=diag_conf,
                upstream_suspect_channel=suspect_ch,
                engine_id=engine_id,
                mission_id=mission_id,
            )

            if math.isnan(raw_hi):
                step_smooth = smoothed_val if smoothed_val is not None else 1.0
            else:
                if smoothed_val is None:
                    step_smooth = raw_hi
                else:
                    step_smooth = alpha * raw_hi + (1.0 - alpha) * smoothed_val
                smoothed_val = step_smooth

            step_smooth = max(0.0, min(1.0, float(step_smooth)))
            projected_health.append(round(step_smooth, 4))

        return projected_health
