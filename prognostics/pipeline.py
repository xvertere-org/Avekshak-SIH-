"""
Authoritative Remaining Useful Life (RUL) Inference Pipeline for Phase 11.
Integrates Phase 9 Health Index, Phase 10 Telemetry Forecasting, Weakest-Link EOL evaluation,
and Monte Carlo trajectory uncertainty propagation under strict state machine exhaustiveness.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple, Set, Any
import numpy as np

from health_index.schema import HealthIndexResult
from forecasting.schema import ForecastResult
from prognostics.schema import RULConfig, RULResult, RULStatus
from prognostics.threshold import WeakestLinkEOLEvaluator
from prognostics.trajectory import DualHorizonSynthesizer, TheilSenExtrapolator
from prognostics.uncertainty import MonteCarloTrajectoryPropagator


class RULPipeline:
    """
    Stateful prognostic inference engine per (engine_id, mission_id) flight session.
    """

    def __init__(self, config: Optional[RULConfig] = None, seed: Optional[int] = 42):
        self.config = config or RULConfig()
        self.weakest_link = WeakestLinkEOLEvaluator(self.config.eol_criteria)
        self.dual_horizon = DualHorizonSynthesizer(self.config)
        self.theil_sen = self.dual_horizon.extrapolator
        self.mc_propagator = MonteCarloTrajectoryPropagator(self.config, seed=seed)

        # State storage per (engine_id, mission_id): dict of history lists
        self._history: Dict[Tuple[str, Optional[str]], Dict[str, Any]] = {}

    def reset(self, engine_id: Optional[str] = None, mission_id: Optional[str] = None) -> None:
        """Reset historical state for a specific mission or all missions."""
        if engine_id is None:
            self._history.clear()
        else:
            keys_to_del = [k for k in self._history if k[0] == engine_id and (mission_id is None or k[1] == mission_id)]
            for k in keys_to_del:
                del self._history[k]

    def _get_history(self, engine_id: str, mission_id: Optional[str]) -> Dict[str, Any]:
        key = (engine_id, mission_id)
        if key not in self._history:
            self._history[key] = {
                "timestamps": [],
                "health_values": [],
                "first_timestamp": None,
            }
        return self._history[key]

    def process_assessment(
        self,
        health_result: HealthIndexResult,
        forecast: Optional[ForecastResult] = None,
        current_telemetry: Optional[Dict[str, float]] = None,
    ) -> RULResult:
        """
        Process incoming health assessment and optional forecast into an authoritative RUL estimate.
        """
        engine_id = health_result.engine_id
        mission_id = health_result.mission_id
        timestamp = float(health_result.timestamp)
        flight_phase = getattr(health_result, "mission_phase", "CRUISE") or "CRUISE"
        hi_smooth = float(health_result.smoothed_health_index)
        excluded_channels = set(getattr(health_result, "excluded_channels", []))
        valid_channels = getattr(health_result, "valid_channels", [])

        # Update historical state
        hist = self._get_history(engine_id, mission_id)
        if hist["first_timestamp"] is None:
            hist["first_timestamp"] = timestamp

        hist["timestamps"].append(timestamp)
        hist["health_values"].append(hi_smooth)

        # Prune old history beyond max window
        cutoff = timestamp - max(self.config.theil_sen_window_s * 2.0, 120.0)
        while len(hist["timestamps"]) > 1 and hist["timestamps"][0] < cutoff:
            hist["timestamps"].pop(0)
            hist["health_values"].pop(0)

        history_ts = np.array(hist["timestamps"], dtype=np.float64)
        history_hi = np.array(hist["health_values"], dtype=np.float64)
        elapsed_time = timestamp - hist["first_timestamp"]

        # Default telemetry dict if not provided
        telemetry_dict = current_telemetry or {}

        # -------------------------------------------------------------
        # STEP 1: PIPELINE CONTEXT GUARDS
        # -------------------------------------------------------------
        # Warmup duration lockout
        if elapsed_time < self.config.warmup_duration_s or timestamp < self.config.warmup_duration_s:
            return RULResult(
                engine_id=engine_id,
                mission_id=mission_id,
                timestamp=timestamp,
                status=RULStatus.INSUFFICIENT_HISTORY,
                rul_seconds_median=None,
                rul_seconds_p05=None,
                rul_seconds_p95=None,
                limiting_factor="INSUFFICIENT_HISTORY",
                confidence_score=0.0,
                active_flight_phase=flight_phase,
                handoff_horizon_s=0.0,
                trajectory_type="NONE",
                provenance={"reason": "warmup_duration_lockout", "elapsed_s": elapsed_time},
            )

        # Valid channel sufficiency (at least 4 valid unisolated physical channels)
        active_valid_count = len([ch for ch in valid_channels if ch not in excluded_channels])
        if active_valid_count < 4:
            return RULResult(
                engine_id=engine_id,
                mission_id=mission_id,
                timestamp=timestamp,
                status=RULStatus.INSUFFICIENT_DATA,
                rul_seconds_median=None,
                rul_seconds_p05=None,
                rul_seconds_p95=None,
                limiting_factor="INSUFFICIENT_DATA",
                confidence_score=0.0,
                active_flight_phase=flight_phase,
                handoff_horizon_s=0.0,
                trajectory_type="NONE",
                provenance={"reason": "active_valid_channels_below_minimum", "count": active_valid_count},
            )

        # -------------------------------------------------------------
        # STEP 2: TERMINAL FAILURE GUARD (Immediate EOL)
        # -------------------------------------------------------------
        is_breached, limiting_factor = self.weakest_link.check_immediate_eol(
            current_health_index=hi_smooth,
            current_telemetry=telemetry_dict,
            excluded_channels=excluded_channels,
        )
        if is_breached:
            return RULResult(
                engine_id=engine_id,
                mission_id=mission_id,
                timestamp=timestamp,
                status=RULStatus.CRITICAL_EOL_REACHED,
                rul_seconds_median=0.0,
                rul_seconds_p05=0.0,
                rul_seconds_p95=0.0,
                limiting_factor=limiting_factor,
                confidence_score=1.0,
                active_flight_phase=flight_phase,
                handoff_horizon_s=0.0,
                trajectory_type="IMMEDIATE_BREACH",
                provenance={"reason": "immediate_eol_breached", "limiting_factor": limiting_factor},
            )

        # -------------------------------------------------------------
        # STEP 3: ESTIMATE ROBUST DEGRADATION SLOPE (Theil-Sen)
        # -------------------------------------------------------------
        slope, slope_se = self.theil_sen.estimate_slope(history_ts, history_hi)
        if not np.isfinite(slope):
            return RULResult(
                engine_id=engine_id,
                mission_id=mission_id,
                timestamp=timestamp,
                status=RULStatus.INSUFFICIENT_HISTORY,
                rul_seconds_median=None,
                rul_seconds_p05=None,
                rul_seconds_p95=None,
                limiting_factor="INSUFFICIENT_POINTS_FOR_SLOPE",
                confidence_score=0.0,
                active_flight_phase=flight_phase,
                handoff_horizon_s=0.0,
                trajectory_type="NONE",
                provenance={"reason": "insufficient_history_for_theil_sen"},
            )

        # -------------------------------------------------------------
        # STEP 4: MATHEMATICALLY EXHAUSTIVE STATE MACHINE
        # Partition finite (HI, dHI_dt) with zero gaps and zero overlaps
        # -------------------------------------------------------------
        if hi_smooth >= 0.85:
            if slope > 0.0005:
                return self._build_non_degrading_result(
                    engine_id, mission_id, timestamp, flight_phase,
                    RULStatus.RECOVERING, slope, slope_se, "thermal_or_dynamic_recovery"
                )
            elif -0.001 <= slope <= 0.0005:
                return self._build_non_degrading_result(
                    engine_id, mission_id, timestamp, flight_phase,
                    RULStatus.NOT_DEGRADING, slope, slope_se, "nominal_cruise_stability"
                )
            else:
                # slope < -0.001: Acute degradation from healthy state
                return self._dispatch_active_prognostics(
                    engine_id, mission_id, timestamp, flight_phase,
                    hi_smooth, history_ts, history_hi, slope, slope_se,
                    excluded_channels, forecast
                )
        else:
            # 0.35 < hi_smooth < 0.85
            if slope > 0.0005:
                return self._build_non_degrading_result(
                    engine_id, mission_id, timestamp, flight_phase,
                    RULStatus.RECOVERING, slope, slope_se, "thermal_or_dynamic_recovery"
                )
            elif -0.0005 <= slope <= 0.0005:
                return self._build_non_degrading_result(
                    engine_id, mission_id, timestamp, flight_phase,
                    RULStatus.INDETERMINATE_TREND, slope, slope_se, "stationary_degraded_health"
                )
            else:
                # slope < -0.0005: Active progressive degradation
                return self._dispatch_active_prognostics(
                    engine_id, mission_id, timestamp, flight_phase,
                    hi_smooth, history_ts, history_hi, slope, slope_se,
                    excluded_channels, forecast
                )

    def _build_non_degrading_result(
        self,
        engine_id: str,
        mission_id: Optional[str],
        timestamp: float,
        flight_phase: str,
        status: RULStatus,
        slope: float,
        slope_se: float,
        reason: str,
    ) -> RULResult:
        return RULResult(
            engine_id=engine_id,
            mission_id=mission_id,
            timestamp=timestamp,
            status=status,
            rul_seconds_median=None,
            rul_seconds_p05=None,
            rul_seconds_p95=None,
            limiting_factor="NONE",
            confidence_score=1.0,
            active_flight_phase=flight_phase,
            handoff_horizon_s=0.0,
            trajectory_type="STATIONARY_OR_RECOVERING",
            provenance={
                "reason": reason,
                "theil_sen_slope": slope,
                "slope_se": slope_se,
            },
        )

    def _dispatch_active_prognostics(
        self,
        engine_id: str,
        mission_id: Optional[str],
        timestamp: float,
        flight_phase: str,
        hi_smooth: float,
        history_ts: np.ndarray,
        history_hi: np.ndarray,
        slope: float,
        slope_se: float,
        excluded_channels: Set[str],
        forecast: Optional[ForecastResult],
    ) -> RULResult:
        """
        Synthesize dual-horizon trajectory, propagate Monte Carlo uncertainty, and evaluate weakest-link EOL.
        """
        # 1. Trajectory synthesis with Phase 10 handoff safety check
        synth = self.dual_horizon.synthesize_trajectory(
            current_time=timestamp,
            current_hi=hi_smooth,
            history_timestamps=history_ts,
            history_hi=history_hi,
            forecast=forecast,
        )

        anchor_t = synth["anchor_time"]
        anchor_hi = synth["anchor_hi"]
        H = synth["handoff_horizon_s"]
        traj_type = synth["trajectory_type"]

        # 2. Monte Carlo uncertainty propagation
        p50, p05, p95, _ = self.mc_propagator.propagate_linear_trajectory(
            anchor_time=anchor_t,
            anchor_hi=anchor_hi,
            median_slope=slope,
            slope_se=slope_se,
            current_time=timestamp,
        )

        limiting_factor = "GLOBAL_HEALTH_INDEX"

        # 3. Check for short-horizon physical redline breach in forecast
        if forecast is not None and H > 0.0 and forecast.predicted_telemetry:
            future_ts = np.array(forecast.forecast_timestamps, dtype=np.float64)
            future_telemetry = {
                k: np.array(v, dtype=np.float64) for k, v in forecast.predicted_telemetry.items()
            }
            proj_hi = getattr(forecast, "projected_health_trajectory", None) or getattr(forecast, "projected_health_index", None)
            future_hi = np.array(proj_hi or [], dtype=np.float64)
            if len(future_hi) == len(future_ts):
                redline_rul, redline_factor = self.weakest_link.find_earliest_trajectory_crossing(
                    future_timestamps=future_ts,
                    future_health_index=future_hi,
                    future_telemetry_dict=future_telemetry,
                    excluded_channels=excluded_channels,
                    current_time=timestamp,
                )
                if redline_rul is not None and p50 is not None and redline_rul < p50:
                    p50 = redline_rul
                    p05 = min(p05 or redline_rul, redline_rul)
                    p95 = min(p95 or redline_rul, redline_rul)
                    limiting_factor = redline_factor

        # 4. Reporting horizon check
        if p50 is not None and p50 > self.config.maximum_reportable_rul_s:
            status = RULStatus.EXCEEDS_HORIZON
        elif len(excluded_channels) > 0:
            status = RULStatus.DEGRADED_PROGNOSTIC
        else:
            status = RULStatus.ACTIVE_DEGRADATION

        # 5. Base confidence score calculation
        # Factors: length of history and presence of isolated sensor faults
        hist_span = history_ts[-1] - history_ts[0]
        history_factor = min(1.0, hist_span / self.config.theil_sen_window_s)
        base_confidence = 0.5 + 0.5 * history_factor
        if len(excluded_channels) > 0:
            base_confidence *= (1.0 - self.config.confidence_discount_sensor_fault)

        return RULResult(
            engine_id=engine_id,
            mission_id=mission_id,
            timestamp=timestamp,
            status=status,
            rul_seconds_median=p50,
            rul_seconds_p05=p05,
            rul_seconds_p95=p95,
            limiting_factor=limiting_factor,
            confidence_score=round(base_confidence, 3),
            active_flight_phase=flight_phase,
            handoff_horizon_s=H,
            trajectory_type=traj_type,
            provenance={
                "theil_sen_slope": slope,
                "slope_se": slope_se,
                "anchor_time": anchor_t,
                "anchor_hi": anchor_hi,
                "excluded_channels": list(excluded_channels),
            },
        )
