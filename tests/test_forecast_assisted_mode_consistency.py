"""
Comprehensive regression tests for authoritative forecast-assisted mode consistency.
Verifies single source of truth across:
Phase 11 Prognostics -> Orchestrator Payload -> Phase 13 Contracts -> Dashboard Adapter -> UI View Model & Formatters.
Tests all engine states: degrading, stable, recovering, unclear trend, blocked weights, blackout, and redline breaches.
"""

import pytest
import numpy as np
from typing import Dict, Any, List

from prognostics.pipeline import RULPipeline
from prognostics.schema import RULResult, RULStatus, RULConfig
from forecasting.schema import ForecastResult, ModelStatus
from health_index.schema import HealthIndexResult, HealthState, DegradationTrend, HealthDataQuality
from orchestrator.schema import DashboardStatePayload, OrchestratorConfig
from orchestrator.pipeline import SystemPipelineOrchestrator
from dashboard.schemas.contracts import Phase13OutputContract
from dashboard.services.adapter import DashboardAdapter
from dashboard.schemas.view_model import PrognosticsViewModel
from dashboard.utils.formatters import format_forecast_assisted_mode


def make_health_result(
    timestamp: float,
    hi_smooth: float,
    engine_id: str = "ENG_01",
    mission_id: str = "MSN_01",
    health_state: str = "DEGRADED",
    degradation_trend: str = DegradationTrend.DEGRADING.value,
    degradation_rate: float = -0.005,
) -> HealthIndexResult:
    all_ch = ["rpm", "cht", "egt", "oil_temp", "oil_pressure", "fuel_flow", "vibration"]
    return HealthIndexResult(
        timestamp=timestamp,
        engine_id=engine_id,
        mission_id=mission_id,
        mission_phase="CRUISE",
        raw_health_index=hi_smooth,
        smoothed_health_index=hi_smooth,
        raw_degradation_score=1.0 - hi_smooth,
        degradation_rate=degradation_rate,
        degradation_trend=degradation_trend,
        health_state=health_state,
        data_quality=HealthDataQuality.VALID.value,
        valid_channels=all_ch,
        missing_channels=[],
        excluded_channels=[],
        dominant_degraded_channels=["cht", "vibration"] if degradation_trend == DegradationTrend.DEGRADING.value else [],
        channel_contributions={"cht": 0.6, "vibration": 0.4} if degradation_trend == DegradationTrend.DEGRADING.value else {},
        channel_degradation_evidence={"cht": 0.5},
        effective_channel_weights={},
    )


def prime_pipeline(
    pipe: RULPipeline,
    engine_id: str,
    mission_id: str,
    start_hi: float = 0.85,
    end_hi: float = 0.55,
    n_points: int = 35,
    degradation_trend: str = DegradationTrend.DEGRADING.value,
    degradation_rate: float = -0.005,
):
    """Accumulate sufficient history (>= 30s) for degradation slope estimation."""
    ts = np.linspace(1.0, 35.0, n_points)
    his = np.linspace(start_hi, end_hi, n_points)
    for t, h in zip(ts, his):
        hr = make_health_result(
            timestamp=float(t),
            hi_smooth=float(h),
            engine_id=engine_id,
            mission_id=mission_id,
            degradation_trend=degradation_trend,
            degradation_rate=degradation_rate,
        )
        pipe.process_assessment(hr)


def make_forecast(
    status: str,
    source: str,
    horizon: int = 16,
    predicted_redline_breach: bool = False,
) -> ForecastResult:
    """Helper to synthesize a ForecastResult under specified status and source."""
    pred_telemetry = {}
    if status == ModelStatus.LOADED_PRETRAINED.value:
        cht_base = 100.0 if not predicted_redline_breach else 140.0
        cht_step = 0.0 if not predicted_redline_breach else 10.0
        pred_telemetry = {
            "rpm": [2450.0 for _ in range(horizon)],
            "cht": [cht_base + cht_step * i for i in range(horizon)],
            "oil_pressure": [4.5 for _ in range(horizon)],
            "egt": [720.0 for _ in range(horizon)],
            "vibration": [1.2 for _ in range(horizon)],
            "oil_temp": [85.0 for _ in range(horizon)],
            "coolant_temp": [85.0 for _ in range(horizon)],
        }
    return ForecastResult(
        engine_id="ENG_01",
        mission_id="MSN_01",
        forecast_start_timestamp=36.0,
        context_start_timestamp=0.0,
        context_length=32,
        forecast_horizon=horizon,
        sampling_interval=1.0,
        target_channels=["rpm", "cht", "egt", "oil_temp", "oil_pressure", "fuel_flow", "vibration"],
        forecast_timestamps=[36.0 + i for i in range(1, horizon + 1)] if pred_telemetry else [],
        predicted_telemetry=pred_telemetry,
        model_name=source,
        model_status=status,
    )


class TestForecastAssistedModeConsistency:
    """Verifies end-to-end consistency of forecast-assisted mode and status."""

    def setup_method(self):
        self.prognostics_pipeline = RULPipeline(RULConfig())
        self.adapter = DashboardAdapter()

    def test_degrading_engine_with_available_forecast(self):
        """When engine is degrading and forecast is available, forecast-assisted mode must be ACTIVE."""
        prime_pipeline(self.prognostics_pipeline, "ENG_01", "MSN_01", start_hi=0.85, end_hi=0.55, n_points=35)
        health = make_health_result(36.0, 0.54, "ENG_01", "MSN_01", degradation_trend=DegradationTrend.DEGRADING.value)
        forecast = make_forecast(ModelStatus.LOADED_PRETRAINED.value, "TIMESFM_PRETRAINED_CHECKPOINT", horizon=16)

        rul_res = self.prognostics_pipeline.process_assessment(
            health_result=health,
            forecast=forecast,
        )

        assert rul_res.status == RULStatus.ACTIVE_DEGRADATION
        assert rul_res.forecast_assisted is True
        assert rul_res.forecast_assisted_mode is True
        assert rul_res.forecast_mode_status == "ACTIVE"
        assert rul_res.handoff_horizon_s == 16.0

        # Pass through Phase 13 Contract
        contract_from_obj = Phase13OutputContract.from_object({"prognostics": rul_res, "forecast": forecast})
        assert contract_from_obj.forecast_assisted_mode is True
        assert contract_from_obj.forecast_mode_status == "ACTIVE"

        # Pass through Dashboard Adapter
        overview = self.adapter.adapt(contract_from_obj)
        prog_vm = overview.prognostics
        assert prog_vm.forecast_assisted_mode is True
        assert prog_vm.forecast_mode_status == "ACTIVE"
        assert format_forecast_assisted_mode(prog_vm.forecast_assisted_mode, prog_vm.forecast_mode_status) == "ACTIVE"

    def test_degrading_engine_with_blocked_forecast(self):
        """When weights are gated or graph is uncheckpointed, mode must be BLOCKED, not falsely OFF."""
        prime_pipeline(self.prognostics_pipeline, "ENG_01", "MSN_01", start_hi=0.85, end_hi=0.55, n_points=35)
        health = make_health_result(36.0, 0.54, "ENG_01", "MSN_01", degradation_trend=DegradationTrend.DEGRADING.value)
        forecast = make_forecast("BLOCKED_UNAUTHENTICATED_GATED", "TIMESFM_WEIGHTS_GATED", horizon=16)

        rul_res = self.prognostics_pipeline.process_assessment(
            health_result=health,
            forecast=forecast,
        )

        assert rul_res.status == RULStatus.ACTIVE_DEGRADATION
        assert rul_res.forecast_assisted is False
        assert rul_res.forecast_assisted_mode is False
        assert rul_res.forecast_mode_status == "BLOCKED"
        assert rul_res.handoff_horizon_s == 0.0

        contract = Phase13OutputContract.from_object({"prognostics": rul_res, "forecast": forecast})
        assert contract.forecast_assisted_mode is False
        assert contract.forecast_mode_status == "BLOCKED"

        overview = self.adapter.adapt(contract)
        prog_vm = overview.prognostics
        assert prog_vm.forecast_assisted_mode is False
        assert prog_vm.forecast_mode_status == "BLOCKED"
        assert format_forecast_assisted_mode(prog_vm.forecast_assisted_mode, prog_vm.forecast_mode_status) == "BLOCKED"

    def test_degrading_engine_with_unavailable_forecast(self):
        """When forecast data is unavailable or buffering, status must be UNAVAILABLE."""
        prime_pipeline(self.prognostics_pipeline, "ENG_01", "MSN_01", start_hi=0.85, end_hi=0.55, n_points=35)
        health = make_health_result(36.0, 0.54, "ENG_01", "MSN_01", degradation_trend=DegradationTrend.DEGRADING.value)
        forecast = make_forecast("INSUFFICIENT_CONTEXT", "TIMESFM", horizon=16)

        rul_res = self.prognostics_pipeline.process_assessment(
            health_result=health,
            forecast=forecast,
        )

        assert rul_res.status == RULStatus.ACTIVE_DEGRADATION
        assert rul_res.forecast_assisted is False
        assert rul_res.forecast_mode_status == "UNAVAILABLE"

        contract = Phase13OutputContract.from_object({"prognostics": rul_res, "forecast": forecast})
        assert contract.forecast_assisted_mode is False
        assert contract.forecast_mode_status == "UNAVAILABLE"

        overview = self.adapter.adapt(contract)
        prog_vm = overview.prognostics
        assert prog_vm.forecast_assisted_mode is False
        assert prog_vm.forecast_mode_status == "UNAVAILABLE"
        assert format_forecast_assisted_mode(prog_vm.forecast_assisted_mode, prog_vm.forecast_mode_status) == "UNAVAILABLE"

    def test_stable_engine_nominal_causal_trend(self):
        """When engine is stable and no redline breach is predicted, mode is OFF (Causal Trend)."""
        prime_pipeline(self.prognostics_pipeline, "ENG_01", "MSN_01", start_hi=0.95, end_hi=0.95, n_points=35,
                       degradation_trend=DegradationTrend.STABLE.value, degradation_rate=0.0)
        health = make_health_result(36.0, 0.95, "ENG_01", "MSN_01",
                                    health_state=HealthState.HEALTHY.value,
                                    degradation_trend=DegradationTrend.STABLE.value,
                                    degradation_rate=0.0)
        forecast = make_forecast(ModelStatus.LOADED_PRETRAINED.value, "TIMESFM_PRETRAINED_CHECKPOINT", horizon=16)

        rul_res = self.prognostics_pipeline.process_assessment(
            health_result=health,
            forecast=forecast,
        )

        assert rul_res.status == RULStatus.NOT_DEGRADING
        assert rul_res.forecast_assisted is False
        assert rul_res.forecast_mode_status == "OFF"

        contract = Phase13OutputContract.from_object({"prognostics": rul_res, "forecast": forecast})
        assert contract.forecast_assisted_mode is False
        assert contract.forecast_mode_status == "OFF"

        overview = self.adapter.adapt(contract)
        prog_vm = overview.prognostics
        assert prog_vm.forecast_assisted_mode is False
        assert prog_vm.forecast_mode_status == "OFF"
        assert format_forecast_assisted_mode(prog_vm.forecast_assisted_mode, prog_vm.forecast_mode_status) == "OFF (Causal Trend)"

    def test_recovering_engine_nominal_causal_trend(self):
        """When engine is recovering, mode is OFF (Causal Trend)."""
        prime_pipeline(self.prognostics_pipeline, "ENG_01", "MSN_01", start_hi=0.70, end_hi=0.85, n_points=35,
                       degradation_trend=DegradationTrend.IMPROVING.value, degradation_rate=0.005)
        health = make_health_result(36.0, 0.85, "ENG_01", "MSN_01",
                                    health_state=HealthState.HEALTHY.value,
                                    degradation_trend=DegradationTrend.IMPROVING.value,
                                    degradation_rate=0.005)
        forecast = make_forecast(ModelStatus.LOADED_PRETRAINED.value, "TIMESFM_PRETRAINED_CHECKPOINT", horizon=16)

        rul_res = self.prognostics_pipeline.process_assessment(
            health_result=health,
            forecast=forecast,
        )

        assert rul_res.status == RULStatus.RECOVERING
        assert rul_res.forecast_assisted is False
        assert rul_res.forecast_mode_status == "OFF"

        contract = Phase13OutputContract.from_object({"prognostics": rul_res, "forecast": forecast})
        assert contract.forecast_assisted_mode is False
        assert contract.forecast_mode_status == "OFF"

        overview = self.adapter.adapt(contract)
        prog_vm = overview.prognostics
        assert prog_vm.forecast_assisted_mode is False
        assert prog_vm.forecast_mode_status == "OFF"
        assert format_forecast_assisted_mode(prog_vm.forecast_assisted_mode, prog_vm.forecast_mode_status) == "OFF (Causal Trend)"

    def test_unclear_trend_mode(self):
        """When degradation trend is indeterminate, mode is OFF."""
        prime_pipeline(self.prognostics_pipeline, "ENG_01", "MSN_01", start_hi=0.60, end_hi=0.60, n_points=10,
                       degradation_trend=DegradationTrend.INSUFFICIENT_HISTORY.value, degradation_rate=0.0)
        health = make_health_result(11.0, 0.60, "ENG_01", "MSN_01",
                                    degradation_trend=DegradationTrend.INSUFFICIENT_HISTORY.value,
                                    degradation_rate=0.0)
        forecast = make_forecast(ModelStatus.LOADED_PRETRAINED.value, "TIMESFM_PRETRAINED_CHECKPOINT", horizon=16)

        rul_res = self.prognostics_pipeline.process_assessment(
            health_result=health,
            forecast=forecast,
        )

        assert rul_res.status == RULStatus.INSUFFICIENT_HISTORY
        assert rul_res.forecast_assisted is False
        assert rul_res.forecast_mode_status == "OFF"

        contract = Phase13OutputContract.from_object({"prognostics": rul_res, "forecast": forecast})
        overview = self.adapter.adapt(contract)
        prog_vm = overview.prognostics
        assert prog_vm.forecast_assisted_mode is False
        assert prog_vm.forecast_mode_status == "OFF"
        assert format_forecast_assisted_mode(prog_vm.forecast_assisted_mode, prog_vm.forecast_mode_status) == "OFF (Causal Trend)"

    def test_forecast_predicts_acute_redline_breach_in_stable_engine(self):
        """If engine is currently stable but forecast predicts imminent physical redline crossing, forecast IS used."""
        prime_pipeline(self.prognostics_pipeline, "ENG_01", "MSN_01", start_hi=0.90, end_hi=0.90, n_points=35,
                       degradation_trend=DegradationTrend.STABLE.value, degradation_rate=0.0)
        health = make_health_result(36.0, 0.90, "ENG_01", "MSN_01",
                                    health_state=HealthState.HEALTHY.value,
                                    degradation_trend=DegradationTrend.STABLE.value,
                                    degradation_rate=0.0)
        # Predicted telemetry breaches CHT redline (>150.0) in step 1
        forecast = make_forecast(ModelStatus.LOADED_PRETRAINED.value, "TIMESFM_PRETRAINED_CHECKPOINT", horizon=16, predicted_redline_breach=True)

        rul_res = self.prognostics_pipeline.process_assessment(
            health_result=health,
            forecast=forecast,
        )

        # Because redline breach occurs within forecast horizon, forecast MUST be utilized
        assert rul_res.forecast_assisted is True
        assert rul_res.forecast_mode_status == "ACTIVE"
        assert rul_res.status == RULStatus.ACTIVE_DEGRADATION
        assert "REDLINE" in rul_res.limiting_factor

        contract = Phase13OutputContract.from_object({"prognostics": rul_res, "forecast": forecast})
        overview = self.adapter.adapt(contract)
        prog_vm = overview.prognostics
        assert prog_vm.forecast_assisted_mode is True
        assert prog_vm.forecast_mode_status == "ACTIVE"
        assert format_forecast_assisted_mode(prog_vm.forecast_assisted_mode, prog_vm.forecast_mode_status) == "ACTIVE"

    def test_phase13_contracts_serialization_roundtrip(self):
        """Test round-trip fidelity between dict, object, contract, and adapter."""
        rul_dict = {
            "status": "ACTIVE_DEGRADATION",
            "state": "ACTIVE_DEGRADATION",
            "point_rul_seconds": 1200.0,
            "forecast_assisted": True,
            "forecast_assisted_mode": True,
            "forecast_mode_status": "ACTIVE",
            "limiting_factor": "HEALTH_INDEX_0.35_THRESHOLD",
        }
        payload_dict = {
            "timestamp": 1234.5,
            "engine_id": "ENG-SERIALIZATION",
            "prognostics": rul_dict,
            "forecast_assisted_mode": True,
            "forecast_mode_status": "ACTIVE",
        }

        # Dict -> Contract
        contract = Phase13OutputContract.from_dict(payload_dict)
        assert contract.forecast_assisted_mode is True
        assert contract.forecast_mode_status == "ACTIVE"

        # Contract -> Object conversion
        contract2 = Phase13OutputContract.from_object(contract)
        assert contract2.forecast_assisted_mode is True
        assert contract2.forecast_mode_status == "ACTIVE"

        # Adapter adaptation
        overview = self.adapter.adapt(contract2)
        prog_vm = overview.prognostics
        assert prog_vm.forecast_assisted_mode is True
        assert prog_vm.forecast_mode_status == "ACTIVE"
        assert format_forecast_assisted_mode(prog_vm.forecast_assisted_mode, prog_vm.forecast_mode_status) == "ACTIVE"

    def test_sensor_blackout_sets_forecast_unavailable(self):
        """During blackout or invalid timestamp, payload and contract must explicitly set UNAVAILABLE."""
        orch = SystemPipelineOrchestrator(
            config=OrchestratorConfig(auto_bootstrap_on_init=True, deterministic_seed=42)
        )
        blackout_telemetry = {
            "timestamp": 1000.0,
            "engine_id": "ENG-BLACKOUT",
            "rpm": None,
            "cht": None,
            "oil_pressure": None,
            "oil_temp": None,
            "coolant_temp": None,
            "vibration": None,
            "egt": None,
        }
        payload = orch.step(blackout_telemetry)
        assert payload.forecast_assisted_mode is False
        assert payload.forecast_mode_status == "UNAVAILABLE"

        contract = Phase13OutputContract.from_object(payload)
        assert contract.forecast_assisted_mode is False
        assert contract.forecast_mode_status == "UNAVAILABLE"

        overview = self.adapter.adapt(contract)
        prog_vm = overview.prognostics
        assert prog_vm.forecast_assisted_mode is False
        assert prog_vm.forecast_mode_status == "UNAVAILABLE"
        assert format_forecast_assisted_mode(prog_vm.forecast_assisted_mode, prog_vm.forecast_mode_status) == "UNAVAILABLE"
