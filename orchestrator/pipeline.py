"""
SystemPipelineOrchestrator for Phase 13.
Unified causal execution layer that binds Phase 6 through Phase 12 into a single
real-time aero-piston digital twin system pipeline.
"""

from __future__ import annotations

import time
import math
from typing import Dict, Any, Optional, List, Union, Tuple
import pandas as pd
import numpy as np

from telemetry.schema import TelemetryRecord, EngineConfig
from digital_twin.twin_model import DigitalTwin
from digital_twin.residuals import ResidualFrame
from anomaly_detection.pipeline import HybridAnomalyDetector
from fault_diagnosis.pipeline import FaultDiagnosisPipeline
from fault_diagnosis.schema import FaultDiagnosisResult
from health_index.pipeline import HealthIndexPipeline
from health_index.schema import HealthIndexResult
from forecasting.pipeline import ForecastingPipeline
from forecasting.schema import ForecastingConfig, ForecastResult, ModelStatus
from prognostics.pipeline import RULPipeline
from prognostics.schema import RULConfig, RULResult
from explainability.pipeline import ExplainabilityPipeline
from explainability.schema import ExplainabilityResult

from orchestrator.schema import (
    DashboardStatePayload,
    OrchestratorConfig,
    SimulationScenario,
    ScenarioFaultType,
)
from orchestrator.bootstrap import SyntheticBootstrapManager
from orchestrator.adapter import PipelineHandoffAdapter
from orchestrator.advisor import OperatorActionAdvisor

from simulator.engine_simulator import EngineSimulator
from simulator.subsystems.mission import MissionProfile, PhaseSegment, FlightPhase
from simulator.fault_interface import (
    FaultSchedule,
    FaultState,
    FaultType,
    FaultSubsystem,
)


class SystemPipelineOrchestrator:
    """
    Production integration orchestrator for SIH26054.
    Connects Phase 6 (Digital Twin) -> Phase 7 (Anomaly) -> Phase 8 (Diagnosis) ->
    Phase 9 (Health) -> Phase 10 (Forecast) -> Phase 11 (RUL) -> Phase 12 (Explainability) ->
    DashboardStatePayload.
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        twin: Optional[DigitalTwin] = None,
        anomaly_detector: Optional[HybridAnomalyDetector] = None,
        diagnosis_pipeline: Optional[FaultDiagnosisPipeline] = None,
        health_pipeline: Optional[HealthIndexPipeline] = None,
        forecasting_pipeline: Optional[ForecastingPipeline] = None,
        rul_pipeline: Optional[RULPipeline] = None,
        explainability_pipeline: Optional[ExplainabilityPipeline] = None,
    ):
        self.config = config or OrchestratorConfig()

        # Session tracking for causal validation and engine/mission isolation
        self.active_engine_id: Optional[str] = None
        self.active_mission_id: Optional[str] = None
        self.last_timestamp: Optional[float] = None
        self.step_count: int = 0
        self.bootstrap_metadata: Dict[str, Any] = {}

        # 1. Phase 6: Digital Twin
        self.twin = twin or DigitalTwin()

        # 2 & 3. Phase 7 & 8: Anomaly Detection & Fault Diagnosis
        if anomaly_detector is None or diagnosis_pipeline is None:
            if self.config.auto_bootstrap_on_init:
                (
                    self.anomaly_detector,
                    self.diagnosis_pipeline,
                    self.bootstrap_metadata,
                ) = SyntheticBootstrapManager.bootstrap_models(
                    seed=self.config.deterministic_seed,
                    fast_mode=True,
                    enable_phase7_gating=self.config.enable_phase7_gating,
                    n_estimators_xgb=self.config.xgboost_n_estimators,
                    max_depth_xgb=self.config.xgboost_max_depth,
                )
            else:
                self.anomaly_detector = anomaly_detector or HybridAnomalyDetector()
                self.diagnosis_pipeline = diagnosis_pipeline or FaultDiagnosisPipeline(
                    classifier=None, feature_extractor=None
                )
        else:
            self.anomaly_detector = anomaly_detector
            self.diagnosis_pipeline = diagnosis_pipeline

        # 4. Phase 9: Health Index & Degradation Tracking
        self.health_pipeline = health_pipeline or HealthIndexPipeline()

        # 5. Phase 10: Forecasting
        forecasting_cfg = ForecastingConfig(
            context_length=self.config.forecast_context_length,
            forecast_horizon=self.config.forecast_horizon,
            sampling_interval_s=self.config.sampling_dt,
        )
        self.forecasting_pipeline = forecasting_pipeline or ForecastingPipeline(
            config=forecasting_cfg
        )

        # 6. Phase 11: Prognostics / RUL
        rul_cfg = RULConfig(
            warmup_duration_s=self.config.rul_warmup_duration_s,
            theil_sen_window_s=self.config.theil_sen_window_s,
        )
        self.rul_pipeline = rul_pipeline or RULPipeline(config=rul_cfg)

        # 7. Phase 12: Explainability
        # Connect Phase 8 classifier if available for SHAP attribution
        clf = getattr(self.diagnosis_pipeline, "classifier", None)
        self.explainability_pipeline = explainability_pipeline or ExplainabilityPipeline(
            classifier=clf,
            top_k_shap=self.config.shap_top_k,
            tau_threshold=self.config.physics_tau_threshold,
        )

    def reset(self, engine_id: Optional[str] = None, mission_id: Optional[str] = None) -> None:
        """
        Coordinate state reset across all stateful pipeline modules for mission/engine isolation.
        """
        self.twin.reset()
        self.anomaly_detector.reset()
        self.health_pipeline.reset(engine_id=engine_id, mission_id=mission_id)
        self.forecasting_pipeline.reset(engine_id=engine_id, mission_id=mission_id)
        self.rul_pipeline.reset(engine_id=engine_id, mission_id=mission_id)
        self.explainability_pipeline.reset(engine_id=engine_id, mission_id=mission_id)

        self.last_timestamp = None
        self.step_count = 0
        if engine_id is not None:
            self.active_engine_id = engine_id
        if mission_id is not None:
            self.active_mission_id = mission_id

    def step(
        self,
        telemetry: Union[TelemetryRecord, Dict[str, Any]],
        scenario_metadata: Optional[Dict[str, Any]] = None,
    ) -> DashboardStatePayload:
        """
        Execute one causal pipeline step for a single telemetry observation.

        Guarantees:
        - Causality: only current and past data available at or before t influence t.
        - Isolation: (engine_id, mission_id) state boundary isolation.
        - Zero leakage: simulation ground-truth fault labels are stripped before inference.
        """
        start_time = time.perf_counter()

        # 1. Parse telemetry dictionary and identify session
        if isinstance(telemetry, TelemetryRecord):
            raw_dict = telemetry.to_dict()
        else:
            raw_dict = dict(telemetry)

        engine_id = str(raw_dict.get("engine_id", self.config.default_engine_id))
        mission_id = str(raw_dict.get("mission_id", self.config.default_mission_id))
        timestamp = float(raw_dict.get("timestamp", 0.0))
        mission_phase = str(raw_dict.get("mission_phase", "CRUISE"))

        # Check session boundary transition -> isolated reset if new mission/engine
        if (
            self.active_engine_id is not None
            and (self.active_engine_id != engine_id or self.active_mission_id != mission_id)
        ):
            self.reset(engine_id=engine_id, mission_id=mission_id)

        self.active_engine_id = engine_id
        self.active_mission_id = mission_id

        # Check causal time monotonicity
        if self.last_timestamp is not None and timestamp < self.last_timestamp:
            # Out-of-order observation: return degraded payload rejecting future/reversal leakage
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            return self._build_out_of_order_payload(
                engine_id=engine_id,
                mission_id=mission_id,
                timestamp=timestamp,
                mission_phase=mission_phase,
                raw_dict=raw_dict,
                latency_ms=latency_ms,
            )

        self.last_timestamp = timestamp
        self.step_count += 1

        # 2. Strict ground-truth sanitization: excise any fault injection metadata
        clean_telemetry = PipelineHandoffAdapter.sanitize_telemetry_for_inference(raw_dict)

        # 3. Phase 6: Digital Twin & Residual Generation
        (
            residual_frame,
            expected_dict,
            raw_residuals,
            norm_residuals,
        ) = PipelineHandoffAdapter.step_digital_twin_streaming(self.twin, clean_telemetry)

        # 4. Phase 7: Hybrid Anomaly Detection
        # reset_state=False preserves causal EWMA and persistence counts
        anomaly_dict = PipelineHandoffAdapter.step_anomaly_detection_streaming(
            self.anomaly_detector, residual_frame
        )
        anomaly_status = anomaly_dict["anomaly_status"]
        anomaly_score = anomaly_dict["anomaly_score"]

        # 5. Phase 8: Supervised Multiclass Fault Diagnosis
        diagnosis_result = PipelineHandoffAdapter.step_fault_diagnosis(
            self.diagnosis_pipeline,
            residual_frame=residual_frame,
            anomaly_status=anomaly_status,
            anomaly_score=anomaly_score,
        )

        # 6. Phase 9: Health Index & Causal Degradation Tracking
        health_result = PipelineHandoffAdapter.step_health_index(
            self.health_pipeline,
            residual_frame=residual_frame,
            diagnosis_result=diagnosis_result,
            anomaly_status=anomaly_status,
            anomaly_score=anomaly_score,
        )

        # 7. Phase 10: Forecasting
        forecast_result = PipelineHandoffAdapter.step_forecasting(
            self.forecasting_pipeline,
            telemetry_dict=clean_telemetry,
            optional_context={"anomaly_status": anomaly_status, "fault_type": diagnosis_result.predicted_fault_type},
        )

        # 8. Phase 11: Authoritative RUL & Weakest-Link EOL
        rul_result = PipelineHandoffAdapter.step_rul(
            self.rul_pipeline,
            health_result=health_result,
            forecast_result=forecast_result,
            telemetry_dict=clean_telemetry,
        )

        # 9. Phase 12: Explainability & Multi-Modal Evidence Fusion
        # Extract Phase 8 16-feature vector for SHAP attribution
        features_df = self.diagnosis_pipeline.feature_extractor.transform(residual_frame.to_dataframe())
        explainability_result = PipelineHandoffAdapter.step_explainability(
            self.explainability_pipeline,
            timestamp=timestamp,
            engine_id=engine_id,
            mission_id=mission_id,
            norm_residuals=norm_residuals,
            features_df=features_df,
            diagnosis_result=diagnosis_result,
            health_result=health_result,
            forecast_result=forecast_result,
            rul_result=rul_result,
            classifier=self.diagnosis_pipeline.classifier,
        )

        # 10. Operator Advisory Decision Support (Section 15)
        advisory = OperatorActionAdvisor.generate_advisory(
            anomaly_status=anomaly_status,
            diagnosis_fault=diagnosis_result.predicted_fault_type,
            diagnosis_confidence=diagnosis_result.diagnostic_confidence,
            health_state=health_result.health_state.value if hasattr(health_result.health_state, "value") else str(health_result.health_state),
            rul_status=rul_result.status.value if hasattr(rul_result.status, "value") else str(rul_result.status),
            dominant_channels=health_result.dominant_degraded_channels,
            recommended_action_from_phase12=explainability_result.summary_explanation,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # 11. Assemble authoritative DashboardStatePayload
        return self._assemble_payload(
            engine_id=engine_id,
            mission_id=mission_id,
            timestamp=timestamp,
            mission_phase=mission_phase,
            clean_telemetry=clean_telemetry,
            expected_dict=expected_dict,
            raw_residuals=raw_residuals,
            norm_residuals=norm_residuals,
            anomaly_dict=anomaly_dict,
            diagnosis_result=diagnosis_result,
            health_result=health_result,
            forecast_result=forecast_result,
            rul_result=rul_result,
            explainability_result=explainability_result,
            advisory=advisory,
            scenario_metadata=scenario_metadata or {},
            latency_ms=latency_ms,
            raw_telemetry=telemetry,
            residual_frame=residual_frame,
        )

    def _assemble_payload(
        self,
        engine_id: str,
        mission_id: Optional[str],
        timestamp: float,
        mission_phase: str,
        clean_telemetry: Dict[str, Any],
        expected_dict: Dict[str, float],
        raw_residuals: Dict[str, float],
        norm_residuals: Dict[str, float],
        anomaly_dict: Dict[str, Any],
        diagnosis_result: FaultDiagnosisResult,
        health_result: HealthIndexResult,
        forecast_result: Optional[ForecastResult],
        rul_result: RULResult,
        explainability_result: ExplainabilityResult,
        advisory: Any,
        scenario_metadata: Dict[str, Any],
        latency_ms: float,
        raw_telemetry: Any,
        residual_frame: ResidualFrame,
    ) -> DashboardStatePayload:
        """Helper to construct DashboardStatePayload containing authoritative results."""
        # Forecast unpacking
        if forecast_result is not None:
            fc_status = forecast_result.model_status
            fc_source = forecast_result.model_name
            fc_horizon = forecast_result.forecast_horizon
            fc_preds = forecast_result.predicted_telemetry
            fc_ts = forecast_result.forecast_timestamps
            fc_qual = forecast_result.forecast_quality
            fc_pretrained = (forecast_result.model_status == ModelStatus.LOADED_PRETRAINED.value)
        else:
            fc_status = "BUFFERING"
            fc_source = "TIMESFM_OR_BASELINE"
            fc_horizon = self.config.forecast_horizon
            fc_preds = None
            fc_ts = None
            fc_qual = "INSUFFICIENT_CONTEXT"
            fc_pretrained = False

        # Explainability evidence unpacking
        shap_attr = None
        if explainability_result.shap_evidence is not None:
            shap_attr = {
                "top_features": [
                    {
                        "feature": f.feature_name,
                        "shap_value": f.shap_value,
                        "feature_value": f.feature_value,
                        "direction": f.direction,
                        "relative_weight": f.relative_weight,
                    }
                    for f in explainability_result.shap_evidence.top_features
                ],
                "status": explainability_result.shap_evidence.status,
            }

        phys_ev = None
        if explainability_result.physics_evidence is not None:
            status_val = (
                explainability_result.physics_evidence.status.value
                if hasattr(explainability_result.physics_evidence.status, "value")
                else str(explainability_result.physics_evidence.status)
            )
            phys_ev = {
                "status": status_val,
                "diagnosed_fault": explainability_result.physics_evidence.diagnosed_fault,
                "consistency_reason": explainability_result.physics_evidence.consistency_reason,
                "supporting_channels": explainability_result.physics_evidence.supporting_channels,
                "conflicting_channels": explainability_result.physics_evidence.conflicting_channels,
            }

        temp_ev = None
        if explainability_result.temporal_evidence is not None:
            temp_ev = {
                "status": explainability_result.temporal_evidence.status,
                "degradation_rate": explainability_result.temporal_evidence.degradation_rate,
                "degradation_trend": explainability_result.temporal_evidence.degradation_trend,
                "trend_direction": explainability_result.temporal_evidence.trend_direction,
                "forecast_status": explainability_result.temporal_evidence.forecast_status,
            }

        return DashboardStatePayload(
            engine_id=engine_id,
            mission_id=mission_id,
            timestamp=timestamp,
            mission_phase=mission_phase,
            simulation_mode="SYNTHETIC_SIMULATION",
            scenario_metadata=scenario_metadata,
            observed_telemetry={
                k: float(v) for k, v in clean_telemetry.items() if isinstance(v, (int, float)) and not math.isnan(v)
            },
            quality_status=str(clean_telemetry.get("quality_status", "NOMINAL")),
            expected_telemetry=expected_dict,
            residuals=raw_residuals,
            normalized_residuals=norm_residuals,
            anomaly_status=anomaly_dict["anomaly_status"],
            anomaly_score=anomaly_dict["anomaly_score"],
            anomaly_contributing_channels=anomaly_dict["contributing_channels"],
            persistence_count=anomaly_dict["persistence_count"],
            anomaly_evidence=anomaly_dict["evidence"],
            predicted_fault_class=diagnosis_result.predicted_fault_type,
            diagnosis_probabilities=diagnosis_result.class_probabilities,
            diagnostic_confidence=diagnosis_result.diagnostic_confidence,
            diagnosis_data_quality=diagnosis_result.data_quality,
            raw_health_index=health_result.raw_health_index,
            smoothed_health_index=health_result.smoothed_health_index,
            health_state=health_result.health_state.value if hasattr(health_result.health_state, "value") else str(health_result.health_state),
            degradation_rate=health_result.degradation_rate,
            degradation_trend=health_result.degradation_trend.value if hasattr(health_result.degradation_trend, "value") else str(health_result.degradation_trend),
            dominant_channels=health_result.dominant_degraded_channels,
            channel_contributions=health_result.channel_contributions,
            forecast_status=fc_status,
            forecast_source=fc_source,
            forecast_horizon=fc_horizon,
            predicted_telemetry=fc_preds,
            forecast_timestamps=fc_ts,
            is_pretrained=fc_pretrained,
            forecast_quality=fc_qual,
            rul_state=rul_result.status.value if hasattr(rul_result.status, "value") else str(rul_result.status),
            point_rul_seconds=rul_result.rul_seconds_median,
            rul_uncertainty_p05=rul_result.rul_seconds_p05,
            rul_uncertainty_p95=rul_result.rul_seconds_p95,
            limiting_factor=rul_result.limiting_factor,
            forecast_assisted_mode=(rul_result.handoff_horizon_s > 0.0),
            eol_provenance=rul_result.provenance,
            summary_explanation=explainability_result.summary_explanation,
            shap_attribution=shap_attr,
            physics_evidence=phys_ev,
            temporal_evidence=temp_ev,
            recommended_operator_action=advisory.recommended_action,
            advisory=advisory,
            provenance={
                "step_count": self.step_count,
                "bootstrap": self.bootstrap_metadata,
                "source_type": "synthetic",
                "disclaimer": "Project-defined simulation criteria, not certified OEM/FAA limits.",
            },
            execution_latency_ms=round(latency_ms, 3),
            _raw_telemetry=raw_telemetry,
            _residual_frame=residual_frame,
            _diagnosis_result=diagnosis_result,
            _health_result=health_result,
            _forecast_result=forecast_result,
            _rul_result=rul_result,
            _explainability_result=explainability_result,
        )

    def _build_out_of_order_payload(
        self,
        engine_id: str,
        mission_id: Optional[str],
        timestamp: float,
        mission_phase: str,
        raw_dict: Dict[str, Any],
        latency_ms: float,
    ) -> DashboardStatePayload:
        """Construct error payload for rejected out-of-order timestamp."""
        advisory = OperatorActionAdvisor.generate_advisory(
            anomaly_status="INSUFFICIENT_DATA",
            diagnosis_fault="none",
            diagnosis_confidence=0.0,
            health_state="INSUFFICIENT_DATA",
            rul_status="INSUFFICIENT_DATA",
            recommended_action_from_phase12="Out-of-order observation rejected by causal sequence guard.",
        )
        return DashboardStatePayload(
            engine_id=engine_id,
            mission_id=mission_id,
            timestamp=timestamp,
            mission_phase=mission_phase,
            simulation_mode="SYNTHETIC_SIMULATION",
            observed_telemetry={k: float(v) for k, v in raw_dict.items() if isinstance(v, (int, float)) and not math.isnan(v)},
            quality_status="OUT_OF_ORDER_REJECTED",
            anomaly_status="INSUFFICIENT_DATA",
            predicted_fault_class="none",
            diagnostic_confidence=0.0,
            diagnosis_data_quality="OUT_OF_ORDER",
            raw_health_index=float("nan"),
            smoothed_health_index=float("nan"),
            health_state="INSUFFICIENT_DATA",
            degradation_trend="INDETERMINATE",
            forecast_status="REJECTED_OUT_OF_ORDER",
            rul_state="INSUFFICIENT_DATA",
            summary_explanation="Rejected observation: timestamp earlier than last observed causal timestep.",
            recommended_operator_action="Verify chronological integrity of telemetry source.",
            advisory=advisory,
            execution_latency_ms=round(latency_ms, 3),
        )

    def run_simulation(
        self,
        scenario: Optional[SimulationScenario] = None,
        dt: Optional[float] = None,
        duration_s: Optional[float] = None,
        fault_type: Optional[ScenarioFaultType] = None,
        fault_start_s: Optional[float] = None,
        fault_severity: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> List[DashboardStatePayload]:
        """
        Execute an end-to-end simulation flight run and causally stream observations through the orchestrator.
        """
        sc = scenario or SimulationScenario()
        sim_dt = dt or sc.dt
        sim_duration = duration_s or sc.duration_s
        sim_fault_type = fault_type or sc.fault_type
        sim_fault_start = fault_start_s if fault_start_s is not None else sc.fault_start_s
        sim_severity = fault_severity if fault_severity is not None else sc.fault_severity
        sim_seed = seed if seed is not None else sc.seed

        # 1. Configure simulator fault schedule
        fault_schedule = None
        if sim_fault_type != ScenarioFaultType.HEALTHY:
            fault_schedule = self._build_fault_schedule(
                fault_type=sim_fault_type,
                fault_start_s=sim_fault_start,
                severity=sim_severity,
            )

        # 2. Configure Mission Profile
        airspeed = getattr(sc, "airspeed_ms", 40.0)
        seg = PhaseSegment(
            phase=FlightPhase.CRUISE,
            duration_s=sim_duration,
            throttle_start_pct=sc.throttle_pct,
            throttle_end_pct=sc.throttle_pct,
            altitude_start_m=sc.altitude_m,
            altitude_end_m=sc.altitude_m,
            airspeed_start_ms=airspeed,
            airspeed_end_ms=airspeed,
        )
        profile = MissionProfile(
            mission_id=sc.mission_id,
            segments=[seg],
        )

        # 3. Reset orchestrator for this mission
        self.reset(engine_id=sc.engine_id, mission_id=sc.mission_id)

        # 4. Run simulator and causally step through pipeline
        sim = EngineSimulator(seed=sim_seed)
        telemetry_stream = sim.run(profile, dt=sim_dt, fault_schedule=fault_schedule)

        scenario_meta = {
            "scenario_name": sc.name,
            "injected_fault": sim_fault_type.value,
            "injected_severity": sim_severity if sim_fault_type != ScenarioFaultType.HEALTHY else 0.0,
            "fault_start_s": sim_fault_start if sim_fault_type != ScenarioFaultType.HEALTHY else None,
        }

        payloads: List[DashboardStatePayload] = []
        for record in telemetry_stream:
            payload = self.step(record, scenario_metadata=scenario_meta)
            payloads.append(payload)

        return payloads

    def _build_fault_schedule(
        self,
        fault_type: ScenarioFaultType,
        fault_start_s: float,
        severity: float,
    ) -> FaultSchedule:
        """Helper to build simulator FaultSchedule without exposing labels to inference."""
        sched = FaultSchedule()
        if fault_type == ScenarioFaultType.COOLING_DEGRADATION:
            sched.add_fault(
                FaultState(
                    fault_type=FaultType.COOLING_DEGRADATION,
                    severity=severity,
                    start_time=fault_start_s,
                    affected_subsystem=FaultSubsystem.THERMAL,
                )
            )
        elif fault_type == ScenarioFaultType.LUBRICATION_DEGRADATION:
            sched.add_fault(
                FaultState(
                    fault_type=FaultType.LUBRICATION_DEGRADATION,
                    severity=severity,
                    start_time=fault_start_s,
                    affected_subsystem=FaultSubsystem.LUBRICATION,
                )
            )
        elif fault_type == ScenarioFaultType.FUEL_ABNORMALITY:
            sched.add_fault(
                FaultState(
                    fault_type=FaultType.FUEL_INJECTION_ABNORMALITY,
                    severity=severity,
                    start_time=fault_start_s,
                    affected_subsystem=FaultSubsystem.FUEL,
                    parameters={"mode": "lean"},
                )
            )
        elif fault_type == ScenarioFaultType.MECHANICAL_DEGRADATION:
            sched.add_fault(
                FaultState(
                    fault_type=FaultType.MECHANICAL_DEGRADATION,
                    severity=severity,
                    start_time=fault_start_s,
                    affected_subsystem=FaultSubsystem.VIBRATION,
                )
            )
        elif fault_type == ScenarioFaultType.SENSOR_FAULT:
            sched.add_fault(
                FaultState(
                    fault_type=FaultType.SENSOR_FAULT,
                    severity=severity,
                    start_time=fault_start_s,
                    affected_subsystem=FaultSubsystem.SENSOR,
                    parameters={"sensor_channel": "cht", "sensor_mode": "bias"},
                )
            )
        return sched

