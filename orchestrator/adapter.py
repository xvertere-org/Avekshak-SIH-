"""
Interface adapter layer for Phase 13 Unified System Pipeline Orchestrator.
Performs clean, non-mutating data transformations and orchestrates handoffs
between Phase 6 through Phase 12 while enforcing strict causality and zero ground-truth leakage.
"""

import math
from typing import Dict, Any, Optional, Tuple, List, Union
import pandas as pd
import numpy as np

from telemetry.schema import TelemetryRecord, DigitalTwinState
from telemetry.ingestion import CanonicalTelemetryFrame
from digital_twin.twin_model import DigitalTwin
from digital_twin.residuals import ResidualFrame, ResidualGenerator, SUPPORTED_RESIDUAL_CHANNELS

from anomaly_detection.pipeline import HybridAnomalyDetector
from anomaly_detection.schema import AnomalyFrame, AnomalyRecord, AnomalyStatus

from fault_diagnosis.pipeline import FaultDiagnosisPipeline
from fault_diagnosis.schema import FaultDiagnosisResult, DiagnosisDataQuality
from fault_diagnosis.features import FeatureExtractor

from health_index.pipeline import HealthIndexPipeline
from health_index.schema import HealthIndexResult, HealthState, DegradationTrend

from forecasting.pipeline import ForecastingPipeline
from forecasting.schema import ForecastResult, ForecastQuality, ModelStatus

from prognostics.pipeline import RULPipeline
from prognostics.schema import RULResult, RULStatus

from explainability.pipeline import ExplainabilityPipeline
from explainability.schema import ExplainabilityResult


class PipelineHandoffAdapter:
    """
    Coordinates safe, type-checked handoffs between Phase 6–12 modules.
    Guarantees no modification of frozen algorithm contracts.
    """

    @staticmethod
    def sanitize_telemetry_for_inference(telemetry: Union[TelemetryRecord, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Strip ground-truth fault simulation labels to prevent any possibility of inference leakage.
        """
        if isinstance(telemetry, TelemetryRecord):
            data = telemetry.to_dict()
        else:
            data = dict(telemetry)

        # Strictly excise simulation ground-truth labels
        data.pop("fault_type", None)
        data.pop("fault_severity", None)
        data.pop("fault_subsystem", None)
        data.pop("injection_metadata", None)

        return data

    @staticmethod
    def step_digital_twin_streaming(
        twin: DigitalTwin,
        telemetry_dict: Dict[str, Any],
    ) -> Tuple[ResidualFrame, Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Causally update Digital Twin model state and compute ResidualFrame for a single observation.
        Does NOT call predict_expected (which would reset nominal state tracking).
        """
        timestamp = float(telemetry_dict.get("timestamp", 0.0))
        raw_th = telemetry_dict.get("throttle")
        throttle = float(raw_th) if raw_th is not None and math.isfinite(float(raw_th)) else 75.0
        raw_alt = telemetry_dict.get("altitude")
        altitude = float(raw_alt) if raw_alt is not None and math.isfinite(float(raw_alt)) else 2000.0
        raw_amb = telemetry_dict.get("ambient_temp")
        ambient_temp = float(raw_amb) if raw_amb is not None and math.isfinite(float(raw_amb)) else 15.0
        phase = str(telemetry_dict.get("mission_phase", "CRUISE"))

        # Compute dynamic dt
        if twin.last_timestamp is not None:
            if timestamp > twin.last_timestamp:
                dt = timestamp - twin.last_timestamp
                twin.last_timestamp = timestamp
            else:
                dt = 0.0
        else:
            dt = twin.sim_config.default_dt
            twin.last_timestamp = timestamp

        # Predict expected state using observable context only
        expected = twin.model.step_expected(
            throttle_pct=throttle,
            altitude_m=altitude,
            ambient_temp_c=ambient_temp,
            dt=dt,
            mission_phase=phase,
        )

        # Fast-path single-sample residual generation bypassing DataFrame churn
        if hasattr(twin.residual_generator, "compute_residuals_sample"):
            residual_frame, expected_dict, raw_res_dict, norm_res_dict = (
                twin.residual_generator.compute_residuals_sample(telemetry_dict, expected)
            )
        else:
            obs_df = pd.DataFrame([telemetry_dict])
            exp_df = pd.DataFrame([expected])
            residual_frame = twin.residual_generator.compute_residuals(obs_df, exp_df)

            res_row = residual_frame.to_dataframe().iloc[0]
            expected_dict = {}
            raw_res_dict = {}
            norm_res_dict = {}

            for ch in SUPPORTED_RESIDUAL_CHANNELS:
                if f"{ch}_expected" in res_row:
                    val = res_row[f"{ch}_expected"]
                    expected_dict[ch] = float(val) if pd.notna(val) else float("nan")
                if f"{ch}_residual" in res_row:
                    val = res_row[f"{ch}_residual"]
                    raw_res_dict[ch] = float(val) if pd.notna(val) else float("nan")
                if f"{ch}_norm_residual" in res_row:
                    val = res_row[f"{ch}_norm_residual"]
                    norm_res_dict[ch] = float(val) if pd.notna(val) else float("nan")

        return residual_frame, expected_dict, raw_res_dict, norm_res_dict

    @staticmethod
    def step_anomaly_detection_streaming(
        detector: HybridAnomalyDetector,
        residual_frame: ResidualFrame,
    ) -> Dict[str, Any]:
        """
        Causally update Phase 7 Anomaly Detection with reset_state=False to preserve EWMA & persistence.
        """
        anomaly_frame = detector.detect(residual_frame, reset_state=False)
        records = anomaly_frame.to_records()
        if not records:
            return {
                "anomaly_status": "INSUFFICIENT_DATA",
                "anomaly_score": float("nan"),
                "contributing_channels": [],
                "persistence_count": 0,
                "evidence": {},
            }

        rec = records[0]
        raw_anom_score = rec.get("anomaly_score")
        if raw_anom_score is None or pd.isna(raw_anom_score):
            parsed_score = float("nan")
        else:
            parsed_score = float(raw_anom_score)

        return {
            "anomaly_status": str(rec.get("anomaly_status", "NORMAL")),
            "anomaly_score": parsed_score,
            "contributing_channels": list(rec.get("contributing_channels", []) or []),
            "persistence_count": int(rec.get("persistence_count", 0)),
            "evidence": {
                "threshold_score": rec.get("threshold_score"),
                "ewma_score": rec.get("ewma_score"),
                "persistence_score": rec.get("persistence_score"),
                "isolation_score": rec.get("isolation_score"),
            },
        }

    @staticmethod
    def step_fault_diagnosis(
        pipeline: FaultDiagnosisPipeline,
        residual_frame: ResidualFrame,
        anomaly_status: Optional[str] = None,
        anomaly_score: Optional[float] = None,
        observed_telemetry: Optional[Dict[str, Any]] = None,
        contributing_channels: Optional[List[str]] = None,
    ) -> FaultDiagnosisResult:
        """
        Execute Phase 8 diagnosis on current residual frame with Phase 7 context.
        """
        if hasattr(residual_frame, "get_first_record"):
            row_dict = residual_frame.get_first_record()
        else:
            df = residual_frame.to_dataframe()
            row_dict = df.iloc[0].to_dict()

        return pipeline.diagnose_sample(
            sample=row_dict,
            anomaly_status=anomaly_status,
            anomaly_score=anomaly_score,
            observed_telemetry=observed_telemetry,
            contributing_channels=contributing_channels,
        )

    @staticmethod
    def step_health_index(
        pipeline: HealthIndexPipeline,
        residual_frame: ResidualFrame,
        diagnosis_result: Optional[FaultDiagnosisResult] = None,
        anomaly_status: Optional[str] = None,
        anomaly_score: Optional[float] = None,
    ) -> HealthIndexResult:
        """
        Execute Phase 9 Health Index calculation and degradation tracking.
        """
        if hasattr(residual_frame, "get_first_record"):
            row_dict = residual_frame.get_first_record()
        else:
            df = residual_frame.to_dataframe()
            row_dict = df.iloc[0].to_dict()

        optional_context = {}
        if diagnosis_result is not None:
            optional_context["predicted_fault_type"] = diagnosis_result.predicted_fault_type
            optional_context["diagnostic_confidence"] = diagnosis_result.diagnostic_confidence
            if diagnosis_result.predicted_fault_type == "sensor_fault":
                suspect = diagnosis_result.suspect_sensor
                if suspect and suspect not in ("unknown", "uncertain", "NONE"):
                    optional_context["suspect_channel"] = suspect
                else:
                    optional_context["suspect_channel"] = None
                if diagnosis_result.suspect_sensors:
                    valid_suspects = [
                        s for s in diagnosis_result.suspect_sensors
                        if s and s not in ("unknown", "uncertain", "NONE")
                    ]
                    if valid_suspects:
                        optional_context["suspect_channels"] = valid_suspects

        if anomaly_status is not None:
            optional_context["anomaly_status"] = anomaly_status
        if anomaly_score is not None:
            optional_context["anomaly_score"] = anomaly_score

        return pipeline.process_sample(row_dict, optional_context=optional_context)

    @staticmethod
    def step_forecasting(
        pipeline: ForecastingPipeline,
        telemetry_dict: Dict[str, Any],
        optional_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[ForecastResult]:
        """
        Feed causal observation into Phase 10 forecasting buffer.
        Returns ForecastResult if context length >= 32, else None during context accumulation.
        """
        return pipeline.process_sample(telemetry_dict, optional_context=optional_context)

    @staticmethod
    def step_rul(
        pipeline: RULPipeline,
        health_result: HealthIndexResult,
        forecast_result: Optional[ForecastResult] = None,
        telemetry_dict: Optional[Dict[str, Any]] = None,
    ) -> RULResult:
        """
        Execute Phase 11 authoritative RUL assessment.
        """
        numeric_telemetry = {}
        if telemetry_dict is not None:
            for k, v in telemetry_dict.items():
                if isinstance(v, (int, float)) and not math.isnan(v):
                    numeric_telemetry[k] = float(v)

        return pipeline.process_assessment(
            health_result=health_result,
            forecast=forecast_result,
            current_telemetry=numeric_telemetry,
        )

    @staticmethod
    def step_explainability(
        pipeline: ExplainabilityPipeline,
        timestamp: float,
        engine_id: str,
        mission_id: Optional[str],
        norm_residuals: Dict[str, float],
        features_df: pd.DataFrame,
        diagnosis_result: Optional[FaultDiagnosisResult],
        health_result: Optional[HealthIndexResult],
        forecast_result: Optional[ForecastResult],
        rul_result: Optional[RULResult],
        classifier: Any,
    ) -> ExplainabilityResult:
        """
        Synthesize Phase 12 multi-modal explainability evidence.
        """
        return pipeline.explain(
            timestamp=timestamp,
            engine_id=engine_id,
            mission_id=mission_id,
            residuals=norm_residuals,
            features=features_df,
            diagnosis_result=diagnosis_result,
            health_result=health_result,
            forecast_result=forecast_result,
            rul_result=rul_result,
            classifier=classifier,
        )
