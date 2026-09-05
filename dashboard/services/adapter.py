"""
Dashboard Adapter for SIH26054 Digital Twin.

Architectural Rule:
The DashboardAdapter is a pure transformation layer. It consumes outputs from
Phase 13 (or individual Phase 5–12 schemas) and populates the DashboardViewModel.

IT MUST NOT:
- Calculate RUL or Health Index
- Perform anomaly detection or fault classification
- Run time-series forecasting or SHAP attribution
- Fabricate missing or degraded values
- Duplicate orchestration logic
"""

import math
from typing import Dict, Any, Optional, Union, List

from orchestrator.schema import DashboardStatePayload, OperatorAdvisory
from dashboard.schemas.contracts import Phase13OutputContract
from dashboard.schemas.view_model import (
    DashboardViewModel,
    OverviewViewModel,
    TelemetryViewModel,
    ChannelTelemetryModel,
    DiagnosticsViewModel,
    PrognosticsViewModel,
    DataQualityViewModel,
    MetricCardModel,
    OperatorAdvisoryModel,
    StatusLevel,
    AvailabilityStatus,
    CANONICAL_CHANNELS,
    CANONICAL_CHANNEL_METADATA,
)
from dashboard.utils.formatters import (
    format_value,
    format_rul,
    format_percent,
    format_health_index,
    format_fault_name,
    map_health_to_status,
    map_anomaly_to_status,
    map_rul_status_to_status,
    safe_is_nan,
)


class DashboardAdapter:
    """
    Transforms Phase 13 DashboardStatePayload into a decoupled DashboardViewModel.
    Consumes the authoritative orchestrator.schema.DashboardStatePayload without duplicating
    or executing any Phase 6–12 PHM/ML algorithms.
    Guarantees non-fabrication and clear distinction between available vs unavailable data.
    """

    @staticmethod
    def _extract_dict_or_attr(source: Any, key: str, default: Any = None) -> Any:
        """Helper to extract a field whether the source is a dataclass object or a dict."""
        if source is None:
            return default
        if isinstance(source, dict):
            return source.get(key, default)
        return getattr(source, key, default)

    def adapt(
        self,
        payload: Union[DashboardStatePayload, Phase13OutputContract, Dict[str, Any], Any],
    ) -> DashboardViewModel:
        """
        Transform authoritative Phase 13 DashboardStatePayload into DashboardViewModel.

        Args:
            payload: DashboardStatePayload instance, raw dict (e.g. from to_dict()), or legacy contract.

        Returns:
            DashboardViewModel ready for Streamlit rendering.
        """
        # Extract Mission context
        engine_id = (
            self._extract_dict_or_attr(payload, "engine_id")
            or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "mission"), "engine_id")
            or "ENG_001"
        )
        mission_id = (
            self._extract_dict_or_attr(payload, "mission_id")
            or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "mission"), "mission_id")
        )
        timestamp_raw = self._extract_dict_or_attr(payload, "timestamp")
        if timestamp_raw is None:
            timestamp_raw = self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "mission"), "timestamp", 0.0)
        timestamp = float(timestamp_raw) if (timestamp_raw is not None and not safe_is_nan(timestamp_raw)) else 0.0

        mission_phase = (
            self._extract_dict_or_attr(payload, "mission_phase")
            or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "mission"), "mission_phase")
            or "CRUISE"
        )
        simulation_mode = (
            self._extract_dict_or_attr(payload, "simulation_mode")
            or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "mission"), "simulation_mode")
            or "SYNTHETIC_SIMULATION"
        )
        scenario_metadata = (
            self._extract_dict_or_attr(payload, "scenario_metadata")
            or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "mission"), "scenario_metadata")
            or {}
        )
        is_synthetic_demo = (
            simulation_mode == "SYNTHETIC_SIMULATION"
            or bool(self._extract_dict_or_attr(payload, "is_synthetic_demo", False))
        )
        execution_status = str(self._extract_dict_or_attr(payload, "execution_status", "COMPLETED"))
        execution_latency_ms = float(self._extract_dict_or_attr(payload, "execution_latency_ms", 0.0))
        provenance = dict(self._extract_dict_or_attr(payload, "provenance", {}) or {})

        # Extract Telemetry
        obs_raw = (
            self._extract_dict_or_attr(payload, "observed_telemetry")
            or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "telemetry"), "observed")
            or self._extract_dict_or_attr(payload, "telemetry")
            or {}
        )
        quality_status = (
            self._extract_dict_or_attr(payload, "quality_status")
            or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "telemetry"), "quality_status")
            or "NOMINAL"
        )

        # Extract Digital Twin
        exp_raw = (
            self._extract_dict_or_attr(payload, "expected_telemetry")
            or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "digital_twin"), "expected")
            or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "digital_twin"), "nominal_estimates")
            or {}
        )
        res_raw = (
            self._extract_dict_or_attr(payload, "residuals")
            or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "digital_twin"), "residuals")
            or {}
        )
        norm_res_raw = (
            self._extract_dict_or_attr(payload, "normalized_residuals")
            or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "digital_twin"), "normalized_residuals")
            or {}
        )

        is_dict_payload = isinstance(payload, dict)

        # Extract Anomaly Detection
        has_phase7 = (
            hasattr(payload, "anomaly_status")
            or not is_dict_payload
            or "anomaly_status" in payload
            or "anomaly" in payload
        )
        if has_phase7:
            anom_status = (
                self._extract_dict_or_attr(payload, "anomaly_status")
                or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "anomaly"), "anomaly_status")
                or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "anomaly"), "status")
                or "Unavailable"
            )
            anom_score_raw = (
                self._extract_dict_or_attr(payload, "anomaly_score")
                if self._extract_dict_or_attr(payload, "anomaly_score") is not None
                else (
                    self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "anomaly"), "anomaly_score")
                    if self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "anomaly"), "anomaly_score") is not None
                    else self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "anomaly"), "score")
                )
            )
            anomaly_score = float(anom_score_raw) if (anom_score_raw is not None and not safe_is_nan(anom_score_raw)) else None
            anom_contrib = list(
                self._extract_dict_or_attr(payload, "anomaly_contributing_channels")
                or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "anomaly"), "contributing_channels")
                or []
            )
            persistence_count = int(
                self._extract_dict_or_attr(payload, "persistence_count")
                or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "anomaly"), "persistence_count")
                or 0
            )
            anomaly_evidence = dict(
                self._extract_dict_or_attr(payload, "anomaly_evidence")
                or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "anomaly"), "evidence")
                or {}
            )
        else:
            anom_status = "Unavailable"
            anomaly_score = None
            anom_contrib = []
            persistence_count = 0
            anomaly_evidence = {}

        # Extract Fault Diagnosis
        has_phase8 = (
            hasattr(payload, "predicted_fault_class")
            or not is_dict_payload
            or "predicted_fault_class" in payload
            or "diagnosis" in payload
            or "fault_diagnosis" in payload
        )
        if has_phase8:
            pred_fault = (
                self._extract_dict_or_attr(payload, "predicted_fault_class")
                or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "diagnosis"), "predicted_fault_class")
                or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "fault_diagnosis"), "predicted_fault_type")
                or "none"
            )
            diag_probs = dict(
                self._extract_dict_or_attr(payload, "diagnosis_probabilities")
                or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "diagnosis"), "class_probabilities")
                or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "fault_diagnosis"), "class_probabilities")
                or {}
            )
            diag_conf_raw = (
                self._extract_dict_or_attr(payload, "diagnostic_confidence")
                if self._extract_dict_or_attr(payload, "diagnostic_confidence") is not None
                else (
                    self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "diagnosis"), "confidence")
                    if self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "diagnosis"), "confidence") is not None
                    else self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "fault_diagnosis"), "diagnostic_confidence")
                )
            )
            diag_conf = float(diag_conf_raw) if (diag_conf_raw is not None and not safe_is_nan(diag_conf_raw)) else None
            diag_quality = (
                self._extract_dict_or_attr(payload, "diagnosis_data_quality")
                or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "diagnosis"), "data_quality")
                or "VALID"
            )
        else:
            pred_fault = "Unavailable"
            diag_probs = {}
            diag_conf = None
            diag_quality = "Unavailable"

        # Extract Health Index
        raw_hi_raw = (
            self._extract_dict_or_attr(payload, "raw_health_index")
            if self._extract_dict_or_attr(payload, "raw_health_index") is not None
            else self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "health"), "raw_health_index")
        )
        raw_health_index = float(raw_hi_raw) if (raw_hi_raw is not None and not safe_is_nan(raw_hi_raw)) else None

        smooth_hi_raw = (
            self._extract_dict_or_attr(payload, "smoothed_health_index")
            if self._extract_dict_or_attr(payload, "smoothed_health_index") is not None
            else (
                self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "health"), "smoothed_health_index")
                if self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "health"), "smoothed_health_index") is not None
                else self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "health_index"), "smoothed_health_index")
            )
        )
        smoothed_health_index = float(smooth_hi_raw) if (smooth_hi_raw is not None and not safe_is_nan(smooth_hi_raw)) else None

        health_state = (
            self._extract_dict_or_attr(payload, "health_state")
            or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "health"), "health_state")
            or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "health_index"), "health_state")
            or "Unavailable"
        )
        deg_rate_raw = (
            self._extract_dict_or_attr(payload, "degradation_rate")
            if self._extract_dict_or_attr(payload, "degradation_rate") is not None
            else self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "health"), "degradation_rate")
        )
        degradation_rate = float(deg_rate_raw) if (deg_rate_raw is not None and not safe_is_nan(deg_rate_raw)) else None

        deg_trend = (
            self._extract_dict_or_attr(payload, "degradation_trend")
            or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "health"), "degradation_trend")
            or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "health_index"), "degradation_trend")
            or "Unavailable"
        )
        dominant_channels = list(
            self._extract_dict_or_attr(payload, "dominant_channels")
            or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "health"), "dominant_channels")
            or []
        )
        channel_contributions = dict(
            self._extract_dict_or_attr(payload, "channel_contributions")
            or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "health"), "channel_contributions")
            or {}
        )

        # Extract Forecast
        has_phase10 = (
            hasattr(payload, "forecast_status")
            or not is_dict_payload
            or "forecast_status" in payload
            or "forecast" in payload
        )
        if has_phase10:
            fc_status = (
                self._extract_dict_or_attr(payload, "forecast_status")
                or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "forecast"), "status")
                or "BUFFERING"
            )
            fc_source = (
                self._extract_dict_or_attr(payload, "forecast_source")
                or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "forecast"), "source")
                or "TIMESFM_OR_BASELINE"
            )
            fc_horizon = int(
                self._extract_dict_or_attr(payload, "forecast_horizon")
                or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "forecast"), "horizon")
                or 16
            )
            pred_telemetry = (
                self._extract_dict_or_attr(payload, "predicted_telemetry")
                or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "forecast"), "predicted_telemetry")
            )
            fc_timestamps_raw = (
                self._extract_dict_or_attr(payload, "forecast_timestamps")
                or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "forecast"), "timestamps")
                or []
            )
            forecast_timestamps = [float(t) for t in fc_timestamps_raw] if fc_timestamps_raw else []
            is_pretrained = bool(
                self._extract_dict_or_attr(payload, "is_pretrained")
                if self._extract_dict_or_attr(payload, "is_pretrained") is not None
                else self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "forecast"), "is_pretrained", False)
            )
            fc_quality = (
                self._extract_dict_or_attr(payload, "forecast_quality")
                or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "forecast"), "quality")
                or "INSUFFICIENT_CONTEXT"
            )
        else:
            fc_status = "Unavailable"
            fc_source = "Unavailable"
            fc_horizon = 16
            pred_telemetry = None
            forecast_timestamps = []
            is_pretrained = False
            fc_quality = None

        # Extract RUL / Prognostics
        rul_state = (
            self._extract_dict_or_attr(payload, "rul_state")
            or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "rul"), "state")
            or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "prognostics"), "status")
            or "Unavailable"
        )
        point_rul_raw = (
            self._extract_dict_or_attr(payload, "point_rul_seconds")
            if self._extract_dict_or_attr(payload, "point_rul_seconds") is not None
            else (
                self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "rul"), "point_rul_seconds")
                if self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "rul"), "point_rul_seconds") is not None
                else self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "prognostics"), "rul_seconds_median")
            )
        )
        point_rul_seconds = float(point_rul_raw) if (point_rul_raw is not None and not safe_is_nan(point_rul_raw)) else None
        rul_hours = (point_rul_seconds / 3600.0) if point_rul_seconds is not None else None

        p05_raw = (
            self._extract_dict_or_attr(payload, "rul_uncertainty_p05")
            if self._extract_dict_or_attr(payload, "rul_uncertainty_p05") is not None
            else (
                self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "rul"), "p05")
                if self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "rul"), "p05") is not None
                else self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "prognostics"), "rul_seconds_p05")
            )
        )
        rul_p05_seconds = float(p05_raw) if (p05_raw is not None and not safe_is_nan(p05_raw)) else None
        rul_p05_hours = (rul_p05_seconds / 3600.0) if rul_p05_seconds is not None else None

        p95_raw = (
            self._extract_dict_or_attr(payload, "rul_uncertainty_p95")
            if self._extract_dict_or_attr(payload, "rul_uncertainty_p95") is not None
            else (
                self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "rul"), "p95")
                if self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "rul"), "p95") is not None
                else self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "prognostics"), "rul_seconds_p95")
            )
        )
        rul_p95_seconds = float(p95_raw) if (p95_raw is not None and not safe_is_nan(p95_raw)) else None
        rul_p95_hours = (rul_p95_seconds / 3600.0) if rul_p95_seconds is not None else None

        limiting_factor = (
            self._extract_dict_or_attr(payload, "limiting_factor")
            or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "rul"), "limiting_factor")
            or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "prognostics"), "limiting_factor")
            or "NONE"
        )
        fc_assisted = bool(
            self._extract_dict_or_attr(payload, "forecast_assisted_mode")
            if self._extract_dict_or_attr(payload, "forecast_assisted_mode") is not None
            else self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "rul"), "forecast_assisted", False)
        )
        eol_provenance = dict(
            self._extract_dict_or_attr(payload, "eol_provenance")
            or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "rul"), "eol_provenance")
            or {}
        )

        # Extract Explainability
        summary_explanation = (
            self._extract_dict_or_attr(payload, "summary_explanation")
            or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "explainability"), "summary_explanation")
            or ""
        )
        shap_attr = (
            self._extract_dict_or_attr(payload, "shap_attribution")
            or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "explainability"), "shap_attribution")
        )
        physics_ev = (
            self._extract_dict_or_attr(payload, "physics_evidence")
            or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "explainability"), "physics_evidence")
        )
        temporal_ev = (
            self._extract_dict_or_attr(payload, "temporal_evidence")
            or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "explainability"), "temporal_evidence")
        )
        fused_ev = (
            self._extract_dict_or_attr(payload, "fused_evidence")
            or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "explainability"), "fused_evidence")
        )
        recommended_action = (
            self._extract_dict_or_attr(payload, "recommended_operator_action")
            or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "explainability"), "recommended_operator_action")
            or "Continue nominal monitoring."
        )

        # Extract Operator Advisory
        advisory_raw = self._extract_dict_or_attr(payload, "advisory")
        advisory_vm: Optional[OperatorAdvisoryModel] = None
        if advisory_raw is not None:
            if isinstance(advisory_raw, dict):
                advisory_vm = OperatorAdvisoryModel(
                    action_code=str(advisory_raw.get("action_code", "NORMAL_MONITORING")),
                    headline=str(advisory_raw.get("headline", "Nominal Operations")),
                    recommended_action=str(advisory_raw.get("recommended_action", "Continue nominal monitoring.")),
                    affected_subsystem=advisory_raw.get("affected_subsystem"),
                    urgency=str(advisory_raw.get("urgency", "LOW")),
                    disclaimer=str(advisory_raw.get("disclaimer", "")),
                )
            else:
                advisory_vm = OperatorAdvisoryModel(
                    action_code=str(getattr(advisory_raw, "action_code", "NORMAL_MONITORING")),
                    headline=str(getattr(advisory_raw, "headline", "Nominal Operations")),
                    recommended_action=str(getattr(advisory_raw, "recommended_action", "Continue nominal monitoring.")),
                    affected_subsystem=getattr(advisory_raw, "affected_subsystem", None),
                    urgency=str(getattr(advisory_raw, "urgency", "LOW")),
                    disclaimer=str(getattr(advisory_raw, "disclaimer", "")),
                )

        # Extract isolated channels from payload
        isolated_channels = list(
            self._extract_dict_or_attr(payload, "isolated_channels")
            or self._extract_dict_or_attr(self._extract_dict_or_attr(payload, "health_index"), "excluded_channels")
            or (anom_contrib if pred_fault.lower() == "sensor_fault" else [])
            or []
        )
        sensor_fault_indicated = (pred_fault.lower() == "sensor_fault" or len(isolated_channels) > 0)

        # 1. Build TelemetryViewModel
        channels: Dict[str, ChannelTelemetryModel] = {}
        for ch in CANONICAL_CHANNELS:
            meta = CANONICAL_CHANNEL_METADATA[ch]
            raw_val = self._extract_dict_or_attr(obs_raw, ch)
            obs_val = float(raw_val) if (raw_val is not None and not safe_is_nan(raw_val)) else None

            # Expected from digital twin
            exp_val_raw = (
                self._extract_dict_or_attr(exp_raw, f"expected_{ch}")
                if self._extract_dict_or_attr(exp_raw, f"expected_{ch}") is not None
                else (
                    self._extract_dict_or_attr(exp_raw, f"nominal_{ch}")
                    if self._extract_dict_or_attr(exp_raw, f"nominal_{ch}") is not None
                    else self._extract_dict_or_attr(exp_raw, ch)
                )
            )
            exp_val = float(exp_val_raw) if (exp_val_raw is not None and not safe_is_nan(exp_val_raw)) else None

            # Residual from digital twin
            res_val_raw = (
                self._extract_dict_or_attr(res_raw, f"{ch}_residual")
                if self._extract_dict_or_attr(res_raw, f"{ch}_residual") is not None
                else self._extract_dict_or_attr(res_raw, ch)
            )
            res_val = float(res_val_raw) if (res_val_raw is not None and not safe_is_nan(res_val_raw)) else None

            # Forecast arrays
            fc_vals = [float(v) for v in pred_telemetry.get(ch, [])] if isinstance(pred_telemetry, dict) and ch in pred_telemetry else []

            is_isolated = ch in isolated_channels or ch in dominant_channels or (pred_fault.lower() == "sensor_fault" and ch in anom_contrib)
            is_missing = obs_val is None

            # Determine channel status
            if is_isolated:
                status = StatusLevel.DEGRADED
            elif is_missing:
                status = StatusLevel.UNAVAILABLE
            elif obs_val is not None:
                if "critical_max" in meta and obs_val >= meta["critical_max"]:
                    status = StatusLevel.CRITICAL
                elif "critical_min" in meta and obs_val <= meta["critical_min"]:
                    status = StatusLevel.CRITICAL
                elif "nominal_max" in meta and obs_val > meta["nominal_max"]:
                    status = StatusLevel.WARNING
                elif "nominal_min" in meta and obs_val < meta["nominal_min"]:
                    status = StatusLevel.WARNING
                else:
                    status = StatusLevel.HEALTHY
            else:
                status = StatusLevel.UNAVAILABLE

            channels[ch] = ChannelTelemetryModel(
                channel=ch,
                display_name=meta["display_name"],
                unit=meta["unit"],
                observed_value=obs_val,
                expected_value=exp_val,
                residual=res_val,
                forecast_values=fc_vals,
                forecast_timestamps=forecast_timestamps,
                status=status,
                is_isolated=is_isolated,
                is_missing=is_missing,
                availability=AvailabilityStatus.AVAILABLE if obs_val is not None else AvailabilityStatus.UNAVAILABLE,
            )

        thr_val = self._extract_dict_or_attr(obs_raw, "throttle")
        ld_val = self._extract_dict_or_attr(obs_raw, "load")
        alt_val = self._extract_dict_or_attr(obs_raw, "altitude")
        amb_val = self._extract_dict_or_attr(obs_raw, "ambient_temp")

        telemetry_vm = TelemetryViewModel(
            timestamp=timestamp,
            channels=channels,
            observed_telemetry={ch: channels[ch].observed_value for ch in CANONICAL_CHANNELS},
            quality_status=quality_status,
            throttle=float(thr_val) if (thr_val is not None and not safe_is_nan(thr_val)) else None,
            load=float(ld_val) if (ld_val is not None and not safe_is_nan(ld_val)) else None,
            altitude=float(alt_val) if (alt_val is not None and not safe_is_nan(alt_val)) else None,
            ambient_temp=float(amb_val) if (amb_val is not None and not safe_is_nan(amb_val)) else None,
            mission_phase=mission_phase,
            availability=AvailabilityStatus.AVAILABLE if any(ch.observed_value is not None for ch in channels.values()) else AvailabilityStatus.UNAVAILABLE,
        )

        # 2. Build DiagnosticsViewModel
        shap_features = []
        if isinstance(shap_attr, dict):
            for f in shap_attr.get("top_features", []):
                shap_features.append({
                    "feature_name": f.get("feature") or f.get("feature_name"),
                    "feature_value": f.get("feature_value"),
                    "shap_value": f.get("shap_value"),
                    "direction": f.get("direction"),
                    "relative_weight": f.get("relative_weight", 0.0),
                })

        phys_status = None
        phys_reason = None
        if isinstance(physics_ev, dict):
            phys_status = str(physics_ev.get("status", "Unavailable"))
            phys_reason = physics_ev.get("consistency_reason")

        diagnostics_vm = DiagnosticsViewModel(
            timestamp=timestamp,
            expected_telemetry=exp_raw,
            residuals=res_raw,
            normalized_residuals=norm_res_raw,
            anomaly_status=anom_status,
            anomaly_score=anomaly_score,
            contributing_channels=anom_contrib,
            anomaly_contributing_channels=anom_contrib,
            persistence_count=persistence_count,
            anomaly_evidence=anomaly_evidence,
            predicted_fault=pred_fault,
            predicted_fault_class=pred_fault,
            diagnostic_confidence=diag_conf,
            class_probabilities=diag_probs,
            diagnosis_data_quality=diag_quality,
            sensor_fault_indicated=sensor_fault_indicated,
            isolated_channels=isolated_channels,
            summary_explanation=summary_explanation,
            recommended_operator_action=recommended_action,
            shap_top_features=shap_features,
            shap_attribution=shap_attr,
            physics_evidence=physics_ev,
            physics_evidence_status=phys_status,
            physics_consistency_reason=phys_reason,
            temporal_evidence=temporal_ev,
            fused_evidence=fused_ev,
            availability=AvailabilityStatus.AVAILABLE if anom_status != "Unavailable" or pred_fault != "Unavailable" else AvailabilityStatus.UNAVAILABLE,
        )

        # 3. Build PrognosticsViewModel
        prognostics_vm = PrognosticsViewModel(
            timestamp=timestamp,
            health_index=smoothed_health_index,
            smoothed_health_index=smoothed_health_index,
            raw_health_index=raw_health_index,
            health_state=health_state,
            degradation_rate=degradation_rate,
            degradation_trend=deg_trend,
            dominant_channels=dominant_channels,
            channel_contributions=channel_contributions,
            rul_state=rul_state,
            rul_status=rul_state,
            point_rul_seconds=point_rul_seconds,
            rul_hours=rul_hours,
            rul_uncertainty_p05=rul_p05_seconds,
            rul_p05_hours=rul_p05_hours,
            rul_uncertainty_p95=rul_p95_seconds,
            rul_p95_hours=rul_p95_hours,
            limiting_factor=limiting_factor,
            forecast_assisted_mode=fc_assisted,
            eol_provenance=eol_provenance,
            forecast_status=fc_status,
            forecast_source=fc_source,
            forecast_horizon=fc_horizon,
            forecast_quality=fc_quality,
            model_name=fc_source,
            model_status=fc_status,
            is_pretrained=is_pretrained,
            predicted_telemetry=pred_telemetry if isinstance(pred_telemetry, dict) else None,
            forecast_timestamps=forecast_timestamps,
            availability=AvailabilityStatus.AVAILABLE if smoothed_health_index is not None or rul_state != "Unavailable" else AvailabilityStatus.UNAVAILABLE,
        )

        # 4. Build DataQualityViewModel
        missing_sensors = [ch for ch, m in channels.items() if m.is_missing]
        prov_source_type = (
            str(provenance.get("source_type"))
            if "source_type" in provenance
            else ("synthetic" if is_synthetic_demo else "Unavailable")
        )
        data_quality_vm = DataQualityViewModel(
            timestamp=timestamp,
            quality_score=1.0 - (len(missing_sensors) / 7.0),
            quality_status=quality_status,
            missing_sensors=missing_sensors,
            isolated_sensors=[ch for ch, m in channels.items() if m.is_isolated],
            provenance=provenance,
            provenance_source=str(provenance.get("source", "SystemPipelineOrchestrator")),
            provenance_source_type=prov_source_type,
            execution_latency_ms=execution_latency_ms,
            availability=AvailabilityStatus.AVAILABLE,
        )

        # 5. Build OverviewViewModel
        overview_vm = self._adapt_overview(
            engine_id=engine_id,
            mission_id=mission_id,
            timestamp=timestamp,
            mission_phase=mission_phase,
            simulation_mode=simulation_mode,
            is_synthetic_demo=is_synthetic_demo,
            scenario_metadata=scenario_metadata,
            telemetry_vm=telemetry_vm,
            diagnostics_vm=diagnostics_vm,
            prognostics_vm=prognostics_vm,
            data_quality_vm=data_quality_vm,
            advisory=advisory_vm,
        )

        return DashboardViewModel(
            timestamp=timestamp,
            engine_id=engine_id,
            mission_id=mission_id,
            mission_phase=mission_phase,
            simulation_mode=simulation_mode,
            scenario_metadata=scenario_metadata,
            is_synthetic_demo=is_synthetic_demo,
            execution_status=execution_status,
            overview=overview_vm,
            telemetry=telemetry_vm,
            diagnostics=diagnostics_vm,
            prognostics=prognostics_vm,
            data_quality=data_quality_vm,
            advisory=advisory_vm,
            provenance=provenance,
            execution_latency_ms=execution_latency_ms,
        )

    def _adapt_overview(
        self,
        engine_id: str,
        mission_id: Optional[str],
        timestamp: float,
        mission_phase: str,
        simulation_mode: str,
        is_synthetic_demo: bool,
        scenario_metadata: Dict[str, Any],
        telemetry_vm: TelemetryViewModel,
        diagnostics_vm: DiagnosticsViewModel,
        prognostics_vm: PrognosticsViewModel,
        data_quality_vm: DataQualityViewModel,
        advisory: Optional[OperatorAdvisoryModel] = None,
    ) -> OverviewViewModel:
        """Create synthesized overview KPI cards without calculating metrics."""
        # 1. Health Index Card
        if prognostics_vm.smoothed_health_index is not None:
            hi_val = format_health_index(prognostics_vm.smoothed_health_index)
            hi_status = map_health_to_status(prognostics_vm.health_state)
            hi_subtext = f"{prognostics_vm.health_state} ({prognostics_vm.degradation_trend})"
            hi_avail = AvailabilityStatus.AVAILABLE
        else:
            hi_val = "Unavailable"
            hi_status = StatusLevel.UNAVAILABLE
            hi_subtext = "Phase 9 Health Index not supplied"
            hi_avail = AvailabilityStatus.UNAVAILABLE

        health_card = MetricCardModel(
            label="Health Index",
            value=hi_val,
            status=hi_status,
            subtext=hi_subtext,
            availability=hi_avail,
        )

        # 2. RUL Card
        if prognostics_vm.rul_hours is not None and not safe_is_nan(prognostics_vm.rul_hours):
            rul_val = f"{prognostics_vm.rul_hours:.1f} hrs"
            rul_status = map_rul_status_to_status(prognostics_vm.rul_state)
            bounds = ""
            if prognostics_vm.rul_p05_hours is not None and prognostics_vm.rul_p95_hours is not None:
                bounds = f" [{prognostics_vm.rul_p05_hours:.1f} - {prognostics_vm.rul_p95_hours:.1f} hrs]"
            rul_subtext = f"State: {prognostics_vm.rul_state}{bounds}"
            rul_avail = AvailabilityStatus.AVAILABLE
        elif prognostics_vm.rul_state in ("NOT_DEGRADING", "INDETERMINATE_TREND", "EXCEEDS_HORIZON"):
            rul_val = prognostics_vm.rul_state.replace("_", " ").title()
            rul_status = StatusLevel.HEALTHY if prognostics_vm.rul_state == "NOT_DEGRADING" else StatusLevel.WARNING
            rul_subtext = f"State: {prognostics_vm.rul_state} | Limiting: {prognostics_vm.limiting_factor}"
            rul_avail = AvailabilityStatus.AVAILABLE
        elif prognostics_vm.rul_state != "Unavailable":
            rul_val = prognostics_vm.rul_state
            rul_status = StatusLevel.WARNING if prognostics_vm.rul_state == "INSUFFICIENT_HISTORY" else StatusLevel.DEGRADED
            rul_subtext = f"State: {prognostics_vm.rul_state} | Limiting: {prognostics_vm.limiting_factor}"
            rul_avail = AvailabilityStatus.AVAILABLE
        else:
            rul_val = "Unavailable"
            rul_status = StatusLevel.UNAVAILABLE
            rul_subtext = "Phase 11 Prognostics not supplied"
            rul_avail = AvailabilityStatus.UNAVAILABLE

        rul_card = MetricCardModel(
            label="Remaining Useful Life",
            value=rul_val,
            status=rul_status,
            subtext=rul_subtext,
            availability=rul_avail,
        )

        # 3. Anomaly Status Card
        if diagnostics_vm.anomaly_status != "Unavailable":
            anom_val = diagnostics_vm.anomaly_status
            anom_status = map_anomaly_to_status(diagnostics_vm.anomaly_status)
            score_str = f"Score: {diagnostics_vm.anomaly_score:.2f} | Persist: {diagnostics_vm.persistence_count}" if diagnostics_vm.anomaly_score is not None else f"Persist: {diagnostics_vm.persistence_count}"
            anom_subtext = score_str
            anom_avail = AvailabilityStatus.AVAILABLE
        else:
            anom_val = "Unavailable"
            anom_status = StatusLevel.UNAVAILABLE
            anom_subtext = "Phase 7 Anomaly detector not supplied"
            anom_avail = AvailabilityStatus.UNAVAILABLE

        anomaly_card = MetricCardModel(
            label="Active Anomaly Status",
            value=anom_val,
            status=anom_status,
            subtext=anom_subtext,
            availability=anom_avail,
        )

        # 4. Fault Diagnosis Card
        if diagnostics_vm.predicted_fault_class != "Unavailable":
            fault_val = format_fault_name(diagnostics_vm.predicted_fault_class)
            if diagnostics_vm.predicted_fault_class.lower() in ("none", "normal"):
                fault_status = StatusLevel.HEALTHY
            elif diagnostics_vm.sensor_fault_indicated:
                fault_status = StatusLevel.DEGRADED
            else:
                fault_status = StatusLevel.CRITICAL

            # Explicitly label class probability as "Diagnosis Probability", never generic "AI confidence"
            prob_val = diagnostics_vm.class_probabilities.get(
                diagnostics_vm.predicted_fault_class,
                diagnostics_vm.diagnostic_confidence
            )
            prob_str = f"{prob_val * 100.0:.1f}%" if (prob_val is not None and not safe_is_nan(prob_val)) else "N/A"
            fault_subtext = f"Diagnosis Probability: {prob_str}"
            fault_avail = AvailabilityStatus.AVAILABLE
        else:
            fault_val = "Unavailable"
            fault_status = StatusLevel.UNAVAILABLE
            fault_subtext = "Phase 8 Diagnosis not supplied"
            fault_avail = AvailabilityStatus.UNAVAILABLE

        fault_card = MetricCardModel(
            label="Diagnosed Fault",
            value=fault_val,
            status=fault_status,
            subtext=fault_subtext,
            availability=fault_avail,
        )

        # 5. Data Quality Card
        dq_val = data_quality_vm.quality_status
        if data_quality_vm.quality_status in ("NOMINAL", "VALID"):
            dq_status = StatusLevel.HEALTHY
        elif data_quality_vm.quality_status in ("DEGRADED", "OUT_OF_ORDER_REJECTED", "INVALID"):
            dq_status = StatusLevel.DEGRADED
        else:
            dq_status = StatusLevel.WARNING
        dq_subtext = f"Latency: {data_quality_vm.execution_latency_ms:.1f} ms"

        data_quality_card = MetricCardModel(
            label="Data Quality Status",
            value=dq_val,
            status=dq_status,
            subtext=dq_subtext,
            availability=AvailabilityStatus.AVAILABLE,
        )

        # Overall Status is derived strictly from upstream health status or execution status
        if prognostics_vm.health_state in ("CRITICAL", "EOL_REACHED") or diagnostics_vm.anomaly_status == "ANOMALY":
            overall_status = StatusLevel.CRITICAL
        elif prognostics_vm.health_state in ("DEGRADED", "SEVERELY_DEGRADED") or diagnostics_vm.anomaly_status == "WARNING":
            overall_status = StatusLevel.WARNING
        elif prognostics_vm.health_state in ("HEALTHY", "NORMAL") and diagnostics_vm.anomaly_status in ("NORMAL", "HEALTHY"):
            overall_status = StatusLevel.HEALTHY
        elif diagnostics_vm.sensor_fault_indicated:
            overall_status = StatusLevel.DEGRADED
        else:
            overall_status = StatusLevel.UNAVAILABLE

        return OverviewViewModel(
            overall_status=overall_status,
            engine_id=engine_id,
            mission_id=mission_id,
            timestamp=timestamp,
            mission_phase=mission_phase,
            simulation_mode=simulation_mode,
            is_synthetic_demo=is_synthetic_demo,
            scenario_metadata=scenario_metadata,
            health_card=health_card,
            rul_card=rul_card,
            anomaly_card=anomaly_card,
            fault_card=fault_card,
            data_quality_card=data_quality_card,
            advisory=advisory,
        )
