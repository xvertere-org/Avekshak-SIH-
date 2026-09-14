"""
Engineering Mission Reporting Module for Phase 14 — SIH26054 / AVEKSHAK Digital Twin.

Architectural Rule:
- Consumes completed mission payloads from Phase 13 without recomputing PHM logic.
- Synthesizes authoritative engineering mission summaries.
- Adheres to strict decision-support terminology ("advisory GO / CAUTION / MAINTENANCE assessment",
  "project-defined simulated functional-failure/EOL assumption", "simulated projection").
- Emits clean Markdown and JSON reports.
"""

from typing import Dict, List, Optional, Any
import numpy as np

from orchestrator.schema import DashboardStatePayload, AdvisoryActionCode
from phase14.schema import MissionReportSummary


class MissionReportGenerator:
    """
    Synthesizes authoritative, non-fabricated mission reports from Phase 13 payloads.
    Provides multi-channel statistical aggregates, PHM events, explainability fusion,
    and structured export capabilities.
    """

    @classmethod
    def generate_report(
        cls,
        payloads: List[DashboardStatePayload],
        scenario_metadata: Optional[Dict[str, Any]] = None,
    ) -> MissionReportSummary:
        """
        Generate a comprehensive MissionReportSummary from a chronological sequence of mission payloads.
        """
        if not payloads:
            raise ValueError("Cannot generate mission report from an empty payload sequence.")

        # Filter out rejected observations (out-of-order, duplicate, invalid timestamp)
        valid_payloads = [
            p for p in payloads
            if "REJECTED" not in (getattr(p, "quality_status", None) or "")
        ]

        if valid_payloads:
            first_p = valid_payloads[0]
            effective_p = valid_payloads[-1]
            engine_id = first_p.engine_id
            mission_id = first_p.mission_id or "UNKNOWN_MISSION"
            duration_s = max(0.0, effective_p.timestamp - first_p.timestamp)
            mission_phase = effective_p.mission_phase
            simulation_mode = effective_p.simulation_mode
            meta = scenario_metadata or effective_p.scenario_metadata or {}
            scenario_name = meta.get("scenario_name", "standard_mission")
            dt = (valid_payloads[1].timestamp - valid_payloads[0].timestamp) if len(valid_payloads) > 1 else 1.0
        else:
            first_p = payloads[0]
            effective_p = payloads[-1]
            engine_id = first_p.engine_id
            mission_id = first_p.mission_id or "UNKNOWN_MISSION"
            duration_s = 0.0
            mission_phase = effective_p.mission_phase
            simulation_mode = effective_p.simulation_mode
            meta = scenario_metadata or effective_p.scenario_metadata or {}
            scenario_name = meta.get("scenario_name", "rejected_mission")
            dt = 1.0

        # 1. Telemetry Aggregation (only over valid payloads if available)
        aggregation_stream = valid_payloads if valid_payloads else payloads
        rpms: List[float] = []
        chts: List[float] = []
        egts: List[float] = []
        oil_temps: List[float] = []
        oil_pressures: List[float] = []
        vibrations: List[float] = []
        fuel_flows: List[float] = []

        for p in aggregation_stream:
            obs = p.observed_telemetry
            if "rpm" in obs and not np.isnan(obs["rpm"]):
                rpms.append(obs["rpm"])
            if "cht" in obs and not np.isnan(obs["cht"]):
                chts.append(obs["cht"])
            if "egt" in obs and not np.isnan(obs["egt"]):
                egts.append(obs["egt"])
            if "oil_temp" in obs and not np.isnan(obs["oil_temp"]):
                oil_temps.append(obs["oil_temp"])
            if "oil_pressure" in obs and not np.isnan(obs["oil_pressure"]):
                oil_pressures.append(obs["oil_pressure"])
            if "vibration" in obs and not np.isnan(obs["vibration"]):
                vibrations.append(obs["vibration"])
            if "fuel_flow" in obs and not np.isnan(obs["fuel_flow"]):
                fuel_flows.append(obs["fuel_flow"])

        rpm_min = float(np.min(rpms)) if rpms else float("nan")
        rpm_max = float(np.max(rpms)) if rpms else float("nan")
        rpm_mean = float(np.mean(rpms)) if rpms else float("nan")

        cht_peak = float(np.max(chts)) if chts else float("nan")
        cht_mean = float(np.mean(chts)) if chts else float("nan")

        egt_peak = float(np.max(egts)) if egts else float("nan")
        egt_mean = float(np.mean(egts)) if egts else float("nan")

        oil_temp_peak = float(np.max(oil_temps)) if oil_temps else float("nan")
        oil_temp_mean = float(np.mean(oil_temps)) if oil_temps else float("nan")

        oil_pressure_min = float(np.min(oil_pressures)) if oil_pressures else float("nan")
        oil_pressure_mean = float(np.mean(oil_pressures)) if oil_pressures else float("nan")

        vibration_max = float(np.max(vibrations)) if vibrations else float("nan")
        vibration_mean = float(np.mean(vibrations)) if vibrations else float("nan")

        fuel_flow_mean = float(np.mean(fuel_flows)) if fuel_flows else float("nan")
        fuel_flow_total = float(np.sum(fuel_flows) * (dt / 3600.0)) if fuel_flows else float("nan")

        # 2. PHM Summary Extraction
        anomaly_count = sum(1 for p in aggregation_stream if p.anomaly_status in ("WARNING", "ANOMALY"))
        
        # Aggregate contributing anomaly channels
        channel_set = set()
        for p in aggregation_stream:
            for ch in p.anomaly_contributing_channels:
                channel_set.add(ch)
        dominant_anomaly_channels = sorted(list(channel_set))

        # Diagnosis (final valid step)
        final_diagnosis = effective_p.predicted_fault_class or "none"
        diag_probs = effective_p.diagnosis_probabilities or {}
        final_diagnosis_prob = float(diag_probs.get(final_diagnosis, effective_p.diagnostic_confidence))

        # Evidence quality from Phase 12 or Phase 8 data quality
        evidence_quality = "VALID"
        if not valid_payloads:
            evidence_quality = "INSUFFICIENT_DATA"
        elif effective_p.fused_evidence and isinstance(effective_p.fused_evidence, dict):
            evidence_quality = str(effective_p.fused_evidence.get("evidence_quality", "HIGH"))
        elif effective_p.diagnosis_data_quality:
            evidence_quality = str(effective_p.diagnosis_data_quality)

        # Health Index
        his = [p.smoothed_health_index for p in aggregation_stream if p.smoothed_health_index is not None and not np.isnan(p.smoothed_health_index)]
        final_hi = effective_p.smoothed_health_index if effective_p.smoothed_health_index is not None else float("nan")
        min_hi = float(np.min(his)) if his else final_hi
        deg_trend = effective_p.degradation_trend

        # Prognostics RUL
        final_rul = effective_p.point_rul_seconds
        rul_p05 = effective_p.rul_uncertainty_p05
        rul_p95 = effective_p.rul_uncertainty_p95
        limiting_factor = effective_p.limiting_factor or "NONE"
        forecast_status = effective_p.forecast_status
        forecast_source = effective_p.forecast_source

        # 3. Explainability
        dominant_shap = []
        if effective_p.shap_attribution and isinstance(effective_p.shap_attribution, dict):
            raw_shap = effective_p.shap_attribution.get("attributions") or effective_p.shap_attribution.get("top_features") or []
            if isinstance(raw_shap, list):
                dominant_shap = raw_shap
            elif isinstance(raw_shap, dict):
                dominant_shap = [{"feature": k, "weight": float(v)} for k, v in raw_shap.items()]

        phys_status = "UNKNOWN"
        phys_reason = ""
        if effective_p.physics_evidence and isinstance(effective_p.physics_evidence, dict):
            phys_status = str(effective_p.physics_evidence.get("status", "VALID"))
            phys_reason = str(effective_p.physics_evidence.get("consistency_reason", ""))

        temp_status = "STABLE"
        if effective_p.temporal_evidence and isinstance(effective_p.temporal_evidence, dict):
            temp_status = str(effective_p.temporal_evidence.get("status", effective_p.degradation_trend))

        default_headline = "Nominal operations verified across all sensors." if valid_payloads else "Insufficient data: all observations rejected."
        default_action = "Continue nominal mission monitoring." if valid_payloads else "Verify chronological and telemetry link integrity."
        fused_headline = effective_p.summary_explanation or default_headline
        recommended_action = effective_p.recommended_operator_action or default_action

        # 4. Decision Support Advisory Assessment
        adv = effective_p.advisory
        if adv is not None:
            adv_action_code = adv.action_code.value if hasattr(adv.action_code, "value") else str(adv.action_code)
            adv_headline = adv.headline
            adv_urgency = adv.urgency
        else:
            adv_action_code = AdvisoryActionCode.NORMAL_MONITORING.value if valid_payloads else AdvisoryActionCode.INSUFFICIENT_DATA.value
            adv_headline = "Nominal System Performance" if valid_payloads else "Telemetry Insufficient Data"
            adv_urgency = "LOW"

        if not valid_payloads:
            advisory_assessment = "INSUFFICIENT_DATA"
        elif adv_action_code in (AdvisoryActionCode.CRITICAL_ABORT_ACTION.value, AdvisoryActionCode.MAINTENANCE_INSPECTION.value):
            advisory_assessment = "MAINTENANCE"
        elif adv_action_code == AdvisoryActionCode.ADVISORY_CAUTION.value:
            advisory_assessment = "CAUTION"
        elif adv_action_code == AdvisoryActionCode.INSUFFICIENT_DATA.value:
            advisory_assessment = "INSUFFICIENT_DATA"
        else:
            advisory_assessment = "GO"

        # 5. Provenance & Limitations
        trailing_rejected = 0
        if valid_payloads and payloads[-1] != valid_payloads[-1]:
            for p in reversed(payloads):
                if "REJECTED" in (getattr(p, "quality_status", None) or ""):
                    trailing_rejected += 1
                else:
                    break

        provenance = {
            "engine_id": engine_id,
            "mission_id": mission_id,
            "total_samples": len(payloads),
            "valid_samples": len(valid_payloads),
            "rejected_samples_count": len(payloads) - len(valid_payloads),
            "trailing_rejected_samples": trailing_rejected,
            "step_dt": dt,
            "orchestrator_source": "Phase 13 Unified System Pipeline",
            "timesfm_status": forecast_status,
            "timesfm_source": forecast_source,
            "eol_definition": "Project-defined simulated functional-failure/EOL assumptions",
        }
        if trailing_rejected > 0:
            provenance["notice"] = (
                f"{trailing_rejected} trailing sample(s) rejected by causal sequence guard; "
                "report reflects the final valid operational observation."
            )

        return MissionReportSummary(
            engine_id=engine_id,
            mission_id=mission_id,
            duration_s=duration_s,
            mission_phase=mission_phase,
            simulation_mode=simulation_mode,
            scenario_name=scenario_name,
            scenario_metadata=meta,
            rpm_min=rpm_min,
            rpm_max=rpm_max,
            rpm_mean=rpm_mean,
            cht_peak=cht_peak,
            cht_mean=cht_mean,
            egt_peak=egt_peak,
            egt_mean=egt_mean,
            oil_temp_peak=oil_temp_peak,
            oil_temp_mean=oil_temp_mean,
            oil_pressure_min=oil_pressure_min,
            oil_pressure_mean=oil_pressure_mean,
            vibration_max=vibration_max,
            vibration_mean=vibration_mean,
            fuel_flow_total=fuel_flow_total,
            fuel_flow_mean=fuel_flow_mean,
            anomaly_events_count=anomaly_count,
            dominant_anomaly_channels=dominant_anomaly_channels,
            final_diagnosis=final_diagnosis,
            final_diagnosis_probability=final_diagnosis_prob,
            evidence_quality=evidence_quality,
            final_health_index=final_hi,
            min_health_index=min_hi,
            degradation_trend=deg_trend,
            final_rul_seconds=final_rul,
            rul_p05_seconds=rul_p05,
            rul_p95_seconds=rul_p95,
            limiting_factor=limiting_factor,
            forecast_status=forecast_status,
            forecast_source=forecast_source,
            dominant_shap_evidence=dominant_shap,
            physics_evidence_status=phys_status,
            physics_consistency_reason=phys_reason,
            temporal_evidence_status=temp_status,
            fused_evidence_headline=fused_headline,
            recommended_operator_action=recommended_action,
            advisory_action_code=adv_action_code,
            advisory_headline=adv_headline,
            advisory_urgency=adv_urgency,
            advisory_assessment=advisory_assessment,
            provenance=provenance,
        )
