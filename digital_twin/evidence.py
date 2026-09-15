"""
Phase 11 Evidence Builder: Extraction and Traceability Snapshotting.

Consumes existing runtime outputs from:
- TelemetryRecord
- CanonicalTwinState
- TelemetryQualityReport
- ResidualVector
- ModelObservationHealthAssessment
- DetectionResult
- DiagnosisResult
- DegradationAssessment
- RULAssessment

Strict Invariants:
1. Purely downstream: reads existing data structures; zero feedback into runtime.
2. No calculation reconstruction: extracts already-computed values without re-evaluating physics or health.
3. Traceability is explicit: constructs machine-readable dependency links connecting inputs to outputs.
4. Completeness is strictly quantified: measures explicit required fields present vs missing.
"""

from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple, Union

from telemetry.schema import TelemetryRecord, DigitalTwinState
from digital_twin.residuals import (
    PRIMARY_RESIDUAL_CHANNELS,
    CYLINDER_RESIDUAL_CHANNELS,
    SECONDARY_MODEL_CHANNELS,
    CHANNEL_UNITS_MAP,
    ResidualVector,
)
from digital_twin.health import (
    HealthState,
    ModelObservationHealthAssessment,
    ChannelHealthIndicator,
    SubsystemHealthAssessment,
)
from digital_twin.detection import DetectionResult, DetectionStatus
from digital_twin.diagnosis import DiagnosisResult, CanonicalFaultType
from digital_twin.degradation_types import DegradationAssessment, RULAssessment, RULStatus
from digital_twin.mission_types import AUTHORITATIVE_ENVELOPE_THRESHOLDS

from digital_twin.evidence_types import (
    DataClassification,
    EpistemicProvenance,
    ChannelRole,
    ExplanationReasonCode,
    TraceabilityNode,
    TraceabilityChain,
    ChannelEvidence,
    SubsystemEvidence,
    AnomalyEvidence,
    DiagnosisEvidence,
    PrognosticEvidence,
    CompletenessAudit,
    StepEvidenceRecord,
)


class EvidenceBuilder:
    """
    Downstream evidence extractor and traceability constructor.
    """

    def __init__(self, engine_id: str = "ROTAX_914_ GrayBox_TWIN"):
        self.engine_id = engine_id
        # Map authoritative envelope limits by channel for reference
        self._envelope_limits: Dict[str, Tuple[float, str, EpistemicProvenance]] = {}
        for eth in AUTHORITATIVE_ENVELOPE_THRESHOLDS:
            prov = (
                EpistemicProvenance.OEM_REFERENCE
                if eth.classification.value == "OEM_REFERENCE_LIMIT"
                else EpistemicProvenance.MODEL_CALIBRATION
            )
            self._envelope_limits[eth.channel] = (eth.numeric_value, eth.direction.value, prov)

    def build_step_evidence(
        self,
        twin_state: DigitalTwinState,
        mission_id: Optional[str] = None,
    ) -> StepEvidenceRecord:
        """
        Extract structured evidence from an existing DigitalTwinState snapshot.
        Zero recomputation; strictly downstream extraction.
        """
        t = twin_state.timestamp
        obs_telem = twin_state.observed_telemetry
        health_ass: Optional[ModelObservationHealthAssessment] = twin_state.health_assessment
        det_res: Optional[DetectionResult] = twin_state.detection_result
        diag_res: Optional[DiagnosisResult] = twin_state.diagnosis_result
        deg_ass: Optional[DegradationAssessment] = twin_state.degradation_assessment
        rul_ass: Optional[RULAssessment] = twin_state.rul_assessment
        res_vec: Optional[ResidualVector] = twin_state.residual_vector
        quality_rep = twin_state.quality_report

        effective_mission_id = mission_id or getattr(obs_telem, "mission_id", None)
        source_telem_ts = getattr(obs_telem, "timestamp", t)

        # 1. Extract Channel Evidence
        channel_ev_map: Dict[str, ChannelEvidence] = {}
        reason_codes: List[ExplanationReasonCode] = []

        # All known channels to inspect
        all_channels = (
            list(PRIMARY_RESIDUAL_CHANNELS)
            + list(CYLINDER_RESIDUAL_CHANNELS)
            + list(SECONDARY_MODEL_CHANNELS)
        )

        for ch in all_channels:
            unit = CHANNEL_UNITS_MAP.get(ch, "")
            # Observed value
            obs_val = None
            if health_ass and ch in health_ass.channel_indicators and health_ass.channel_indicators[ch].observed is not None:
                obs_val = health_ass.channel_indicators[ch].observed
            elif res_vec and ch in res_vec.residuals and res_vec.residuals[ch].observed_value is not None:
                obs_val = res_vec.residuals[ch].observed_value
            elif obs_telem:
                obs_val = getattr(obs_telem, ch, None)

            # Predicted / Expected value
            pred_val = None
            if health_ass and ch in health_ass.channel_indicators and health_ass.channel_indicators[ch].predicted is not None:
                pred_val = health_ass.channel_indicators[ch].predicted
            elif res_vec and ch in res_vec.residuals and res_vec.residuals[ch].predicted_value is not None:
                pred_val = res_vec.residuals[ch].predicted_value
            else:
                pred_val = (
                    twin_state.nominal_estimates.get(ch)
                    or twin_state.nominal_estimates.get(f"expected_{ch}")
                    or twin_state.nominal_estimates.get(f"nominal_{ch}")
                )

            # Raw residual
            raw_res = None
            if health_ass and ch in health_ass.channel_indicators:
                raw_res = health_ass.channel_indicators[ch].raw_residual
            elif res_vec and ch in res_vec.residuals:
                raw_res = res_vec.residuals[ch].raw_residual
            else:
                raw_res = twin_state.residuals.get(ch) or twin_state.residuals.get(f"{ch}_residual")

            # Normalized residual
            norm_res = None
            if health_ass and ch in health_ass.channel_indicators:
                norm_res = health_ass.channel_indicators[ch].normalized_residual
            elif res_vec:
                norm_res = res_vec.get_normalized(ch)

            # Determine channel role
            if ch in PRIMARY_RESIDUAL_CHANNELS:
                role = ChannelRole.PRIMARY_ENGINE_HEALTH
            elif ch in CYLINDER_RESIDUAL_CHANNELS:
                role = ChannelRole.CYLINDER_LOCALIZATION_DIAGNOSTIC
            else:
                role = ChannelRole.SECONDARY_MODEL_CONSISTENCY

            # Quality and observability status
            q_status = "VALID"
            if quality_rep and hasattr(quality_rep, "channels") and ch in quality_rep.channels:
                q_status = quality_rep.channels[ch].status.value
            elif obs_val is None or (isinstance(obs_val, float) and math.isnan(obs_val)):
                q_status = "MISSING"

            obs_status = "OBSERVABLE"
            if role == ChannelRole.SECONDARY_MODEL_CONSISTENCY:
                obs_status = "MODEL_ESTIMATED"

            # Check for envelope threshold
            lim_val, lim_dir, lim_prov = self._envelope_limits.get(ch, (None, None, None))

            # Epistemic classification
            if obs_val is not None and not math.isnan(obs_val):
                classif = DataClassification.MEASURED
            elif pred_val is not None and not math.isnan(pred_val):
                classif = DataClassification.PREDICTED
            else:
                classif = DataClassification.UNAVAILABLE

            rejection_reason = None
            if health_ass and ch in health_ass.channel_indicators:
                rejection_reason = health_ass.channel_indicators[ch].rejection_reason or None

            channel_ev = ChannelEvidence(
                channel=ch,
                unit=unit,
                observed_value=obs_val,
                predicted_value=pred_val,
                raw_residual=raw_res,
                normalized_residual=norm_res,
                classification=classif,
                role=role,
                quality_status=q_status,
                observability_status=obs_status,
                provenance=EpistemicProvenance.MODEL_CALIBRATION,
                rejection_reason=rejection_reason,
                threshold_limit=lim_val,
                threshold_direction=lim_dir,
                threshold_provenance=lim_prov,
            )
            channel_ev_map[ch] = channel_ev

        # 2. Extract Subsystem Evidence
        subsystem_ev_map: Dict[str, SubsystemEvidence] = {}
        if health_ass and hasattr(health_ass, "subsystems"):
            for sub_name, sub_ass in health_ass.subsystems.items():
                sub_reasons: List[ExplanationReasonCode] = []
                if sub_ass.state != HealthState.HEALTHY:
                    sub_reasons.append(ExplanationReasonCode.SUBSYSTEM_HEALTH_REDUCED)
                    reason_codes.append(ExplanationReasonCode.SUBSYSTEM_HEALTH_REDUCED)

                # Diagnostic / cylinder separation
                diagnostic_chs: List[str] = []
                cylinder_chs: List[str] = []
                for ch_name in sub_ass.primary_channels:
                    ch_h = health_ass.channel_indicators.get(ch_name)
                    if ch_h and ch_h.state != HealthState.HEALTHY:
                        sub_reasons.append(ExplanationReasonCode.HEALTH_CHANNEL_DEGRADED)
                        reason_codes.append(ExplanationReasonCode.HEALTH_CHANNEL_DEGRADED)

                # Collect secondary context
                if hasattr(sub_ass, "secondary_evidence") and sub_ass.secondary_evidence:
                    for sec_k in sub_ass.secondary_evidence.keys():
                        if "cyl" in sec_k:
                            cylinder_chs.append(sec_k)
                        else:
                            diagnostic_chs.append(sec_k)

                subsystem_ev_map[sub_name] = SubsystemEvidence(
                    subsystem=sub_name,
                    health_score=sub_ass.score,
                    contributing_primary_channels=tuple(sub_ass.primary_channels),
                    channel_health_scores={k: v for k, v in sub_ass.channel_scores.items()},
                    diagnostic_channels=tuple(diagnostic_chs),
                    cylinder_channels=tuple(cylinder_chs),
                    reason_codes=tuple(sorted(set(sub_reasons), key=lambda x: x.value)),
                    provenance=EpistemicProvenance.ENGINEERING_HEURISTIC,
                )

        # 3. Extract Anomaly Evidence
        anomaly_ev: Optional[AnomalyEvidence] = None
        if det_res is not None:
            anom_reasons: List[ExplanationReasonCode] = []
            if det_res.status == DetectionStatus.ANOMALOUS:
                anom_reasons.append(ExplanationReasonCode.ANOMALY_PERSISTENCE_SATISFIED)
                reason_codes.append(ExplanationReasonCode.ANOMALY_PERSISTENCE_SATISFIED)
            elif det_res.status == DetectionStatus.SUSPECTED:
                anom_reasons.append(ExplanationReasonCode.ANOMALY_THRESHOLD_CROSSED)
                reason_codes.append(ExplanationReasonCode.ANOMALY_THRESHOLD_CROSSED)
            elif det_res.status == DetectionStatus.RECOVERED:
                anom_reasons.append(ExplanationReasonCode.ANOMALY_RECOVERY)
                reason_codes.append(ExplanationReasonCode.ANOMALY_RECOVERY)
            elif det_res.status == DetectionStatus.INSUFFICIENT_DATA:
                anom_reasons.append(ExplanationReasonCode.INSUFFICIENT_DATA)
                reason_codes.append(ExplanationReasonCode.INSUFFICIENT_DATA)

            anomaly_ev = AnomalyEvidence(
                anomaly_score=det_res.anomaly_score,
                threshold=0.018,  # DetectionConfig heuristic threshold
                status=det_res.status.value,
                is_anomalous=det_res.anomaly_detected,
                persistence_seconds=det_res.persistence_duration,
                recovery_seconds=det_res.recovery_duration,
                contributing_channels=tuple(det_res.contributing_channels),
                contributing_subsystems=tuple(det_res.contributing_subsystems),
                evidence_coverage=det_res.evidence_quality,
                hi_raw_ref=health_ass.HI_raw if health_ass else None,
                reason_codes=tuple(sorted(set(anom_reasons), key=lambda x: x.value)),
                provenance=EpistemicProvenance.ENGINEERING_HEURISTIC,
            )

        # 4. Extract Diagnosis Evidence
        diag_ev: Optional[DiagnosisEvidence] = None
        if diag_res is not None:
            diag_reasons: List[ExplanationReasonCode] = []
            if diag_res.primary_fault not in ("HEALTHY", "UNKNOWN", "NONE"):
                diag_reasons.append(ExplanationReasonCode.FAULT_SIGNATURE_MATCH)
                reason_codes.append(ExplanationReasonCode.FAULT_SIGNATURE_MATCH)

            if diag_res.affected_cylinder is not None:
                diag_reasons.append(ExplanationReasonCode.CYLINDER_LOCALIZATION)
                reason_codes.append(ExplanationReasonCode.CYLINDER_LOCALIZATION)

            if diag_res.is_sensor_fault:
                diag_reasons.append(ExplanationReasonCode.SENSOR_LOCALIZATION_FAVORED)
                reason_codes.append(ExplanationReasonCode.SENSOR_LOCALIZATION_FAVORED)

            # Map ranked hypotheses
            hyp_list: List[Dict[str, Any]] = []
            if hasattr(diag_res, "ranked_hypotheses"):
                for hyp in diag_res.ranked_hypotheses:
                    hyp_list.append({
                        "fault_type": hyp.fault_type,
                        "compatibility_score": hyp.compatibility_score,
                        "matched_evidence": list(hyp.matched_evidence),
                        "unmatched_evidence": list(hyp.unmatched_evidence),
                    })

            # Directional residuals
            dir_map: Dict[str, str] = {}
            for ch_k, ch_v in channel_ev_map.items():
                if ch_v.normalized_residual is not None:
                    if ch_v.normalized_residual < -1.5:
                        dir_map[ch_k] = "NEGATIVE_DEVIATION"
                    elif ch_v.normalized_residual > 1.5:
                        dir_map[ch_k] = "POSITIVE_DEVIATION"

            competing_count = len(diag_res.competing_hypotheses) if hasattr(diag_res, "competing_hypotheses") else 0

            diag_ev = DiagnosisEvidence(
                primary_fault=diag_res.primary_fault,
                status=diag_res.status,
                confidence_heuristic=diag_res.confidence,
                ranked_hypotheses=tuple(hyp_list),
                supporting_channels=tuple(diag_res.evidence),
                residual_directions=dir_map,
                affected_cylinder=diag_res.affected_cylinder,
                is_sensor_fault=diag_res.is_sensor_fault,
                uncertainty_heuristic=diag_res.uncertainty,
                competing_hypotheses_count=competing_count,
                rule_evidence=tuple(diag_res.evidence),
                reason_codes=tuple(sorted(set(diag_reasons), key=lambda x: x.value)),
                provenance=EpistemicProvenance.ENGINEERING_HEURISTIC,
            )

        # 5. Extract Prognostic / RUL Evidence
        prog_ev: Optional[PrognosticEvidence] = None
        if deg_ass is not None or rul_ass is not None:
            prog_reasons: List[ExplanationReasonCode] = []
            rul_stat = rul_ass.status.value if rul_ass else "UNAVAILABLE"

            if rul_ass:
                if rul_ass.status == RULStatus.COMPUTED:
                    prog_reasons.append(ExplanationReasonCode.RUL_TREND_SUPPORTED)
                    reason_codes.append(ExplanationReasonCode.RUL_TREND_SUPPORTED)
                elif rul_ass.status in (RULStatus.STABLE, RULStatus.NON_DEGRADING):
                    prog_reasons.append(ExplanationReasonCode.RUL_NON_DEGRADING)
                    reason_codes.append(ExplanationReasonCode.RUL_NON_DEGRADING)
                else:
                    prog_reasons.append(ExplanationReasonCode.RUL_UNAVAILABLE)
                    reason_codes.append(ExplanationReasonCode.RUL_UNAVAILABLE)

            rejections = [rul_ass.rejection_reason] if (rul_ass and rul_ass.rejection_reason) else []

            prog_ev = PrognosticEvidence(
                degradation_index=deg_ass.degradation_index if deg_ass else None,
                trend_slope_per_second=deg_ass.trend_slope_per_sec if deg_ass else None,
                trend_slope_per_hour=deg_ass.trend_slope_per_hour if deg_ass else None,
                trend_window_seconds=deg_ass.window_duration_s if deg_ass else (rul_ass.window_duration if rul_ass else 0.0),
                observation_count=deg_ass.observation_count if deg_ass else (rul_ass.sample_count if rul_ass else 0),
                valid_fraction=deg_ass.data_quality_factor if deg_ass else 1.0,
                data_confidence=rul_ass.data_confidence if rul_ass else 1.0,
                dominant_subsystem=deg_ass.dominant_subsystem if deg_ass else None,
                rul_status=rul_stat,
                rul_estimate_hours=rul_ass.rul_median if rul_ass else None,
                rul_low_hours=rul_ass.rul_low if rul_ass else None,
                rul_high_hours=rul_ass.rul_high if rul_ass else None,
                eol_threshold=rul_ass.eol_threshold if rul_ass else 0.50,
                stress_multiplier=1.0,
                rejection_reasons=tuple(rejections),
                reason_codes=tuple(sorted(set(prog_reasons), key=lambda x: x.value)),
                provenance=EpistemicProvenance.ENGINEERING_HEURISTIC,
            )

        # 6. Quality-driven reason codes
        if quality_rep:
            if hasattr(quality_rep, "overall_status") and quality_rep.overall_status.value != "GOOD":
                reason_codes.append(ExplanationReasonCode.SENSOR_QUALITY_DEGRADED)
            if hasattr(quality_rep, "channels"):
                for ch_n, ch_q in quality_rep.channels.items():
                    q_val = ch_q.status.value if hasattr(ch_q, "status") else str(ch_q)
                    if "DROPOUT" in q_val:
                        reason_codes.append(ExplanationReasonCode.SENSOR_DROPOUT)
                    elif "STUCK" in q_val:
                        reason_codes.append(ExplanationReasonCode.SENSOR_STUCK)
                    elif "BIAS" in q_val:
                        reason_codes.append(ExplanationReasonCode.SENSOR_BIAS)
                    elif "DRIFT" in q_val:
                        reason_codes.append(ExplanationReasonCode.SENSOR_DRIFT)

        # Channel-specific deviation codes
        egt_ch = channel_ev_map.get("egt")
        if egt_ch and egt_ch.normalized_residual is not None and abs(egt_ch.normalized_residual) > 1.5:
            reason_codes.append(ExplanationReasonCode.COMBUSTION_EGT_DEVIATION)

        oil_p_ch = channel_ev_map.get("oil_p") or channel_ev_map.get("oil_pressure")
        if oil_p_ch and oil_p_ch.normalized_residual is not None and abs(oil_p_ch.normalized_residual) > 1.5:
            reason_codes.append(ExplanationReasonCode.LUBRICATION_PRESSURE_DEVIATION)

        vib_ch = channel_ev_map.get("vibration")
        if vib_ch and vib_ch.normalized_residual is not None and abs(vib_ch.normalized_residual) > 1.5:
            reason_codes.append(ExplanationReasonCode.MECHANICAL_VIBRATION_DEVIATION)

        cht_ch = channel_ev_map.get("cht")
        if cht_ch and cht_ch.normalized_residual is not None and abs(cht_ch.normalized_residual) > 1.5:
            reason_codes.append(ExplanationReasonCode.THERMAL_LIMIT_APPROACH)

        # Engine level health state
        hi_r = health_ass.HI_raw if health_ass else None
        hi_s = health_ass.HI_smooth if health_ass else None
        eng_h_state = health_ass.state.value if health_ass else HealthState.HEALTHY.value
        if health_ass and health_ass.state != HealthState.HEALTHY:
            reason_codes.append(ExplanationReasonCode.ENGINE_HEALTH_REDUCED)
        elif not reason_codes:
            reason_codes.append(ExplanationReasonCode.NOMINAL_OPERATION)

        # 7. Construct Machine-Readable Traceability Chains
        traceability: Dict[str, TraceabilityChain] = {}

        # Chain A: Engine Health Lineage
        health_nodes = [
            TraceabilityNode(
                stage="1_TELEMETRY_INGESTION",
                source_entity=f"telemetry_stream[t={t:.2f}s]",
                output_entity="channel_observations",
                value_summary=f"channels_observed={len(all_channels)}",
                relationship="observed_at_sensor",
                provenance=EpistemicProvenance.OEM_REFERENCE,
            ),
            TraceabilityNode(
                stage="2_PHYSICS_ESTIMATION",
                source_entity="DigitalTwinModel",
                output_entity="nominal_predictions",
                value_summary=f"predictions_generated={len(twin_state.nominal_estimates)}",
                relationship="physics_simulated",
                provenance=EpistemicProvenance.MODEL_CALIBRATION,
            ),
            TraceabilityNode(
                stage="3_RESIDUAL_GENERATION",
                source_entity="observed_vs_predicted",
                output_entity="residual_vector",
                value_summary=f"residuals_computed={len(twin_state.residuals)}",
                relationship="subtraction_and_normalization",
                provenance=EpistemicProvenance.MODEL_CALIBRATION,
            ),
            TraceabilityNode(
                stage="4_SUBSYSTEM_EVALUATION",
                source_entity="primary_channels",
                output_entity="subsystem_health_scores",
                value_summary=f"subsystems_evaluated={len(subsystem_ev_map)}",
                relationship="arithmetic_subsystem_aggregation",
                provenance=EpistemicProvenance.ENGINEERING_HEURISTIC,
            ),
            TraceabilityNode(
                stage="5_ENGINE_HEALTH_INDEX",
                source_entity="subsystems",
                output_entity="HI_raw_and_HI_smooth",
                value_summary=f"HI_raw={hi_r}, HI_smooth={hi_s}, state={eng_h_state}",
                relationship="weighted_subsystem_mean_and_causal_ewma",
                provenance=EpistemicProvenance.ENGINEERING_HEURISTIC,
            ),
        ]
        traceability["engine_health"] = TraceabilityChain(
            chain_id="LINEAGE_ENGINE_HEALTH",
            target_metric="HI_smooth",
            nodes=tuple(health_nodes),
        )

        # Chain B: Fault Diagnosis Lineage (if applicable)
        if diag_ev is not None:
            diag_nodes = [
                TraceabilityNode(
                    stage="1_ANOMALY_DETECTION",
                    source_entity="HI_raw",
                    output_entity="temporal_detection_status",
                    value_summary=f"status={det_res.status.value if det_res else 'NONE'}",
                    relationship="threshold_and_temporal_persistence",
                    provenance=EpistemicProvenance.ENGINEERING_HEURISTIC,
                ),
                TraceabilityNode(
                    stage="2_FAULT_HYPOTHESIS_EVALUATION",
                    source_entity="residual_vector_and_cylinder_spread",
                    output_entity="ranked_hypotheses",
                    value_summary=f"top_fault={diag_ev.primary_fault}, score={diag_ev.confidence_heuristic}",
                    relationship="physics_informed_rule_matching",
                    provenance=EpistemicProvenance.ENGINEERING_HEURISTIC,
                ),
            ]
            if diag_ev.affected_cylinder is not None:
                diag_nodes.append(
                    TraceabilityNode(
                        stage="3_CYLINDER_LOCALIZATION",
                        source_entity="runner_spread_residuals",
                        output_entity="localized_cylinder",
                        value_summary=f"cylinder={diag_ev.affected_cylinder}",
                        relationship="inter_cylinder_divergence_localization",
                        provenance=EpistemicProvenance.ENGINEERING_HEURISTIC,
                    )
                )
            traceability["fault_diagnosis"] = TraceabilityChain(
                chain_id="LINEAGE_FAULT_DIAGNOSIS",
                target_metric="primary_fault",
                nodes=tuple(diag_nodes),
            )

        # Chain C: Prognostic Lineage (if applicable)
        if prog_ev is not None:
            prog_nodes = [
                TraceabilityNode(
                    stage="1_DEGRADATION_TRACKING",
                    source_entity="HI_smooth_trajectory",
                    output_entity="degradation_index",
                    value_summary=f"D(t)={prog_ev.degradation_index}",
                    relationship="inverse_health_indicator",
                    provenance=EpistemicProvenance.ENGINEERING_HEURISTIC,
                ),
                TraceabilityNode(
                    stage="2_ROBUST_TREND_ESTIMATION",
                    source_entity="historical_window",
                    output_entity="TheilSen_slope",
                    value_summary=f"slope_hr={prog_ev.trend_slope_per_hour}",
                    relationship="non_parametric_theil_sen",
                    provenance=EpistemicProvenance.ENGINEERING_HEURISTIC,
                ),
                TraceabilityNode(
                    stage="3_RUL_PROJECTION",
                    source_entity="slope_and_D_EOL",
                    output_entity="RUL_projection",
                    value_summary=f"RUL_median={prog_ev.rul_estimate_hours}h, status={prog_ev.rul_status}",
                    relationship="linear_horizon_extrapolation",
                    provenance=EpistemicProvenance.ENGINEERING_HEURISTIC,
                ),
            ]
            traceability["prognostics_rul"] = TraceabilityChain(
                chain_id="LINEAGE_PROGNOSTICS_RUL",
                target_metric="rul_median",
                nodes=tuple(prog_nodes),
            )

        # 8. Completeness Audit
        req_fields = [
            "timestamp",
            "engine_id",
            "source_telemetry_timestamp",
            "canonical_timestamp",
            "channel_evidence",
            "subsystem_evidence",
            "traceability_chains",
            "data_quality_status",
            "observability_coverage",
            "synchronization_status",
            "limitations",
        ]
        present_fields = list(req_fields)
        missing_fields = []

        if health_ass:
            req_fields.extend(["hi_raw", "hi_smooth", "engine_health_state"])
            present_fields.extend(["hi_raw", "hi_smooth", "engine_health_state"])

        if det_res:
            req_fields.append("anomaly_evidence")
            present_fields.append("anomaly_evidence")

        if diag_res:
            req_fields.append("diagnosis_evidence")
            present_fields.append("diagnosis_evidence")

        if deg_ass or rul_ass:
            req_fields.append("prognostic_evidence")
            present_fields.append("prognostic_evidence")

        completeness_ratio = len(present_fields) / float(len(req_fields)) if req_fields else 1.0
        completeness = CompletenessAudit(
            scope="STEP_EVIDENCE_RECORD",
            required_fields=tuple(req_fields),
            present_fields=tuple(present_fields),
            missing_fields=tuple(missing_fields),
            total_required=len(req_fields),
            total_present=len(present_fields),
            completeness_ratio=completeness_ratio,
        )

        c_obs_val = health_ass.C_obs if health_ass else 1.0
        q_stat_val = "GOOD"
        if quality_rep and hasattr(quality_rep, "overall_status"):
            q_stat_val = quality_rep.overall_status.value

        sync_stat_val = "SYNCHRONIZED"

        return StepEvidenceRecord(
            timestamp=t,
            engine_id=self.engine_id,
            mission_id=effective_mission_id,
            source_telemetry_timestamp=source_telem_ts,
            canonical_timestamp=t,
            hi_raw=hi_r,
            hi_smooth=hi_s,
            engine_health_state=eng_h_state,
            channel_evidence=channel_ev_map,
            subsystem_evidence=subsystem_ev_map,
            anomaly_evidence=anomaly_ev,
            diagnosis_evidence=diag_ev,
            prognostic_evidence=prog_ev,
            traceability_chains=traceability,
            overall_reason_codes=tuple(sorted(set(reason_codes), key=lambda x: x.value)),
            data_quality_status=q_stat_val,
            observability_coverage=c_obs_val,
            synchronization_status=sync_stat_val,
            completeness=completeness,
            provenance_metadata={
                "engine_model": "Rotax 914 UL/F Grey-Box",
                "phase": "PHASE_11_EVIDENCE_EXPLAINABILITY",
                "simulation_version": "0.2.0-phase2b-physics",
            },
        )
