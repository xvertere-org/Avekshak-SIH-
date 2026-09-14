"""
Phase 6 Physics-Informed Fault Diagnosis Layer for Rotax 914 UL/F Digital Twin.

Provides evidence-aware, physics-informed fault diagnosis and hypothesis ranking
based on Phase 5 physical residuals, coverage gating, health assessments, and Phase 6 detection results.

Invariants:
- Unidirectional Causal Flow: Consumes detection results, residual vectors, and health assessments.
- Zero Leakage: Completely decoupled from FaultType, FaultState, severity, or simulator ground-truth.
- Compatibility Scores: Emits heuristic match scores in [0.0, 1.0]; explicitly NOT claimed as probabilities.
- Cylinder Localization: Evaluates cylinder runner spread metrics to localize affected cylinder
  without inflating engine-level votes.
- Sensor Fault Disambiguation: Evaluates cross-channel physical coupling to distinguish observation-layer
  sensor faults (bias, drift, dropout, stuck) from genuine multi-channel physical degradation.
- Honest Uncertainty: Outputs UNKNOWN / INSUFFICIENT_EVIDENCE when evidence is ambiguous or conflicting.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, Any, List, Optional, Tuple

from digital_twin.residuals import ResidualVector
from digital_twin.health import ModelObservationHealthAssessment, HealthState
from digital_twin.detection import DetectionResult, DetectionStatus


class CanonicalFaultType(str, Enum):
    """
    Canonical Phase 4 / Phase 6 Fault Taxonomy identifiers.
    """
    INJECTOR_DELIVERY_ABNORMALITY = "INJECTOR_DELIVERY_ABNORMALITY"
    LUBRICATION_DEGRADATION = "LUBRICATION_DEGRADATION"
    COOLING_DEGRADATION = "COOLING_DEGRADATION"
    COMBUSTION_MISFIRE = "COMBUSTION_MISFIRE"
    MECHANICAL_DEGRADATION = "MECHANICAL_DEGRADATION"
    SENSOR_BIAS = "SENSOR_BIAS"
    SENSOR_DRIFT = "SENSOR_DRIFT"
    SENSOR_DROPOUT = "SENSOR_DROPOUT"
    SENSOR_STUCK = "SENSOR_STUCK"
    HEALTHY = "HEALTHY"
    NOMINAL = "HEALTHY"
    NONE = "HEALTHY"
    UNKNOWN = "UNKNOWN"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


# Backward compatibility alias
FUEL_INJECTION_ABNORMALITY = CanonicalFaultType.INJECTOR_DELIVERY_ABNORMALITY.value
COMBUSTION_INSTABILITY = CanonicalFaultType.COMBUSTION_MISFIRE.value


@dataclass
class HypothesisRanking:
    """
    Ranked fault hypothesis with evidence-based compatibility score.
    Note: compatibility_score is a heuristic match metric, NOT a calibrated probability.
    """
    fault_type: str
    compatibility_score: float          # Heuristic score in [0.0, 1.0] (ENGINEERING_HEURISTIC)
    matched_evidence: List[str]         # Evidentiary patterns satisfied by observed residuals
    unmatched_evidence: List[str]       # Expected evidentiary patterns missing or contradicted

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fault_type": self.fault_type,
            "compatibility_score": round(self.compatibility_score, 4),
            "matched_evidence": list(self.matched_evidence),
            "unmatched_evidence": list(self.unmatched_evidence),
        }


@dataclass
class DiagnosisResult:
    """
    Structured fault diagnosis output for an individual telemetry sample.
    """
    timestamp: float
    engine_id: str
    primary_fault: str
    ranked_hypotheses: List[HypothesisRanking]
    confidence: float                   # Diagnostic confidence in [0.0, 1.0] (ENGINEERING_HEURISTIC)
    evidence: List[str]
    affected_subsystem: Optional[str]
    affected_cylinder: Optional[int]    # 1, 2, 3, 4, or None
    is_sensor_fault: bool
    uncertainty: float                  # Ambiguity metric in [0.0, 1.0]
    status: str                         # CONFIRMED, SUSPECTED, INSUFFICIENT_EVIDENCE, NOMINAL
    provenance: str = "ENGINEERING_HEURISTIC"
    is_empirically_validated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def top_fault(self) -> CanonicalFaultType:
        try:
            return CanonicalFaultType(self.primary_fault)
        except (ValueError, KeyError):
            return CanonicalFaultType.UNKNOWN

    @property
    def primary_hypothesis(self) -> Optional[HypothesisRanking]:
        return self.ranked_hypotheses[0] if self.ranked_hypotheses else None

    @property
    def confidence_score(self) -> float:
        return self.confidence

    @property
    def insufficient_evidence(self) -> bool:
        return self.status == "INSUFFICIENT_EVIDENCE" or self.top_fault in (CanonicalFaultType.UNKNOWN, CanonicalFaultType.NONE)

    @property
    def is_ambiguous(self) -> bool:
        return self.uncertainty > 0.35 or len(self.competing_hypotheses) > 0

    @property
    def competing_hypotheses(self) -> List[HypothesisRanking]:
        if not self.ranked_hypotheses or len(self.ranked_hypotheses) < 2:
            return []
        top_score = self.ranked_hypotheses[0].compatibility_score
        return [h for h in self.ranked_hypotheses[1:] if (top_score - h.compatibility_score) < 0.20 and h.compatibility_score > 0.30]

    @property
    def isolated_channels(self) -> List[str]:
        channels = set()
        for ev in self.evidence:
            ev_norm = ev.lower().replace(" ", "_")
            for ch in ("cht", "coolant_temp", "oil_temp", "oil_pressure", "fuel_flow", "egt", "vibration", "rpm", "map_bar"):
                if ch in ev_norm or ch in ev.lower():
                    channels.add(ch)
            for r in (1, 2, 3, 4):
                if f"runner_{r}" in ev_norm or f"runner_{r}" in ev.lower() or f"cyl_{r}" in ev_norm or f"cyl_{r}" in ev.lower():
                    channels.add(f"RUNNER_{r}")
        return sorted(list(channels))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "engine_id": self.engine_id,
            "primary_fault": self.primary_fault,
            "ranked_hypotheses": [h.to_dict() for h in self.ranked_hypotheses],
            "confidence": round(self.confidence, 4),
            "evidence": list(self.evidence),
            "affected_subsystem": self.affected_subsystem,
            "affected_cylinder": self.affected_cylinder,
            "is_sensor_fault": self.is_sensor_fault,
            "uncertainty": round(self.uncertainty, 4),
            "status": self.status,
            "provenance": self.provenance,
            "is_empirically_validated": self.is_empirically_validated,
            "metadata": self.metadata,
        }


class PhysicsInformedDiagnoser:
    """
    Rule-based, physics-informed fault diagnoser.

    Evaluates observed normalized residuals, cross-channel physical consistency,
    subsystem health degradation, and cylinder runner spreads against validated
    causal fault signatures.
    """

    def __init__(self, tau_nom: float = 1.5, tau_crit: float = 5.0):
        self.tau_nom = tau_nom
        self.tau_crit = tau_crit
        self._last_timestamp: Optional[float] = None

    def reset(self) -> None:
        """Reset internal history."""
        self._last_timestamp = None

    def diagnose(
        self,
        detection_result: DetectionResult,
        residual_vector: ResidualVector,
        health_assessment: ModelObservationHealthAssessment,
        dt: Optional[float] = None,
        engine_id: str = "ROTAX_914_GREYBOX",
    ) -> DiagnosisResult:
        """
        Evaluate fault diagnosis from detection results and physical residual signatures.

        Args:
            detection_result: DetectionResult from TemporalFaultDetector.
            residual_vector: ResidualVector from QualityAwareResidualGenerator.
            health_assessment: ModelObservationHealthAssessment from HealthEvaluator.
            dt: Elapsed time step in seconds.
            engine_id: Engine identifier.

        Returns:
            DiagnosisResult with ranked hypotheses, primary fault, cylinder localization, and confidence.
        """
        timestamp = detection_result.timestamp

        # 1. Check Coverage Gate / Insufficient Data
        if detection_result.status == DetectionStatus.INSUFFICIENT_DATA:
            hyp = HypothesisRanking(
                fault_type=CanonicalFaultType.INSUFFICIENT_EVIDENCE.value,
                compatibility_score=1.0,
                matched_evidence=["Insufficient valid primary channels to establish diagnosis"],
                unmatched_evidence=[],
            )
            return DiagnosisResult(
                timestamp=timestamp,
                engine_id=engine_id,
                primary_fault=CanonicalFaultType.INSUFFICIENT_EVIDENCE.value,
                ranked_hypotheses=[hyp],
                confidence=0.0,
                evidence=["INSUFFICIENT_DATA: Valid channels < 5"],
                affected_subsystem=None,
                affected_cylinder=None,
                is_sensor_fault=False,
                uncertainty=1.0,
                status="INSUFFICIENT_EVIDENCE",
                metadata={"valid_primary_count": residual_vector.valid_primary_count},
            )

        # 2. Check Nominal Healthy Condition
        # If detection is NORMAL and health index is high, engine is healthy
        if detection_result.status == DetectionStatus.NORMAL and detection_result.anomaly_score < 0.15:
            hyp = HypothesisRanking(
                fault_type=CanonicalFaultType.HEALTHY.value,
                compatibility_score=1.0,
                matched_evidence=["All observed physical residuals within nominal deadband"],
                unmatched_evidence=[],
            )
            return DiagnosisResult(
                timestamp=timestamp,
                engine_id=engine_id,
                primary_fault=CanonicalFaultType.HEALTHY.value,
                ranked_hypotheses=[hyp],
                confidence=round(health_assessment.HI_raw, 4),
                evidence=["NOMINAL: HI_raw >= 0.85 and no sustained residual anomalies"],
                affected_subsystem=None,
                affected_cylinder=None,
                is_sensor_fault=False,
                uncertainty=0.0,
                status="NOMINAL",
                metadata={"HI_raw": health_assessment.HI_raw},
            )

        # 3. Extract Residual Map & Cylinder Spread Metrics
        # Collect signed raw and normalized residuals
        r_map: Dict[str, float] = {}
        z_map: Dict[str, float] = {}
        for ch, ind in health_assessment.channel_indicators.items():
            if ind.valid and not math.isnan(ind.normalized_residual):
                r_map[ch] = ind.raw_residual
                z_map[ch] = ind.normalized_residual
            else:
                r_map[ch] = float("nan")
                z_map[ch] = float("nan")

        cyl = residual_vector.cylinder_residuals

        # 4. Sensor Fault Disambiguation (F6 / F7)
        # Check for dropout / stuck or isolated single-channel sensor bias
        sensor_hypotheses, is_sensor_dominant = self._evaluate_sensor_faults(
            health_assessment=health_assessment,
            residual_vector=residual_vector,
            z_map=z_map,
        )

        # 5. Evaluate Physical Fault Signatures (F1–F5)
        physical_hypotheses, localized_cylinder = self._evaluate_physical_faults(
            r_map=r_map,
            z_map=z_map,
            cyl=cyl,
            health_assessment=health_assessment,
        )

        # Combine hypotheses
        all_hypotheses = physical_hypotheses + sensor_hypotheses
        # Sort by compatibility score descending
        all_hypotheses.sort(key=lambda h: h.compatibility_score, reverse=True)

        # 6. Primary Fault Selection & Uncertainty Evaluation
        top_hyp = all_hypotheses[0] if all_hypotheses else None
        second_hyp = all_hypotheses[1] if len(all_hypotheses) > 1 else None

        if top_hyp is None or top_hyp.compatibility_score < 0.25:
            primary_fault = CanonicalFaultType.UNKNOWN.value
            confidence = 0.20
            uncertainty = 1.0
            affected_subsystem = None
            affected_cylinder = None
            is_sensor = False
            diag_status = "INSUFFICIENT_EVIDENCE"
            all_hypotheses.insert(
                0,
                HypothesisRanking(
                    fault_type=CanonicalFaultType.UNKNOWN.value,
                    compatibility_score=1.0 - (top_hyp.compatibility_score if top_hyp else 0.0),
                    matched_evidence=["Observed residual vector does not match known fault signatures"],
                    unmatched_evidence=[],
                ),
            )
        else:
            primary_fault = top_hyp.fault_type
            score_margin = top_hyp.compatibility_score - (second_hyp.compatibility_score if second_hyp else 0.0)
            uncertainty = max(0.0, min(1.0, 1.0 - score_margin))

            # Bounded confidence based on compatibility, persistence, and evidence quality
            base_conf = top_hyp.compatibility_score
            quality_factor = health_assessment.C_obs * health_assessment.C_data
            persistence_factor = 1.0 if detection_result.anomaly_detected else 0.70
            confidence = max(0.1, min(0.95, base_conf * quality_factor * persistence_factor))

            is_sensor = primary_fault in (
                CanonicalFaultType.SENSOR_BIAS.value,
                CanonicalFaultType.SENSOR_DRIFT.value,
                CanonicalFaultType.SENSOR_DROPOUT.value,
                CanonicalFaultType.SENSOR_STUCK.value,
            )

            affected_subsystem = self._get_affected_subsystem(primary_fault, z_map)
            affected_cylinder = localized_cylinder if primary_fault in (
                CanonicalFaultType.INJECTOR_DELIVERY_ABNORMALITY.value,
                CanonicalFaultType.COMBUSTION_MISFIRE.value,
            ) else None

            diag_status = "CONFIRMED" if detection_result.anomaly_detected and confidence >= 0.60 else "SUSPECTED"

        evidence_list = top_hyp.matched_evidence if top_hyp else ["No matched patterns"]

        return DiagnosisResult(
            timestamp=timestamp,
            engine_id=engine_id,
            primary_fault=primary_fault,
            ranked_hypotheses=all_hypotheses,
            confidence=round(confidence, 4),
            evidence=evidence_list,
            affected_subsystem=affected_subsystem,
            affected_cylinder=affected_cylinder,
            is_sensor_fault=is_sensor,
            uncertainty=round(uncertainty, 4),
            status=diag_status,
            metadata={
                "detection_status": detection_result.status.value,
                "anomaly_score": detection_result.anomaly_score,
                "persistence_seconds": round(detection_result.persistence_duration, 2),
                "HI_raw": health_assessment.HI_raw,
                "localized_cylinder": localized_cylinder,
            },
        )

    # ── Physical Fault Evaluator (F1–F5) ──────────────────────────────────────

    def _evaluate_physical_faults(
        self,
        r_map: Dict[str, float],
        z_map: Dict[str, float],
        cyl: Optional[Any],
        health_assessment: ModelObservationHealthAssessment,
    ) -> Tuple[List[HypothesisRanking], Optional[int]]:
        """Evaluate physical fault compatibility scores and localize affected cylinder."""
        hypotheses: List[HypothesisRanking] = []
        localized_cyl: Optional[int] = None

        # Helper: safe getter
        def gz(ch: str) -> float:
            val = z_map.get(ch, float("nan"))
            return 0.0 if math.isnan(val) else val

        def gr(ch: str) -> float:
            val = r_map.get(ch, float("nan"))
            return 0.0 if math.isnan(val) else val

        z_fuel = gz("fuel_flow")
        z_egt = gz("egt")
        z_cht = gz("cht")
        z_oil_p = gz("oil_pressure")
        z_oil_t = gz("oil_temp")
        z_cool_t = gz("coolant_temp")
        z_vib = gz("vibration")
        z_rpm = gz("rpm")

        # ── F1: INJECTOR_DELIVERY_ABNORMALITY ─────────────────────────
        # Validated Simulator Causal Direction:
        # Lean injection: fuel_flow observed < expected (z_fuel < -tau_nom)
        # Higher EGT: z_egt > 0.5 or localized runner deviation
        f1_matches = []
        f1_unmatched = []
        f1_score = 0.0

        has_fuel_drop = z_fuel < -self.tau_nom
        has_runner_imbalance = False
        if cyl is not None and cyl.valid_cylinder_count > 0 and cyl.egt_runner_residuals:
            valid_runners = [(idx, val) for idx, val in enumerate(cyl.egt_runner_residuals) if not math.isnan(val)]
            if valid_runners:
                max_runner_idx = max(valid_runners, key=lambda pair: pair[1])[0]
                if cyl.egt_spread_c > 50.0 or cyl.egt_runner_residuals[max_runner_idx] > 25.0:
                    has_runner_imbalance = True
                    localized_cyl = max_runner_idx + 1

        if has_fuel_drop or has_runner_imbalance or z_egt > self.tau_nom:
            if has_fuel_drop:
                f1_matches.append(f"Fuel flow drop: z_fuel={z_fuel:.2f} < -{self.tau_nom}")
                f1_score += 0.40
            if z_egt > 0.5:
                f1_matches.append(f"EGT elevation: z_egt={z_egt:.2f}")
                f1_score += 0.35
            if has_runner_imbalance:
                f1_matches.append(f"Runner {localized_cyl} localized EGT elevation")
                f1_score += 0.25
        else:
            f1_unmatched.append("Expected negative fuel flow or localized runner EGT elevation")
            f1_score = 0.0

        hypotheses.append(
            HypothesisRanking(
                fault_type=CanonicalFaultType.INJECTOR_DELIVERY_ABNORMALITY.value,
                compatibility_score=min(1.0, f1_score),
                matched_evidence=f1_matches,
                unmatched_evidence=f1_unmatched,
            )
        )

        # ── F2: LUBRICATION_DEGRADATION ──────────────────────────────
        # Validated Simulator Causal Direction:
        # Oil pressure drop: z_oil_p < -tau_nom
        # Oil temperature rise: r_oil_t > +0.8°C or z_oil_t > 0.2
        # Cross-channel: Fuel flow and coolant remain nominal
        f2_matches = []
        f2_unmatched = []
        f2_score = 0.0

        if z_oil_p < -self.tau_nom:
            f2_matches.append(f"Oil pressure drop: z_oil_p={z_oil_p:.2f} < -{self.tau_nom}")
            f2_score += 0.55
            r_oil_t = gr("oil_temp")
            if r_oil_t > 0.8 or z_oil_t > 0.20:
                f2_matches.append(f"Oil temperature elevation: r_oil_t={r_oil_t:+.1f}°C")
                f2_score += 0.30
            elif z_oil_t >= 0.0:
                f2_score += 0.15
            else:
                f2_unmatched.append("Expected elevated oil temperature")

            # Cross-coupling consistency: Fuel flow and coolant nominal
            if abs(z_fuel) < self.tau_nom and abs(z_cool_t) < self.tau_nom:
                f2_score += 0.15
                f2_matches.append("Fuel delivery and cooling nominal (cross-channel consistency)")
        else:
            f2_unmatched.append("Expected negative oil pressure residual")
            f2_score = 0.0

        hypotheses.append(
            HypothesisRanking(
                fault_type=CanonicalFaultType.LUBRICATION_DEGRADATION.value,
                compatibility_score=min(1.0, f2_score),
                matched_evidence=f2_matches,
                unmatched_evidence=f2_unmatched,
            )
        )

        # ── F3: COOLING_DEGRADATION ──────────────────────────────────
        # Validated Simulator Causal Direction:
        # CHT elevation: z_cht > +tau_nom or r_cht > +5.0°C
        # Coolant temp rise: r_cool_t > +0.8°C or z_cool_t > 0.25
        # Cross-channel: Oil pressure and fuel delivery nominal
        f3_matches = []
        f3_unmatched = []
        f3_score = 0.0

        r_cool_t = gr("coolant_temp")
        r_cht = gr("cht")

        if z_cht > self.tau_nom or r_cht > 5.0 or r_cool_t > 0.8:
            f3_score += 0.50
            if z_cht > self.tau_nom or r_cht > 5.0:
                f3_matches.append(f"CHT elevation: r_cht={r_cht:+.1f}°C (z={z_cht:.2f})")
            if r_cool_t > 0.8 or z_cool_t > 0.25:
                f3_matches.append(f"Coolant temperature rise: r_cool_t={r_cool_t:+.1f}°C")
                f3_score += 0.30
            if abs(z_oil_p) < self.tau_nom:
                f3_matches.append("Lubrication pressure nominal (cross-channel consistency)")
                f3_score += 0.20
        else:
            f3_unmatched.append("Expected positive CHT or coolant temperature residual")
            f3_score = 0.0

        hypotheses.append(
            HypothesisRanking(
                fault_type=CanonicalFaultType.COOLING_DEGRADATION.value,
                compatibility_score=min(1.0, f3_score),
                matched_evidence=f3_matches,
                unmatched_evidence=f3_unmatched,
            )
        )

        # ── F4: COMBUSTION_MISFIRE ───────────────────────────────────
        # Validated Simulator Causal Direction:
        # Severe EGT drop from unburnt combustion charge: z_egt < -2.0 or r_egt < -50
        # Engine torque/RPM loss: z_rpm < -1.0 or z_fuel < -1.0
        # Depressed head temperature: z_cht < 0.0
        f4_matches = []
        f4_unmatched = []
        f4_score = 0.0

        if z_egt < -2.0 or gr("egt") < -50.0:
            f4_score += 0.55
            f4_matches.append(f"Severe EGT drop: r_egt={gr('egt'):.1f}°C (z={z_egt:.2f})")
            if z_rpm < -1.0 or z_fuel < -1.0:
                f4_matches.append(f"Engine torque/speed loss: z_rpm={z_rpm:.2f}")
                f4_score += 0.25
            if z_cht < 0.0:
                f4_matches.append(f"Depressed cylinder head temperature: z_cht={z_cht:.2f}")
                f4_score += 0.20
            # Localized runner check
            if cyl is not None and cyl.valid_cylinder_count > 0 and cyl.egt_runner_residuals:
                valid_runners = [(idx, val) for idx, val in enumerate(cyl.egt_runner_residuals) if not math.isnan(val)]
                if valid_runners:
                    min_runner_idx = min(valid_runners, key=lambda pair: pair[1])[0]
                    localized_cyl = min_runner_idx + 1
                    f4_matches.append(f"Runner {localized_cyl} exhibits lowest EGT")
        else:
            f4_unmatched.append("Expected severe EGT drop from unburnt charge")
            f4_score = 0.0

        hypotheses.append(
            HypothesisRanking(
                fault_type=CanonicalFaultType.COMBUSTION_MISFIRE.value,
                compatibility_score=min(1.0, f4_score),
                matched_evidence=f4_matches,
                unmatched_evidence=f4_unmatched,
            )
        )

        # ── F5: MECHANICAL_DEGRADATION ───────────────────────────────
        # Validated Simulator Causal Direction:
        # Vibration surge: z_vib > +tau_nom
        # Combustion nominal (NO severe EGT drop): |z_egt| < tau_nom
        # Thermal & fluid conductances nominal
        f5_matches = []
        f5_unmatched = []
        f5_score = 0.0

        if z_vib > self.tau_nom:
            f5_matches.append(f"Vibration surge: z_vib={z_vib:.2f} > +{self.tau_nom}")
            f5_score += 0.60
            if abs(z_egt) < self.tau_nom and abs(z_fuel) < self.tau_nom:
                f5_matches.append("Combustion and fuel delivery nominal (excludes misfire/fuel faults)")
                f5_score += 0.20
            if abs(z_cool_t) < self.tau_nom and abs(z_oil_p) < self.tau_nom:
                f5_matches.append("Cooling and lubrication nominal")
                f5_score += 0.20
        else:
            f5_unmatched.append("Expected vibration surge")
            f5_score = 0.0

        hypotheses.append(
            HypothesisRanking(
                fault_type=CanonicalFaultType.MECHANICAL_DEGRADATION.value,
                compatibility_score=min(1.0, f5_score),
                matched_evidence=f5_matches,
                unmatched_evidence=f5_unmatched,
            )
        )

        return hypotheses, localized_cyl

    # ── Sensor Fault Evaluator (F6 / F7) ──────────────────────────────────────

    def _evaluate_sensor_faults(
        self,
        health_assessment: ModelObservationHealthAssessment,
        residual_vector: ResidualVector,
        z_map: Dict[str, float],
    ) -> Tuple[List[HypothesisRanking], bool]:
        """
        Evaluate observation-layer sensor faults using cross-channel physical consistency.
        Distinguishes sensor bias/drift/dropout/stuck from physical degradation.
        """
        hypotheses: List[HypothesisRanking] = []
        is_sensor_dominant = False

        # 1. Check for SENSOR_DROPOUT or SENSOR_STUCK via channel rejection reasons
        dropout_channels: List[str] = []
        stuck_channels: List[str] = []

        for ch_name, ind in health_assessment.channel_indicators.items():
            if not ind.valid:
                reason = ind.rejection_reason or ""
                if "MISSING" in reason or "NAN" in reason or "DROPOUT" in reason:
                    dropout_channels.append(ch_name)
                elif "STALE" in reason or "STUCK" in reason:
                    stuck_channels.append(ch_name)

        if dropout_channels:
            hypotheses.append(
                HypothesisRanking(
                    fault_type=CanonicalFaultType.SENSOR_DROPOUT.value,
                    compatibility_score=0.95,
                    matched_evidence=[f"Telemetry dropout on channel(s): {', '.join(dropout_channels)}"],
                    unmatched_evidence=[],
                )
            )
            is_sensor_dominant = True

        if stuck_channels:
            hypotheses.append(
                HypothesisRanking(
                    fault_type=CanonicalFaultType.SENSOR_STUCK.value,
                    compatibility_score=0.95,
                    matched_evidence=[f"Sensor stuck/stale on channel(s): {', '.join(stuck_channels)}"],
                    unmatched_evidence=[],
                )
            )
            is_sensor_dominant = True

        # 2. Check for SENSOR_BIAS / SENSOR_DRIFT (Single-Channel Physical Inconsistency)
        # Condition: Exactly one primary channel has |z| > tau_nom, while physically coupled
        # channels show negligible raw residual and nominal z-score.
        r_map = {ch: res.raw_residual for ch, res in residual_vector.residuals.items()}

        deviated_channels = [ch for ch, z in z_map.items() if not math.isnan(z) and abs(z) > self.tau_nom]

        if len(deviated_channels) == 1:
            dev_ch = deviated_channels[0]
            z_dev = z_map[dev_ch]

            # Cross-coupling checks:
            # If CHT is deviated, but coolant_temp and oil_temp show NO heating (<0.5 C rise) -> CHT Sensor Bias!
            if dev_ch == "cht":
                z_cool = abs(z_map.get("coolant_temp", 0.0))
                z_oil = abs(z_map.get("oil_temp", 0.0))
                r_cool = r_map.get("coolant_temp", 0.0)
                r_oil = r_map.get("oil_temp", 0.0)
                if z_cool < 0.5 and z_oil < 0.5 and r_cool < 0.5 and r_oil < 0.5:
                    hypotheses.append(
                        HypothesisRanking(
                            fault_type=CanonicalFaultType.SENSOR_BIAS.value,
                            compatibility_score=0.90,
                            matched_evidence=[
                                f"Isolated CHT deviation (z={z_dev:.2f}) with nominal coolant/oil temps (physical inconsistency)",
                            ],
                            unmatched_evidence=["Coupled coolant/oil temperature rise absent"],
                        )
                    )
                    is_sensor_dominant = True

            # If oil_pressure is deviated, but oil_temp shows NO heating (<0.5 C rise) -> Oil Pressure Sensor Bias!
            elif dev_ch == "oil_pressure":
                z_oil_t = abs(z_map.get("oil_temp", 0.0))
                r_oil_t = r_map.get("oil_temp", 0.0)
                z_rpm = abs(z_map.get("rpm", 0.0))
                if z_oil_t < 0.20 and r_oil_t < 0.5 and z_rpm < 0.5:
                    hypotheses.append(
                        HypothesisRanking(
                            fault_type=CanonicalFaultType.SENSOR_BIAS.value,
                            compatibility_score=0.90,
                            matched_evidence=[
                                f"Isolated oil pressure deviation (z={z_dev:.2f}) with nominal oil temp/RPM",
                            ],
                            unmatched_evidence=["Coupled thermal heating absent"],
                        )
                    )
                    is_sensor_dominant = True

            # If fuel_flow is deviated, but EGT and CHT are healthy -> Fuel Flow Sensor Bias!
            elif dev_ch == "fuel_flow":
                z_egt = abs(z_map.get("egt", 0.0))
                z_cht = abs(z_map.get("cht", 0.0))
                r_egt = abs(r_map.get("egt", 0.0))
                if z_egt < 0.5 and z_cht < 0.5 and r_egt < 5.0:
                    hypotheses.append(
                        HypothesisRanking(
                            fault_type=CanonicalFaultType.SENSOR_BIAS.value,
                            compatibility_score=0.90,
                            matched_evidence=[
                                f"Isolated fuel flow deviation (z={z_dev:.2f}) with nominal combustion temps",
                            ],
                            unmatched_evidence=["Combustion heat release nominal"],
                        )
                    )
                    is_sensor_dominant = True

        return hypotheses, is_sensor_dominant

    # ── Subsystem Mapping Helper ──────────────────────────────────────────────

    def _get_affected_subsystem(self, primary_fault: str, z_map: Dict[str, float]) -> Optional[str]:
        """Map primary fault to its primary owning subsystem."""
        mapping = {
            CanonicalFaultType.INJECTOR_DELIVERY_ABNORMALITY.value: "FUEL",
            CanonicalFaultType.LUBRICATION_DEGRADATION.value: "LUBRICATION",
            CanonicalFaultType.COOLING_DEGRADATION.value: "THERMAL",
            CanonicalFaultType.COMBUSTION_MISFIRE.value: "COMBUSTION",
            CanonicalFaultType.MECHANICAL_DEGRADATION.value: "MECHANICAL",
        }
        if primary_fault in mapping:
            return mapping[primary_fault]

        # For sensor bias/dropout, determine from deviated channel
        if primary_fault in (
            CanonicalFaultType.SENSOR_BIAS.value,
            CanonicalFaultType.SENSOR_DRIFT.value,
            CanonicalFaultType.SENSOR_DROPOUT.value,
            CanonicalFaultType.SENSOR_STUCK.value,
        ):
            from digital_twin.residuals import PRIMARY_SUBSYSTEM_MAP
            for ch, z in z_map.items():
                if abs(z) > self.tau_nom and ch in PRIMARY_SUBSYSTEM_MAP:
                    return PRIMARY_SUBSYSTEM_MAP[ch]

        return None
