"""
Deterministic rule-based physics consistency evaluation for Phase 12.
Evaluates concordance between diagnosed fault and physical residual signatures.
NOTE: Physics consistency is supporting engineering evidence, not proof of fault causation.
"""

from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np

from explainability.schema import EvidenceStatus, PhysicsEvidence


class PhysicsEvidenceEvaluator:
    """
    Evaluates physical consistency of Phase 8 fault diagnosis against domain physics relationships.
    Uses Phase 9 deadband (tau_nominal = 1.5 sigma) for elevation/depression classification.
    """

    def __init__(self, tau_threshold: float = 1.5):
        self.tau_threshold = tau_threshold

    def _classify_direction(self, val: Optional[float]) -> str:
        """Classify residual value into ELEVATED, DEPRESSED, NOMINAL, or MISSING."""
        if val is None or not np.isfinite(val):
            return "MISSING"
        if val > self.tau_threshold:
            return "ELEVATED"
        elif val < -self.tau_threshold:
            return "DEPRESSED"
        else:
            return "NOMINAL"

    def evaluate(
        self,
        diagnosed_fault: Optional[str],
        residuals: Dict[str, Optional[float]],
        excluded_channels: Optional[Set[str]] = None,
    ) -> PhysicsEvidence:
        """
        Evaluate physical consistency for a diagnosed condition.

        Args:
            diagnosed_fault: Diagnosed fault type string (e.g. "cooling_degradation")
            residuals: Dictionary of normalized residuals per channel
            excluded_channels: Channels isolated by Phase 9 as sensor faults

        Returns:
            PhysicsEvidence structure
        """
        excluded = set(excluded_channels or [])
        fault = (diagnosed_fault or "none").lower()

        # Check data sufficiency: at least 4 valid numeric residuals
        valid_channels = [k for k, v in residuals.items() if v is not None and np.isfinite(v)]
        if len(valid_channels) < 4:
            return PhysicsEvidence(
                status=EvidenceStatus.INSUFFICIENT_DATA,
                diagnosed_fault=fault,
                evidence_channels=[],
                observed_residual_directions={},
                expected_residual_directions={},
                consistency_reason="Insufficient valid physical residual channels (< 4 valid) to evaluate physical consistency.",
                supporting_channels=[],
                conflicting_channels=[],
                missing_channels=[k for k, v in residuals.items() if v is None or not np.isfinite(v)],
                provenance={"valid_count": len(valid_channels), "threshold_sigma": self.tau_threshold},
            )

        # Classify observed directions
        obs_dirs = {ch: self._classify_direction(residuals.get(ch)) for ch in residuals}

        # -----------------------------------------------------------------
        # FAULT 1: COOLING_DEGRADATION
        # Expected: CHT elevated, oil_temp elevated (coupled thermal conduction)
        # -----------------------------------------------------------------
        if fault == "cooling_degradation":
            evidence_channels = ["cht", "oil_temp"]
            expected = {"cht": "ELEVATED", "oil_temp": "ELEVATED"}
            cht_dir = obs_dirs.get("cht", "MISSING")
            ot_dir = obs_dirs.get("oil_temp", "MISSING")

            # Check if CHT is isolated as a sensor fault
            if "cht" in excluded:
                return PhysicsEvidence(
                    status=EvidenceStatus.CONFLICTING,
                    diagnosed_fault=fault,
                    evidence_channels=evidence_channels,
                    observed_residual_directions={k: obs_dirs.get(k, "MISSING") for k in evidence_channels},
                    expected_residual_directions=expected,
                    consistency_reason="Primary thermal channel (CHT) is isolated as an observation sensor fault and cannot support cooling degradation.",
                    supporting_channels=[],
                    conflicting_channels=["cht"],
                    missing_channels=[],
                    provenance={"isolated_sensor": "cht"},
                )

            if cht_dir == "ELEVATED":
                if ot_dir == "ELEVATED":
                    return PhysicsEvidence(
                        status=EvidenceStatus.SUPPORTED,
                        diagnosed_fault=fault,
                        evidence_channels=evidence_channels,
                        observed_residual_directions={k: obs_dirs.get(k, "MISSING") for k in evidence_channels},
                        expected_residual_directions=expected,
                        consistency_reason="Observed elevated CHT and coupled oil temperature residuals are consistent with the project's cooling-degradation thermal relationship.",
                        supporting_channels=["cht", "oil_temp"],
                        conflicting_channels=[],
                        missing_channels=[],
                    )
                elif ot_dir in {"NOMINAL", "MISSING"}:
                    return PhysicsEvidence(
                        status=EvidenceStatus.PARTIALLY_SUPPORTED,
                        diagnosed_fault=fault,
                        evidence_channels=evidence_channels,
                        observed_residual_directions={k: obs_dirs.get(k, "MISSING") for k in evidence_channels},
                        expected_residual_directions=expected,
                        consistency_reason="Elevated CHT residual provides supporting evidence, but secondary oil temperature residual has not yet responded (delayed conduction coupling).",
                        supporting_channels=["cht"],
                        conflicting_channels=[],
                        missing_channels=["oil_temp"] if ot_dir == "MISSING" else [],
                    )
                else:  # ot_dir == "DEPRESSED"
                    return PhysicsEvidence(
                        status=EvidenceStatus.CONFLICTING,
                        diagnosed_fault=fault,
                        evidence_channels=evidence_channels,
                        observed_residual_directions={k: obs_dirs.get(k, "MISSING") for k in evidence_channels},
                        expected_residual_directions=expected,
                        consistency_reason="Elevated CHT residual conflicts with depressed oil temperature residual.",
                        supporting_channels=["cht"],
                        conflicting_channels=["oil_temp"],
                        missing_channels=[],
                    )
            elif cht_dir == "DEPRESSED":
                return PhysicsEvidence(
                    status=EvidenceStatus.CONFLICTING,
                    diagnosed_fault=fault,
                    evidence_channels=evidence_channels,
                    observed_residual_directions={k: obs_dirs.get(k, "MISSING") for k in evidence_channels},
                    expected_residual_directions=expected,
                    consistency_reason="Depressed CHT residual contradicts expected thermal elevation for cooling degradation.",
                    supporting_channels=[],
                    conflicting_channels=["cht"],
                    missing_channels=[],
                )
            else:  # cht_dir == "NOMINAL"
                return PhysicsEvidence(
                    status=EvidenceStatus.CONFLICTING,
                    diagnosed_fault=fault,
                    evidence_channels=evidence_channels,
                    observed_residual_directions={k: obs_dirs.get(k, "MISSING") for k in evidence_channels},
                    expected_residual_directions=expected,
                    consistency_reason="CHT residual remains within nominal band, lacking supporting evidence for cooling degradation.",
                    supporting_channels=[],
                    conflicting_channels=["cht"],
                    missing_channels=[],
                )

        # -----------------------------------------------------------------
        # FAULT 2: LUBRICATION_DEGRADATION
        # Expected: oil_pressure depressed, oil_temp elevated
        # -----------------------------------------------------------------
        elif fault == "lubrication_degradation":
            evidence_channels = ["oil_pressure", "oil_temp"]
            expected = {"oil_pressure": "DEPRESSED", "oil_temp": "ELEVATED"}
            op_dir = obs_dirs.get("oil_pressure", "MISSING")
            ot_dir = obs_dirs.get("oil_temp", "MISSING")

            if "oil_pressure" in excluded:
                return PhysicsEvidence(
                    status=EvidenceStatus.CONFLICTING,
                    diagnosed_fault=fault,
                    evidence_channels=evidence_channels,
                    observed_residual_directions={k: obs_dirs.get(k, "MISSING") for k in evidence_channels},
                    expected_residual_directions=expected,
                    consistency_reason="Primary lubrication channel (oil_pressure) is isolated as an observation sensor fault and cannot support lubrication degradation.",
                    supporting_channels=[],
                    conflicting_channels=["oil_pressure"],
                    missing_channels=[],
                    provenance={"isolated_sensor": "oil_pressure"},
                )

            if op_dir == "DEPRESSED":
                if ot_dir == "ELEVATED":
                    return PhysicsEvidence(
                        status=EvidenceStatus.SUPPORTED,
                        diagnosed_fault=fault,
                        evidence_channels=evidence_channels,
                        observed_residual_directions={k: obs_dirs.get(k, "MISSING") for k in evidence_channels},
                        expected_residual_directions=expected,
                        consistency_reason="Observed drop in oil pressure residual coupled with elevated oil temperature residual is consistent with the project's lubrication degradation relationship.",
                        supporting_channels=["oil_pressure", "oil_temp"],
                        conflicting_channels=[],
                        missing_channels=[],
                    )
                elif ot_dir in {"NOMINAL", "MISSING"}:
                    return PhysicsEvidence(
                        status=EvidenceStatus.PARTIALLY_SUPPORTED,
                        diagnosed_fault=fault,
                        evidence_channels=evidence_channels,
                        observed_residual_directions={k: obs_dirs.get(k, "MISSING") for k in evidence_channels},
                        expected_residual_directions=expected,
                        consistency_reason="Depressed oil pressure residual provides primary supporting evidence, while oil temperature residual remains in nominal deadband.",
                        supporting_channels=["oil_pressure"],
                        conflicting_channels=[],
                        missing_channels=["oil_temp"] if ot_dir == "MISSING" else [],
                    )
                else:  # ot_dir == "DEPRESSED"
                    return PhysicsEvidence(
                        status=EvidenceStatus.CONFLICTING,
                        diagnosed_fault=fault,
                        evidence_channels=evidence_channels,
                        observed_residual_directions={k: obs_dirs.get(k, "MISSING") for k in evidence_channels},
                        expected_residual_directions=expected,
                        consistency_reason="Depressed oil pressure with depressed oil temperature conflicts with frictional heating expectations.",
                        supporting_channels=["oil_pressure"],
                        conflicting_channels=["oil_temp"],
                        missing_channels=[],
                    )
            elif op_dir == "ELEVATED":
                return PhysicsEvidence(
                    status=EvidenceStatus.CONFLICTING,
                    diagnosed_fault=fault,
                    evidence_channels=evidence_channels,
                    observed_residual_directions={k: obs_dirs.get(k, "MISSING") for k in evidence_channels},
                    expected_residual_directions=expected,
                    consistency_reason="Elevated oil pressure residual contradicts expected hydrodynamic pressure drop for lubrication degradation.",
                    supporting_channels=[],
                    conflicting_channels=["oil_pressure"],
                    missing_channels=[],
                )
            else:  # op_dir == "NOMINAL"
                return PhysicsEvidence(
                    status=EvidenceStatus.CONFLICTING,
                    diagnosed_fault=fault,
                    evidence_channels=evidence_channels,
                    observed_residual_directions={k: obs_dirs.get(k, "MISSING") for k in evidence_channels},
                    expected_residual_directions=expected,
                    consistency_reason="Oil pressure residual remains within nominal deadband, conflicting with active lubrication degradation.",
                    supporting_channels=[],
                    conflicting_channels=["oil_pressure"],
                    missing_channels=[],
                )

        # -----------------------------------------------------------------
        # FAULT 3: FUEL_INJECTION_ABNORMALITY
        # Lean branch: EGT elevated, fuel_flow depressed
        # Rich branch: EGT depressed, fuel_flow elevated
        # -----------------------------------------------------------------
        elif fault == "fuel_injection_abnormality":
            evidence_channels = ["egt", "fuel_flow"]
            egt_dir = obs_dirs.get("egt", "MISSING")
            ff_dir = obs_dirs.get("fuel_flow", "MISSING")

            # Check Lean branch
            if egt_dir == "ELEVATED" and ff_dir == "DEPRESSED":
                return PhysicsEvidence(
                    status=EvidenceStatus.SUPPORTED,
                    diagnosed_fault=fault,
                    evidence_channels=evidence_channels,
                    observed_residual_directions={k: obs_dirs.get(k, "MISSING") for k in evidence_channels},
                    expected_residual_directions={"egt": "ELEVATED", "fuel_flow": "DEPRESSED"},
                    consistency_reason="Elevated EGT residual combined with depressed fuel flow residual is consistent with the project's lean mixture abnormality relationship.",
                    supporting_channels=["egt", "fuel_flow"],
                    conflicting_channels=[],
                    missing_channels=[],
                    provenance={"mode": "lean_mixture"},
                )
            # Check Rich branch
            elif egt_dir == "DEPRESSED" and ff_dir == "ELEVATED":
                return PhysicsEvidence(
                    status=EvidenceStatus.SUPPORTED,
                    diagnosed_fault=fault,
                    evidence_channels=evidence_channels,
                    observed_residual_directions={k: obs_dirs.get(k, "MISSING") for k in evidence_channels},
                    expected_residual_directions={"egt": "DEPRESSED", "fuel_flow": "ELEVATED"},
                    consistency_reason="Depressed EGT residual combined with elevated fuel flow residual is consistent with the project's rich mixture abnormality relationship.",
                    supporting_channels=["egt", "fuel_flow"],
                    conflicting_channels=[],
                    missing_channels=[],
                    provenance={"mode": "rich_mixture"},
                )
            # Check partial support
            elif (egt_dir == "ELEVATED" and ff_dir in {"NOMINAL", "MISSING"}) or (ff_dir == "DEPRESSED" and egt_dir in {"NOMINAL", "MISSING"}):
                return PhysicsEvidence(
                    status=EvidenceStatus.PARTIALLY_SUPPORTED,
                    diagnosed_fault=fault,
                    evidence_channels=evidence_channels,
                    observed_residual_directions={k: obs_dirs.get(k, "MISSING") for k in evidence_channels},
                    expected_residual_directions={"egt": "ELEVATED", "fuel_flow": "DEPRESSED"},
                    consistency_reason="One combustion residual is deviating toward lean condition while coupled channel remains in nominal band.",
                    supporting_channels=["egt"] if egt_dir == "ELEVATED" else ["fuel_flow"],
                    conflicting_channels=[],
                    missing_channels=[ch for ch in evidence_channels if obs_dirs.get(ch) == "MISSING"],
                    provenance={"mode": "partial_lean"},
                )
            elif (egt_dir == "DEPRESSED" and ff_dir in {"NOMINAL", "MISSING"}) or (ff_dir == "ELEVATED" and egt_dir in {"NOMINAL", "MISSING"}):
                return PhysicsEvidence(
                    status=EvidenceStatus.PARTIALLY_SUPPORTED,
                    diagnosed_fault=fault,
                    evidence_channels=evidence_channels,
                    observed_residual_directions={k: obs_dirs.get(k, "MISSING") for k in evidence_channels},
                    expected_residual_directions={"egt": "DEPRESSED", "fuel_flow": "ELEVATED"},
                    consistency_reason="One combustion residual is deviating toward rich condition while coupled channel remains in nominal band.",
                    supporting_channels=["egt"] if egt_dir == "DEPRESSED" else ["fuel_flow"],
                    conflicting_channels=[],
                    missing_channels=[ch for ch in evidence_channels if obs_dirs.get(ch) == "MISSING"],
                    provenance={"mode": "partial_rich"},
                )
            # Same direction (both ELEVATED or both DEPRESSED) contradicts combustion stoichiometry
            elif egt_dir in {"ELEVATED", "DEPRESSED"} and ff_dir == egt_dir:
                return PhysicsEvidence(
                    status=EvidenceStatus.CONFLICTING,
                    diagnosed_fault=fault,
                    evidence_channels=evidence_channels,
                    observed_residual_directions={k: obs_dirs.get(k, "MISSING") for k in evidence_channels},
                    expected_residual_directions={"egt": "OPPOSITE_TO_FUEL_FLOW", "fuel_flow": "OPPOSITE_TO_EGT"},
                    consistency_reason="Observed residual direction (both EGT and fuel flow deviating in same direction) contradicts coupled combustion stoichiometry.",
                    supporting_channels=[],
                    conflicting_channels=["egt", "fuel_flow"],
                    missing_channels=[],
                )
            else:  # Both NOMINAL
                return PhysicsEvidence(
                    status=EvidenceStatus.CONFLICTING,
                    diagnosed_fault=fault,
                    evidence_channels=evidence_channels,
                    observed_residual_directions={k: obs_dirs.get(k, "MISSING") for k in evidence_channels},
                    expected_residual_directions={"egt": "ELEVATED_OR_DEPRESSED", "fuel_flow": "OPPOSITE"},
                    consistency_reason="Both EGT and fuel flow residuals remain in nominal band, lacking combustion abnormality evidence.",
                    supporting_channels=[],
                    conflicting_channels=["egt", "fuel_flow"],
                    missing_channels=[],
                )

        # -----------------------------------------------------------------
        # FAULT 4: MECHANICAL_DEGRADATION
        # Expected: vibration elevated
        # -----------------------------------------------------------------
        elif fault == "mechanical_degradation":
            evidence_channels = ["vibration"]
            expected = {"vibration": "ELEVATED"}
            vib_dir = obs_dirs.get("vibration", "MISSING")

            if vib_dir == "ELEVATED":
                return PhysicsEvidence(
                    status=EvidenceStatus.SUPPORTED,
                    diagnosed_fault=fault,
                    evidence_channels=evidence_channels,
                    observed_residual_directions={"vibration": vib_dir},
                    expected_residual_directions=expected,
                    consistency_reason="Elevated structural vibration residual is consistent with mechanical degradation and structural unbalance.",
                    supporting_channels=["vibration"],
                    conflicting_channels=[],
                    missing_channels=[],
                )
            elif vib_dir == "NOMINAL":
                return PhysicsEvidence(
                    status=EvidenceStatus.CONFLICTING,
                    diagnosed_fault=fault,
                    evidence_channels=evidence_channels,
                    observed_residual_directions={"vibration": vib_dir},
                    expected_residual_directions=expected,
                    consistency_reason="Structural vibration residual remains in nominal band, lacking supporting evidence for mechanical degradation.",
                    supporting_channels=[],
                    conflicting_channels=["vibration"],
                    missing_channels=[],
                )
            else:
                return PhysicsEvidence(
                    status=EvidenceStatus.CONFLICTING,
                    diagnosed_fault=fault,
                    evidence_channels=evidence_channels,
                    observed_residual_directions={"vibration": vib_dir},
                    expected_residual_directions=expected,
                    consistency_reason="Depressed structural vibration residual contradicts mechanical unbalance expectations.",
                    supporting_channels=[],
                    conflicting_channels=["vibration"],
                    missing_channels=[],
                )

        # -----------------------------------------------------------------
        # FAULT 5: SENSOR_FAULT
        # Expected: Isolated observation anomaly without multi-subsystem physical response
        # -----------------------------------------------------------------
        elif fault == "sensor_fault":
            evidence_channels = list(residuals.keys())
            non_nominal = [ch for ch, d in obs_dirs.items() if d in {"ELEVATED", "DEPRESSED"}]

            if len(excluded) > 0:
                iso_ch = list(excluded)[0]
                return PhysicsEvidence(
                    status=EvidenceStatus.SUPPORTED,
                    diagnosed_fault=fault,
                    evidence_channels=[iso_ch],
                    observed_residual_directions={iso_ch: obs_dirs.get(iso_ch, "MISSING")},
                    expected_residual_directions={iso_ch: "ISOLATED_ANOMALY"},
                    consistency_reason=f"Sensor channel '{iso_ch}' is isolated by Phase 9 heuristic as an observation anomaly without coupled physical divergence.",
                    supporting_channels=[iso_ch],
                    conflicting_channels=[],
                    missing_channels=[],
                    provenance={"isolated_sensor": iso_ch},
                )
            elif len(non_nominal) == 1:
                anom_ch = non_nominal[0]
                return PhysicsEvidence(
                    status=EvidenceStatus.SUPPORTED,
                    diagnosed_fault=fault,
                    evidence_channels=[anom_ch],
                    observed_residual_directions={anom_ch: obs_dirs.get(anom_ch, "MISSING")},
                    expected_residual_directions={anom_ch: "ISOLATED_ANOMALY"},
                    consistency_reason=f"Single isolated residual divergence observed on '{anom_ch}' with all other physical channels nominal.",
                    supporting_channels=[anom_ch],
                    conflicting_channels=[],
                    missing_channels=[],
                )
            elif len(non_nominal) > 2:
                return PhysicsEvidence(
                    status=EvidenceStatus.CONFLICTING,
                    diagnosed_fault=fault,
                    evidence_channels=non_nominal,
                    observed_residual_directions={ch: obs_dirs.get(ch, "MISSING") for ch in non_nominal},
                    expected_residual_directions={ch: "NOMINAL_EXCEPT_ONE" for ch in non_nominal},
                    consistency_reason="Multi-channel physical residual divergence is consistent with physical engine degradation rather than single-sensor failure.",
                    supporting_channels=[],
                    conflicting_channels=non_nominal,
                    missing_channels=[],
                )
            else:
                return PhysicsEvidence(
                    status=EvidenceStatus.PARTIALLY_SUPPORTED,
                    diagnosed_fault=fault,
                    evidence_channels=non_nominal,
                    observed_residual_directions={ch: obs_dirs.get(ch, "MISSING") for ch in non_nominal},
                    expected_residual_directions={},
                    consistency_reason="Isolated sensor anomaly criteria partially met.",
                    supporting_channels=non_nominal,
                    conflicting_channels=[],
                    missing_channels=[],
                )

        # -----------------------------------------------------------------
        # FAULT 6: NONE (Healthy Baseline)
        # Expected: All channels nominal
        # -----------------------------------------------------------------
        else:
            evidence_channels = list(residuals.keys())
            non_nominal = [ch for ch, d in obs_dirs.items() if d in {"ELEVATED", "DEPRESSED"}]

            if len(non_nominal) == 0:
                return PhysicsEvidence(
                    status=EvidenceStatus.SUPPORTED,
                    diagnosed_fault="none",
                    evidence_channels=evidence_channels,
                    observed_residual_directions={ch: obs_dirs.get(ch, "MISSING") for ch in evidence_channels},
                    expected_residual_directions={ch: "NOMINAL" for ch in evidence_channels},
                    consistency_reason="All observed physical residuals remain within the healthy nominal operating envelope.",
                    supporting_channels=evidence_channels,
                    conflicting_channels=[],
                    missing_channels=[],
                )
            elif len(non_nominal) == 1:
                return PhysicsEvidence(
                    status=EvidenceStatus.PARTIALLY_SUPPORTED,
                    diagnosed_fault="none",
                    evidence_channels=evidence_channels,
                    observed_residual_directions={ch: obs_dirs.get(ch, "MISSING") for ch in evidence_channels},
                    expected_residual_directions={ch: "NOMINAL" for ch in evidence_channels},
                    consistency_reason="Minor single-channel deviation observed, but overall system state remains predominantly nominal.",
                    supporting_channels=[ch for ch in evidence_channels if ch not in non_nominal],
                    conflicting_channels=non_nominal,
                    missing_channels=[],
                )
            else:
                return PhysicsEvidence(
                    status=EvidenceStatus.CONFLICTING,
                    diagnosed_fault="none",
                    evidence_channels=evidence_channels,
                    observed_residual_directions={ch: obs_dirs.get(ch, "MISSING") for ch in evidence_channels},
                    expected_residual_directions={ch: "NOMINAL" for ch in evidence_channels},
                    consistency_reason="Significant multi-channel residual divergence conflicts with healthy nominal engine state.",
                    supporting_channels=[],
                    conflicting_channels=non_nominal,
                    missing_channels=[],
                )
