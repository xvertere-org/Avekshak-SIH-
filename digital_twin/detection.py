"""
Phase 6 Temporal Anomaly Detection Layer for Rotax 914 UL/F Digital Twin.

Provides generic, operating-point-aware temporal anomaly detection based on
Phase 5 physical residuals, coverage gating, and health indicators.

Invariants:
- Separation of Concerns: Zero knowledge of fault taxonomy (F1-F7) or fault modes.
- Operating-point aware: Consumes normalized residuals (z = r / sigma) and Phase 5 HI_raw.
- Temporal persistence: Sustained abnormal residual duration >= 3.0s required for ANOMALOUS.
- Recovery semantics: Sustained nominal duration >= 5.0s required to return from RECOVERED to NORMAL.
- Coverage gating: Insufficient primary channels (< 5 of 9) immediately yields INSUFFICIENT_DATA.
- Non-circular: Emits DetectionResult downstream; never feeds back into estimator or prediction.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, Any, List, Optional

from digital_twin.residuals import ResidualVector
from digital_twin.health import ModelObservationHealthAssessment, HealthState


class DetectionStatus(str, Enum):
    """
    Standard anomaly detection states for Phase 6.
    """
    NORMAL = "NORMAL"
    SUSPECTED = "SUSPECTED"
    ANOMALOUS = "ANOMALOUS"
    RECOVERED = "RECOVERED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass
class DetectionConfig:
    """
    Configuration parameters for temporal anomaly detection.
    Explicitly tags all heuristic thresholds with ENGINEERING_HEURISTIC provenance.
    """
    anomaly_threshold: float = 0.018    # Anomaly active when S_anom >= 0.018 (HI_raw <= 0.982)
    persistence_seconds: float = 3.0    # Duration required to confirm ANOMALOUS from SUSPECTED
    recovery_seconds: float = 5.0       # Duration of nominal conditions required to clear RECOVERED to NORMAL
    min_valid_channels: int = 5         # Minimum valid primary channels (5/9 coverage gate)
    tau_nom: float = 1.5                # Nominal normalized residual threshold
    tau_crit: float = 5.0               # Critical normalized residual threshold
    provenance: str = "ENGINEERING_HEURISTIC"
    is_empirically_validated: bool = False


@dataclass
class DetectionResult:
    """
    Structured anomaly detection output for a single streaming time step.
    Contains no fault-type labels; represents pure anomaly existence and persistence.
    """
    timestamp: float
    engine_id: str
    status: DetectionStatus
    anomaly_score: float                # S_anom in [0.0, 1.0], or NaN if INSUFFICIENT_DATA
    anomaly_detected: bool              # True iff status == DetectionStatus.ANOMALOUS
    persistence_duration: float         # Continuous seconds in active abnormal condition
    recovery_duration: float            # Continuous seconds in recovery condition
    contributing_channels: List[str]    # Primary channels with |z| > tau_nom
    contributing_subsystems: List[str]  # Subsystems with degraded health
    evidence_quality: float             # C_obs * C_data in [0.0, 1.0]
    reason_codes: List[str]             # Explanatory trigger codes
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_anomalous(self) -> bool:
        return self.anomaly_detected

    @property
    def is_recovering(self) -> bool:
        return self.status == DetectionStatus.RECOVERED

    @property
    def anomaly_persistence_s(self) -> float:
        return self.persistence_duration

    def to_dict(self) -> Dict[str, Any]:
        """Serialize detection result to dictionary."""
        return {
            "timestamp": self.timestamp,
            "engine_id": self.engine_id,
            "status": self.status.value,
            "anomaly_score": round(self.anomaly_score, 4) if not math.isnan(self.anomaly_score) else None,
            "anomaly_detected": self.anomaly_detected,
            "persistence_duration": round(self.persistence_duration, 2),
            "recovery_duration": round(self.recovery_duration, 2),
            "contributing_channels": list(self.contributing_channels),
            "contributing_subsystems": list(self.contributing_subsystems),
            "evidence_quality": round(self.evidence_quality, 4),
            "reason_codes": list(self.reason_codes),
            "metadata": self.metadata,
        }


class TemporalFaultDetector:
    """
    Stateful temporal anomaly detector.

    Tracks anomaly persistence, enforces debounce and recovery transitions,
    and identifies contributing channels and subsystems without reference to fault taxonomy.
    """

    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()
        self._abnormal_duration: float = 0.0
        self._healthy_duration: float = 0.0
        self._confirmed_status: DetectionStatus = DetectionStatus.NORMAL
        self._last_timestamp: Optional[float] = None

    def reset(self) -> None:
        """Reset internal temporal persistence and state history."""
        self._abnormal_duration = 0.0
        self._healthy_duration = 0.0
        self._confirmed_status = DetectionStatus.NORMAL
        self._last_timestamp = None

    def update(self, *args, **kwargs) -> DetectionResult:
        """Alias for detect() for API consistency."""
        return self.detect(*args, **kwargs)

    def detect(
        self,
        residual_vector: ResidualVector,
        health_assessment: ModelObservationHealthAssessment,
        dt: Optional[float] = None,
        engine_id: str = "ROTAX_914_GREYBOX",
    ) -> DetectionResult:
        """
        Evaluate temporal anomaly status from Phase 5 residual vector and health assessment.

        Args:
            residual_vector: ResidualVector from QualityAwareResidualGenerator.
            health_assessment: ModelObservationHealthAssessment from HealthEvaluator.
            dt: Optional time increment in seconds. If None, inferred from timestamp.
            engine_id: Engine identifier.

        Returns:
            DetectionResult summarizing anomaly status, score, persistence, and contributing channels.
        """
        timestamp = residual_vector.timestamp
        if dt is None:
            if self._last_timestamp is not None and timestamp > self._last_timestamp:
                step_dt = timestamp - self._last_timestamp
            else:
                step_dt = 0.1
        else:
            step_dt = max(1e-4, float(dt))
        self._last_timestamp = timestamp

        evidence_quality = max(0.0, min(1.0, health_assessment.C_obs * health_assessment.C_data))

        # 1. Coverage Gate Evaluation
        if (
            residual_vector.valid_primary_count < self.config.min_valid_channels
            or health_assessment.state == HealthState.UNAVAILABLE
            or math.isnan(health_assessment.HI_raw)
        ):
            self._abnormal_duration = 0.0
            self._healthy_duration = 0.0
            self._confirmed_status = DetectionStatus.INSUFFICIENT_DATA

            return DetectionResult(
                timestamp=timestamp,
                engine_id=engine_id,
                status=DetectionStatus.INSUFFICIENT_DATA,
                anomaly_score=float("nan"),
                anomaly_detected=False,
                persistence_duration=0.0,
                recovery_duration=0.0,
                contributing_channels=[],
                contributing_subsystems=[],
                evidence_quality=evidence_quality,
                reason_codes=["COVERAGE_GATE_INSUFFICIENT_CHANNELS"],
                metadata={
                    "valid_primary_count": residual_vector.valid_primary_count,
                    "coverage_fraction": residual_vector.coverage_fraction,
                },
            )

        # 2. Compute Authoritative Engine-Level Anomaly Score: S_anom = 1.0 - HI_raw
        # In strict adherence to authorized Phase 5/6 architecture, engine-level anomaly detection
        # is governed exclusively by composite physical health HI_raw.
        # Cylinder runner spreads (cht_cyl1..4, egt_cyl1..4) are reserved strictly for the
        # diagnostic layer (cylinder localization & imbalance evidence) and do NOT inflate
        # engine-level anomaly votes or drive the temporal persistence timer.
        hi_raw = health_assessment.HI_raw
        s_anom = max(0.0, min(1.0, 1.0 - hi_raw))

        # 3. Identify Contributing Channels (|z| > tau_nom) and Subsystems
        contributing_channels: List[str] = []
        reason_codes: List[str] = []

        for ch_name, ind in health_assessment.channel_indicators.items():
            if ind.valid and ind.is_primary:
                if ind.z_score > self.config.tau_nom:
                    contributing_channels.append(ch_name)
                    reason_codes.append(f"CHANNEL_DEVIATION:{ch_name}(|z|={ind.z_score:.2f})")

        contributing_subsystems: List[str] = []
        for sub_name, sub_ass in health_assessment.subsystems.items():
            if not math.isnan(sub_ass.score) and sub_ass.score < (1.0 - self.config.anomaly_threshold):
                contributing_subsystems.append(sub_name)
                reason_codes.append(f"SUBSYSTEM_DEGRADATION:{sub_name}(score={sub_ass.score:.2f})")

        # 4. Temporal State Machine
        # Mandatory correction 1: Explicit anomaly condition S_anom >= threshold (0.15)
        is_anomaly_active = s_anom >= self.config.anomaly_threshold

        if is_anomaly_active:
            self._healthy_duration = 0.0
            self._abnormal_duration += step_dt

            if self._abnormal_duration >= self.config.persistence_seconds:
                current_status = DetectionStatus.ANOMALOUS
            else:
                current_status = DetectionStatus.SUSPECTED
        else:
            self._abnormal_duration = 0.0
            self._healthy_duration += step_dt

            # Mandatory correction 4: Recovery semantics ANOMALOUS -> RECOVERED -> NORMAL
            if self._confirmed_status in (
                DetectionStatus.ANOMALOUS,
                DetectionStatus.RECOVERED,
            ):
                if self._healthy_duration >= self.config.recovery_seconds:
                    current_status = DetectionStatus.NORMAL
                else:
                    current_status = DetectionStatus.RECOVERED
                    reason_codes.append("RECOVERY_IN_PROGRESS")
            else:
                current_status = DetectionStatus.NORMAL

        self._confirmed_status = current_status
        anomaly_detected = current_status == DetectionStatus.ANOMALOUS

        return DetectionResult(
            timestamp=timestamp,
            engine_id=engine_id,
            status=current_status,
            anomaly_score=s_anom,
            anomaly_detected=anomaly_detected,
            persistence_duration=self._abnormal_duration,
            recovery_duration=self._healthy_duration,
            contributing_channels=contributing_channels,
            contributing_subsystems=contributing_subsystems,
            evidence_quality=evidence_quality,
            reason_codes=reason_codes,
            metadata={
                "HI_raw": round(hi_raw, 4),
                "HI_smooth": round(health_assessment.HI_smooth, 4),
                "threshold": self.config.anomaly_threshold,
                "persistence_target": self.config.persistence_seconds,
                "recovery_target": self.config.recovery_seconds,
            },
        )
