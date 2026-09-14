"""
Physics-Based Health Assessment and Health Indicator Layer for SIH26054 Digital Twin.

Evaluates engine physical health purely from normalized residuals, observing strict
causal dependency and architectural non-circularity:
Telemetry -> Quality/Observability -> Synchronized Twin Prediction -> Residuals -> Health Assessment

Key Architectural Rules:
- No feedback from Health Assessment into Twin state estimation or prediction.
- Primary Subsystem Ownership: exactly 9 primary engine channels mapped to 6 primary subsystems;
  no double-counting of physical channels.
- Secondary channels (e.g. charge_air_temp) and cylinder-level imbalances are reported as
  diagnostic/localization context with 0 weight in engine-level HI.
- Strict Independence: HI_raw (physics consistency), C_obs (observability coverage),
  and C_data (telemetry quality / state confidence) are strictly separated.
  Missing telemetry drops C_obs; it does NOT fabricate physical engine degradation in HI_raw.
- Coverage Gate: minimum 5 valid primary channels (C_obs >= 5/9). Below 5 valid channels,
  state is UNAVAILABLE, HI_raw is NaN.
- Causal Persistence & Smoothing: channel/subsystem health states require physical time persistence;
  HI_smooth uses strictly causal EWMA with no lookahead.
- Disclaimer: Research prototype implementation. Not certified for flight operations or airworthiness.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Union
import math

from digital_twin.residuals import (
    PRIMARY_RESIDUAL_CHANNELS,
    CYLINDER_RESIDUAL_CHANNELS,
    SECONDARY_MODEL_CHANNELS,
    PRIMARY_SUBSYSTEM_MAP,
    SECONDARY_CONTEXT_MAP,
    CHANNEL_UNITS_MAP,
    PhysicalResidual,
    CylinderResiduals,
    ResidualVector,
)


class HealthState(str, Enum):
    """
    Categorical health state classification.
    """
    HEALTHY = "HEALTHY"
    WATCH = "WATCH"
    DEGRADED = "DEGRADED"
    CRITICAL = "CRITICAL"
    UNAVAILABLE = "UNAVAILABLE"


# Default primary subsystem weight allocation (Sums to 1.0)
DEFAULT_SUBSYSTEM_WEIGHTS: Dict[str, float] = {
    "THERMAL": 0.25,
    "LUBRICATION": 0.20,
    "FUEL": 0.20,
    "COMBUSTION": 0.15,
    "MECHANICAL": 0.10,
    "ROTATIONAL": 0.10,
}


@dataclass
class HealthIndicatorConfig:
    """
    Configuration and heuristic thresholds for physics-based health assessment.
    Explicitly tags all heuristic parameters with ENGINEERING_HEURISTIC provenance.
    """
    tau_nom: float = 1.5                # Nominal normalized residual threshold (|z| <= 1.5 is healthy)
    tau_crit: float = 5.0               # Critical normalized residual threshold (|z| >= 5.0 is critical)
    persistence_seconds: float = 3.0    # Time persistence required to confirm abnormal health state
    recovery_seconds: float = 5.0       # Time persistence required to clear abnormal state back to healthy
    ewma_alpha: float = 0.15            # Causal exponential smoothing factor
    min_primary_channels: int = 5       # Minimum valid primary channels required for HI_raw (5/9 coverage)
    primary_subsystem_weights: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SUBSYSTEM_WEIGHTS))
    provenance: str = "ENGINEERING_HEURISTIC"
    is_empirically_validated: bool = False


@dataclass
class ChannelHealthIndicator:
    """
    Individual channel health and residual severity metric.
    """
    channel: str
    primary_subsystem: str
    raw_residual: float
    normalized_residual: float
    z_score: float                      # abs(normalized_residual)
    channel_score: float                # 1.0 (nominal) down to 0.0 (critical penalty)
    state: HealthState
    is_primary: bool
    valid: bool
    units: str
    observed: Optional[float] = None
    predicted: Optional[float] = None
    rejection_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel,
            "primary_subsystem": self.primary_subsystem,
            "raw_residual": round(self.raw_residual, 4) if not math.isnan(self.raw_residual) else None,
            "normalized_residual": round(self.normalized_residual, 4) if not math.isnan(self.normalized_residual) else None,
            "z_score": round(self.z_score, 4) if not math.isnan(self.z_score) else None,
            "channel_score": round(self.channel_score, 4) if not math.isnan(self.channel_score) else None,
            "state": self.state.value,
            "is_primary": self.is_primary,
            "valid": self.valid,
            "units": self.units,
            "observed": self.observed,
            "predicted": self.predicted,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class SubsystemHealthAssessment:
    """
    Health assessment for a single physical subsystem.
    Calculated exclusively as the arithmetic mean of primary channels owned by this subsystem.
    """
    subsystem: str
    score: float                        # Arithmetic mean of valid primary channels in [0.0, 1.0] (or NaN)
    state: HealthState
    primary_channels: List[str]
    valid_channels: List[str]
    channel_scores: Dict[str, float]
    secondary_evidence: Dict[str, Any] = field(default_factory=dict)
    worst_channel_score: Optional[float] = None  # Explicitly named separate diagnostic metric (does NOT affect score)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subsystem": self.subsystem,
            "score": round(self.score, 4) if not math.isnan(self.score) else None,
            "state": self.state.value,
            "primary_channels": self.primary_channels,
            "valid_channels": self.valid_channels,
            "channel_scores": {k: round(v, 4) for k, v in self.channel_scores.items()},
            "secondary_evidence": self.secondary_evidence,
            "worst_channel_score": round(self.worst_channel_score, 4) if self.worst_channel_score is not None and not math.isnan(self.worst_channel_score) else None,
        }


@dataclass
class ModelObservationHealthAssessment:
    """
    Complete engine health assessment evaluated at a single time step.
    Strictly separates physics consistency (HI_raw), coverage (C_obs), and data quality (C_data).
    HI_raw is the arithmetic mean of active subsystem scores.
    """
    timestamp: float
    engine_id: str
    state: HealthState
    HI_raw: float                       # Arithmetic mean of active subsystem scores in [0.0, 1.0] (NaN if UNAVAILABLE)
    HI_smooth: float                    # Causal EWMA smoothed health index (NaN if UNAVAILABLE)
    HI_cov_adj: Optional[float]         # Optional coverage-adjusted index: HI_raw * C_obs
    C_obs: float                        # Observability coverage: valid_primary / 9.0
    C_data: float                       # Data quality/confidence from Phase 3 CanonicalTwinState
    subsystems: Dict[str, SubsystemHealthAssessment]
    channel_indicators: Dict[str, ChannelHealthIndicator]
    min_subsystem_score: Optional[float] = None  # Explicitly named separate diagnostic metric (does NOT affect HI_raw)
    worst_channel_score: Optional[float] = None  # Explicitly named separate diagnostic metric (does NOT affect HI_raw)
    cylinder_assessment: Optional[Dict[str, Any]] = None
    active_anomalies: List[str] = field(default_factory=list)
    config_provenance: str = "ENGINEERING_HEURISTIC"
    is_empirically_validated: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "engine_id": self.engine_id,
            "state": self.state.value,
            "HI_raw": round(self.HI_raw, 4) if not math.isnan(self.HI_raw) else None,
            "HI_smooth": round(self.HI_smooth, 4) if not math.isnan(self.HI_smooth) else None,
            "HI_cov_adj": round(self.HI_cov_adj, 4) if self.HI_cov_adj is not None and not math.isnan(self.HI_cov_adj) else None,
            "C_obs": round(self.C_obs, 4),
            "C_data": round(self.C_data, 4),
            "subsystems": {k: v.to_dict() for k, v in self.subsystems.items()},
            "channel_indicators": {k: v.to_dict() for k, v in self.channel_indicators.items()},
            "min_subsystem_score": round(self.min_subsystem_score, 4) if self.min_subsystem_score is not None and not math.isnan(self.min_subsystem_score) else None,
            "worst_channel_score": round(self.worst_channel_score, 4) if self.worst_channel_score is not None and not math.isnan(self.worst_channel_score) else None,
            "cylinder_assessment": self.cylinder_assessment,
            "active_anomalies": self.active_anomalies,
            "config_provenance": self.config_provenance,
            "is_empirically_validated": self.is_empirically_validated,
            "metadata": self.metadata,
        }


class HealthEvaluator:
    """
    Stateful physics-based health assessment evaluator.

    Tracks channel abnormal persistence, computes subsystem scores without double counting,
    evaluates overall engine health state, and maintains EWMA smoothing.
    """

    def __init__(self, config: Optional[HealthIndicatorConfig] = None):
        self.config = config or HealthIndicatorConfig()
        # Internal state tracking: channel -> elapsed duration in abnormal state
        self._channel_abnormal_duration: Dict[str, float] = {}
        # Channel recovery tracking: channel -> elapsed duration in normal state while recovering
        self._channel_healthy_duration: Dict[str, float] = {}
        # Confirmed channel states
        self._confirmed_channel_states: Dict[str, HealthState] = {}
        # Causal EWMA state
        self._last_hi_smooth: Optional[float] = None
        self._last_timestamp: Optional[float] = None

    def reset(self) -> None:
        """Reset internal temporal persistence and EWMA history."""
        self._channel_abnormal_duration.clear()
        self._channel_healthy_duration.clear()
        self._confirmed_channel_states.clear()
        self._last_hi_smooth = None
        self._last_timestamp = None

    def evaluate(
        self,
        residual_vector: ResidualVector,
        data_confidence: float = 1.0,
        dt: Optional[float] = None,
        engine_id: str = "ROTAX_914_GREYBOX",
    ) -> ModelObservationHealthAssessment:
        """
        Evaluate engine health assessment from a ResidualVector.

        Args:
            residual_vector: ResidualVector from QualityAwareResidualGenerator.
            data_confidence: Telemetry quality / heuristic confidence from Phase 3 estimator (C_data).
            dt: Optional elapsed time increment. If None, inferred from timestamp.
            engine_id: Engine identifier.

        Returns:
            ModelObservationHealthAssessment with full subsystem and channel breakdowns.
        """
        # Determine physical time step dt
        timestamp = residual_vector.timestamp
        if dt is None:
            if self._last_timestamp is not None and timestamp > self._last_timestamp:
                step_dt = timestamp - self._last_timestamp
            else:
                step_dt = 0.1
        else:
            step_dt = max(1e-4, float(dt))
        self._last_timestamp = timestamp

        # Coverage assessment
        c_obs = residual_vector.coverage_fraction
        valid_primary_count = residual_vector.valid_primary_count
        c_data = max(0.0, min(1.0, float(data_confidence)))

        channel_indicators: Dict[str, ChannelHealthIndicator] = {}
        subsystem_channels: Dict[str, List[str]] = {
            "THERMAL": ["cht", "coolant_temp", "oil_temp"],
            "LUBRICATION": ["oil_pressure"],
            "FUEL": ["fuel_flow"],
            "COMBUSTION": ["egt"],
            "MECHANICAL": ["vibration"],
            "ROTATIONAL": ["rpm", "map_bar"],
        }

        active_anomalies: List[str] = []

        # 1. Evaluate individual channels
        for ch_name, res in residual_vector.residuals.items():
            if not res.valid or math.isnan(res.normalized_residual):
                # Channel is invalid or unobserved
                self._channel_abnormal_duration[ch_name] = 0.0
                self._channel_healthy_duration[ch_name] = 0.0
                self._confirmed_channel_states[ch_name] = HealthState.UNAVAILABLE

                channel_indicators[ch_name] = ChannelHealthIndicator(
                    channel=ch_name,
                    primary_subsystem=res.primary_subsystem,
                    raw_residual=float("nan"),
                    normalized_residual=float("nan"),
                    z_score=float("nan"),
                    channel_score=float("nan"),
                    state=HealthState.UNAVAILABLE,
                    is_primary=res.is_primary_engine_vote,
                    valid=False,
                    units=res.units,
                    observed=res.observed_value,
                    predicted=res.predicted_value,
                    rejection_reason=res.reason or "INVALID_OR_DROPOUT",
                )
                continue

            z = abs(res.normalized_residual)
            raw_res = res.raw_residual
            norm_res = res.normalized_residual

            # Calculate raw penalty: 0 below tau_nom, ramps linearly to 1.0 at tau_crit
            if z <= self.config.tau_nom:
                penalty = 0.0
                instant_state = HealthState.HEALTHY
            elif z < self.config.tau_crit:
                penalty = (z - self.config.tau_nom) / max(1e-4, (self.config.tau_crit - self.config.tau_nom))
                instant_state = HealthState.DEGRADED
            else:
                penalty = 1.0
                instant_state = HealthState.CRITICAL

            ch_score = max(0.0, min(1.0, 1.0 - penalty))

            # Temporal persistence tracking for state classification
            current_confirmed = self._confirmed_channel_states.get(ch_name, HealthState.HEALTHY)

            if instant_state in (HealthState.DEGRADED, HealthState.CRITICAL):
                self._channel_healthy_duration[ch_name] = 0.0
                self._channel_abnormal_duration[ch_name] = (
                    self._channel_abnormal_duration.get(ch_name, 0.0) + step_dt
                )

                if instant_state == HealthState.CRITICAL:
                    # Critical threshold violation triggers CRITICAL immediately or upon persistence
                    confirmed_state = HealthState.CRITICAL
                elif self._channel_abnormal_duration[ch_name] >= self.config.persistence_seconds:
                    confirmed_state = HealthState.DEGRADED
                else:
                    confirmed_state = HealthState.WATCH
            else:
                # Instant state is HEALTHY
                self._channel_abnormal_duration[ch_name] = 0.0
                self._channel_healthy_duration[ch_name] = (
                    self._channel_healthy_duration.get(ch_name, 0.0) + step_dt
                )

                if current_confirmed in (HealthState.WATCH, HealthState.DEGRADED, HealthState.CRITICAL):
                    # Requires recovery persistence to return fully to HEALTHY
                    if self._channel_healthy_duration[ch_name] >= self.config.recovery_seconds:
                        confirmed_state = HealthState.HEALTHY
                    else:
                        confirmed_state = HealthState.WATCH
                else:
                    confirmed_state = HealthState.HEALTHY

            self._confirmed_channel_states[ch_name] = confirmed_state

            if confirmed_state in (HealthState.WATCH, HealthState.DEGRADED, HealthState.CRITICAL):
                active_anomalies.append(f"{ch_name}:{confirmed_state.value}(z={z:.2f})")

            channel_indicators[ch_name] = ChannelHealthIndicator(
                channel=ch_name,
                primary_subsystem=res.primary_subsystem,
                raw_residual=raw_res,
                normalized_residual=norm_res,
                z_score=z,
                channel_score=ch_score,
                state=confirmed_state,
                is_primary=res.is_primary_engine_vote,
                valid=True,
                units=res.units,
                observed=res.observed_value,
                predicted=res.predicted_value,
                rejection_reason="",
            )

        # 2. Evaluate Primary Subsystems
        subsystem_assessments: Dict[str, SubsystemHealthAssessment] = {}
        subsystem_scores: Dict[str, float] = {}

        for sub_name, channels in subsystem_channels.items():
            valid_ch_in_sub = [ch for ch in channels if channel_indicators.get(ch) and channel_indicators[ch].valid]
            ch_scores = {ch: channel_indicators[ch].channel_score for ch in valid_ch_in_sub}

            # Secondary contextual evidence (ZERO engine vote)
            secondary_ev: Dict[str, Any] = {}
            if sub_name == "COMBUSTION":
                if residual_vector.cylinder_residuals is not None:
                    secondary_ev["egt_spread_c"] = residual_vector.cylinder_residuals.egt_spread_c
                    secondary_ev["egt_imbalance_max_c"] = residual_vector.cylinder_residuals.egt_imbalance_max_c
            elif sub_name == "THERMAL":
                if residual_vector.cylinder_residuals is not None:
                    secondary_ev["cht_spread_c"] = residual_vector.cylinder_residuals.cht_spread_c
                    secondary_ev["cht_imbalance_max_c"] = residual_vector.cylinder_residuals.cht_imbalance_max_c
                cat_ind = channel_indicators.get("charge_air_temp")
                if cat_ind and cat_ind.valid:
                    secondary_ev["charge_air_temp_residual"] = cat_ind.raw_residual
            elif sub_name == "LUBRICATION":
                ot_ind = channel_indicators.get("oil_temp")
                if ot_ind and ot_ind.valid:
                    secondary_ev["oil_temp_context_c"] = ot_ind.raw_residual

            if valid_ch_in_sub:
                sub_score = sum(ch_scores.values()) / len(valid_ch_in_sub)
                subsystem_scores[sub_name] = sub_score
                worst_ch = min(ch_scores.values())

                # Determine subsystem categorical state from confirmed channel states
                sub_states = [channel_indicators[ch].state for ch in valid_ch_in_sub]
                if HealthState.CRITICAL in sub_states:
                    sub_state = HealthState.CRITICAL
                elif HealthState.DEGRADED in sub_states:
                    sub_state = HealthState.DEGRADED
                elif HealthState.WATCH in sub_states:
                    sub_state = HealthState.WATCH
                else:
                    sub_state = HealthState.HEALTHY
            else:
                sub_score = float("nan")
                worst_ch = None
                sub_state = HealthState.UNAVAILABLE

            subsystem_assessments[sub_name] = SubsystemHealthAssessment(
                subsystem=sub_name,
                score=sub_score,
                state=sub_state,
                primary_channels=channels,
                valid_channels=valid_ch_in_sub,
                channel_scores=ch_scores,
                secondary_evidence=secondary_ev,
                worst_channel_score=worst_ch,
            )

        # 3. Evaluate Engine-Level Health Index and Coverage Gate
        # Invariant: minimum_valid_primary_channels = 5 (4/9 is UNAVAILABLE, 5/9 is available)
        if valid_primary_count < self.config.min_primary_channels:
            overall_state = HealthState.UNAVAILABLE
            hi_raw = float("nan")
            hi_smooth = float("nan")
            hi_cov_adj = None
            min_sub_score = None
            worst_ch_score = None
        else:
            # H_phys = arithmetic mean of active subsystem scores (equal weighting across active subsystems)
            active_subsystems = [s for s, sc in subsystem_scores.items() if not math.isnan(sc)]
            if active_subsystems:
                hi_raw = sum(subsystem_scores[s] for s in active_subsystems) / len(active_subsystems)
                hi_raw = max(0.0, min(1.0, float(hi_raw)))
                min_sub_score = min(subsystem_scores[s] for s in active_subsystems)
                valid_ch_scores = [
                    ind.channel_score for ind in channel_indicators.values()
                    if ind.valid and not math.isnan(ind.channel_score) and ind.is_primary
                ]
                worst_ch_score = min(valid_ch_scores) if valid_ch_scores else None
            else:
                hi_raw = float("nan")
                min_sub_score = None
                worst_ch_score = None

            # Coverage-adjusted HI (explicitly independent from HI_raw)
            hi_cov_adj = hi_raw * c_obs if not math.isnan(hi_raw) else None

            # Causal EWMA Smoothing
            if not math.isnan(hi_raw):
                if self._last_hi_smooth is None or math.isnan(self._last_hi_smooth):
                    hi_smooth = hi_raw
                else:
                    alpha = self.config.ewma_alpha
                    hi_smooth = alpha * hi_raw + (1.0 - alpha) * self._last_hi_smooth
                self._last_hi_smooth = hi_smooth
            else:
                hi_smooth = float("nan")

            # Determine overall engine health state from subsystem states and smoothed health index
            all_sub_states = [s.state for s in subsystem_assessments.values() if s.state != HealthState.UNAVAILABLE]
            if HealthState.CRITICAL in all_sub_states or (not math.isnan(hi_smooth) and hi_smooth < 0.45):
                overall_state = HealthState.CRITICAL
            elif HealthState.DEGRADED in all_sub_states or (not math.isnan(hi_smooth) and hi_smooth < 0.70):
                overall_state = HealthState.DEGRADED
            elif HealthState.WATCH in all_sub_states or (not math.isnan(hi_smooth) and hi_smooth < 0.85):
                overall_state = HealthState.WATCH
            else:
                overall_state = HealthState.HEALTHY

        # 4. Pack Cylinder Localization Assessment
        cyl_assessment = None
        if residual_vector.cylinder_residuals is not None:
            cr = residual_vector.cylinder_residuals
            cyl_assessment = {
                "cht_runner_residuals": cr.cht_runner_residuals,
                "egt_runner_residuals": cr.egt_runner_residuals,
                "cht_spread_c": cr.cht_spread_c,
                "egt_spread_c": cr.egt_spread_c,
                "cht_imbalance_max_c": cr.cht_imbalance_max_c,
                "egt_imbalance_max_c": cr.egt_imbalance_max_c,
                "valid_cylinder_count": cr.valid_cylinder_count,
            }

        return ModelObservationHealthAssessment(
            timestamp=timestamp,
            engine_id=engine_id,
            state=overall_state,
            HI_raw=hi_raw,
            HI_smooth=hi_smooth,
            HI_cov_adj=hi_cov_adj,
            C_obs=c_obs,
            C_data=c_data,
            subsystems=subsystem_assessments,
            channel_indicators=channel_indicators,
            min_subsystem_score=min_sub_score,
            worst_channel_score=worst_ch_score,
            cylinder_assessment=cyl_assessment,
            active_anomalies=active_anomalies,
            config_provenance=self.config.provenance,
            is_empirically_validated=self.config.is_empirically_validated,
            metadata={
                "valid_primary_count": valid_primary_count,
                "total_primary_channels": 9,
                "dt_seconds": round(step_dt, 4),
            },
        )
