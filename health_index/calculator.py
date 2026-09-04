"""
Health Index calculation, channel degradation evidence, and deterministic sensor isolation.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
import math
import numpy as np
import pandas as pd

from health_index.schema import (
    HealthIndexConfig,
    HealthDataQuality,
    HealthState,
)


def compute_channel_evidence(
    residual: float,
    tau_nominal: float = 1.5,
    tau_critical: float = 5.0,
) -> float:
    """
    Map normalized residual magnitude to channel degradation evidence d_i in [0.0, 1.0].

    Args:
        residual: Normalized residual value (in sigma units).
        tau_nominal: Nominal noise deadband. Below this, d_i = 0.0.
        tau_critical: Critical threshold. At or above this, d_i = 1.0.

    Returns:
        d_i degradation evidence in [0.0, 1.0], or NaN if residual is NaN.
    """
    if math.isnan(residual):
        return float("nan")

    mag = abs(residual)
    if mag <= tau_nominal:
        return 0.0
    if mag >= tau_critical:
        return 1.0

    return float((mag - tau_nominal) / (tau_critical - tau_nominal))


def _make_key(engine_id: str, mission_id: Optional[str]) -> Tuple[str, str]:
    return (str(engine_id), str(mission_id) if mission_id is not None else "")


class SensorIsolationTracker:
    """
    Tracks and applies deterministic observation-quality heuristics for sensor-fault isolation.

    Heuristic Rule:
    A channel is isolated if:
    - Case A: Explicit upstream Phase 8 diagnostic context indicates sensor_fault (confidence >= 0.60).
    - Case B: One channel exhibits |r_k| >= outlier_sigma while all other valid physical channels
      exhibit |r_j| <= correlated_max_sigma continuously for >= persist_s seconds with timestamp gaps <= max_gap_s.

    This rule is an engineering heuristic, not a certified physical impossibility criterion.
    """

    def __init__(self, config: HealthIndexConfig):
        self.config = config
        self._suspect_channel: Dict[Tuple[str, str], Optional[str]] = {}
        self._persist_start_time: Dict[Tuple[str, str], Optional[float]] = {}
        self._last_timestamp: Dict[Tuple[str, str], Optional[float]] = {}

    def reset(
        self,
        engine_id: Optional[str] = None,
        mission_id: Optional[str] = None,
    ) -> None:
        """Reset internal tracking state."""
        if engine_id is not None and mission_id is not None:
            key = _make_key(engine_id, mission_id)
            self._suspect_channel.pop(key, None)
            self._persist_start_time.pop(key, None)
            self._last_timestamp.pop(key, None)
        elif engine_id is not None:
            keys_to_remove = [k for k in self._suspect_channel if k[0] == str(engine_id)]
            for k in keys_to_remove:
                self._suspect_channel.pop(k, None)
                self._persist_start_time.pop(k, None)
                self._last_timestamp.pop(k, None)
        else:
            self._suspect_channel.clear()
            self._persist_start_time.clear()
            self._last_timestamp.clear()

    def evaluate_isolation(
        self,
        timestamp: float,
        residuals: Dict[str, float],
        upstream_fault_type: Optional[str] = None,
        upstream_confidence: Optional[float] = None,
        upstream_suspect_channel: Optional[str] = None,
        engine_id: str = "ENG_001",
        mission_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Determine if any channel should be isolated as an observation-layer sensor fault.

        Returns:
            Name of isolated channel if isolation condition is met, else None.
        """
        key = _make_key(engine_id, mission_id)

        # Case A: Explicit upstream diagnostic context
        if (
            upstream_fault_type == "sensor_fault"
            and upstream_confidence is not None
            and upstream_confidence >= 0.60
        ):
            if upstream_suspect_channel and upstream_suspect_channel in residuals:
                return upstream_suspect_channel
            # If specific channel not passed, identify channel with maximum absolute residual
            valid_items = [(ch, abs(r)) for ch, r in residuals.items() if not math.isnan(r)]
            if valid_items:
                max_ch, max_val = max(valid_items, key=lambda x: x[1])
                if max_val >= self.config.tau_nominal:
                    return max_ch

        # Case B: Deterministic Multi-Channel Disconnect Heuristic
        valid_res = {ch: abs(r) for ch, r in residuals.items() if not math.isnan(r)}
        if len(valid_res) < self.config.min_valid_channels:
            self.reset(engine_id=engine_id, mission_id=mission_id)
            return None

        # Find candidates exceeding outlier_sigma
        outliers = [
            ch for ch, mag in valid_res.items()
            if mag >= self.config.sensor_isolation_outlier_sigma
        ]

        if len(outliers) == 1:
            candidate = outliers[0]
            # Check if all other channels are within correlated_max_sigma
            other_mags = [mag for ch, mag in valid_res.items() if ch != candidate]
            all_others_nominal = all(
                mag <= self.config.sensor_isolation_correlated_max_sigma
                for mag in other_mags
            )

            if all_others_nominal:
                last_t = self._last_timestamp.get(key, None)
                suspect = self._suspect_channel.get(key, None)

                # Continuity check
                if suspect == candidate and last_t is not None:
                    gap = timestamp - last_t
                    if gap <= self.config.sensor_isolation_max_gap_s and gap >= 0.0:
                        self._last_timestamp[key] = timestamp
                    else:
                        # Gap exceeded: reset persistence window
                        self._persist_start_time[key] = timestamp
                        self._last_timestamp[key] = timestamp
                else:
                    self._suspect_channel[key] = candidate
                    self._persist_start_time[key] = timestamp
                    self._last_timestamp[key] = timestamp

                # Check persistence duration
                persist_start = self._persist_start_time.get(key, None)
                if (
                    persist_start is not None
                    and (timestamp - persist_start) >= self.config.sensor_isolation_persist_s
                ):
                    return candidate
                return None

        # Condition not met: reset tracker for this key
        self.reset(engine_id=engine_id, mission_id=mission_id)
        return None


class HealthCalculator:
    """
    Computes instantaneous raw degradation and Health Index from normalized residuals.
    """

    def __init__(self, config: Optional[HealthIndexConfig] = None):
        self.config = config or HealthIndexConfig()
        self.sensor_tracker = SensorIsolationTracker(self.config)

    def compute(
        self,
        timestamp: float,
        residuals: Dict[str, float],
        upstream_fault_type: Optional[str] = None,
        upstream_confidence: Optional[float] = None,
        upstream_suspect_channel: Optional[str] = None,
        engine_id: str = "ENG_001",
        mission_id: Optional[str] = None,
    ) -> Tuple[float, float, Dict[str, float], Dict[str, float], List[str], List[str], List[str], List[str], Dict[str, float], str]:
        """
        Compute instantaneous Health Index and attribution for a single timestep.

        Returns:
            (raw_health_index, raw_degradation_score, channel_contributions,
             channel_degradation_evidence, dominant_degraded_channels,
             valid_channels, missing_channels, excluded_channels,
             effective_channel_weights, data_quality)
        """
        all_configured_channels = list(self.config.channel_weights.keys())

        # 1. Audit channel validity
        valid_channels: List[str] = []
        missing_channels: List[str] = []
        clean_residuals: Dict[str, float] = {}

        for ch in all_configured_channels:
            val = residuals.get(ch, float("nan"))
            if val is not None and not math.isnan(val):
                valid_channels.append(ch)
                clean_residuals[ch] = float(val)
            else:
                missing_channels.append(ch)
                clean_residuals[ch] = float("nan")

        # 2. Check sensor isolation
        isolated_ch = self.sensor_tracker.evaluate_isolation(
            timestamp=timestamp,
            residuals=clean_residuals,
            upstream_fault_type=upstream_fault_type,
            upstream_confidence=upstream_confidence,
            upstream_suspect_channel=upstream_suspect_channel,
            engine_id=engine_id,
            mission_id=mission_id,
        )

        excluded_channels: List[str] = [isolated_ch] if isolated_ch else []
        active_channels = [ch for ch in valid_channels if ch not in excluded_channels]

        # 3. Check sufficiency of physical evidence
        if len(active_channels) < self.config.min_valid_channels:
            data_quality = HealthDataQuality.INSUFFICIENT_DATA.value
            return (
                float("nan"),
                float("nan"),
                {},
                {},
                [],
                valid_channels,
                missing_channels,
                excluded_channels,
                {},
                data_quality,
            )

        data_quality = (
            HealthDataQuality.SENSOR_ISOLATED.value
            if excluded_channels
            else HealthDataQuality.VALID.value
        )

        # 4. Dynamic Weight Renormalization
        active_base_weights = {ch: self.config.channel_weights[ch] for ch in active_channels}
        total_active_w = sum(active_base_weights.values())
        if total_active_w <= 0.0:
            effective_weights = {ch: 1.0 / len(active_channels) for ch in active_channels}
        else:
            effective_weights = {ch: w / total_active_w for ch, w in active_base_weights.items()}

        # 5. Calculate Channel Evidence and Contributions
        evidence_dict: Dict[str, float] = {}
        contrib_dict: Dict[str, float] = {}
        raw_degradation = 0.0

        for ch in all_configured_channels:
            res_val = clean_residuals[ch]
            ev = compute_channel_evidence(
                res_val,
                tau_nominal=self.config.tau_nominal,
                tau_critical=self.config.tau_critical,
            )
            evidence_dict[ch] = ev

            if ch in active_channels:
                c_i = effective_weights[ch] * ev
                contrib_dict[ch] = float(c_i)
                raw_degradation += c_i
            else:
                contrib_dict[ch] = 0.0

        # Bound degradation score to [0.0, 1.0]
        raw_degradation = float(min(1.0, max(0.0, raw_degradation)))
        raw_hi = float(min(1.0, max(0.0, 1.0 - raw_degradation)))

        # 6. Determine Dominant Degraded Channels (evidence >= 0.20, sorted descending by contribution)
        degraded_candidates = [
            (ch, contrib_dict[ch])
            for ch in active_channels
            if evidence_dict[ch] >= 0.20
        ]
        degraded_candidates.sort(key=lambda x: x[1], reverse=True)
        dominant_degraded = [ch for ch, _ in degraded_candidates]

        return (
            raw_hi,
            raw_degradation,
            contrib_dict,
            evidence_dict,
            dominant_degraded,
            valid_channels,
            missing_channels,
            excluded_channels,
            effective_weights,
            data_quality,
        )
