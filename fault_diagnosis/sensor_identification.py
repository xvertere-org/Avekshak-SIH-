"""
Sensor-fault channel identification for Phase 8 Fault Diagnosis.
Deterministically identifies suspect faulty sensor(s) based on:
1. Missingness / NaN dropout validation
2. Normalized residual magnitudes and ranking
3. Phase 7 anomaly detection contributing channels
4. Physical decoupling criteria (isolated divergence vs. multi-channel physical response)

Strictly adheres to:
- No hardcoded channel defaults (e.g. never default to 'cht').
- Unambiguous reporting of "unknown" / uncertain sensor when evidence is ambiguous or confidence is low.
- Zero contamination of unrelated healthy channels.
"""

import math
from typing import Dict, Any, List, Optional, Tuple, Set


CANONICAL_CHANNELS: List[str] = [
    "rpm",
    "cht",
    "egt",
    "oil_temp",
    "oil_pressure",
    "fuel_flow",
    "vibration",
]


class SensorIdentificationResult(dict):
    """
    Structured result for sensor-fault identification supporting dict access,
    attribute access, and tuple unpacking: (suspect_sensor, suspect_sensors, sensor_isolation_status).
    """
    @property
    def suspect_sensor(self) -> Optional[str]:
        return self.get("suspect_sensor")

    @property
    def suspect_sensors(self) -> List[str]:
        return self.get("suspect_sensors", [])

    @property
    def sensor_isolation_status(self) -> str:
        return self.get("sensor_isolation_status", "NONE")

    def __iter__(self):
        return iter((self.suspect_sensor, self.suspect_sensors, self.sensor_isolation_status))


class SensorFaultIdentifier:
    """
    Identifies specific faulty sensor channel(s) when an observation-layer sensor fault occurs.
    """

    def __init__(
        self,
        outlier_threshold_sigma: float = 2.5,
        nominal_bound_sigma: float = 1.5,
        min_valid_channels_required: int = 4,
        confidence_threshold: float = 0.50,
    ):
        self.outlier_threshold_sigma = outlier_threshold_sigma
        self.nominal_bound_sigma = nominal_bound_sigma
        self.min_valid_channels_required = min_valid_channels_required
        self.confidence_threshold = confidence_threshold

    @staticmethod
    def _normalize_channel_name(ch: str) -> str:
        """Strip suffixes like _norm_residual, _residual, etc."""
        clean = ch.replace("_norm_residual", "").replace("_residual", "").replace("_expected", "")
        return clean.strip().lower()

    def identify_suspect_sensors(
        self,
        sample: Optional[Dict[str, Any]] = None,
        observed_telemetry: Optional[Dict[str, Any]] = None,
        contributing_channels: Optional[List[str]] = None,
        diagnostic_confidence: float = 1.0,
        residual_sample: Optional[Dict[str, Any]] = None,
        model_confidence: Optional[float] = None,
        **kwargs,
    ) -> SensorIdentificationResult:
        """
        Evaluate multi-source evidence to determine suspect sensor channel(s).

        Returns:
            SensorIdentificationResult:
            - suspect_sensor: str (e.g. "cht", "rpm", or "unknown")
            - suspect_sensors: List[str] (all identified channels)
            - sensor_isolation_status: str ("CONFIRMED", "UNCERTAIN", "NONE")
        """
        sample_dict = sample if sample is not None else (residual_sample or {})
        conf = model_confidence if model_confidence is not None else diagnostic_confidence

        # Low diagnostic confidence -> uncertain
        if conf < self.confidence_threshold:
            return SensorIdentificationResult({
                "suspect_sensor": "unknown",
                "suspect_sensors": [],
                "sensor_isolation_status": "UNCERTAIN",
            })

        # 1. Parse observed telemetry values and detect NaN dropouts
        obs_dict: Dict[str, Optional[float]] = {}
        if observed_telemetry is not None:
            for ch in CANONICAL_CHANNELS:
                val = observed_telemetry.get(ch)
                if val is not None and math.isfinite(float(val)):
                    obs_dict[ch] = float(val)
                else:
                    obs_dict[ch] = None
        else:
            for ch in CANONICAL_CHANNELS:
                val = sample_dict.get(ch)
                if val is not None and math.isfinite(float(val)):
                    obs_dict[ch] = float(val)
                elif ch in sample_dict:
                    obs_dict[ch] = None

        # Dropout candidates: explicitly present in telemetry schema but value is NaN/None
        dropout_candidates = [
            ch for ch, v in obs_dict.items()
            if v is None and ch in (observed_telemetry or sample_dict)
        ]

        # 2. Parse normalized residuals
        norm_residuals: Dict[str, float] = {}
        for ch in CANONICAL_CHANNELS:
            norm_key = f"{ch}_norm_residual"
            if norm_key in sample_dict and sample_dict[norm_key] is not None and math.isfinite(float(sample_dict[norm_key])):
                norm_residuals[ch] = abs(float(sample_dict[norm_key]))
            elif ch in sample_dict and sample_dict[ch] is not None and math.isfinite(float(sample_dict[ch])):
                # If dictionary contains raw/pre-normalized channel
                norm_residuals[ch] = abs(float(sample_dict[ch]))

        # 3. Incorporate Phase 7 contributing channels
        phase7_channels: Set[str] = set()
        if contributing_channels:
            for c in contributing_channels:
                norm_c = self._normalize_channel_name(c)
                if norm_c in CANONICAL_CHANNELS:
                    phase7_channels.add(norm_c)

        # 4. Identify residual outlier candidates
        outlier_candidates: List[Tuple[str, float]] = []
        for ch, mag in norm_residuals.items():
            if mag >= self.outlier_threshold_sigma or ch in phase7_channels:
                outlier_candidates.append((ch, mag))

        # Sort outlier candidates descending by residual magnitude
        outlier_candidates.sort(key=lambda x: x[1], reverse=True)

        # Combine dropout candidates and residual outlier candidates
        all_candidate_channels: List[str] = []
        for ch in dropout_candidates:
            if ch not in all_candidate_channels:
                all_candidate_channels.append(ch)
        for ch, _ in outlier_candidates:
            if ch not in all_candidate_channels:
                all_candidate_channels.append(ch)

        # No candidates found
        if not all_candidate_channels:
            return SensorIdentificationResult({
                "suspect_sensor": "unknown",
                "suspect_sensors": [],
                "sensor_isolation_status": "UNCERTAIN",
            })

        # Safety Check: Maximum allowed sensor isolations
        # Must retain at least min_valid_channels_required (>= 4 channels)
        max_allowable_isolated = len(CANONICAL_CHANNELS) - self.min_valid_channels_required
        if len(all_candidate_channels) > max_allowable_isolated:
            # Too many corrupted channels -> systemic physical breakdown or blackout, not isolated sensor faults
            return SensorIdentificationResult({
                "suspect_sensor": "unknown",
                "suspect_sensors": [],
                "sensor_isolation_status": "UNCERTAIN",
            })

        # 5. Physical Decoupling Check
        # Check non-candidate channels: in a genuine sensor fault, other channels should be within nominal bounds
        non_candidates = [ch for ch in CANONICAL_CHANNELS if ch not in all_candidate_channels]
        coupled_deviations = [
            ch for ch in non_candidates
            if norm_residuals.get(ch, 0.0) > self.nominal_bound_sigma
        ]

        # If coupled non-candidate channels also deviate significantly, this is a multi-subsystem physical fault
        if len(coupled_deviations) >= 2:
            return SensorIdentificationResult({
                "suspect_sensor": "unknown",
                "suspect_sensors": [],
                "sensor_isolation_status": "UNCERTAIN",
            })

        # 6. Finalize Confirmed Identification
        primary_ch = all_candidate_channels[0]
        return SensorIdentificationResult({
            "suspect_sensor": primary_ch,
            "suspect_sensors": all_candidate_channels,
            "sensor_isolation_status": "CONFIRMED",
        })
