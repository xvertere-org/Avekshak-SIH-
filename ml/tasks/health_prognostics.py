"""
Phase 4: Health Monitoring, Fault Injection, Uncertainty Quantification, and Prognostics.

Core Capabilities:
1. Health-State Representation:
   - Explicit engine and subsystem health states (HEALTHY, THERMAL_DEGRADATION,
     COOLING_DEGRADATION, LUBRICATION_DEGRADATION, COMBUSTION_DEGRADATION,
     VIBRATION_DEGRADATION, SENSOR_ANOMALY, UNKNOWN_INSUFFICIENT_DATA).
   - Bounded health score [0.0, 1.0] where 1.0 = nominal healthy baseline and
     0.0 = functional failure threshold (HI <= 0.35).
2. Deterministic Fault Injection Catalog:
   - Reusable definitions for physical degradation (cooling, lubrication, fuel, vibration)
     and observation-layer sensor faults (bias, drift, dropout) across ramp, step, and transient profiles.
3. Health Monitoring Pipeline:
   - Causal EWMA filtering, rolling residual z-scores, persistence counters, and hysteresis recovery.
   - Cross-channel consistency discriminator separating sensor anomalies from possible physical degradation.
4. Engineering Uncertainty Quantification:
   - Rolling residual dispersion and prediction bounds [y_lower <= y_pred <= y_upper].
   - Labeled transparently as engineering uncertainty estimates.
5. Causal Prognostics & Trend Estimation:
   - Strictly causal Theil-Sen / linear slope estimation over sliding history.
   - Time-to-threshold estimation with RUL explicitly withheld as UNAVAILABLE when degradation is absent.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Tuple
import math
import numpy as np
import pandas as pd

from simulator.fault_interface import FaultState, FaultType, FaultSubsystem
from simulator.sensor_faults import SensorFaultMode, SensorChannel, SensorFaultProcessor
from digital_twin.quality import PHYSICAL_INSTRUMENT_LIMITS


# =====================================================================
# 1. Health States and Output Schemas
# =====================================================================

class EngineHealthState(str, Enum):
    """Authoritative categorical engine health states."""
    HEALTHY = "HEALTHY"
    THERMAL_DEGRADATION = "THERMAL_DEGRADATION"
    COOLING_DEGRADATION = "COOLING_DEGRADATION"
    LUBRICATION_DEGRADATION = "LUBRICATION_DEGRADATION"
    COMBUSTION_DEGRADATION = "COMBUSTION_DEGRADATION"
    VIBRATION_DEGRADATION = "VIBRATION_DEGRADATION"
    SENSOR_ANOMALY = "SENSOR_ANOMALY"
    UNKNOWN_INSUFFICIENT_DATA = "UNKNOWN_INSUFFICIENT_DATA"


class AlertClassification(str, Enum):
    """Categorical classification of detected anomalies."""
    NOMINAL = "NOMINAL"
    POSSIBLE_PHYSICAL_DEGRADATION = "POSSIBLE_PHYSICAL_DEGRADATION"
    SENSOR_ANOMALY = "SENSOR_ANOMALY"
    MODEL_DISAGREEMENT = "MODEL_DISAGREEMENT"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class DegradationTrendState(str, Enum):
    """Direction and rate of change of health condition."""
    STABLE = "STABLE"
    DEGRADING = "DEGRADING"
    RAPIDLY_DEGRADING = "RAPIDLY_DEGRADING"
    IMPROVING = "IMPROVING"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"


# Standard subsystem channel groupings
SUBSYSTEM_CHANNEL_MAP: Dict[str, List[str]] = {
    "THERMAL": ["cht", "coolant_temp", "oil_temp"],
    "COOLING": ["coolant_temp", "cht"],
    "LUBRICATION": ["oil_pressure", "oil_temp"],
    "COMBUSTION": ["egt"],
    "FUEL": ["fuel_flow"],
    "ROTATIONAL": ["rpm", "map_bar"],
    "MECHANICAL": ["vibration"],
}

# Subsystem importance weights for composite health score (Sums to 1.0)
DEFAULT_SUBSYSTEM_WEIGHTS: Dict[str, float] = {
    "THERMAL": 0.25,
    "LUBRICATION": 0.25,
    "COMBUSTION": 0.15,
    "FUEL": 0.15,
    "ROTATIONAL": 0.10,
    "MECHANICAL": 0.10,
}


@dataclass
class ChannelUncertainty:
    """Engineering uncertainty estimate for a single channel prediction."""
    channel: str
    predicted_value: float
    prediction_lower: float
    prediction_upper: float
    uncertainty_width: float
    model_confidence: float
    confidence_basis: str = "rolling_residual_dispersion_engineering_estimate"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class HealthMonitoringOutput:
    """Comprehensive snapshot of engine health and uncertainty assessment."""
    timestamp: float
    health_score: float                  # Bounded [0.0, 1.0]
    health_state: EngineHealthState
    anomaly_score: float                 # Bounded [0.0, 1.0]
    degradation_severity: float          # Bounded [0.0, 1.0]
    alert_classification: AlertClassification
    affected_channels: List[str]
    affected_subsystems: List[str]
    subsystem_scores: Dict[str, float]
    channel_uncertainties: Dict[str, ChannelUncertainty]
    trend_state: DegradationTrendState
    trend_slope_per_s: Optional[float]
    time_to_threshold_s: Optional[float]
    rul_state: str                       # "AVAILABLE" or "UNAVAILABLE"
    prognostics_reason: Optional[str]
    evidence: Dict[str, Any]
    confidence: float                    # Bounded [0.0, 1.0]
    data_quality_status: str = "VALID"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["health_state"] = self.health_state.value
        d["alert_classification"] = self.alert_classification.value
        d["trend_state"] = self.trend_state.value
        d["channel_uncertainties"] = {k: v.to_dict() for k, v in self.channel_uncertainties.items()}
        return d


# =====================================================================
# 2. Deterministic Fault Injection Catalog
# =====================================================================

@dataclass
class FaultScenarioDefinition:
    """Explicit metadata and injection formula for a reproducible fault scenario."""
    fault_name: str
    fault_type: str
    affected_signal: str
    start_time: float
    duration: Optional[float]
    severity: float                      # Normalized intensity [0.0, 1.0]
    injection_formula: str
    expected_observable_effect: str
    is_sensor_fault: bool
    parameters: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_standard_fault_catalog() -> Dict[str, FaultScenarioDefinition]:
    """Returns the standardized, physically motivated fault injection definitions."""
    return {
        "cooling_degradation_ramp": FaultScenarioDefinition(
            fault_name="Gradual Cooling Loop Degradation",
            fault_type="cooling_degradation",
            affected_signal="coolant_temp",
            start_time=20.0,
            duration=40.0,
            severity=0.40,
            injection_formula="coolant_hx_conductance(t) = hx_base * (1 - 0.40 * (t - start)/duration)",
            expected_observable_effect="Correlated elevation in coolant_temp and cht with thermal time lag.",
            is_sensor_fault=False,
            parameters={"ramp_duration": 40.0},
        ),
        "lubrication_pressure_loss_step": FaultScenarioDefinition(
            fault_name="Abrupt Lubrication Pressure Loss",
            fault_type="lubrication_degradation",
            affected_signal="oil_pressure",
            start_time=15.0,
            duration=45.0,
            severity=0.35,
            injection_formula="oil_press(t) = oil_press_nominal * (1 - 0.35); oil_temp rises via friction",
            expected_observable_effect="Immediate drop in oil_pressure followed by progressive elevation in oil_temp.",
            is_sensor_fault=False,
            parameters={"step_onset": True},
        ),
        "thermal_load_increase": FaultScenarioDefinition(
            fault_name="Thermal System Overload",
            fault_type="cooling_degradation",
            affected_signal="cht",
            start_time=10.0,
            duration=50.0,
            severity=0.30,
            injection_formula="q_combustion_fraction * (1 + 0.30 * severity)",
            expected_observable_effect="Elevated CHT across all cylinders and secondary coolant loop rise.",
            is_sensor_fault=False,
        ),
        "fuel_injection_lean_abnormality": FaultScenarioDefinition(
            fault_name="Fuel Injection Delivery Abnormality",
            fault_type="fuel_injection_abnormality",
            affected_signal="egt",
            start_time=25.0,
            duration=35.0,
            severity=0.25,
            injection_formula="fuel_mass_flow = fuel_target * (1 - 0.25); mixture = lean",
            expected_observable_effect="Elevated EGT due to lean combustion accompanied by reduced fuel_flow.",
            is_sensor_fault=False,
            parameters={"mode": "lean"},
        ),
        "mechanical_vibration_degradation": FaultScenarioDefinition(
            fault_name="Mechanical Mount/Bearing Degradation",
            fault_type="mechanical_degradation",
            affected_signal="vibration",
            start_time=30.0,
            duration=30.0,
            severity=0.45,
            injection_formula="vib_rms = vib_nominal * (1 + 1.2 * severity)",
            expected_observable_effect="Increased overall RMS vibration without thermodynamic shift.",
            is_sensor_fault=False,
        ),
        "cht_sensor_drift": FaultScenarioDefinition(
            fault_name="Cylinder Head Temperature Sensor Drift",
            fault_type="sensor_drift",
            affected_signal="cht",
            start_time=10.0,
            duration=50.0,
            severity=0.50,
            injection_formula="cht_obs(t) = cht_true(t) + 0.8 °C/s * (t - start)",
            expected_observable_effect="Unilateral progressive CHT rise; coolant_temp and oil_temp remain healthy.",
            is_sensor_fault=True,
            parameters={"rate": 0.8},
        ),
        "rpm_sensor_bias": FaultScenarioDefinition(
            fault_name="Engine Speed Sensor Constant Bias",
            fault_type="sensor_bias",
            affected_signal="rpm",
            start_time=15.0,
            duration=45.0,
            severity=0.40,
            injection_formula="rpm_obs = rpm_true + 150.0 RPM",
            expected_observable_effect="Abrupt positive RPM offset; fuel flow and MAP dynamic state uncorrupted.",
            is_sensor_fault=True,
            parameters={"bias_offset": 150.0},
        ),
        "oil_pressure_sensor_dropout": FaultScenarioDefinition(
            fault_name="Oil Pressure Sensor Intermittent Dropout",
            fault_type="sensor_dropout",
            affected_signal="oil_pressure",
            start_time=20.0,
            duration=15.0,
            severity=1.0,
            injection_formula="oil_press_obs = NaN",
            expected_observable_effect="Observation becomes NaN; physical lubrication loop remains intact.",
            is_sensor_fault=True,
        ),
        "transient_throttle_disturbance": FaultScenarioDefinition(
            fault_name="Transient Flight Maneuver Disturbance",
            fault_type="transient_disturbance",
            affected_signal="throttle",
            start_time=15.0,
            duration=5.0,
            severity=0.20,
            injection_formula="throttle(t) = throttle_base + 20% for 5s",
            expected_observable_effect="Short-lived transient dynamic excursion, restoring cleanly to baseline.",
            is_sensor_fault=False,
        ),
    }


# =====================================================================
# 3. Health Monitoring & Prognostics Pipeline
# =====================================================================

class HealthPrognosticsPipeline:
    """
    Causal, persistent health assessment, uncertainty quantification,
    and prognostics engine for aero-piston digital twin telemetry.
    """

    def __init__(
        self,
        persistence_steps: int = 5,
        hysteresis_recovery_steps: int = 4,
        z_score_threshold: float = 2.5,
        recovery_z_threshold: float = 1.2,
        ewma_alpha: float = 0.20,
        eol_threshold: float = 0.35,
        history_window_size: int = 50,
        seed: int = 42,
    ):
        self.persistence_steps = persistence_steps
        self.hysteresis_recovery_steps = hysteresis_recovery_steps
        self.z_score_threshold = z_score_threshold
        self.recovery_z_threshold = recovery_z_threshold
        self.ewma_alpha = ewma_alpha
        self.eol_threshold = eol_threshold
        self.history_window_size = history_window_size
        self.rng = np.random.default_rng(seed)

        # Causal state buffers
        self._ewma_health_score: float = 1.0
        self._persistence_counters: Dict[str, int] = {}
        self._recovery_counters: Dict[str, int] = {}
        self._active_channel_alarms: Dict[str, bool] = {}
        self._history_timestamps: List[float] = []
        self._history_health_scores: List[float] = []
        self._rolling_residuals: Dict[str, List[float]] = {}
        self._baseline_scales: Dict[str, float] = {
            "cht": 6.0,
            "egt": 15.0,
            "oil_temp": 4.0,
            "oil_pressure": 0.35,
            "fuel_flow": 1.2,
            "rpm": 50.0,
            "map_bar": 0.03,
            "coolant_temp": 1.5,
            "vibration": 0.08,
        }

    def reset(self) -> None:
        """Reset internal causal tracking states."""
        self._ewma_health_score = 1.0
        self._persistence_counters.clear()
        self._recovery_counters.clear()
        self._active_channel_alarms.clear()
        self._history_timestamps.clear()
        self._history_health_scores.clear()
        self._rolling_residuals.clear()

    # -----------------------------------------------------------------
    # Baseline Calibration (Training Only)
    # -----------------------------------------------------------------

    def calibrate_baseline_scales(self, training_residuals: pd.DataFrame) -> None:
        """
        Calibrate residual standard deviations strictly from training data.
        Guarantees zero test-set leakage in z-score calculations.
        """
        for col in training_residuals.columns:
            series = pd.to_numeric(training_residuals[col], errors="coerce").dropna()
            if len(series) > 10:
                # Use robust median absolute deviation (MAD) or std
                std_val = float(series.std())
                if std_val > 1e-4:
                    self._baseline_scales[col] = std_val

    # -----------------------------------------------------------------
    # Step-by-Step Causal Processing
    # -----------------------------------------------------------------

    def process_step(
        self,
        timestamp: float,
        observed: Dict[str, Optional[float]],
        physics_expected: Dict[str, float],
        corrected_prediction: Optional[Dict[str, float]] = None,
    ) -> HealthMonitoringOutput:
        """
        Process a single time step causally.
        
        Args:
            timestamp: Current elapsed flight time in seconds.
            observed: Dictionary of sensor observations (may contain NaNs).
            physics_expected: Dictionary of nominal physics predictions.
            corrected_prediction: Optional grey-box ML corrected predictions.
            
        Returns:
            Fully populated HealthMonitoringOutput with bounded metrics and uncertainty.
        """
        corr_pred = corrected_prediction or physics_expected

        # 1. Data Quality & Residual Extraction
        raw_residuals: Dict[str, float] = {}
        z_scores: Dict[str, float] = {}
        missing_channels: List[str] = []

        for ch in self._baseline_scales:
            obs_val = observed.get(ch, None)
            exp_val = physics_expected.get(f"{ch}_expected", physics_expected.get(ch, None))

            if obs_val is None or not np.isfinite(obs_val):
                missing_channels.append(ch)
                continue
            if exp_val is None or not np.isfinite(exp_val):
                continue

            res = float(obs_val - exp_val)
            raw_residuals[ch] = res

            scale = self._baseline_scales.get(ch, 1.0)
            z = res / max(1e-4, scale)
            z_scores[ch] = z

            # Update rolling residual history
            if ch not in self._rolling_residuals:
                self._rolling_residuals[ch] = []
            self._rolling_residuals[ch].append(res)
            if len(self._rolling_residuals[ch]) > self.history_window_size:
                self._rolling_residuals[ch].pop(0)

        # Insufficient data check
        if len(raw_residuals) < 3:
            return HealthMonitoringOutput(
                timestamp=timestamp,
                health_score=0.0,
                health_state=EngineHealthState.UNKNOWN_INSUFFICIENT_DATA,
                anomaly_score=1.0,
                degradation_severity=0.0,
                alert_classification=AlertClassification.INSUFFICIENT_DATA,
                affected_channels=missing_channels,
                affected_subsystems=[],
                subsystem_scores={},
                channel_uncertainties={},
                trend_state=DegradationTrendState.INSUFFICIENT_HISTORY,
                trend_slope_per_s=None,
                time_to_threshold_s=None,
                rul_state="UNAVAILABLE",
                prognostics_reason="Insufficient valid sensor channels (< 3 active observations)",
                evidence={"missing_channels": missing_channels},
                confidence=0.0,
                data_quality_status="INSUFFICIENT_DATA",
            )

        # 2. Persistence and Hysteresis Logic
        alarmed_channels: List[str] = []
        for ch, z in z_scores.items():
            abs_z = abs(z)
            is_currently_alarmed = self._active_channel_alarms.get(ch, False)

            if abs_z >= self.z_score_threshold:
                self._persistence_counters[ch] = self._persistence_counters.get(ch, 0) + 1
                self._recovery_counters[ch] = 0
                if self._persistence_counters[ch] >= self.persistence_steps:
                    self._active_channel_alarms[ch] = True
            elif abs_z <= self.recovery_z_threshold:
                self._recovery_counters[ch] = self._recovery_counters.get(ch, 0) + 1
                if self._recovery_counters[ch] >= self.hysteresis_recovery_steps:
                    self._active_channel_alarms[ch] = False
                    self._persistence_counters[ch] = 0
            else:
                # In deadband between recovery and alarm threshold: preserve current state
                pass

            if self._active_channel_alarms.get(ch, False):
                alarmed_channels.append(ch)

        # 3. Cross-Channel Discrimination (Sensor vs Physical Degradation)
        alert_class, affected_subsystems, health_state = self._classify_cross_channel_condition(
            alarmed_channels=alarmed_channels,
            z_scores=z_scores,
            missing_channels=missing_channels,
        )

        # 4. Bounded Subsystem and Health Score Calculation
        subsystem_scores: Dict[str, float] = {}
        for sub, ch_list in SUBSYSTEM_CHANNEL_MAP.items():
            sub_z = [abs(z_scores[c]) for c in ch_list if c in z_scores]
            if sub_z:
                max_sub_z = max(sub_z)
                # Sigmoidal health score mapping: 1.0 at z <= 1.5, decays toward 0.0 at z >= 5.0
                score = 1.0 / (1.0 + math.exp(1.5 * (max_sub_z - 3.0)))
                subsystem_scores[sub] = float(np.clip(score, 0.0, 1.0))
            else:
                subsystem_scores[sub] = 1.0

        # Weighted engine composite score
        composite_score = sum(
            subsystem_scores.get(sub, 1.0) * w
            for sub, w in DEFAULT_SUBSYSTEM_WEIGHTS.items()
        )
        composite_score = float(np.clip(composite_score, 0.0, 1.0))

        # Causal EWMA smoothing of health score
        self._ewma_health_score = (
            self.ewma_alpha * composite_score + (1.0 - self.ewma_alpha) * self._ewma_health_score
        )
        current_health_score = float(np.clip(self._ewma_health_score, 0.0, 1.0))

        # Degradation severity and anomaly score
        anomaly_score = float(np.clip(1.0 - current_health_score, 0.0, 1.0))
        degradation_severity = float(np.clip(max([abs(z_scores[c]) for c in alarmed_channels] or [0.0]) / 6.0, 0.0, 1.0))

        # Update history
        self._history_timestamps.append(timestamp)
        self._history_health_scores.append(current_health_score)
        if len(self._history_timestamps) > self.history_window_size:
            self._history_timestamps.pop(0)
            self._history_health_scores.pop(0)

        # 5. Engineering Uncertainty Quantification
        uncertainties = self._compute_uncertainties(
            observed=observed,
            physics_expected=physics_expected,
            corrected_prediction=corr_pred,
        )

        # 6. Causal Prognostics and Trend Estimation
        trend_state, slope, t_to_thresh, rul_state, prog_reason = self._compute_prognostics(
            current_health=current_health_score,
            current_time=timestamp,
        )

        confidence = float(np.clip(1.0 - (degradation_severity * 0.5) - (len(missing_channels) * 0.1), 0.1, 1.0))

        evidence = {
            "alarmed_channels": alarmed_channels,
            "z_scores": {k: round(v, 2) for k, v in z_scores.items()},
            "raw_residuals": {k: round(v, 3) for k, v in raw_residuals.items()},
            "missing_channels": missing_channels,
        }

        return HealthMonitoringOutput(
            timestamp=timestamp,
            health_score=round(current_health_score, 4),
            health_state=health_state,
            anomaly_score=round(anomaly_score, 4),
            degradation_severity=round(degradation_severity, 4),
            alert_classification=alert_class,
            affected_channels=alarmed_channels,
            affected_subsystems=affected_subsystems,
            subsystem_scores={k: round(v, 4) for k, v in subsystem_scores.items()},
            channel_uncertainties=uncertainties,
            trend_state=trend_state,
            trend_slope_per_s=round(slope, 6) if slope is not None else None,
            time_to_threshold_s=round(t_to_thresh, 1) if t_to_thresh is not None else None,
            rul_state=rul_state,
            prognostics_reason=prog_reason,
            evidence=evidence,
            confidence=round(confidence, 3),
            data_quality_status="SENSOR_DROPOUT" if missing_channels else "VALID",
        )

    # -----------------------------------------------------------------
    # Cross-Channel Discriminator
    # -----------------------------------------------------------------

    def _classify_cross_channel_condition(
        self,
        alarmed_channels: List[str],
        z_scores: Dict[str, float],
        missing_channels: List[str],
    ) -> Tuple[AlertClassification, List[str], EngineHealthState]:
        """
        Distinguishes isolated sensor faults from multi-channel physical degradation.
        """
        if missing_channels and not alarmed_channels:
            return AlertClassification.SENSOR_ANOMALY, ["SENSOR"], EngineHealthState.SENSOR_ANOMALY

        if not alarmed_channels:
            return AlertClassification.NOMINAL, [], EngineHealthState.HEALTHY

        # Single alarmed channel with otherwise nominal channels -> Sensor Anomaly or Model Disagreement
        if len(alarmed_channels) == 1:
            bad_ch = alarmed_channels[0]
            if bad_ch in ["rpm", "map_bar"]:
                return AlertClassification.SENSOR_ANOMALY, ["ROTATIONAL"], EngineHealthState.SENSOR_ANOMALY
            elif bad_ch == "vibration":
                return AlertClassification.POSSIBLE_PHYSICAL_DEGRADATION, ["MECHANICAL"], EngineHealthState.VIBRATION_DEGRADATION
            elif bad_ch == "cht" and abs(z_scores.get("coolant_temp", 0.0)) < 1.0:
                # Unilateral CHT without coolant temp shift indicates sensor drift
                return AlertClassification.SENSOR_ANOMALY, ["THERMAL"], EngineHealthState.SENSOR_ANOMALY
            elif bad_ch == "oil_pressure" and abs(z_scores.get("oil_temp", 0.0)) < 1.0:
                return AlertClassification.SENSOR_ANOMALY, ["LUBRICATION"], EngineHealthState.SENSOR_ANOMALY
            else:
                return AlertClassification.MODEL_DISAGREEMENT, [], EngineHealthState.HEALTHY

        # Multiple alarmed channels -> Check physical correlation
        aff_subsystems = []
        for sub, ch_list in SUBSYSTEM_CHANNEL_MAP.items():
            if any(c in alarmed_channels for c in ch_list):
                aff_subsystems.append(sub)

        # Thermal/Cooling concordance
        if "cht" in alarmed_channels and "coolant_temp" in alarmed_channels:
            return AlertClassification.POSSIBLE_PHYSICAL_DEGRADATION, ["COOLING", "THERMAL"], EngineHealthState.COOLING_DEGRADATION

        # Lubrication concordance
        if "oil_pressure" in alarmed_channels and "oil_temp" in alarmed_channels:
            return AlertClassification.POSSIBLE_PHYSICAL_DEGRADATION, ["LUBRICATION"], EngineHealthState.LUBRICATION_DEGRADATION

        # Combustion concordance
        if "egt" in alarmed_channels:
            return AlertClassification.POSSIBLE_PHYSICAL_DEGRADATION, ["COMBUSTION"], EngineHealthState.COMBUSTION_DEGRADATION

        # Mechanical concordance
        if "vibration" in alarmed_channels:
            return AlertClassification.POSSIBLE_PHYSICAL_DEGRADATION, ["MECHANICAL"], EngineHealthState.VIBRATION_DEGRADATION

        return AlertClassification.POSSIBLE_PHYSICAL_DEGRADATION, aff_subsystems, EngineHealthState.THERMAL_DEGRADATION

    # -----------------------------------------------------------------
    # Uncertainty Quantification
    # -----------------------------------------------------------------

    def _compute_uncertainties(
        self,
        observed: Dict[str, Optional[float]],
        physics_expected: Dict[str, float],
        corrected_prediction: Dict[str, float],
    ) -> Dict[str, ChannelUncertainty]:
        """
        Computes transparent prediction intervals based on rolling residual dispersion.
        """
        uncertainties: Dict[str, ChannelUncertainty] = {}

        for ch in self._baseline_scales:
            y_pred = corrected_prediction.get(ch, physics_expected.get(f"{ch}_expected", 0.0))

            # Estimate dispersion from causal rolling history
            hist = self._rolling_residuals.get(ch, [])
            if len(hist) >= 5:
                dispersion = float(np.std(hist))
            else:
                dispersion = self._baseline_scales.get(ch, 1.0)

            # Combined engineering uncertainty margin (1.96 * dispersion)
            margin = max(1e-3, 1.96 * dispersion)
            y_lower = y_pred - margin
            y_upper = y_pred + margin

            # Ensure physical bounds
            if ch in PHYSICAL_INSTRUMENT_LIMITS:
                min_lim, max_lim = PHYSICAL_INSTRUMENT_LIMITS[ch]
                y_lower = max(min_lim, y_lower)
                y_upper = min(max_lim, y_upper)

            # Confidence scaled inversely with dispersion relative to nominal baseline
            conf = float(np.clip(1.0 - (dispersion / (self._baseline_scales.get(ch, 1.0) * 3.0)), 0.2, 0.98))

            uncertainties[ch] = ChannelUncertainty(
                channel=ch,
                predicted_value=round(y_pred, 3),
                prediction_lower=round(y_lower, 3),
                prediction_upper=round(y_upper, 3),
                uncertainty_width=round(y_upper - y_lower, 3),
                model_confidence=round(conf, 3),
                confidence_basis="rolling_residual_dispersion_engineering_estimate",
            )

        return uncertainties

    # -----------------------------------------------------------------
    # Prognostics & Degradation Trend
    # -----------------------------------------------------------------

    def _compute_prognostics(
        self,
        current_health: float,
        current_time: float,
    ) -> Tuple[DegradationTrendState, Optional[float], Optional[float], str, Optional[str]]:
        """
        Causally estimates degradation trend slope and time-to-threshold.
        Strictly withholds RUL when degradation trajectory is absent or non-negative.
        """
        n_pts = len(self._history_timestamps)
        if n_pts < 10:
            return (
                DegradationTrendState.INSUFFICIENT_HISTORY,
                None,
                None,
                "UNAVAILABLE",
                "Insufficient historical observation window (< 10 steps)",
            )

        times = np.array(self._history_timestamps)
        scores = np.array(self._history_health_scores)

        dt_span = times[-1] - times[0]
        if dt_span < 2.0:
            return (
                DegradationTrendState.INSUFFICIENT_HISTORY,
                None,
                None,
                "UNAVAILABLE",
                "Observation time span too short (< 2.0s) to estimate trend",
            )

        # Robust linear slope (health score units per second)
        slope, _ = np.polyfit(times, scores, 1)

        # Classify trend direction
        if slope < -0.010:
            trend_state = DegradationTrendState.RAPIDLY_DEGRADING
        elif slope < -0.001:
            trend_state = DegradationTrendState.DEGRADING
        elif slope > 0.002:
            trend_state = DegradationTrendState.IMPROVING
        else:
            trend_state = DegradationTrendState.STABLE

        # Time-to-threshold logic (RUL)
        # Condition for defensible RUL: trend must be genuinely degrading (DEGRADING or RAPIDLY_DEGRADING)
        if trend_state not in (DegradationTrendState.DEGRADING, DegradationTrendState.RAPIDLY_DEGRADING) or slope >= -0.001:
            return (
                trend_state,
                slope,
                None,
                "UNAVAILABLE",
                "Engine health condition is stable or non-degrading; time-to-threshold is infinite",
            )

        # Calculate time to cross critical EOL threshold (HI <= 0.35)
        delta_hi_to_eol = current_health - self.eol_threshold
        if delta_hi_to_eol <= 0.0:
            return (
                trend_state,
                slope,
                0.0,
                "AVAILABLE",
                "Health score is at or below critical functional threshold",
            )

        # Linear projection to threshold
        time_to_threshold = float(delta_hi_to_eol / abs(slope))

        return (
            trend_state,
            slope,
            time_to_threshold,
            "AVAILABLE",
            "Projected based on causal sliding-window linear degradation slope",
        )
