"""
Residual Generation and Storage for SIH26054 Digital Twin.

Computes raw (observed - expected) and normalized residuals across all core telemetry channels.
Serves as the primary observable interface for Phase 5 Health Assessment and Phase 6 PHM.

Rules:
- Non-destructive: preserves raw observations, expected values, and explicit physical units.
- Missingness preservation: observed NaN (e.g. sensor dropout) produces NaN residual.
- Quality-aware: rejects STALE, OUT_OF_RANGE, DUPLICATE_TIMESTAMP, etc. with explicit reasons.
- Observability-aware: respects CANONICAL_OBSERVABILITY_CATALOG contracts.
- Frozen calibration: immutable reference scales (ENGINEERING_HEURISTIC or FROZEN_SYNTHETIC_BASELINE_MAD).
- Primary subsystem ownership: enforces single primary ownership per channel to eliminate double-counting.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional, Union, Set, Tuple
import math
import numpy as np
import pandas as pd

from telemetry.ingestion import CanonicalTelemetryFrame, OPTIONAL_PHYSICAL_CHANNELS
from telemetry.schema import TelemetryRecord
from digital_twin.quality import DataQualityStatus, TelemetryQualityReport
from digital_twin.observability import CANONICAL_OBSERVABILITY_CATALOG, ObservabilityType


# =====================================================================
# Channel Sets, Mappings, and Subsystem Ownership Architecture
# =====================================================================

# Primary engine-level channels (exactly 9 channels evaluated for engine-level health)
PRIMARY_RESIDUAL_CHANNELS: List[str] = [
    "rpm",
    "map_bar",
    "fuel_flow",
    "cht",
    "coolant_temp",
    "oil_temp",
    "oil_pressure",
    "egt",
    "vibration",
]

# Discrete per-cylinder channels (discrete localization evidence; zero additional engine votes)
CYLINDER_RESIDUAL_CHANNELS: List[str] = [
    "cht_cyl1",
    "cht_cyl2",
    "cht_cyl3",
    "cht_cyl4",
    "egt_cyl1",
    "egt_cyl2",
    "egt_cyl3",
    "egt_cyl4",
]

# Secondary / model-consistency channels (zero primary subsystem ownership, zero engine vote)
SECONDARY_MODEL_CHANNELS: List[str] = [
    "charge_air_temp",
]

# Authoritative Primary Subsystem Ownership (Enforces single-vote invariant)
PRIMARY_SUBSYSTEM_MAP: Dict[str, str] = {
    "rpm": "ROTATIONAL",
    "map_bar": "ROTATIONAL",
    "fuel_flow": "FUEL",
    "cht": "THERMAL",
    "coolant_temp": "THERMAL",
    "oil_temp": "THERMAL",
    "oil_pressure": "LUBRICATION",
    "egt": "COMBUSTION",
    "vibration": "MECHANICAL",
}

# Secondary Contextual Links (Documentation / diagnostic context only; ZERO engine weight)
SECONDARY_CONTEXT_MAP: Dict[str, List[str]] = {
    "rpm": ["MECHANICAL"],
    "map_bar": ["COMBUSTION"],
    "fuel_flow": ["COMBUSTION"],
    "oil_temp": ["LUBRICATION"],
    "egt": ["THERMAL"],
    "vibration": ["ROTATIONAL"],
    "charge_air_temp": ["ROTATIONAL", "THERMAL"],
}

# Physical units map
CHANNEL_UNITS_MAP: Dict[str, str] = {
    "rpm": "RPM",
    "map_bar": "bar",
    "fuel_flow": "L/h",
    "cht": "°C",
    "coolant_temp": "°C",
    "oil_temp": "°C",
    "oil_pressure": "bar",
    "egt": "°C",
    "vibration": "g",
    "cht_cyl1": "°C",
    "cht_cyl2": "°C",
    "cht_cyl3": "°C",
    "cht_cyl4": "°C",
    "egt_cyl1": "°C",
    "egt_cyl2": "°C",
    "egt_cyl3": "°C",
    "egt_cyl4": "°C",
    "charge_air_temp": "°C",
}

# Default engineering heuristic reference scales
DEFAULT_ENGINEERING_SCALES: Dict[str, float] = {
    "rpm": 100.0,            # RPM
    "map_bar": 0.05,         # bar
    "fuel_flow": 2.0,        # L/h
    "cht": 10.0,             # °C
    "coolant_temp": 6.0,     # °C
    "oil_temp": 8.0,         # °C
    "oil_pressure": 0.50,    # bar
    "egt": 25.0,             # °C
    "vibration": 0.20,       # g
    "cht_cyl1": 12.0,        # °C
    "cht_cyl2": 12.0,        # °C
    "cht_cyl3": 12.0,        # °C
    "cht_cyl4": 12.0,        # °C
    "egt_cyl1": 30.0,        # °C
    "egt_cyl2": 30.0,        # °C
    "egt_cyl3": 30.0,        # °C
    "egt_cyl4": 30.0,        # °C
    "charge_air_temp": 5.0,  # °C
}

# Legacy scale factors and channels for backward compatibility with Phase 1-3 tests
DEFAULT_RESIDUAL_SCALES: Dict[str, float] = {
    "rpm": 100.0,
    "cht": 10.0,
    "egt": 20.0,
    "oil_temp": 10.0,
    "oil_pressure": 0.5,
    "fuel_flow": 2.0,
    "vibration": 0.2,
}

SUPPORTED_RESIDUAL_CHANNELS: List[str] = [
    "rpm",
    "cht",
    "egt",
    "oil_temp",
    "oil_pressure",
    "fuel_flow",
    "vibration",
]


# =====================================================================
# Typed Data Models & Frozen Scale Calibration Container
# =====================================================================

@dataclass(frozen=True)
class FrozenScaleCalibration:
    """
    Immutable calibration container holding frozen scale factors and dataset provenance.
    Lifecycle:
        calibration dataset window -> calculate MAD -> freeze calibration -> evaluation data.
    Never mutates scales during streaming or evaluation.
    """
    calibration_id: str
    calibration_mode: str             # "ENGINEERING_HEURISTIC" or "FROZEN_SYNTHETIC_BASELINE_MAD"
    dataset_type: str                 # "NONE_HEURISTIC" or "SYNTHETIC_GREY_BOX_HEALTHY"
    scales: Dict[str, float]
    provenance: Dict[str, str]        # Provenance classification per channel
    sample_count: Optional[int] = None
    calibration_dataset_window: Optional[str] = None
    created_at: str = ""

    def get_scale(self, channel: str, default: float = 1.0) -> float:
        """Retrieve frozen normalization scale factor for a channel with positive safety floor."""
        val = self.scales.get(channel, default)
        return max(1e-4, float(val))


def create_default_calibration(calibration_id: str = "DEFAULT_HEURISTIC_V1") -> FrozenScaleCalibration:
    """Factory creating default immutable engineering heuristic scale calibration."""
    prov = {ch: "ENGINEERING_HEURISTIC" for ch in DEFAULT_ENGINEERING_SCALES}
    return FrozenScaleCalibration(
        calibration_id=calibration_id,
        calibration_mode="ENGINEERING_HEURISTIC",
        dataset_type="NONE_HEURISTIC",
        scales=dict(DEFAULT_ENGINEERING_SCALES),
        provenance=prov,
        created_at="2026-09-14T12:00:00Z",
    )


@dataclass
class PhysicalResidual:
    """
    A typed physical residual preserving engineering units, observation/prediction values,
    and quality/observability audit metadata.
    """
    channel: str
    timestamp: float
    observed_value: Optional[float]
    predicted_value: Optional[float]
    raw_residual: float               # observed - predicted (or NaN if invalid)
    normalized_residual: float        # raw_residual / scale (or NaN if invalid)
    units: str
    quality_status: str               # e.g. "VALID", "STALE", "OUT_OF_RANGE", "MISSING"
    observability_status: str         # e.g. "DIRECTLY_OBSERVED", "UNOBSERVED", "UNAVAILABLE"
    valid: bool
    reason: str = ""
    scale_source: str = "ENGINEERING_HEURISTIC"
    scale_value: float = 1.0
    primary_subsystem: str = "UNASSIGNED"
    is_primary_engine_vote: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "channel": self.channel,
            "timestamp": self.timestamp,
            "observed_value": self.observed_value,
            "predicted_value": self.predicted_value,
            "raw_residual": round(self.raw_residual, 4) if not math.isnan(self.raw_residual) else None,
            "normalized_residual": round(self.normalized_residual, 4) if not math.isnan(self.normalized_residual) else None,
            "units": self.units,
            "quality_status": self.quality_status,
            "observability_status": self.observability_status,
            "valid": self.valid,
            "reason": self.reason,
            "scale_source": self.scale_source,
            "scale_value": self.scale_value,
            "primary_subsystem": self.primary_subsystem,
            "is_primary_engine_vote": self.is_primary_engine_vote,
        }


@dataclass
class CylinderResiduals:
    """
    Discrete per-cylinder runner residuals, spreads, and imbalance metrics.
    Preserves cylinder localization without double-voting against overall engine health.
    """
    cht_runner_residuals: List[float]    # Cylinders 1-4 raw CHT residuals in °C
    egt_runner_residuals: List[float]    # Cylinders 1-4 raw EGT residuals in °C
    cht_spread_c: float                  # max(r_cht) - min(r_cht)
    egt_spread_c: float                  # max(r_egt) - min(r_egt)
    cht_imbalance_max_c: float           # max(|r_cht,i - mean(r_cht)|)
    egt_imbalance_max_c: float           # max(|r_egt,i - mean(r_egt)|)
    valid_cylinder_count: int            # count of finite runner pairs


@dataclass
class ResidualVector:
    """
    Sample-level container holding all channel residuals, cylinder metrics, and aggregations.
    """
    timestamp: float
    residuals: Dict[str, PhysicalResidual]
    cylinder_residuals: CylinderResiduals
    primary_channel_count: int = 9
    valid_primary_count: int = 0
    coverage_fraction: float = 0.0
    mean_abs_normalized_residual: float = float("nan")
    rms_normalized_residual: float = float("nan")
    max_abs_normalized_residual: float = float("nan")
    subsystem_rms: Dict[str, float] = field(default_factory=dict)

    def get_residual(self, channel: str) -> Optional[PhysicalResidual]:
        """Retrieve PhysicalResidual object for a channel."""
        return self.residuals.get(channel)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "primary_channel_count": self.primary_channel_count,
            "valid_primary_count": self.valid_primary_count,
            "coverage_fraction": round(self.coverage_fraction, 4),
            "mean_abs_normalized_residual": round(self.mean_abs_normalized_residual, 4) if not math.isnan(self.mean_abs_normalized_residual) else None,
            "rms_normalized_residual": round(self.rms_normalized_residual, 4) if not math.isnan(self.rms_normalized_residual) else None,
            "max_abs_normalized_residual": round(self.max_abs_normalized_residual, 4) if not math.isnan(self.max_abs_normalized_residual) else None,
            "subsystem_rms": {k: round(v, 4) for k, v in self.subsystem_rms.items()},
            "residuals": {k: v.to_dict() for k, v in self.residuals.items()},
        }


# =====================================================================
# Quality-Aware & Observability-Gated Residual Generator
# =====================================================================

class QualityAwareResidualGenerator:
    """
    Authoritative Phase 5 residual generator.
    Enforces:
    1. Temporal and physical quality filtering via TelemetryQualityReport.
    2. Observability constraints via CANONICAL_OBSERVABILITY_CATALOG.
    3. Frozen immutable reference scales.
    4. Anti-double counting via Primary Subsystem Ownership.
    """

    def __init__(
        self,
        calibration: Optional[FrozenScaleCalibration] = None,
        epsilon: float = 1e-4,
    ):
        self.calibration = calibration or create_default_calibration()
        self.epsilon = epsilon

    def generate(
        self,
        telemetry: Union[TelemetryRecord, Dict[str, Any]],
        expected: Dict[str, float],
        quality_report: Optional[TelemetryQualityReport] = None,
    ) -> ResidualVector:
        """
        Compute quality-audited, observability-constrained residual vector for a single sample.

        Args:
            telemetry: TelemetryRecord or dictionary of sensor observations.
            expected: Dictionary of twin predicted/expected values (e.g. 'rpm_expected', etc.).
            quality_report: Optional TelemetryQualityReport from Phase 3 quality layer.

        Returns:
            ResidualVector containing all channel residuals, cylinder spread, and aggregations.
        """
        # Extract timestamp
        if isinstance(telemetry, TelemetryRecord):
            timestamp = float(telemetry.timestamp)
        elif isinstance(telemetry, dict):
            timestamp = float(telemetry.get("timestamp", 0.0))
        else:
            timestamp = 0.0

        # Build quality status map
        quality_map: Dict[str, str] = {}
        if quality_report is not None:
            ch_dict = getattr(quality_report, "channel_reports", getattr(quality_report, "channels", {}))
            if isinstance(ch_dict, dict):
                for ch_name, ch_q in ch_dict.items():
                    quality_map[ch_name] = ch_q.status.value if hasattr(ch_q.status, "value") else str(ch_q.status)

        # Map telemetry values
        all_channels = (
            PRIMARY_RESIDUAL_CHANNELS
            + CYLINDER_RESIDUAL_CHANNELS
            + SECONDARY_MODEL_CHANNELS
        )

        residual_map: Dict[str, PhysicalResidual] = {}
        valid_primary_norm_res: List[float] = []
        subsystem_norm_res: Dict[str, List[float]] = {
            "THERMAL": [],
            "LUBRICATION": [],
            "FUEL": [],
            "COMBUSTION": [],
            "MECHANICAL": [],
            "ROTATIONAL": [],
        }

        for ch in all_channels:
            # 1. Observability taxonomy check
            obs_entry = CANONICAL_OBSERVABILITY_CATALOG.get(ch)
            if obs_entry is None:
                # Check for cylinder channel variants in catalog
                if ch.startswith("cht_cyl"):
                    obs_entry = CANONICAL_OBSERVABILITY_CATALOG.get(f"{ch}_c")
                elif ch.startswith("egt_cyl"):
                    obs_entry = CANONICAL_OBSERVABILITY_CATALOG.get(f"{ch}_c")

            if obs_entry is not None:
                obs_type = obs_entry.observability_type.value if hasattr(obs_entry.observability_type, "value") else str(obs_entry.observability_type)
            else:
                obs_type = ObservabilityType.DIRECTLY_OBSERVED.value

            # 2. Extract observed value
            obs_val: Optional[float] = None
            if isinstance(telemetry, TelemetryRecord):
                raw = getattr(telemetry, ch, None)
                if raw is not None:
                    try:
                        f_val = float(raw)
                        obs_val = f_val if not math.isnan(f_val) else None
                    except (ValueError, TypeError):
                        obs_val = None
            elif isinstance(telemetry, dict):
                raw = telemetry.get(ch, None)
                if raw is not None:
                    try:
                        f_val = float(raw)
                        obs_val = f_val if not math.isnan(f_val) else None
                    except (ValueError, TypeError):
                        obs_val = None

            # If channel is a primary residual channel with sensor telemetry, it is directly observed
            if ch in PRIMARY_RESIDUAL_CHANNELS and obs_val is not None:
                is_observable = True
                obs_type = ObservabilityType.DIRECTLY_OBSERVED.value
            else:
                is_observable = obs_type in (
                    ObservabilityType.DIRECTLY_OBSERVED.value,
                    ObservabilityType.INDIRECTLY_OBSERVED.value,
                )

            # 3. Extract expected value
            exp_val: Optional[float] = None
            for exp_key in (f"{ch}_expected", ch, f"{ch}_pred"):
                if exp_key in expected:
                    try:
                        f_val = float(expected[exp_key])
                        if not math.isnan(f_val):
                            exp_val = f_val
                            break
                    except (ValueError, TypeError):
                        pass

            # 4. Determine Data Quality Status
            q_status = quality_map.get(ch, DataQualityStatus.VALID.value)

            # Fallback checks if quality_report was not explicitly passed
            if obs_val is None and q_status == DataQualityStatus.VALID.value:
                q_status = DataQualityStatus.MISSING.value

            # 5. Evaluate validity and compute residual
            primary_sub = PRIMARY_SUBSYSTEM_MAP.get(ch, "UNASSIGNED")
            is_primary_vote = ch in PRIMARY_RESIDUAL_CHANNELS
            units = CHANNEL_UNITS_MAP.get(ch, "")
            scale_val = self.calibration.get_scale(ch)
            scale_src = self.calibration.provenance.get(ch, self.calibration.calibration_mode)

            valid = False
            reason = ""
            raw_res = float("nan")
            norm_res = float("nan")

            if not is_observable:
                valid = False
                reason = f"UNOBSERVABLE: {obs_type}"
            elif q_status != DataQualityStatus.VALID.value:
                valid = False
                reason = f"QUALITY_INVALID: {q_status}"
            elif obs_val is None:
                valid = False
                reason = "OBSERVATION_NONE_OR_NAN"
            elif exp_val is None:
                valid = False
                reason = "PREDICTION_NONE_OR_NAN"
            else:
                # Valid physical residual: observed - expected
                valid = True
                raw_res = round(obs_val - exp_val, 4)
                norm_res = round(raw_res / max(self.epsilon, scale_val), 4)

            phys_res = PhysicalResidual(
                channel=ch,
                timestamp=timestamp,
                observed_value=obs_val,
                predicted_value=exp_val,
                raw_residual=raw_res,
                normalized_residual=norm_res,
                units=units,
                quality_status=q_status,
                observability_status=obs_type,
                valid=valid,
                reason=reason,
                scale_source=scale_src,
                scale_value=scale_val,
                primary_subsystem=primary_sub,
                is_primary_engine_vote=is_primary_vote,
            )
            residual_map[ch] = phys_res

            # Accumulate primary vote statistics
            if is_primary_vote and valid:
                valid_primary_norm_res.append(norm_res)
                if primary_sub in subsystem_norm_res:
                    subsystem_norm_res[primary_sub].append(norm_res)

        # 6. Evaluate Per-Cylinder Metrics
        cht_runners = [
            residual_map[f"cht_cyl{k}"].raw_residual
            if residual_map[f"cht_cyl{k}"].valid
            else float("nan")
            for k in range(1, 5)
        ]
        egt_runners = [
            residual_map[f"egt_cyl{k}"].raw_residual
            if residual_map[f"egt_cyl{k}"].valid
            else float("nan")
            for k in range(1, 5)
        ]

        valid_cht_runners = [r for r in cht_runners if not math.isnan(r)]
        valid_egt_runners = [r for r in egt_runners if not math.isnan(r)]

        if len(valid_cht_runners) >= 2:
            cht_spread = max(valid_cht_runners) - min(valid_cht_runners)
            cht_mean = sum(valid_cht_runners) / len(valid_cht_runners)
            cht_imb = max(abs(r - cht_mean) for r in valid_cht_runners)
        else:
            cht_spread = float("nan")
            cht_imb = float("nan")

        if len(valid_egt_runners) >= 2:
            egt_spread = max(valid_egt_runners) - min(valid_egt_runners)
            egt_mean = sum(valid_egt_runners) / len(valid_egt_runners)
            egt_imb = max(abs(r - egt_mean) for r in valid_egt_runners)
        else:
            egt_spread = float("nan")
            egt_imb = float("nan")

        cyl_residuals = CylinderResiduals(
            cht_runner_residuals=cht_runners,
            egt_runner_residuals=egt_runners,
            cht_spread_c=round(cht_spread, 2) if not math.isnan(cht_spread) else float("nan"),
            egt_spread_c=round(egt_spread, 2) if not math.isnan(egt_spread) else float("nan"),
            cht_imbalance_max_c=round(cht_imb, 2) if not math.isnan(cht_imb) else float("nan"),
            egt_imbalance_max_c=round(egt_imb, 2) if not math.isnan(egt_imb) else float("nan"),
            valid_cylinder_count=min(len(valid_cht_runners), len(valid_egt_runners)),
        )

        # 7. Compute Aggregate Primary Metrics
        valid_primary_count = len(valid_primary_norm_res)
        coverage = valid_primary_count / 9.0

        if valid_primary_count > 0:
            abs_z = [abs(z) for z in valid_primary_norm_res]
            mean_abs_z = float(np.mean(abs_z))
            rms_z = float(np.sqrt(np.mean([z ** 2 for z in valid_primary_norm_res])))
            max_abs_z = float(np.max(abs_z))
        else:
            mean_abs_z = float("nan")
            rms_z = float("nan")
            max_abs_z = float("nan")

        sub_rms: Dict[str, float] = {}
        for sub_name, z_list in subsystem_norm_res.items():
            if z_list:
                sub_rms[sub_name] = float(np.sqrt(np.mean([z ** 2 for z in z_list])))
            else:
                sub_rms[sub_name] = float("nan")

        return ResidualVector(
            timestamp=timestamp,
            residuals=residual_map,
            cylinder_residuals=cyl_residuals,
            primary_channel_count=9,
            valid_primary_count=valid_primary_count,
            coverage_fraction=coverage,
            mean_abs_normalized_residual=mean_abs_z,
            rms_normalized_residual=rms_z,
            max_abs_normalized_residual=max_abs_z,
            subsystem_rms=sub_rms,
        )

    @staticmethod
    def calibrate_from_dataframe(
        df_healthy: pd.DataFrame,
        calibration_id: str = "SYNTHETIC_HEALTHY_CAL_V1",
        window_name: str = "STEADY_CRUISE_CALIBRATION_WINDOW",
        epsilon: float = 1e-4,
    ) -> FrozenScaleCalibration:
        """
        Derive robust frozen scale calibration from an audited synthetic healthy baseline dataset.
        Estimator: sigma = max(epsilon, 1.4826 * MAD).
        Lifecycle: Calibrate -> Freeze -> Return read-only FrozenScaleCalibration.
        """
        all_channels = (
            PRIMARY_RESIDUAL_CHANNELS
            + CYLINDER_RESIDUAL_CHANNELS
            + SECONDARY_MODEL_CHANNELS
        )

        scales: Dict[str, float] = {}
        prov: Dict[str, str] = {}

        for ch in all_channels:
            res_col = f"{ch}_residual"
            if res_col in df_healthy.columns:
                series = pd.to_numeric(df_healthy[res_col], errors="coerce").dropna()
            elif ch in df_healthy.columns and f"{ch}_expected" in df_healthy.columns:
                series = (
                    pd.to_numeric(df_healthy[ch], errors="coerce")
                    - pd.to_numeric(df_healthy[f"{ch}_expected"], errors="coerce")
                ).dropna()
            else:
                scales[ch] = DEFAULT_ENGINEERING_SCALES.get(ch, 1.0)
                prov[ch] = "ENGINEERING_HEURISTIC"
                continue

            if len(series) >= 20:
                med = float(np.median(series))
                mad = float(np.median(np.abs(series - med)))
                sigma_robust = max(epsilon, 1.4826 * mad)
                scales[ch] = round(sigma_robust, 4)
                prov[ch] = "FROZEN_SYNTHETIC_BASELINE_MAD"
            else:
                scales[ch] = DEFAULT_ENGINEERING_SCALES.get(ch, 1.0)
                prov[ch] = "ENGINEERING_HEURISTIC"

        return FrozenScaleCalibration(
            calibration_id=calibration_id,
            calibration_mode="FROZEN_SYNTHETIC_BASELINE_MAD",
            dataset_type="SYNTHETIC_GREY_BOX_HEALTHY",
            scales=scales,
            provenance=prov,
            sample_count=len(df_healthy),
            calibration_dataset_window=window_name,
            created_at="2026-09-14T12:00:00Z",
        )


# =====================================================================
# Backward Compatibility: ResidualFrame and Legacy ResidualGenerator
# =====================================================================

class ResidualFrame:
    """
    Tabular container pairing observed telemetry, expected states, and calculated residuals.
    Maintains 100% backward compatibility with Phase 1-3 test suite.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self._data = data.copy()
        self._metadata = dict(metadata) if metadata is not None else {}

    @property
    def data(self) -> pd.DataFrame:
        """Access underlying DataFrame."""
        return self._data

    @property
    def metadata(self) -> Dict[str, Any]:
        """Metadata and residual configuration dictionary."""
        return self._metadata

    @property
    def residual_channels(self) -> List[str]:
        """List of channels that have residuals present in this frame."""
        return [c for c in SUPPORTED_RESIDUAL_CHANNELS if f"{c}_residual" in self._data.columns]

    def to_dataframe(self) -> pd.DataFrame:
        """Return a copy of the underlying DataFrame."""
        return self._data.copy()

    def to_records(self) -> List[Dict[str, Any]]:
        """Return frame rows as a list of dictionaries."""
        return self._data.to_dict(orient="records")

    def copy(self) -> "ResidualFrame":
        """Return a deep copy."""
        return ResidualFrame(data=self._data.copy(), metadata=dict(self._metadata))

    def get_residual(self, channel: str) -> pd.Series:
        """Retrieve raw residual series for a channel."""
        col = f"{channel}_residual"
        if col not in self._data.columns:
            raise KeyError(f"Residual column '{col}' not found in ResidualFrame.")
        return self._data[col]

    def get_normalized_residual(self, channel: str) -> pd.Series:
        """Retrieve normalized residual series for a channel."""
        col = f"{channel}_norm_residual"
        if col not in self._data.columns:
            raise KeyError(f"Normalized residual column '{col}' not found in ResidualFrame.")
        return self._data[col]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key: Any) -> Any:
        return self._data[key]

    def __repr__(self) -> str:
        rows = len(self._data)
        channels = len(self.residual_channels)
        return f"<ResidualFrame: {rows} samples, {channels} residual channels evaluated>"


class ResidualGenerator:
    """
    Computes raw and normalized residuals between observed telemetry and expected states.
    Maintains full backward compatibility with Phase 1-3 tests and anomaly detection.
    """

    def __init__(
        self,
        scale_factors: Optional[Dict[str, float]] = None,
        epsilon: float = 1e-6,
    ):
        self.scale_factors = scale_factors or DEFAULT_RESIDUAL_SCALES
        self.epsilon = epsilon

    def compute_residuals(
        self,
        observed: Union[CanonicalTelemetryFrame, pd.DataFrame],
        expected: pd.DataFrame,
    ) -> ResidualFrame:
        """
        Generate residuals from observed telemetry and expected states.
        """
        if isinstance(observed, CanonicalTelemetryFrame):
            df_obs = observed.to_dataframe().copy()
            meta = dict(observed.metadata)
        else:
            df_obs = observed.copy()
            meta = {}

        df_exp = expected.copy()

        if len(df_obs) != len(df_exp):
            raise ValueError(
                f"Row count mismatch between observed ({len(df_obs)}) and expected ({len(df_exp)}) dataframes."
            )

        result_df = pd.DataFrame(index=df_obs.index)

        # Preserve flight context columns
        id_cols = [
            "timestamp",
            "engine_id",
            "mission_id",
            "mission_phase",
            "altitude",
            "ambient_temp",
            "throttle",
            "load",
            "fault_type",
            "fault_severity",
            "source",
            "source_type",
            "simulation_version",
            "quality_status",
            "missing_mask",
        ]
        for col in id_cols:
            if col in df_obs.columns:
                result_df[col] = df_obs[col]

        # Evaluate legacy channels
        for ch in SUPPORTED_RESIDUAL_CHANNELS:
            if ch not in df_obs.columns:
                continue

            obs_series = pd.to_numeric(df_obs[ch], errors="coerce").astype(float)
            result_df[ch] = obs_series

            if f"{ch}_expected" in df_exp.columns:
                exp_series = pd.to_numeric(df_exp[f"{ch}_expected"], errors="coerce").astype(float)
            elif ch in df_exp.columns:
                exp_series = pd.to_numeric(df_exp[ch], errors="coerce").astype(float)
            else:
                continue

            result_df[f"{ch}_expected"] = exp_series

            raw_residual = obs_series - exp_series
            result_df[f"{ch}_residual"] = raw_residual.round(4)

            scale = max(self.epsilon, self.scale_factors.get(ch, 1.0))
            norm_residual = raw_residual / scale
            result_df[f"{ch}_norm_residual"] = norm_residual.round(4)

        if "order_1x_freq_hz" in df_exp.columns:
            result_df["order_1x_freq_hz_expected"] = df_exp["order_1x_freq_hz"]
        if "order_2x_freq_hz" in df_exp.columns:
            result_df["order_2x_freq_hz_expected"] = df_exp["order_2x_freq_hz"]

        meta["residual_scales"] = dict(self.scale_factors)
        return ResidualFrame(result_df, metadata=meta)
