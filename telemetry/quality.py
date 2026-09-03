"""
Data Quality Pipeline and Assessment for SIH26054 Aero Piston Engine Digital Twin.

Provides non-destructive, deterministic quality evaluation:
- Timestamp integrity (monotonicity, ordering, interval regularities, duplicate detection)
- Missingness & Phase 4F sensor dropout identification (NaN preserved, never converted to 0)
- Physical validity vs. operational envelope warnings (severe fault excursions flagged as WARNING,
  not marked as INVALID data, and NEVER conflated with engine fault diagnosis)
- Machine-readable DataQualityReport generation with channel-level granular summaries
"""

import math
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Set
import numpy as np
import pandas as pd

from telemetry.ingestion import CanonicalTelemetryFrame, OPTIONAL_PHYSICAL_CHANNELS


class QualityStatus(str, Enum):
    """Data quality classification status."""
    VALID = "valid"              # Clean observation within nominal operational envelopes
    WARNING = "warning"          # Operational envelope excursion (e.g. high CHT during cooling fault)
    INVALID = "invalid"          # Physically impossible or corrupt value (e.g. negative RPM, Inf)
    MISSING = "missing"          # Sensor dropout or unrecorded value (NaN)


@dataclass
class QualityEnvelope:
    """
    Physical vs. Operational envelopes for a single telemetry channel.

    - physical_range: Absolute thermodynamic/physical limits (violations = INVALID).
    - warning_envelope: Nominal operational limits (violations = WARNING, e.g. active faults).
    """
    physical_min: float
    physical_max: float
    warning_min: float
    warning_max: float


# Default physical validity vs. operational warning boundaries (Tier A & physical thermodynamics)
DEFAULT_QUALITY_ENVELOPES: Dict[str, QualityEnvelope] = {
    "rpm": QualityEnvelope(
        physical_min=0.0, physical_max=7500.0,
        warning_min=1000.0, warning_max=5850.0
    ),
    "cht": QualityEnvelope(
        physical_min=-50.0, physical_max=300.0,
        warning_min=20.0, warning_max=150.0
    ),
    "egt": QualityEnvelope(
        physical_min=0.0, physical_max=1200.0,
        warning_min=400.0, warning_max=950.0
    ),
    "oil_temp": QualityEnvelope(
        physical_min=-50.0, physical_max=200.0,
        warning_min=20.0, warning_max=130.0
    ),
    "oil_pressure": QualityEnvelope(
        physical_min=0.0, physical_max=15.0,
        warning_min=0.8, warning_max=7.0
    ),
    "fuel_flow": QualityEnvelope(
        physical_min=0.0, physical_max=100.0,
        warning_min=0.5, warning_max=45.0
    ),
    "vibration": QualityEnvelope(
        physical_min=0.0, physical_max=20.0,
        warning_min=0.05, warning_max=3.5
    ),
    "altitude": QualityEnvelope(
        physical_min=-500.0, physical_max=15000.0,
        warning_min=0.0, warning_max=8000.0
    ),
    "ambient_temp": QualityEnvelope(
        physical_min=-80.0, physical_max=80.0,
        warning_min=-40.0, warning_max=55.0
    ),
    "throttle": QualityEnvelope(
        physical_min=0.0, physical_max=100.0,
        warning_min=0.0, warning_max=100.0
    ),
    "load": QualityEnvelope(
        physical_min=0.0, physical_max=120.0,
        warning_min=0.0, warning_max=100.0
    ),
}


@dataclass
class ChannelQualitySummary:
    """Summary of data quality checks for a single telemetry channel."""
    channel: str
    total_count: int = 0
    valid_count: int = 0
    warning_count: int = 0
    invalid_count: int = 0
    missing_count: int = 0
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    mean_value: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataQualityReport:
    """
    Machine-readable quality evaluation report for telemetry streams.
    Preserves granular issue counts without reducing everything to a single score.
    """
    total_samples: int
    valid_samples: int
    warning_samples: int
    invalid_samples: int
    missing_samples: int
    duplicate_samples: int
    out_of_order_samples: int
    bound_violations: int
    quality_score: float                  # Normalized [0.0, 1.0]
    sampling_interval_mean: Optional[float] = None
    sampling_interval_std: Optional[float] = None
    is_regular_sampling: bool = True
    nominal_dt_s: Optional[float] = 0.1
    channel_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    issues_detected: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DataQualityChecker:
    """
    Evaluates telemetry data quality across schema, timestamps, bounds, and missingness.
    Does not modify raw physical measurements.
    """

    def __init__(
        self,
        envelopes: Optional[Dict[str, QualityEnvelope]] = None,
        expected_dt_s: float = 0.1,
        dt_tolerance_pct: float = 10.0,
    ):
        """
        Initialize DataQualityChecker.

        Args:
            envelopes: Channel-specific physical vs warning envelopes.
            expected_dt_s: Expected nominal sampling interval in seconds (default 0.1s for 10 Hz).
            dt_tolerance_pct: Percentage tolerance for declaring sampling regular.
        """
        self.envelopes = envelopes or DEFAULT_QUALITY_ENVELOPES
        self.expected_dt_s = expected_dt_s
        self.dt_tolerance_pct = dt_tolerance_pct

    def assess(
        self,
        frame: CanonicalTelemetryFrame,
    ) -> Tuple[CanonicalTelemetryFrame, DataQualityReport]:
        """
        Perform complete non-destructive quality assessment.

        Returns:
            Tuple of:
            - Annotated CanonicalTelemetryFrame (adds 'quality_status' and 'missing_mask' columns)
            - Machine-readable DataQualityReport
        """
        df = frame.to_dataframe()
        n_samples = len(df)
        issues: List[str] = []

        if n_samples == 0:
            report = DataQualityReport(
                total_samples=0,
                valid_samples=0,
                warning_samples=0,
                invalid_samples=0,
                missing_samples=0,
                duplicate_samples=0,
                out_of_order_samples=0,
                bound_violations=0,
                quality_score=1.0,
                is_regular_sampling=True,
                issues_detected=["Dataset is empty"],
            )
            annotated_df = df.copy()
            annotated_df["quality_status"] = QualityStatus.VALID.value
            annotated_df["missing_mask"] = False
            return CanonicalTelemetryFrame(annotated_df, metadata=frame.metadata), report

        # Initialize row-level status tracker: default VALID
        row_status = np.full(n_samples, QualityStatus.VALID.value, dtype=object)
        row_has_missing = np.zeros(n_samples, dtype=bool)

        # 1. Timestamp validation
        dup_count = 0
        ooo_count = 0
        mean_dt = None
        std_dt = None
        is_regular = True

        if "timestamp" in df.columns:
            ts = df["timestamp"].to_numpy(dtype=float)

            # Check for non-monotonic / out-of-order timestamps
            diffs = np.diff(ts)
            ooo_indices = np.where(diffs < 0)[0] + 1
            ooo_count = len(ooo_indices)
            if ooo_count > 0:
                issues.append(f"Detected {ooo_count} out-of-order timestamp(s)")
                row_status[ooo_indices] = QualityStatus.INVALID.value

            # Check for duplicates across (engine_id, mission_id, timestamp)
            ident_cols = [c for c in ["engine_id", "mission_id", "timestamp"] if c in df.columns]
            dups = df.duplicated(subset=ident_cols, keep=False)
            dup_count = int(dups.sum())
            if dup_count > 0:
                issues.append(f"Detected {dup_count} duplicate sample(s)")
                # Only elevate to WARNING if current status is VALID (do not overwrite INVALID)
                elevate_dup = (row_status == QualityStatus.VALID.value) & dups.to_numpy()
                row_status[elevate_dup] = QualityStatus.WARNING.value

            # Sampling interval statistics
            if len(diffs) > 0:
                pos_diffs = diffs[diffs > 0]
                if len(pos_diffs) > 0:
                    mean_dt = float(np.mean(pos_diffs))
                    std_dt = float(np.std(pos_diffs))
                    tol = self.expected_dt_s * (self.dt_tolerance_pct / 100.0)
                    if abs(mean_dt - self.expected_dt_s) > tol or std_dt > tol:
                        is_regular = False
                        issues.append(f"Irregular sampling detected (mean dt={mean_dt:.4f}s, std={std_dt:.4f}s)")

        # 2. Channel-by-channel quality & envelope evaluation
        channel_summaries: Dict[str, Dict[str, Any]] = {}
        total_bound_violations = 0

        for col in df.columns:
            if col not in self.envelopes:
                continue

            series = df[col]
            arr = series.to_numpy(dtype=float)

            n_total = len(arr)
            is_nan = np.isnan(arr)
            is_inf = np.isinf(arr)

            missing_cnt = int(is_nan.sum())
            invalid_cnt = int(is_inf.sum())
            warning_cnt = 0
            valid_cnt = 0

            # Record missingness (e.g. sensor dropout from Phase 4F)
            if missing_cnt > 0:
                row_has_missing |= is_nan
                # For rows where status is still VALID, set to MISSING
                valid_mask = (row_status == QualityStatus.VALID.value)
                row_status[is_nan & valid_mask] = QualityStatus.MISSING.value

            # Check physical limits and warning envelopes
            env = self.envelopes[col]
            finite_mask = ~(is_nan | is_inf)

            if finite_mask.any():
                finite_vals = arr[finite_mask]
                min_v = float(np.min(finite_vals))
                max_v = float(np.max(finite_vals))
                mean_v = float(np.mean(finite_vals))

                # Physical impossibility check -> INVALID
                phys_invalid = (finite_vals < env.physical_min) | (finite_vals > env.physical_max)
                n_phys_invalid = int(phys_invalid.sum())
                invalid_cnt += n_phys_invalid

                if n_phys_invalid > 0:
                    orig_indices = np.where(finite_mask)[0][phys_invalid]
                    row_status[orig_indices] = QualityStatus.INVALID.value
                    issues.append(f"Channel '{col}' has {n_phys_invalid} physically impossible value(s)")

                # Operational envelope warning check (does NOT invalidate data)
                # Only evaluate on physically valid finite values
                phys_valid_mask = ~phys_invalid
                warn_mask = (
                    (finite_vals[phys_valid_mask] < env.warning_min) |
                    (finite_vals[phys_valid_mask] > env.warning_max)
                )
                warning_cnt = int(warn_mask.sum())
                total_bound_violations += warning_cnt + n_phys_invalid

                if warning_cnt > 0:
                    orig_warn_indices = np.where(finite_mask)[0][phys_valid_mask][warn_mask]
                    # Only elevate to WARNING if current status is VALID
                    elevate_mask = (row_status[orig_warn_indices] == QualityStatus.VALID.value)
                    row_status[orig_warn_indices[elevate_mask]] = QualityStatus.WARNING.value

                valid_cnt = int(phys_valid_mask.sum()) - warning_cnt
            else:
                min_v, max_v, mean_v = None, None, None

            c_summary = ChannelQualitySummary(
                channel=col,
                total_count=n_total,
                valid_count=valid_cnt,
                warning_count=warning_cnt,
                invalid_count=invalid_cnt,
                missing_count=missing_cnt,
                min_value=min_v,
                max_value=max_v,
                mean_value=mean_v,
            )
            channel_summaries[col] = c_summary.to_dict()

        # 3. Aggregate totals
        n_valid = int((row_status == QualityStatus.VALID.value).sum())
        n_warn = int((row_status == QualityStatus.WARNING.value).sum())
        n_invalid = int((row_status == QualityStatus.INVALID.value).sum())
        n_missing = int((row_status == QualityStatus.MISSING.value).sum())

        # Quality score: penalizes invalid and missing heavily, warnings lightly
        score = 1.0 - (
            (n_invalid * 1.0 + n_missing * 0.5 + n_warn * 0.1) / max(1, n_samples)
        )
        score = max(0.0, min(1.0, round(score, 4)))

        report = DataQualityReport(
            total_samples=n_samples,
            valid_samples=n_valid,
            warning_samples=n_warn,
            invalid_samples=n_invalid,
            missing_samples=n_missing,
            duplicate_samples=dup_count,
            out_of_order_samples=ooo_count,
            bound_violations=total_bound_violations,
            quality_score=score,
            sampling_interval_mean=round(mean_dt, 5) if mean_dt is not None else None,
            sampling_interval_std=round(std_dt, 5) if std_dt is not None else None,
            is_regular_sampling=is_regular,
            nominal_dt_s=self.expected_dt_s,
            channel_summaries=channel_summaries,
            issues_detected=issues,
        )

        annotated_df = df.copy()
        annotated_df["quality_status"] = row_status
        annotated_df["missing_mask"] = row_has_missing

        return CanonicalTelemetryFrame(annotated_df, metadata=frame.metadata), report
