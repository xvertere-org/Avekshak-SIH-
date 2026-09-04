"""
Residual preprocessing and data quality validation for Phase 7 Anomaly Detection.

Enforces strict feature hygiene:
- Consumes only normalized residuals.
- Prohibits substitution of raw physical channels for missing normalized residuals.
- Preserves NaNs without imputing zeros.
- Evaluates per-row feature validity against configurable minimum-valid-feature thresholds.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from anomaly_detection.schema import (
    PRIMARY_NORMALIZED_RESIDUAL_CHANNELS,
    RAW_RESIDUAL_CHANNELS,
)
from digital_twin.residuals import ResidualFrame


# Default minimum valid normalized features required to evaluate a row (4 out of 7)
DEFAULT_MIN_VALID_FEATURES: int = 4


class ResidualPreprocessor:
    """
    Validates and prepares residual telemetry for downstream anomaly detectors.
    """

    def __init__(
        self,
        feature_channels: Optional[List[str]] = None,
        min_valid_features: int = DEFAULT_MIN_VALID_FEATURES,
    ):
        """
        Args:
            feature_channels: List of normalized residual channel names to extract.
            min_valid_features: Minimum non-null features required for valid scoring.
        """
        self.feature_channels = list(feature_channels or PRIMARY_NORMALIZED_RESIDUAL_CHANNELS)
        self.min_valid_features = int(min_valid_features)

        # Prohibit dangerous configuration where raw physical channels might be passed as features
        for ch in self.feature_channels:
            if not ch.endswith("_norm_residual"):
                raise ValueError(
                    f"Invalid feature channel '{ch}'. Anomaly detection must operate "
                    f"strictly on normalized residuals ending in '_norm_residual'."
                )

    def extract_features(
        self,
        residuals: Union[ResidualFrame, pd.DataFrame],
    ) -> Tuple[pd.DataFrame, pd.Series, List[List[str]]]:
        """
        Extract normalized residual feature matrix and compute per-sample validity.

        Args:
            residuals: ResidualFrame or pandas DataFrame containing normalized residual columns.

        Returns:
            feature_df: DataFrame with exactly the selected normalized residual channels.
            valid_mask: Boolean Series indicating whether each row meets min_valid_features.
            missing_per_row: List of missing/NaN feature names for each row.
        """
        if isinstance(residuals, ResidualFrame):
            df = residuals.to_dataframe()
        elif isinstance(residuals, pd.DataFrame):
            df = residuals.copy()
        else:
            raise TypeError(f"Expected ResidualFrame or pd.DataFrame, got {type(residuals).__name__}")

        # Build feature DataFrame preserving NaNs (NEVER impute NaN to 0)
        feature_dict = {}
        for col in self.feature_channels:
            if col in df.columns:
                feature_dict[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
            else:
                # If a channel column does not exist at all, populate with NaN
                # STRICT REQUIREMENT: Do NOT substitute raw physical channel!
                feature_dict[col] = pd.Series(np.nan, index=df.index, dtype=float)

        feature_df = pd.DataFrame(feature_dict, index=df.index)

        # Count non-null, finite values per row
        valid_counts = feature_df.notna().sum(axis=1)
        valid_mask = valid_counts >= self.min_valid_features

        # Identify missing features per row
        is_missing = feature_df.isna()
        missing_per_row = [
            [col for col in self.feature_channels if is_missing.at[idx, col]]
            for idx in df.index
        ]

        return feature_df, valid_mask, missing_per_row

    def extract_provenance(
        self,
        residuals: Union[ResidualFrame, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Extract flight metadata, engine context, and quality provenance.
        Guarantees that fault labels/severity are NOT carried into detector features.
        """
        if isinstance(residuals, ResidualFrame):
            df = residuals.to_dataframe()
        else:
            df = residuals

        prov_cols = [
            "timestamp",
            "engine_id",
            "mission_id",
            "mission_phase",
            "source",
            "source_type",
            "quality_status",
        ]

        prov_df = pd.DataFrame(index=df.index)
        for col in prov_cols:
            if col in df.columns:
                prov_df[col] = df[col]
            else:
                # Fallback defaults for missing provenance
                if col == "timestamp":
                    prov_df[col] = np.arange(len(df), dtype=float)
                elif col == "engine_id":
                    prov_df[col] = "UNKNOWN_ENGINE"
                elif col == "mission_id":
                    prov_df[col] = "UNKNOWN_MISSION"
                elif col == "mission_phase":
                    prov_df[col] = "CRUISE"
                elif col == "source":
                    prov_df[col] = "residual_stream"
                elif col == "source_type":
                    prov_df[col] = "simulated"
                elif col == "quality_status":
                    prov_df[col] = "GOOD"

        return prov_df
