"""
Residual Generation and Storage for SIH26054 Digital Twin.

Computes raw (observed - expected) and normalized residuals across all core telemetry channels.
Serves as the primary observable interface for Phase 7 Anomaly Detection and PHM.

Rules:
- Non-destructive: preserves raw observations and expected values.
- Missingness preservation: observed NaN (e.g. sensor dropout) produces NaN residual.
- Configurable reference scales for normalized residuals without whole-dataset standardization.
"""

from typing import Dict, Any, List, Optional, Union, Set
import numpy as np
import pandas as pd

from telemetry.ingestion import CanonicalTelemetryFrame, OPTIONAL_PHYSICAL_CHANNELS


# Default operational scale factors for normalized residual calculation
# (nominal operating range variation per channel, avoiding division by near-zero)
DEFAULT_RESIDUAL_SCALES: Dict[str, float] = {
    "rpm": 100.0,            # RPM
    "cht": 10.0,             # °C
    "egt": 20.0,             # °C
    "oil_temp": 10.0,        # °C
    "oil_pressure": 0.5,     # bar
    "fuel_flow": 2.0,        # L/h
    "vibration": 0.2,        # g
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


class ResidualFrame:
    """
    Tabular container pairing observed telemetry, expected states, and calculated residuals.
    Wraps a pandas DataFrame with schema-aware helper accessors.
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
    """

    def __init__(
        self,
        scale_factors: Optional[Dict[str, float]] = None,
        epsilon: float = 1e-6,
    ):
        """
        Initialize ResidualGenerator.

        Args:
            scale_factors: Optional custom normalization scales per channel.
            epsilon: Small floor preventing division by zero during normalization.
        """
        self.scale_factors = scale_factors or DEFAULT_RESIDUAL_SCALES
        self.epsilon = epsilon

    def compute_residuals(
        self,
        observed: Union[CanonicalTelemetryFrame, pd.DataFrame],
        expected: pd.DataFrame,
    ) -> ResidualFrame:
        """
        Generate residuals from observed telemetry and expected states.

        Args:
            observed: CanonicalTelemetryFrame or DataFrame of actual observations.
            expected: DataFrame containing '<channel>_expected' or plain channel columns.

        Returns:
            ResidualFrame containing observed, expected, raw residuals, and normalized residuals.
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

        # 1. Preserve flight identifiers, timestamps, and context
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

        # 2. For each supported channel, compute observed, expected, residual, and normalized residual
        for ch in SUPPORTED_RESIDUAL_CHANNELS:
            # Check if observed channel exists
            if ch not in df_obs.columns:
                continue

            obs_series = pd.to_numeric(df_obs[ch], errors="coerce").astype(float)
            result_df[ch] = obs_series

            # Retrieve expected series: check for 'ch_expected' then 'ch'
            if f"{ch}_expected" in df_exp.columns:
                exp_series = pd.to_numeric(df_exp[f"{ch}_expected"], errors="coerce").astype(float)
            elif ch in df_exp.columns:
                exp_series = pd.to_numeric(df_exp[ch], errors="coerce").astype(float)
            else:
                continue

            result_df[f"{ch}_expected"] = exp_series

            # Raw residual = observed - expected
            # Note: if observed is NaN (e.g. sensor dropout), residual naturally evaluates to NaN!
            raw_residual = obs_series - exp_series
            result_df[f"{ch}_residual"] = raw_residual.round(4)

            # Normalized residual = residual / scale
            scale = max(self.epsilon, self.scale_factors.get(ch, 1.0))
            norm_residual = raw_residual / scale
            result_df[f"{ch}_norm_residual"] = norm_residual.round(4)

        # Auxiliary analytical expected harmonic frequencies if available
        if "order_1x_freq_hz" in df_exp.columns:
            result_df["order_1x_freq_hz_expected"] = df_exp["order_1x_freq_hz"]
        if "order_2x_freq_hz" in df_exp.columns:
            result_df["order_2x_freq_hz_expected"] = df_exp["order_2x_freq_hz"]

        meta["residual_scales"] = dict(self.scale_factors)
        return ResidualFrame(result_df, metadata=meta)
