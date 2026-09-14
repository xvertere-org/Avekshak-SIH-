"""
Phase 9 Health Index Pipeline.

End-to-end integration combining:
- HealthCalculator (evidence mapping, dynamic renormalization, sensor isolation)
- CausalEWMASmoother (noise reduction with zero future leakage)
- DegradationTracker (timestamp-based rate and trend tracking)
- Optional Phase 7/8 passthrough context
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd

from health_index.schema import (
    HealthIndexConfig,
    HealthIndexResult,
    HealthState,
    DegradationTrend,
)
from health_index.calculator import HealthCalculator
from health_index.smoothing import CausalEWMASmoother
from health_index.degradation import DegradationTracker


# Mapping from standard residual channel names to expected DataFrame columns
RESIDUAL_CHANNEL_NAMES: List[str] = [
    "rpm",
    "cht",
    "egt",
    "oil_temp",
    "oil_pressure",
    "fuel_flow",
    "vibration",
]


class HealthIndexPipeline:
    """
    Inference pipeline for Phase 9 Health Index and Degradation Tracking.

    Operates purely on ResidualFrame data (single record dict or batch DataFrame).
    Produces structured HealthIndexResult objects with full provenance,
    channel contributions, effective weights, data quality, and trend tracking.

    Phase 7 and Phase 8 Integration:
    - Purely OPTIONAL. The pipeline functions completely without Phase 7 or Phase 8.
    - If provided, Phase 8 diagnosis can assist sensor-channel isolation, and
      diagnostic context is preserved in result provenance.
    """

    def __init__(self, config: Optional[HealthIndexConfig] = None):
        self.config = config or HealthIndexConfig()
        self.calculator = HealthCalculator(self.config)
        self.smoother = CausalEWMASmoother(alpha=self.config.ewma_alpha)
        self.tracker = DegradationTracker(self.config)

    def reset(self, engine_id: Optional[str] = None, mission_id: Optional[str] = None) -> None:
        """Reset internal pipeline state (smoother, trackers)."""
        self.calculator.sensor_tracker.reset(engine_id=engine_id, mission_id=mission_id)
        self.smoother.reset(engine_id=engine_id, mission_id=mission_id)
        self.tracker.reset(engine_id=engine_id, mission_id=mission_id)

    def process_sample(
        self,
        sample: Dict[str, Any],
        optional_context: Optional[Dict[str, Any]] = None,
    ) -> HealthIndexResult:
        """
        Process a single telemetry/residual sample.

        Args:
            sample: Dict containing residual channels and operating context.
            optional_context: Optional upstream Phase 7/8 diagnosis and anomaly metadata.

        Returns:
            HealthIndexResult instance.
        """
        df = pd.DataFrame([sample])
        ctx_list = [optional_context] if optional_context is not None else None
        results = self.process_dataframe(df, optional_context=ctx_list)
        return results[0]

    def process_dataframe(
        self,
        df: pd.DataFrame,
        optional_context: Optional[Union[List[Optional[Dict[str, Any]]], pd.DataFrame]] = None,
    ) -> List[HealthIndexResult]:
        """
        Process a batch of ResidualFrame observations in causal chronological order.

        Args:
            df: DataFrame containing ResidualFrame columns.
            optional_context: Optional list/DataFrame of upstream Phase 7/8 contexts per sample.

        Returns:
            List of HealthIndexResult instances.
        """
        n_samples = len(df)
        if n_samples == 0:
            return []

        # Parse optional context
        contexts: List[Optional[Dict[str, Any]]] = [None] * n_samples
        if optional_context is not None:
            if isinstance(optional_context, pd.DataFrame):
                contexts = [row.to_dict() for _, row in optional_context.iterrows()]
            elif isinstance(optional_context, list):
                contexts = list(optional_context)

        results: List[HealthIndexResult] = []

        for i in range(n_samples):
            row = df.iloc[i]
            timestamp = float(row.get("timestamp", float(i)))
            raw_engine = row.get("engine_id")
            engine_id = str(raw_engine) if (raw_engine is not None and pd.notna(raw_engine)) else "ENG_001"
            raw_mission = row.get("mission_id")
            if raw_mission is None or pd.isna(raw_mission):
                raw_mission = row.get("mission_run_id")
            mission_id = str(raw_mission) if (raw_mission is not None and pd.notna(raw_mission)) else None
            mission_phase = str(row.get("mission_phase", "UNKNOWN"))

            # Extract normalized residuals for core channels
            residuals: Dict[str, float] = {}
            for ch in RESIDUAL_CHANNEL_NAMES:
                col = f"{ch}_norm_residual"
                if col in row:
                    residuals[ch] = float(row[col]) if pd.notna(row[col]) else float("nan")
                elif ch in row:
                    residuals[ch] = float(row[ch]) if pd.notna(row[ch]) else float("nan")
                else:
                    residuals[ch] = float("nan")

            # Extract upstream context if provided
            ctx = contexts[i] if i < len(contexts) else None
            upstream_fault_type = None
            upstream_confidence = None
            upstream_suspect_channel = None
            upstream_suspect_channels = None
            anomaly_status = None
            anomaly_score = None

            if ctx:
                upstream_fault_type = ctx.get("predicted_fault_type") or ctx.get("diagnosed_fault")
                upstream_confidence = ctx.get("diagnostic_confidence")
                upstream_suspect_channel = ctx.get("sensor_channel") or ctx.get("suspect_channel")
                upstream_suspect_channels = ctx.get("suspect_channels")
                anomaly_status = ctx.get("anomaly_status")
                anomaly_score = ctx.get("anomaly_score")
            else:
                # Check if embedded in row
                upstream_fault_type = row.get("predicted_fault_type")
                upstream_confidence = row.get("diagnostic_confidence")
                anomaly_status = row.get("anomaly_status")
                anomaly_score = row.get("anomaly_score")

            # 1. Compute raw degradation and channel attribution
            (
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
            ) = self.calculator.compute(
                timestamp=timestamp,
                residuals=residuals,
                upstream_fault_type=upstream_fault_type,
                upstream_confidence=upstream_confidence,
                upstream_suspect_channel=upstream_suspect_channel,
                upstream_suspect_channels=upstream_suspect_channels,
                engine_id=engine_id,
                mission_id=mission_id,
            )

            # 2. Causal EWMA Smoothing
            smoothed_hi = self.smoother.update(
                raw_hi,
                engine_id=engine_id,
                mission_id=mission_id,
                timestamp=timestamp,
            )

            # 3. Degradation Rate and Trend Tracking
            rate, trend, health_state = self.tracker.update_and_evaluate(
                timestamp=timestamp,
                smoothed_hi=smoothed_hi,
                engine_id=engine_id,
                mission_id=mission_id,
            )

            res = HealthIndexResult(
                timestamp=timestamp,
                engine_id=engine_id,
                mission_id=mission_id,
                mission_phase=mission_phase,
                raw_health_index=raw_hi,
                smoothed_health_index=smoothed_hi,
                health_state=health_state,
                raw_degradation_score=raw_degradation,
                degradation_rate=rate,
                degradation_trend=trend,
                channel_contributions=contrib_dict,
                channel_degradation_evidence=evidence_dict,
                dominant_degraded_channels=dominant_degraded,
                valid_channels=valid_channels,
                missing_channels=missing_channels,
                excluded_channels=excluded_channels,
                effective_channel_weights=effective_weights,
                data_quality=data_quality,
                diagnosed_fault=upstream_fault_type,
                diagnostic_confidence=upstream_confidence,
                anomaly_status=anomaly_status,
                anomaly_score=anomaly_score,
            )
            results.append(res)

        return results
