"""
Telemetry Pipeline Orchestrator for SIH26054 Aero Piston Engine Digital Twin.

Coordinates the end-to-end telemetry engineering flow:
Raw Telemetry -> Ingestion -> Quality Assessment -> Synchronization/Resampling -> Feature Engineering
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, List
import pandas as pd

from telemetry.ingestion import CanonicalTelemetryFrame, TelemetryIngestor
from telemetry.quality import DataQualityChecker, DataQualityReport
from telemetry.preprocessing import TelemetryResampler, TelemetryCleaner, TelemetrySynchronizer
from telemetry.features import TelemetryFeatureExtractor


@dataclass
class PipelineResult:
    """
    Standardized payload emitted by TelemetryPipeline.
    Preserves raw observations, quality diagnostics, processed series, and ML features.
    """
    raw_frame: CanonicalTelemetryFrame
    quality_report: DataQualityReport
    processed_frame: CanonicalTelemetryFrame
    feature_frame: pd.DataFrame

    def summary(self) -> Dict[str, Any]:
        """Return high-level summary of pipeline outputs."""
        return {
            "raw_samples": len(self.raw_frame),
            "processed_samples": len(self.processed_frame),
            "feature_count": self.feature_frame.shape[1] if not self.feature_frame.empty else 0,
            "quality_score": self.quality_report.quality_score,
            "valid_samples": self.quality_report.valid_samples,
            "warning_samples": self.quality_report.warning_samples,
            "invalid_samples": self.quality_report.invalid_samples,
            "missing_samples": self.quality_report.missing_samples,
            "provenance": self.raw_frame.provenance,
        }


class TelemetryPipeline:
    """
    Modular, lightweight telemetry processing pipeline.
    Ensures raw data immutability, quality auditing, and leakage-free feature synthesis.
    """

    def __init__(
        self,
        quality_checker: Optional[DataQualityChecker] = None,
        target_resample_dt_s: Optional[float] = None,
        deduplicate: bool = True,
        impute_brief_dropout: bool = False,
        feature_extractor: Optional[TelemetryFeatureExtractor] = None,
    ):
        """
        Initialize TelemetryPipeline.

        Args:
            quality_checker: Custom DataQualityChecker or default if None.
            target_resample_dt_s: If specified, resamples to this interval (seconds), e.g. 1.0 for 1 Hz.
            deduplicate: Whether to remove duplicate (engine_id, mission_id, timestamp) rows.
            impute_brief_dropout: Whether to traceably impute short missing gaps after quality checking.
            feature_extractor: Custom TelemetryFeatureExtractor or default if None.
        """
        self.quality_checker = quality_checker or DataQualityChecker()
        self.target_resample_dt_s = target_resample_dt_s
        self.deduplicate = deduplicate
        self.impute_brief_dropout = impute_brief_dropout
        self.feature_extractor = feature_extractor or TelemetryFeatureExtractor()

    def process(
        self,
        raw_telemetry: Union[CanonicalTelemetryFrame, Any],
        **ingest_kwargs,
    ) -> PipelineResult:
        """
        Execute full telemetry processing pipeline.

        Args:
            raw_telemetry: TelemetryRecord(s), DataFrame, streamer, or CanonicalTelemetryFrame.
            **ingest_kwargs: Optional kwargs forwarded to TelemetryIngestor.

        Returns:
            PipelineResult packaging raw frame, quality report, processed frame, and features.
        """
        # 1. Ingestion into CanonicalTelemetryFrame
        if isinstance(raw_telemetry, CanonicalTelemetryFrame):
            raw_frame = raw_telemetry.copy()
        else:
            raw_frame = TelemetryIngestor.ingest(raw_telemetry, **ingest_kwargs)

        # 2. Quality Assessment (non-destructive, flags observations without altering values)
        annotated_frame, quality_report = self.quality_checker.assess(raw_frame)

        processed = annotated_frame.copy()

        # 3. Optional Deduplication
        if self.deduplicate and quality_report.duplicate_samples > 0:
            processed = TelemetryCleaner.deduplicate(processed)

        # 4. Optional Traceable Imputation (after quality checking has recorded missingness)
        if self.impute_brief_dropout and quality_report.missing_samples > 0:
            processed = TelemetryCleaner.impute_missing(processed, limit=5)

        # 5. Optional Causal Resampling (e.g. 10 Hz -> 1 Hz)
        if self.target_resample_dt_s is not None and self.target_resample_dt_s > 0:
            resampler = TelemetryResampler(target_dt_s=self.target_resample_dt_s)
            processed = resampler.resample(processed)

        # 6. Causal Feature Engineering
        feature_df = self.feature_extractor.extract_features(processed)

        return PipelineResult(
            raw_frame=raw_frame,
            quality_report=quality_report,
            processed_frame=processed,
            feature_frame=feature_df,
        )
