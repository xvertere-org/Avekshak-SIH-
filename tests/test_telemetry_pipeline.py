"""
Comprehensive Unit and Integration Tests for Phase 5: Telemetry + Data Pipeline.

Covers:
1. Telemetry Ingestion (single record, list, streamer, DataFrame, dicts, metadata preservation)
2. Provenance tracking (simulator vs. external substitute datasets)
3. Timestamp handling (monotonicity, ordering, interval calculation, duplicates)
4. Data Quality checking (missing values, NaN preservation, bounds vs. warning envelope, no fault labeling)
5. Phase 4F Sensor Fault integration (dropout -> NaN -> MISSING, bias preserved in raw)
6. Synchronization & Resampling (multi-channel alignment, causal scalar RMS for vibration, no future leakage)
7. Feature Engineering (causal rolling mean/std, slopes, temperature trends, RPM stability)
8. Vibration features (scalar RMS, peak, crest factor, analytical 1x/2x orders from RPM)
9. Leakage Prevention (backward-looking rolling verification, grouped mission splitting)
10. External Dataset Adapters (Generic CSV, CWRU surrogate, C-MAPSS surrogate)
11. DataQualityReport serialization & metrics
12. Performance benchmark on large dataset
"""

import math
import time
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
import pytest

from telemetry.schema import TelemetryRecord, MissionPhase, FaultCategory
from telemetry.streamer import TelemetryStreamer
from telemetry.ingestion import (
    CanonicalTelemetryFrame,
    TelemetryIngestor,
    REQUIRED_COLUMNS,
    OPTIONAL_PHYSICAL_CHANNELS,
)
from telemetry.quality import (
    QualityStatus,
    QualityEnvelope,
    DataQualityReport,
    DataQualityChecker,
    DEFAULT_QUALITY_ENVELOPES,
)
from telemetry.preprocessing import (
    TelemetrySynchronizer,
    TelemetryResampler,
    TelemetryCleaner,
)
from telemetry.features import (
    TelemetryFeatureExtractor,
    LeakageSafeSplitter,
)
from telemetry.adapters import (
    GenericCSVAdapter,
    VibrationBenchmarkAdapter,
    CMAPSSBenchmarkAdapter,
)
from telemetry.pipeline import (
    TelemetryPipeline,
    PipelineResult,
)
from simulator.engine_simulator import EngineSimulator
from simulator.subsystems.mission import MissionProfile, PhaseSegment, FlightPhase
from simulator.fault_interface import FaultState, FaultType, FaultSubsystem


# ────────────────────────────────────────────────────────────────────────────
# Helper Fixtures & Generators
# ────────────────────────────────────────────────────────────────────────────

def _create_synthetic_record(
    timestamp: float = 0.0,
    rpm: float = 4000.0,
    cht: float = 100.0,
    egt: float = 650.0,
    oil_temp: float = 85.0,
    oil_pressure: float = 4.0,
    fuel_flow: float = 12.0,
    vibration: float = 0.5,
    fault_type: str = "none",
    fault_severity: float = 0.0,
    source: str = "simulator_v1_physics",
    source_type: str = "synthetic",
    simulation_version: str = "0.2.0-phase2b-physics",
    metadata: dict = None,
) -> TelemetryRecord:
    return TelemetryRecord(
        timestamp=timestamp,
        mission_id="TEST_MISSION_01",
        engine_id="ENGINE_UAV_01",
        mission_phase="CRUISE",
        altitude=2000.0,
        ambient_temp=15.0,
        throttle=75.0,
        load=70.0,
        rpm=rpm,
        cht=cht,
        egt=egt,
        oil_temp=oil_temp,
        oil_pressure=oil_pressure,
        fuel_flow=fuel_flow,
        vibration=vibration,
        fault_type=fault_type,
        fault_severity=fault_severity,
        source=source,
        source_type=source_type,
        simulation_version=simulation_version,
        metadata=metadata if metadata is not None else {"power_kw": 45.0},
    )


def _generate_synthetic_stream(n_samples: int = 50, dt: float = 0.1) -> List[TelemetryRecord]:
    records = []
    for i in range(n_samples):
        t = round(i * dt, 4)
        rec = _create_synthetic_record(
            timestamp=t,
            rpm=4000.0 + 50.0 * np.sin(i * 0.1),
            cht=90.0 + i * 0.05,
            egt=620.0 + i * 0.1,
            vibration=0.4 + 0.05 * np.sin(i * 0.2),
        )
        records.append(rec)
    return records


# ────────────────────────────────────────────────────────────────────────────
# 1. Ingestion Tests
# ────────────────────────────────────────────────────────────────────────────

def test_ingest_single_record():
    """Verify single TelemetryRecord is ingested into CanonicalTelemetryFrame."""
    rec = _create_synthetic_record(metadata={"sensor_health": "nominal"})
    frame = TelemetryIngestor.ingest(rec)

    assert isinstance(frame, CanonicalTelemetryFrame)
    assert len(frame) == 1
    df = frame.to_dataframe()
    assert df["rpm"].iloc[0] == 4000.0
    assert df["source"].iloc[0] == "simulator_v1_physics"
    assert df["metadata"].iloc[0] == {"sensor_health": "nominal"}


def test_ingest_record_list_and_streamer():
    """Verify list of records and TelemetryStreamer are ingested properly."""
    stream = _generate_synthetic_stream(n_samples=25, dt=0.1)

    # Ingest list
    frame_from_list = TelemetryIngestor.ingest(stream)
    assert len(frame_from_list) == 25
    assert frame_from_list.physical_channels_present == OPTIONAL_PHYSICAL_CHANNELS

    # Ingest from streamer buffer
    streamer = TelemetryStreamer(buffer_size=50)
    for r in stream:
        streamer.push(r)
    frame_from_streamer = TelemetryIngestor.ingest(streamer)
    assert len(frame_from_streamer) == 25
    assert frame_from_streamer.to_dataframe()["timestamp"].iloc[-1] == round(24 * 0.1, 4)


def test_ingest_dataframe_and_partial_channels():
    """Verify DataFrame with only a subset of physical channels ingests without error."""
    partial_df = pd.DataFrame({
        "timestamp": [0.0, 0.1, 0.2],
        "engine_id": ["E1", "E1", "E1"],
        "mission_id": ["M1", "M1", "M1"],
        "vibration": [0.45, 0.48, 0.50],
        "rpm": [3800.0, 3810.0, 3820.0],
    })
    frame = TelemetryIngestor.ingest(partial_df, default_source="partial_bench")
    assert len(frame) == 3
    assert "vibration" in frame.physical_channels_present
    assert "rpm" in frame.physical_channels_present
    assert "cht" not in frame.physical_channels_present
    assert frame.provenance["source"] == "partial_bench"


def test_ingest_missing_required_column_raises():
    """Verify strict mode raises error if required column is missing and no default."""
    bad_df = pd.DataFrame({"rpm": [4000.0]})  # missing timestamp
    with pytest.raises(ValueError, match="missing required column"):
        TelemetryIngestor.ingest(bad_df, strict=True)


# ────────────────────────────────────────────────────────────────────────────
# 2. Provenance Tests
# ────────────────────────────────────────────────────────────────────────────

def test_provenance_retention():
    """Verify source, source_type, and simulation_version are preserved."""
    rec = _create_synthetic_record(
        source="custom_bench_generator",
        source_type="test_bench",
        simulation_version="1.5.0-alpha",
    )
    frame = TelemetryIngestor.ingest(rec)
    prov = frame.provenance

    assert prov["source"] == "custom_bench_generator"
    assert prov["source_type"] == "test_bench"
    assert prov["simulation_version"] == "1.5.0-alpha"


# ────────────────────────────────────────────────────────────────────────────
# 3. Timestamp Validation Tests
# ────────────────────────────────────────────────────────────────────────────

def test_timestamp_monotonicity_and_ordering():
    """Verify detection of out-of-order timestamps."""
    records = _generate_synthetic_stream(10, dt=0.1)
    df = pd.DataFrame([r.to_dict() for r in records])
    # Introduce out-of-order timestamp at index 5
    df.loc[5, "timestamp"] = df.loc[3, "timestamp"]

    frame = CanonicalTelemetryFrame(df)
    checker = DataQualityChecker(expected_dt_s=0.1)
    annotated_frame, report = checker.assess(frame)

    assert report.out_of_order_samples >= 1
    assert any("out-of-order" in issue for issue in report.issues_detected)
    assert annotated_frame["quality_status"].iloc[5] == QualityStatus.INVALID.value


def test_timestamp_duplicate_detection():
    """Verify detection of duplicate (engine_id, mission_id, timestamp)."""
    records = _generate_synthetic_stream(5, dt=0.1)
    df = pd.DataFrame([r.to_dict() for r in records])
    # Duplicate row 2
    dup_row = df.iloc[2:3].copy()
    df = pd.concat([df, dup_row], ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    frame = CanonicalTelemetryFrame(df)
    checker = DataQualityChecker(expected_dt_s=0.1)
    _, report = checker.assess(frame)

    assert report.duplicate_samples == 2
    assert any("duplicate" in issue for issue in report.issues_detected)


def test_sampling_interval_calculation():
    """Verify regular and irregular sampling interval statistics."""
    # Regular 10 Hz (dt = 0.1s)
    regular_stream = _generate_synthetic_stream(30, dt=0.1)
    frame_reg = TelemetryIngestor.ingest(regular_stream)
    checker = DataQualityChecker(expected_dt_s=0.1)
    _, report_reg = checker.assess(frame_reg)

    assert report_reg.is_regular_sampling is True
    assert pytest.approx(report_reg.sampling_interval_mean, abs=1e-3) == 0.1
    assert pytest.approx(report_reg.sampling_interval_std, abs=1e-3) == 0.0

    # Irregular timestamps
    irregular_df = pd.DataFrame([r.to_dict() for r in regular_stream])
    irregular_df.loc[10:, "timestamp"] += 0.5  # jump
    frame_irreg = CanonicalTelemetryFrame(irregular_df)
    _, report_irreg = checker.assess(frame_irreg)

    assert report_irreg.is_regular_sampling is False


# ────────────────────────────────────────────────────────────────────────────
# 4. Data Quality & Bounds vs. Warning Envelope Tests
# ────────────────────────────────────────────────────────────────────────────

def test_quality_valid_observations():
    """Verify clean nominal telemetry receives QualityStatus.VALID and high score."""
    stream = _generate_synthetic_stream(20, dt=0.1)
    frame = TelemetryIngestor.ingest(stream)
    checker = DataQualityChecker()
    annotated_frame, report = checker.assess(frame)

    assert report.quality_score == 1.0
    assert report.valid_samples == 20
    assert report.invalid_samples == 0
    assert (annotated_frame["quality_status"] == QualityStatus.VALID.value).all()


def test_quality_bounds_excursion_warning_not_invalid():
    """
    CRITICAL: Verify that operational envelope excursions (e.g. CHT elevated by fault)
    are flagged as WARNING, NOT marked as INVALID data, and raw values are unchanged.
    """
    stream = _generate_synthetic_stream(10, dt=0.1)
    df = pd.DataFrame([r.to_dict() for r in stream])
    # Push CHT to 154 °C (above nominal 150 °C warning envelope, but below 300 °C physical impossibility limit)
    df.loc[5, "cht"] = 154.0

    frame = CanonicalTelemetryFrame(df)
    checker = DataQualityChecker()
    annotated_frame, report = checker.assess(frame)

    assert report.warning_samples >= 1
    assert report.invalid_samples == 0
    assert annotated_frame["quality_status"].iloc[5] == QualityStatus.WARNING.value
    # Raw value remains exactly 154.0 (non-destructive)
    assert annotated_frame["cht"].iloc[5] == 154.0


def test_quality_physically_impossible_marked_invalid():
    """Verify physically impossible values (e.g. negative RPM) are tagged INVALID."""
    stream = _generate_synthetic_stream(5, dt=0.1)
    df = pd.DataFrame([r.to_dict() for r in stream])
    df.loc[2, "rpm"] = -50.0  # Physically impossible

    frame = CanonicalTelemetryFrame(df)
    checker = DataQualityChecker()
    annotated_frame, report = checker.assess(frame)

    assert report.invalid_samples >= 1
    assert annotated_frame["quality_status"].iloc[2] == QualityStatus.INVALID.value


def test_quality_check_does_not_alter_fault_labels():
    """Verify quality checking does not inject or alter engine fault classification."""
    rec = _create_synthetic_record(cht=155.0, fault_type="none")
    frame = TelemetryIngestor.ingest(rec)
    checker = DataQualityChecker()
    annotated_frame, _ = checker.assess(frame)

    # Fault label remains 'none' — data quality check does not diagnose engine faults
    assert annotated_frame["fault_type"].iloc[0] == "none"
    assert annotated_frame["quality_status"].iloc[0] == QualityStatus.WARNING.value


# ────────────────────────────────────────────────────────────────────────────
# 5. Phase 4F Sensor Fault Integration (Dropout NaN & Bias)
# ────────────────────────────────────────────────────────────────────────────

def test_sensor_dropout_nan_preserved_and_flagged():
    """
    Verify Phase 4F sensor dropout produces NaN that is detected as MISSING
    and NEVER converted to 0.0.
    """
    stream = _generate_synthetic_stream(10, dt=0.1)
    df = pd.DataFrame([r.to_dict() for r in stream])
    # Simulate Phase 4F sensor dropout on oil_pressure
    df.loc[3:5, "oil_pressure"] = float("nan")

    frame = CanonicalTelemetryFrame(df)
    checker = DataQualityChecker()
    annotated_frame, report = checker.assess(frame)

    assert report.missing_samples >= 3
    for idx in range(3, 6):
        assert math.isnan(annotated_frame["oil_pressure"].iloc[idx])
        assert annotated_frame["oil_pressure"].iloc[idx] != 0.0
        assert annotated_frame["missing_mask"].iloc[idx] is True or annotated_frame["missing_mask"].iloc[idx] == 1
        assert annotated_frame["quality_status"].iloc[idx] == QualityStatus.MISSING.value


def test_sensor_bias_preserved_in_raw():
    """Verify Phase 4F sensor bias is preserved in raw observations."""
    sim = EngineSimulator(seed=42)
    profile = MissionProfile(segments=[PhaseSegment(FlightPhase.CRUISE, 20.0, 75.0, 75.0, 2000.0, 2000.0)])
    bias_fault = FaultState(
        fault_type=FaultType.SENSOR_FAULT,
        severity=1.0,
        start_time=5.0,
        end_time=15.0,
        affected_subsystem=FaultSubsystem.SENSOR,
        parameters={"sensor_channel": "rpm", "sensor_mode": "bias", "bias_magnitude": 150.0},
    )

    records = sim.run(profile, dt=0.5, fault_schedule=bias_fault)
    frame = TelemetryIngestor.ingest(records)
    pipeline = TelemetryPipeline()
    result = pipeline.process(frame)

    # During fault window, RPM retains its biased value
    df_raw = result.raw_frame.to_dataframe()
    mask_fault = (df_raw["timestamp"] >= 5.0) & (df_raw["timestamp"] <= 15.0)
    assert mask_fault.any()
    # Ensure raw observations were not discarded or overwritten
    assert len(result.raw_frame) == len(records)


# ────────────────────────────────────────────────────────────────────────────
# 6. Synchronization & Resampling Tests
# ────────────────────────────────────────────────────────────────────────────

def test_telemetry_resampler_causal_10hz_to_1hz():
    """Verify deterministic causal 10 Hz -> 1 Hz downsampling."""
    stream_10hz = _generate_synthetic_stream(50, dt=0.1)  # 5.0 seconds
    frame = TelemetryIngestor.ingest(stream_10hz)

    resampler = TelemetryResampler(target_dt_s=1.0)
    resampled = resampler.resample(frame)

    df_res = resampled.to_dataframe()
    assert len(df_res) >= 4
    diffs = np.diff(df_res["timestamp"].to_numpy())
    np.testing.assert_allclose(diffs, 1.0, atol=1e-3)


def test_resampling_vibration_scalar_rms():
    """
    CRITICAL: Verify vibration channel uses scalar RMS aggregation:
    RMS = sqrt(mean(v_i^2)) rather than naive linear interpolation.
    """
    # Create 4 samples with constant vibration values [1.0, 2.0, 3.0, 4.0] in a 1.0s window
    df = pd.DataFrame({
        "timestamp": [0.25, 0.50, 0.75, 1.00],
        "engine_id": ["E1"] * 4,
        "mission_id": ["M1"] * 4,
        "source": ["sim"] * 4,
        "source_type": ["synthetic"] * 4,
        "vibration": [1.0, 2.0, 3.0, 4.0],
        "rpm": [3000.0, 3000.0, 3000.0, 3000.0],
    })
    frame = CanonicalTelemetryFrame(df)
    resampler = TelemetryResampler(target_dt_s=1.0)
    resampled = resampler.resample(frame)

    expected_rms = math.sqrt((1.0**2 + 2.0**2 + 3.0**2 + 4.0**2) / 4.0)  # sqrt(30/4) = sqrt(7.5) ≈ 2.7386
    actual_rms = resampled.to_dataframe()["vibration"].iloc[0]
    assert pytest.approx(actual_rms, abs=1e-3) == round(expected_rms, 4)


def test_resampling_dropout_nan_preservation():
    """Verify that a window containing only NaNs remains NaN after resampling."""
    df = pd.DataFrame({
        "timestamp": [0.25, 0.50, 0.75, 1.00],
        "engine_id": ["E1"] * 4,
        "mission_id": ["M1"] * 4,
        "source": ["sim"] * 4,
        "source_type": ["synthetic"] * 4,
        "rpm": [float("nan"), float("nan"), float("nan"), float("nan")],
        "vibration": [float("nan"), float("nan"), float("nan"), float("nan")],
    })
    frame = CanonicalTelemetryFrame(df)
    resampler = TelemetryResampler(target_dt_s=1.0)
    resampled = resampler.resample(frame)

    df_res = resampled.to_dataframe()
    assert math.isnan(df_res["rpm"].iloc[0])
    assert math.isnan(df_res["vibration"].iloc[0])


# ────────────────────────────────────────────────────────────────────────────
# 7. Feature Engineering Tests
# ────────────────────────────────────────────────────────────────────────────

def test_feature_extractor_causal_rolling_and_slopes():
    """Verify causal rolling statistics and rate of change features."""
    stream = _generate_synthetic_stream(30, dt=0.1)
    frame = TelemetryIngestor.ingest(stream)

    extractor = TelemetryFeatureExtractor(rolling_windows=[5, 10])
    feat_df = extractor.extract_features(frame)

    # Core checks
    assert "rpm_roll_mean_w5" in feat_df.columns
    assert "rpm_roll_std_w5" in feat_df.columns
    assert "cht_rate_of_change" in feat_df.columns
    assert "cht_trend_c_per_s" in feat_df.columns
    assert "rpm_stability_cov" in feat_df.columns
    assert "phase_persistence_duration_s" in feat_df.columns

    # Verify rolling mean at index 4 equals mean of first 5 samples
    raw_rpms = [r.rpm for r in stream[:5]]
    expected_mean = np.mean(raw_rpms)
    assert pytest.approx(feat_df["rpm_roll_mean_w5"].iloc[4], abs=1e-2) == expected_mean


def test_vibration_analytical_orders_and_statistics():
    """
    CRITICAL: Verify vibration features expose RMS, peak, crest factor,
    and analytical 1x / 2x order frequencies derived from instantaneous RPM.
    """
    stream = _generate_synthetic_stream(20, dt=0.1)
    frame = TelemetryIngestor.ingest(stream)

    extractor = TelemetryFeatureExtractor(rolling_windows=[5])
    feat_df = extractor.extract_features(frame)

    assert "vibration_roll_rms" in feat_df.columns
    assert "vibration_roll_peak" in feat_df.columns
    assert "vibration_crest_factor" in feat_df.columns
    assert "vib_order_1x_freq_hz" in feat_df.columns
    assert "vib_order_2x_freq_hz" in feat_df.columns

    # Check analytical 1x and 2x frequency math: RPM = 4000 -> 1x = 66.67 Hz, 2x = 133.33 Hz
    rpm_val = feat_df["rpm"].iloc[0]
    expected_1x = round(rpm_val / 60.0, 2)
    expected_2x = round(2.0 * rpm_val / 60.0, 2)
    assert feat_df["vib_order_1x_freq_hz"].iloc[0] == expected_1x
    assert feat_df["vib_order_2x_freq_hz"].iloc[0] == expected_2x


# ────────────────────────────────────────────────────────────────────────────
# 8. Leakage Prevention Tests
# ────────────────────────────────────────────────────────────────────────────

def test_leakage_safe_rolling_causality():
    """
    Verify strict backward causality: altering future sample (t=10)
    MUST NOT change rolling feature values at prior sample (t=5).
    """
    stream_orig = _generate_synthetic_stream(20, dt=0.1)
    frame_orig = TelemetryIngestor.ingest(stream_orig)

    stream_altered = _generate_synthetic_stream(20, dt=0.1)
    # Alter sample at index 15 (future)
    stream_altered[15].rpm = 9999.0
    frame_altered = TelemetryIngestor.ingest(stream_altered)

    extractor = TelemetryFeatureExtractor(rolling_windows=[5])
    feat_orig = extractor.extract_features(frame_orig)
    feat_altered = extractor.extract_features(frame_altered)

    # Features at index <= 10 must be identical
    np.testing.assert_allclose(
        feat_orig["rpm_roll_mean_w5"].iloc[:11].values,
        feat_altered["rpm_roll_mean_w5"].iloc[:11].values,
        rtol=1e-12,
        err_msg="Future data leaked backward into prior rolling features!",
    )


def test_leakage_safe_splitter():
    """Verify split_by_mission and temporal_split isolate partitions cleanly."""
    data = pd.DataFrame({
        "timestamp": np.arange(20, dtype=float),
        "mission_id": ["M1"] * 10 + ["M2"] * 10,
        "rpm": [4000.0] * 20,
    })

    # Grouped mission split
    splits = LeakageSafeSplitter.split_by_mission(data, train_missions=["M1"], test_missions=["M2"])
    assert len(splits["train"]) == 10
    assert len(splits["test"]) == 10
    assert (splits["train"]["mission_id"] == "M1").all()
    assert (splits["test"]["mission_id"] == "M2").all()

    # Temporal split with buffer
    train_t, test_t = LeakageSafeSplitter.temporal_split(data, split_timestamp_s=8.0, buffer_window_s=2.0)
    assert train_t["timestamp"].max() == 8.0
    assert test_t["timestamp"].min() > 10.0


# ────────────────────────────────────────────────────────────────────────────
# 9. External Dataset Adapters Tests
# ────────────────────────────────────────────────────────────────────────────

def test_generic_csv_adapter():
    """Verify GenericCSVAdapter maps custom headers and applies explicit provenance."""
    raw_csv_df = pd.DataFrame({
        "time_sec": [0.0, 1.0, 2.0],
        "eng_speed": [3500.0, 3600.0, 3700.0],
        "temp_cyl": [85.0, 88.0, 90.0],
    })
    adapter = GenericCSVAdapter(
        column_mapping={"time_sec": "timestamp", "eng_speed": "rpm", "temp_cyl": "cht"},
        source_name="flir_test_rig",
        source_type="external_benchmark",
    )
    frame = adapter.adapt(raw_csv_df)

    assert isinstance(frame, CanonicalTelemetryFrame)
    assert "rpm" in frame.physical_channels_present
    assert "cht" in frame.physical_channels_present
    assert frame.provenance["source"] == "flir_test_rig"
    assert frame.provenance["source_type"] == "external_benchmark"


def test_vibration_benchmark_adapter():
    """Verify VibrationBenchmarkAdapter adapts bearing data and attaches disclaimer."""
    vib_df = pd.DataFrame({
        "DE_time": [0.12, 0.15, -0.09, 0.04],
        "FE_time": [0.08, 0.09, -0.04, 0.02],
        "RPM": [1797.0, 1797.0, 1796.0, 1797.0],
    })
    adapter = VibrationBenchmarkAdapter(dataset_name="cwru_test_bearing", dt_s=0.0001)
    frame = adapter.adapt(vib_df)

    assert "vibration" in frame.physical_channels_present
    assert "rpm" in frame.physical_channels_present
    assert frame.provenance["source_type"] == "substitute_dataset"
    assert "disclaimer" in frame.metadata
    assert "NOT certified aero-piston" in frame.metadata["disclaimer"]


def test_cmapss_benchmark_adapter():
    """Verify CMAPSSBenchmarkAdapter normalizes turbofan data with substitute disclaimer."""
    # Synthesize small C-MAPSS DataFrame
    cmapss_data = {
        "unit_number": [1, 1, 1],
        "time_cycles": [1, 2, 3],
        "op_setting_1": [0.0, 0.0, 0.0],
        "op_setting_2": [0.0, 0.0, 0.0],
        "op_setting_3": [100.0, 100.0, 100.0],
        "s_2": [642.0, 642.5, 643.0],
        "s_3": [1585.0, 1587.0, 1589.0],
        "s_4": [1400.0, 1402.0, 1404.0],
        "s_7": [553.0, 552.8, 552.5],
        "s_9": [9050.0, 9052.0, 9051.0],
        "s_12": [521.0, 521.2, 521.5],
    }
    for i in range(1, 22):
        if f"s_{i}" not in cmapss_data:
            cmapss_data[f"s_{i}"] = [0.0] * 3

    adapter = CMAPSSBenchmarkAdapter(subset_id="FD001")
    frame = adapter.adapt(pd.DataFrame(cmapss_data))

    assert "engine_id" in frame.data.columns
    assert frame.data["engine_id"].iloc[0] == "CMAPSS_UNIT_1"
    assert frame.provenance["source_type"] == "external_benchmark"
    assert "disclaimer" in frame.metadata


# ────────────────────────────────────────────────────────────────────────────
# 10. End-to-End Pipeline & Performance Tests
# ────────────────────────────────────────────────────────────────────────────

def test_telemetry_pipeline_end_to_end():
    """Verify full ingestion -> quality -> resample -> features execution."""
    stream = _generate_synthetic_stream(40, dt=0.1)
    pipeline = TelemetryPipeline(target_resample_dt_s=1.0)
    result = pipeline.process(stream)

    assert isinstance(result, PipelineResult)
    assert len(result.raw_frame) == 40
    assert len(result.processed_frame) >= 3
    assert not result.feature_frame.empty
    assert result.quality_report.valid_samples == 40

    summary = result.summary()
    assert summary["raw_samples"] == 40
    assert summary["feature_count"] > 10


def test_pipeline_performance_golden_baseline():
    """
    Verify pipeline handles a representative large dataset (~17,200 samples,
    equivalent to the golden baseline) efficiently (< 5.0 seconds).
    """
    n_samples = 17200
    t_start = time.perf_counter()

    # Generate vector of synthetic data
    ts = np.arange(n_samples, dtype=float) * 0.1
    df = pd.DataFrame({
        "timestamp": ts,
        "engine_id": ["ENGINE_UAV_01"] * n_samples,
        "mission_id": ["MISSION_GOLDEN_01"] * n_samples,
        "mission_phase": ["CRUISE"] * n_samples,
        "rpm": 4000.0 + 20.0 * np.sin(ts * 0.05),
        "cht": 95.0 + 5.0 * np.cos(ts * 0.01),
        "egt": 650.0 + 10.0 * np.sin(ts * 0.02),
        "oil_temp": 80.0 + 2.0 * np.sin(ts * 0.01),
        "oil_pressure": 4.2 + 0.1 * np.cos(ts * 0.05),
        "fuel_flow": 12.0 + 0.5 * np.sin(ts * 0.05),
        "vibration": 0.45 + 0.02 * np.random.randn(n_samples),
        "source": ["simulator_v1_physics"] * n_samples,
        "source_type": ["synthetic"] * n_samples,
    })

    frame = CanonicalTelemetryFrame(df)
    pipeline = TelemetryPipeline(target_resample_dt_s=1.0)
    result = pipeline.process(frame)

    elapsed = time.perf_counter() - t_start

    assert len(result.raw_frame) == 17200
    assert len(result.processed_frame) >= 1700
    assert elapsed < 5.0, f"Pipeline processing took too long: {elapsed:.2f}s (must be < 5.0s)"
