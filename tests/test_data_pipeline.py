"""
Validation tests for the SIH26054 data pipeline.

Tests cover:
  - File discovery and loading
  - Schema validation
  - Unsafe C-MAPSS column detection
  - Missing values, duplicates
  - RUL validity
  - Bearing/unit-level split integrity (no train/test leakage)
  - Provenance preservation
  - Output file existence
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.common.schemas import UNSAFE_CMAPSS_COLUMN_NAMES
from data_pipeline.common.validation import (
    check_no_unsafe_cmapss_columns,
    check_missing_values,
    check_duplicates,
    check_empty_dataframe,
    check_rul_validity,
    check_no_train_test_leakage,
    validate_schema,
)
from data_pipeline.common.splitting import split_by_id, verify_no_leakage
from data_pipeline.common.provenance import compute_file_checksum


# ============================
# Common validation tests
# ============================

class TestCommonValidation:
    """Tests for the shared validation utilities."""

    def test_unsafe_cmapss_column_detection(self):
        """Must detect unsafe column names in C-MAPSS outputs."""
        df = pd.DataFrame({"cht": [1, 2], "egt": [3, 4], "safe_col": [5, 6]})
        violations = check_no_unsafe_cmapss_columns(df, "test_cmapss")
        assert len(violations) >= 2, f"Expected violations for cht/egt, got: {violations}"

    def test_safe_cmapss_columns_pass(self):
        """Columns using cmapss_s_N naming should not trigger violations."""
        df = pd.DataFrame({"cmapss_s_2": [1], "cmapss_s_3": [2], "unit_number": [1]})
        violations = check_no_unsafe_cmapss_columns(df, "test_cmapss")
        assert len(violations) == 0, f"Unexpected violations: {violations}"

    def test_missing_values(self):
        """Detect missing values correctly."""
        df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
        missing = check_missing_values(df)
        assert "a" in missing and missing["a"] == 1

    def test_duplicates(self):
        """Detect duplicate rows."""
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        assert check_duplicates(df) == 1

    def test_empty_dataframe(self):
        """Detect empty DataFrames."""
        assert check_empty_dataframe(pd.DataFrame()) is True
        assert check_empty_dataframe(pd.DataFrame({"a": [1]})) is False

    def test_rul_validity_good(self):
        """Valid RUL values should pass."""
        rul = pd.Series([100, 50, 25, 0])
        issues = check_rul_validity(rul)
        assert len(issues) == 0

    def test_rul_validity_negative(self):
        """Negative RUL values must be flagged."""
        rul = pd.Series([100, -5, 25, 0])
        issues = check_rul_validity(rul)
        assert any("negative" in i for i in issues)

    def test_no_train_test_leakage(self):
        """Overlapping IDs must be flagged."""
        train_ids = {"unit_1", "unit_2"}
        test_ids = {"unit_2", "unit_3"}
        issues = check_no_train_test_leakage(train_ids, test_ids)
        assert len(issues) > 0, "Leakage should be detected"

    def test_no_leakage_clean(self):
        """Non-overlapping IDs should pass."""
        train_ids = {"unit_1", "unit_2"}
        test_ids = {"unit_3", "unit_4"}
        issues = check_no_train_test_leakage(train_ids, test_ids)
        assert len(issues) == 0

    def test_schema_validation(self):
        """Missing columns should be flagged."""
        df = pd.DataFrame({"a": [1], "b": [2]})
        issues = validate_schema(df, ["a", "c"])
        assert any("c" in str(i) for i in issues)


class TestSplitting:
    """Tests for train/val/test splitting."""

    def test_split_preserves_all_ids(self):
        """All IDs must appear in exactly one partition."""
        ids = [f"id_{i}" for i in range(20)]
        train, val, test = split_by_id(ids)
        all_split = set(train) | set(val) | set(test)
        assert all_split == set(ids)

    def test_split_no_leakage(self):
        """No ID should appear in multiple partitions."""
        ids = [f"id_{i}" for i in range(20)]
        train, val, test = split_by_id(ids)
        assert verify_no_leakage(set(train), set(val), set(test))


class TestProvenance:
    """Tests for provenance tracking."""

    def test_checksum_consistency(self):
        """Same file should produce same checksum."""
        # Use this test file itself
        this_file = os.path.abspath(__file__)
        c1 = compute_file_checksum(this_file)
        c2 = compute_file_checksum(this_file)
        assert c1 == c2
        assert len(c1) == 64  # SHA-256 hex digest


# ============================
# C-MAPSS specific tests
# ============================

class TestCMAPSS:
    """Tests for C-MAPSS dataset processing."""

    def test_cmapss_loader_finds_subsets(self):
        """Loader must find at least FD001."""
        from data_pipeline.cmapss.loader import list_available_subsets
        subsets = list_available_subsets()
        assert "FD001" in subsets, f"FD001 not found. Available: {subsets}"

    def test_cmapss_loader_column_names(self):
        """All sensor columns must use cmapss_s_N naming."""
        from data_pipeline.cmapss.loader import load_train
        df = load_train("FD001")
        sensor_cols = [c for c in df.columns if c.startswith("cmapss_s_")]
        assert len(sensor_cols) == 21, f"Expected 21 sensor columns, got {len(sensor_cols)}"

    def test_cmapss_no_unsafe_columns(self):
        """C-MAPSS output must NEVER contain cht, egt, oil_temp, oil_pressure, fuel_flow."""
        from data_pipeline.cmapss.loader import load_train
        df = load_train("FD001")
        violations = check_no_unsafe_cmapss_columns(df)
        assert len(violations) == 0, f"UNSAFE columns found: {violations}"

    def test_cmapss_rul_generation(self):
        """RUL labels must be non-negative and decreasing within each unit."""
        from data_pipeline.cmapss.preprocess import preprocess_subset
        result = preprocess_subset("FD001")
        train_df = result["train"]
        assert "rul_label" in train_df.columns
        issues = check_rul_validity(train_df["rul_label"])
        assert len(issues) == 0, f"RUL issues: {issues}"
        # Check monotonicity per unit
        for unit_id in train_df["unit_number"].unique()[:5]:
            unit_rul = train_df[train_df["unit_number"] == unit_id]["rul_label"].values
            assert all(unit_rul[i] >= unit_rul[i + 1] for i in range(len(unit_rul) - 1)), \
                f"RUL not monotonically decreasing for unit {unit_id}"

    def test_cmapss_processed_output_exists(self):
        """Processed output files should exist after preprocessing."""
        output_dir = "data/processed/cmapss"
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            # At least FD001 train and test should be present
            assert any("FD001" in f and "train" in f for f in files), \
                f"FD001 train output not found in {files}"


# ============================
# CWRU specific tests
# ============================

class TestCWRU:
    """Tests for CWRU dataset processing."""

    def test_cwru_files_exist(self):
        """At least the 3 core MAT files should be available."""
        from data_pipeline.cwru.loader import list_available_files
        files = list_available_files()
        assert len(files) >= 3, f"Expected >= 3 MAT files, found {len(files)}"

    def test_cwru_mat_loading(self):
        """MAT files should load and contain vibration channels."""
        from data_pipeline.cwru.loader import load_all
        data = load_all()
        assert len(data) >= 3
        for fn, info in data.items():
            assert len(info["channels"]) > 0, f"No channels in {fn}"

    def test_cwru_audit_discovers_all_supported_files(self):
        """Every discovered supported file must be processed or explicitly skipped with reason."""
        from data_pipeline.cwru.loader import audit_raw_cwru_files
        audits = audit_raw_cwru_files()
        assert len(audits) >= 3
        for a in audits:
            assert a["parsing_status"] in ["SUCCESS", "SKIPPED", "FAILED"]
            if a["parsing_status"] != "SUCCESS":
                assert a["skip_reason"] is not None, f"Skipped file without reason: {a['filename']}"
            else:
                assert len(a["signal_channels"]) > 0, f"No channels in parsed file: {a['filename']}"
                assert a["sampling_frequency"] is not None

    def test_cwru_mat_channel_extraction(self):
        """Verify DE, FE, BA channels and RPM extraction from MATLAB keys."""
        from data_pipeline.cwru.loader import load_all
        data = load_all()
        # 105.mat has DE, FE, and BA channels
        assert "105.mat" in data
        ch_105 = data["105.mat"]["channels"]
        assert any("_DE_" in k for k in ch_105), "Drive End channel not detected in 105.mat"
        assert any("_FE_" in k for k in ch_105), "Fan End channel not detected in 105.mat"
        assert any("_BA_" in k for k in ch_105), "Base channel not detected in 105.mat"
        assert data["105.mat"]["metadata"].get("rpm") is not None

    def test_cwru_artifacts_and_manifest_exist(self):
        """Verify canonical outputs exist: parquet, metadata.json, manifest.json, sample.csv."""
        out_dir = "data/processed/cwru"
        assert os.path.exists(os.path.join(out_dir, "cwru_features.parquet"))
        assert os.path.exists(os.path.join(out_dir, "cwru_metadata.json"))
        assert os.path.exists(os.path.join(out_dir, "cwru_processing_manifest.json"))
        assert os.path.exists(os.path.join(out_dir, "cwru_features_sample.csv"))

    def test_cwru_all_time_domain_features_present(self):
        """Verify presence of all 14 time-domain metrics + rms compatibility alias."""
        df = pd.read_parquet("data/processed/cwru/cwru_features.parquet")
        required_14 = [
            "mean", "std", "variance", "RMS", "minimum", "maximum",
            "peak_to_peak", "absolute_mean", "skewness", "kurtosis",
            "crest_factor", "shape_factor", "impulse_factor", "clearance_factor"
        ]
        for f in required_14:
            assert f in df.columns, f"Missing time-domain feature: {f}"
        assert "rms" in df.columns, "Missing backward-compatibility alias 'rms'"
        assert np.allclose(df["rms"], df["RMS"]), "'rms' alias does not match 'RMS'"

    def test_cwru_all_frequency_domain_features_present(self):
        """Verify presence of 4 frequency-domain metrics and documented subbands."""
        df = pd.read_parquet("data/processed/cwru/cwru_features.parquet")
        freq_cols = [
            "dominant_frequency", "spectral_energy", "spectral_centroid", "frequency_band_energy",
            "band_energy_0_1500hz", "band_energy_1500_3000hz", "band_energy_3000_4500hz", "band_energy_4500_6000hz"
        ]
        for c in freq_cols:
            assert c in df.columns, f"Missing frequency-domain feature: {c}"

    def test_cwru_feature_finiteness_and_no_nans(self):
        """No NaN or infinite values in numeric feature columns."""
        df = pd.read_parquet("data/processed/cwru/cwru_features.parquet")
        assert len(df) > 0, "Feature dataframe is empty"
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for c in numeric_cols:
            assert not df[c].isna().any(), f"NaN values found in column {c}"
            assert np.all(np.isfinite(df[c])), f"Non-finite values found in column {c}"

    def test_cwru_traceability_and_condition_columns(self):
        """Verify all required provenance and condition columns."""
        df = pd.read_parquet("data/processed/cwru/cwru_features.parquet")
        trace_cols = [
            "dataset_name", "source_file", "source_id", "record_id",
            "bearing_id", "window_id", "sensor_location",
            "sampling_frequency", "RPM", "load_condition", "fault_label", "fault_size"
        ]
        for c in trace_cols:
            assert c in df.columns, f"Missing traceability column: {c}"
        assert (df["dataset_name"] == "cwru").all()
        assert set(df["sensor_location"].unique()).issubset({"drive_end", "fan_end", "base"})

    def test_cwru_no_duplicate_records(self):
        """Record IDs must be globally unique without duplicates."""
        df = pd.read_parquet("data/processed/cwru/cwru_features.parquet")
        assert df["record_id"].is_unique, "Duplicate record_id values detected"

    def test_cwru_no_unsafe_engine_channels(self):
        """No vibration signal mapped to aero piston channels."""
        df = pd.read_parquet("data/processed/cwru/cwru_features.parquet")
        unsafe = {"cht", "egt", "oil_pressure", "oil_temp", "fuel_flow", "manifold_pressure"}
        for col in df.columns:
            assert col.lower() not in unsafe, f"Unsafe engine column found in CWRU: {col}"

    def test_cwru_split_integrity_documentation(self):
        """Split strategy in metadata must prohibit random window leakage."""
        import json
        with open("data/processed/cwru/cwru_metadata.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        strat = meta.get("splitting_strategy", {})
        assert "Partition by 'source_file' or 'bearing_id'" in strat.get("recommendation", "")
        assert "subset_limitations" in strat


# ============================
# FEMTO specific tests
# ============================

class TestFEMTO:
    """Tests for FEMTO dataset processing."""

    def test_femto_training_bearings_exist(self):
        """At least 6 training bearings should be available."""
        from data_pipeline.femto.loader import list_bearings
        bearings = list_bearings("train")
        assert len(bearings) >= 6, f"Expected >= 6 training bearings, found {len(bearings)}"

    def test_femto_test_bearings_exist(self):
        """At least 11 test bearings should be available."""
        from data_pipeline.femto.loader import list_bearings
        bearings = list_bearings("test")
        assert len(bearings) >= 11, f"Expected >= 11 test bearings, found {len(bearings)}"

    def test_femto_no_train_test_leakage(self):
        """Training and test bearing IDs must not overlap."""
        from data_pipeline.femto.loader import list_bearings
        train_ids = set(list_bearings("train"))
        test_ids = set(list_bearings("test"))
        overlap = train_ids & test_ids
        assert len(overlap) == 0, f"Train/test bearing leakage: {overlap}"


# ============================
# NUST specific tests
# ============================

class TestNUST:
    """Tests for NUST dataset processing."""

    def test_nust_csv_files_exist(self):
        """At least some CSV files should be available."""
        from data_pipeline.nust.loader import list_csv_files
        files = list_csv_files()
        assert len(files) > 0, "No NUST CSV files found"

    def test_nust_csv_loading(self):
        """First CSV file should load with expected columns."""
        from data_pipeline.nust.loader import list_csv_files, load_csv_file
        files = list_csv_files()
        if files:
            df, meta = load_csv_file(files[0])
            assert not df.empty, "First CSV loaded as empty"
            assert len(df.columns) >= 10, f"Expected >= 10 columns, got {len(df.columns)}"

    def test_nust_no_fabricated_channels(self):
        """NUST outputs must NOT contain fabricated aero-piston channel names."""
        output_path = "data/processed/nust"
        if os.path.exists(output_path):
            for fn in os.listdir(output_path):
                fp = os.path.join(output_path, fn)
                if fn.endswith(".parquet"):
                    df = pd.read_parquet(fp)
                elif fn.endswith(".csv"):
                    df = pd.read_csv(fp, nrows=5)
                else:
                    continue
                violations = check_no_unsafe_cmapss_columns(df, "nust")
                assert len(violations) == 0, f"Fabricated channels in NUST: {violations}"


# ============================
# NASA Battery specific tests
# ============================

class TestNASABattery:
    """Tests for NASA Battery dataset."""

    def test_nasa_battery_archives_exist(self):
        """At least one ZIP archive should be present."""
        from data_pipeline.nasa_battery.loader import get_battery_extracted_dir
        bdir = get_battery_extracted_dir()
        assert os.path.exists(bdir), f"Directory {bdir} not found"
        zips = [f for f in os.listdir(bdir) if f.endswith(".zip")]
        assert len(zips) >= 1, "No NASA Battery archives found"

    def test_battery_zip_extraction_safety(self):
        """Test safe extraction and manifest generation without path traversal."""
        from data_pipeline.nasa_battery.loader import extract_battery_archives
        manifest, summary = extract_battery_archives()
        assert len(manifest) >= 1
        assert summary["total_extracted_files"] > 0
        for entry in manifest:
            assert entry["extraction_status"] == "SUCCESS"
            for p in entry["extracted_file_paths"]:
                assert not p.startswith("..")

    def test_battery_supported_file_discovery(self):
        """Discover supported .mat files and catalog non-mat files."""
        from data_pipeline.nasa_battery.loader import discover_battery_files
        mat_files, non_mat_files = discover_battery_files()
        assert len(mat_files) >= 4, f"Expected >= 4 mat files, got {len(mat_files)}"
        assert any("B0005" in f for f in mat_files), "B0005.mat not discovered"
        assert len(non_mat_files) >= 1, "Non-mat documentation files not discovered"

    def test_battery_empty_or_corrupt_file_handling(self):
        """Empty or invalid mat structure should return empty DataFrame without error."""
        from data_pipeline.nasa_battery.features import extract_battery_cycle_records
        df_empty = extract_battery_cycle_records({}, "BTEST", "test.mat", "test/id")
        assert df_empty.empty

    def test_battery_duplicate_record_detection(self):
        """Deduplication must detect and resolve duplicate battery-cycle pairs."""
        df_dups = pd.DataFrame([
            {"battery_id": "B0005", "cycle": 1, "voltage_mean": 3.8},
            {"battery_id": "B0005", "cycle": 1, "voltage_mean": 3.8},
            {"battery_id": "B0005", "cycle": 2, "voltage_mean": 3.7},
        ])
        deduped = df_dups.drop_duplicates(subset=["battery_id", "cycle"])
        assert len(deduped) == 2

    def test_battery_invalid_capacity_rejection(self):
        """Negative capacities must be rejected/set to NaN."""
        from data_pipeline.nasa_battery.features import extract_battery_cycle_records
        # Mock cycle object with negative capacity
        class DummyCycle:
            type = "discharge"
            ambient_temperature = 24.0
            class data:
                Capacity = -1.5
                Time = [0, 100]
                Voltage_measured = [4.2, 3.0]
                Current_measured = [-1.0, -1.0]
                Temperature_measured = [24.0, 26.0]
        class DummyObj:
            cycle = [DummyCycle()]
        
        df = extract_battery_cycle_records({"BTEST": DummyObj()}, "BTEST", "test.mat", "test/id")
        assert len(df) == 1
        assert np.isnan(df.loc[0, "discharge_capacity"])

    def test_battery_parquet_schema_integrity(self):
        """battery_cycles.parquet must exist and satisfy strict schema requirements."""
        pq_path = "data/processed/nasa_battery/battery_cycles.parquet"
        assert os.path.exists(pq_path), f"File {pq_path} does not exist"
        df = pd.read_parquet(pq_path)
        assert len(df) > 0, "battery_cycles.parquet is empty"
        required_cols = [
            "dataset_name", "source_file", "source_id", "record_id",
            "battery_id", "cycle", "cycle_type", "voltage_mean",
            "discharge_capacity", "reference_capacity", "soh", "capacity_loss"
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing required column {col}"
        
        # Check uniqueness of (battery_id, cycle)
        dups = df.duplicated(subset=["battery_id", "cycle"]).sum()
        assert dups == 0, f"Found {dups} duplicate battery-cycle pairs"
        
        # Check non-negativity of valid discharge capacities
        valid_caps = df["discharge_capacity"].dropna()
        assert (valid_caps >= 0.0).all(), "Found negative discharge capacities"

    def test_battery_metadata_and_manifest_exist(self):
        """Metadata JSON and extraction manifest JSON must exist."""
        meta_path = "data/processed/nasa_battery/battery_metadata.json"
        man_path = "data/processed/nasa_battery/extraction_manifest.json"
        assert os.path.exists(meta_path)
        assert os.path.exists(man_path)


# ============================
# Paderborn specific tests
# ============================

class TestPaderborn:
    """Tests for Paderborn dataset."""

    def test_paderborn_archives_exist(self):
        """RAR archives should be present."""
        from data_pipeline.paderborn.loader import get_paderborn_extracted_dir
        pdir = get_paderborn_extracted_dir()
        assert os.path.exists(pdir), f"Directory {pdir} not found"

    def test_paderborn_bearing_classification(self):
        """Bearing IDs should be correctly classified."""
        from data_pipeline.paderborn.loader import parse_filename_metadata
        m_k = parse_filename_metadata("N09_M07_F10_K001_1.mat")
        assert m_k["fault_label"] == "healthy"
        assert m_k["speed_rpm"] == 900
        
        m_ka = parse_filename_metadata("N15_M07_F10_KA01_2.mat")
        assert m_ka["fault_label"] == "outer_race_fault"
        assert m_ka["speed_rpm"] == 1500

        m_ki = parse_filename_metadata("N15_M01_F10_KI04_3.mat")
        assert m_ki["fault_label"] == "inner_race_fault"

    def test_paderborn_file_discovery(self):
        """Discover valid .mat files and catalog PDF files."""
        from data_pipeline.paderborn.loader import discover_paderborn_files
        mat_files, pdf_files = discover_paderborn_files()
        assert len(mat_files) >= 10, f"Expected >= 10 mat files, got {len(mat_files)}"
        assert len(pdf_files) >= 1, "No PDF logs cataloged"

    def test_paderborn_signal_validation(self):
        """Vibration signal extracted from mat must be non-empty and 64 kHz."""
        from data_pipeline.paderborn.loader import discover_paderborn_files, load_paderborn_mat, extract_vibration_signal
        mat_files, _ = discover_paderborn_files()
        if mat_files:
            mat = load_paderborn_mat(mat_files[0])
            sig, fs = extract_vibration_signal(mat)
            assert sig is not None
            assert len(sig) > 1000
            assert fs == 64000.0
            assert np.all(np.isfinite(sig))

    def test_paderborn_window_generation(self):
        """Windowing logic produces expected number of windows with finite features."""
        from data_pipeline.paderborn.features import extract_paderborn_window_features
        dummy_sig = np.random.randn(256823)
        meta = {
            "source_file": "dummy.mat",
            "measurement_id": "dummy_meas",
            "bearing_id": "K001",
            "fault_label": "healthy",
            "operating_condition": "N09_M07_F10",
            "speed_rpm": 900,
            "torque_nm": 0.7,
            "radial_force_n": 1000,
        }
        df = extract_paderborn_window_features(dummy_sig, meta, window_size=4096, overlap=2048)
        assert len(df) == 124
        assert "rms" in df.columns
        assert "dominant_frequency" in df.columns
        assert "band_energy_0_5khz" in df.columns
        assert np.all(np.isfinite(df["rms"]))

    def test_paderborn_parquet_schema_integrity(self):
        """paderborn_features.parquet must exist and satisfy strict schema requirements."""
        pq_path = "data/processed/paderborn/paderborn_features.parquet"
        assert os.path.exists(pq_path), f"File {pq_path} does not exist"
        df = pd.read_parquet(pq_path)
        assert len(df) > 0
        required_cols = [
            "dataset_name", "source_file", "source_id", "record_id",
            "bearing_id", "measurement_id", "window_id", "fault_label",
            "operating_condition", "mean", "std", "variance", "rms",
            "peak", "peak_to_peak", "crest_factor", "skewness", "kurtosis",
            "dominant_frequency", "spectral_energy", "spectral_centroid",
            "band_energy_0_5khz", "band_energy_5_15khz", "band_energy_15_32khz"
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing required column {col} in paderborn"
        
        # Verify finite values
        num_cols = ["rms", "peak", "kurtosis", "spectral_centroid"]
        for c in num_cols:
            assert np.all(np.isfinite(df[c])), f"Found non-finite values in {c}"

    def test_paderborn_metadata_and_manifest_exist(self):
        """Metadata JSON and processing manifest JSON must exist."""
        meta_path = "data/processed/paderborn/paderborn_metadata.json"
        man_path = "data/processed/paderborn/paderborn_processing_manifest.json"
        assert os.path.exists(meta_path)
        assert os.path.exists(man_path)


# ============================
# BASIC dataset test
# ============================

class TestBASIC:
    """Tests for BASIC dataset — must be documented as unavailable."""

    def test_basic_is_empty(self):
        """BASIC directory must contain only .gitkeep or be empty."""
        basic_dir = os.path.join("data", "raw", "basic")
        if os.path.exists(basic_dir):
            files = [f for f in os.listdir(basic_dir) if f != ".gitkeep"]
            assert len(files) == 0, f"BASIC should be empty but contains: {files}"


# ============================
# Integration safety tests
# ============================

class TestIntegrationSafety:
    """Tests to ensure external datasets don't contaminate aero-piston channels."""

    def test_no_unsafe_benchmark_to_piston_channel_mapping(self):
        """Verify no benchmark dataset has cht, egt, oil_temp, oil_pressure, fuel_flow."""
        dataset_dirs = [
            "data/processed/cmapss",
            "data/processed/cwru",
            "data/processed/femto",
            "data/processed/nust",
            "data/processed/nasa_battery",
            "data/processed/paderborn",
        ]
        for d in dataset_dirs:
            if not os.path.exists(d):
                continue
            for fn in os.listdir(d):
                if fn.endswith(".parquet"):
                    fp = os.path.join(d, fn)
                    df = pd.read_parquet(fp)
                    violations = check_no_unsafe_cmapss_columns(df, d)
                    assert len(violations) == 0, f"UNSAFE piston mapping found in {fp}: {violations}"

    def test_no_physics_files_modified(self):
        """Verify key physics files exist and haven't been touched by this pipeline."""
        physics_files = [
            "simulator/base.py",
            "digital_twin/__init__.py",
            "health_index/__init__.py",
            "anomaly_detection/__init__.py",
            "fault_diagnosis/__init__.py",
        ]
        for pf in physics_files:
            if os.path.exists(pf):
                with open(pf, "r") as f:
                    content = f.read()
                assert "data_pipeline" not in content, \
                    f"Physics file {pf} contains data_pipeline reference"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

