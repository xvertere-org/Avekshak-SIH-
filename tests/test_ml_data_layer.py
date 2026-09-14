"""
Comprehensive Unit & Integration Test Suite for Grey-Box ML Data Layer (Phase 1).

Tests cover:
1. Dataset registry completeness & metadata accuracy.
2. Processed file discovery & missing dataset handling (BASIC).
3. Required column validation & column preservation.
4. Feature schema taxonomy (vibration, degradation, reserved physics).
5. Numerical cleanliness (no NaNs or Infs in feature matrices).
6. Dynamic grouping column selection (no hardcoded column names).
7. Leakage-safe group partitioning (strictly zero group overlap).
8. Deterministic splitting under identical random seeds.
9. CWRU 3-file limitation handling & controlled demonstration guard.
10. Rejection of raw file formats (.zip, .mat, .txt, .csv) as canonical ML inputs.
11. Rejection of incompatible cross-domain dataset concatenation.
12. Zero Rotax piston channel collision.
13. Domain protection test verifying zero modifications to core protected directories.
"""

from pathlib import Path
import os
import sys
import pytest
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

from ml.dataset_registry import (
    DatasetRegistry,
    DatasetMetadata,
    DatasetUnavailableError,
    IncompatibleDatasetError,
    GroupColumnNotFoundError,
)
from ml.data_loader import (
    DataLoader,
    DatasetSummary,
    InvalidFileFormatError,
    SchemaValidationError,
)
from ml.feature_schema import (
    FeatureSchema,
    VIBRATION_TIME_DOMAIN_FEATURES,
    VIBRATION_SPECTRAL_FEATURES,
    ALL_VIBRATION_FEATURES,
    DEGRADATION_FEATURES,
    RESERVED_PHYSICS_DERIVED_FEATURES,
)
from ml.split_strategy import (
    GroupSplitter,
    SplitResult,
    InsufficientGroupsError,
)
from ml.validation import (
    validate_no_leakage,
    validate_numerical_cleanliness,
    validate_protected_domains,
    DataLeakageError,
    NumericalIntegrityError,
    ProtectedDomainViolationError,
    PROTECTED_DOMAINS,
)


@pytest.fixture(scope="module")
def registry():
    return DatasetRegistry()


@pytest.fixture(scope="module")
def loader(registry):
    return DataLoader(project_root=PROJECT_ROOT, registry=registry)


@pytest.fixture(scope="module")
def splitter(registry):
    return GroupSplitter(registry=registry)


# ==============================================================================
# 1. Dataset Registry Tests
# ==============================================================================

class TestDatasetRegistry:
    """Tests for dataset registry completeness and metadata governance."""

    def test_registry_completeness(self, registry):
        """All 7 required datasets must be registered."""
        expected = {"cmapss", "femto", "cwru", "paderborn", "nasa_battery", "nust", "basic"}
        registered = set(registry.list_datasets())
        assert expected.issubset(registered), f"Missing datasets in registry: {expected - registered}"

    def test_basic_is_marked_unavailable(self, registry):
        """BASIC dataset must be explicitly marked as UNAVAILABLE."""
        assert not registry.is_available("basic")
        meta = registry.get("basic")
        assert meta.availability == "UNAVAILABLE"
        assert meta.integration_role == "UNAVAILABLE"
        assert any("unavailable" in lim.lower() for lim in meta.limitations)

    def test_benchmark_roles_and_segregation(self, registry):
        """C-MAPSS and NASA Battery must be designated as benchmark datasets, not Rotax telemetry."""
        cmapss_meta = registry.get("cmapss")
        assert cmapss_meta.integration_role == "RUL_METHODOLOGY_BENCHMARK"
        assert not cmapss_meta.is_direct_rotax_telemetry

        battery_meta = registry.get("nasa_battery")
        assert battery_meta.integration_role == "SOH_RUL_BENCHMARK"
        assert not battery_meta.is_direct_rotax_telemetry

    def test_cwru_limitation_documented(self, registry):
        """CWRU must have the 3-source-file limitation explicitly documented."""
        meta = registry.get("cwru")
        assert any("3 independent source files" in lim for lim in meta.limitations)
        assert any("feature validation" in lim.lower() for lim in meta.limitations)

    def test_dynamic_grouping_column_resolution(self, registry):
        """Resolution must succeed based on actual DataFrame columns without hardcoding."""
        # C-MAPSS supports either unit_number or unit_id
        resolved_1 = registry.resolve_grouping_column("cmapss", ["unit_number", "time_cycles", "sensor_1"])
        assert resolved_1 == "unit_number"

        resolved_2 = registry.resolve_grouping_column("cmapss", ["unit_id", "time_cycles", "sensor_1"])
        assert resolved_2 == "unit_id"

        # Missing grouping column raises GroupColumnNotFoundError
        with pytest.raises(GroupColumnNotFoundError):
            registry.resolve_grouping_column("cmapss", ["arbitrary_col", "another_col"])


# ==============================================================================
# 2. Processed Data Loader Tests
# ==============================================================================

class TestDataLoader:
    """Tests for file discovery, loading, and format validation."""

    def test_processed_file_discovery(self, loader):
        """Discovers existing processed parquet files for available datasets."""
        for d_id in ["cmapss", "femto", "cwru", "paderborn", "nasa_battery", "nust"]:
            files = loader.discover_files(d_id)
            assert len(files) > 0, f"No files discovered for {d_id}"
            for f in files:
                assert f.suffix == ".parquet"
                assert f.exists()

    def test_unavailable_dataset_raises_error(self, loader):
        """Attempting to discover or load an unavailable dataset must raise DatasetUnavailableError."""
        with pytest.raises(DatasetUnavailableError) as exc_info:
            loader.discover_files("basic")
        assert "UNAVAILABLE" in str(exc_info.value)

    def test_reject_raw_file_formats(self, loader, tmp_path):
        """Raw file formats (.zip, .mat, .txt, .csv) must be rejected with InvalidFileFormatError."""
        raw_formats = ["data.zip", "test.mat", "log.txt", "samples.csv", "archive.tar.gz"]
        for rf in raw_formats:
            dummy_file = tmp_path / rf
            dummy_file.write_text("raw dummy content")
            with pytest.raises(InvalidFileFormatError):
                loader.validate_file_path(dummy_file)

    def test_load_dataset_preserves_columns(self, loader):
        """Loading processed data must preserve original columns and types."""
        df = loader.load_dataset("cwru")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "source_file" in df.columns
        assert "fault_label" in df.columns

    def test_required_columns_validation(self, loader):
        """Loader must raise SchemaValidationError if required columns are missing."""
        with pytest.raises(SchemaValidationError):
            loader.load_dataset("cwru", required_columns=["non_existent_column_123"])

    def test_rotax_channel_collision_prevention(self, loader, tmp_path, registry):
        """Non-engine datasets containing Rotax channels (e.g. CHT, EGT) must be rejected."""
        tainted_df = pd.DataFrame({
            "bearing_id": ["B1"],
            "cht": [120.0],  # Forbidden Rotax channel in bearing dataset!
            "mean": [0.5],
        })
        tainted_file = tmp_path / "tainted.parquet"
        tainted_df.to_parquet(tainted_file)

        with pytest.raises(SchemaValidationError) as exc:
            loader.load_dataset("femto", file_path=tainted_file)
        assert "protected Rotax engine channels" in str(exc.value)

    def test_incompatible_dataset_combination_rejection(self, loader):
        """Combining incompatible datasets (e.g. C-MAPSS and FEMTO) must be strictly forbidden."""
        with pytest.raises(IncompatibleDatasetError) as exc:
            loader.combine_compatible_datasets(["cmapss", "femto"])
        assert "incompatible integration roles" in str(exc.value)

    def test_dataset_summary_generation(self, loader):
        """Summaries must contain actual row counts, feature counts, and limitations."""
        summary = loader.summarize_dataset("cwru")
        assert isinstance(summary, DatasetSummary)
        assert summary.num_rows == 1179
        assert summary.num_features == 47
        assert summary.grouping_column == "source_file"
        assert summary.missing_value_count == 0
        assert summary.infinite_value_count == 0
        assert len(summary.dataset_limitations) > 0


# ==============================================================================
# 3. Feature Schema Tests
# ==============================================================================

class TestFeatureSchema:
    """Tests for feature taxonomy classification and reserved physics attributes."""

    def test_vibration_feature_classification(self):
        """Known vibration metrics must be correctly classified as vibration features."""
        test_cols = ["mean", "RMS", "kurtosis", "crest_factor", "spectral_centroid", "band_energy_0_1500hz"]
        classification = FeatureSchema.classify_columns(test_cols)
        assert set(classification.vibration_features) == set(test_cols)
        assert len(classification.degradation_features) == 0

    def test_degradation_feature_classification(self):
        """Degradation metrics must be classified as degradation features."""
        test_cols = ["time_cycles", "cycle", "health_index", "soh", "capacity_loss", "rul_label"]
        classification = FeatureSchema.classify_columns(test_cols)
        assert set(classification.degradation_features) == set(test_cols)
        assert len(classification.vibration_features) == 0

    def test_reserved_physics_features_not_fabricated(self):
        """Reserved physics features must not be present in Phase 1 processed data."""
        test_cols = ["physics_residuals", "normalized_residuals", "subsystem_health_scores"]
        with pytest.raises(ValueError) as exc:
            FeatureSchema.assert_no_fabricated_physics(test_cols)
        assert "Fabricated physics features detected" in str(exc.value)

    def test_processed_datasets_have_no_fabricated_physics(self, loader):
        """Verify that none of the real processed datasets contain fabricated physics columns."""
        for d_id in ["cmapss", "femto", "cwru", "paderborn", "nasa_battery", "nust"]:
            df = loader.load_dataset(d_id)
            FeatureSchema.assert_no_fabricated_physics(list(df.columns))


# ==============================================================================
# 4. Leakage-Safe Splitting Tests
# ==============================================================================

class TestGroupSplitter:
    """Tests for group-aware splitting and leakage prevention."""

    def test_cmapss_group_split_zero_leakage(self, loader, splitter):
        """C-MAPSS split by unit_number must have 0 overlapping units."""
        df = loader.load_dataset("cmapss", subset="FD001", split="train")
        train_df, val_df, test_df, res = splitter.split(
            df, "cmapss", train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=123
        )
        assert not res.leakage_detected
        train_units = set(train_df["unit_number"])
        val_units = set(val_df["unit_number"])
        test_units = set(test_df["unit_number"])

        assert train_units.isdisjoint(val_units)
        assert train_units.isdisjoint(test_units)
        assert val_units.isdisjoint(test_units)
        assert len(train_df) + len(val_df) + len(test_df) == len(df)

    def test_femto_group_split_zero_leakage(self, loader, splitter):
        """FEMTO split by bearing_id must have 0 overlapping bearings."""
        df = loader.load_dataset("femto", split="train")
        train_df, val_df, test_df, res = splitter.split(
            df, "femto", train_ratio=0.67, val_ratio=0.0, test_ratio=0.33, seed=42
        )
        assert not res.leakage_detected
        assert set(train_df["bearing_id"]).isdisjoint(set(test_df["bearing_id"]))

    def test_cwru_limitation_and_controlled_demo(self, loader, splitter):
        """CWRU must reject conventional 3-way split and require controlled demo flag."""
        df = loader.load_dataset("cwru")

        # Conventional split must raise InsufficientGroupsError
        with pytest.raises(InsufficientGroupsError) as exc:
            splitter.split(df, "cwru", allow_cwru_demo=False)
        assert "3 independent source files" in str(exc.value)

        # Controlled demo split must succeed with zero leakage
        train_df, val_df, test_df, res = splitter.split(
            df, "cwru", allow_cwru_demo=True, seed=42
        )
        assert not res.leakage_detected
        assert res.num_train_groups == 2
        assert res.num_test_groups == 1
        assert set(train_df["source_file"]).isdisjoint(set(test_df["source_file"]))

    def test_deterministic_seed_consistency(self, loader, splitter):
        """Splits under identical seeds must yield strictly identical group assignments."""
        df = loader.load_dataset("paderborn")
        _, _, _, res1 = splitter.split(df, "paderborn", group_column="source_file", seed=999)
        _, _, _, res2 = splitter.split(df, "paderborn", group_column="source_file", seed=999)

        assert res1.train_groups == res2.train_groups
        assert res1.val_groups == res2.val_groups
        assert res1.test_groups == res2.test_groups

    def test_leakage_validator_catches_synthetic_overlap(self):
        """validate_no_leakage must raise DataLeakageError if groups overlap."""
        train_g = {"unit_1", "unit_2", "unit_3"}
        test_g = {"unit_3", "unit_4"}  # unit_3 leaked!

        with pytest.raises(DataLeakageError) as exc:
            validate_no_leakage(train_g, test_g, group_col_name="unit")
        assert "Critical Data Leakage Detected" in str(exc.value)


# ==============================================================================
# 5. Numerical Integrity Tests
# ==============================================================================

class TestNumericalCleanliness:
    """Tests for absence of unexpected NaNs and Infs in feature matrices."""

    def test_cwru_features_cleanliness(self, loader):
        """CWRU feature matrix must have 0 NaNs and 0 Infs."""
        df = loader.load_dataset("cwru")
        vibration_cols = [c for c in df.columns if c in ALL_VIBRATION_FEATURES]
        assert len(vibration_cols) > 0
        stats = validate_numerical_cleanliness(df, columns=vibration_cols)
        assert stats["columns_checked"] == len(vibration_cols)

    def test_paderborn_features_cleanliness(self, loader):
        """Paderborn feature matrix must have 0 NaNs and 0 Infs."""
        df = loader.load_dataset("paderborn")
        vibration_cols = [c for c in df.columns if c in ALL_VIBRATION_FEATURES]
        assert len(vibration_cols) > 0
        stats = validate_numerical_cleanliness(df, columns=vibration_cols)
        assert stats["columns_checked"] == len(vibration_cols)

    def test_numerical_cleanliness_fails_on_nan(self):
        """validate_numerical_cleanliness must raise NumericalIntegrityError if NaNs present."""
        df = pd.DataFrame({"feat_1": [1.0, np.nan, 3.0], "feat_2": [4.0, 5.0, 6.0]})
        with pytest.raises(NumericalIntegrityError) as exc:
            validate_numerical_cleanliness(df, columns=["feat_1", "feat_2"])
        assert "null/NaN values" in str(exc.value)


# ==============================================================================
# 6. Realistic Core Domain Protection Test
# ==============================================================================

class TestCoreDomainProtection:
    """
    Realistic verification comparing git changed files against protected directories.
    """

    def test_zero_modifications_in_protected_domains(self):
        """
        Asserts that NO files inside the protected core domains were modified or created.
        Protected domains:
          - physics/
          - simulator/
          - telemetry/
          - digital_twin/
          - health_index/
          - anomaly_detection/
          - fault_diagnosis/
          - frontend/
          - backend/
        """
        changed_files = validate_protected_domains(PROJECT_ROOT)

        violations = []
        for f in changed_files:
            first_dir = f.split("/")[0].lower()
            if first_dir in PROTECTED_DOMAINS:
                violations.append(f)

        assert len(violations) == 0, (
            f"VIOLATION OF PROTECTED DOMAINS! The following files in protected directories were modified: {violations}"
        )
