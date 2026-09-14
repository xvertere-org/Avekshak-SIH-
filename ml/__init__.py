"""
ML Integration Layer for SIH26054 Aero-Piston Engine Grey-Box Digital Twin.

Provides:
- DatasetRegistry: Unified registry with metadata and strict physics separation.
- DataLoader: Leakage-safe, parquet-only loader with validation.
- GroupSplitter: Group-aware splitting preventing train/test data leakage.
- FeatureSchema: Systematic classification of vibration, degradation, and reserved physics features.
- Validation: Robust leakage, numerical cleanliness, and domain protection checks.
"""

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
    FeatureClassification,
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
)

__all__ = [
    "DatasetRegistry",
    "DatasetMetadata",
    "DatasetUnavailableError",
    "IncompatibleDatasetError",
    "GroupColumnNotFoundError",
    "DataLoader",
    "DatasetSummary",
    "InvalidFileFormatError",
    "SchemaValidationError",
    "FeatureSchema",
    "FeatureClassification",
    "VIBRATION_TIME_DOMAIN_FEATURES",
    "VIBRATION_SPECTRAL_FEATURES",
    "ALL_VIBRATION_FEATURES",
    "DEGRADATION_FEATURES",
    "RESERVED_PHYSICS_DERIVED_FEATURES",
    "GroupSplitter",
    "SplitResult",
    "InsufficientGroupsError",
    "validate_no_leakage",
    "validate_numerical_cleanliness",
    "validate_protected_domains",
    "DataLeakageError",
    "NumericalIntegrityError",
    "ProtectedDomainViolationError",
]
