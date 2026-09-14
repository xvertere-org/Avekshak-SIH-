"""
Task-Specific Exceptions for Grey-Box ML Baselines (Phase 2).
"""


class MLTaskError(Exception):
    """Base exception for all ML baseline task errors."""
    pass


class TargetNotFoundError(MLTaskError):
    """Raised when an expected target column is missing from the dataset schema."""
    pass


class FeatureSelectionError(MLTaskError):
    """Raised when invalid, leakage-prone, or empty feature sets are encountered."""
    pass


class HealthyReferenceNotFoundError(MLTaskError):
    """Raised when a dataset does not possess a verified healthy/normal reference class."""
    pass


class InsufficientEntityError(MLTaskError):
    """Raised when an entity-held-out split cannot be satisfied with available entities."""
    pass


class TemporalOrderingError(MLTaskError):
    """Raised when temporal or cycle ordering within an entity is violated."""
    pass


class IncompatibleDomainError(MLTaskError):
    """Raised when an invalid cross-domain dataset combination is attempted."""
    pass
