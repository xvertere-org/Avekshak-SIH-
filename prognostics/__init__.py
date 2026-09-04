"""
Phase 11: Remaining Useful Life (RUL) Estimation & Prognostics Package.
Authoritative prognostic health trajectory projection and failure time estimation.
"""

from prognostics.schema import (
    RULStatus,
    EOLCriterion,
    EOLCriteriaConfig,
    RULConfig,
    RULResult,
)
from prognostics.threshold import WeakestLinkEOLEvaluator
from prognostics.trajectory import TheilSenExtrapolator, DualHorizonSynthesizer
from prognostics.uncertainty import MonteCarloTrajectoryPropagator
from prognostics.pipeline import RULPipeline
from prognostics.evaluation import (
    compute_phm08_score,
    compute_picp_and_mpiw,
    compute_prognostic_metrics,
)

__all__ = [
    "RULStatus",
    "EOLCriterion",
    "EOLCriteriaConfig",
    "RULConfig",
    "RULResult",
    "WeakestLinkEOLEvaluator",
    "TheilSenExtrapolator",
    "DualHorizonSynthesizer",
    "MonteCarloTrajectoryPropagator",
    "RULPipeline",
    "compute_phm08_score",
    "compute_picp_and_mpiw",
    "compute_prognostic_metrics",
]
