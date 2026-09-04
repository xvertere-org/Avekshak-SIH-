"""
Phase 12: Explainability & Evidence Fusion Package.
Interprets and explains Phase 8 fault diagnosis, Phase 9 Health Index, Phase 10 forecasting, and Phase 11 RUL.
"""

from explainability.schema import (
    EvidenceStatus,
    EvidenceQuality,
    SHAPFeatureContribution,
    SHAPEvidence,
    PhysicsEvidence,
    HealthEvidence,
    TemporalEvidence,
    RULEvidence,
    EvidenceProvenance,
    ExplainabilityResult,
)
from explainability.shap_explainer import SHAPExplainer
from explainability.physics_evidence import PhysicsEvidenceEvaluator
from explainability.temporal_evidence import TemporalEvidenceEvaluator
from explainability.fusion import EvidenceFusionEngine
from explainability.pipeline import ExplainabilityPipeline

__all__ = [
    "EvidenceStatus",
    "EvidenceQuality",
    "SHAPFeatureContribution",
    "SHAPEvidence",
    "PhysicsEvidence",
    "HealthEvidence",
    "TemporalEvidence",
    "RULEvidence",
    "EvidenceProvenance",
    "ExplainabilityResult",
    "SHAPExplainer",
    "PhysicsEvidenceEvaluator",
    "TemporalEvidenceEvaluator",
    "EvidenceFusionEngine",
    "ExplainabilityPipeline",
]
