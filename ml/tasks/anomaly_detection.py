"""
Anomaly Detection Baseline Pipeline for Phase 2 Grey-Box ML.

Implements semi-supervised anomaly detection:
- Models are fitted STRICTLY on verified healthy/normal training records.
- Evaluation is performed on strictly held-out groups containing both healthy and anomalous samples.
- Healthy label definitions are verified dynamically from schema before execution.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from ml.tasks.adapters import DatasetAdapter
from ml.tasks.preprocessing import LeakageSafePreprocessor
from ml.tasks.evaluation import evaluate_anomaly_detection
from ml.tasks.exceptions import HealthyReferenceNotFoundError
from ml.split_strategy import GroupSplitter, SplitResult


@dataclass
class AnomalyModelResult:
    """Evaluation result for a single anomaly detection model."""
    model_name: str
    model_type: str
    threshold_method: str
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnomalyDetectionReport:
    """Structured report for an anomaly detection baseline."""
    dataset_id: str
    task_name: str = "ANOMALY_DETECTION"
    status: str = "completed"  # "completed", "controlled_demonstration", "skipped"
    controlled_demonstration: bool = False
    generalization_claim: bool = True
    healthy_label_definition: str = ""
    target_column: str = ""
    group_column: str = ""
    feature_count: int = 0
    feature_names: List[str] = field(default_factory=list)
    train_groups: List[Any] = field(default_factory=list)
    test_groups: List[Any] = field(default_factory=list)
    num_train_healthy_samples: int = 0
    num_test_samples: int = 0
    test_healthy_count: int = 0
    test_anomaly_count: int = 0
    models: Dict[str, AnomalyModelResult] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["models"] = {k: v.to_dict() if isinstance(v, AnomalyModelResult) else v for k, v in self.models.items()}
        return d


class AnomalyDetectionPipeline:
    """
    Executes leakage-safe semi-supervised anomaly detection baselines.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.splitter = GroupSplitter()

    def run(
        self,
        dataset_id: str,
        target_column: Optional[str] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.0,
        test_ratio: float = 0.3,
    ) -> AnomalyDetectionReport:
        """
        Run anomaly detection baseline on the specified dataset.
        """
        adapter = DatasetAdapter(dataset_id)
        df, target_col, group_col, feature_cols = adapter.load_and_adapt(target_column=target_column)

        # Verify and resolve healthy label
        healthy_label = adapter.resolve_healthy_label(df, target_col)

        is_cwru = (dataset_id.lower() == "cwru")
        controlled_demo = is_cwru
        gen_claim = not is_cwru

        # Group-aware splitting
        if is_cwru:
            train_df, _, test_df, split_res = self.splitter.split(
                df,
                dataset_id,
                group_column=group_col,
                allow_cwru_demo=True,
                label_column=target_col,
                seed=self.seed,
            )
            limitations = [
                "CWRU controlled demonstration: normal class originates from 97.mat.",
                "Evaluates anomaly discrimination against held-out fault files.",
                "Zero statistical generalization claim to unobserved operational machinery.",
            ]
        else:
            train_df, _, test_df, split_res = self.splitter.split(
                df,
                dataset_id,
                group_column=group_col,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                label_column=target_col,
                seed=self.seed,
            )
            limitations = [
                f"Semi-supervised anomaly detection fitted strictly on '{healthy_label}' records in training groups.",
                f"Source-file level grouping on '{group_col}' guarantees 0 file-level leakage.",
            ]

        # Filter training data to HEALTHY ONLY
        train_healthy_mask = train_df[target_col] == healthy_label
        train_healthy_df = train_df[train_healthy_mask]

        if len(train_healthy_df) == 0:
            raise HealthyReferenceNotFoundError(
                f"Training split for '{dataset_id}' contains zero samples with healthy label '{healthy_label}'."
            )

        X_train_raw = train_healthy_df[feature_cols].to_numpy(dtype=float)
        preprocessor = LeakageSafePreprocessor(with_imputer=True)
        X_train = preprocessor.fit_transform(X_train_raw, feature_names=feature_cols)

        # Test data evaluation: 0 = healthy, 1 = anomaly
        X_test_raw = test_df[feature_cols].to_numpy(dtype=float)
        X_test = preprocessor.transform(X_test_raw)
        y_test_binary = (test_df[target_col] != healthy_label).astype(int).to_numpy()

        n_test_healthy = int(np.sum(y_test_binary == 0))
        n_test_anomaly = int(np.sum(y_test_binary == 1))

        # Model 1: Isolation Forest
        iso_forest = IsolationForest(
            n_estimators=100,
            contamination=0.05,
            random_state=self.seed,
            n_jobs=-1,
        )
        iso_forest.fit(X_train)

        # In IsolationForest, score_samples is opposite of anomaly score (lower is more anomalous)
        # We invert so higher score = more anomalous
        raw_test_scores = -iso_forest.score_samples(X_test)
        
        # Decision threshold calibrated on training healthy percentile (95th percentile)
        raw_train_scores = -iso_forest.score_samples(X_train)
        threshold_95 = float(np.percentile(raw_train_scores, 95))
        y_pred_iso = (raw_test_scores >= threshold_95).astype(int)

        metrics_iso = evaluate_anomaly_detection(y_test_binary, y_pred_iso, scores=raw_test_scores)
        metrics_iso["calibrated_threshold"] = threshold_95

        # Model 2: Simple Distance / L2 Norm from Training Center Baseline
        train_center = np.mean(X_train, axis=0)
        dist_train = np.linalg.norm(X_train - train_center, axis=1)
        dist_threshold = float(np.percentile(dist_train, 95))

        dist_test = np.linalg.norm(X_test - train_center, axis=1)
        y_pred_dist = (dist_test >= dist_threshold).astype(int)
        metrics_dist = evaluate_anomaly_detection(y_test_binary, y_pred_dist, scores=dist_test)
        metrics_dist["calibrated_threshold"] = dist_threshold

        status_str = "controlled_demonstration" if controlled_demo else "completed"

        return AnomalyDetectionReport(
            dataset_id=dataset_id,
            task_name="ANOMALY_DETECTION",
            status=status_str,
            controlled_demonstration=controlled_demo,
            generalization_claim=gen_claim,
            healthy_label_definition=healthy_label,
            target_column=target_col,
            group_column=group_col,
            feature_count=len(feature_cols),
            feature_names=feature_cols,
            train_groups=split_res.train_groups,
            test_groups=split_res.test_groups,
            num_train_healthy_samples=len(train_healthy_df),
            num_test_samples=len(test_df),
            test_healthy_count=n_test_healthy,
            test_anomaly_count=n_test_anomaly,
            models={
                "isolation_forest": AnomalyModelResult(
                    model_name="isolation_forest",
                    model_type="IsolationForest",
                    threshold_method="95th_percentile_train_healthy",
                    metrics=metrics_iso,
                ),
                "distance_centroid": AnomalyModelResult(
                    model_name="distance_centroid",
                    model_type="EuclideanCentroidDistance",
                    threshold_method="95th_percentile_train_healthy",
                    metrics=metrics_dist,
                ),
            },
            limitations=limitations,
            seed=self.seed,
        )
