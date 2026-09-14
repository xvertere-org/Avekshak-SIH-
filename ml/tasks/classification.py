"""
Fault Classification Baseline Pipeline for Phase 2 Grey-Box ML.

Implements reusable, leakage-safe classification baselines for CWRU, Paderborn, and NUST.
Evaluates on strictly held-out group-level test sets.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from ml.tasks.adapters import DatasetAdapter
from ml.tasks.preprocessing import LeakageSafePreprocessor
from ml.tasks.evaluation import evaluate_classification
from ml.split_strategy import GroupSplitter, SplitResult, InsufficientGroupsError


@dataclass
class ClassificationModelResult:
    """Evaluation result for a single classification model."""
    model_name: str
    model_type: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FaultClassificationReport:
    """Comprehensive report for a dataset classification baseline."""
    dataset_id: str
    task_name: str = "FAULT_CLASSIFICATION"
    status: str = "completed"  # "completed", "controlled_demonstration", "skipped"
    controlled_demonstration: bool = False
    generalization_claim: bool = True
    target_column: str = ""
    group_column: str = ""
    feature_count: int = 0
    feature_names: List[str] = field(default_factory=list)
    num_classes: int = 0
    class_names: List[str] = field(default_factory=list)
    train_groups: List[Any] = field(default_factory=list)
    test_groups: List[Any] = field(default_factory=list)
    val_groups: List[Any] = field(default_factory=list)
    num_train_samples: int = 0
    num_test_samples: int = 0
    num_val_samples: int = 0
    train_class_distribution: Dict[str, int] = field(default_factory=dict)
    test_class_distribution: Dict[str, int] = field(default_factory=dict)
    models: Dict[str, ClassificationModelResult] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["models"] = {k: v.to_dict() if isinstance(v, ClassificationModelResult) else v for k, v in self.models.items()}
        return d


class FaultClassificationPipeline:
    """
    Executes leakage-safe fault classification baselines.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.splitter = GroupSplitter()

    def run(
        self,
        dataset_id: str,
        target_column: Optional[str] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        run_rf: bool = True,
    ) -> FaultClassificationReport:
        """
        Run classification baseline on the specified dataset.
        """
        adapter = DatasetAdapter(dataset_id)
        df, target_col, group_col, feature_cols = adapter.load_and_adapt(target_column=target_column)

        is_cwru = (dataset_id.lower() == "cwru")
        controlled_demo = is_cwru
        gen_claim = not is_cwru

        # Split data safely
        if is_cwru:
            # Deterministic controlled demonstration on CWRU
            train_df, val_df, test_df, split_res = self.splitter.split(
                df,
                dataset_id,
                group_column=group_col,
                allow_cwru_demo=True,
                label_column=target_col,
                seed=self.seed,
            )
            limitations = [
                "CONTROLLED DEMONSTRATION ONLY: CWRU contains only 3 source files (97.mat, 130.mat, 105.mat).",
                "Each file represents an isolated bearing/fault condition; holding out an entire file holds out an entire class.",
                "Zero statistical generalization claim to unobserved operational machinery.",
            ]
        else:
            train_df, val_df, test_df, split_res = self.splitter.split(
                df,
                dataset_id,
                group_column=group_col,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio,
                label_column=target_col,
                seed=self.seed,
            )
            limitations = adapter.config.excluded_features + [
                f"Source-file level grouping on '{group_col}' guarantees 0 file-level leakage.",
                "Benchmarking results are component-specific and not validated on complete Rotax aero engines.",
            ]

        # Extract feature arrays
        X_train_raw = train_df[feature_cols].to_numpy(dtype=float)
        y_train = train_df[target_col].to_numpy()

        X_test_raw = test_df[feature_cols].to_numpy(dtype=float)
        y_test = test_df[target_col].to_numpy()

        # Training-only preprocessor
        preprocessor = LeakageSafePreprocessor(with_imputer=True)
        X_train = preprocessor.fit_transform(X_train_raw, feature_names=feature_cols)
        X_test = preprocessor.transform(X_test_raw)

        # Classes
        train_classes = sorted(list(set(y_train)))
        all_eval_classes = sorted(list(set(y_train).union(set(y_test))))

        # Train Baseline 1: DummyClassifier
        dummy = DummyClassifier(strategy="most_frequent", random_state=self.seed)
        dummy.fit(X_train, y_train)
        y_pred_dummy = dummy.predict(X_test)
        metrics_dummy = evaluate_classification(y_test, y_pred_dummy, labels=all_eval_classes)

        # Train Baseline 2: LogisticRegression
        lr = LogisticRegression(max_iter=1000, random_state=self.seed)
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        metrics_lr = evaluate_classification(y_test, y_pred_lr, labels=all_eval_classes)

        model_results = {
            "dummy_most_frequent": ClassificationModelResult(
                model_name="dummy_most_frequent",
                model_type="DummyClassifier",
                hyperparameters={"strategy": "most_frequent"},
                metrics=metrics_dummy,
            ),
            "logistic_regression": ClassificationModelResult(
                model_name="logistic_regression",
                model_type="LogisticRegression",
                hyperparameters={"max_iter": 1000, "random_state": self.seed},
                metrics=metrics_lr,
            ),
        }

        # Train Baseline 3: RandomForestClassifier (if requested)
        if run_rf:
            rf = RandomForestClassifier(
                n_estimators=50, max_depth=10, random_state=self.seed, n_jobs=-1
            )
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            metrics_rf = evaluate_classification(y_test, y_pred_rf, labels=all_eval_classes)
            model_results["random_forest"] = ClassificationModelResult(
                model_name="random_forest",
                model_type="RandomForestClassifier",
                hyperparameters={"n_estimators": 50, "max_depth": 10, "random_state": self.seed},
                metrics=metrics_rf,
            )

        status_str = "controlled_demonstration" if controlled_demo else "completed"

        return FaultClassificationReport(
            dataset_id=dataset_id,
            task_name="FAULT_CLASSIFICATION",
            status=status_str,
            controlled_demonstration=controlled_demo,
            generalization_claim=gen_claim,
            target_column=target_col,
            group_column=group_col,
            feature_count=len(feature_cols),
            feature_names=feature_cols,
            num_classes=len(train_classes),
            class_names=[str(c) for c in train_classes],
            train_groups=split_res.train_groups,
            test_groups=split_res.test_groups,
            val_groups=split_res.val_groups,
            num_train_samples=len(train_df),
            num_test_samples=len(test_df),
            num_val_samples=len(val_df),
            train_class_distribution=split_res.train_class_distribution,
            test_class_distribution=split_res.test_class_distribution,
            models=model_results,
            limitations=limitations,
            seed=self.seed,
        )
