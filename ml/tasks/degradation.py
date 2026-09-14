"""
Degradation & Remaining Useful Life (RUL) Baseline Pipeline for Phase 2 Grey-Box ML.

Guarantees:
- Temporal / cycle ordering is strictly preserved per entity (zero shuffling across time).
- Preprocessing fitted strictly on training entities.
- Supports entity-held-out evaluation.
- Computes MAE, RMSE, R2, per-entity error, and near-EOL metrics.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

from ml.tasks.adapters import DatasetAdapter
from ml.tasks.preprocessing import LeakageSafePreprocessor
from ml.tasks.evaluation import evaluate_regression
from ml.tasks.exceptions import TargetNotFoundError, TemporalOrderingError
from ml.split_strategy import GroupSplitter, SplitResult


@dataclass
class RegressionModelResult:
    """Evaluation metrics for a single degradation / RUL model."""
    model_name: str
    model_type: str
    hyperparameters: Dict[str, Any]
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DegradationReport:
    """Comprehensive evaluation report for degradation and RUL tasks."""
    dataset_id: str
    task_name: str = "DEGRADATION_RUL_REGRESSION"
    status: str = "completed"
    target_column: str = ""
    group_column: str = ""
    time_column: Optional[str] = None
    split_type: str = "entity_held_out"  # "entity_held_out", "temporal", "combined"
    feature_count: int = 0
    feature_names: List[str] = field(default_factory=list)
    num_total_entities: int = 0
    num_train_entities: int = 0
    num_test_entities: int = 0
    train_entities: List[Any] = field(default_factory=list)
    test_entities: List[Any] = field(default_factory=list)
    num_train_samples: int = 0
    num_test_samples: int = 0
    per_entity_test_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    models: Dict[str, RegressionModelResult] = field(default_factory=dict)
    limitations: List[str] = field(default_factory=list)
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["models"] = {k: v.to_dict() if isinstance(v, RegressionModelResult) else v for k, v in self.models.items()}
        return d


class DegradationPipeline:
    """
    Executes leakage-safe degradation and RUL regression baselines.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.splitter = GroupSplitter()

    def run(
        self,
        dataset_id: str,
        target_column: Optional[str] = None,
        train_ratio: float = 0.7,
        test_ratio: float = 0.3,
        eol_threshold: Optional[float] = None,
        run_rf: bool = True,
    ) -> DegradationReport:
        """
        Run degradation / RUL baseline for FEMTO, NASA Battery, or C-MAPSS.
        """
        adapter = DatasetAdapter(dataset_id)

        # Dataset-specific file selection
        if dataset_id.lower() == "femto":
            df, target_col, group_col, feature_cols = adapter.load_and_adapt(
                target_column=target_column, split="train"
            )
        elif dataset_id.lower() == "cmapss":
            df, target_col, group_col, feature_cols = adapter.load_and_adapt(
                target_column=target_column, subset="FD001", split="train"
            )
        else:
            df, target_col, group_col, feature_cols = adapter.load_and_adapt(
                target_column=target_column
            )

        # Drop rows with null target (e.g. non-discharge battery cycles)
        valid_mask = df[target_col].notnull()
        df = df[valid_mask].copy()

        if len(df) == 0:
            raise TargetNotFoundError(f"No records with valid target '{target_col}' found in '{dataset_id}'.")

        time_col = adapter.config.time_column

        # Guarantee temporal ordering per entity
        if time_col and time_col in df.columns:
            df = df.sort_values(by=[group_col, time_col]).reset_index(drop=True)

        # Partition by entity (entity-held-out)
        # For FEMTO: 6 total bearings (4 train, 2 test)
        if dataset_id.lower() == "femto":
            train_df, _, test_df, split_res = self.splitter.split(
                df,
                dataset_id,
                group_column=group_col,
                train_ratio=0.67,
                val_ratio=0.0,
                test_ratio=0.33,
                seed=self.seed,
            )
            limitations = [
                "FEMTO evaluation uses 6 total bearings from the processed training inventory: 4 train, 2 test.",
                "Zero validation split is created at entity level to avoid starving training variance.",
                "Test set consists of complete run-to-failure bearing trajectories with preserved temporal ordering.",
            ]
            if eol_threshold is None:
                eol_threshold = 200.0  # Final 200 time-steps before failure
        elif dataset_id.lower() == "cmapss":
            train_df, _, test_df, split_res = self.splitter.split(
                df,
                dataset_id,
                group_column=group_col,
                train_ratio=train_ratio,
                val_ratio=0.0,
                test_ratio=test_ratio,
                seed=self.seed,
            )
            limitations = [
                "C-MAPSS FD001 turbofan degradation benchmark: evaluated on 100 engine units.",
                "Entity-held-out split guarantees zero engine unit leakage.",
                "Target is piecewise linear RUL (capped at 125 cycles).",
            ]
            if eol_threshold is None:
                eol_threshold = 30.0  # Last 30 cycles before failure
        else:  # nasa_battery
            train_df, _, test_df, split_res = self.splitter.split(
                df,
                dataset_id,
                group_column=group_col,
                train_ratio=train_ratio,
                val_ratio=0.0,
                test_ratio=test_ratio,
                seed=self.seed,
            )
            limitations = [
                "NASA Battery aging benchmark: evaluated on 34 lithium-ion cells.",
                "Filtered strictly to discharge cycles where SOH is physically defined.",
                "SOH is an electrochemical benchmark and must never be treated as Rotax telemetry.",
            ]
            if eol_threshold is None:
                eol_threshold = 0.75  # 75% SOH end-of-life boundary

        # Feature matrices
        X_train_raw = train_df[feature_cols].to_numpy(dtype=float)
        y_train = train_df[target_col].to_numpy(dtype=float)

        X_test_raw = test_df[feature_cols].to_numpy(dtype=float)
        y_test = test_df[target_col].to_numpy(dtype=float)

        # Preprocessor fitted strictly on training entities
        preprocessor = LeakageSafePreprocessor(with_imputer=True)
        X_train = preprocessor.fit_transform(X_train_raw, feature_names=feature_cols)
        X_test = preprocessor.transform(X_test_raw)

        # Baseline 1: DummyRegressor (Mean)
        dummy = DummyRegressor(strategy="mean")
        dummy.fit(X_train, y_train)
        y_pred_dummy = dummy.predict(X_test)
        metrics_dummy = evaluate_regression(y_test, y_pred_dummy, eol_threshold=eol_threshold)

        # Baseline 2: Ridge Regression
        ridge = Ridge(alpha=1.0, random_state=self.seed)
        ridge.fit(X_train, y_train)
        y_pred_ridge = ridge.predict(X_test)
        metrics_ridge = evaluate_regression(y_test, y_pred_ridge, eol_threshold=eol_threshold)

        model_results = {
            "dummy_mean": RegressionModelResult(
                model_name="dummy_mean",
                model_type="DummyRegressor",
                hyperparameters={"strategy": "mean"},
                metrics=metrics_dummy,
            ),
            "ridge_regression": RegressionModelResult(
                model_name="ridge_regression",
                model_type="Ridge",
                hyperparameters={"alpha": 1.0, "random_state": self.seed},
                metrics=metrics_ridge,
            ),
        }

        # Baseline 3: RandomForestRegressor (if requested)
        if run_rf:
            rf = RandomForestRegressor(
                n_estimators=50, max_depth=10, random_state=self.seed, n_jobs=-1
            )
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            metrics_rf = evaluate_regression(y_test, y_pred_rf, eol_threshold=eol_threshold)
            model_results["random_forest"] = RegressionModelResult(
                model_name="random_forest",
                model_type="RandomForestRegressor",
                hyperparameters={"n_estimators": 50, "max_depth": 10, "random_state": self.seed},
                metrics=metrics_rf,
            )

        # Per-entity test evaluation (using primary ML model, e.g. RF or Ridge)
        primary_preds = y_pred_rf if run_rf else y_pred_ridge
        per_entity = {}
        for entity_val in split_res.test_groups:
            ent_mask = (test_df[group_col] == entity_val).to_numpy()
            if np.any(ent_mask):
                ent_y_true = y_test[ent_mask]
                ent_y_pred = primary_preds[ent_mask]
                per_entity[str(entity_val)] = {
                    "sample_count": int(np.sum(ent_mask)),
                    "mae": float(np.mean(np.abs(ent_y_true - ent_y_pred))),
                    "rmse": float(np.sqrt(np.mean((ent_y_true - ent_y_pred) ** 2))),
                }

        return DegradationReport(
            dataset_id=dataset_id,
            task_name="DEGRADATION_RUL_REGRESSION",
            status="completed",
            target_column=target_col,
            group_column=group_col,
            time_column=time_col,
            split_type="entity_held_out",
            feature_count=len(feature_cols),
            feature_names=feature_cols,
            num_total_entities=split_res.num_total_groups,
            num_train_entities=split_res.num_train_groups,
            num_test_entities=split_res.num_test_groups,
            train_entities=split_res.train_groups,
            test_entities=split_res.test_groups,
            num_train_samples=len(train_df),
            num_test_samples=len(test_df),
            per_entity_test_metrics=per_entity,
            models=model_results,
            limitations=limitations,
            seed=self.seed,
        )
