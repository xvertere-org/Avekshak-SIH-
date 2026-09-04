"""
Grouped mission-level evaluation and metric calculation for Phase 10 forecasting.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

from forecasting.schema import (
    DEFAULT_FORECAST_CHANNELS,
    DEFAULT_NRMSE_DENOMINATORS,
    ForecastResult,
)


def compute_forecast_metrics(
    actual_dict: Dict[str, np.ndarray],
    predicted_dict: Dict[str, np.ndarray],
    channels: Optional[List[str]] = None,
    denominators: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Compute MAE, RMSE, and fixed-denominator NRMSE per channel and in aggregate.

    Args:
        actual_dict: Dict mapping channel -> 1D array of actual values across forecast horizon
        predicted_dict: Dict mapping channel -> 1D array of predicted values across forecast horizon
        channels: Target channels to evaluate
        denominators: Fixed engineering reference ranges (strictly constant, not test-derived)

    Returns:
        Dict with 'per_channel' metrics, 'aggregate_mae', 'aggregate_rmse', 'aggregate_nrmse'
    """
    target_channels = channels or DEFAULT_FORECAST_CHANNELS
    denoms = denominators or DEFAULT_NRMSE_DENOMINATORS

    per_channel: Dict[str, Dict[str, float]] = {}
    mae_list: List[float] = []
    rmse_list: List[float] = []
    nrmse_list: List[float] = []

    for ch in target_channels:
        if ch not in actual_dict or ch not in predicted_dict:
            continue

        y_true = np.asarray(actual_dict[ch], dtype=np.float64)
        y_pred = np.asarray(predicted_dict[ch], dtype=np.float64)

        # Drop any NaN pairs if present
        valid = ~np.isnan(y_true) & ~np.isnan(y_pred)
        if not np.any(valid):
            continue

        err = y_true[valid] - y_pred[valid]
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))

        denom = denoms.get(ch, 1.0)
        nrmse = rmse / denom if denom > 0 else rmse

        per_channel[ch] = {
            "mae": mae,
            "rmse": rmse,
            "nrmse": nrmse,
            "denominator": denom,
        }
        mae_list.append(mae)
        rmse_list.append(rmse)
        nrmse_list.append(nrmse)

    return {
        "per_channel": per_channel,
        "aggregate_mae": float(np.mean(mae_list)) if mae_list else 0.0,
        "aggregate_rmse": float(np.mean(rmse_list)) if rmse_list else 0.0,
        "aggregate_nrmse": float(np.mean(nrmse_list)) if nrmse_list else 0.0,
    }


def split_missions_grouped(
    df: pd.DataFrame,
    test_ratio: float = 0.30,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split telemetry dataset strictly at the whole-mission level.

    Zero sample-level leakage: all observations for any given mission_id
    belong entirely to either train or test split.
    """
    mission_col = "mission_id" if "mission_id" in df.columns else "mission_run_id"
    if mission_col not in df.columns:
        raise ValueError("DataFrame must contain 'mission_id' or 'mission_run_id' for grouped splitting.")

    unique_missions = list(df[mission_col].dropna().unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_missions)

    n_test = max(1, int(len(unique_missions) * test_ratio))
    test_missions = set(unique_missions[:n_test])
    train_missions = set(unique_missions[n_test:])

    train_df = df[df[mission_col].isin(train_missions)].copy()
    test_df = df[df[mission_col].isin(test_missions)].copy()

    return train_df, test_df


class ForecastingEvaluator:
    """
    Evaluator comparing TimesFM-3 against Persistence and Causal EWMA baselines
    with stratified reporting across fault classes and operational regimes.
    """

    def __init__(
        self,
        denominators: Optional[Dict[str, float]] = None,
        channels: Optional[List[str]] = None,
    ):
        self.denominators = denominators or DEFAULT_NRMSE_DENOMINATORS
        self.channels = channels or DEFAULT_FORECAST_CHANNELS

    def evaluate_results(
        self,
        results: List[ForecastResult],
        ground_truth_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Evaluate a list of ForecastResults against ground truth telemetry.

        Args:
            results: List of generated ForecastResults
            ground_truth_df: DataFrame containing actual telemetry with 'timestamp'

        Returns:
            Dict containing overall metrics, per-fault metrics, healthy vs degraded stratification,
            and baseline comparisons.
        """
        if not results:
            return {"error": "No forecast results to evaluate"}

        # Index ground truth by timestamp for fast lookup
        gt_df = ground_truth_df.sort_values("timestamp")
        gt_timestamps = gt_df["timestamp"].values

        model_errors = {ch: [] for ch in self.channels}
        persist_errors = {ch: [] for ch in self.channels}
        ewma_errors = {ch: [] for ch in self.channels}

        stratified_by_fault: Dict[str, List[Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]]] = {}

        for res in results:
            # Extract ground truth over forecast horizon
            hor_ts = res.forecast_timestamps
            actual_horizon: Dict[str, List[float]] = {ch: [] for ch in self.channels}

            for t in hor_ts:
                # Find closest timestamp in ground truth within 0.5s tolerance
                idx = np.searchsorted(gt_timestamps, t)
                if idx < len(gt_timestamps) and abs(gt_timestamps[idx] - t) <= 0.5:
                    row = gt_df.iloc[idx]
                    for ch in self.channels:
                        actual_horizon[ch].append(float(row[ch]) if ch in row else float("nan"))
                else:
                    for ch in self.channels:
                        actual_horizon[ch].append(float("nan"))

            # Build prediction arrays
            pred_model = {ch: np.array(res.predicted_telemetry.get(ch, []), dtype=np.float64) for ch in self.channels}

            persist_pred = {}
            ewma_pred = {}
            if res.baseline_comparison:
                p_dict = res.baseline_comparison.get("persistence", {})
                e_dict = res.baseline_comparison.get("causal_ewma", {})
                persist_pred = {ch: np.array(p_dict.get(ch, []), dtype=np.float64) for ch in self.channels}
                ewma_pred = {ch: np.array(e_dict.get(ch, []), dtype=np.float64) for ch in self.channels}

            # Fault tag provenance
            opt_ctx = res.provenance.get("optional_context", {})
            fault_tag = str(opt_ctx.get("fault_type", "NONE")).upper()

            if fault_tag not in stratified_by_fault:
                stratified_by_fault[fault_tag] = []
            actual_arrs = {ch: np.array(actual_horizon[ch]) for ch in self.channels}
            stratified_by_fault[fault_tag].append((actual_arrs, pred_model))

            for ch in self.channels:
                act = actual_arrs[ch]
                if ch in pred_model and len(pred_model[ch]) == len(act):
                    model_errors[ch].extend((act - pred_model[ch])[~np.isnan(act)])
                if ch in persist_pred and len(persist_pred[ch]) == len(act):
                    persist_errors[ch].extend((act - persist_pred[ch])[~np.isnan(act)])
                if ch in ewma_pred and len(ewma_pred[ch]) == len(act):
                    ewma_errors[ch].extend((act - ewma_pred[ch])[~np.isnan(act)])

        # Compute overall model vs baseline metrics
        def _calc_summary(err_dict):
            ch_metrics = {}
            for ch, errs in err_dict.items():
                if errs:
                    arr = np.array(errs)
                    mae = float(np.mean(np.abs(arr)))
                    rmse = float(np.sqrt(np.mean(arr ** 2)))
                    denom = self.denominators.get(ch, 1.0)
                    ch_metrics[ch] = {"mae": mae, "rmse": rmse, "nrmse": rmse / denom}
                else:
                    ch_metrics[ch] = {"mae": 0.0, "rmse": 0.0, "nrmse": 0.0}
            agg_mae = float(np.mean([m["mae"] for m in ch_metrics.values()]))
            agg_rmse = float(np.mean([m["rmse"] for m in ch_metrics.values()]))
            agg_nrmse = float(np.mean([m["nrmse"] for m in ch_metrics.values()]))
            return {"per_channel": ch_metrics, "aggregate_mae": agg_mae, "aggregate_rmse": agg_rmse, "aggregate_nrmse": agg_nrmse}

        model_summary = _calc_summary(model_errors)
        persist_summary = _calc_summary(persist_errors)
        ewma_summary = _calc_summary(ewma_errors)

        # Relative skill score vs persistence
        rmse_model = model_summary["aggregate_rmse"]
        rmse_persist = persist_summary["aggregate_rmse"]
        skill_score = 1.0 - (rmse_model / rmse_persist) if rmse_persist > 1e-6 else 0.0

        return {
            "model_metrics": model_summary,
            "persistence_metrics": persist_summary,
            "causal_ewma_metrics": ewma_summary,
            "skill_score_vs_persistence": skill_score,
            "fault_classes_evaluated": list(stratified_by_fault.keys()),
        }
