"""
Standardized Evaluation Metrics for Phase 2 Grey-Box ML Baselines.

Computes exact mathematical metrics for classification, anomaly detection,
and degradation/RUL regression tasks.
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def evaluate_classification(
    y_true: Union[List[Any], np.ndarray],
    y_pred: Union[List[Any], np.ndarray],
    labels: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """
    Calculate comprehensive classification performance metrics.
    """
    y_t = np.asarray(y_true)
    y_p = np.asarray(y_pred)

    if labels is None:
        unique_labels = sorted(list(set(y_t).union(set(y_p))))
    else:
        unique_labels = labels

    acc = float(accuracy_score(y_t, y_p))
    bal_acc = float(balanced_accuracy_score(y_t, y_p))
    macro_f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_t, y_p, average="weighted", zero_division=0))
    cm = confusion_matrix(y_t, y_p, labels=unique_labels).tolist()

    report_dict = classification_report(
        y_t, y_p, labels=unique_labels, output_dict=True, zero_division=0
    )

    per_class = {}
    for lbl in unique_labels:
        lbl_str = str(lbl)
        if lbl_str in report_dict:
            per_class[lbl_str] = {
                "precision": float(report_dict[lbl_str]["precision"]),
                "recall": float(report_dict[lbl_str]["recall"]),
                "f1_score": float(report_dict[lbl_str]["f1-score"]),
                "support": int(report_dict[lbl_str]["support"]),
            }

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "labels": [str(lbl) for lbl in unique_labels],
        "confusion_matrix": cm,
        "per_class": per_class,
    }


def evaluate_anomaly_detection(
    y_true_binary: Union[List[int], np.ndarray],
    y_pred_binary: Union[List[int], np.ndarray],
    scores: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Calculate anomaly detection metrics where:
      0 = Normal / Healthy reference
      1 = Anomaly / Faulty condition
    """
    y_t = np.asarray(y_true_binary, dtype=int)
    y_p = np.asarray(y_pred_binary, dtype=int)

    prec = float(precision_score(y_t, y_p, pos_label=1, zero_division=0))
    rec = float(recall_score(y_t, y_p, pos_label=1, zero_division=0))
    f1 = float(f1_score(y_t, y_p, pos_label=1, zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_t, y_p))

    # Confusion matrix breakdown: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    score_stats = {}
    if scores is not None:
        sc = np.asarray(scores, dtype=float)
        score_stats = {
            "min": float(np.min(sc)),
            "mean": float(np.mean(sc)),
            "std": float(np.std(sc)),
            "median": float(np.median(sc)),
            "max": float(np.max(sc)),
        }

    return {
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "balanced_accuracy": bal_acc,
        "false_positive_rate": fpr,
        "false_negative_rate": fnr,
        "confusion_counts": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
        },
        "score_distribution": score_stats,
    }


def evaluate_regression(
    y_true: Union[List[float], np.ndarray],
    y_pred: Union[List[float], np.ndarray],
    eol_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Calculate regression performance metrics for degradation and RUL tasks.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    mae = float(mean_absolute_error(y_t, y_p))
    rmse = float(np.sqrt(mean_squared_error(y_t, y_p)))
    
    # R2 can be negative for poor predictors
    try:
        r2 = float(r2_score(y_t, y_p))
    except Exception:
        r2 = float("nan")

    res = {
        "mae": mae,
        "rmse": rmse,
        "r2_score": r2,
        "max_error": float(np.max(np.abs(y_t - y_p))),
        "target_range": [float(np.min(y_t)), float(np.max(y_t))],
    }

    if eol_threshold is not None:
        # Near-EOL subset: where true target is <= eol_threshold (e.g. low RUL remaining)
        mask = y_t <= eol_threshold
        if np.any(mask):
            late_mae = float(mean_absolute_error(y_t[mask], y_p[mask]))
            late_rmse = float(np.sqrt(mean_squared_error(y_t[mask], y_p[mask])))
            res["near_eol_metrics"] = {
                "threshold": eol_threshold,
                "sample_count": int(np.sum(mask)),
                "mae": late_mae,
                "rmse": late_rmse,
            }

    return res
