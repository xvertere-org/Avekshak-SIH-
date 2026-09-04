"""
Evaluation module for Phase 8 Fault Diagnosis.

Computes comprehensive classification metrics on held-out test data:
- Primary: Macro F1 across all 6 canonical fault classes
- Per-class Precision, Recall, and F1
- Balanced Accuracy and Weighted F1
- 6x6 Confusion Matrix
- NONE False-Positive Rate and False Alarm Rate
- SENSOR_FAULT Recall
- Independent-run counts and validation disclaimers
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    balanced_accuracy_score,
    confusion_matrix,
)

from fault_diagnosis.schema import CANONICAL_FAULT_LABELS


def evaluate_classifier(
    classifier: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_labels: Optional[List[str]] = None,
    test_run_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Evaluate trained fault classifier on held-out test data.

    Args:
        classifier: Trained XGBoostFaultClassifier or compatible model.
        X_test: Test feature matrix (16 columns).
        y_test: Test true fault labels (canonical strings).
        class_labels: Canonical fault label list (defaults to CANONICAL_FAULT_LABELS).
        test_run_counts: Optional dict mapping class -> number of test runs.

    Returns:
        Dict containing:
        - macro_f1: Macro-averaged F1 across all classes (Primary metric)
        - balanced_accuracy: Balanced accuracy across classes
        - weighted_f1: Sample-weighted F1
        - per_class: Dict mapping class -> {precision, recall, f1, support, test_runs}
        - confusion_matrix: 2D list (rows: true, cols: predicted)
        - confusion_matrix_labels: Class labels for the matrix rows/columns
        - none_false_positive_rate: False positive rate for the NONE class
        - none_false_alarm_rate: False alarm rate when true label is NONE
        - sensor_fault_recall: Recall specifically for sensor_fault
        - disclaimers: List of important evaluation caveats
    """
    if class_labels is None:
        class_labels = list(CANONICAL_FAULT_LABELS)

    # Make predictions
    y_pred = classifier.predict(X_test)
    if isinstance(y_pred, np.ndarray):
        y_pred = list(y_pred)
    y_true = list(y_test)

    # 1. Primary & overall metrics
    macro_f1 = float(f1_score(y_true, y_pred, labels=class_labels, average="macro", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=class_labels, average="weighted", zero_division=0))

    # 2. Per-class metrics
    precisions = precision_score(y_true, y_pred, labels=class_labels, average=None, zero_division=0)
    recalls = recall_score(y_true, y_pred, labels=class_labels, average=None, zero_division=0)
    f1s = f1_score(y_true, y_pred, labels=class_labels, average=None, zero_division=0)

    per_class: Dict[str, Dict[str, Any]] = {}
    for i, cls in enumerate(class_labels):
        support = sum(1 for y in y_true if y == cls)
        run_count = test_run_counts.get(cls, None) if test_run_counts else None
        per_class[cls] = {
            "precision": float(precisions[i]),
            "recall": float(recalls[i]),
            "f1": float(f1s[i]),
            "support": support,
            "test_runs": run_count,
        }

    # 3. Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    cm_list = cm.tolist()

    # 4. NONE-specific metrics
    # Case A: False Alarm Rate — True is NONE, but predicted is a FAULT
    none_total = sum(1 for y in y_true if y == "none")
    none_false_alarms = sum(1 for yt, yp in zip(y_true, y_pred) if yt == "none" and yp != "none")
    none_false_alarm_rate = float(none_false_alarms / none_total) if none_total > 0 else 0.0

    # Case B: NONE False Positive Rate — True is FAULT, but predicted is NONE (missed fault as healthy)
    fault_total = sum(1 for y in y_true if y != "none")
    false_none_preds = sum(1 for yt, yp in zip(y_true, y_pred) if yt != "none" and yp == "none")
    none_false_positive_rate = float(false_none_preds / fault_total) if fault_total > 0 else 0.0

    # 5. SENSOR_FAULT recall
    sensor_recall = per_class.get("sensor_fault", {}).get("recall", 0.0)

    # 6. Honest Disclaimers
    disclaimers = [
        "High synthetic-data performance does not establish real-world MALE-UAV fault classification validity.",
        "Classes with only 1 independent test run (e.g. physical faults) reflect evaluation on a single trajectory (~1,720 temporally correlated samples).",
        "Temporally correlated telemetry samples within the same mission run do NOT constitute thousands of independent observations.",
    ]

    if test_run_counts:
        for cls, count in test_run_counts.items():
            if count <= 1:
                disclaimers.append(
                    f"Class '{cls}' has only {count} test run(s). Metrics should be interpreted with extreme caution."
                )

    return {
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "confusion_matrix": cm_list,
        "confusion_matrix_labels": class_labels,
        "none_false_positive_rate": none_false_positive_rate,
        "none_false_alarm_rate": none_false_alarm_rate,
        "sensor_fault_recall": float(sensor_recall),
        "disclaimers": disclaimers,
    }


def format_confusion_matrix_ascii(
    cm: List[List[int]],
    labels: List[str],
) -> str:
    """Format confusion matrix as readable ASCII table."""
    short_labels = [lbl[:10] for lbl in labels]
    title_col = "True \\ Pred"
    header = f"{title_col:>14} | " + " | ".join(f"{lbl:>10}" for lbl in short_labels)
    sep = "-" * len(header)
    lines = [header, sep]

    for i, true_lbl in enumerate(short_labels):
        row_str = f"{true_lbl:>14} | " + " | ".join(f"{cm[i][j]:>10}" for j in range(len(labels)))
        lines.append(row_str)

    return "\n".join(lines)
