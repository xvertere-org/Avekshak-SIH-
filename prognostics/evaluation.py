"""
Prognostics evaluation metrics: NASA PHM08 asymmetric scoring, PICP target, MPIW, MAE, and RMSE.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np


def compute_phm08_score(
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Compute the NASA PHM08 Prognostics asymmetric scoring function (Saxena et al., 2008).

    Error convention:
        d_j = y_pred_j - y_true_j = RUL_hat - RUL_true

    Penalty function:
        s_j = exp(-d_j / 13) - 1   if d_j < 0  (early prediction / conservative)
        s_j = exp(d_j / 10) - 1    if d_j >= 0 (late prediction / hazardous)

    Returns:
        Tuple of (total_score: float, per_sample_penalties: np.ndarray)
    """
    y_p = np.asarray(y_pred, dtype=np.float64)
    y_t = np.asarray(y_true, dtype=np.float64)

    if len(y_p) != len(y_t):
        raise ValueError(f"Length mismatch: y_pred ({len(y_p)}) vs y_true ({len(y_t)})")

    valid = np.isfinite(y_p) & np.isfinite(y_t)
    if not np.any(valid):
        return float("nan"), np.array([], dtype=np.float64)

    d = y_p[valid] - y_t[valid]
    penalties = np.zeros_like(d)

    # Early predictions (d < 0)
    early_mask = d < 0
    penalties[early_mask] = np.exp(-d[early_mask] / 13.0) - 1.0

    # Late predictions (d >= 0)
    late_mask = ~early_mask
    penalties[late_mask] = np.exp(d[late_mask] / 10.0) - 1.0

    total_score = float(np.sum(penalties))
    return total_score, penalties


def compute_picp_and_mpiw(
    y_true: np.ndarray,
    y_p05: np.ndarray,
    y_p95: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute Prediction Interval Coverage Probability (PICP) and Mean Prediction Interval Width (MPIW).

    PICP: Fraction of true RUL values that fall within [P05, P95].
          Note: >= 90% is an evaluation target, not a hard acceptance gate.
    MPIW: Average width (P95 - P05). Always reported alongside PICP to assess interval sharpness.

    Returns:
        Tuple of (picp: float, mpiw: float)
    """
    yt = np.asarray(y_true, dtype=np.float64)
    p05 = np.asarray(y_p05, dtype=np.float64)
    p95 = np.asarray(y_p95, dtype=np.float64)

    valid = np.isfinite(yt) & np.isfinite(p05) & np.isfinite(p95)
    if not np.any(valid):
        return float("nan"), float("nan")

    v_yt = yt[valid]
    v_p05 = p05[valid]
    v_p95 = p95[valid]

    covered = (v_yt >= v_p05) & (v_yt <= v_p95)
    picp = float(np.mean(covered))

    widths = v_p95 - v_p05
    mpiw = float(np.mean(widths))

    return picp, mpiw


def compute_prognostic_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_p05: Optional[np.ndarray] = None,
    y_p95: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive prognostic performance metrics for Phase 11.
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)

    valid = np.isfinite(yt) & np.isfinite(yp)
    n_valid = int(np.sum(valid))

    if n_valid == 0:
        return {
            "n_samples": 0,
            "mae_s": float("nan"),
            "rmse_s": float("nan"),
            "phm08_score": float("nan"),
            "picp": float("nan"),
            "mpiw_s": float("nan"),
        }

    errors = yp[valid] - yt[valid]
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    phm08, _ = compute_phm08_score(yp[valid], yt[valid])

    metrics: Dict[str, Any] = {
        "n_samples": n_valid,
        "mae_s": round(mae, 2),
        "rmse_s": round(rmse, 2),
        "phm08_score": round(phm08, 4),
    }

    if y_p05 is not None and y_p95 is not None:
        picp, mpiw = compute_picp_and_mpiw(yt, y_p05, y_p95)
        metrics["picp"] = round(picp, 4)
        metrics["picp_meets_target"] = picp >= 0.90
        metrics["mpiw_s"] = round(mpiw, 2)

    return metrics
