"""Classification metrics used throughout the project.

The research paper reports Accuracy, Precision, Recall and F1-score. These are
computed on the failure class (``pos_label=1``) using scikit-learn, exactly as
a binary predictive-maintenance evaluation would be performed.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

METRIC_NAMES = ("accuracy", "precision", "recall", "f1")


def compute_metrics(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
    prefix: str = "",
) -> dict[str, float]:
    """Compute accuracy / precision / recall / F1 (failure = positive class)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length.")

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }
    if prefix:
        metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
    return metrics


def confusion(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def format_metrics(metrics: dict[str, float], decimals: int = 4) -> str:
    return ", ".join(f"{k}={v:.{decimals}f}" for k, v in metrics.items())


def weighted_average_metrics(
    per_client_metrics: list[dict[str, float]], weights: list[int]
) -> dict[str, float]:
    """Aggregate per-client metrics weighted by the number of samples."""
    total = sum(weights)
    if total <= 0:
        raise ValueError("Total weight must be positive.")
    out: dict[str, float] = {}
    for key in METRIC_NAMES:
        out[key] = sum(
            m.get(key, 0.0) * w for m, w in zip(per_client_metrics, weights)
        ) / total
    return out