from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    # Compute standard regression metrics for one target.
    """Return MAE, RMSE, and R2 for numeric predictions."""
    truth = np.asarray(y_true, dtype="float64")
    prediction = np.asarray(y_pred, dtype="float64")
    return {
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(mean_squared_error(truth, prediction) ** 0.5),
        "r2": float(r2_score(truth, prediction)),
    }


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_probability: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    # Compute binary classification metrics from probabilities.
    """Return ROC-AUC, F1, accuracy, and Brier score for binary probabilities."""
    truth = np.asarray(y_true, dtype="int64")
    probability = np.asarray(y_probability, dtype="float64")
    prediction = (probability >= threshold).astype("int64")
    metrics = {
        "f1": float(f1_score(truth, prediction, zero_division=0)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "brier": float(brier_score_loss(truth, probability)),
    }
    if len(np.unique(truth)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(truth, probability))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def range_metrics(
    y_true: pd.Series | np.ndarray,
    expected: pd.Series | np.ndarray,
    lower: pd.Series | np.ndarray,
    upper: pd.Series | np.ndarray,
) -> dict[str, float]:
    # Evaluate point prediction quality and interval coverage.
    """Return regression metrics plus coverage and average interval width."""
    truth = np.asarray(y_true, dtype="float64")
    expected_values = np.asarray(expected, dtype="float64")
    lower_values = np.asarray(lower, dtype="float64")
    upper_values = np.asarray(upper, dtype="float64")
    output = regression_metrics(truth, expected_values)
    output["coverage_rate"] = float(((truth >= lower_values) & (truth <= upper_values)).mean())
    output["avg_range_width"] = float((upper_values - lower_values).mean())
    return output

