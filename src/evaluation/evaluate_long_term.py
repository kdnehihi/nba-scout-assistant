from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.long_term_config import LongTermTaskConfig
from src.evaluation.metrics import classification_metrics, regression_metrics


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def predict_sklearn_long_term(
    model: Pipeline,
    X: pd.DataFrame,
    task_config: LongTermTaskConfig,
) -> np.ndarray:
    """Return regression predictions or positive-class probabilities from a sklearn pipeline."""
    if task_config.task_type == "classification":
        if not hasattr(model, "predict_proba"):
            raise TypeError("Classification model must expose predict_proba.")
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def predict_mlp_long_term(
    model: torch.nn.Module,
    X: np.ndarray,
    batch_size: int,
    task_config: LongTermTaskConfig,
    device: str = DEVICE,
) -> np.ndarray:
    """Return MLP regression predictions or classification probabilities."""
    model.eval()
    predictions: list[np.ndarray] = []
    loader = DataLoader(torch.from_numpy(X.astype("float32")), batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for X_batch in loader:
            raw = model(X_batch.to(device)).detach().cpu().numpy()
            if task_config.task_type == "classification":
                raw = 1 / (1 + np.exp(-raw))
            predictions.append(raw)

    return np.concatenate(predictions)


def evaluate_long_term_predictions_by_split(
    df: pd.DataFrame,
    predictions: np.ndarray,
    task_config: LongTermTaskConfig,
    splits: tuple[str, ...] = ("validation", "test"),
) -> pd.DataFrame:
    """Evaluate long-term predictions separately for validation and test splits."""
    return evaluate_long_term_predictions(
        split=df["split"],
        y_true=df[task_config.target_col],
        y_pred=predictions,
        task_config=task_config,
        splits=splits,
    )


def evaluate_long_term_predictions(
    split: pd.Series | np.ndarray,
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    task_config: LongTermTaskConfig,
    splits: tuple[str, ...] = ("validation", "test"),
) -> pd.DataFrame:
    """Evaluate long-term predictions from split labels, targets, and predictions."""
    rows: list[dict[str, object]] = []
    split_values = np.asarray(split)
    target = np.asarray(y_true)
    prediction = np.asarray(y_pred)

    for split_name in splits:
        split_mask = split_values == split_name
        if not split_mask.any():
            continue

        if task_config.task_type == "classification":
            metrics = classification_metrics(target[split_mask], prediction[split_mask])
        else:
            metrics = regression_metrics(target[split_mask], prediction[split_mask])

        rows.append(
            {
                "task": task_config.task,
                "horizon": task_config.horizon,
                "model_family": task_config.model_family,
                "target": task_config.target_col,
                "split": split_name,
                "rows": int(split_mask.sum()),
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def evaluate_long_term_split_predictions(
    split_name: str,
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    task_config: LongTermTaskConfig,
) -> dict[str, object]:
    """Evaluate one long-term split from true and predicted arrays."""
    if task_config.task_type == "classification":
        metrics = classification_metrics(y_true, y_pred)
    else:
        metrics = regression_metrics(y_true, y_pred)

    return {
        "task": task_config.task,
        "horizon": task_config.horizon,
        "model_family": task_config.model_family,
        "target": task_config.target_col,
        "split": split_name,
        "rows": len(y_true),
        **metrics,
    }


def evaluate_long_term_prediction_frame(
    prediction_df: pd.DataFrame,
    task_config: LongTermTaskConfig,
    splits: tuple[str, ...] = ("validation", "test"),
) -> pd.DataFrame:
    """Evaluate a dataframe with split, target, and prediction columns."""
    return evaluate_long_term_predictions(
        split=prediction_df["split"],
        y_true=prediction_df["target"],
        y_pred=prediction_df["prediction"],
        task_config=task_config,
        splits=splits,
    )
