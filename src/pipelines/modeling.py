from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.long_term_config import LONG_TERM_HORIZONS, LONG_TERM_TASKS
from src.training.train_long_term import train_long_term_model, train_long_term_models
from src.training.train_short_term import train_short_term


SHORT_TERM_TASKS = ("points", "assists", "rebounds")


def run_short_term_training_pipeline(
    data_dir: Path | str = "data",
    artifact_dir: Path | str = "artifacts",
    tasks: tuple[str, ...] = SHORT_TERM_TASKS,
    epochs: int = 100,
) -> pd.DataFrame:
    # Train selected short-term LSTM tasks and combine their metrics.
    """Run short-term model training for selected tasks."""
    metrics = [
        train_short_term(
            task=task,
            data_dir=data_dir,
            artifact_dir=artifact_dir,
            epochs=epochs,
        )
        for task in tasks
    ]
    return pd.concat(metrics, ignore_index=True) if metrics else pd.DataFrame()


def run_long_term_training_pipeline(
    data_dir: Path | str = "data",
    artifact_dir: Path | str = "artifacts",
    task: str = "all",
    horizon: int | None = None,
) -> pd.DataFrame:
    # Train selected long-term models using the selected model family per task.
    """Run long-term model training for selected task and horizon filters."""
    if task == "all" and horizon is None:
        return train_long_term_models(data_dir=data_dir, artifact_dir=artifact_dir)

    if task == "all":
        metrics = [
            train_long_term_model(
                task=task_name,
                horizon=int(horizon),
                data_dir=data_dir,
                artifact_dir=artifact_dir,
            )
            for task_name in LONG_TERM_TASKS
        ]
        return pd.concat(metrics, ignore_index=True)

    horizons = [int(horizon)] if horizon is not None else list(LONG_TERM_HORIZONS)
    metrics = [
        train_long_term_model(
            task=task,
            horizon=horizon_value,
            data_dir=data_dir,
            artifact_dir=artifact_dir,
        )
        for horizon_value in horizons
    ]
    return pd.concat(metrics, ignore_index=True)
