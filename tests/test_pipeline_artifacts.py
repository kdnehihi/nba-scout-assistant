from __future__ import annotations

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from config.long_term_config import resolve_long_term_task_config
from config.lstm_config import LSTM_TASK_CONFIG
from models.lstm import ShortTermLSTM
from models.mlp import LongTermMLP
from pipelines.artifacts import (
    load_long_term_model_artifact,
    load_long_term_model_artifacts,
    load_short_term_model_artifact,
    load_short_term_model_artifacts,
)


def test_load_short_term_model_artifact_restores_lstm(tmp_path):
    task = "points"
    config = LSTM_TASK_CONFIG[task]
    seq_scaler = StandardScaler().fit(np.zeros((4, 2)))
    static_scaler = StandardScaler().fit(np.zeros((4, 2)))
    model = ShortTermLSTM(input_size=2, hidden_size=config.hidden_size, static_size=2, dropout=config.dropout)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "task": task,
            "hidden_size": config.hidden_size,
            "dropout": config.dropout,
            "seq_scaler": seq_scaler,
            "static_scaler": static_scaler,
            "y_scaler": None,
        },
        tmp_path / "short_term_lstm_points.pt",
    )

    artifact = load_short_term_model_artifact(tmp_path, task=task, device="cpu")
    artifacts = load_short_term_model_artifacts(tmp_path, tasks=(task,), device="cpu")

    assert artifact.task == "points"
    assert artifact.model.training is False
    assert artifacts["points"].task_config == config


def test_load_long_term_random_forest_artifact_uses_saved_schema(tmp_path):
    config = resolve_long_term_task_config("pts_per_36", 1)
    model_stem = f"long_term_{config.task}_h{config.horizon}_{config.model_family}"
    joblib.dump(
        {
            "model": "fitted-random-forest-placeholder",
            "task_config": config,
            "feature_cols": ["age", "minutes"],
            "metrics": [],
        },
        tmp_path / f"{model_stem}.joblib",
    )

    artifact = load_long_term_model_artifact(tmp_path, config, device="cpu")

    assert artifact.model == "fitted-random-forest-placeholder"
    assert artifact.feature_cols == ["age", "minutes"]
    assert artifact.preprocessor is None


def test_load_long_term_mlp_artifact_restores_model_and_preprocessor(tmp_path):
    config = resolve_long_term_task_config("pts_per_36", 3)
    model_stem = f"long_term_{config.task}_h{config.horizon}_{config.model_family}"
    input_size = 3
    params = config.model_params
    model = LongTermMLP(
        input_size=input_size,
        hidden_sizes=tuple(params["hidden_sizes"]),
        dropout=float(params["dropout"]),
        batch_norm=bool(params["batch_norm"]),
    )
    preprocessor = StandardScaler().fit(np.zeros((4, input_size)))

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "task_config": config,
            "feature_cols": ["age", "minutes", "usage_pct"],
            "target_scaler": None,
            "input_size": input_size,
            "metrics": [],
        },
        tmp_path / f"{model_stem}.pt",
    )
    joblib.dump(preprocessor, tmp_path / f"{model_stem}_preprocessor.joblib")

    artifact = load_long_term_model_artifact(tmp_path, config, device="cpu")
    artifacts = load_long_term_model_artifacts(tmp_path, task_configs=(config,), device="cpu")

    assert artifact.model.training is False
    assert artifact.feature_cols == ["age", "minutes", "usage_pct"]
    assert artifact.preprocessor is not None
    assert artifacts[(config.task, config.horizon)].model_family == "mlp"
