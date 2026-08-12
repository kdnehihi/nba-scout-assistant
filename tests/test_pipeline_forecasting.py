from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config.long_term_config import resolve_long_term_task_config
from src.config.lstm_config import LSTM_TASK_CONFIG
from src.dataset.sequence import make_lstm_delta_inference_window
from src.modeling.long_term_baseline import build_long_term_preprocessor
from src.models.lstm import ShortTermLSTM
from src.pipelines.artifacts import LongTermModelArtifact, ShortTermModelArtifact
from src.pipelines.forecasting import predict_long_term_task, predict_long_term_tasks, predict_short_term_task, predict_short_term_tasks


def make_short_term_artifact(task: str = "points") -> ShortTermModelArtifact:
    config = LSTM_TASK_CONFIG[task]
    model = ShortTermLSTM(input_size=2, hidden_size=config.hidden_size, static_size=2, dropout=config.dropout)
    seq_scaler = StandardScaler().fit(np.zeros((4, 2)))
    static_scaler = StandardScaler().fit(np.zeros((4, 2)))
    return ShortTermModelArtifact(
        task=task,
        task_config=config,
        model=model,
        seq_scaler=seq_scaler,
        static_scaler=static_scaler,
        y_scaler=None,
        checkpoint={},
    )


def make_short_term_rows(task: str = "points", n_rows: int = 12) -> pd.DataFrame:
    config = LSTM_TASK_CONFIG[task]
    rows = []
    for idx in range(n_rows):
        rows.append(
            {
                "player_id": 1,
                "player_name": "Player One",
                "season": "2024-25",
                "as_of_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=idx),
                "game_id": idx,
                config.stat_col: 20 + idx,
                config.stat_avg_col: 22.0,
                "min": 30 + idx % 3,
                "min_season_avg": 31.0,
            }
        )
    return pd.DataFrame(rows)


def test_predict_short_term_task_returns_one_prediction():
    artifact = make_short_term_artifact()
    rows = make_short_term_rows()

    prediction = predict_short_term_task(rows, artifact, player_id=1, season="2024-25", device="cpu")

    assert prediction["task"] == "points"
    assert prediction["player_id"] == 1
    assert isinstance(prediction["prediction"], float)


def test_lstm_inference_window_reuses_training_sequence_shape():
    config = LSTM_TASK_CONFIG["points"]

    X_seq, X_static, baseline, anchor = make_lstm_delta_inference_window(
        make_short_term_rows("points", n_rows=12),
        config,
    )

    assert X_seq.shape == (1, config.sequence_length, 2)
    assert X_static.shape == (1, 2)
    assert baseline == 22.0
    assert anchor["game_id"] == 11


def test_predict_short_term_tasks_filters_requested_tasks():
    artifacts = {"points": make_short_term_artifact("points")}
    rows = make_short_term_rows("points")

    predictions = predict_short_term_tasks(rows, artifacts, tasks=["points"], player_name="Player One", device="cpu")

    assert set(predictions) == {"points"}


def make_long_term_rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "player_name": ["Player One", "Player Two", "Player Three"],
            "anchor_season": ["2021-22", "2021-22", "2021-22"],
            "anchor_season_start_year": [2021, 2021, 2021],
            "age": [27, 28, 29],
            "minutes": [1800, 1600, 1400],
            "usage_pct": [0.24, 0.20, 0.18],
            "pts_per_36_h1": [20.0, 16.0, 12.0],
            "split": ["test", "test", "test"],
        }
    )


def make_long_term_artifact() -> LongTermModelArtifact:
    config = resolve_long_term_task_config("pts_per_36", 1)
    feature_cols = ["age", "minutes", "usage_pct"]
    train_X = make_long_term_rows()[feature_cols]
    model = Pipeline(
        [
            ("preprocess", build_long_term_preprocessor(train_X)),
            ("model", DummyRegressor(strategy="constant", constant=18.0)),
        ]
    )
    model.fit(train_X, make_long_term_rows()["pts_per_36_h1"])
    return LongTermModelArtifact(
        task=config.task,
        horizon=config.horizon,
        task_config=config,
        model_family="random_forest",
        model=model,
        feature_cols=feature_cols,
    )


def test_predict_long_term_task_returns_one_prediction():
    artifact = make_long_term_artifact()

    prediction = predict_long_term_task(make_long_term_rows(), artifact, player_id=1)

    assert prediction["task"] == "pts_per_36"
    assert prediction["horizon"] == 1
    assert prediction["prediction"] == 18.0


def test_predict_long_term_tasks_filters_task_and_horizon():
    artifact = make_long_term_artifact()
    artifacts = {(artifact.task, artifact.horizon): artifact}

    predictions = predict_long_term_tasks(
        make_long_term_rows(),
        artifacts,
        tasks=["pts_per_36"],
        horizons=[1],
        player_name="Player One",
    )

    assert predictions["pts_per_36"][1]["prediction"] == 18.0
