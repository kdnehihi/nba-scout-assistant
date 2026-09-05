# ruff: noqa: I001
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import xgboost as _xgboost  # noqa: F401  # Load its OpenMP runtime before torch on macOS.
import torch

from src.config.long_term_config import LONG_TERM_HORIZONS, LONG_TERM_FORECAST_TASKS, LongTermTaskConfig
from src.config.lstm_config import LSTM_TASK_CONFIG, LSTMTaskConfig
from src.models.lstm import ShortTermLSTM
from src.models.mlp import LongTermMLP
from src.config.recommendation_config import RECOMMENDATION_RANKER_FILENAME
from src.scouting.ranking import (
    RecommendationRankerArtifact,
    load_recommendation_ranker_artifact as _load_recommendation_ranker_artifact,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class ShortTermModelArtifact:
    """Loaded short-term LSTM model and its preprocessing objects."""

    task: str
    task_config: LSTMTaskConfig
    model: ShortTermLSTM
    seq_scaler: Any
    static_scaler: Any
    y_scaler: Any
    checkpoint: dict[str, Any]


@dataclass(frozen=True)
class LongTermModelArtifact:
    """Loaded long-term model artifact for one task and horizon."""

    task: str
    horizon: int
    task_config: LongTermTaskConfig
    model_family: str
    model: Any
    feature_cols: list[str]
    preprocessor: Any | None = None
    target_scaler: Any | None = None
    checkpoint: dict[str, Any] | None = None


def _require_file(path: Path) -> Path:
    # Fail early when a model artifact expected by serving is missing.
    """Return an existing file path or raise FileNotFoundError."""
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Expected model artifact file, got: {path}")
    return path


def load_short_term_model_artifact(
    artifact_dir: Path | str,
    task: str,
    device: str = DEVICE,
) -> ShortTermModelArtifact:
    # Load one saved short-term LSTM checkpoint for API inference.
    """Load one short-term LSTM artifact from disk."""
    artifact_root = Path(artifact_dir)
    checkpoint_path = _require_file(artifact_root / f"short_term_lstm_{task}.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    task_name = str(checkpoint.get("task", task))
    task_config = LSTM_TASK_CONFIG[task_name]

    model = ShortTermLSTM(
        input_size=int(checkpoint["seq_scaler"].n_features_in_),
        hidden_size=int(checkpoint.get("hidden_size", task_config.hidden_size)),
        static_size=int(checkpoint["static_scaler"].n_features_in_),
        dropout=float(checkpoint.get("dropout", task_config.dropout)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return ShortTermModelArtifact(
        task=task_name,
        task_config=task_config,
        model=model,
        seq_scaler=checkpoint["seq_scaler"],
        static_scaler=checkpoint["static_scaler"],
        y_scaler=checkpoint.get("y_scaler"),
        checkpoint=checkpoint,
    )


def load_short_term_model_artifacts(
    artifact_dir: Path | str,
    tasks: tuple[str, ...] = tuple(LSTM_TASK_CONFIG),
    device: str = DEVICE,
) -> dict[str, ShortTermModelArtifact]:
    # Load all requested short-term LSTM checkpoints.
    """Load short-term LSTM artifacts keyed by task name."""
    return {
        task: load_short_term_model_artifact(
            artifact_dir=artifact_dir,
            task=task,
            device=device,
        )
        for task in tasks
    }


def _long_term_model_stem(task_config: LongTermTaskConfig) -> str:
    # Match the naming convention used by src.training.train_long_term.
    """Return the artifact filename stem for one long-term config."""
    return f"long_term_{task_config.task}_h{task_config.horizon}_{task_config.model_family}"


def _build_long_term_mlp_from_checkpoint(
    checkpoint: dict[str, Any],
    task_config: LongTermTaskConfig,
    device: str,
) -> LongTermMLP:
    # Reconstruct the configured MLP architecture before loading weights.
    """Build and restore one long-term MLP model from checkpoint data."""
    params = task_config.model_params
    model = LongTermMLP(
        input_size=int(checkpoint["input_size"]),
        hidden_sizes=tuple(params["hidden_sizes"]),
        output_size=1,
        dropout=float(params["dropout"]),
        batch_norm=bool(params["batch_norm"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_long_term_model_artifact(
    artifact_dir: Path | str,
    task_config: LongTermTaskConfig,
    device: str = DEVICE,
) -> LongTermModelArtifact:
    # Load one selected long-term model artifact for API inference.
    """Load one long-term model artifact from disk."""
    artifact_root = Path(artifact_dir)
    model_stem = _long_term_model_stem(task_config)

    if task_config.model_family in {"random_forest", "ridge", "logistic"}:
        artifact = joblib.load(_require_file(artifact_root / f"{model_stem}.joblib"))
        return LongTermModelArtifact(
            task=task_config.task,
            horizon=task_config.horizon,
            task_config=artifact.get("task_config", task_config),
            model_family=task_config.model_family,
            model=artifact["model"],
            feature_cols=list(artifact["feature_cols"]),
            checkpoint=artifact,
        )

    if task_config.model_family == "mlp":
        checkpoint_path = _require_file(artifact_root / f"{model_stem}.pt")
        preprocessor_path = _require_file(artifact_root / f"{model_stem}_preprocessor.joblib")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = _build_long_term_mlp_from_checkpoint(checkpoint, task_config, device)
        return LongTermModelArtifact(
            task=task_config.task,
            horizon=task_config.horizon,
            task_config=checkpoint.get("task_config", task_config),
            model_family=task_config.model_family,
            model=model,
            feature_cols=list(checkpoint["feature_cols"]),
            preprocessor=joblib.load(preprocessor_path),
            target_scaler=checkpoint.get("target_scaler"),
            checkpoint=checkpoint,
        )

    raise ValueError(f"Unsupported long-term model family: {task_config.model_family}")


def load_long_term_model_artifacts(
    artifact_dir: Path | str,
    task_configs: tuple[LongTermTaskConfig, ...] | None = None,
    device: str = DEVICE,
) -> dict[tuple[str, int], LongTermModelArtifact]:
    # Load all selected long-term task/horizon artifacts.
    """Load long-term model artifacts keyed by (task, horizon)."""
    if task_configs is None:
        from src.config.long_term_config import resolve_long_term_task_config

        task_configs = tuple(
            resolve_long_term_task_config(task, horizon)
            for task in LONG_TERM_FORECAST_TASKS
            for horizon in LONG_TERM_HORIZONS
        )

    return {
        (task_config.task, task_config.horizon): load_long_term_model_artifact(
            artifact_dir=artifact_dir,
            task_config=task_config,
            device=device,
        )
        for task_config in task_configs
    }


def load_recommendation_ranker_artifact(
    artifact_dir: Path | str,
    required: bool = False,
) -> RecommendationRankerArtifact | None:
    """Load the selected playing-profile ranker when it is available."""
    path = Path(artifact_dir) / RECOMMENDATION_RANKER_FILENAME
    return _load_recommendation_ranker_artifact(path, required=required)
