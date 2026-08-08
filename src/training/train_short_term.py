from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.lstm_config import resolve_lstm_task_config
from src.dataset.loaders import DataPaths, load_performance_training_clean, resolve_data_paths
from src.dataset.lstm_dataset import ShortTermLSTMDataset
from src.dataset.scaling import scale_lstm_inputs
from src.dataset.sequence import make_lstm_delta_sequences, prepare_sequence_training
from src.evaluation.evaluate_short_term import evaluate_lstm_predictions_by_split, predict_lstm_actuals
from src.models.lstm import ShortTermLSTM
from src.training.mlflow_utils import configure_mlflow


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_ARTIFACT_DIR = Path("artifacts")
DEFAULT_EXPERIMENT_NAME = "nba_scout_short_term_lstm"
DEFAULT_MLFLOW_DB_PATH = Path("mlflow.db")
DEFAULT_MLFLOW_ARTIFACT_DIR = Path("mlartifacts")
DEFAULT_PATIENCE = 8
DEFAULT_MIN_DELTA = 1e-4


def load_short_term_dataframe(paths: DataPaths) -> pd.DataFrame:
    """Load the clean gold short-term performance training dataframe."""
    return load_performance_training_clean(paths)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Adam,
) -> float:
    """Run one training epoch and return sample-weighted loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x_seq, x_static, y in loader:
        x_seq = x_seq.to(DEVICE)
        x_static = x_static.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()
        prediction = model(x_seq, x_static)
        loss = criterion(prediction, y)
        loss.backward()
        optimizer.step()

        batch_size = y.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> float:
    """Evaluate model loss on a dataloader."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for x_seq, x_static, y in loader:
            x_seq = x_seq.to(DEVICE)
            x_static = x_static.to(DEVICE)
            y = y.to(DEVICE)

            prediction = model(x_seq, x_static)
            loss = criterion(prediction, y)

            batch_size = y.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def train_short_term(
    task: str = "points",
    data_dir: Path | str = "data",
    artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR,
    epochs: int = 100,
    patience: int = DEFAULT_PATIENCE,
    min_delta: float = DEFAULT_MIN_DELTA,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    mlflow_db_path: Path | str = DEFAULT_MLFLOW_DB_PATH,
    mlflow_artifact_dir: Path | str = DEFAULT_MLFLOW_ARTIFACT_DIR,
) -> pd.DataFrame:
    """Train one short-term LSTM task and return validation/test metrics."""
    task_name, task_config = resolve_lstm_task_config(task)
    artifact_dir = Path(artifact_dir)
    model_path = artifact_dir / f"short_term_lstm_{task_name}.pt"

    print(f"Using device: {DEVICE}")
    print(f"Training task: {task_name}")

    paths = resolve_data_paths(data_dir)
    df = load_short_term_dataframe(paths)
    sequence_df = prepare_sequence_training(df, task_config)

    X_seq, X_static, y_delta, y_actual, baseline, split = make_lstm_delta_sequences(
        sequence_df,
        task_config,
    )

    train_mask = split == "train"
    validation_mask = split == "validation"
    test_mask = split == "test"

    X_seq_scaled, X_static_scaled, y_delta_model, seq_scaler, static_scaler, y_scaler = scale_lstm_inputs(
        X_seq,
        X_static,
        y_delta,
        train_mask,
        scale_target_delta=task_config.scale_target_delta,
    )

    train_dataset = ShortTermLSTMDataset(
        X_seq_scaled[train_mask],
        X_static_scaled[train_mask],
        y_delta_model[train_mask],
    )
    validation_dataset = ShortTermLSTMDataset(
        X_seq_scaled[validation_mask],
        X_static_scaled[validation_mask],
        y_delta_model[validation_mask],
    )
    test_dataset = ShortTermLSTMDataset(
        X_seq_scaled[test_mask],
        X_static_scaled[test_mask],
        y_delta_model[test_mask],
    )

    train_loader = DataLoader(train_dataset, batch_size=task_config.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=task_config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=task_config.batch_size, shuffle=False)

    model = ShortTermLSTM(
        input_size=X_seq.shape[-1],
        hidden_size=task_config.hidden_size,
        static_size=X_static.shape[-1],
        dropout=task_config.dropout,
    ).to(DEVICE)

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = Adam(model.parameters(), lr=task_config.learning_rate)

    best_val_loss = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    mlflow = configure_mlflow(
        tracking_db_path=mlflow_db_path,
        artifact_dir=mlflow_artifact_dir,
        experiment_name=experiment_name,
    )
    with mlflow.start_run(run_name=f"lstm_{task_name}"):
        mlflow.log_params(
            {
                "task": task_name,
                "epochs": epochs,
                "batch_size": task_config.batch_size,
                "learning_rate": task_config.learning_rate,
                "hidden_size": task_config.hidden_size,
                "sequence_length": task_config.sequence_length,
                "dropout": task_config.dropout,
                "optimizer": "Adam",
                "loss": "HuberLoss",
                "scale_target_delta": task_config.scale_target_delta,
                "sequence_features": X_seq.shape[-1],
                "static_features": X_static.shape[-1],
                "patience": patience,
                "min_delta": min_delta,
            }
        )

        for epoch in range(epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
            validation_loss = evaluate_loss(model, validation_loader, criterion)

            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("validation_loss", validation_loss, step=epoch)
            print(f"Epoch {epoch + 1}/{epochs} train_loss={train_loss:.5f} validation_loss={validation_loss:.5f}")

            if validation_loss < best_val_loss - min_delta:
                best_val_loss = validation_loss
                best_model_state = deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        test_loss = evaluate_loss(model, test_loader, criterion)
        y_pred = predict_lstm_actuals(
            model=model,
            X_seq_scaled=X_seq_scaled,
            X_static_scaled=X_static_scaled,
            baseline=baseline,
            y_scaler=y_scaler,
            batch_size=task_config.batch_size,
            device=DEVICE,
        )
        metrics = evaluate_lstm_predictions_by_split(
            y_actual=y_actual,
            y_pred=y_pred,
            split=split,
            task=task_name,
            splits=("validation", "test"),
        )

        mlflow.log_metric("best_validation_loss", best_val_loss)
        mlflow.log_metric("test_loss", test_loss)
        for _, row in metrics.iterrows():
            split_name = row["split"]
            mlflow.log_metric(f"{split_name}_rows", int(row["rows"]))
            mlflow.log_metric(f"{split_name}_mae", float(row["mae"]))
            mlflow.log_metric(f"{split_name}_rmse", float(row["rmse"]))
            mlflow.log_metric(f"{split_name}_r2", float(row["r2"]))

        artifact_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "task": task_name,
                "best_validation_loss": best_val_loss,
                "test_loss": test_loss,
                "sequence_length": task_config.sequence_length,
                "hidden_size": task_config.hidden_size,
                "dropout": task_config.dropout,
                "seq_scaler": seq_scaler,
                "static_scaler": static_scaler,
                "y_scaler": y_scaler,
                "scale_target_delta": task_config.scale_target_delta,
                "metrics": metrics.to_dict("records"),
            },
            model_path,
        )
        mlflow.log_artifact(str(model_path), artifact_path="checkpoints")

    print(metrics)
    print(f"Saved {model_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for short-term LSTM training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="points", choices=["points", "assists", "rebounds", "pts", "ast", "reb"])
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--mlflow-db-path", default=str(DEFAULT_MLFLOW_DB_PATH))
    parser.add_argument("--mlflow-artifact-dir", default=str(DEFAULT_MLFLOW_ARTIFACT_DIR))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_short_term(
        task=args.task,
        data_dir=args.data_dir,
        artifact_dir=args.artifact_dir,
        epochs=args.epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        experiment_name=args.experiment_name,
        mlflow_db_path=args.mlflow_db_path,
        mlflow_artifact_dir=args.mlflow_artifact_dir,
    )
