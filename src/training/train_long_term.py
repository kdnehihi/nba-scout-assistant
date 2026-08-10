from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.long_term_config import LONG_TERM_HORIZONS, LONG_TERM_TASKS, LongTermTaskConfig, resolve_long_term_task_config
from src.dataset.loaders import DataPaths, load_long_term_training, resolve_data_paths
from src.dataset.long_term import prepare_long_term_modeling_data
from src.evaluation.evaluate_long_term import (
    evaluate_long_term_split_predictions,
    predict_mlp_long_term,
    predict_sklearn_long_term,
)
from src.modeling.long_term_baseline import build_long_term_preprocessor
from src.models.mlp import LongTermMLP
from src.models.randomforest import build_random_forest_classifier, build_random_forest_regressor
from src.training.mlflow_utils import configure_mlflow, log_metrics_flat, log_params_flat
from src.training.splitters import make_supervised_splits


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_ARTIFACT_DIR = Path("artifacts")
DEFAULT_EXPERIMENT_NAME = "nba_scout_long_term_forecasting"
DEFAULT_MLFLOW_DB_PATH = Path("mlflow.db")
DEFAULT_MLFLOW_ARTIFACT_DIR = Path("mlartifacts")
DEFAULT_MLP_EPOCHS = 100
DEFAULT_MLP_PATIENCE = 8
DEFAULT_MLP_MIN_DELTA = 1e-4


def load_long_term_dataframe(paths: DataPaths) -> pd.DataFrame:
    """Load the clean gold long-term training dataframe."""
    return load_long_term_training(paths)


def build_long_term_model(task_config: LongTermTaskConfig) -> Any:
    """Build the selected unfitted model for one long-term task config."""
    if task_config.model_family == "random_forest":
        if task_config.task_type == "classification":
            return build_random_forest_classifier(**task_config.model_params)
        return build_random_forest_regressor(**task_config.model_params)

    if task_config.model_family == "mlp":
        return None

    raise ValueError(f"Unsupported long-term model family: {task_config.model_family}")


def fit_random_forest_long_term(
    prepared_df: pd.DataFrame,
    feature_cols: list[str],
    task_config: LongTermTaskConfig,
) -> tuple[Pipeline, pd.DataFrame]:
    """Fit the selected random forest pipeline and return validation/test metrics."""
    splits = make_supervised_splits(prepared_df, feature_cols, task_config.target_col)
    estimator = build_long_term_model(task_config)
    model = Pipeline(
        [
            ("preprocess", build_long_term_preprocessor(splits.X_train)),
            ("model", estimator),
        ]
    )
    model.fit(splits.X_train, splits.y_train)

    validation_df = prepared_df[prepared_df["split"].eq("validation")]
    test_df = prepared_df[prepared_df["split"].eq("test")]

    validation_pred = predict_sklearn_long_term(
        model,
        validation_df[feature_cols],
        task_config,
    )
    test_pred = predict_sklearn_long_term(
        model,
        test_df[feature_cols],
        task_config,
    )

    metrics = pd.DataFrame(
        [
            evaluate_long_term_split_predictions("validation", validation_df[task_config.target_col], validation_pred, task_config),
            evaluate_long_term_split_predictions("test", test_df[task_config.target_col], test_pred, task_config),
        ]
    )
    return model, metrics


def dense_float32(values: Any) -> np.ndarray:
    """Convert preprocessed tabular data to a dense float32 numpy array for torch."""
    if sparse.issparse(values):
        values = values.toarray()
    return np.asarray(values, dtype="float32")


def make_mlp_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """Build a torch dataloader from dense tabular arrays."""
    dataset = TensorDataset(
        torch.from_numpy(X.astype("float32")),
        torch.from_numpy(y.astype("float32")),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def build_mlp_criterion(loss_name: str) -> nn.Module:
    """Build the configured MLP loss function."""
    if loss_name == "bce":
        return nn.BCEWithLogitsLoss()
    if loss_name == "huber":
        return nn.HuberLoss(delta=1.0)
    if loss_name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unsupported MLP loss: {loss_name}")


def train_mlp_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Adam,
) -> float:
    """Run one MLP training epoch and return sample-weighted loss."""
    model.train()
    total_loss = 0.0
    total_samples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()
        prediction = model(X_batch)
        loss = criterion(prediction, y_batch)
        loss.backward()
        optimizer.step()

        batch_size = y_batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def evaluate_mlp_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> float:
    """Evaluate MLP loss on one split dataloader."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            prediction = model(X_batch)
            loss = criterion(prediction, y_batch)

            batch_size = y_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def fit_mlp_long_term(
    prepared_df: pd.DataFrame,
    feature_cols: list[str],
    task_config: LongTermTaskConfig,
    mlflow_module: Any | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit the selected MLP and return artifacts plus validation/test metrics."""
    params = task_config.model_params
    splits = make_supervised_splits(prepared_df, feature_cols, task_config.target_col)

    preprocessor = build_long_term_preprocessor(splits.X_train)
    X_train = dense_float32(preprocessor.fit_transform(splits.X_train))
    X_validation = dense_float32(preprocessor.transform(splits.X_validation))
    X_test = dense_float32(preprocessor.transform(splits.X_test))

    y_train = splits.y_train.to_numpy(dtype="float32")
    y_validation = splits.y_validation.to_numpy(dtype="float32")
    y_test = splits.y_test.to_numpy(dtype="float32")

    target_scaler: StandardScaler | None = None
    y_train_model = y_train.copy()
    y_validation_model = y_validation.copy()
    y_test_model = y_test.copy()
    if task_config.task_type == "regression":
        target_scaler = StandardScaler()
        y_train_model = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel().astype("float32")
        y_validation_model = target_scaler.transform(y_validation.reshape(-1, 1)).ravel().astype("float32")
        y_test_model = target_scaler.transform(y_test.reshape(-1, 1)).ravel().astype("float32")

    batch_size = int(params["batch_size"])
    train_loader = make_mlp_loader(X_train, y_train_model, batch_size=batch_size, shuffle=True)
    validation_loader = make_mlp_loader(X_validation, y_validation_model, batch_size=batch_size, shuffle=False)
    test_loader = make_mlp_loader(X_test, y_test_model, batch_size=batch_size, shuffle=False)

    torch.manual_seed(int(params.get("random_state", 42)))
    model = LongTermMLP(
        input_size=X_train.shape[1],
        hidden_sizes=tuple(params["hidden_sizes"]),
        output_size=1,
        dropout=float(params["dropout"]),
        batch_norm=bool(params["batch_norm"]),
    ).to(DEVICE)
    criterion = build_mlp_criterion(str(params["loss_name"]))
    optimizer = Adam(
        model.parameters(),
        lr=float(params["learning_rate"]),
        weight_decay=float(params["weight_decay"]),
    )

    best_validation_loss = float("inf")
    best_model_state = None
    stale_epochs = 0
    max_epochs = int(params.get("max_epochs", DEFAULT_MLP_EPOCHS))
    patience = int(params.get("patience", DEFAULT_MLP_PATIENCE))
    min_delta = float(params.get("min_delta", DEFAULT_MLP_MIN_DELTA))

    for epoch in range(max_epochs):
        train_loss = train_mlp_one_epoch(model, train_loader, criterion, optimizer)
        validation_loss = evaluate_mlp_loss(model, validation_loader, criterion)

        if mlflow_module is not None:
            mlflow_module.log_metric("train_loss", train_loss, step=epoch)
            mlflow_module.log_metric("validation_loss", validation_loss, step=epoch)

        print(f"Epoch {epoch + 1}/{max_epochs} train_loss={train_loss:.5f} validation_loss={validation_loss:.5f}")

        if validation_loss < best_validation_loss - min_delta:
            best_validation_loss = validation_loss
            best_model_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    validation_loss = evaluate_mlp_loss(model, validation_loader, criterion)
    test_loss = evaluate_mlp_loss(model, test_loader, criterion)

    validation_pred = predict_mlp_long_term(model, X_validation, batch_size, task_config)
    test_pred = predict_mlp_long_term(model, X_test, batch_size, task_config)

    if target_scaler is not None:
        validation_pred = target_scaler.inverse_transform(validation_pred.reshape(-1, 1)).ravel()
        test_pred = target_scaler.inverse_transform(test_pred.reshape(-1, 1)).ravel()

    metrics = pd.DataFrame(
        [
            evaluate_long_term_split_predictions("validation", y_validation, validation_pred, task_config),
            evaluate_long_term_split_predictions("test", y_test, test_pred, task_config),
        ]
    )

    artifacts = {
        "model": model,
        "preprocessor": preprocessor,
        "target_scaler": target_scaler,
        "best_validation_loss": best_validation_loss,
        "validation_loss": validation_loss,
        "test_loss": test_loss,
        "input_size": X_train.shape[1],
        "epochs_ran": epoch + 1,
    }
    return artifacts, metrics


def train_long_term_model(
    task: str,
    horizon: int,
    data_dir: Path | str = "data",
    artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    mlflow_db_path: Path | str = DEFAULT_MLFLOW_DB_PATH,
    mlflow_artifact_dir: Path | str = DEFAULT_MLFLOW_ARTIFACT_DIR,
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Train one selected long-term model and return validation/test metrics."""
    task_config = resolve_long_term_task_config(task, horizon)
    paths = resolve_data_paths(data_dir)
    df = load_long_term_dataframe(paths)
    prepared_df, selected_feature_cols = prepare_long_term_modeling_data(
        df,
        task_config=task_config,
        feature_cols=feature_cols,
    )

    artifact_dir = Path(artifact_dir)
    model_stem = f"long_term_{task_config.task}_h{task_config.horizon}_{task_config.model_family}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device: {DEVICE}")
    print(f"Training long-term task: {task_config.task} h{task_config.horizon}")
    print(f"Target: {task_config.target_col}")
    print(f"Model family: {task_config.model_family}")
    print(f"Features: {len(selected_feature_cols)}")

    mlflow = configure_mlflow(
        tracking_db_path=mlflow_db_path,
        artifact_dir=mlflow_artifact_dir,
        experiment_name=experiment_name,
    )
    with mlflow.start_run(run_name=model_stem):
        log_params_flat(
            mlflow,
            {
                "task": task_config.task,
                "horizon": task_config.horizon,
                "task_type": task_config.task_type,
                "target_col": task_config.target_col,
                "model_family": task_config.model_family,
                "selection_metric": task_config.selection_metric,
                "selected_validation_value": task_config.validation_value,
                "feature_count": len(selected_feature_cols),
                "data_rows": len(prepared_df),
            },
        )
        log_params_flat(mlflow, task_config.model_params, prefix="model")

        if task_config.model_family == "random_forest":
            model, metrics = fit_random_forest_long_term(prepared_df, selected_feature_cols, task_config)
            model_path = artifact_dir / f"{model_stem}.joblib"
            joblib.dump(
                {
                    "model": model,
                    "task_config": task_config,
                    "feature_cols": selected_feature_cols,
                    "metrics": metrics.to_dict("records"),
                },
                model_path,
            )
            mlflow.log_artifact(str(model_path), artifact_path="models")

        elif task_config.model_family == "mlp":
            artifacts, metrics = fit_mlp_long_term(
                prepared_df,
                selected_feature_cols,
                task_config,
                mlflow_module=mlflow,
            )
            model_path = artifact_dir / f"{model_stem}.pt"
            preprocessor_path = artifact_dir / f"{model_stem}_preprocessor.joblib"
            torch.save(
                {
                    "model_state_dict": artifacts["model"].state_dict(),
                    "task_config": task_config,
                    "feature_cols": selected_feature_cols,
                    "target_scaler": artifacts["target_scaler"],
                    "input_size": artifacts["input_size"],
                    "metrics": metrics.to_dict("records"),
                    "best_validation_loss": artifacts["best_validation_loss"],
                },
                model_path,
            )
            joblib.dump(artifacts["preprocessor"], preprocessor_path)
            mlflow.log_artifact(str(model_path), artifact_path="models")
            mlflow.log_artifact(str(preprocessor_path), artifact_path="models")
            mlflow.log_metric("best_validation_loss", float(artifacts["best_validation_loss"]))
            mlflow.log_metric("validation_loss", float(artifacts["validation_loss"]))
            mlflow.log_metric("test_loss", float(artifacts["test_loss"]))
            mlflow.log_metric("epochs_ran", int(artifacts["epochs_ran"]))

        else:
            raise ValueError(f"Unsupported long-term model family: {task_config.model_family}")

        metrics_path = artifact_dir / f"{model_stem}_metrics.csv"
        feature_path = artifact_dir / f"{model_stem}_features.txt"
        metrics.to_csv(metrics_path, index=False)
        feature_path.write_text("\n".join(selected_feature_cols), encoding="utf-8")
        mlflow.log_artifact(str(metrics_path), artifact_path="metrics")
        mlflow.log_artifact(str(feature_path), artifact_path="features")

        for _, row in metrics.iterrows():
            split_name = row["split"]
            mlflow.log_metric(f"{split_name}_rows", int(row["rows"]))
            metric_values = {
                key: value
                for key, value in row.to_dict().items()
                if key not in {"task", "horizon", "model_family", "target", "split", "rows"} and pd.notna(value)
            }
            log_metrics_flat(mlflow, metric_values, prefix=str(split_name))

    print(metrics)
    return metrics


def train_long_term_models(
    data_dir: Path | str = "data",
    artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR,
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    mlflow_db_path: Path | str = DEFAULT_MLFLOW_DB_PATH,
    mlflow_artifact_dir: Path | str = DEFAULT_MLFLOW_ARTIFACT_DIR,
) -> pd.DataFrame:
    """Train every selected long-term task/horizon model and return all metrics."""
    all_metrics = []
    for task in LONG_TERM_TASKS:
        for horizon in LONG_TERM_HORIZONS:
            metrics = train_long_term_model(
                task=task,
                horizon=horizon,
                data_dir=data_dir,
                artifact_dir=artifact_dir,
                experiment_name=experiment_name,
                mlflow_db_path=mlflow_db_path,
                mlflow_artifact_dir=mlflow_artifact_dir,
            )
            all_metrics.append(metrics)
    return pd.concat(all_metrics, ignore_index=True)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for long-term model training."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=[*LONG_TERM_TASKS, "all"], default="all")
    parser.add_argument("--horizon", type=int, choices=[*LONG_TERM_HORIZONS], default=None)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default=str(DEFAULT_ARTIFACT_DIR))
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--mlflow-db-path", default=str(DEFAULT_MLFLOW_DB_PATH))
    parser.add_argument("--mlflow-artifact-dir", default=str(DEFAULT_MLFLOW_ARTIFACT_DIR))
    return parser.parse_args()


def main() -> None:
    """Run long-term training from the command line."""
    args = parse_args()
    if args.task == "all":
        if args.horizon is not None:
            for task in LONG_TERM_TASKS:
                train_long_term_model(
                    task=task,
                    horizon=args.horizon,
                    data_dir=args.data_dir,
                    artifact_dir=args.artifact_dir,
                    experiment_name=args.experiment_name,
                    mlflow_db_path=args.mlflow_db_path,
                    mlflow_artifact_dir=args.mlflow_artifact_dir,
                )
        else:
            train_long_term_models(
                data_dir=args.data_dir,
                artifact_dir=args.artifact_dir,
                experiment_name=args.experiment_name,
                mlflow_db_path=args.mlflow_db_path,
                mlflow_artifact_dir=args.mlflow_artifact_dir,
            )
        return

    if args.horizon is None:
        raise ValueError("--horizon is required when --task is not 'all'.")

    train_long_term_model(
        task=args.task,
        horizon=args.horizon,
        data_dir=args.data_dir,
        artifact_dir=args.artifact_dir,
        experiment_name=args.experiment_name,
        mlflow_db_path=args.mlflow_db_path,
        mlflow_artifact_dir=args.mlflow_artifact_dir,
    )


if __name__ == "__main__":
    main()
