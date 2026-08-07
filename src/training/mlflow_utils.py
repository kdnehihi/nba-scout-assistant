from __future__ import annotations

from pathlib import Path
from typing import Any


def configure_mlflow(
    tracking_db_path: str | Path,
    artifact_dir: str | Path,
    experiment_name: str,
) -> Any:
    # Configure MLflow with a SQLite backend and file artifact store.
    """Configure MLflow and return the imported mlflow module."""
    import mlflow

    tracking_uri = f"sqlite:///{Path(tracking_db_path).expanduser().resolve()}"
    artifact_uri = Path(artifact_dir).expanduser().resolve().as_uri()
    Path(artifact_dir).expanduser().resolve().mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name, artifact_location=artifact_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow


def log_params_flat(mlflow_module: Any, params: dict[str, Any], prefix: str | None = None) -> None:
    # Log a flat dictionary of MLflow parameters with optional key prefixing.
    """Log parameter values to the active MLflow run."""
    for key, value in params.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        mlflow_module.log_param(name, value)


def log_metrics_flat(mlflow_module: Any, metrics: dict[str, float], prefix: str | None = None) -> None:
    # Log a flat dictionary of MLflow metrics with optional key prefixing.
    """Log numeric metric values to the active MLflow run."""
    for key, value in metrics.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        mlflow_module.log_metric(name, float(value))

