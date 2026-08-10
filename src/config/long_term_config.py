from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


LongTermTaskType = Literal["classification", "regression"]
LongTermModelFamily = Literal["random_forest", "mlp"]

LONG_TERM_TASKS = (
    "active_probability",
    "pts_per_36",
    "ast_per_36",
    "reb_per_36",
    "pts_per_100",
    "ast_per_100",
    "reb_per_100",
)
LONG_TERM_HORIZONS = (1, 2, 3)


@dataclass(frozen=True)
class LongTermTaskConfig:
    task: str
    horizon: int
    task_type: LongTermTaskType
    target_col: str
    model_family: LongTermModelFamily
    model_params: dict[str, Any]
    selection_metric: str
    validation_value: float


def _rf_regressor_params(
    n_estimators: int,
    min_samples_leaf: int,
    max_features: str | float,
) -> dict[str, Any]:
    return {
        "n_estimators": n_estimators,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "n_jobs": -1,
        "random_state": 42,
    }


def _rf_classifier_params(
    n_estimators: int,
    min_samples_leaf: int,
    max_features: str | float,
) -> dict[str, Any]:
    return {
        **_rf_regressor_params(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
        ),
        "class_weight": "balanced",
    }


def _mlp_params(
    hidden_sizes: tuple[int, ...],
    dropout: float,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    loss_name: str,
    batch_norm: bool,
) -> dict[str, Any]:
    return {
        "hidden_sizes": hidden_sizes,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "batch_size": batch_size,
        "loss_name": loss_name,
        "batch_norm": batch_norm,
        "max_epochs": 100,
        "patience": 8,
        "min_delta": 1e-4,
        "random_state": 42,
    }


LONG_TERM_SELECTED_CONFIGS: dict[tuple[str, int], LongTermTaskConfig] = {
    ("active_probability", 1): LongTermTaskConfig(
        task="active_probability",
        horizon=1,
        task_type="classification",
        target_col="active_h1",
        model_family="random_forest",
        model_params=_rf_classifier_params(500, 4, 0.45),
        selection_metric="brier",
        validation_value=0.10690462671467102,
    ),
    ("active_probability", 2): LongTermTaskConfig(
        task="active_probability",
        horizon=2,
        task_type="classification",
        target_col="active_h2",
        model_family="random_forest",
        model_params=_rf_classifier_params(300, 4, "sqrt"),
        selection_metric="brier",
        validation_value=0.1387025898556649,
    ),
    ("active_probability", 3): LongTermTaskConfig(
        task="active_probability",
        horizon=3,
        task_type="classification",
        target_col="active_h3",
        model_family="mlp",
        model_params=_mlp_params((192, 96), 0.1649670079325266, 0.0006168967960669716, 0.00015832037046909113, 64, "bce", False),
        selection_metric="brier",
        validation_value=0.1503136244105211,
    ),
    ("pts_per_36", 1): LongTermTaskConfig(
        task="pts_per_36",
        horizon=1,
        task_type="regression",
        target_col="pts_per_36_h1",
        model_family="random_forest",
        model_params=_rf_regressor_params(300, 4, "sqrt"),
        selection_metric="mae",
        validation_value=2.394319839647154,
    ),
    ("pts_per_36", 2): LongTermTaskConfig(
        task="pts_per_36",
        horizon=2,
        task_type="regression",
        target_col="pts_per_36_h2",
        model_family="random_forest",
        model_params=_rf_regressor_params(300, 12, 0.85),
        selection_metric="mae",
        validation_value=2.584225314209544,
    ),
    ("pts_per_36", 3): LongTermTaskConfig(
        task="pts_per_36",
        horizon=3,
        task_type="regression",
        target_col="pts_per_36_h3",
        model_family="mlp",
        model_params=_mlp_params((128, 64), 0.30893102776267806, 0.0008330803890301997, 1.414401002080816e-05, 256, "huber", True),
        selection_metric="mae",
        validation_value=2.6230733394622803,
    ),
    ("ast_per_36", 1): LongTermTaskConfig(
        task="ast_per_36",
        horizon=1,
        task_type="regression",
        target_col="ast_per_36_h1",
        model_family="random_forest",
        model_params=_rf_regressor_params(500, 12, 0.45),
        selection_metric="mae",
        validation_value=0.799767329414566,
    ),
    ("ast_per_36", 2): LongTermTaskConfig(
        task="ast_per_36",
        horizon=2,
        task_type="regression",
        target_col="ast_per_36_h2",
        model_family="random_forest",
        model_params=_rf_regressor_params(300, 4, "sqrt"),
        selection_metric="mae",
        validation_value=0.7823843632227498,
    ),
    ("ast_per_36", 3): LongTermTaskConfig(
        task="ast_per_36",
        horizon=3,
        task_type="regression",
        target_col="ast_per_36_h3",
        model_family="mlp",
        model_params=_mlp_params((192, 96), 0.23355586841671383, 0.000160712385120399, 1.037104137505588e-05, 256, "huber", False),
        selection_metric="mae",
        validation_value=0.892236590385437,
    ),
    ("reb_per_36", 1): LongTermTaskConfig(
        task="reb_per_36",
        horizon=1,
        task_type="regression",
        target_col="reb_per_36_h1",
        model_family="random_forest",
        model_params=_rf_regressor_params(300, 4, "sqrt"),
        selection_metric="mae",
        validation_value=0.8795698484333203,
    ),
    ("reb_per_36", 2): LongTermTaskConfig(
        task="reb_per_36",
        horizon=2,
        task_type="regression",
        target_col="reb_per_36_h2",
        model_family="random_forest",
        model_params=_rf_regressor_params(200, 8, "sqrt"),
        selection_metric="mae",
        validation_value=1.0917502083365629,
    ),
    ("reb_per_36", 3): LongTermTaskConfig(
        task="reb_per_36",
        horizon=3,
        task_type="regression",
        target_col="reb_per_36_h3",
        model_family="mlp",
        model_params=_mlp_params((192, 96), 0.26918349112517104, 0.0003636817400938816, 0.00018198265284956783, 64, "huber", True),
        selection_metric="mae",
        validation_value=1.0506700277328491,
    ),
    ("pts_per_100", 1): LongTermTaskConfig(
        task="pts_per_100",
        horizon=1,
        task_type="regression",
        target_col="pts_per_100_h1",
        model_family="random_forest",
        model_params=_rf_regressor_params(300, 4, "sqrt"),
        selection_metric="mae",
        validation_value=3.1206774901010346,
    ),
    ("pts_per_100", 2): LongTermTaskConfig(
        task="pts_per_100",
        horizon=2,
        task_type="regression",
        target_col="pts_per_100_h2",
        model_family="random_forest",
        model_params=_rf_regressor_params(300, 8, 0.85),
        selection_metric="mae",
        validation_value=3.359991549193136,
    ),
    ("pts_per_100", 3): LongTermTaskConfig(
        task="pts_per_100",
        horizon=3,
        task_type="regression",
        target_col="pts_per_100_h3",
        model_family="mlp",
        model_params=_mlp_params((128, 64), 0.30893102776267806, 0.0008330803890301997, 1.414401002080816e-05, 256, "huber", True),
        selection_metric="mae",
        validation_value=3.4342381954193115,
    ),
    ("ast_per_100", 1): LongTermTaskConfig(
        task="ast_per_100",
        horizon=1,
        task_type="regression",
        target_col="ast_per_100_h1",
        model_family="random_forest",
        model_params=_rf_regressor_params(500, 12, 0.45),
        selection_metric="mae",
        validation_value=1.057053616475143,
    ),
    ("ast_per_100", 2): LongTermTaskConfig(
        task="ast_per_100",
        horizon=2,
        task_type="regression",
        target_col="ast_per_100_h2",
        model_family="random_forest",
        model_params=_rf_regressor_params(300, 4, "sqrt"),
        selection_metric="mae",
        validation_value=1.0334815285317311,
    ),
    ("ast_per_100", 3): LongTermTaskConfig(
        task="ast_per_100",
        horizon=3,
        task_type="regression",
        target_col="ast_per_100_h3",
        model_family="mlp",
        model_params=_mlp_params((192, 96), 0.23355586841671383, 0.000160712385120399, 1.037104137505588e-05, 256, "huber", False),
        selection_metric="mae",
        validation_value=1.168047308921814,
    ),
    ("reb_per_100", 1): LongTermTaskConfig(
        task="reb_per_100",
        horizon=1,
        task_type="regression",
        target_col="reb_per_100_h1",
        model_family="random_forest",
        model_params=_rf_regressor_params(300, 4, "sqrt"),
        selection_metric="mae",
        validation_value=1.1439175789495484,
    ),
    ("reb_per_100", 2): LongTermTaskConfig(
        task="reb_per_100",
        horizon=2,
        task_type="regression",
        target_col="reb_per_100_h2",
        model_family="random_forest",
        model_params=_rf_regressor_params(300, 4, "sqrt"),
        selection_metric="mae",
        validation_value=1.4098317863958942,
    ),
    ("reb_per_100", 3): LongTermTaskConfig(
        task="reb_per_100",
        horizon=3,
        task_type="regression",
        target_col="reb_per_100_h3",
        model_family="mlp",
        model_params=_mlp_params((192, 96), 0.21280882494747455, 0.00016149614799999194, 0.0006156532440760017, 128, "huber", True),
        selection_metric="mae",
        validation_value=1.394848346710205,
    ),
}


def resolve_long_term_task_config(task: str, horizon: int) -> LongTermTaskConfig:
    """Return the selected model config for a long-term task and horizon."""
    key = (task, horizon)
    if key not in LONG_TERM_SELECTED_CONFIGS:
        raise KeyError(f"Unknown long-term task config: task={task!r}, horizon={horizon!r}")
    return LONG_TERM_SELECTED_CONFIGS[key]

