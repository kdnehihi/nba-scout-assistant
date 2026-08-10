from __future__ import annotations

import pytest

from config.long_term_config import (
    LONG_TERM_HORIZONS,
    LONG_TERM_SELECTED_CONFIGS,
    LONG_TERM_TASKS,
    resolve_long_term_task_config,
)


def test_long_term_config_covers_all_tasks_and_horizons():
    expected_keys = {
        (task, horizon)
        for task in LONG_TERM_TASKS
        for horizon in LONG_TERM_HORIZONS
    }

    assert set(LONG_TERM_SELECTED_CONFIGS) == expected_keys


def test_long_term_config_uses_random_forest_for_h1_h2_and_mlp_for_h3():
    for task in LONG_TERM_TASKS:
        assert resolve_long_term_task_config(task, 1).model_family == "random_forest"
        assert resolve_long_term_task_config(task, 2).model_family == "random_forest"
        assert resolve_long_term_task_config(task, 3).model_family == "mlp"


def test_long_term_config_target_and_metric_contracts():
    active = resolve_long_term_task_config("active_probability", 3)
    scoring = resolve_long_term_task_config("pts_per_36", 2)

    assert active.task_type == "classification"
    assert active.target_col == "active_h3"
    assert active.selection_metric == "brier"
    assert active.model_params["loss_name"] == "bce"

    assert scoring.task_type == "regression"
    assert scoring.target_col == "pts_per_36_h2"
    assert scoring.selection_metric == "mae"
    assert "n_estimators" in scoring.model_params


def test_long_term_config_rejects_unknown_task():
    with pytest.raises(KeyError):
        resolve_long_term_task_config("unknown", 1)

