from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from config.long_term_config import resolve_long_term_task_config
from evaluation.evaluate_long_term import evaluate_long_term_split_predictions
from training.train_long_term import build_long_term_model


def test_build_long_term_model_matches_config_family_and_task_type():
    classifier_config = resolve_long_term_task_config("active_probability", 1)
    regressor_config = resolve_long_term_task_config("pts_per_36", 1)

    classifier = build_long_term_model(classifier_config)
    regressor = build_long_term_model(regressor_config)

    assert isinstance(classifier, RandomForestClassifier)
    assert isinstance(regressor, RandomForestRegressor)


def test_evaluate_long_term_split_predictions_handles_regression_metrics():
    task_config = resolve_long_term_task_config("pts_per_36", 1)

    metrics = evaluate_long_term_split_predictions(
        split_name="validation",
        y_true=np.array([10.0, 12.0]),
        y_pred=np.array([11.0, 11.0]),
        task_config=task_config,
    )

    assert metrics["split"] == "validation"
    assert metrics["rows"] == 2
    assert metrics["mae"] == 1.0
