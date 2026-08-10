from __future__ import annotations

import torch
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from models.mlp import LongTermMLP
from models.randomforest import build_random_forest_classifier, build_random_forest_regressor


def test_long_term_mlp_returns_one_value_per_row():
    model = LongTermMLP(input_size=5, hidden_sizes=(8, 4), dropout=0.0, batch_norm=False)
    x = torch.randn(3, 5)

    prediction = model(x)

    assert prediction.shape == (3,)


def test_random_forest_builders_return_unfitted_estimators():
    regressor = build_random_forest_regressor(n_estimators=10, min_samples_leaf=2, max_features="sqrt")
    classifier = build_random_forest_classifier(n_estimators=10, min_samples_leaf=2, max_features=0.5)

    assert isinstance(regressor, RandomForestRegressor)
    assert regressor.n_estimators == 10
    assert regressor.min_samples_leaf == 2
    assert regressor.max_features == "sqrt"

    assert isinstance(classifier, RandomForestClassifier)
    assert classifier.n_estimators == 10
    assert classifier.min_samples_leaf == 2
    assert classifier.max_features == 0.5
    assert classifier.class_weight == "balanced"
