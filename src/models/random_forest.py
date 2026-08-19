from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


def build_random_forest_regressor(
    n_estimators: int,
    min_samples_leaf: int,
    max_features: str | float | int | None,
    n_jobs: int | None = -1,
    random_state: int = 42,
) -> RandomForestRegressor:
    """Return an unfitted RandomForestRegressor."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=n_jobs,
        random_state=random_state,
    )


def build_random_forest_classifier(
    n_estimators: int,
    min_samples_leaf: int,
    max_features: str | float | int | None,
    class_weight: str | dict | None = "balanced",
    n_jobs: int | None = -1,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Return an unfitted RandomForestClassifier."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        n_jobs=n_jobs,
        random_state=random_state,
    )
