from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def get_one_hot_encoder() -> OneHotEncoder:
    """Return a OneHotEncoder compatible with the installed sklearn version."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def normalize_categorical_missing_values(X: pd.DataFrame) -> pd.DataFrame:
    """Return categorical columns with Python None normalized to pandas missing values."""
    return pd.DataFrame(X).replace({None: np.nan})


def build_long_term_preprocessor(X: pd.DataFrame, scale_numeric: bool = False) -> ColumnTransformer:
    """Return a long-term tabular preprocessor for numeric and categorical features."""
    categorical_cols = [
        column
        for column in X.columns
        if X[column].dtype == "object" or str(X[column].dtype) == "category"
    ]
    numeric_cols = [column for column in X.columns if column not in categorical_cols]

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(numeric_steps),
                numeric_cols,
            ),
            (
                "categorical",
                Pipeline([
                    ("normalize_missing", FunctionTransformer(normalize_categorical_missing_values)),
                    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                    ("one_hot", get_one_hot_encoder()),
                ]),
                categorical_cols,
            ),
        ],
        remainder="drop",
    )


def attach_long_term_preprocessor(model_pipeline: Pipeline, X: pd.DataFrame) -> Pipeline:
    """Return an unfitted pipeline with long-term preprocessing before the model steps."""
    return Pipeline([
        ("preprocess", build_long_term_preprocessor(X)),
        *model_pipeline.steps,
    ])


def build_long_term_ridge_baseline(alpha: float = 5.0) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])


def build_long_term_logistic_baseline(
    max_iter: int = 2000,
    class_weight: str | dict | None = "balanced",
    C: float = 0.5,
    random_state: int = 42,
) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=max_iter,
            class_weight=class_weight,
            C=C,
            random_state=random_state,
        )),
    ])


def build_long_term_random_forest_regressor(
    n_estimators: int = 300,
    min_samples_leaf: int = 8,
    max_features: str | float | int | None = "sqrt",
    n_jobs: int | None = -1,
    random_state: int = 42,
) -> Pipeline:
    return Pipeline([("model", RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=n_jobs,
        random_state=random_state
    ))])

def build_long_term_random_forest_classifier(
    n_estimators: int = 400,
    min_samples_leaf: int = 8,
    max_features: str | float | int | None = "sqrt",
    class_weight: str | dict | None = "balanced",
    n_jobs: int | None = -1,
    random_state: int = 42,
) -> Pipeline:
    return Pipeline([("model", RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state
        ))])

def build_long_term_hist_gradient_boosting_regressor(
    max_iter: int = 250,
    learning_rate: float = 0.04,
    l2_regularization: float = 0.05,
    random_state: int = 42,
) -> Pipeline:
    return Pipeline([("model", HistGradientBoostingRegressor(
        max_iter=max_iter,
        learning_rate=learning_rate,
        l2_regularization=l2_regularization,
        random_state=random_state,
    ))])
