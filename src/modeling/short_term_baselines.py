from __future__ import annotations
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd


def build_ridge_short_term_baseline(alpha: float = 1.0) -> Pipeline:
    """Return an unfitted ridge baseline pipeline for short-term forecasting."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=alpha)),
    ])

def build_random_forest_short_term_baseline(
    n_estimators: int = 260,
    min_samples_leaf: int = 10,
    max_features: str | float | None = "sqrt",
    n_jobs: int = -1,
    random_state: int = 42,
) -> Pipeline:
    """Return an unfitted random forest baseline pipeline for short-term forecasting."""
    return Pipeline([
        ("model", RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
        )),
    ])

def naive_last_5_prediction(df: pd.DataFrame, stat: str) -> pd.Series:
    # Predict next-five-game average with the latest five-game average.
    """Return naive short-term baseline predictions from last-5 averages."""
    return df[f"{stat}_last_5"]


def season_average_prediction(df: pd.DataFrame, stat: str) -> pd.Series:
    # Predict next-five-game average with season-to-date average.
    """Return season-average short-term baseline predictions."""
    return df[f"{stat}_season_avg"]

