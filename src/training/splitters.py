from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DatasetSplit:
    """Container for model-ready train, validation, and test matrices."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_validation: pd.DataFrame
    y_validation: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def validate_split_column(
    df: pd.DataFrame,
    split_col: str = "split",
    required_splits: tuple[str, ...] = ("train", "validation", "test"),
) -> None:
    """Validate that a dataframe has the expected temporal split labels."""
    if split_col not in df.columns:
        raise KeyError(f"Missing split column: {split_col}")

    observed = set(df[split_col].dropna().astype(str))
    missing = set(required_splits) - observed

    if missing:
        raise ValueError(f"Missing required split labels: {sorted(missing)}")


def split_by_column(
    df: pd.DataFrame,
    split_col: str = "split",
    train_label: str = "train",
    validation_label: str = "validation",
    test_label: str = "test",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train, validation, and test dataframes from a split-labeled table."""
    validate_split_column(
        df,
        split_col=split_col,
        required_splits=(train_label, validation_label, test_label),
    )

    return (
        df[df[split_col].eq(train_label)].copy(),
        df[df[split_col].eq(validation_label)].copy(),
        df[df[split_col].eq(test_label)].copy(),
    )


def feature_target_split(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix X and target vector y after dropping incomplete rows."""
    required_cols = [*feature_cols, target_col]
    missing_cols = sorted(set(required_cols) - set(df.columns))

    if missing_cols:
        raise KeyError(f"Missing feature/target columns: {missing_cols}")

    clean = df.dropna(subset=required_cols).copy()

    return clean[feature_cols].copy(), clean[target_col].copy()


def make_supervised_splits(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    split_col: str = "split",
) -> DatasetSplit:
    """Return model-ready X/y splits for supervised training."""
    train_df, validation_df, test_df = split_by_column(df, split_col=split_col)

    X_train, y_train = feature_target_split(train_df, feature_cols, target_col)
    X_validation, y_validation = feature_target_split(validation_df, feature_cols, target_col)
    X_test, y_test = feature_target_split(test_df, feature_cols, target_col)

    return DatasetSplit(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        X_test=X_test,
        y_test=y_test,
    )