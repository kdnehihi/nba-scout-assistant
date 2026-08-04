from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DataQualityConfig:
    """Configuration for pre-training dataframe validation."""

    required_columns: tuple[str, ...]
    feature_columns: tuple[str, ...]
    target_columns: tuple[str, ...]
    split_column: str = "split"
    allowed_splits: tuple[str, ...] = ("train", "validation", "test")
    key_columns: tuple[str, ...] = ()
    max_feature_missing_pct: float = 0.0
    max_target_missing_pct: float = 0.0
    require_all_splits: bool = True


def missing_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    # Find required columns that are absent from a dataframe.
    """Return missing column names from a dataframe."""
    return sorted(set(columns) - set(df.columns))


def missing_summary(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    # Summarize missingness for selected columns.
    """Return dtype, missing count, missing percent, and unique count by column."""
    rows = []
    for column in columns:
        if column not in df.columns:
            rows.append(
                {
                    "column": column,
                    "dtype": "MISSING_COLUMN",
                    "missing_count": len(df),
                    "missing_pct": 1.0,
                    "n_unique": 0,
                }
            )
            continue
        series = df[column]
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "missing_count": int(series.isna().sum()),
                "missing_pct": float(series.isna().mean()),
                "n_unique": int(series.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows)


def numeric_finite_summary(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    # Count non-finite numeric values before model training.
    """Return inf and non-finite counts for numeric columns."""
    rows = []
    for column in columns:
        if column not in df.columns:
            continue
        numeric = pd.to_numeric(df[column], errors="coerce")
        finite_mask = np.isfinite(numeric.to_numpy(dtype="float64", na_value=np.nan))
        rows.append(
            {
                "column": column,
                "inf_count": int(np.isinf(numeric.to_numpy(dtype="float64", na_value=np.nan)).sum()),
                "non_finite_count": int((~finite_mask & numeric.notna().to_numpy()).sum()),
            }
        )
    return pd.DataFrame(rows)


def duplicate_key_summary(df: pd.DataFrame, key_columns: Iterable[str]) -> dict[str, object]:
    # Detect duplicate modeling rows when a natural key is available.
    """Return duplicate count and duplicate rate for selected key columns."""
    keys = list(key_columns)
    if not keys:
        return {"key_columns": "", "duplicate_count": 0, "duplicate_rate": 0.0}
    missing = missing_columns(df, keys)
    if missing:
        return {"key_columns": ", ".join(keys), "duplicate_count": None, "duplicate_rate": None, "missing_key_columns": missing}
    duplicate_mask = df.duplicated(keys)
    return {
        "key_columns": ", ".join(keys),
        "duplicate_count": int(duplicate_mask.sum()),
        "duplicate_rate": float(duplicate_mask.mean()) if len(df) else 0.0,
    }


def split_summary(df: pd.DataFrame, split_column: str = "split") -> pd.DataFrame:
    # Summarize row counts by split label.
    """Return row counts and percentage by split."""
    if split_column not in df.columns:
        return pd.DataFrame(columns=[split_column, "rows", "row_pct"])
    counts = df[split_column].value_counts(dropna=False).rename_axis(split_column).reset_index(name="rows")
    counts["row_pct"] = counts["rows"] / len(df) if len(df) else 0.0
    return counts.sort_values(split_column).reset_index(drop=True)


def validate_training_dataframe(
    df: pd.DataFrame,
    config: DataQualityConfig,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame | dict[str, object]]]:
    # Run pre-training data quality checks and return issues plus audit tables.
    """Validate a model-training dataframe before fitting models."""
    issues: list[dict[str, object]] = []
    all_required = set(config.required_columns) | set(config.feature_columns) | set(config.target_columns) | {config.split_column}

    missing_required = missing_columns(df, all_required)
    for column in missing_required:
        issues.append({"severity": "error", "check": "required_column", "column": column, "message": f"Missing required column: {column}"})

    split_counts = split_summary(df, config.split_column)
    if config.split_column in df.columns:
        observed_splits = set(df[config.split_column].dropna().astype(str))
        unexpected_splits = sorted(observed_splits - set(config.allowed_splits))
        for split in unexpected_splits:
            issues.append({"severity": "error", "check": "split_label", "column": config.split_column, "message": f"Unexpected split label: {split}"})
        if config.require_all_splits:
            missing_splits = sorted(set(config.allowed_splits) - observed_splits)
            for split in missing_splits:
                issues.append({"severity": "error", "check": "split_presence", "column": config.split_column, "message": f"Missing required split: {split}"})

    feature_missing = missing_summary(df, config.feature_columns)
    for row in feature_missing.to_dict("records"):
        if row["missing_pct"] > config.max_feature_missing_pct:
            issues.append(
                {
                    "severity": "error",
                    "check": "feature_missing",
                    "column": row["column"],
                    "message": f"Feature missing pct {row['missing_pct']:.4f} exceeds {config.max_feature_missing_pct:.4f}",
                }
            )

    target_missing = missing_summary(df, config.target_columns)
    for row in target_missing.to_dict("records"):
        if row["missing_pct"] > config.max_target_missing_pct:
            issues.append(
                {
                    "severity": "error",
                    "check": "target_missing",
                    "column": row["column"],
                    "message": f"Target missing pct {row['missing_pct']:.4f} exceeds {config.max_target_missing_pct:.4f}",
                }
            )

    numeric_summary = numeric_finite_summary(df, list(config.feature_columns) + list(config.target_columns))
    for row in numeric_summary.to_dict("records"):
        if row["non_finite_count"] > 0 or row["inf_count"] > 0:
            issues.append(
                {
                    "severity": "error",
                    "check": "numeric_finite",
                    "column": row["column"],
                    "message": f"Column contains non-finite numeric values: {row}",
                }
            )

    duplicate_summary = duplicate_key_summary(df, config.key_columns)
    if duplicate_summary.get("duplicate_count"):
        issues.append(
            {
                "severity": "warning",
                "check": "duplicate_key",
                "column": duplicate_summary["key_columns"],
                "message": f"Duplicate key rows found: {duplicate_summary['duplicate_count']}",
            }
        )

    issues_df = pd.DataFrame(issues, columns=["severity", "check", "column", "message"])
    audit = {
        "split_summary": split_counts,
        "feature_missing_summary": feature_missing,
        "target_missing_summary": target_missing,
        "numeric_finite_summary": numeric_summary,
        "duplicate_key_summary": duplicate_summary,
    }
    return issues_df, audit


def assert_training_dataframe_is_valid(df: pd.DataFrame, config: DataQualityConfig) -> None:
    # Raise a clear error when pre-training validation fails.
    """Raise ValueError if a training dataframe has data-quality errors."""
    issues, _ = validate_training_dataframe(df, config)
    errors = issues[issues["severity"].eq("error")] if not issues.empty else issues
    if not errors.empty:
        messages = "\n".join(errors["message"].astype(str).tolist())
        raise ValueError(f"Training dataframe failed validation:\n{messages}")

