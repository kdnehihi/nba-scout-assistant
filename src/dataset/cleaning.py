from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd


TEAM_ABBREVIATION_MAP = {
    "BRK": "BKN",
    "CHO": "CHA",
    "PHO": "PHX",
    "NOH": "NOP",
    "NOK": "NOP",
    "NJN": "BKN",
}


def to_snake_case(column: object) -> str:
    # Standardize source column names before schema matching.
    # Example: "Player Name!" -> "player_name"; "trueShooting%" -> "true_shooting".
    """Convert a source column name to snake_case."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(column))
    text = re.sub(r"[^a-zA-Z0-9]+", "_", text)
    return text.strip("_").lower()


def normalize_name_key(value: object) -> str | None:
    # Create a stable player-name key for fallback joins.
    # Example: "Luka Dončić Jr." -> "lukadoncicjr".
    """Normalize a player name for cross-source joins."""
    if pd.isna(value):
        return None
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", text.lower())


def normalize_team_abbreviation(value: object) -> str | None:
    # Map historical/source team abbreviations to canonical IDs.
    # Example: "BRK" -> "BKN"; "pho" -> "PHX"; "LAL" -> "LAL".
    """Normalize team abbreviations across common NBA source variants."""
    if pd.isna(value):
        return None
    text = str(value).strip().upper()
    return TEAM_ABBREVIATION_MAP.get(text, text)


def percent_to_ratio(values: pd.Series) -> pd.Series:
    # Convert mixed percentage/ratio inputs to ratio scale.
    # Example: [55.0, 0.55] -> [0.55, 0.55].
    """Convert percentage-like values to ratio scale."""
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.where(numeric <= 1.5, numeric / 100)


def parse_salary(value: object) -> float | None:
    # Convert currency-like salary strings into numeric USD values.
    # Example: "$12,345,678" -> 12345678.0.
    """Parse currency-like salary values into float USD."""
    if pd.isna(value):
        return None
    cleaned = re.sub(r"[^\d.]", "", str(value))
    return float(cleaned) if cleaned else None


def normalize_season(value: object) -> tuple[int | None, int | None, str | None]:
    # Parse season text into numeric years and canonical season label.
    # Example: "2023-24" -> (2023, 2024, "2023-24").
    """Normalize a season value into start year, end year, and label."""
    if pd.isna(value):
        return None, None, None
    text = str(value).strip()
    match = re.search(r"(\d{4})\D+(\d{2,4})", text)
    if not match:
        return None, None, None
    start_year = int(match.group(1))
    end_text = match.group(2)
    if len(end_text) == 2:
        end_year = (start_year // 100) * 100 + int(end_text)
        if end_year < start_year:
            end_year += 100
    else:
        end_year = int(end_text)
    return start_year, end_year, f"{start_year}-{str(end_year)[-2:]}"


def season_label_to_start_year(value: object) -> int | None:
    # Extract start year for temporal comparisons and split assignment.
    # Example: "2024-25" -> 2024.
    """Extract the season start year from a season label."""
    if pd.isna(value):
        return None
    match = re.search(r"(\d{4})", str(value))
    return int(match.group(1)) if match else None


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    # Avoid invalid divisions when computing rates.
    # Example: numerator [10, 5] / denominator [2, 0] -> [5.0, NaN].
    """Divide two numeric series and return NaN where denominator is zero/missing."""
    numerator = pd.to_numeric(numerator, errors="coerce")
    denominator = pd.to_numeric(denominator, errors="coerce")
    return numerator.divide(denominator.replace(0, np.nan))


def safe_per_36(numerator: pd.Series, minutes: pd.Series) -> pd.Series:
    # Scale counting stats to per-36 production rates.
    # Example: 10 points in 20 minutes -> 18 points per 36.
    """Convert counting stats to per-36 minute rates."""
    return safe_divide(numerator, minutes) * 36


def add_missing_flags(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    # Preserve missingness information before imputation.
    # Example: age NaN -> age_was_missing=True before age is filled.
    """Add boolean missing flags for selected columns."""
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[f"{column}_was_missing"] = result[column].isna()
    return result


def fill_numeric_median(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    # Impute numeric model features with robust central values.
    # Example: [10, NaN, 20] -> [10, 15, 20].
    """Fill numeric columns with their median, falling back to zero when all missing."""
    result = df.copy()
    for column in columns:
        if column not in result.columns:
            continue
        numeric = pd.to_numeric(result[column], errors="coerce")
        median = numeric.median()
        result[column] = numeric.fillna(0 if pd.isna(median) else median)
    return result


def fill_categorical_unknown(df: pd.DataFrame, columns: list[str], value: str = "UNK") -> pd.DataFrame:
    # Impute categorical model features with a stable unknown token.
    # Example: position NaN -> "UNK".
    """Fill categorical columns with an unknown marker."""
    result = df.copy()
    for column in columns:
        if column in result.columns:
            result[column] = result[column].fillna(value).astype(str)
    return result
