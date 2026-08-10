from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


TABULAR_EXTENSIONS = {".csv", ".tsv", ".parquet", ".json", ".jsonl", ".xlsx"}


@dataclass(frozen=True)
class DataPaths:
    """Resolved paths for project data layers."""

    data_dir: Path = Path("data")

    @property
    def raw_dir(self) -> Path:
        # Build the raw data layer path from the configured data root.
        return self.data_dir / "raw"

    @property
    def bronze_dir(self) -> Path:
        # Build the bronze data layer path from the configured data root.
        return self.data_dir / "bronze"

    @property
    def silver_dir(self) -> Path:
        # Build the silver data layer path from the configured data root.
        return self.data_dir / "silver"

    @property
    def gold_dir(self) -> Path:
        # Build the gold data layer path from the configured data root.
        return self.data_dir / "gold"


def resolve_data_paths(data_dir: Path | str = "data") -> DataPaths:
    # Normalize the user-provided data root before constructing layer paths.
    """Resolve the data root directory and return standard layer paths."""
    root = Path(data_dir).expanduser().resolve()
    return DataPaths(data_dir=root)


def require_file(path: Path | str) -> Path:
    # Fail early when a required data file is missing or not a file.
    """Return a resolved file path or raise a clear error."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Expected a file, got: {resolved}")
    return resolved


def require_dir(path: Path | str) -> Path:
    # Fail early when a required data directory is missing or not a directory.
    """Return a resolved directory path or raise a clear error."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Directory not found: {resolved}")
    if not resolved.is_dir():
        raise FileNotFoundError(f"Expected a directory, got: {resolved}")
    return resolved


def list_tabular_files(
    directory: Path | str,
    extensions: Iterable[str] = TABULAR_EXTENSIONS,
) -> list[Path]:
    # Discover source files that can be read by the tabular loader.
    """List supported tabular files under a directory."""
    root = require_dir(directory)
    normalized = {extension.lower() for extension in extensions}
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in normalized
    )


def load_tabular_data(file_path: Path | str, **kwargs) -> pd.DataFrame:
    # Dispatch to the correct pandas reader based on file extension.
    """Load a supported tabular file into a pandas dataframe."""
    path = require_file(file_path)
    suffix = path.suffix.lower()
    if suffix not in TABULAR_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {suffix}")
    if suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", **kwargs)
    if suffix == ".parquet":
        return pd.read_parquet(path, **kwargs)
    if suffix == ".json":
        return pd.read_json(path, **kwargs)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True, **kwargs)
    if suffix == ".xlsx":
        return pd.read_excel(path, **kwargs)
    raise ValueError(f"Unsupported file extension: {suffix}")


def require_columns(df: pd.DataFrame, required_columns: Iterable[str], dataset_name: str) -> None:
    # Validate the minimum schema contract for a loaded dataset.
    """Validate that a dataframe has required columns."""
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise KeyError(f"{dataset_name} is missing required columns: {missing}")


def load_players(paths: DataPaths) -> pd.DataFrame:
    # Read player bio/profile data used by feature builders.
    """Load normalized player bio/profile data."""
    df = load_tabular_data(paths.raw_dir / "players.parquet")
    require_columns(df, ["player_id", "player_name"], "players")
    return df


def load_player_game_logs(paths: DataPaths) -> pd.DataFrame:
    # Read canonical game logs used for performance and long-term features.
    """Load canonical player game logs from the silver layer."""
    df = load_tabular_data(paths.silver_dir / "player_game_logs.parquet")
    require_columns(
        df,
        ["player_id", "game_id", "game_date", "season", "team_id", "minutes"],
        "player_game_logs",
    )
    return df


def load_player_season_stats(paths: DataPaths) -> pd.DataFrame:
    # Read season-level rate stats used for role features.
    """Load player-season rate and role stats from the raw layer."""
    df = load_tabular_data(paths.raw_dir / "player_season_stats.parquet")
    require_columns(df, ["player_id", "season"], "player_season_stats")
    return df


def load_player_season_salaries(paths: DataPaths) -> pd.DataFrame:
    # Read normalized salary rows used for player compensation context.
    """Load normalized player-season salary data from the silver layer."""
    df = load_tabular_data(paths.silver_dir / "player_season_salaries.parquet")
    require_columns(df, ["player_name", "season_label", "salary_usd"], "player_season_salaries")
    return df


def load_salary_cap(paths: DataPaths) -> pd.DataFrame:
    # Read salary-cap context used to normalize salary history.
    """Load salary-cap data by NBA season."""
    df = load_tabular_data(paths.raw_dir / "salary_cap" / "salary_cap_by_season.csv")
    require_columns(df, ["season", "salary_cap_usd"], "salary_cap_by_season")
    return df


def load_role_features_clean(paths: DataPaths) -> pd.DataFrame:
    # Read the materialized role feature table from the gold layer.
    """Load clean player role features from the gold layer."""
    return load_tabular_data(paths.gold_dir / "player_role_features_clean.parquet")


def load_performance_training_clean(paths: DataPaths) -> pd.DataFrame:
    # Read the materialized short-term training table from the gold layer.
    """Load clean short-term performance training data from the gold layer."""
    return load_tabular_data(paths.gold_dir / "performance_training_clean.parquet")


def load_player_salary_history_clean(paths: DataPaths) -> pd.DataFrame:
    # Read the materialized player salary history table from the gold layer.
    """Load clean player salary history from the gold layer."""
    return load_tabular_data(paths.gold_dir / "player_salary_history_clean.parquet")


def load_contract_history(paths: DataPaths, required: bool = False) -> pd.DataFrame:
    # Read optional contract-event history for player detail pages.
    """Load optional contract history rows from the raw data layer."""
    path = paths.raw_dir / "contract_value" / "contract_events.csv"
    if not path.exists() and not required:
        return pd.DataFrame()
    return load_tabular_data(path)


def load_long_term_training(paths: DataPaths) -> pd.DataFrame:
    # Read the materialized long-term training table from the gold layer.
    """Load clean long-term player forecast training data from the gold layer."""
    return load_tabular_data(paths.gold_dir / "long_term_player_forecast_training.parquet")
