from __future__ import annotations

import pandas as pd

from dataset.cleaning import normalize_name_key, parse_salary


def _name_key_series(values: pd.Series) -> pd.Series:
    # Normalize names once so lookup works across slightly different source formats.
    """Return normalized player-name keys for matching."""
    return values.map(normalize_name_key)


def _first_existing(columns: list[str], candidates: list[str]) -> str | None:
    # Resolve the first available source column from a list of possible names.
    """Return the first candidate column present in columns."""
    lower_to_original = {column.lower(): column for column in columns}
    for candidate in candidates:
        if candidate.lower() in lower_to_original:
            return lower_to_original[candidate.lower()]
    return None


def normalize_contract_history(contract_df: pd.DataFrame) -> pd.DataFrame:
    # Convert external contract-event files into a compact display schema.
    """Normalize optional contract history rows for player detail pages."""
    if contract_df.empty:
        return pd.DataFrame()

    source = contract_df.copy()
    columns = list(source.columns)
    rename_map = {
        _first_existing(columns, ["NAME", "Player", "player_name"]): "player_name",
        _first_existing(columns, ["CONTRACT_START", "contract_start"]): "contract_start",
        _first_existing(columns, ["CONTRACT_END", "contract_end"]): "contract_end",
        _first_existing(columns, ["AVG_SALARY", "average_salary", "average_salary_usd"]): "average_salary_usd",
        _first_existing(columns, ["AGE", "age"]): "age",
        _first_existing(columns, ["GP", "games_played", "games"]): "games_played",
        _first_existing(columns, ["MIN", "minutes"]): "minutes",
        _first_existing(columns, ["PTS", "points"]): "points",
        _first_existing(columns, ["AST", "assists"]): "assists",
        _first_existing(columns, ["TRB", "REB", "rebounds"]): "rebounds",
    }
    rename_map = {key: value for key, value in rename_map.items() if key is not None}
    normalized = source.rename(columns=rename_map)
    if "player_name" not in normalized.columns:
        raise KeyError("contract history is missing a player-name column")

    if "average_salary_usd" in normalized.columns:
        normalized["average_salary_usd"] = normalized["average_salary_usd"].map(parse_salary)
    for column in ["contract_start", "contract_end", "age", "games_played", "minutes", "points", "assists", "rebounds"]:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    output_cols = [
        "player_name",
        "contract_start",
        "contract_end",
        "average_salary_usd",
        "age",
        "games_played",
        "minutes",
        "points",
        "assists",
        "rebounds",
    ]
    existing_cols = [column for column in output_cols if column in normalized.columns]
    return normalized[existing_cols].sort_values([col for col in ["player_name", "contract_start"] if col in existing_cols]).reset_index(drop=True)


def player_salary_history(
    salary_history: pd.DataFrame,
    player_name: str | None = None,
    player_id: int | str | None = None,
) -> pd.DataFrame:
    # Select salary rows for one player by ID when possible, otherwise by normalized name.
    """Return historical salaries for one player."""
    history = salary_history.copy()
    if player_id is not None and "player_id" in history.columns:
        selected = history[history["player_id"].astype(str).eq(str(player_id))]
    elif player_name is not None and "player_name" in history.columns:
        selected = history[_name_key_series(history["player_name"]).eq(normalize_name_key(player_name))]
    else:
        raise ValueError("player_name or player_id is required")
    return selected.sort_values([col for col in ["season_start_year", "season_label"] if col in selected.columns]).reset_index(drop=True)


def player_contract_history(contract_history: pd.DataFrame, player_name: str) -> pd.DataFrame:
    # Select optional contract-event rows for one player.
    """Return historical contract events for one player."""
    normalized = normalize_contract_history(contract_history)
    if normalized.empty:
        return normalized
    selected = normalized[_name_key_series(normalized["player_name"]).eq(normalize_name_key(player_name))]
    return selected.reset_index(drop=True)


def build_player_compensation_context(
    salary_history: pd.DataFrame,
    player_name: str | None = None,
    player_id: int | str | None = None,
    contract_history: pd.DataFrame | None = None,
) -> dict[str, object]:
    # Package salary and contract rows for the player detail view.
    """Return latest salary, salary history, and optional contract history."""
    salaries = player_salary_history(salary_history, player_name=player_name, player_id=player_id)
    latest_salary = salaries.tail(1).to_dict("records")[0] if not salaries.empty else None
    contracts = (
        player_contract_history(contract_history, player_name or str(salaries["player_name"].iloc[-1]))
        if contract_history is not None and (player_name is not None or not salaries.empty)
        else pd.DataFrame()
    )
    return {
        "latest_salary": latest_salary,
        "salary_history": salaries,
        "contract_history": contracts,
    }
