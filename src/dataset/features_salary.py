from __future__ import annotations

import pandas as pd

from .cleaning import (
    add_missing_flags,
    fill_categorical_unknown,
    fill_numeric_median,
    normalize_name_key,
    normalize_team_abbreviation,
    parse_salary,
)
from .splits import assign_salary_temporal_split


SALARY_MODEL_FEATURES = [
    "age",
    "height",
    "weight",
    "minutes",
    "usage_pct",
    "points_per_100",
    "assists_per_100",
    "rebounds_per_100",
    "true_shooting_pct",
    "three_point_attempt_rate",
    "free_throw_rate",
    "turnover_rate",
    "steal_rate",
    "block_rate",
    "defensive_rebound_rate",
    "foul_rate",
    "scoring_creation",
    "playmaking",
    "shooting",
    "rim_pressure",
    "rebounding",
    "perimeter_defense",
    "interior_defense",
    "two_way_impact",
]


def build_salary_training(
    salaries: pd.DataFrame,
    salary_cap: pd.DataFrame,
    players: pd.DataFrame,
    role_features: pd.DataFrame,
) -> pd.DataFrame:
    # Merge salary, cap, bio, and role data into salary modeling rows.
    """Build model-ready salary training data."""
    salary = salaries.copy()
    salary["salary_usd"] = salary["salary_usd"].map(parse_salary) if salary["salary_usd"].dtype == "object" else pd.to_numeric(salary["salary_usd"], errors="coerce")
    salary["team"] = salary.get("team", "UNK")
    salary["team_id"] = salary["team"].map(normalize_team_abbreviation).fillna("UNK")
    salary["player_name_key"] = salary["player_name"].map(normalize_name_key)

    caps = salary_cap.copy()
    caps["salary_cap_usd"] = pd.to_numeric(caps["salary_cap_usd"], errors="coerce")
    salary = salary.merge(
        caps.rename(columns={"season": "season_label"}),
        on="season_label",
        how="left",
    )
    salary["salary_cap_share"] = salary["salary_usd"] / salary["salary_cap_usd"]
    salary["target_salary_usd"] = salary["salary_usd"]

    player_meta = players.copy()
    player_meta["player_name_key"] = player_meta["player_name"].map(normalize_name_key)
    player_cols = [column for column in ["player_name_key", "player_id", "birth_date", "position", "height", "weight"] if column in player_meta.columns]
    salary = salary.merge(player_meta[player_cols].drop_duplicates("player_name_key"), on="player_name_key", how="left")

    if "birth_date" in salary.columns:
        birth_date = pd.to_datetime(salary["birth_date"], errors="coerce")
        season_date = pd.to_datetime(salary["season_start_year"].astype(str) + "-10-01", errors="coerce")
        salary["age"] = (season_date - birth_date).dt.days / 365.25

    role = role_features.copy()
    if "player_id" in salary.columns and "player_id" in role.columns:
        role_cols = [
            column
            for column in ["player_id", "season", "team_id", *SALARY_MODEL_FEATURES]
            if column in role.columns
        ]
        salary = salary.merge(
            role[role_cols].drop_duplicates(["player_id", "season"]),
            left_on=["player_id", "season_label"],
            right_on=["player_id", "season"],
            how="left",
            suffixes=("", "_role"),
        )

    if "position_role" in salary.columns:
        salary["position"] = salary["position"].fillna(salary["position_role"])
        salary = salary.drop(columns=["position_role"])
    for column in ["height", "weight", "age"]:
        role_column = f"{column}_role"
        if role_column in salary.columns:
            salary[column] = salary[column].fillna(salary[role_column])
            salary = salary.drop(columns=[role_column])

    model_numeric = [column for column in SALARY_MODEL_FEATURES if column in salary.columns]
    salary = add_missing_flags(salary, ["salary_usd", "salary_cap_usd", "salary_cap_share", *model_numeric])
    salary = fill_numeric_median(salary, ["salary_usd", "salary_cap_usd", "salary_cap_share", "target_salary_usd", *model_numeric])
    salary = fill_categorical_unknown(salary, ["team", "team_id", "position"])
    salary["split"] = salary["season_label"].map(assign_salary_temporal_split)
    salary = salary.drop(columns=[column for column in ["player_name_key", "season"] if column in salary.columns])
    return salary.reset_index(drop=True)

