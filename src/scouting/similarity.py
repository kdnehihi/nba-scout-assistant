from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import SimilarityConfig
from .signals import season_start_year


def build_similarity_base(role_features: pd.DataFrame, salary: pd.DataFrame) -> pd.DataFrame:
    # Join role features with salary context for replacement candidate ranking.
    """Return player-season role rows with salary context."""
    role = role_features.copy()
    role["season_start_year"] = role["season"].map(season_start_year)
    salary_context = salary[["player_id", "season_start_year", "salary_usd", "salary_cap_share", "team_id"]].rename(
        columns={"team_id": "salary_team_id"}
    )
    return role.merge(salary_context, on=["player_id", "season_start_year"], how="left")


def standardized_similarity_matrix(df: pd.DataFrame, features: list[str] | tuple[str, ...]) -> tuple[pd.DataFrame, list[str]]:
    # Standardize numeric role features before distance calculations.
    """Return standardized feature matrix and retained feature names."""
    available = [feature for feature in features if feature in df.columns]
    matrix = df[available].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.fillna(matrix.median(numeric_only=True)).fillna(0)
    scaled = StandardScaler().fit_transform(matrix)
    return pd.DataFrame(scaled, index=df.index, columns=available), available


def find_replacement_candidates(
    base_df: pd.DataFrame,
    player_name: str,
    season: str | None = None,
    top_n: int = 10,
    cheaper_only: bool = False,
    younger_only: bool = False,
    same_position_group: bool | None = None,
    salary_cap_share_max: float | None = None,
    age_max: float | None = None,
    minutes_min: float | None = None,
    config: SimilarityConfig = SimilarityConfig(),
) -> pd.DataFrame:
    # Rank same-season players by standardized role-profile distance.
    """Return replacement candidates for a player-season query."""
    base = base_df.copy()
    position_filter = config.same_position_group if same_position_group is None else same_position_group
    name_mask = base["player_name"].str.contains(player_name, case=False, na=False, regex=False)
    if season is not None:
        name_mask &= base["season"].eq(season)
    target_rows = base[name_mask].sort_values(["season_start_year", "minutes"], ascending=[False, False])
    if target_rows.empty:
        raise ValueError(f"No player-season row matched player_name={player_name!r}, season={season!r}")

    target = target_rows.iloc[0]
    season_df = base[base["season"].eq(target["season"])].copy()
    season_df = season_df[season_df["player_id"].ne(target["player_id"])]
    if position_filter and pd.notna(target.get("position")):
        target_position = str(target["position"])
        season_df = season_df[season_df["position"].astype(str).str.contains(target_position[:1], na=False)]
    if cheaper_only and pd.notna(target.get("salary_cap_share")):
        season_df = season_df[season_df["salary_cap_share"].fillna(np.inf) <= float(target["salary_cap_share"])]
    if younger_only and pd.notna(target.get("age")):
        season_df = season_df[season_df["age"].fillna(np.inf) <= float(target["age"])]
    if salary_cap_share_max is not None:
        season_df = season_df[season_df["salary_cap_share"].fillna(np.inf) <= salary_cap_share_max]
    if age_max is not None:
        season_df = season_df[season_df["age"].fillna(np.inf) <= age_max]
    if minutes_min is not None:
        season_df = season_df[season_df["minutes"].fillna(0) >= minutes_min]

    scoring_df = pd.concat([target.to_frame().T, season_df], ignore_index=True)
    matrix, retained_features = standardized_similarity_matrix(scoring_df, config.features)
    target_vector = matrix.iloc[0].to_numpy(dtype="float64")
    candidate_matrix = matrix.iloc[1:].to_numpy(dtype="float64")
    distances = np.sqrt(((candidate_matrix - target_vector) ** 2).mean(axis=1))

    candidates = scoring_df.iloc[1:].copy()
    candidates["target_player_name"] = target["player_name"]
    candidates["target_player_id"] = target["player_id"]
    candidates["target_season"] = target["season"]
    candidates["similarity_distance"] = distances
    candidates["similarity_score"] = 1 / (1 + candidates["similarity_distance"])
    candidates["salary_cap_share_gap"] = candidates["salary_cap_share"] - target.get("salary_cap_share", np.nan)
    candidates["age_gap"] = candidates["age"] - target.get("age", np.nan)
    candidates["features_used"] = ", ".join(retained_features)
    output_cols = [
        "target_player_name",
        "target_player_id",
        "target_season",
        "player_id",
        "player_name",
        "season",
        "team_id",
        "position",
        "age",
        "minutes",
        "usage_pct",
        "points_per_100",
        "assists_per_100",
        "rebounds_per_100",
        "true_shooting_pct",
        "salary_usd",
        "salary_cap_share",
        "salary_cap_share_gap",
        "age_gap",
        "similarity_distance",
        "similarity_score",
        "features_used",
    ]
    return candidates[output_cols].sort_values("similarity_distance").head(top_n).reset_index(drop=True)

