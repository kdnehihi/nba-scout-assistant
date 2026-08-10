from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


STAT_FEATURE_GROUPS: dict[str, list[str]] = {
    "workload": [
        "minutes",
        "usage_pct",
    ],
    "scoring": [
        "points_per_100",
        "usage_pct",
        "true_shooting_pct",
        "three_point_attempt_rate",
        "free_throw_rate",
        "scoring_creation",
        "shooting",
        "rim_pressure",
    ],
    "playmaking": [
        "assists_per_100",
        "turnover_rate",
        "playmaking",
    ],
    "rebounding": [
        "rebounds_per_100",
        "defensive_rebound_rate",
        "rebounding",
    ],
    "defense": [
        "steal_rate",
        "block_rate",
        "defensive_rebound_rate",
        "foul_rate",
        "perimeter_defense",
        "interior_defense",
        "two_way_impact",
    ],
}

PHYSICAL_FEATURES = ["height", "weight"]
ALL_RECOMMENDER_FEATURES = sorted(
    {feature for features in STAT_FEATURE_GROUPS.values() for feature in features}
    | set(PHYSICAL_FEATURES)
)

RANKING_PRESETS: dict[str, dict[str, float]] = {
    "role_similarity": {
        "role_similarity_score": 1.00,
        "workload_reliability_score": 0.00,
        "physical_match_score": 0.00,
    },
    "playing_profile": {
        "role_similarity_score": 0.85,
        "workload_reliability_score": 0.10,
        "physical_match_score": 0.05,
    },
    "workload_fit": {
        "role_similarity_score": 0.75,
        "workload_reliability_score": 0.25,
        "physical_match_score": 0.00,
    },
    "physical_role_fit": {
        "role_similarity_score": 0.75,
        "workload_reliability_score": 0.05,
        "physical_match_score": 0.20,
    },
}


def season_start_year(season: object) -> int | None:
    """Return the start year from an NBA season label."""
    try:
        return int(str(season)[:4])
    except (TypeError, ValueError):
        return None


def position_group(position: object) -> str:
    """Map raw positions into guard, wing, big, or unknown groups."""
    text = str(position).upper()
    has_g = "G" in text
    has_f = "F" in text
    has_c = "C" in text
    if has_c and not has_g:
        return "big"
    if has_f and not has_c:
        return "wing"
    if has_g and not has_c:
        return "guard"
    if has_f and has_c:
        return "big"
    return "unknown"


def latest_form_by_player_season(performance_df: pd.DataFrame) -> pd.DataFrame:
    """Return latest short-term form deltas by player-season."""
    if performance_df.empty:
        return pd.DataFrame(columns=["player_id", "season"])
    sort_cols = [column for column in ["player_id", "season", "as_of_date", "game_id"] if column in performance_df.columns]
    latest = (
        performance_df.sort_values(sort_cols)
        .groupby(["player_id", "season"], as_index=False)
        .tail(1)
        .copy()
    )
    cols = [
        "player_id",
        "season",
        "pts_last_5_minus_season_avg",
        "ast_last_5_minus_season_avg",
        "reb_last_5_minus_season_avg",
        "min_last_5_minus_season_avg",
    ]
    return latest[[col for col in cols if col in latest.columns]].reset_index(drop=True)


def physical_context_by_player_season(physical_df: pd.DataFrame) -> pd.DataFrame:
    """Return height and weight by player-season when available."""
    required = ["player_id", "season_start_year", "height", "weight"]
    if not set(required).issubset(physical_df.columns):
        return pd.DataFrame(columns=required)
    physical = physical_df[required].copy()
    for column in ["height", "weight"]:
        physical[column] = pd.to_numeric(physical[column], errors="coerce")
    return physical.drop_duplicates(["player_id", "season_start_year"])


def build_recommendation_base(
    role_df: pd.DataFrame,
    physical_df: pd.DataFrame | None = None,
    performance_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a player-season recommendation base from gold scouting tables."""
    base = role_df.copy()
    base["season_start_year"] = base["season"].map(season_start_year)
    base["position_group"] = base["position"].map(position_group)

    if physical_df is not None and not physical_df.empty:
        physical_context = physical_context_by_player_season(physical_df)
        if not physical_context.empty:
            base = base.merge(
                physical_context,
                on=["player_id", "season_start_year"],
                how="left",
                suffixes=("", "_physical"),
            )
            for column in PHYSICAL_FEATURES:
                physical_col = f"{column}_physical"
                if physical_col in base.columns:
                    base[column] = pd.to_numeric(base.get(column), errors="coerce").fillna(base[physical_col])
                    base = base.drop(columns=[physical_col])

    if performance_df is not None and not performance_df.empty:
        form_context = latest_form_by_player_season(performance_df)
        if not form_context.empty:
            base = base.merge(form_context, on=["player_id", "season"], how="left")

    for column in [*ALL_RECOMMENDER_FEATURES, "minutes"]:
        if column in base.columns:
            base[column] = pd.to_numeric(base[column], errors="coerce")

    return base


def robust_minmax_score(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Return clipped 0-1 scores that are less sensitive to outliers."""
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(0.5, index=values.index)
    lower = numeric.quantile(0.05)
    upper = numeric.quantile(0.95)
    if pd.isna(lower) or pd.isna(upper) or lower == upper:
        return pd.Series(0.5, index=values.index)
    clipped = numeric.astype("float64").clip(float(lower), float(upper))
    score = (clipped - float(lower)) / (float(upper) - float(lower))
    if not higher_is_better:
        score = 1 - score
    return score.fillna(0.5)


def select_target_row(
    base_df: pd.DataFrame,
    player_name: str,
    season: str | None = None,
) -> pd.Series:
    """Select one target player-season row for a recommendation query."""
    mask = base_df["player_name"].str.contains(player_name, case=False, na=False, regex=False)
    if season is not None:
        mask &= base_df["season"].eq(season)
    matches = base_df[mask].sort_values(["season_start_year", "minutes"], ascending=[False, False])
    if matches.empty:
        raise ValueError(f"No target matched player_name={player_name!r}, season={season!r}")
    return matches.iloc[0]


def generate_candidates(
    base_df: pd.DataFrame,
    target: pd.Series,
    same_season: bool = True,
    same_position_group: bool = True,
    minutes_min: float | None = 500,
) -> pd.DataFrame:
    """Filter the recommendation candidate pool before scoring."""
    candidates = base_df[base_df["player_id"].ne(target["player_id"])].copy()
    if same_season:
        candidates = candidates[candidates["season"].eq(target["season"])]
    if same_position_group and pd.notna(target.get("position_group")):
        candidates = candidates[candidates["position_group"].eq(target["position_group"])]
    if minutes_min is not None:
        candidates = candidates[candidates["minutes"].fillna(0) >= minutes_min]
    return candidates.reset_index(drop=True)


def standardized_group_distance(
    scoring_df: pd.DataFrame,
    target_index: int,
    features: list[str],
) -> pd.Series:
    """Return standardized profile distances from one target row."""
    available = [feature for feature in features if feature in scoring_df.columns]
    if not available:
        return pd.Series(np.nan, index=scoring_df.index)
    matrix = scoring_df[available].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.fillna(matrix.median(numeric_only=True)).fillna(0)
    scaled = StandardScaler().fit_transform(matrix)
    target_vector = scaled[target_index]
    distances = np.sqrt(((scaled - target_vector) ** 2).mean(axis=1))
    return pd.Series(distances, index=scoring_df.index)


def add_similarity_scores(target: pd.Series, candidates: pd.DataFrame) -> pd.DataFrame:
    """Add role-group and physical similarity scores to candidate rows."""
    scoring_df = pd.concat([target.to_frame().T, candidates], ignore_index=True)
    result = scoring_df.iloc[1:].copy()

    group_distance_cols = []
    for group_name, features in STAT_FEATURE_GROUPS.items():
        distances = standardized_group_distance(scoring_df, target_index=0, features=features)
        distance_col = f"{group_name}_distance"
        score_col = f"{group_name}_score"
        result[distance_col] = distances.iloc[1:].to_numpy(dtype="float64")
        result[score_col] = 1 / (1 + result[distance_col])
        group_distance_cols.append(distance_col)

    result["role_similarity_distance"] = result[group_distance_cols].mean(axis=1)
    result["role_similarity_score"] = 1 / (1 + result["role_similarity_distance"])

    physical_distances = standardized_group_distance(scoring_df, target_index=0, features=PHYSICAL_FEATURES)
    result["physical_distance"] = physical_distances.iloc[1:].to_numpy(dtype="float64")
    result["physical_match_score"] = 1 / (1 + result["physical_distance"])
    return result


def add_practical_scores(target: pd.Series, candidates: pd.DataFrame) -> pd.DataFrame:
    """Add workload and practical gap scores to candidate rows."""
    result = candidates.copy()
    result["workload_reliability_score"] = robust_minmax_score(result["minutes"], higher_is_better=True)
    result["height_gap"] = result["height"] - target.get("height", np.nan) if "height" in result.columns else np.nan
    result["weight_gap"] = result["weight"] - target.get("weight", np.nan) if "weight" in result.columns else np.nan
    result["minutes_gap"] = result["minutes"] - target.get("minutes", np.nan)
    if "physical_match_score" not in result.columns:
        result["physical_match_score"] = 0.5
    return result


def add_final_score(candidates: pd.DataFrame, preset: str = "playing_profile") -> pd.DataFrame:
    """Apply a deterministic ranking preset to scored candidates."""
    if preset not in RANKING_PRESETS:
        raise KeyError(f"Unknown ranking preset: {preset}")
    weights = RANKING_PRESETS[preset]
    result = candidates.copy()
    result["ranking_preset"] = preset
    result["recommendation_score"] = 0.0
    for column, weight in weights.items():
        result["recommendation_score"] += weight * result[column].fillna(0.5)
    return result


def matched_groups(row: pd.Series, top_n: int = 3) -> str:
    """Return the strongest matching role groups for explanation."""
    score_cols = [f"{group}_score" for group in STAT_FEATURE_GROUPS]
    available = [(col.replace("_score", ""), row[col]) for col in score_cols if col in row and pd.notna(row[col])]
    if not available:
        return ""
    ranked = sorted(available, key=lambda item: item[1], reverse=True)[:top_n]
    return ", ".join(f"{name}:{score:.2f}" for name, score in ranked)


def recommendation_reason(row: pd.Series) -> str:
    """Return a compact human-readable recommendation reason."""
    parts = [f"matched groups [{row['matched_groups']}]"]
    if pd.notna(row.get("height_gap")):
        parts.append(f"height gap {row['height_gap']:+.1f}")
    if pd.notna(row.get("weight_gap")):
        parts.append(f"weight gap {row['weight_gap']:+.1f}")
    if pd.notna(row.get("minutes_gap")):
        parts.append(f"minutes gap {row['minutes_gap']:+.0f}")
    return "; ".join(parts)


def recommend_players(
    base_df: pd.DataFrame,
    player_name: str,
    season: str | None = None,
    top_n: int = 5,
    preset: str = "playing_profile",
    same_season: bool = True,
    same_position_group: bool = True,
    minutes_min: float | None = 500,
) -> pd.DataFrame:
    """Return ranked player recommendations with grouped explanations."""
    target = select_target_row(base_df, player_name=player_name, season=season)
    candidates = generate_candidates(
        base_df=base_df,
        target=target,
        same_season=same_season,
        same_position_group=same_position_group,
        minutes_min=minutes_min,
    )
    if candidates.empty:
        raise ValueError("No candidates matched the requested filters.")

    scored = add_similarity_scores(target, candidates)
    scored = add_practical_scores(target, scored)
    scored = add_final_score(scored, preset=preset)
    scored["target_player_name"] = target["player_name"]
    scored["target_player_id"] = target["player_id"]
    scored["target_season"] = target["season"]
    scored["matched_groups"] = scored.apply(matched_groups, axis=1)
    scored["recommendation_reason"] = scored.apply(recommendation_reason, axis=1)

    output_cols = [
        "target_player_name",
        "target_player_id",
        "target_season",
        "player_id",
        "player_name",
        "season",
        "team_id",
        "position",
        "position_group",
        "height",
        "weight",
        "minutes",
        "recommendation_score",
        "role_similarity_score",
        "workload_reliability_score",
        "physical_match_score",
        "height_gap",
        "weight_gap",
        "minutes_gap",
        "matched_groups",
        "recommendation_reason",
        "ranking_preset",
    ]
    output_cols = [col for col in output_cols if col in scored.columns]
    return scored[output_cols].sort_values("recommendation_score", ascending=False).head(top_n).reset_index(drop=True)
