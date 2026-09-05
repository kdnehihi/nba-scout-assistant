from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from src.config.recommendation_config import (
    DEFAULT_RECOMMENDATION_SPLIT_POLICY,
    DEFAULT_SHRINKAGE_PRIOR_STRENGTH,
    PAIR_FEATURES,
    RECOMMENDATION_FEATURES,
    RecommendationSplitPolicy,
)
from src.scouting.ranking import (
    SeasonFeaturePreprocessor,
    build_pair_features,
    normalized_feature_name,
)


def recommendation_query_split(
    season: object,
    policy: RecommendationSplitPolicy = DEFAULT_RECOMMENDATION_SPLIT_POLICY,
) -> str:
    """Map a query season to train, validation, test, inference, or excluded."""
    label = str(season)
    if policy.train_start <= label <= policy.train_end:
        return "train"
    if label == policy.validation:
        return "validation"
    if label == policy.test:
        return "test"
    if label == policy.inference_only:
        return "inference"
    return "excluded"


def _next_season_label(season: object) -> str | None:
    try:
        start = int(str(season)[:4]) + 1
    except (TypeError, ValueError):
        return None
    return f"{start}-{str(start + 1)[-2:]}"


def _eligible_pool(
    season_df: pd.DataFrame,
    minutes_min: float | None,
) -> pd.DataFrame:
    pool = season_df.copy()
    if minutes_min is not None:
        pool = pool[pd.to_numeric(pool["minutes"], errors="coerce").fillna(0) >= minutes_min]
    return pool


def build_temporal_ranking_dataset(
    base_df: pd.DataFrame,
    prior_strength: float,
    relevant_n: int = 5,
    same_position_group: bool = True,
    minutes_min: float | None = 500,
    policy: RecommendationSplitPolicy = DEFAULT_RECOMMENDATION_SPLIT_POLICY,
    label_prior_strength: float = DEFAULT_SHRINKAGE_PRIOR_STRENGTH,
) -> tuple[pd.DataFrame, SeasonFeaturePreprocessor]:
    """Build current-season ranker inputs and graded next-season proxy labels."""
    if relevant_n < 1:
        raise ValueError("relevant_n must be at least 1.")

    required = {
        "player_id",
        "player_name",
        "season",
        "position_group",
        "minutes",
        *RECOMMENDATION_FEATURES,
    }
    missing = sorted(required - set(base_df.columns))
    if missing:
        raise KeyError(f"Missing recommendation modeling columns: {missing}")

    base = base_df.drop_duplicates(["player_id", "season"]).copy()
    input_preprocessor = SeasonFeaturePreprocessor(prior_strength=prior_strength).fit(base)
    input_profiles = input_preprocessor.transform(base)

    # Proxy outcomes use one fixed transform so tuning the input prior cannot move
    # the definition of relevance on every Optuna trial.
    label_preprocessor = SeasonFeaturePreprocessor(prior_strength=label_prior_strength).fit(base)
    outcome_profiles = label_preprocessor.transform(base)
    outcome_by_season = {
        str(season): season_df.set_index("player_id", drop=False)
        for season, season_df in outcome_profiles.groupby("season", sort=False)
    }

    rows: list[pd.DataFrame] = []
    query_id = 0
    modeled_seasons = [
        season
        for season in sorted(input_profiles["season"].dropna().astype(str).unique())
        if recommendation_query_split(season, policy) in {"train", "validation", "test"}
    ]

    for season in modeled_seasons:
        next_season = _next_season_label(season)
        if next_season is None or next_season not in outcome_by_season:
            continue
        current_pool = _eligible_pool(
            input_profiles[input_profiles["season"].eq(season)],
            minutes_min=minutes_min,
        ).reset_index(drop=True)
        future_pool = outcome_by_season[next_season]

        for _, target in current_pool.iterrows():
            target_player_id = target["player_id"]
            if target_player_id not in future_pool.index:
                continue
            candidates = current_pool[current_pool["player_id"].ne(target_player_id)].copy()
            if same_position_group:
                candidates = candidates[candidates["position_group"].eq(target["position_group"])]
            if candidates.empty:
                continue

            future_target = future_pool.loc[target_player_id]
            if isinstance(future_target, pd.DataFrame):
                future_target = future_target.iloc[0]
            available_future = candidates[candidates["player_id"].isin(future_pool.index)].copy()
            relevance_by_player: dict[object, int] = {}
            if not available_future.empty:
                future_candidates = future_pool.loc[available_future["player_id"].tolist()]
                if isinstance(future_candidates, pd.Series):
                    future_candidates = future_candidates.to_frame().T
                normalized_columns = [normalized_feature_name(feature) for feature in RECOMMENDATION_FEATURES]
                differences = (
                    future_candidates[normalized_columns].to_numpy(dtype="float64")
                    - future_target[normalized_columns].to_numpy(dtype="float64")
                )
                distances = np.sqrt(np.square(differences).mean(axis=1))
                future_order = pd.DataFrame(
                    {
                        "player_id": future_candidates["player_id"].to_numpy(),
                        "distance": distances,
                    }
                ).sort_values(["distance", "player_id"], kind="stable")
                for rank, player_id in enumerate(future_order.head(relevant_n)["player_id"], start=1):
                    relevance_by_player[player_id] = relevant_n - rank + 1

            pair_features = build_pair_features(target, candidates)
            query_rows = pair_features.reset_index(drop=True)
            query_rows.insert(0, "relevance", candidates["player_id"].map(relevance_by_player).fillna(0).astype("int8").to_numpy())
            query_rows.insert(0, "candidate_minutes", candidates["minutes"].to_numpy())
            query_rows.insert(0, "candidate_position_group", candidates["position_group"].to_numpy())
            query_rows.insert(0, "candidate_player_name", candidates["player_name"].to_numpy())
            query_rows.insert(0, "candidate_player_id", candidates["player_id"].to_numpy())
            query_rows.insert(0, "target_position_group", target["position_group"])
            query_rows.insert(0, "target_player_name", target["player_name"])
            query_rows.insert(0, "target_player_id", target_player_id)
            query_rows.insert(0, "query_season", season)
            query_rows.insert(0, "split", recommendation_query_split(season, policy))
            query_rows.insert(0, "query_id", query_id)
            rows.append(query_rows)
            query_id += 1

    if not rows:
        return pd.DataFrame(columns=["query_id", "split", "relevance", *PAIR_FEATURES]), input_preprocessor

    ranking = pd.concat(rows, ignore_index=True)
    ranking = ranking.sort_values(["query_id", "candidate_player_id"], kind="stable").reset_index(drop=True)
    return ranking, input_preprocessor


def split_ranking_dataset(
    ranking_df: pd.DataFrame,
    splits: Iterable[str] = ("train", "validation", "test"),
) -> dict[str, pd.DataFrame]:
    """Return query-contiguous ranking rows for each requested temporal split."""
    output: dict[str, pd.DataFrame] = {}
    for split in splits:
        rows = ranking_df[ranking_df["split"].eq(split)].copy()
        output[split] = rows.sort_values(["query_id", "candidate_player_id"], kind="stable").reset_index(drop=True)
    return output


def ranking_arrays(ranking_df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return model features, graded labels, and sorted numeric query IDs."""
    ordered = ranking_df.sort_values(["query_id", "candidate_player_id"], kind="stable")
    features = ordered[list(PAIR_FEATURES)].astype("float32")
    labels = ordered["relevance"].to_numpy(dtype="float32")
    query_ids = ordered["query_id"].to_numpy(dtype="int64")
    return features, labels, query_ids


def assert_point_in_time_ranker_inputs(ranking_df: pd.DataFrame) -> None:
    """Raise when ranker feature columns contain outcome-season information."""
    actual_features = [column for column in ranking_df.columns if column.startswith("abs_diff__")]
    if set(actual_features) != set(PAIR_FEATURES):
        raise ValueError("Recommendation ranker feature contract does not match PAIR_FEATURES.")
    leaked = [column for column in actual_features if "future" in column or "next" in column]
    if leaked:
        raise ValueError(f"Future information found in recommendation ranker inputs: {leaked}")
