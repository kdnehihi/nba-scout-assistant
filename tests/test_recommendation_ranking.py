from __future__ import annotations

import numpy as np
import pandas as pd

from src.config.recommendation_config import (
    PAIR_FEATURES,
    RECOMMENDATION_FEATURES,
    SHRINKAGE_FEATURES,
)
from src.dataset.recommendation_modeling import (
    assert_point_in_time_ranker_inputs,
    build_temporal_ranking_dataset,
    ranking_arrays,
    split_ranking_dataset,
)
from src.evaluation.evaluate_recommendation_ranking import (
    evaluate_ranking_queries,
    paired_query_bootstrap,
)
from src.scouting.ranking import (
    RecommendationRankerArtifact,
    SeasonFeaturePreprocessor,
    load_recommendation_ranker_artifact,
    normalized_feature_name,
    save_recommendation_ranker_artifact,
)
from src.scouting.recommendation import build_recommendation_base
from tests.test_scouting_recommendation import sample_role_features


def temporal_recommendation_base() -> pd.DataFrame:
    seasons = ["2021-22", "2022-23", "2023-24", "2024-25"]
    frames = []
    for season_index, season in enumerate(seasons):
        frame = sample_role_features().copy()
        frame["season"] = season
        frame["points_per_100"] += season_index * np.array([0.6, 0.5, -0.2, 0.1])
        frame["assists_per_100"] += season_index * np.array([0.2, 0.2, -0.1, 0.0])
        frames.append(frame)
    return build_recommendation_base(pd.concat(frames, ignore_index=True))


def test_recommendation_contract_contains_unique_observed_features():
    assert len(RECOMMENDATION_FEATURES) == len(set(RECOMMENDATION_FEATURES)) == 15
    assert {"scoring_creation", "playmaking", "shooting", "two_way_impact"}.isdisjoint(
        RECOMMENDATION_FEATURES
    )


def test_minutes_shrinkage_pulls_small_samples_toward_season_prior():
    base = temporal_recommendation_base()
    preprocessor = SeasonFeaturePreprocessor(prior_strength=750).fit(base)
    transformed = preprocessor.transform(base)
    season = "2024-25"
    feature = "points_per_100"
    season_rows = transformed[transformed["season"].eq(season)].sort_values("minutes")
    stats = preprocessor.season_statistics[season][feature]
    adjusted = (
        season_rows[normalized_feature_name(feature)] * stats["scale"] + stats["center"]
    )
    raw = season_rows[feature].astype(float)
    distance_ratio = (adjusted - stats["prior_mean"]).abs() / (raw - stats["prior_mean"]).abs()

    assert feature in SHRINKAGE_FEATURES
    assert distance_ratio.iloc[0] < distance_ratio.iloc[-1]


def test_persisted_season_normalization_does_not_refit_on_filtered_candidates():
    base = temporal_recommendation_base()
    preprocessor = SeasonFeaturePreprocessor(prior_strength=500).fit(base)
    full = preprocessor.transform(base)
    subset_source = base[base["position_group"].eq("guard")]
    subset = preprocessor.transform(subset_source)
    column = normalized_feature_name("usage_pct")

    expected = full.loc[subset_source.index, column].to_numpy()
    assert np.allclose(subset[column].to_numpy(), expected)


def test_temporal_labels_and_query_groups_are_point_in_time():
    ranking, _ = build_temporal_ranking_dataset(
        temporal_recommendation_base(),
        prior_strength=750,
        relevant_n=2,
    )
    assert_point_in_time_ranker_inputs(ranking)

    assert set(ranking["split"]) == {"train", "validation", "test"}
    positive_labels = ranking.groupby("query_id")["relevance"].apply(
        lambda values: sorted(values[values > 0])
    )
    assert positive_labels.map(lambda values: values == [1, 2]).all()
    assert not any("future" in column or "next" in column for column in PAIR_FEATURES)

    split_frames = split_ranking_dataset(ranking)
    X, y, qid = ranking_arrays(split_frames["validation"])
    assert list(X.columns) == list(PAIR_FEATURES)
    assert len(X) == len(y) == len(qid)
    assert np.all(qid[:-1] <= qid[1:])


def test_ranking_metrics_and_paired_bootstrap_reward_better_ordering():
    ranking = pd.DataFrame(
        {
            "query_id": [0, 0, 0, 1, 1, 1],
            "query_season": ["2023-24"] * 6,
            "candidate_player_id": [1, 2, 3, 1, 2, 3],
            "relevance": [5, 1, 0, 0, 1, 5],
        }
    )
    champion_metrics, champion_queries = evaluate_ranking_queries(
        ranking, np.array([0.0, 0.5, 1.0, 1.0, 0.5, 0.0]), algorithm="champion"
    )
    challenger_metrics, challenger_queries = evaluate_ranking_queries(
        ranking, np.array([1.0, 0.5, 0.0, 0.0, 0.5, 1.0]), algorithm="challenger"
    )
    interval = paired_query_bootstrap(challenger_queries, champion_queries, samples=200, seed=42)

    assert challenger_metrics["ndcg_at_5"] > champion_metrics["ndcg_at_5"]
    assert interval["mean_improvement"] > 0
    assert interval["ci_lower"] > 0


def test_recommendation_artifact_round_trip_preserves_scores(tmp_path):
    base = temporal_recommendation_base()
    preprocessor = SeasonFeaturePreprocessor().fit(base)
    transformed = preprocessor.transform(base)
    target = transformed.iloc[0]
    candidates = transformed.iloc[1:3]
    artifact = RecommendationRankerArtifact(
        algorithm="season_normalized_euclidean",
        preprocessor=preprocessor,
    )
    expected = artifact.score(target, candidates)

    path = save_recommendation_ranker_artifact(artifact, tmp_path / "ranker.joblib")
    restored = load_recommendation_ranker_artifact(path, required=True)

    assert restored is not None
    assert restored.version == artifact.version
    assert np.allclose(restored.score(target, candidates), expected)
