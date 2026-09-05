from __future__ import annotations

from dataclasses import dataclass

RECOMMENDATION_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "workload": ("minutes", "usage_pct"),
    "scoring": (
        "points_per_100",
        "true_shooting_pct",
        "three_point_attempt_rate",
        "free_throw_rate",
    ),
    "playmaking": ("assists_per_100", "turnover_rate"),
    "rebounding": ("rebounds_per_100", "defensive_rebound_rate"),
    "defense": ("steal_rate", "block_rate", "foul_rate"),
    "physical": ("height", "weight"),
}

RECOMMENDATION_FEATURES = tuple(
    feature
    for group_features in RECOMMENDATION_FEATURE_GROUPS.values()
    for feature in group_features
)

SHRINKAGE_FEATURES = tuple(
    feature
    for feature in RECOMMENDATION_FEATURES
    if feature not in {"minutes", "height", "weight"}
)

PRESET_FEATURES: dict[str, tuple[str, ...]] = {
    "playing_profile": RECOMMENDATION_FEATURES,
    "role_similarity": tuple(
        feature
        for group in ("workload", "scoring", "playmaking", "rebounding", "defense")
        for feature in RECOMMENDATION_FEATURE_GROUPS[group]
    ),
    "scoring_profile": (
        "usage_pct",
        *RECOMMENDATION_FEATURE_GROUPS["scoring"],
    ),
    "defensive_profile": (
        *RECOMMENDATION_FEATURE_GROUPS["defense"],
        "defensive_rebound_rate",
    ),
    "workload_fit": RECOMMENDATION_FEATURE_GROUPS["workload"],
    "physical_role_fit": (
        *RECOMMENDATION_FEATURE_GROUPS["workload"],
        *RECOMMENDATION_FEATURE_GROUPS["physical"],
    ),
}

PAIR_FEATURES = tuple(f"abs_diff__{feature}" for feature in RECOMMENDATION_FEATURES)
SHRINKAGE_PRIOR_STRENGTHS = (250, 500, 750, 1000, 1500)


@dataclass(frozen=True)
class RecommendationSplitPolicy:
    """Define query seasons used for recommendation ranker selection."""

    train_start: str = "2016-17"
    train_end: str = "2021-22"
    validation: str = "2022-23"
    test: str = "2023-24"
    inference_only: str = "2024-25"


DEFAULT_RECOMMENDATION_SPLIT_POLICY = RecommendationSplitPolicy()
DEFAULT_SHRINKAGE_PRIOR_STRENGTH = 750
RECOMMENDATION_RANKER_FILENAME = "recommendation_playing_profile_ranker.joblib"
RECOMMENDATION_RANKER_VERSION = "recommendation-ranker-v1"
