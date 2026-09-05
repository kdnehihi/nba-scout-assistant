from __future__ import annotations

from collections.abc import Callable
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config.recommendation_config import PAIR_FEATURES

LEGACY_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "workload": ("minutes", "usage_pct"),
    "scoring": (
        "points_per_100",
        "usage_pct",
        "true_shooting_pct",
        "three_point_attempt_rate",
        "free_throw_rate",
        "scoring_creation",
        "shooting",
        "rim_pressure",
    ),
    "playmaking": ("assists_per_100", "turnover_rate", "playmaking"),
    "rebounding": ("rebounds_per_100", "defensive_rebound_rate", "rebounding"),
    "defense": (
        "steal_rate",
        "block_rate",
        "defensive_rebound_rate",
        "foul_rate",
        "perimeter_defense",
        "interior_defense",
        "two_way_impact",
    ),
}


def _discounted_gain(relevance: np.ndarray) -> float:
    gains = np.power(2.0, relevance.astype("float64")) - 1.0
    discounts = np.log2(np.arange(2, len(relevance) + 2, dtype="float64"))
    return float(np.sum(gains / discounts))


def evaluate_ranking_queries(
    ranking_df: pd.DataFrame,
    scores: np.ndarray | pd.Series,
    top_k: int = 5,
    algorithm: str = "ranker",
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate graded recommendations and return aggregate plus per-query metrics."""
    if len(ranking_df) != len(scores):
        raise ValueError("ranking_df and scores must have the same number of rows.")
    evaluated = ranking_df.copy()
    evaluated["ranking_score"] = np.asarray(scores, dtype="float64")
    query_rows: list[dict[str, float | int | str]] = []
    recommended_players: set[object] = set()

    for query_id, group in evaluated.groupby("query_id", sort=True):
        ranked = group.sort_values(
            ["ranking_score", "candidate_player_id"],
            ascending=[False, True],
            kind="stable",
        )
        top = ranked.head(top_k)
        relevance = top["relevance"].to_numpy(dtype="float64")
        ideal = np.sort(group["relevance"].to_numpy(dtype="float64"))[::-1][:top_k]
        ideal_dcg = _discounted_gain(ideal)
        ndcg = _discounted_gain(relevance) / ideal_dcg if ideal_dcg > 0 else np.nan
        relevant_total = int(group["relevance"].gt(0).sum())
        hit_positions = np.flatnonzero(relevance > 0)
        hit_count = len(hit_positions)
        recommended_players.update(top["candidate_player_id"].tolist())
        query_rows.append(
            {
                "algorithm": algorithm,
                "query_id": int(query_id),
                "query_season": str(group["query_season"].iloc[0]),
                "ndcg_at_5": float(ndcg),
                "recall_at_5": float(hit_count / relevant_total) if relevant_total else np.nan,
                "hit_rate_at_5": float(hit_count > 0),
                "mrr": float(1.0 / (hit_positions[0] + 1)) if hit_count else 0.0,
            }
        )

    per_query = pd.DataFrame(query_rows)
    candidate_universe = max(1, evaluated["candidate_player_id"].nunique())
    metrics = {
        "queries": float(len(per_query)),
        "ndcg_at_5": float(per_query["ndcg_at_5"].mean()),
        "recall_at_5": float(per_query["recall_at_5"].mean()),
        "hit_rate_at_5": float(per_query["hit_rate_at_5"].mean()),
        "mrr": float(per_query["mrr"].mean()),
        "coverage_at_5": float(len(recommended_players) / candidate_universe),
    }
    return metrics, per_query


def paired_query_bootstrap(
    challenger: pd.DataFrame,
    champion: pd.DataFrame,
    metric: str = "ndcg_at_5",
    samples: int = 2000,
    seed: int = 42,
) -> dict[str, float]:
    """Return a paired query-bootstrap confidence interval for metric improvement."""
    paired = challenger[["query_id", metric]].merge(
        champion[["query_id", metric]],
        on="query_id",
        suffixes=("_challenger", "_champion"),
        validate="one_to_one",
    ).dropna()
    differences = (
        paired[f"{metric}_challenger"] - paired[f"{metric}_champion"]
    ).to_numpy(dtype="float64")
    if len(differences) == 0:
        return {"mean_improvement": np.nan, "ci_lower": np.nan, "ci_upper": np.nan}
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(samples, dtype="float64")
    for sample_idx in range(samples):
        bootstrap_means[sample_idx] = rng.choice(differences, size=len(differences), replace=True).mean()
    return {
        "mean_improvement": float(differences.mean()),
        "ci_lower": float(np.quantile(bootstrap_means, 0.025)),
        "ci_upper": float(np.quantile(bootstrap_means, 0.975)),
    }


def benchmark_query_latency(
    scorer: Callable[[pd.DataFrame], np.ndarray],
    query_frames: list[pd.DataFrame],
    repeats: int = 3,
) -> dict[str, float]:
    """Measure p50 and p95 model-scoring latency on complete query candidate pools."""
    timings_ms: list[float] = []
    for query_df in query_frames:
        for _ in range(repeats):
            started = perf_counter()
            scorer(query_df)
            timings_ms.append((perf_counter() - started) * 1000.0)
    return {
        "latency_p50_ms": float(np.quantile(timings_ms, 0.50)) if timings_ms else np.nan,
        "latency_p95_ms": float(np.quantile(timings_ms, 0.95)) if timings_ms else np.nan,
    }


def _legacy_standardized_distance(scoring_df: pd.DataFrame, features: tuple[str, ...]) -> np.ndarray:
    available = [feature for feature in features if feature in scoring_df.columns]
    matrix = scoring_df[available].apply(pd.to_numeric, errors="coerce")
    matrix = matrix.fillna(matrix.median(numeric_only=True)).fillna(0)
    scaled = StandardScaler().fit_transform(matrix)
    return np.sqrt(np.square(scaled[1:] - scaled[0]).mean(axis=1))


def _legacy_reliability(minutes: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(minutes, errors="coerce")
    lower, upper = numeric.quantile([0.05, 0.95])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
        return np.full(len(numeric), 0.5)
    return ((numeric.clip(lower, upper) - lower) / (upper - lower)).fillna(0.5).to_numpy()


def score_legacy_weighted_euclidean(
    ranking_df: pd.DataFrame,
    raw_base: pd.DataFrame,
) -> np.ndarray:
    """Reproduce the pre-upgrade playing-profile score for champion comparison."""
    profile_lookup = raw_base.drop_duplicates(["player_id", "season"]).set_index(["season", "player_id"])
    output = pd.Series(index=ranking_df.index, dtype="float64")
    for _, query in ranking_df.groupby("query_id", sort=True):
        season = query["query_season"].iloc[0]
        target_id = query["target_player_id"].iloc[0]
        target = profile_lookup.loc[(season, target_id)]
        candidate_keys = [(season, player_id) for player_id in query["candidate_player_id"]]
        candidates = profile_lookup.loc[candidate_keys].reset_index()
        scoring_df = pd.concat([target.to_frame().T, candidates], ignore_index=True)
        group_distances = np.column_stack(
            [_legacy_standardized_distance(scoring_df, features) for features in LEGACY_FEATURE_GROUPS.values()]
        )
        role_score = 1.0 / (1.0 + group_distances.mean(axis=1))
        physical_distance = _legacy_standardized_distance(scoring_df, ("height", "weight"))
        physical_score = 1.0 / (1.0 + physical_distance)
        reliability = _legacy_reliability(candidates["minutes"])
        output.loc[query.index] = 0.85 * role_score + 0.10 * reliability + 0.05 * physical_score
    return output.to_numpy(dtype="float64")


def model_scores(model: object, ranking_df: pd.DataFrame) -> np.ndarray:
    """Score ranking rows with an estimator that accepts the persisted pair contract."""
    return np.asarray(model.predict(ranking_df[list(PAIR_FEATURES)]), dtype="float64")
