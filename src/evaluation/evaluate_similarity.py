from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def similarity_diagnostics(candidates: pd.DataFrame) -> dict[str, float]:
    # Summarize replacement-candidate ranking outputs without supervised labels.
    """Return simple diagnostics for a similarity candidate table."""
    if candidates.empty:
        return {
            "rows": 0.0,
            "avg_similarity_score": float("nan"),
            "avg_similarity_distance": float("nan"),
            "avg_salary_cap_share_gap": float("nan"),
            "avg_age_gap": float("nan"),
        }
    return {
        "rows": float(len(candidates)),
        "avg_similarity_score": float(candidates["similarity_score"].mean()),
        "avg_similarity_distance": float(candidates["similarity_distance"].mean()),
        "avg_salary_cap_share_gap": float(candidates["salary_cap_share_gap"].mean()),
        "avg_age_gap": float(candidates["age_gap"].mean()),
    }


def build_profile_clusters(
    base_df: pd.DataFrame,
    features: list[str] | tuple[str, ...],
    target_cluster_size: int = 8,
    random_state: int = 42,
) -> pd.DataFrame:
    # Cluster player-season profiles as an unsupervised recommendation diagnostic.
    """Return player-season profile clusters from standardized recommendation features."""
    available = [feature for feature in features if feature in base_df.columns]
    if not available:
        raise ValueError("No clustering features were available.")
    if target_cluster_size < 2:
        raise ValueError("target_cluster_size must be at least 2.")

    cluster_rows = []
    for season, season_df in base_df.groupby("season", sort=False):
        season_df = season_df.reset_index(drop=True).copy()
        if len(season_df) < target_cluster_size:
            season_df["profile_cluster"] = 0
        else:
            matrix = season_df[available].apply(pd.to_numeric, errors="coerce")
            matrix = matrix.fillna(matrix.median(numeric_only=True)).fillna(0)
            scaled = StandardScaler().fit_transform(matrix)
            n_clusters = max(2, int(round(len(season_df) / target_cluster_size)))
            n_clusters = min(n_clusters, len(season_df))
            season_df["profile_cluster"] = KMeans(
                n_clusters=n_clusters,
                random_state=random_state,
                n_init=10,
            ).fit_predict(scaled)
        season_df["cluster_season"] = season
        output_cols = [
            "player_id",
            "player_name",
            "season",
            "position",
            "position_group",
            "profile_cluster",
            "cluster_season",
        ]
        cluster_rows.append(season_df[[column for column in output_cols if column in season_df.columns]])

    clusters = pd.concat(cluster_rows, ignore_index=True)
    cluster_sizes = clusters.groupby(["season", "profile_cluster"])["player_id"].transform("count")
    clusters["profile_cluster_size"] = cluster_sizes
    clusters["cluster_features"] = ", ".join(available)
    return clusters


def recommendation_cluster_agreement(
    recommendations: pd.DataFrame,
    clusters: pd.DataFrame,
    top_n: int = 5,
) -> dict[str, float]:
    # Measure how often top recommendations share the target's unsupervised profile cluster.
    """Return cluster-agreement diagnostics for a recommendation result table."""
    if recommendations.empty:
        return {
            "rows": 0.0,
            "top_n": float(top_n),
            "same_cluster_count": 0.0,
            "same_cluster_rate": float("nan"),
            "target_cluster_size": float("nan"),
        }

    recs = recommendations.head(top_n).copy()
    target_player_id = recs["target_player_id"].iloc[0]
    target_season = recs["target_season"].iloc[0]
    target_cluster_row = clusters[
        clusters["player_id"].eq(target_player_id)
        & clusters["season"].eq(target_season)
    ]
    if target_cluster_row.empty:
        return {
            "rows": float(len(recs)),
            "top_n": float(top_n),
            "same_cluster_count": 0.0,
            "same_cluster_rate": float("nan"),
            "target_cluster_size": float("nan"),
        }

    target_cluster = target_cluster_row["profile_cluster"].iloc[0]
    target_cluster_size = target_cluster_row["profile_cluster_size"].iloc[0]
    candidate_clusters = recs.merge(
        clusters[["player_id", "season", "profile_cluster"]],
        on=["player_id", "season"],
        how="left",
    )
    same_cluster = candidate_clusters["profile_cluster"].eq(target_cluster)
    return {
        "rows": float(len(recs)),
        "top_n": float(top_n),
        "same_cluster_count": float(same_cluster.sum()),
        "same_cluster_rate": float(same_cluster.mean()) if len(recs) else np.nan,
        "target_cluster_size": float(target_cluster_size),
    }


def _season_start_year(season: object) -> int | None:
    # Parse the start year from an NBA season label such as 2023-24.
    """Return the start year from a season label."""
    try:
        return int(str(season)[:4])
    except (TypeError, ValueError):
        return None


def build_future_similarity_ground_truth(
    base_df: pd.DataFrame,
    features: list[str] | tuple[str, ...],
    relevant_n: int = 5,
    same_position_group: bool = True,
    minutes_min: float | None = 500,
) -> pd.DataFrame:
    # Label relevant recommendations by next-season profile similarity.
    """Return proxy ground-truth top-N similar outcomes for each player-season."""
    if relevant_n < 1:
        raise ValueError("relevant_n must be at least 1.")

    available = [feature for feature in features if feature in base_df.columns]
    if not available:
        raise ValueError("No ground-truth features were available.")

    current = base_df.copy()
    current["season_start_year"] = current["season"].map(_season_start_year)
    future = current[
        ["player_id", "season_start_year", *available]
    ].rename(columns={feature: f"future_{feature}" for feature in available})
    future["season_start_year"] = future["season_start_year"] - 1

    labeled = current.merge(
        future,
        on=["player_id", "season_start_year"],
        how="inner",
    )
    if minutes_min is not None and "minutes" in labeled.columns:
        labeled = labeled[labeled["minutes"].fillna(0) >= minutes_min].copy()

    future_features = [f"future_{feature}" for feature in available]
    rows = []
    for (_, season), season_df in labeled.groupby(["season_start_year", "season"], sort=False):
        season_df = season_df.reset_index(drop=True)
        if len(season_df) < 2:
            continue
        matrix = season_df[future_features].apply(pd.to_numeric, errors="coerce")
        matrix = matrix.fillna(matrix.median(numeric_only=True)).fillna(0)
        scaled = StandardScaler().fit_transform(matrix)

        for target_idx, target in season_df.iterrows():
            candidates = season_df[season_df["player_id"].ne(target["player_id"])].copy()
            candidate_indices = candidates.index.to_numpy()
            if same_position_group and "position_group" in season_df.columns:
                candidates = candidates[candidates["position_group"].eq(target.get("position_group"))].copy()
                candidate_indices = candidates.index.to_numpy()
            if len(candidate_indices) == 0:
                continue

            distances = np.sqrt(((scaled[candidate_indices] - scaled[target_idx]) ** 2).mean(axis=1))
            candidate_rows = candidates.copy()
            candidate_rows["future_similarity_distance"] = distances
            candidate_rows["future_similarity_score"] = 1 / (1 + candidate_rows["future_similarity_distance"])
            candidate_rows = candidate_rows.sort_values("future_similarity_distance").head(relevant_n)
            for rank, (_, candidate) in enumerate(candidate_rows.iterrows(), start=1):
                rows.append(
                    {
                        "target_player_id": target["player_id"],
                        "target_player_name": target["player_name"],
                        "target_season": target["season"],
                        "player_id": candidate["player_id"],
                        "player_name": candidate["player_name"],
                        "season": candidate["season"],
                        "future_relevance_rank": rank,
                        "future_similarity_distance": candidate["future_similarity_distance"],
                        "future_similarity_score": candidate["future_similarity_score"],
                        "ground_truth_features": ", ".join(available),
                    }
                )

    return pd.DataFrame(rows)


def evaluate_recommendations_against_ground_truth(
    recommendations: pd.DataFrame,
    ground_truth: pd.DataFrame,
    top_n: int = 5,
) -> dict[str, float]:
    # Evaluate top-K recommendations against future-outcome proxy labels.
    """Return hit-rate and reciprocal-rank metrics for one recommendation result."""
    if recommendations.empty:
        return {
            "rows": 0.0,
            "top_n": float(top_n),
            "relevant_count": 0.0,
            "hit_count": 0.0,
            "hit_rate": float("nan"),
            "recall_at_k": float("nan"),
            "mrr": float("nan"),
        }

    recs = recommendations.head(top_n).copy()
    target_player_id = recs["target_player_id"].iloc[0]
    target_season = recs["target_season"].iloc[0]
    truth = ground_truth[
        ground_truth["target_player_id"].eq(target_player_id)
        & ground_truth["target_season"].eq(target_season)
    ].copy()
    if truth.empty:
        return {
            "rows": float(len(recs)),
            "top_n": float(top_n),
            "relevant_count": 0.0,
            "hit_count": 0.0,
            "hit_rate": float("nan"),
            "recall_at_k": float("nan"),
            "mrr": float("nan"),
        }

    truth_keys = set(zip(truth["player_id"], truth["season"]))
    rec_keys = list(zip(recs["player_id"], recs["season"]))
    hit_positions = [idx for idx, key in enumerate(rec_keys, start=1) if key in truth_keys]
    hit_count = len(hit_positions)
    return {
        "rows": float(len(recs)),
        "top_n": float(top_n),
        "relevant_count": float(len(truth_keys)),
        "hit_count": float(hit_count),
        "hit_rate": float(hit_count / len(recs)) if len(recs) else np.nan,
        "recall_at_k": float(hit_count / len(truth_keys)) if truth_keys else np.nan,
        "mrr": float(1 / min(hit_positions)) if hit_positions else 0.0,
    }
