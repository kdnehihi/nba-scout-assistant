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
