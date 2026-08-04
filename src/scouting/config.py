from __future__ import annotations

from dataclasses import dataclass, field


STAT_TARGETS = {
    "pts": "target_next_5_pts_avg",
    "ast": "target_next_5_ast_avg",
    "reb": "target_next_5_reb_avg",
}


@dataclass(frozen=True)
class TrendConfig:
    """Thresholds used to classify recent production trends."""

    points_tolerance: float = 1.5
    assists_tolerance: float = 0.6
    rebounds_tolerance: float = 0.8
    minutes_tolerance: float = 3.0
    overall_tolerance: float = 0.5


@dataclass(frozen=True)
class RangeConfig:
    """Weights and volatility settings for deterministic short-term ranges."""

    season_avg_weight: float = 0.50
    last_10_weight: float = 0.30
    last_5_weight: float = 0.20
    rolling_std_window: int = 10
    rolling_std_min_periods: int = 5
    volatility_multiplier: float = 0.80


@dataclass(frozen=True)
class SimilarityConfig:
    """Feature set and default filters for deterministic replacement ranking."""

    features: tuple[str, ...] = field(
        default_factory=lambda: (
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
        )
    )
    same_position_group: bool = True

