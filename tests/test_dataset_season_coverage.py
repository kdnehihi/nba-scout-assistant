from __future__ import annotations

import pandas as pd

from dataset.season_coverage import (
    filter_to_modeling_seasons,
    latest_complete_modeling_season,
    modeling_seasons_through_latest_complete,
    summarize_game_log_season_coverage,
)


def make_coverage_logs() -> pd.DataFrame:
    rows = []
    game_id = 1
    for season, games in [("2022-23", 1230), ("2023-24", 1230), ("2024-25", 600)]:
        for idx in range(games):
            rows.append(
                {
                    "season": season,
                    "game_id": game_id,
                    "player_id": idx,
                    "team_id": f"T{idx % 30:02d}",
                    "game_date": f"{season[:4]}-10-01",
                }
            )
            game_id += 1
    return pd.DataFrame(rows)


def test_summarize_game_log_season_coverage_marks_partial_latest_season():
    coverage = summarize_game_log_season_coverage(make_coverage_logs())

    assert coverage.loc[coverage["season"].eq("2023-24"), "is_modeling_complete"].iloc[0]
    assert not coverage.loc[coverage["season"].eq("2024-25"), "is_modeling_complete"].iloc[0]


def test_modeling_seasons_stop_at_latest_complete_season():
    logs = make_coverage_logs()

    assert latest_complete_modeling_season(logs) == "2023-24"
    assert modeling_seasons_through_latest_complete(logs) == {"2022-23", "2023-24"}


def test_filter_to_modeling_seasons_drops_partial_future_rows():
    df = pd.DataFrame({"season": ["2022-23", "2023-24", "2024-25"], "value": [1, 2, 3]})

    filtered = filter_to_modeling_seasons(df, {"2022-23", "2023-24"})

    assert filtered["season"].tolist() == ["2022-23", "2023-24"]
