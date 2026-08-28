from __future__ import annotations

from datetime import date
import gzip
import json

import pandas as pd

from src.dataset.incremental import (
    load_balldontlie_checkpoint,
    run_balldontlie_incremental_pipeline,
    upsert_player_game_logs,
)
from src.dataset.loaders import resolve_data_paths
from tests.test_dataset_features import sample_game_logs, sample_players, sample_season_stats


class StubBallDontLieClient:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.calls: list[tuple[date, date, bool]] = []

    def fetch_player_game_stats(
        self,
        start_date: date,
        end_date: date,
        postseason: bool = False,
    ) -> list[dict[str, object]]:
        self.calls.append((start_date, end_date, postseason))
        return self.rows


def api_stat_row() -> dict[str, object]:
    return {
        "id": 9001,
        "min": "31:30",
        "fgm": 8,
        "fga": 16,
        "fg3m": 2,
        "fg3a": 6,
        "ftm": 4,
        "fta": 5,
        "oreb": 1,
        "dreb": 6,
        "reb": 7,
        "ast": 5,
        "stl": 2,
        "blk": 1,
        "turnover": 2,
        "pf": 2,
        "pts": 22,
        "player": {"id": 100, "first_name": "Player", "last_name": "One"},
        "team": {"id": 1, "abbreviation": "AAA"},
        "game": {
            "id": 8001,
            "date": "2025-10-20",
            "season": 2025,
            "status": "Final",
            "status_state": "final",
            "postseason": False,
            "home_team_id": 1,
            "visitor_team_id": 2,
            "home_team": {"id": 1, "abbreviation": "AAA"},
            "visitor_team": {"id": 2, "abbreviation": "BBB"},
        },
    }


def test_upsert_uses_cross_source_key_and_preserves_existing_game_id():
    existing = pd.DataFrame(
        {
            "player_id": [1],
            "game_date": [pd.Timestamp("2025-10-20")],
            "team_id": ["AAA"],
            "game_id": [12345],
            "points": [18],
            "rest_days": [2.0],
        }
    )
    incoming = pd.DataFrame(
        {
            "player_id": [1, pd.NA],
            "game_date": [pd.Timestamp("2025-10-20"), pd.Timestamp("2025-10-20")],
            "team_id": ["AAA", "BBB"],
            "game_id": [8001, 8001],
            "points": [22, 4],
            "rest_days": [pd.NA, pd.NA],
            "source_player_id": [100, 999],
        }
    )

    result, summary = upsert_player_game_logs(existing, incoming)

    assert len(result) == 1
    assert result.iloc[0]["game_id"] == 12345
    assert result.iloc[0]["points"] == 22
    assert summary.replaced_rows == 1
    assert summary.inserted_rows == 0
    assert summary.unmatched_rows == 1


def test_incremental_pipeline_writes_all_layers_and_checkpoint(tmp_path):
    paths = resolve_data_paths(tmp_path)
    paths.raw_dir.mkdir(parents=True)
    paths.silver_dir.mkdir(parents=True)
    sample_players().to_parquet(paths.raw_dir / "players.parquet", index=False)
    sample_season_stats().to_parquet(paths.raw_dir / "player_season_stats.parquet", index=False)
    sample_game_logs().to_parquet(paths.silver_dir / "player_game_logs.parquet", index=False)
    api_rows = []
    for game_number in range(14):
        row = api_stat_row()
        row["id"] = 9001 + game_number
        row["game"] = {
            **row["game"],
            "id": 8001 + game_number,
            "date": f"2025-10-{game_number + 1:02d}",
        }
        api_rows.append(row)
    client = StubBallDontLieClient(api_rows)

    result = run_balldontlie_incremental_pipeline(
        paths,
        end_date="2025-10-20",
        overlap_days=0,
        client=client,
    )

    assert result.bronze_snapshot_path is not None
    assert result.bronze_snapshot_path.exists()
    with gzip.open(result.bronze_snapshot_path, "rt", encoding="utf-8") as file:
        snapshot = json.load(file)
    assert snapshot["row_count"] == 14
    assert snapshot["data"][0]["id"] == 9001

    silver = pd.read_parquet(result.silver_game_logs_path)
    assert pd.to_datetime(silver["game_date"]).max() == pd.Timestamp("2025-10-14")
    assert result.upsert.inserted_rows == 14
    assert all(path.exists() for path in result.gold_output_paths.values())

    short_term = pd.read_parquet(result.gold_output_paths["short_term_inference_latest"])
    long_term = pd.read_parquet(
        result.gold_output_paths["long_term_player_forecast_inference_latest"]
    )
    assert short_term["game_id"].max() == 8014
    assert long_term["anchor_season"].eq("2025-26").any()
    assert load_balldontlie_checkpoint(paths) == date(2025, 10, 20)

    next_client = StubBallDontLieClient([])
    run_balldontlie_incremental_pipeline(
        paths,
        end_date="2025-10-21",
        overlap_days=2,
        client=next_client,
    )
    assert next_client.calls == [(date(2025, 10, 19), date(2025, 10, 21), False)]
