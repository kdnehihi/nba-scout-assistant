from __future__ import annotations

import pandas as pd

from dataset.loaders import resolve_data_paths
from dataset.pipeline import build_all_gold_datasets
from tests.test_dataset_features import sample_game_logs, sample_players, sample_season_stats


def test_build_all_gold_datasets_writes_outputs(tmp_path):
    paths = resolve_data_paths(tmp_path)
    paths.raw_dir.mkdir(parents=True)
    paths.silver_dir.mkdir(parents=True)

    sample_players().to_parquet(paths.raw_dir / "players.parquet", index=False)
    sample_game_logs().to_parquet(paths.silver_dir / "player_game_logs.parquet", index=False)
    sample_season_stats().to_parquet(paths.raw_dir / "player_season_stats.parquet", index=False)
    pd.DataFrame(
        {
            "player_name": ["Player One"],
            "team": ["AAA"],
            "season_start_year": [2022],
            "season_end_year": [2023],
            "season_label": ["2022-23"],
            "salary_usd": [10_000_000],
            "source": ["test"],
            "source_file": ["test.csv"],
            "collected_at": ["2026-01-01"],
        }
    ).to_parquet(paths.silver_dir / "player_season_salaries.parquet", index=False)
    (paths.raw_dir / "salary_cap").mkdir()
    pd.DataFrame({"season": ["2022-23"], "salary_cap_usd": [100_000_000], "tax_level_usd": [120_000_000]}).to_csv(
        paths.raw_dir / "salary_cap" / "salary_cap_by_season.csv",
        index=False,
    )

    outputs = build_all_gold_datasets(paths)

    assert "performance_training_clean" in outputs
    assert (paths.gold_dir / "player_salary_history_clean.parquet").exists()
