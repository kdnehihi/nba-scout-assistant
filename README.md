# NBA Scout Assistant

NBA Scout Assistant is an end-to-end machine learning scouting project for NBA
player evaluation. It builds clean player datasets, trains forecasting models,
tracks experiments with MLflow, and prepares data products for player
similarity, performance forecasting, long-term trajectory forecasting, and
salary valuation.

## Core Tasks

- Short-term performance forecasting: predict a player's next-five-game average
  points, assists, and rebounds.
- Long-term player trajectory forecasting: estimate future availability and
  per-minute/per-possession production across H1, H2, and H3 seasons.
- Salary valuation: estimate salary value through salary-cap-normalized targets.
- Player recommendation and similarity: compare players using role, production,
  physical profile, and salary-aware scouting signals.

## Data Sources

The project uses multiple sources because NBA scouting features are spread
across box scores, player bio tables, advanced-stat snapshots, salary history,
and salary-cap context.

| Data domain | Source | Canonical local file |
| --- | --- | --- |
| Historical game logs | KaggleHub dataset `szymonjwiak/nba-traditional` | `data/silver/player_game_logs.parquet` |
| Player bio/profile | Historical NBA `Players.csv` staged as `data/raw/player_bio/eoin_players.csv` | `data/raw/players.parquet` |
| Advanced/rate stats | Staged regular-season advanced, usage, and defense CSV snapshots | `data/raw/player_season_stats.parquet` |
| 2023-24 advanced patch | KaggleHub dataset `rodneycarroll78/nba-stats-1980-2024` | `data/raw/player_stats_advanced_patch/2023_24/Advanced.csv` |
| 2024-25 advanced patch | KaggleHub dataset `ratin21/nba-player-stats-2024-25-per-game` | `data/raw/player_stats_advanced_patch/2024_25/*.csv` |
| Historical salaries | Kaggle salary history staged into the silver salary table | `data/silver/player_season_salaries.parquet` |
| Salary cap history | Curated NBA salary-cap and tax-level history | `data/raw/salary_cap/salary_cap_by_season.csv` |
| NBA API fallback | `nba_api` / `stats.nba.com` | `data/raw/nba_api_cache/` |

Notebook workflows can fetch or refresh some external sources. Local source
code expects the canonical raw/silver/gold files to already exist under `data/`
or another directory passed through `--data-dir`.

More detail is documented in [docs/data_pipeline.md](docs/data_pipeline.md).

## Data Pipeline

The project follows a layered data layout:

```text
data/raw      source-level files
data/bronze   downloaded or staged source snapshots
data/silver   normalized intermediate tables
data/gold     final model-ready datasets and reports
```

Important gold outputs:

```text
data/gold/player_role_features_clean.parquet
data/gold/performance_training_clean.parquet
data/gold/salary_training_clean.parquet
data/gold/long_term_player_forecast_training.parquet
```

Local data builders live under `src/dataset/`:

- `loaders.py`: path resolution and tabular loading
- `cleaning.py`: shared cleaning utilities
- `features_role.py`: player-season role features
- `features_performance.py`: short-term rolling features and next-five targets
- `features_salary.py`: salary, cap, bio, and role joins
- `features_long_term.py`: season-anchor long-term forecasting data
- `long_term.py`: final long-term modeling data preparation and schema checks
- `pipeline.py`: orchestration for rebuilding gold datasets

## Modeling

Short-term forecasting uses a PyTorch LSTM over recent game sequences. Classical
time-series approaches such as ARIMA were explored conceptually but not selected
for the main short-term task because player-level NBA sequences are short,
irregular, and better treated as panel sequence data across many players.

Long-term forecasting uses selected model families from notebook experiments:

- H1 and H2: Random Forest
- H3: MLP

The selected long-term model family and hyperparameters are stored in:

```text
src/config/long_term_config.py
```

Training and evaluation code is split by responsibility:

```text
src/training/train_short_term.py
src/training/train_long_term.py
src/evaluation/evaluate_short_term.py
src/evaluation/evaluate_long_term.py
```

MLflow is used for experiment tracking with a SQLite backend and file artifact
store.

## Example Commands

Run tests:

```bash
pytest -q
```

Train one long-term model:

```bash
python -m src.training.train_long_term --task pts_per_36 --horizon 1
```

Train all selected long-term task/horizon models:

```bash
python -m src.training.train_long_term --task all
```

Train one short-term LSTM task:

```bash
python -m src.training.train_short_term --task points
```

## Project Direction

The codebase is structured so the notebook experiments can be promoted into a
production-style ML application. The intended deployment path includes a FastAPI
inference service, Dockerized runtime, persisted model artifacts, cloud object
storage for data/model assets, scheduled data refresh jobs, MLflow experiment
tracking, and monitoring for data drift, model drift, and API health.
