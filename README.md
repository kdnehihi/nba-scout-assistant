# NBA Scout Assistant

NBA Scout Assistant is an end-to-end machine learning scouting project for NBA
player evaluation. It builds clean player datasets, trains forecasting models,
tracks experiments with MLflow, and prepares data products for player
similarity, performance forecasting, long-term trajectory forecasting, and
compensation context.

## Core Tasks

- Short-term performance forecasting: predict a player's next-five-game average
  points, assists, and rebounds.
- Long-term player trajectory forecasting: estimate future availability and
  per-36-minute production across H1, H2, and H3 seasons.
- Player recommendation and similarity: compare players using role, production,
  physical profile, and scouting signals.
- Player detail context: show current/historical salary and contract history for
  reference alongside forecasting outputs.

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
| Optional contract history | Manually staged contract-event reference data | `data/raw/contract_value/contract_events.csv` |
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
data/gold/player_salary_history_clean.parquet
data/gold/long_term_player_forecast_training.parquet
data/gold/long_term_player_forecast_inference.parquet
data/gold/season_coverage.parquet
```

Local data builders live under `src/dataset/`:

- `loaders.py`: path resolution and tabular loading
- `cleaning.py`: shared cleaning utilities
- `features_role.py`: player-season role features
- `features_performance.py`: short-term rolling features and next-five targets
- `features_compensation.py`: player salary history and salary-cap context
- `features_long_term.py`: season-anchor long-term feature engineering
- `long_term_modeling.py`: final long-term modeling data preparation and schema checks
- `season_coverage.py`: complete-season filtering for modeling data
- `pipeline.py`: orchestration for rebuilding gold datasets

## Modeling

Short-term forecasting uses a PyTorch LSTM over recent game sequences. Classical
time-series approaches such as ARIMA were explored conceptually but not selected
for the main short-term task because player-level NBA sequences are short,
irregular, and better treated as panel sequence data across many players.

Long-term forecasting uses selected model families from notebook experiments:

- Availability H1/H2/H3: Random Forest
- Per-36 production H1/H2: Random Forest
- Per-36 production H3: MLP

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

Per-100-possession production fields may appear in role profiles,
recommendation inputs, and research artifacts. They are not exposed as selected
long-term forecast targets in the API.

## API Surface

FastAPI entrypoint:

```text
app/main.py
```

The API is split into schemas and services:

```text
app/schemas/
app/services/
```

Current endpoints:

```text
GET  /health
GET  /metadata
POST /recommendations
POST /forecasts/short-term
POST /forecasts/long-term
POST /players/scouting-report
```

The scouting report endpoint composes player profile context, recommendation
data, compensation history, deterministic scouting signals, and optional
forecast outputs.

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
