# Data Pipeline Documentation

This document describes the local data-processing flow under `src/dataset`.
It is intended as a reference for rebuilding, reviewing, and extending the
project data layer after notebook experiments are promoted into source code.

## High-Level Flow

The project uses layered data processing:

```text
raw / bronze / silver inputs
-> loaders
-> shared cleaning helpers
-> complete-season filtering
-> split assignment
-> task-specific feature builders
-> gold modeling datasets
```

Current online updates use a separate inference path so incomplete current
seasons never alter historical train/validation/test datasets:

```text
BALLDONTLIE /stats
-> immutable Bronze JSON snapshot + ingestion checkpoint
-> canonical normalization and player-ID review
-> cross-source upsert into Silver player_game_logs
-> label-free short-term and long-term Gold inference datasets
```

The local source-code flow is orchestrated by `src/dataset/pipeline.py`:

```text
load_players()
load_player_game_logs()
load_player_season_stats()
load_player_season_salaries()
load_salary_cap()
        |
        v
build_role_features()
build_performance_training()
build_player_salary_history()
build_long_term_training()
build_long_term_inference()
summarize_game_log_season_coverage()
        |
        v
data/gold/*.parquet
```

The key gold outputs are:

```text
data/gold/player_role_features_clean.parquet
data/gold/performance_training_clean.parquet
data/gold/player_salary_history_clean.parquet
data/gold/long_term_player_forecast_training.parquet
data/gold/long_term_player_forecast_inference.parquet
data/gold/season_coverage.parquet
data/gold/short_term_inference_latest.parquet
data/gold/long_term_player_forecast_inference_latest.parquet
```

The data pipeline intentionally keeps split labels inside the output
dataframes. A split file assigns labels such as `train`, `validation`, `test`,
or `ignore`; actual `X_train`, `X_validation`, and `X_test` slicing should
happen in the model-training layer.

## Data Layers

The project uses these data-layer conventions:

- `raw`: source-level files with minimal transformation.
- `bronze`: downloaded or staged source files, when applicable.
- `silver`: normalized intermediate tables with stable schema.
- `gold`: final model-ready datasets and derived scouting artifacts.

Local source code currently expects several canonical files:

```text
data/raw/players.parquet
data/raw/player_season_stats.parquet
data/raw/salary_cap/salary_cap_by_season.csv
data/silver/player_game_logs.parquet
data/silver/player_season_salaries.parquet
```

Notebook workflows may materialize additional gold artifacts, but the local
pipeline source focuses on the core clean modeling tables.

## Source Data Inventory

The ingestion notebook `notebooks/00_data_ingestion.ipynb` uses several
external and curated sources because no single public dataset contains all
fields required for player performance, role similarity, long-term forecasting,
recommendations, and compensation context.

The current source inventory is:

| Data domain | Primary source | Local path | Main fields used | Downstream use |
| --- | --- | --- | --- | --- |
| Historical player game logs | KaggleHub dataset `szymonjwiak/nba-traditional` | `data/raw/nba_traditional/` -> `data/silver/player_game_logs.parquet` | `player_id`, `player_name`, `game_id`, `game_date`, `season`, `team_id`, `minutes`, `points`, `assists`, `rebounds`, rebounds by type, steals, blocks, fouls, turnovers, FGA, FTA, 3PA, team/opponent context | Short-term forecasting, long-term season summaries, defensive proxies, rolling form |
| Player bio/profile | `Players.csv` staged as `eoin_players.csv` from the historical NBA data/player box-score source | `data/raw/player_bio/eoin_players.csv` -> `data/raw/players.parquet` | `player_id`, `player_name`, `birth_date`, `position`, `height`, `weight` | Age at season/anchor, role filters, similarity, compensation joins, long-term features |
| Base advanced/rate season stats | Staged advanced, usage, and defense regular-season CSV snapshots | `data/raw/player_stats_advanced/player_stats_advanced_rs.csv`, `player_stats_usage_rs.csv`, `player_stats_defense_rs.csv` -> `data/raw/player_season_stats.parquet` | usage, true shooting, three-point attempt rate, free-throw rate, turnover rate, steal/block/defensive rebound/foul rates, pace, possessions, offensive/defensive rating | Role features, recommendation features, long-term normalized production context |
| 2023-24 advanced stat patch | KaggleHub dataset `rodneycarroll78/nba-stats-1980-2024` | `data/raw/player_stats_advanced_patch/2023_24/Advanced.csv` | Basketball-Reference-style advanced/rate stats for recent-season coverage | Fills sparse advanced coverage for 2023-24 |
| 2024-25 advanced stat patch | KaggleHub dataset `ratin21/nba-player-stats-2024-25-per-game` | `data/raw/player_stats_advanced_patch/2024_25/NBA Player Advanced Stats_2024-25.csv`, `NBA Player Stats_2024-25_Total.csv` | 2024-25 advanced rates plus totals/minutes/team context | Fills sparse advanced coverage for 2024-25 |
| Historical player salaries | Kaggle salary dataset staged into silver salary table | `data/silver/player_season_salaries.parquet` | `player_name`, `team`, `season_start_year`, `season_end_year`, `season_label`, `salary_usd`, `source`, `source_file`, `collected_at` | Current and historical salary context on player detail pages |
| Salary cap and tax level | Curated NBA salary-cap history, backed by official NBA cap releases where available | `data/raw/salary_cap/salary_cap_by_season.csv` | `season`, `salary_cap_usd`, `tax_level_usd` | `salary_cap_share` for cross-era salary context |
| Optional contract history | Manually staged contract-event reference data | `data/raw/contract_value/contract_events.csv` | player name, contract start/end, average salary, pre-contract box-score context | Contract history display only |
| NBA API fallback | `nba_api` / `stats.nba.com` endpoints | `data/raw/nba_api_cache/`, `data/raw/player_game_logs_nba_api.parquet` | fallback player metadata, game logs, season stats | Fallback only; not the primary historical source because live requests can timeout/throttle |
| Incremental player game logs | BALLDONTLIE Game Player Stats API | `data/bronze/balldontlie/` -> `data/silver/player_game_logs.parquet` | current regular-season box scores, game/team context, source IDs and fetch audit fields | Nightly short-term and long-term inference refresh |

### Source Selection Notes

- Game logs prefer KaggleHub snapshots over live `stats.nba.com` requests. The
  notebook keeps `nba_api` as a fallback because live NBA Stats requests can
  timeout or be throttled.
- Player bio is kept as a separate raw source so age is calculated relative to
  the modeling date instead of stored as one fixed value.
- Advanced/rate stats are assembled from a base regular-season snapshot plus
  recent-season patches. This avoids losing important role features such as
  `usage_pct`, `true_shooting_pct`, `stl_pct`/steal rate, `blk_pct`/block rate,
  pace, and rating fields in 2023-24 and 2024-25.
- Salary history uses `salary_cap_share = salary_usd / salary_cap_usd` only as
  context. The local pipeline does not train a salary model.
- The canonical local training pipeline expects staged canonical files. It does
  not download from Kaggle or call `nba_api` directly; source acquisition is
  handled by notebooks or manual data refresh steps before local training.

## Incremental Online Update

`src/dataset/balldontlie.py` owns HTTP access and source normalization.
`src/dataset/incremental.py` owns persistence and layer transitions. The
application wrapper is `run_online_data_update_pipeline()` in
`src/pipelines/data.py`.

Run one update from the repository root:

```bash
python -m scripts.update_online_data --data-dir data
```

The command reads `BALLDONTLIE_API_KEY` from the environment or local `.env`.
It performs these steps in order:

1. Read `data/silver/player_game_logs.parquet` as canonical historical state.
2. Use `data/bronze/balldontlie/ingestion_state.json` as the API watermark
   after the first successful run. On the first run, use the latest Silver
   game date instead.
3. Refetch two prior dates by default so delayed source corrections can be
   incorporated without downloading the full history.
4. Save the unmodified API rows under partitioned gzip JSON files in
   `data/bronze/balldontlie/player_game_stats/`.
5. Normalize source fields to the canonical game-log schema. BALLDONTLIE IDs
   remain in `source_*` audit columns and are never silently substituted for
   canonical NBA player IDs.
6. Store players that cannot be mapped in
   `data/bronze/balldontlie/unmatched_players_latest.parquet`. These rows are
   excluded from Silver until a reviewed mapping or explicit override exists.
7. Upsert mapped rows using `player_id + game_date + team_id`. This key is used
   instead of `game_id` because game identifiers can differ across sources.
   Existing canonical game IDs are retained for overlapping rows while updated
   box-score values replace stale values.
8. Atomically replace `data/silver/player_game_logs.parquet` and recompute
   player rest days over the full timeline.
9. Rebuild `short_term_inference_latest.parquet` without requiring next-five
   labels and rebuild `long_term_player_forecast_inference_latest.parquet`
   without future-season labels.
10. Write the checkpoint only after Bronze, Silver, and Gold operations finish
    successfully. A failed job therefore retries the same date range safely.

The historical Gold training tables are deliberately not rebuilt by this
command. Their temporal split policy remains stable until a reviewed retraining
cycle explicitly promotes newer complete seasons.

## Source-To-Gold Lineage

The source-to-output flow is:

```text
Kaggle nba-traditional
  -> data/silver/player_game_logs.parquet
  -> performance_training_clean.parquet
  -> long_term_player_forecast_training.parquet
  -> long_term_player_forecast_inference.parquet
  -> season_coverage.parquet

Player bio/profile CSV
  -> data/raw/players.parquet
  -> player_role_features_clean.parquet
  -> player_salary_history_clean.parquet
  -> long_term_player_forecast_training.parquet
  -> long_term_player_forecast_inference.parquet

Advanced / usage / defense season stats + recent-season patches
  -> data/raw/player_season_stats.parquet
  -> player_role_features_clean.parquet
  -> long_term_player_forecast_training.parquet
  -> long_term_player_forecast_inference.parquet

Historical salaries + salary cap table
  -> data/silver/player_season_salaries.parquet
  -> player_salary_history_clean.parquet
```

## File-by-File Reference

## `src/dataset/loaders.py`

### Main Responsibility

`loaders.py` owns data path resolution, file validation, tabular file loading,
and basic schema checks. It is the entrypoint for reading data from disk.

### Important Objects And Functions

`DataPaths`

- Frozen dataclass containing the configured project data root.
- Provides standard layer paths:
  - `raw_dir`
  - `bronze_dir`
  - `silver_dir`
  - `gold_dir`

`resolve_data_paths(data_dir="data")`

- Resolves a user-provided data root into an absolute path.
- Returns a `DataPaths` object.

`require_file(path)` and `require_dir(path)`

- Fail early with clear errors when required files or directories are missing.
- These functions prevent silent path fallback issues.

`list_tabular_files(directory, extensions=TABULAR_EXTENSIONS)`

- Recursively lists supported tabular files.
- Supported extensions include CSV, TSV, Parquet, JSON, JSONL, and XLSX.

`load_tabular_data(file_path, **kwargs)`

- Dispatches to the correct pandas reader based on file suffix.
- This centralizes tabular loading instead of scattering `pd.read_*` calls.

`require_columns(df, required_columns, dataset_name)`

- Validates minimum schema contracts after loading.
- Raises a `KeyError` when required columns are missing.

Dataset-specific loaders:

- `load_players(paths)`
- `load_player_game_logs(paths)`
- `load_player_season_stats(paths)`
- `load_player_season_salaries(paths)`
- `load_salary_cap(paths)`
- `load_role_features_clean(paths)`
- `load_performance_training_clean(paths)`
- `load_player_salary_history_clean(paths)`
- `load_contract_history(paths, required=False)`
- `load_long_term_training(paths)`
- `load_long_term_inference(paths)`

### Notable Details

- `load_player_game_logs` reads from the silver layer because game logs are
  expected to be canonicalized before feature engineering.
- `load_player_season_salaries` reads from the silver layer because salaries
  are expected to be normalized from source files first.
- Gold loaders are read-only convenience functions for downstream modeling.

## `src/dataset/cleaning.py`

### Main Responsibility

`cleaning.py` contains shared low-level transformations used by multiple
feature builders. It should contain reusable primitive operations, not
task-specific feature engineering.

### Important Constants

`TEAM_ABBREVIATION_MAP`

- Maps historical or source-specific NBA abbreviations to canonical team IDs.
- Examples:
  - `BRK -> BKN`
  - `CHO -> CHA`
  - `PHO -> PHX`

### Important Functions

`to_snake_case(column)`

- Normalizes source column names.
- Example: `Player Name! -> player_name`.

`normalize_name_key(value)`

- Creates a stable player-name key for fallback joins.
- Example: `LeBron James Jr. -> lebronjamesjr`.

`normalize_team_abbreviation(value)`

- Uppercases and canonicalizes team abbreviations.
- Example: `pho -> PHX`.

`percent_to_ratio(values)`

- Converts mixed percentage/ratio values to ratio scale.
- Example: `[55.0, 0.55] -> [0.55, 0.55]`.

`parse_salary(value)`

- Converts currency-like salary strings to numeric USD.
- Example: `$12,345,678 -> 12345678.0`.

`normalize_season(value)`

- Parses season text into start year, end year, and canonical label.
- Example: `2023-24 -> (2023, 2024, "2023-24")`.

`season_label_to_start_year(value)`

- Extracts start year for temporal comparisons.
- Example: `2024-25 -> 2024`.

`safe_divide(numerator, denominator)`

- Performs numeric division and returns `NaN` when the denominator is zero.
- This is used to avoid invalid rate calculations.

`safe_per_36(numerator, minutes)`

- Converts counting stats into per-36-minute rates.
- Example: `10 points in 20 minutes -> 18 points per 36`.

`add_missing_flags(df, columns)`

- Adds boolean missingness indicators before imputation.
- Example: `age = NaN -> age_was_missing = True`.

`fill_numeric_median(df, columns)`

- Fills numeric missing values with column medians.
- Falls back to zero when an entire column is missing.

`fill_categorical_unknown(df, columns, value="UNK")`

- Fills categorical missing values with a stable unknown token.

### Notable Details

- Missing flags are created before imputation so downstream models can still
  learn whether a value was originally missing.
- `safe_per_36` can be useful for normalized production, but per-36 rates can
  exaggerate very-low-minute samples. Filtering or minutes context should be
  considered when using these features.

## `src/dataset/splits.py`

### Main Responsibility

`splits.py` defines temporal split policy and assigns split labels. It does not
perform actual model-array splitting.

### Current Split Policy

Short-term performance:

```text
train: seasons with start year <= 2022-23
validation: 2023-24
test: 2024-25
ignore: all other seasons
```

Salary:

```text
train: 2016-17 through 2022-23
validation: 2023-24
test: 2024-25
ignore: all other seasons
```

Long-term forecasting:

```text
train: seasons with start year <= 2019-20
validation: 2020-21
test: 2021-22
ignore: all other seasons
```

The long-term split is shifted earlier so that future H1/H2/H3 labels can be
complete without using seasons beyond the available data.

### Important Functions

`assign_temporal_split(season)`

- Assigns short-term performance split labels.
- Uses start-year comparison for the train period.

`assign_long_term_temporal_split(season)`

- Assigns long-term anchor split labels.
- Uses an earlier cutoff to support complete future horizons.

### Notable Details

The typical usage is:

```python
df["split"] = df["season"].map(assign_temporal_split)
df = df[df["split"].isin(["train", "validation", "test"])].copy()
```

Training code should later perform:

```python
train_df = df[df["split"].eq("train")]
validation_df = df[df["split"].eq("validation")]
test_df = df[df["split"].eq("test")]
```

## `src/dataset/season_coverage.py`

### Main Responsibility

`season_coverage.py` identifies which game-log seasons are complete enough for
modeling and filters out later sparse seasons before feature generation.

### Important Functions

`summarize_game_log_season_coverage(game_logs)`

- Returns one row per season with row count, unique games, player count, team
  count, date range, and `is_modeling_complete`.

`latest_complete_modeling_season(game_logs)`

- Returns the latest season whose coverage passes the configured minimum unique
  games and teams checks.

`modeling_seasons_through_latest_complete(game_logs)`

- Returns all seasons up to the latest complete modeling season.

`filter_to_modeling_seasons(df, seasons, season_col="season")`

- Filters any season-based dataframe to the approved modeling seasons.

### Notable Details

- This prevents partially available recent seasons from entering training data.
- The coverage summary is saved as `data/gold/season_coverage.parquet` for
  auditing.

## `src/dataset/features_role.py`

### Main Responsibility

`features_role.py` builds player-season role features by combining season-level
stats with player metadata. The output is used by similarity search, long-term
modeling, recommendations, and scouting signals.

### Important Constants

`ROLE_NUMERIC_FEATURES`

- Numeric columns expected in role feature construction.
- Includes bio, workload, usage, per-100-possession production, shooting,
  defense proxies, pace, possessions, and ratings.

### Important Functions

`add_role_dimensions(df)`

- Creates interpretable composite role dimensions:
  - `scoring_creation`
  - `playmaking`
  - `shooting`
  - `rim_pressure`
  - `rebounding`
  - `perimeter_defense`
  - `interior_defense`
  - `two_way_impact`

These are heuristic scouting dimensions. The current weights are not official
NBA formulas. They are designed to convert raw rate stats into more readable
role signals for MVP modeling and similarity.

`two_way_impact` is computed as:

```python
offensive_rating - defensive_rating
```

This is subtraction, not addition, because higher offensive rating is better
while lower defensive rating is better.

`build_role_features(players, season_stats, game_logs=None)`

- Uses `game_logs` when available to compute `points_per_100`,
  `assists_per_100`, and `rebounds_per_100` as per-100-possession rates.
- Falls back to `season_stats` production columns when game logs are not
  supplied.
- Converts percentage columns to ratio scale.
- Merges player bio data onto season stats by `player_id`.
- Resolves duplicated columns such as `position` and `position_player`.
- Adds missing flags.
- Fills numeric missing values with medians.
- Fills missing categorical values with `UNK`.
- Adds composite role dimensions.
- Drops duplicate `player_id`, `season`, `team_id` rows.

### Output Grain

One row per:

```text
player_id + season + team_id
```

Including `team_id` matters because a player may change teams within a season.

### Notable Details

- The composite role-dimension weights are heuristic. A future revision should
  consider season-standardized z-score inputs before weighted sums.
- `points_per_100`, `assists_per_100`, and `rebounds_per_100` mean
  per-100-possession production, not per-100-minute production.
- Role features are a core join table for other tasks, so schema stability is
  important.

## `src/dataset/features_performance.py`

### Main Responsibility

`features_performance.py` builds the short-term forecasting dataset. It creates
point-in-time rolling features and next-five-game average targets for points,
assists, and rebounds.

### Important Functions

`_column(df, canonical, fallback)`

- Resolves canonical or fallback stat names.
- Example: use `points` if present, otherwise `pts`.

`future_average(values, horizon=5)`

- Creates future average targets.
- Row `t` uses `t+1` through `t+horizon`.
- The current row is not included in the target.

Example with `horizon=3`:

```text
values: [10, 20, 15, 25, 30]
row 0 target: mean(20, 15, 25) = 20.0
row 1 target: mean(15, 25, 30) = 23.33
```

`add_rolling_player_features(game_logs, min_history=5)`

- Sorts game logs by `player_id`, `season`, `as_of_date`, and `game_id`.
- Resolves points, assists, rebounds, and minutes column names.
- Converts stat columns to numeric.
- Groups by `player_id` and `season`.
- Creates:
  - last-5 averages
  - last-10 averages
  - season-to-date averages
  - recent-minus-season deltas
  - last-5-minus-last-10 deltas
  - next-five-game average targets

The rolling features include the current game row. This is correct when the
prediction moment is defined as after game `t`, predicting games `t+1` through
`t+5`. If the product later predicts before game `t`, these rolling features
must be shifted by one row.

`build_performance_training(game_logs)`

- Calls `add_rolling_player_features`.
- Assigns split labels with `assign_temporal_split`.
- Drops rows missing required historical features or future targets.
- Filters out `ignore` split rows.

### Output Grain

One row per player-game prediction anchor.

Each row means:

```text
After this player's game t, use history through t to predict average
PTS/AST/REB over games t+1 through t+5.
```

### Notable Details

- Early-season rows are dropped if they do not have enough history.
- End-of-season rows are dropped if they do not have enough future games to
  create next-five-game targets.
- The current implementation builds complete supervised rows for all three
  targets together.

## `src/dataset/features_compensation.py`

### Main Responsibility

`features_compensation.py` builds a clean player salary history table by merging
player-season salaries, salary cap context, and player IDs.

### Important Constants

There are no salary modeling feature constants in the local pipeline. Salary
history is retained as reference data.

### Important Functions

`build_player_salary_history(salaries, salary_cap, players)`

- Parses `salary_usd` when it is stored as text.
- Normalizes team abbreviations into `team_id`.
- Creates `player_name_key` for fallback player joins.
- Merges salary cap by `season_label`.
- Computes:

```python
salary_cap_share = salary_usd / salary_cap_usd
```

- Merges `player_id` from player metadata by normalized player name key when
  salary rows do not already contain player IDs.
- Drops helper columns from the final output.

### Output Grain

One row per player-season salary record.

### Notable Details

- `salary_cap_share` is retained only for reporting and cross-season context.
- Name-based player joins are fragile; player IDs should be preferred whenever
  reliable source IDs are available.

## `src/dataset/features_long_term.py`

### Main Responsibility

`features_long_term.py` builds season-anchor data for long-term player
forecasting. It summarizes game logs into player-season rows, creates lagged
history features, and attaches future horizon targets.

### Important Constants

`NBA_REGULAR_SEASON_GAMES = 82`

- Used to compute availability rate.

`LONG_TERM_HORIZONS = [1, 2, 3]`

- Future horizons for H1, H2, and H3 labels.

`LONG_TERM_LAGS = 4`

- Number of lag rows created from prior/current seasons.

### Important Functions

`build_player_season_summary(game_logs, players, season_stats)`

- Aggregates game logs to player-season summaries.
- Computes:
  - games played
  - total minutes
  - minutes per game
  - total points, assists, rebounds
  - availability rate
  - points/assists/rebounds per 36 minutes
  - points/assists/rebounds per 100 estimated possessions
- Merges selected advanced season stats when available.
- Merges player metadata.
- Computes `age_at_anchor`.
- Computes career-level cumulative features:
  - `career_games`
  - `career_minutes`
  - `years_in_league`

`build_recent_game_anchor_features(game_logs)`

- Summarizes each player's most recent games within an anchor season.
- Creates recent workload, efficiency, rest, and trend features.

`add_lagged_season_features(summary)`

- Adds lagged trajectory features for each player.
- `lag_0` is the current anchor season value.
- `lag_1` is the previous season.
- `lag_2` is two seasons back.
- `lag_3` is three seasons back.

Examples:

```text
pts_per_36_lag_0
pts_per_36_lag_1
minutes_per_game_lag_2
availability_rate_lag_3
```

`add_future_horizon_targets(anchor_features, season_summary)`

- Adds H1/H2/H3 future targets by shifting each player's future seasons
  backward onto the current anchor row.
- The gold table retains several future labels for evaluation and research:
  - `games_played_h1/h2/h3`
  - `pts_per_36_h1/h2/h3`
  - `ast_per_36_h1/h2/h3`
  - `reb_per_36_h1/h2/h3`
  - `pts_per_100_h1/h2/h3`
  - `ast_per_100_h1/h2/h3`
  - `reb_per_100_h1/h2/h3`
  - `pts_per_game_h1/h2/h3`
  - `ast_per_game_h1/h2/h3`
  - `reb_per_game_h1/h2/h3`
  - `active_h1/h2/h3`
  - `low_availability_h1/h2/h3`
  - `high_availability_h1/h2/h3`

The selected product forecast tasks are narrower:

```text
active_probability
pts_per_36
ast_per_36
reb_per_36
```

Per-100 fields here mean per-100-possession rates. They are retained for
research and feature context, not as selected API forecast targets.

`build_long_term_training(game_logs, players, season_stats)`

- Builds player-season summaries.
- Adds recent game anchor features.
- Adds lagged season features.
- Adds future targets.
- Drops rows without required active horizon labels.
- Creates anchor metadata:
  - `anchor_season`
  - `anchor_season_start_year`
  - `anchor_date`
- Assigns long-term split labels.
- Filters out `ignore` rows.

`build_long_term_inference(game_logs, players, season_stats)`

- Builds the same anchor-time feature rows without requiring future labels.
- Supports API inference for recent complete seasons that are not part of the
  training split.

### Output Grain

One row per player-season anchor.

Each row means:

```text
At the end of anchor season t, use season/career history through t to forecast
future availability and production in t+1, t+2, and t+3.
```

### Notable Details

- Long-term split years are earlier than short-term split years to ensure H3
  future labels are complete.
- Per-36 rates normalize production for playing time but should be interpreted
  together with minutes and availability features.
- Per-100 rates normalize production by estimated possessions and are kept as
  context/research columns.
- Availability and minutes remain difficult targets because they are heavily
  affected by injuries, team role, coaching, trades, and career exits.

## `src/dataset/long_term_modeling.py`

### Main Responsibility

`long_term_modeling.py` prepares the gold long-term dataframe for model
training. It does not create raw features; it validates schema, removes leaking
future-target columns from the feature list, and returns task-specific modeling
rows.

### Important Functions

`is_long_term_model_feature(column)`

- Returns `False` for metadata and future-horizon target columns.
- Prevents labels such as `pts_per_36_h1`, `active_h2`, or
  `pts_per_100_h3` from being used as anchor-time input features.

`infer_long_term_feature_columns(df)`

- Infers the allowed feature columns from a gold long-term dataframe.

`validate_long_term_columns(df, task_config, feature_cols)`

- Checks that required metadata, selected features, and the requested target
  column exist.

`validate_long_term_split(df)`

- Confirms split labels are valid and match the configured temporal split map.

`prepare_long_term_modeling_data(df, task_config, feature_cols=None)`

- Returns the filtered modeling dataframe and selected feature list for one
  long-term task/horizon.

### Notable Details

- Actual `train`, `validation`, and `test` slicing happens later in the training
  layer.
- Per-100-possession future labels may exist in the gold table, but this file
  still excludes them from model features to avoid leakage.

## `src/dataset/pipeline.py`

### Main Responsibility

`pipeline.py` orchestrates the dataset build. It is the file to call when the
gold layer should be rebuilt from canonical source inputs.

### Important Function

`build_all_gold_datasets(paths)`

- Loads source datasets through `loaders.py`.
- Builds role features.
- Builds short-term performance training data.
- Builds clean salary history for player detail context.
- Builds long-term forecasting data.
- Builds long-term inference data.
- Saves season coverage for data completeness audit.
- Saves all outputs to `paths.gold_dir`.
- Returns a dictionary of output dataframes.

### Output Files

```text
player_role_features_clean.parquet
performance_training_clean.parquet
player_salary_history_clean.parquet
long_term_player_forecast_training.parquet
long_term_player_forecast_inference.parquet
season_coverage.parquet
```

### Notable Details

- The pipeline currently writes full parquet outputs every time it runs.
- It does not run DVC commands, MLflow logging, or incremental cache checks.
- Season coverage is saved as a data audit artifact, but detailed validation
  reports should remain in the data-evaluation layer.
- It should remain thin: detailed transformation logic belongs in the feature
  builder modules.

## Recommended Reading Order

For code review or onboarding, read the files in this order:

```text
1. src/dataset/loaders.py
2. src/dataset/cleaning.py
3. src/dataset/splits.py
4. src/dataset/features_role.py
5. src/dataset/features_performance.py
6. src/dataset/features_compensation.py
7. src/dataset/season_coverage.py
8. src/dataset/features_long_term.py
9. src/dataset/long_term_modeling.py
10. src/dataset/pipeline.py
```

This order moves from data access, to shared transformations, to split policy,
to task-specific datasets, and finally to orchestration.

## Maintenance Notes

- Keep split assignment centralized in `splits.py`.
- Keep primitive cleaning helpers in `cleaning.py`.
- Keep task-specific feature engineering inside the relevant `features_*.py`
  file.
- Keep `pipeline.py` as an orchestration layer, not a transformation layer.
- Prefer explicit schema validation before feature construction.
- Avoid random splits for forecasting tasks; use temporal splits.
- When a feature uses current-row game data, document whether prediction occurs
  before or after that game.
- If notebooks introduce new gold artifacts, promote only stable logic into
  `src/dataset` after the notebook output has been reviewed.

## Notebook Decisions To Promote Later

The following notebook-level decisions have been reviewed and should be kept in
mind when promoting experimental logic into local source code.

Keep the existing modeling decisions:

- Keep per-36 production features for long-term production modeling.
- Keep per-100-possession fields as comparison artifacts or profile context, but
  do not expose them as production forecast targets unless evaluation clearly
  improves and the API contract is updated.
- Keep short-term next-five-game forecasting targets for PTS, AST, and REB.
- Keep salary and contract history as reference context, not as a forecasting
  target.

Notebook `notebooks/06_scouting_signals.ipynb` evaluates these scout-facing
signals with low model risk:

- `player_trend_signals`: recent-vs-season trend labels for production and
  minutes.
- `player_consistency_signals`: player-season consistency, volatility, and
  floor/ceiling descriptors from observed game logs.
- `short_term_floor_ceiling_signals`: deterministic expected/floor/ceiling
  next-five-game ranges.
- `short_term_floor_ceiling_evaluation`: validation metrics for the deterministic
  range estimates.
- replacement candidate ranking: role/profile similarity with optional salary,
  age, and position filters.

When these signals are promoted, keep them deterministic unless a supervised
model clearly beats the heuristic on temporal validation and remains stable on
test.
