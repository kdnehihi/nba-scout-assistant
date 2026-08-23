# NBA Scout Assistant

[![CI](https://github.com/kdnehihi/nba-scout-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/kdnehihi/nba-scout-assistant/actions/workflows/ci.yml)
[![CD](https://github.com/kdnehihi/nba-scout-assistant/actions/workflows/cd.yml/badge.svg)](https://github.com/kdnehihi/nba-scout-assistant/actions/workflows/cd.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Docker Hub](https://img.shields.io/badge/Docker_Hub-khoatran1%2Fnba--scout--assistant-2496ED?logo=docker&logoColor=white)](https://hub.docker.com/r/khoatran1/nba-scout-assistant)

NBA Scout Assistant is an end-to-end machine learning product for NBA player
discovery and evaluation. A user searches for a target player, receives ranked
similar-player recommendations, and opens a candidate report containing recent
form, short-term forecasts, long-term trajectory estimates, physical and role
context, and salary history.

The repository covers the full delivery path: multi-source data processing,
feature engineering, reproducible model training and evaluation, MLflow
tracking, deterministic recommendation diagnostics, a FastAPI inference layer,
a responsive browser interface, Docker packaging, GitHub Actions CI/CD, and an
AWS ECS Fargate deployment.

## Demo

[![Watch the NBA Scout Assistant product demo](https://img.youtube.com/vi/_Y7J8QKMIbc/maxresdefault.jpg)](https://youtu.be/_Y7J8QKMIbc)

[Watch the full NBA Scout Assistant product demo on YouTube.](https://youtu.be/_Y7J8QKMIbc)

## Project Status

| Area | Status | Current implementation |
| --- | --- | --- |
| Data pipeline | Complete | Raw/bronze/silver/gold processing with validation and temporal splits |
| Player recommendation | Complete | Six configurable ranking presets, position/minutes filters, explanations, and proxy evaluation |
| Short-term forecasting | Complete | Three deployed PyTorch LSTMs for next-five-game PTS, AST, and REB averages |
| Long-term forecasting | Complete | Availability and per-36 production forecasts for H1, H2, and H3 |
| Player report | Complete | Profile, recent performance, forecasts, salary history, and optional contract context |
| API and frontend | Complete | FastAPI endpoints and a responsive static dashboard served by the same container |
| Experiment tracking | Complete | MLflow parameters, metrics, artifacts, and reusable checkpoints |
| Testing and delivery | Complete | 69 automated tests, GitHub Actions CI, Docker Hub publishing, and ECS deployment |
| Cloud runtime | Live demo | Single-container AWS ECS Fargate service in `us-east-2` |
| Automated data refresh | Not implemented | Serving data is an immutable snapshot through the 2024-25 season |

The deployed serving snapshot currently contains:

```text
4,910 player-season recommendation profiles
186,937 short-term performance rows
4,910 long-term inference anchors
12,386 salary-history rows
3 short-term model artifacts
12 long-term task/horizon artifacts
Seasons 2016-17 through 2024-25
```

## Product Workflow

```text
Player search
    -> candidate filtering by season, position group, and minimum minutes
    -> standardized statistical-profile similarity
    -> preset-based Top-K ranking with explanations
    -> candidate scouting report
         -> recent performance
         -> next-five-game forecasts
         -> H1/H2/H3 long-term forecasts
         -> salary and optional contract history
```

The current UI automatically opens the highest-ranked candidate and retrieves
the complete report without reloading the page.

## Core Capabilities

### Player recommendation

The recommender works on one player-season profile per player. Multi-team
stints are collapsed with minutes-weighted statistics before scoring. Candidate
features are standardized within the scoring pool, and distance is converted to
a readable score using `1 / (1 + distance)`.

Available ranking presets:

| Preset | Ranking focus |
| --- | --- |
| `playing_profile` | 85% role similarity, 10% workload reliability, 5% physical match |
| `role_similarity` | Pure statistical role similarity |
| `scoring_profile` | Scoring volume, usage, efficiency, 3PA rate, and free-throw rate |
| `defensive_profile` | Steal, block, defensive-rebound, and foul rates |
| `workload_fit` | Role similarity with greater minutes reliability weight |
| `physical_role_fit` | Role similarity with greater height/weight match weight |

Role similarity covers workload, scoring, playmaking, rebounding, and defensive
feature groups. Age is not used as a similarity feature. Candidate filtering
can enforce the same season, same position group, and a minimum-minutes floor.

Recommendation quality is audited with two proxy ground-truth layers:

- KMeans profile-cluster agreement checks whether recommendations occupy a
  similar unsupervised statistical cluster.
- Next-season profile similarity provides precision, recall, hit-rate, and MRR
  diagnostics against future statistical outcomes.

### Short-term forecasting

Three independent PyTorch LSTM models predict a player's average production
over the next five games.

| Task | Sequence design | Test rows | MAE | RMSE | R2 |
| --- | --- | ---: | ---: | ---: | ---: |
| Points | 10 recent games | 17,058 | 2.582 | 3.338 | 0.780 |
| Assists | 10 recent games | 17,058 | 0.738 | 0.984 | 0.789 |
| Rebounds | 15 recent games | 14,810 | 1.055 | 1.400 | 0.739 |

Each sequence contains target-stat and minutes deltas from the point-in-time
season averages. Static context contains the anchor season averages. The model
predicts a future delta, which is restored to the original stat scale for API
output. This design prevents future-game features from leaking into inference.

ARIMA was considered but not selected because NBA player histories form short,
irregular panel sequences. An attention-LSTM variant was also tested and did
not consistently outperform the selected LSTM.

### Long-term forecasting

Long-term models operate on player-season anchors and forecast:

- probability that the player remains active
- points per 36 minutes
- assists per 36 minutes
- rebounds per 36 minutes

H1 and H2 production use Random Forest models. H3 production uses MLP models.
Availability uses Random Forest for all three horizons. Selected configurations
are centralized in `src/config/long_term_config.py`.

| Target | H1 validation | H2 validation | H3 validation |
| --- | ---: | ---: | ---: |
| Active probability, Brier | 0.107 RF | 0.139 RF | 0.157 RF |
| Points per 36, MAE | 2.394 RF | 2.584 RF | 2.687 MLP |
| Assists per 36, MAE | 0.800 RF | 0.782 RF | 0.929 MLP |
| Rebounds per 36, MAE | 0.880 RF | 1.092 RF | 1.096 MLP |

Per-100-possession statistics remain useful recommendation and research
features, but they are not exposed as long-term API forecast targets.

### Deterministic scouting context

Alongside trained models, the project provides deterministic and explainable
signals for:

- recent improving, stable, or declining production trends
- player consistency and volatility
- validation-calibrated expected/floor/ceiling ranges
- salary and salary-cap-share history
- recommendation explanations and profile gaps

The range weights and trend thresholds were selected from validation data and
are documented in [docs/deterministic_scouting.md](docs/deterministic_scouting.md).

Direct salary forecasting was removed from the product after experiments showed
that previous salary was difficult to beat reliably because of multi-year
contracts, role transitions, and discontinuous market changes. Salary is now
presented as decision context rather than as an unsupported point forecast.

## Architecture

```text
External NBA and Kaggle datasets
        |
        v
Raw -> Bronze -> Silver -> Gold parquet datasets
        |                    |
        |                    +-> deterministic recommender and scouting signals
        |                    +-> short-term LSTM training
        |                    +-> long-term RF/MLP training
        v
MLflow tracking + persisted model artifacts
        |
        v
FastAPI schemas -> services -> inference/reporting pipelines
        |
        +-> JSON API
        +-> static HTML/CSS/JavaScript dashboard
        |
        v
Docker Hub -> AWS ECS Fargate
```

Application resources are loaded once during FastAPI startup and reused across
requests. The service does not reload datasets or model artifacts for every
prediction.

## Data Sources

Multiple sources are combined because game-level, advanced, physical, and
compensation data are not available in one stable dataset.

| Data domain | Source | Canonical local file |
| --- | --- | --- |
| Historical game logs | KaggleHub `szymonjwiak/nba-traditional` | `data/silver/player_game_logs.parquet` |
| Player bio/profile | Historical NBA `Players.csv` | `data/raw/players.parquet` |
| Advanced/rate statistics | Regular-season advanced, usage, and defense snapshots | `data/raw/player_season_stats.parquet` |
| 2023-24 advanced patch | KaggleHub `rodneycarroll78/nba-stats-1980-2024` | `data/raw/player_stats_advanced_patch/2023_24/Advanced.csv` |
| 2024-25 advanced patch | KaggleHub `ratin21/nba-player-stats-2024-25-per-game` | `data/raw/player_stats_advanced_patch/2024_25/*.csv` |
| Historical salaries | Kaggle salary history | `data/silver/player_season_salaries.parquet` |
| Salary-cap history | Curated NBA cap and tax history | `data/raw/salary_cap/salary_cap_by_season.csv` |
| Contract history | Optional manually staged reference table | `data/raw/contract_value/contract_events.csv` |
| NBA API fallback | `nba_api` and `stats.nba.com` | `data/raw/nba_api_cache/` |

Detailed source-specific joins and cleaning rules are documented in
[docs/data_pipeline.md](docs/data_pipeline.md).

## Data Pipeline

```text
data/raw      source-level files
data/bronze   downloaded or staged source snapshots
data/silver   normalized intermediate tables
data/gold     final training and inference datasets
```

Primary serving datasets:

```text
data/gold/player_role_features_clean.parquet
data/gold/performance_training_clean.parquet
data/gold/player_salary_history_clean.parquet
data/gold/long_term_player_forecast_inference.parquet
```

The local pipeline handles name normalization, season normalization,
percentage/rate conversion, player-ID joins, physical context, multi-team
stints, point-in-time rolling features, future targets, missing-value policy,
season-coverage checks, and leakage-safe temporal split labels.

Data and model binaries are versioned outside normal Git history. DVC setup and
workflow details are available in [docs/dvc.md](docs/dvc.md).

## API

FastAPI entrypoint: `app/main.py`

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Container and service health |
| `GET` | `/metadata` | Loaded row counts, seasons, and model artifacts |
| `POST` | `/recommendations` | Ranked Top-K similar players plus diagnostics |
| `POST` | `/forecasts/short-term` | Requested next-five-game forecasts |
| `POST` | `/forecasts/long-term` | Requested H1/H2/H3 trajectory forecasts |
| `POST` | `/players/scouting-report` | Composed player profile, forecasts, recent form, and compensation context |

Open `/docs` for the generated OpenAPI interface. Player requests accept either
`player_id` or normalized `player_name`; omitted seasons resolve to the latest
available compatible row.

## Frontend

The demo frontend is a dependency-free static application:

```text
app/static/index.html
app/static/styles.css
app/static/app.js
```

FastAPI mounts these files at `/`, so the UI and API share one origin and one
container. The interface supports recommendation filters, six ranking presets,
candidate selection, profile metrics, short- and long-term forecasts, salary
history, and recent-game summaries.

## Local Usage

### Run the published image

```bash
docker pull khoatran1/nba-scout-assistant:latest

docker run --rm --name nba-scout-api \
  -p 8001:8001 \
  khoatran1/nba-scout-assistant:latest
```

Open:

```text
http://localhost:8001
http://localhost:8001/docs
http://localhost:8001/health
```

### Run from source

Source execution requires the four serving datasets and selected artifacts in
`data/gold/` and `artifacts/`.

```bash
python -m pip install -r requirements-api.txt
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

### Build from the local serving snapshot

```bash
docker build -f docker/Dockerfile -t nba-scout-assistant-api .
```

## Training And Evaluation

```bash
# Train the three short-term tasks
python -m src.training.train_short_term --task points
python -m src.training.train_short_term --task assists
python -m src.training.train_short_term --task rebounds

# Train one or all long-term task/horizon combinations
python -m src.training.train_long_term --task pts_per_36 --horizon 1
python -m src.training.train_long_term --task all

# Evaluate a deployed short-term checkpoint
python -m src.evaluation.evaluate_short_term \
  --model-path artifacts/short_term_lstm_points.pt \
  --task points \
  --splits validation test

# Run the complete test suite
pytest -q
```

MLflow records parameters, validation/test metrics, artifacts, and model
configuration fingerprints. Training can reuse compatible tracked results
instead of recomputing unchanged experiments.

## CI/CD And Cloud Deployment

CI runs on every push and pull request targeting `main`:

```text
checkout -> Python 3.12 -> install dev dependencies -> compile -> pytest
```

After CI succeeds on `main`, CD automatically:

```text
checks out the tested SHA
    -> builds linux/amd64 with Buildx
    -> pushes Docker Hub tags <commit-sha> and latest
    -> downloads the current ECS task definition
    -> replaces the nba-scout-api container image with the immutable SHA tag
    -> registers a new task-definition revision
    -> updates the ECS service and waits for stability
```

The deployment uses:

```text
Docker Hub: khoatran1/nba-scout-assistant
AWS region: us-east-2
ECS cluster: nba-scout-assistant-cluster
ECS service: nba-scout-assistant-service
Task family: nba-scout-assistant
Container: nba-scout-api
Runtime: AWS Fargate, Linux/X86_64
```

Because serving binaries are ignored by Git, `docker/Dockerfile.cd` takes
`/app/data` and `/app/artifacts` from the immutable `0.2.0` runtime snapshot,
then rebuilds dependencies and application code from the SHA that passed CI.

## Repository Structure

```text
app/                 FastAPI entrypoint, schemas, services, and static frontend
src/config/          selected short- and long-term model configurations
src/dataset/         loaders, cleaning, feature engineering, and split logic
src/models/          LSTM, MLP, and Random Forest model definitions
src/training/        training orchestration and MLflow integration
src/evaluation/      metrics, data checks, model evaluation, and recommender diagnostics
src/scouting/        deterministic signals, ranges, similarity, and ranking
src/pipelines/       data, artifact, forecasting, recommendation, and reporting pipelines
notebooks/           exploratory processing and model-selection experiments
tests/               automated unit and integration-style tests
docs/                data, DVC, and deterministic-scouting documentation
docker/              local and CI/CD Dockerfiles
.github/workflows/   CI and CD workflows
```

## Current Limitations

- Serving data is a fixed snapshot through 2024-25; scheduled ingestion and
  retraining are not automated yet.
- Contract history is optional and is not populated in the current cloud
  snapshot; salary history remains available as reference context.
- The current ECS demo uses a basic single-task deployment without a custom
  domain, HTTPS termination, or an Application Load Balancer.
- Recommendation relevance uses defensible proxy ground truth because no
  universal labeled dataset of correct NBA player replacements exists.
- Long-horizon forecasts should be interpreted as estimates, especially H3,
  where role changes, injuries, and retirement create substantial uncertainty.

These constraints are explicit product boundaries rather than hidden model
assumptions. The current system is a complete portfolio-grade ML application
and a stable base for scheduled data refresh, managed artifact storage,
monitoring, authentication, and a richer frontend.
