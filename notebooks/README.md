# Notebook Workflow

Run notebooks in numerical order when rebuilding the project from source data. Experiment notebooks can be rerun independently once their required Gold tables exist.

| Order | Notebook | Purpose | Main input | Main output |
| --- | --- | --- | --- | --- |
| 00 | `00_data_ingestion.ipynb` | Download, inspect, and normalize source datasets. | Kaggle snapshots, bio files, advanced-stat patches, salary sources | Canonical Raw and Silver tables |
| 01 | `01_build_gold_datasets.ipynb` | Run the source-controlled data pipeline. | Canonical Raw and Silver tables | Six current Gold datasets |
| 02 | `02_data_visualization.ipynb` | Inspect source, intermediate, and Gold tables before and after processing. | Raw, Silver, and Gold layers | Visual schema and transformation audit |
| 03 | `03_feature_target_exploration.ipynb` | Analyze feature relationships before model comparison. | Short-term and long-term Gold training tables | Correlation, mutual information, nonlinearity, importance, and redundancy tables |
| 04 | `04_short_term_forecasting.ipynb` | Compare the current LSTM with an attention variant. | `performance_training_clean.parquet` | Short-term validation and test metrics |
| 05 | `05_long_term_forecasting.ipynb` | Compare tuned tabular MLP and classical ML models for H1-H3. | `long_term_player_forecast_training.parquet` | Long-term model selection metrics and configs |
| 06 | `06_scouting_signals.ipynb` | Calibrate deterministic trend, consistency, volatility, and range signals. | Short-term and role Gold tables | Deterministic scouting artifacts |
| 07 | `07_player_recommendation.ipynb` | Test candidate generation, similarity scoring, ranking presets, and explanations. | Role, performance, and scouting tables | Recommendation examples and ranking audits |

## Data Timeline

```text
external sources
-> 00 ingestion
-> canonical raw / silver
-> 01 Gold build
-> 02 visual audit
-> 03 feature exploration
-> 04-05 forecasting experiments
-> 06 deterministic scouting signals
-> 07 player recommendation
```

## Boundaries

- Data acquisition and canonicalization belong only in notebook 00.
- Gold feature construction belongs only in notebook 01 and `src/dataset`.
- Notebook 02 reads and reconstructs stages for inspection; it does not overwrite data.
- Notebook 03 analyzes feature-target relationships but does not select production models.
- Notebooks 04 and 05 own predictive model experiments.
- Notebooks 06 and 07 own deterministic scouting and recommendation experiments.
- Salary and contract rows are retained as player-detail context. Salary forecasting is not an active task.
