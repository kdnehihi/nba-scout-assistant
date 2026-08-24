# Deterministic Scouting Process

This document describes the deterministic scouting layer under `src/scouting`.
These outputs are designed to support scout-facing interpretation without
training a supervised model.

## Scope

The deterministic layer currently covers:

- player trend signals
- player consistency and volatility signals
- short-term expected/floor/ceiling ranges
- replacement candidate ranking
- recommendation ground-truth proxy evaluation
- compensation context for player detail pages

The goal is to keep these signals simple, auditable, and low-risk. When a
heuristic contains tunable weights or thresholds, the preferred approach is to
select those values using temporal validation rather than manual judgment.

## Parameter Audit

The deterministic layer contains three kinds of numeric parameters:

```text
validation-calibrated parameters
statistical definitions
descriptive or business-rule parameters
```

Only the first group should be described as data-calibrated.

| Parameter | Current Value | Status | Basis |
|---|---:|---|---|
| `points_tolerance` | `3.0` | validation-calibrated | selected on validation directional hit rate |
| `assists_tolerance` | `1.0` | validation-calibrated | selected on validation directional hit rate |
| `rebounds_tolerance` | `1.2` | validation-calibrated | selected on validation directional hit rate |
| `season_avg_weight` | `0.563203` | validation-calibrated | optimized validation normalized MAE |
| `last_10_weight` | `0.308359` | validation-calibrated | optimized validation normalized MAE |
| `last_5_weight` | `0.128438` | validation-calibrated | optimized validation normalized MAE |
| `volatility_multiplier` | `0.825154` | validation-calibrated | optimized validation range coverage |
| `rolling_std_window` | `10` | product/statistical definition | matches the recent 10-game horizon used by short-term features |
| `rolling_std_min_periods` | `5` | product/statistical definition | avoids volatility estimates from very short samples |
| `min_games` | `10` | product/statistical definition | minimum sample for season-level consistency summaries |
| `p20/p50/p80` | `0.20/0.50/0.80` | statistical definition | distribution summary bands, not fitted weights |
| `0.6745` in robust z-score | `0.6745` | statistical definition | median absolute deviation scaling constant |
| consistency label bands | `-0.5/0.5` | descriptive | interpretability bands around season-relative robust z-score |
| similarity feature weights | equal | descriptive | standardized role similarity without relevance labels |
| `similarity_score` formula | `1 / (1 + distance)` | deterministic transform | monotonic conversion from distance to score |

The descriptive parameters are intentionally centralized in config classes where
possible. They should be recalibrated only after the project has labels or a
clear downstream metric for the relevant decision.

## Source Notebook

The reviewed calibration output came from:

```text
notebooks/06_scouting_signals.ipynb
```

The notebook reads the same clean gold data used by local modules:

```text
data/gold/performance_training_clean.parquet
data/gold/player_role_features_clean.parquet
data/gold/player_salary_history_clean.parquet
```

## Trend Signals

### Purpose

Trend signals answer:

```text
Is the player's recent production above, near, or below his season baseline?
```

The local implementation is:

```text
src/scouting/signals.py
```

The main function is:

```python
build_player_trend_signals(performance_df)
```

### Calculation

For each player-season, the function keeps the latest available row and
computes:

```text
pts_recent_delta = pts_last_5 - pts_season_avg
ast_recent_delta = ast_last_5 - ast_season_avg
reb_recent_delta = reb_last_5 - reb_season_avg
min_recent_delta = min_last_5 - min_season_avg
```

Each delta is converted into:

```text
improving / stable / declining
```

using:

```text
improving: delta > tolerance
declining: delta < -tolerance
stable: otherwise
```

### Validation Calibration

Trend thresholds for PTS/AST/REB were selected from validation data by comparing
the current recent-vs-season delta with the future next-five-game delta.

Selection metric:

```text
directional_hit_rate among non-stable labels
```

with a minimum non-stable coverage requirement.

Selected validation thresholds:

```text
points_tolerance = 3.0
assists_tolerance = 1.0
rebounds_tolerance = 1.2
```

Validation evidence:

```text
PTS: tolerance 3.0, non_stable_rate 0.256, directional_hit_rate 0.636
AST: tolerance 1.0, non_stable_rate 0.204, directional_hit_rate 0.649
REB: tolerance 1.2, non_stable_rate 0.256, directional_hit_rate 0.643
```

These values are now reflected in `TrendConfig`.

### Non-Calibrated Trend Setting

`minutes_tolerance` remains descriptive rather than validated because the
short-term performance table currently does not contain a next-five-minutes
target used for directional validation.

Current value:

```text
minutes_tolerance = 3.0
```

This should be revisited if a reliable future minutes label is added.

## Consistency And Volatility Signals

### Purpose

Consistency signals answer:

```text
Is this player's production stable or volatile within a season?
```

The local implementation is:

```text
src/scouting/signals.py
```

The main function is:

```python
build_player_consistency_signals(performance_df)
```

### Calculation

For each player-season-team group with at least 10 games, the function computes
for PTS, AST, REB, and MIN:

```text
mean
standard deviation
coefficient of variation
p20 / p50 / p80
above_mean_rate
```

Then it normalizes each coefficient of variation by season using robust
z-scores. The final consistency label is:

```text
consistent
balanced
volatile
```

### Calibration Status

This signal is descriptive. It does not use future labels and therefore is not
selected by supervised validation.

The current label thresholds:

```text
overall_volatility_score <= -0.5 -> consistent
overall_volatility_score >= 0.5 -> volatile
otherwise -> balanced
```

These thresholds should be treated as interpretability bands. They should only
be recalibrated if the project later defines a downstream target such as
future prediction error, role volatility, or human-labeled consistency classes.

## Short-Term Floor/Expected/Ceiling Ranges

### Purpose

Short-term ranges answer:

```text
What is a reasonable expected/floor/ceiling range for the player's next-five-game average production?
```

The local implementation is:

```text
src/scouting/ranges.py
```

The main functions are:

```python
build_short_term_floor_ceiling_signals(performance_df)
evaluate_floor_ceiling_signals(signals)
```

### Calculation

For each stat:

```text
PTS, AST, REB
```

the expected value is:

```text
expected =
    season_avg_weight * season_avg
  + last_10_weight * last_10
  + last_5_weight * last_5
```

The range is:

```text
floor = max(0, expected - volatility_multiplier * rolling_std)
ceiling = expected + volatility_multiplier * rolling_std
```

`rolling_std` is computed from the last 10 games within the same player-season,
with at least 5 games required.

### Validation Calibration

The original fixed heuristic was:

```text
season_avg_weight = 0.50
last_10_weight = 0.30
last_5_weight = 0.20
volatility_multiplier = 0.80
```

This was replaced with validation-calibrated values.

Expected weights were selected with:

```text
scipy.optimize.differential_evolution
```

Objective:

```text
minimize average normalized validation MAE across PTS/AST/REB
```

The volatility multiplier was selected with:

```text
scipy.optimize.minimize_scalar
```

Objective:

```text
target validation coverage around 0.80 while avoiding unnecessarily wide ranges
```

Selected calibrated config:

```text
season_avg_weight = 0.563203
last_10_weight = 0.308359
last_5_weight = 0.128438
volatility_multiplier = 0.825154
```

These values are now reflected in `RangeConfig`.

### Validation And Test Evidence

Calibrated validation metrics:

```text
AST: MAE 0.7675, R2 0.7766, coverage 0.8042
PTS: MAE 2.6268, R2 0.7897, coverage 0.7941
REB: MAE 1.0542, R2 0.7463, coverage 0.8015
```

Calibrated test metrics:

```text
AST: MAE 0.7527, R2 0.7775, coverage 0.8029
PTS: MAE 2.6212, R2 0.7734, coverage 0.7963
REB: MAE 1.0807, R2 0.7274, coverage 0.7999
```

Compared with the original fixed config, the calibrated config slightly improves
MAE and R2 while moving coverage closer to 0.80. The tradeoff is slightly wider
ranges.

## Replacement Candidate Ranking

### Purpose

Replacement ranking answers:

```text
Which players have similar statistical role profiles in the same season?
```

The local implementation is:

```text
src/scouting/similarity.py
```

The main functions are:

```python
build_similarity_base(role_features, salary)
find_replacement_candidates(...)
```

### Calculation

The function:

1. Builds one player-season role table with salary context.
2. Filters candidates by season, position group, salary, age, and minutes when
   requested.
3. Standardizes role features. `points_per_100`, `assists_per_100`, and
   `rebounds_per_100` are treated as per-100-possession production fields.
4. Computes Euclidean distance to the target player.
5. Converts distance into:

```text
similarity_score = 1 / (1 + similarity_distance)
```

### Calibration Status

Similarity uses equal-weight standardized features. This is deterministic and
auditable, but it is not supervised-calibrated.

Calibrating similarity weights requires one of:

- human-labeled similar-player pairs
- downstream retrieval success labels
- a business objective such as cheaper replacement hit rate

Until one of those exists, similarity should be described as:

```text
statistical role similarity
```

not as:

```text
equivalent player quality
```

### Ranking Profiles

The focused ranking presets use a single standardized feature group, so their
scores do not depend on manually invented cross-group weights:

- `scoring_profile` compares points per 100 possessions, usage percentage,
  true shooting percentage, three-point attempt rate, and free-throw rate.
- `defensive_profile` compares steal rate, block rate, defensive rebound rate,
  and foul rate.

Each feature is standardized within the target and filtered candidate pool.
The group distance is the root mean squared standardized feature difference,
and the displayed score is `1 / (1 + distance)`. Derived role dimensions are
excluded from these focused profiles because they duplicate their source
statistics and would implicitly overweight them. Position, season, and minimum
minutes remain candidate filters rather than hidden ranking weights.

When a player has multiple team stints in one season, the recommender first
creates one player-season profile by minutes-weighting the rate statistics and
summing minutes. This prevents traded players from occupying multiple places
in the top-K result while retaining information from every stint.

## Recommendation Ground Truth Proxy

### Purpose

The project does not have human-labeled player-comparison data. To make
recommendation quality auditable, the local evaluation layer builds a proxy
ground truth from future outcomes.

The proxy answers:

```text
If the recommendation is made at season t, did it retrieve players whose
season t+1 profile became similar to the target player's season t+1 profile?
```

The local implementation is:

```text
src/evaluation/evaluate_similarity.py
```

The main functions are:

```python
build_future_similarity_ground_truth(...)
evaluate_recommendations_against_ground_truth(...)
```

### Label Construction

For every target player-season:

1. Find the same player's next-season row.
2. Find candidate players from the same anchor season.
3. Compare target and candidate next-season profiles using standardized role
   and physical features.
4. Mark the closest `relevant_n` candidates as proxy-relevant.

This creates labels such as:

```text
target_player_id
target_season
candidate_player_id
future_relevance_rank
future_similarity_score
```

### Metrics

The recommendation output is evaluated with:

- `hit_rate`: share of returned top-K recommendations that are proxy-relevant.
- `recall_at_k`: share of proxy-relevant players recovered in the top-K list.
- `mrr`: reciprocal rank of the first proxy-relevant recommendation.

### Limitations

This is not a scout-labeled ground truth. It is a future-outcome proxy. It is
useful for sanity checks and model comparison, but final recommendation quality
should still be reviewed qualitatively on known player cases.

## Compensation Context

### Purpose

Compensation context attaches current and historical salary data to player
detail pages after recommendations are generated. It is descriptive reference
data, not a prediction task.

```text
latest salary
salary cap share
salary history
optional contract history
```

The local implementation is:

```text
src/scouting/compensation.py
```

The main function is:

```python
build_player_compensation_context(...)
```

### Status

The salary and contract datasets are too sparse and contract-dependent for a
reliable local salary forecast. The product should display this information as
context for scout review rather than presenting a model-estimated salary value.

## Local Promotion Rules

Use this standard before promoting notebook heuristics into local source:

```text
1. If the heuristic has labels, select parameters on validation.
2. If the heuristic does not have labels, document it as descriptive.
3. Do not present descriptive signals as supervised predictions.
4. Keep test metrics for final audit only after validation selection.
5. Store final selected values in config classes.
```

Current calibrated config values are stored in:

```text
src/scouting/config.py
```
