# Recommendation Ranking

## Product Objective

The recommendation system retrieves players whose observed playing profiles
are comparable to a requested player-season. It supports six presets, but only
`playing_profile` uses a learned ranker. The focused presets remain
deterministic so their meaning stays explicit.

This is a retrieval and ranking problem. It is not a player-quality rating:
the system ranks similarity, not which candidate is universally better.

## Feature Contract

The ranker uses 15 non-overlapping observed features:

| Group | Features |
| --- | --- |
| Workload | `minutes`, `usage_pct` |
| Scoring | `points_per_100`, `true_shooting_pct`, `three_point_attempt_rate`, `free_throw_rate` |
| Playmaking | `assists_per_100`, `turnover_rate` |
| Rebounding | `rebounds_per_100`, `defensive_rebound_rate` |
| Defense | `steal_rate`, `block_rate`, `foul_rate` |
| Physical | `height`, `weight` |

Derived dimensions such as `scoring_creation`, `playmaking`, `shooting`, and
defensive composite scores are excluded. Including them beside their source
statistics would count the same evidence more than once.

Age is not a similarity feature. Position group, season, and minimum minutes
are candidate filters.

## Point-In-Time Preprocessing

Rates from small minute samples are noisy. The preprocessing layer therefore
shrinks each rate toward its minutes-weighted season mean:

```text
reliability = minutes / (minutes + prior_strength)

adjusted_rate =
    reliability * observed_rate
    + (1 - reliability) * season_prior
```

Optuna selects `prior_strength` from `250`, `500`, `750`, `1000`, and `1500`
using validation NDCG@5. Height, weight, and minutes are not shrunk.

Every adjusted feature is standardized against the complete player pool for
that season. The scaler is never fitted on the candidates left after one API
request applies its filters. The fitted season statistics and feature order are
stored in the ranker artifact.

For one target-candidate pair, the model input is the absolute difference in
each normalized observed feature. No next-season field is present in model
input.

## Proxy Relevance Labels

There is no universal expert-labeled dataset of correct NBA replacement
players. Model selection therefore uses next-season profile similarity as a
proxy outcome:

1. Construct the target and eligible candidate profiles using season `t` only.
2. Find the same players' observed profiles in season `t+1`.
3. Rank the available candidates by future profile distance to the target.
4. Assign relevance `5, 4, 3, 2, 1` to future ranks 1 through 5.
5. Assign relevance `0` to every other candidate, including players without a
   next-season profile.

The target must have a next-season row for its query to be evaluable. Candidate
filters are the same position group and at least 500 season minutes by default.

Temporal query splits are:

| Split | Query seasons | Outcome needed |
| --- | --- | --- |
| Train | 2016-17 through 2021-22 | 2017-18 through 2022-23 |
| Validation | 2022-23 | 2023-24 |
| Locked test | 2023-24 | 2024-25 |
| Inference only | 2024-25 | no 2025-26 proxy available |

This label is useful for reproducible comparison, but it remains a temporal
proxy rather than expert scouting truth.

## Champion And Challengers

Three algorithms are evaluated on identical queries:

1. The previous weighted Euclidean implementation is the production champion.
2. Ledoit-Wolf Mahalanobis distance is the covariance-aware deterministic
   challenger.
3. XGBoost LambdaMART is the learned challenger, using `rank:ndcg` and
   validation `ndcg@5`.

LambdaMART follows the standard learning-to-rank query-group design: each
target player-season is one query and all eligible candidates form its document
group. Rows are sorted by numeric query ID before fitting. See the
[XGBoost learning-to-rank guide](https://xgboost.readthedocs.io/en/release_3.2.0/tutorials/learning_to_rank.html)
and the [LambdaMART overview](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/).

The training entrypoint runs 40 Optuna trials and verifies the selected
configuration with seeds 42, 123, and 2026:

```bash
python -m src.training.train_recommendation --trials 40
```

MLflow records trial parameters, validation metrics, selected metrics, feature
importance, and the final artifact.

## Promotion Gate

A challenger replaces the previous ranker only when locked-test evaluation
satisfies every condition:

- mean NDCG@5 improvement is positive;
- the paired query-bootstrap 95% confidence interval has a non-negative lower
  bound;
- Recall@5 and MRR each decline by no more than 0.01;
- p95 model-scoring latency is below 100 ms on complete eligible candidate
  pools.

The selected LambdaMART ranker passed all four conditions. Mahalanobis improved
the point estimate slightly but its confidence interval crossed zero, so it was
not promoted.

Locked-test results from 338 player-season queries are:

| Algorithm | NDCG@5 | Recall@5 | Hit rate@5 | MRR | Coverage@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous weighted Euclidean | 0.1713 | 0.1740 | 0.5503 | 0.3600 | 0.8583 |
| Ledoit-Wolf Mahalanobis | 0.1725 | 0.1757 | 0.6036 | 0.3606 | 0.9194 |
| Selected LambdaMART | 0.2114 | 0.2077 | 0.6864 | 0.4209 | 0.9278 |

LambdaMART's mean per-query NDCG@5 improvement is `0.0402`, with paired
bootstrap 95% confidence interval `[0.0144, 0.0665]`. Its measured p95 model
scoring latency is `0.615 ms` on the complete eligible query pools used by the
test benchmark.

The selected artifact is:

```text
artifacts/recommendation_playing_profile_ranker.joblib
```

It contains the model, season preprocessing, raw and pair feature contracts,
selected parameters, temporal split policy, evaluation metrics, and version.

## Serving Behavior

FastAPI loads the ranker once at startup. `playing_profile` uses LambdaMART when
the artifact is present. If the artifact is absent, serving remains available
through season-normalized Euclidean ranking.

The remaining presets use fixed season normalization and direct, non-duplicated
feature subsets:

- `role_similarity`: observed workload and performance profile;
- `scoring_profile`: usage, scoring volume, efficiency, and shot mix;
- `defensive_profile`: steals, blocks, defensive rebounding, and foul rate;
- `workload_fit`: minutes and usage;
- `physical_role_fit`: workload, height, and weight.

Every recommendation row includes `ranking_algorithm` and `ranker_version` in
addition to the existing response fields.

## Evaluation Outputs

The experiment writes:

```text
reports/recommendation_ranking/lambdamart_optuna_trials.parquet
reports/recommendation_ranking/lambdamart_seed_evaluation.parquet
reports/recommendation_ranking/recommendation_ranker_evaluation.parquet
reports/recommendation_ranking/lambdamart_feature_importance.csv
reports/recommendation_ranking/selection_summary.json
```

Reported metrics are NDCG@5, Recall@5, hit rate, MRR, recommendation coverage,
paired-bootstrap confidence intervals, and p50/p95 scoring latency.
