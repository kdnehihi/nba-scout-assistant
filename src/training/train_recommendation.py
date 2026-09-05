from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.recommendation_config import (
    DEFAULT_RECOMMENDATION_SPLIT_POLICY,
    DEFAULT_SHRINKAGE_PRIOR_STRENGTH,
    PAIR_FEATURES,
    RECOMMENDATION_RANKER_FILENAME,
    RECOMMENDATION_RANKER_VERSION,
    SHRINKAGE_PRIOR_STRENGTHS,
)
from src.dataset.loaders import (
    load_performance_training_clean,
    load_role_features_clean,
    resolve_data_paths,
)
from src.dataset.recommendation_modeling import (
    assert_point_in_time_ranker_inputs,
    build_temporal_ranking_dataset,
    ranking_arrays,
    split_ranking_dataset,
)
from src.evaluation.evaluate_recommendation_ranking import (
    benchmark_query_latency,
    evaluate_ranking_queries,
    model_scores,
    paired_query_bootstrap,
    score_legacy_weighted_euclidean,
)
from src.scouting.ranking import (
    RecommendationRankerArtifact,
    fit_ledoit_wolf_precision,
    mahalanobis_similarity,
    save_recommendation_ranker_artifact,
)
from src.scouting.recommendation import build_recommendation_base
from src.training.mlflow_utils import (
    configure_mlflow,
    log_metrics_flat,
    log_params_flat,
)


def build_lambdamart(parameters: dict[str, Any], seed: int):
    """Construct an XGBoost LambdaMART ranker from one tuned configuration."""
    from xgboost import XGBRanker

    return XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@5",
        tree_method="hist",
        lambdarank_pair_method="topk",
        lambdarank_num_pair_per_sample=8,
        early_stopping_rounds=50,
        n_jobs=4,
        random_state=seed,
        **parameters,
    )


def fit_lambdamart(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    parameters: dict[str, Any],
    seed: int,
):
    """Fit one LambdaMART candidate with query-contiguous train and validation rows."""
    X_train, y_train, qid_train = ranking_arrays(train_df)
    X_validation, y_validation, qid_validation = ranking_arrays(validation_df)
    model = build_lambdamart(parameters, seed=seed)
    model.fit(
        X_train,
        y_train,
        qid=qid_train,
        eval_set=[(X_validation, y_validation)],
        eval_qid=[qid_validation],
        verbose=False,
    )
    return model


def _fit_final_lambdamart(
    train_validation_df: pd.DataFrame,
    parameters: dict[str, Any],
    seed: int,
    n_estimators: int,
):
    """Refit the selected LambdaMART configuration without the locked test labels."""
    from xgboost import XGBRanker

    X, y, qid = ranking_arrays(train_validation_df)
    final_parameters = {**parameters, "n_estimators": n_estimators}
    model = XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@5",
        tree_method="hist",
        lambdarank_pair_method="topk",
        lambdarank_num_pair_per_sample=8,
        n_jobs=4,
        random_state=seed,
        **final_parameters,
    )
    model.fit(X, y, qid=qid, verbose=False)
    return model


def _trial_parameters(trial: Any) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
        "max_depth": trial.suggest_int("max_depth", 2, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.18, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 24.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
    }


def _promotion_gate(
    challenger_metrics: dict[str, float],
    champion_metrics: dict[str, float],
    confidence_interval: dict[str, float],
    latency: dict[str, float],
) -> tuple[bool, dict[str, bool]]:
    checks = {
        "positive_ndcg_improvement": confidence_interval["mean_improvement"] > 0,
        "non_negative_ci_lower": confidence_interval["ci_lower"] >= 0,
        "recall_regression_within_0_01": (
            challenger_metrics["recall_at_5"] - champion_metrics["recall_at_5"] >= -0.01
        ),
        "mrr_regression_within_0_01": challenger_metrics["mrr"] - champion_metrics["mrr"] >= -0.01,
        "p95_latency_below_100_ms": latency["latency_p95_ms"] < 100.0,
    }
    return all(checks.values()), checks


def run_recommendation_experiment(
    data_dir: str | Path = "data",
    artifact_dir: str | Path = "artifacts",
    report_dir: str | Path = "reports/recommendation_ranking",
    trials: int = 40,
    seeds: tuple[int, ...] = (42, 123, 2026),
    use_mlflow: bool = True,
    mlflow_db_path: str | Path = "mlflow.db",
    mlflow_artifact_dir: str | Path = "mlartifacts",
) -> tuple[RecommendationRankerArtifact, pd.DataFrame]:
    """Tune, gate, persist, and report the recommendation champion/challengers."""
    import optuna

    paths = resolve_data_paths(data_dir)
    role = load_role_features_clean(paths)
    performance = load_performance_training_clean(paths)
    base = build_recommendation_base(role, performance_df=performance)
    report_root = Path(report_dir).expanduser().resolve()
    report_root.mkdir(parents=True, exist_ok=True)
    artifact_path = Path(artifact_dir).expanduser().resolve() / RECOMMENDATION_RANKER_FILENAME

    dataset_cache: dict[int, tuple[pd.DataFrame, Any]] = {}

    def dataset_for(prior_strength: int):
        if prior_strength not in dataset_cache:
            ranking, preprocessor = build_temporal_ranking_dataset(
                base,
                prior_strength=prior_strength,
            )
            assert_point_in_time_ranker_inputs(ranking)
            dataset_cache[prior_strength] = (ranking, preprocessor)
        return dataset_cache[prior_strength]

    reference_ranking, _ = dataset_for(DEFAULT_SHRINKAGE_PRIOR_STRENGTH)
    reference_splits = split_ranking_dataset(reference_ranking)
    baseline_validation_scores = score_legacy_weighted_euclidean(reference_splits["validation"], base)
    baseline_validation, _ = evaluate_ranking_queries(
        reference_splits["validation"], baseline_validation_scores, algorithm="weighted_euclidean_v1"
    )

    mlflow_module = None
    if use_mlflow:
        mlflow_module = configure_mlflow(
            tracking_db_path=mlflow_db_path,
            artifact_dir=mlflow_artifact_dir,
            experiment_name="nba_scout_recommendation_ranking",
        )

    trial_records: list[dict[str, Any]] = []

    def objective(trial: Any) -> float:
        prior_strength = trial.suggest_categorical("prior_strength", list(SHRINKAGE_PRIOR_STRENGTHS))
        parameters = _trial_parameters(trial)
        ranking, _ = dataset_for(int(prior_strength))
        split_frames = split_ranking_dataset(ranking)
        model = fit_lambdamart(
            split_frames["train"],
            split_frames["validation"],
            parameters=parameters,
            seed=42,
        )
        validation_scores = model_scores(model, split_frames["validation"])
        metrics, _ = evaluate_ranking_queries(
            split_frames["validation"], validation_scores, algorithm="lambdamart"
        )
        best_iteration = int(getattr(model, "best_iteration", parameters["n_estimators"] - 1))
        trial.set_user_attr("best_iteration", best_iteration)
        trial.set_user_attr("recall_at_5", metrics["recall_at_5"])
        trial.set_user_attr("mrr", metrics["mrr"])
        record = {
            "trial": trial.number,
            "prior_strength": prior_strength,
            **parameters,
            "best_iteration": best_iteration,
            **{f"validation_{key}": value for key, value in metrics.items()},
        }
        trial_records.append(record)
        if mlflow_module is not None:
            with mlflow_module.start_run(run_name=f"lambdamart_trial_{trial.number}"):
                log_params_flat(mlflow_module, {"prior_strength": prior_strength, "seed": 42, **parameters})
                log_metrics_flat(mlflow_module, metrics, prefix="validation")
                mlflow_module.log_metric("best_iteration", best_iteration)
        return metrics["ndcg_at_5"]

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler, study_name="recommendation_lambdamart")
    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    best = study.best_trial
    selected_prior = int(best.params["prior_strength"])
    selected_parameters = {key: value for key, value in best.params.items() if key != "prior_strength"}
    selected_ranking, selected_preprocessor = dataset_for(selected_prior)
    selected_splits = split_ranking_dataset(selected_ranking)

    seed_results: list[dict[str, Any]] = []
    fitted_by_seed: dict[int, Any] = {}
    query_metrics_by_seed: dict[int, dict[str, pd.DataFrame]] = {}
    for seed in seeds:
        model = fit_lambdamart(
            selected_splits["train"],
            selected_splits["validation"],
            parameters=selected_parameters,
            seed=seed,
        )
        fitted_by_seed[seed] = model
        query_metrics_by_seed[seed] = {}
        for split_name in ("validation", "test"):
            frame = selected_splits[split_name]
            metrics, per_query = evaluate_ranking_queries(
                frame,
                model_scores(model, frame),
                algorithm="lambdamart",
            )
            query_metrics_by_seed[seed][split_name] = per_query
            seed_results.append({"algorithm": "lambdamart", "seed": seed, "split": split_name, **metrics})

    seed_evaluation = pd.DataFrame(seed_results)
    median_seed = int(
        seed_evaluation[seed_evaluation["split"].eq("validation")]
        .sort_values("ndcg_at_5")
        .iloc[len(seeds) // 2]["seed"]
    )
    selected_model = fitted_by_seed[median_seed]

    # Tune the deterministic covariance challenger over the same validation-only prior grid.
    mahalanobis_validation: list[tuple[float, int, np.ndarray]] = []
    for prior_strength in SHRINKAGE_PRIOR_STRENGTHS:
        ranking, preprocessor = dataset_for(prior_strength)
        frames = split_ranking_dataset(ranking)
        normalized = preprocessor.transform(base)
        train_profiles = normalized[
            normalized["season"].between(
                DEFAULT_RECOMMENDATION_SPLIT_POLICY.train_start,
                DEFAULT_RECOMMENDATION_SPLIT_POLICY.train_end,
            )
        ]
        precision = fit_ledoit_wolf_precision(train_profiles)
        validation_scores = mahalanobis_similarity(frames["validation"][list(PAIR_FEATURES)], precision)
        metrics, _ = evaluate_ranking_queries(frames["validation"], validation_scores, algorithm="mahalanobis")
        mahalanobis_validation.append((metrics["ndcg_at_5"], prior_strength, precision))
    _, mahalanobis_prior, mahalanobis_precision = max(mahalanobis_validation, key=lambda item: item[0])
    mahalanobis_ranking, mahalanobis_preprocessor = dataset_for(mahalanobis_prior)
    mahalanobis_splits = split_ranking_dataset(mahalanobis_ranking)

    evaluations: list[dict[str, Any]] = []
    per_query_outputs: dict[tuple[str, str], pd.DataFrame] = {}
    scores_by_algorithm: dict[tuple[str, str], np.ndarray] = {}
    for split_name in ("validation", "test"):
        reference_frame = reference_splits[split_name]
        legacy_scores = score_legacy_weighted_euclidean(reference_frame, base)
        metrics, per_query = evaluate_ranking_queries(
            reference_frame, legacy_scores, algorithm="weighted_euclidean_v1"
        )
        evaluations.append({"algorithm": "weighted_euclidean_v1", "split": split_name, **metrics})
        per_query_outputs[("weighted_euclidean_v1", split_name)] = per_query
        scores_by_algorithm[("weighted_euclidean_v1", split_name)] = legacy_scores

        mahal_frame = mahalanobis_splits[split_name]
        mahal_scores = mahalanobis_similarity(mahal_frame[list(PAIR_FEATURES)], mahalanobis_precision)
        metrics, per_query = evaluate_ranking_queries(mahal_frame, mahal_scores, algorithm="mahalanobis")
        evaluations.append({"algorithm": "mahalanobis", "split": split_name, **metrics})
        per_query_outputs[("mahalanobis", split_name)] = per_query
        scores_by_algorithm[("mahalanobis", split_name)] = mahal_scores

        lambda_frame = selected_splits[split_name]
        lambda_scores = model_scores(selected_model, lambda_frame)
        metrics, per_query = evaluate_ranking_queries(lambda_frame, lambda_scores, algorithm="lambdamart")
        evaluations.append({"algorithm": "lambdamart", "split": split_name, **metrics})
        per_query_outputs[("lambdamart", split_name)] = per_query
        scores_by_algorithm[("lambdamart", split_name)] = lambda_scores

    evaluation = pd.DataFrame(evaluations)
    test_baseline = evaluation[
        evaluation["algorithm"].eq("weighted_euclidean_v1") & evaluation["split"].eq("test")
    ].iloc[0].to_dict()

    latency_frames = [group for _, group in selected_splits["test"].groupby("query_id", sort=True)]
    lambda_latency = benchmark_query_latency(
        lambda frame: model_scores(selected_model, frame),
        latency_frames,
        repeats=3,
    )
    mahal_latency_frames = [group for _, group in mahalanobis_splits["test"].groupby("query_id", sort=True)]
    mahal_latency = benchmark_query_latency(
        lambda frame: mahalanobis_similarity(frame[list(PAIR_FEATURES)], mahalanobis_precision),
        mahal_latency_frames,
        repeats=3,
    )

    gates: dict[str, Any] = {}
    for algorithm, latency in (("lambdamart", lambda_latency), ("mahalanobis", mahal_latency)):
        challenger = evaluation[
            evaluation["algorithm"].eq(algorithm) & evaluation["split"].eq("test")
        ].iloc[0].to_dict()
        interval = paired_query_bootstrap(
            per_query_outputs[(algorithm, "test")],
            per_query_outputs[("weighted_euclidean_v1", "test")],
        )
        passed, checks = _promotion_gate(challenger, test_baseline, interval, latency)
        gates[algorithm] = {"passed": passed, "checks": checks, "bootstrap": interval, "latency": latency}

    if gates["lambdamart"]["passed"]:
        selected_algorithm = "lambdamart"
    elif gates["mahalanobis"]["passed"]:
        selected_algorithm = "mahalanobis"
    else:
        selected_algorithm = "weighted_euclidean_v1"

    if selected_algorithm == "lambdamart":
        best_iterations = [
            int(getattr(model, "best_iteration", selected_parameters["n_estimators"] - 1)) + 1
            for model in fitted_by_seed.values()
        ]
        final_estimators = max(1, int(np.median(best_iterations)))
        train_validation = pd.concat(
            [selected_splits["train"], selected_splits["validation"]], ignore_index=True
        ).sort_values(["query_id", "candidate_player_id"], kind="stable")
        final_model = _fit_final_lambdamart(
            train_validation,
            parameters={key: value for key, value in selected_parameters.items() if key != "n_estimators"},
            seed=median_seed,
            n_estimators=final_estimators,
        )
        final_preprocessor = selected_preprocessor
        final_precision = None
        final_parameters = {
            **selected_parameters,
            "n_estimators": final_estimators,
            "prior_strength": selected_prior,
            "seed": median_seed,
        }
    elif selected_algorithm == "mahalanobis":
        final_model = None
        final_preprocessor = mahalanobis_preprocessor
        normalized = final_preprocessor.transform(base)
        fit_profiles = normalized[
            normalized["season"].between(
                DEFAULT_RECOMMENDATION_SPLIT_POLICY.train_start,
                DEFAULT_RECOMMENDATION_SPLIT_POLICY.validation,
            )
        ]
        final_precision = fit_ledoit_wolf_precision(fit_profiles)
        final_parameters = {"prior_strength": mahalanobis_prior}
    else:
        final_model = None
        _, final_preprocessor = dataset_for(DEFAULT_SHRINKAGE_PRIOR_STRENGTH)
        final_precision = None
        final_parameters = {"prior_strength": DEFAULT_SHRINKAGE_PRIOR_STRENGTH}

    artifact = RecommendationRankerArtifact(
        algorithm=selected_algorithm,
        preprocessor=final_preprocessor,
        model=final_model,
        precision=final_precision,
        version=RECOMMENDATION_RANKER_VERSION,
        split_policy=asdict(DEFAULT_RECOMMENDATION_SPLIT_POLICY),
        parameters=final_parameters,
        metrics={
            "selection_gate": gates,
            "evaluation": evaluation.to_dict("records"),
            "seed_evaluation": seed_evaluation.to_dict("records"),
            "baseline_validation": baseline_validation,
            "proxy_ground_truth": "next-season graded profile similarity",
        },
    )
    saved_artifact = save_recommendation_ranker_artifact(artifact, artifact_path)

    pd.DataFrame(trial_records).to_parquet(report_root / "lambdamart_optuna_trials.parquet", index=False)
    seed_evaluation.to_parquet(report_root / "lambdamart_seed_evaluation.parquet", index=False)
    evaluation.to_parquet(report_root / "recommendation_ranker_evaluation.parquet", index=False)
    with (report_root / "selection_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "selected_algorithm": selected_algorithm,
                "ranker_version": RECOMMENDATION_RANKER_VERSION,
                "parameters": final_parameters,
                "gates": gates,
                "split_policy": asdict(DEFAULT_RECOMMENDATION_SPLIT_POLICY),
            },
            handle,
            indent=2,
        )

    if selected_algorithm == "lambdamart" and final_model is not None:
        importance = pd.DataFrame(
            {
                "feature": list(PAIR_FEATURES),
                "importance": final_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)
        importance.to_csv(report_root / "lambdamart_feature_importance.csv", index=False)

    if mlflow_module is not None:
        with mlflow_module.start_run(run_name="recommendation_ranker_selection"):
            log_params_flat(
                mlflow_module,
                {
                    "selected_algorithm": selected_algorithm,
                    "ranker_version": RECOMMENDATION_RANKER_VERSION,
                    **final_parameters,
                },
            )
            selected_test = evaluation[
                evaluation["algorithm"].eq(selected_algorithm) & evaluation["split"].eq("test")
            ]
            if not selected_test.empty:
                log_metrics_flat(
                    mlflow_module,
                    {
                        key: float(selected_test.iloc[0][key])
                        for key in ("ndcg_at_5", "recall_at_5", "hit_rate_at_5", "mrr", "coverage_at_5")
                    },
                    prefix="test",
                )
            mlflow_module.log_artifact(str(saved_artifact), artifact_path="selected_ranker")
            mlflow_module.log_artifacts(str(report_root), artifact_path="evaluation")

    print(evaluation.to_string(index=False))
    print(json.dumps({"selected_algorithm": selected_algorithm, "gates": gates}, indent=2))
    print(f"Saved recommendation ranker: {saved_artifact}")
    return artifact, evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune and select the player recommendation ranker.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--artifact-dir", default="artifacts")
    parser.add_argument("--report-dir", default="reports/recommendation_ranking")
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--no-mlflow", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_recommendation_experiment(
        data_dir=args.data_dir,
        artifact_dir=args.artifact_dir,
        report_dir=args.report_dir,
        trials=args.trials,
        use_mlflow=not args.no_mlflow,
    )


if __name__ == "__main__":
    main()
