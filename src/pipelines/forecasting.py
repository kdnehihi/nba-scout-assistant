from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from src.config.long_term_config import LongTermTaskConfig
from src.dataset.cleaning import normalize_name_key
from src.dataset.loaders import (
    load_long_term_inference,
    load_players,
    load_short_term_inference,
    resolve_data_paths,
)
from src.dataset.sequence import make_lstm_delta_inference_window
from src.evaluation.evaluate_long_term import predict_mlp_long_term, predict_sklearn_long_term
from src.pipelines.artifacts import LongTermModelArtifact, ShortTermModelArtifact


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _filter_player_rows(
    df: pd.DataFrame,
    player_id: int | str | None = None,
    player_name: str | None = None,
) -> pd.DataFrame:
    # Select player rows by ID when available, otherwise by normalized player name.
    """Return rows for one requested player."""
    if player_id is not None and "player_id" in df.columns:
        return df[df["player_id"].astype(str).eq(str(player_id))].copy()
    if player_name is not None and "player_name" in df.columns:
        player_key = normalize_name_key(player_name)
        name_keys = (
            df["player_name_key"]
            if "player_name_key" in df.columns
            else df["player_name"].map(normalize_name_key)
        )
        mask = name_keys.eq(player_key)
        if not mask.any() and player_key:
            mask = name_keys.fillna("").str.contains(player_key, regex=False)
        return df[mask].copy()
    raise ValueError("player_id or player_name is required.")


def _select_short_term_anchor_row(
    df: pd.DataFrame,
    player_id: int | str | None = None,
    player_name: str | None = None,
    season: str | None = None,
    as_of_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    # Select the ordered player-season history ending at the requested game/date.
    """Return player game rows up to the selected short-term anchor."""
    rows = _filter_player_rows(df, player_id=player_id, player_name=player_name)
    if season is not None:
        rows = rows[rows["season"].eq(season)].copy()
    if rows.empty:
        raise ValueError("No short-term rows matched the requested player and season.")

    rows["as_of_date"] = pd.to_datetime(rows["as_of_date"], errors="coerce")
    rows = rows.dropna(subset=["as_of_date"]).sort_values(["as_of_date", "game_id"]).copy()
    if as_of_date is not None:
        cutoff = pd.to_datetime(as_of_date)
        rows = rows[rows["as_of_date"].le(cutoff)].copy()
    if rows.empty:
        raise ValueError("No short-term rows were available at or before the requested date.")
    return rows


def predict_short_term_task(
    performance_df: pd.DataFrame,
    artifact: ShortTermModelArtifact,
    player_id: int | str | None = None,
    player_name: str | None = None,
    season: str | None = None,
    as_of_date: str | pd.Timestamp | None = None,
    device: str = DEVICE,
) -> dict[str, Any]:
    # Predict one next-five-game short-term stat from the latest available sequence.
    """Return one short-term prediction for a player request."""
    config = artifact.task_config
    rows = _select_short_term_anchor_row(
        performance_df,
        player_id=player_id,
        player_name=player_name,
        season=season,
        as_of_date=as_of_date,
    )

    X_seq, static, baseline, anchor = make_lstm_delta_inference_window(rows, config)
    X_seq_scaled = artifact.seq_scaler.transform(X_seq.reshape(-1, X_seq.shape[-1])).reshape(X_seq.shape).astype("float32")
    X_static_scaled = artifact.static_scaler.transform(static).astype("float32")

    artifact.model.eval()
    with torch.no_grad():
        pred_delta = (
            artifact.model(
                torch.tensor(X_seq_scaled, dtype=torch.float32, device=device),
                torch.tensor(X_static_scaled, dtype=torch.float32, device=device),
            )
            .detach()
            .cpu()
            .numpy()
        )

    if artifact.y_scaler is not None:
        pred_delta = artifact.y_scaler.inverse_transform(pred_delta.reshape(-1, 1)).reshape(-1)

    prediction = baseline + float(pred_delta[0])
    return {
        "task": artifact.task,
        "player_id": anchor.get("player_id"),
        "player_name": anchor.get("player_name"),
        "season": anchor.get("season"),
        "as_of_date": anchor.get("as_of_date"),
        "baseline": baseline,
        "prediction": prediction,
    }


def predict_short_term_tasks(
    performance_df: pd.DataFrame,
    artifacts: dict[str, ShortTermModelArtifact],
    tasks: list[str] | tuple[str, ...] | None = None,
    **request: Any,
) -> dict[str, dict[str, Any]]:
    # Predict one or more short-term tasks from cached loaded artifacts.
    """Return short-term predictions keyed by task."""
    selected_tasks = tuple(tasks or artifacts.keys())
    return {
        task: predict_short_term_task(
            performance_df=performance_df,
            artifact=artifacts[task],
            **request,
        )
        for task in selected_tasks
    }


def load_short_term_prediction_data(data_dir: Path | str = "data") -> pd.DataFrame:
    # Load clean short-term gold data for inference window selection.
    """Load short-term prediction source data from the gold layer."""
    paths = resolve_data_paths(data_dir)
    performance = load_short_term_inference(paths)
    if "player_name" not in performance.columns and "player_id" in performance.columns:
        players = load_players(paths)
        player_lookup = players[["player_id", "player_name"]].drop_duplicates("player_id")
        performance = performance.merge(player_lookup, on="player_id", how="left")
    if "player_name" in performance.columns and "player_name_key" not in performance.columns:
        performance["player_name_key"] = performance["player_name"].map(normalize_name_key)
    return performance


def _select_long_term_anchor_row(
    df: pd.DataFrame,
    player_id: int | str | None = None,
    player_name: str | None = None,
    anchor_season: str | None = None,
) -> pd.Series:
    # Select one long-term anchor row for a player request.
    """Return the selected long-term anchor row."""
    rows = _filter_player_rows(df, player_id=player_id, player_name=player_name)
    if anchor_season is not None:
        rows = rows[rows["anchor_season"].eq(anchor_season)].copy()
    if rows.empty:
        raise ValueError("No long-term rows matched the requested player and anchor season.")
    return rows.sort_values(["anchor_season_start_year"]).iloc[-1]


def _dense_float32(values: Any) -> np.ndarray:
    # Convert sklearn preprocessor output into torch-compatible dense float32 arrays.
    """Return a dense float32 numpy array."""
    if sparse.issparse(values):
        values = values.toarray()
    return np.asarray(values, dtype="float32")


def _clip_probability(prediction: np.ndarray) -> np.ndarray:
    # Keep served classification probabilities inside the valid probability range.
    """Return probability predictions clipped to [0, 1]."""
    return np.clip(prediction.astype("float64"), 0.0, 1.0)


def predict_long_term_task(
    long_term_df: pd.DataFrame,
    artifact: LongTermModelArtifact,
    player_id: int | str | None = None,
    player_name: str | None = None,
    anchor_season: str | None = None,
    device: str = DEVICE,
) -> dict[str, Any]:
    # Predict one long-term task/horizon from the selected anchor-season row.
    """Return one long-term prediction for a player request."""
    anchor = _select_long_term_anchor_row(
        long_term_df,
        player_id=player_id,
        player_name=player_name,
        anchor_season=anchor_season,
    )
    missing = sorted(set(artifact.feature_cols) - set(long_term_df.columns))
    if missing:
        raise KeyError(f"Missing long-term prediction columns: {missing}")

    X = anchor.to_frame().T[artifact.feature_cols]
    task_config: LongTermTaskConfig = artifact.task_config

    if artifact.model_family in {"random_forest", "ridge", "logistic"}:
        prediction = predict_sklearn_long_term(artifact.model, X, task_config)
    elif artifact.model_family == "mlp":
        if artifact.preprocessor is None:
            raise ValueError("MLP long-term artifact is missing its preprocessor.")
        X_model = _dense_float32(artifact.preprocessor.transform(X))
        prediction = predict_mlp_long_term(
            artifact.model,
            X_model,
            batch_size=int(task_config.model_params["batch_size"]),
            task_config=task_config,
            device=device,
        )
        if artifact.target_scaler is not None and task_config.task_type == "regression":
            prediction = artifact.target_scaler.inverse_transform(prediction.reshape(-1, 1)).reshape(-1)
    else:
        raise ValueError(f"Unsupported long-term model family: {artifact.model_family}")

    if task_config.task_type == "classification":
        prediction = _clip_probability(prediction)

    return {
        "task": artifact.task,
        "horizon": artifact.horizon,
        "model_family": artifact.model_family,
        "target": task_config.target_col,
        "player_id": anchor.get("player_id"),
        "player_name": anchor.get("player_name"),
        "anchor_season": anchor.get("anchor_season"),
        "prediction": float(prediction[0]),
    }


def predict_long_term_tasks(
    long_term_df: pd.DataFrame,
    artifacts: dict[tuple[str, int], LongTermModelArtifact],
    tasks: list[str] | tuple[str, ...] | None = None,
    horizons: list[int] | tuple[int, ...] | None = None,
    **request: Any,
) -> dict[str, dict[int, dict[str, Any]]]:
    # Predict selected long-term tasks and horizons from cached artifacts.
    """Return long-term predictions keyed by task and horizon."""
    task_filter = set(tasks) if tasks is not None else None
    horizon_filter = set(horizons) if horizons is not None else None
    output: dict[str, dict[int, dict[str, Any]]] = {}

    for (task, horizon), artifact in artifacts.items():
        if task_filter is not None and task not in task_filter:
            continue
        if horizon_filter is not None and horizon not in horizon_filter:
            continue
        output.setdefault(task, {})[horizon] = predict_long_term_task(
            long_term_df=long_term_df,
            artifact=artifact,
            **request,
        )
    if "active_probability" in output:
        previous_probability: float | None = None
        for horizon in sorted(output["active_probability"]):
            current = float(output["active_probability"][horizon]["prediction"])
            if previous_probability is not None:
                current = min(current, previous_probability)
                output["active_probability"][horizon]["prediction"] = current
            previous_probability = current
    return output


def load_long_term_prediction_data(data_dir: Path | str = "data") -> pd.DataFrame:
    # Load long-term inference anchors, not target-complete training anchors.
    """Load long-term prediction source data from the gold layer."""
    return load_long_term_inference(resolve_data_paths(data_dir), build_if_missing=True)
