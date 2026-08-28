from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import gzip
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .balldontlie import (
    BallDontLieClient,
    BallDontLieFetchResult,
    fetch_missing_balldontlie_game_logs,
)
from .cleaning import normalize_team_abbreviation
from .features_long_term import build_long_term_inference
from .features_performance import build_performance_inference
from .loaders import (
    DataPaths,
    load_player_game_logs,
    load_player_season_stats,
    load_players,
)


BALLDONTLIE_BRONZE_DIR = "balldontlie"
BALLDONTLIE_STATE_FILE = "ingestion_state.json"
SILVER_GAME_LOG_FILE = "player_game_logs.parquet"
SHORT_TERM_INFERENCE_FILE = "short_term_inference_latest.parquet"
LONG_TERM_INFERENCE_FILE = "long_term_player_forecast_inference_latest.parquet"
UPSERT_KEY_COLUMNS = ["player_id", "game_date", "team_id"]


@dataclass(frozen=True)
class GameLogUpsertSummary:
    """Counts describing one canonical Silver game-log upsert."""

    existing_rows: int
    incoming_rows: int
    unmatched_rows: int
    output_rows: int
    inserted_rows: int
    replaced_rows: int


@dataclass(frozen=True)
class IncrementalPipelineResult:
    """Artifacts and row counts produced by one incremental API update."""

    fetch: BallDontLieFetchResult
    upsert: GameLogUpsertSummary
    bronze_snapshot_path: Path | None
    unmatched_players_path: Path | None
    silver_game_logs_path: Path
    gold_output_paths: dict[str, Path]
    checkpoint_path: Path


def _utc_timestamp(value: datetime | None = None) -> datetime:
    """Return a timezone-aware UTC timestamp for snapshot names and metadata."""
    resolved = value or datetime.now(timezone.utc)
    return resolved.astimezone(timezone.utc)


def _atomic_write_parquet(dataframe: pd.DataFrame, path: Path) -> None:
    """Write a parquet file through a temporary path before replacing the target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    dataframe.to_parquet(temporary_path, index=False)
    temporary_path.replace(path)


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    """Write JSON metadata atomically so interrupted jobs do not corrupt state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def load_balldontlie_checkpoint(paths: DataPaths) -> date | None:
    """Return the last successfully processed API end date, when available."""
    state_path = paths.bronze_dir / BALLDONTLIE_BRONZE_DIR / BALLDONTLIE_STATE_FILE
    if not state_path.exists():
        return None
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    value = payload.get("last_successful_end_date")
    return pd.Timestamp(value).date() if value else None


def save_balldontlie_bronze_snapshot(
    paths: DataPaths,
    fetch_result: BallDontLieFetchResult,
    fetched_at: datetime | None = None,
) -> Path | None:
    """Persist an immutable gzip JSON snapshot of the API rows in the Bronze layer."""
    if fetch_result.date_range is None:
        return None

    timestamp = _utc_timestamp(fetched_at)
    snapshot_dir = (
        paths.bronze_dir
        / BALLDONTLIE_BRONZE_DIR
        / "player_game_stats"
        / f"year={timestamp:%Y}"
        / f"month={timestamp:%m}"
    )
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"stats_{timestamp:%Y%m%dT%H%M%S%fZ}.json.gz"
    temporary_path = snapshot_path.with_name(f".{snapshot_path.name}.tmp")
    payload = {
        "source": "balldontlie",
        "fetched_at": timestamp.isoformat(),
        "start_date": fetch_result.date_range.start_date.isoformat(),
        "end_date": fetch_result.date_range.end_date.isoformat(),
        "row_count": fetch_result.raw_row_count,
        "data": fetch_result.raw_stats,
    }
    with gzip.open(temporary_path, "wt", encoding="utf-8") as file:
        json.dump(payload, file, default=str)
    temporary_path.replace(snapshot_path)
    return snapshot_path


def _canonical_upsert_keys(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return normalized cross-source keys for one player, game date, and team."""
    keys = pd.DataFrame(index=dataframe.index)
    keys["player_id"] = pd.to_numeric(dataframe["player_id"], errors="coerce").astype("Int64")
    keys["game_date"] = pd.to_datetime(dataframe["game_date"], errors="coerce").dt.normalize()
    keys["team_id"] = dataframe["team_id"].map(normalize_team_abbreviation)
    return keys


def recompute_rest_days(game_logs: pd.DataFrame) -> pd.DataFrame:
    """Recompute rest days over the complete canonical player timeline."""
    result = game_logs.copy()
    result["game_date"] = pd.to_datetime(result["game_date"], errors="coerce")
    result = result.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)
    result["rest_days"] = result.groupby("player_id")["game_date"].diff().dt.days.astype("float64")
    return result


def upsert_player_game_logs(
    clean_game_logs: pd.DataFrame,
    incoming_game_logs: pd.DataFrame,
) -> tuple[pd.DataFrame, GameLogUpsertSummary]:
    """Upsert mapped API rows into canonical Silver history using cross-source keys."""
    missing_existing = sorted(set(UPSERT_KEY_COLUMNS) - set(clean_game_logs.columns))
    missing_incoming = sorted(set(UPSERT_KEY_COLUMNS) - set(incoming_game_logs.columns))
    if missing_existing:
        raise KeyError(f"clean_game_logs is missing upsert columns: {missing_existing}")
    if missing_incoming:
        raise KeyError(f"incoming_game_logs is missing upsert columns: {missing_incoming}")

    existing = clean_game_logs.copy()
    incoming = incoming_game_logs.copy()
    unmatched_rows = int(incoming["player_id"].isna().sum())
    incoming = incoming.dropna(subset=UPSERT_KEY_COLUMNS).copy()

    if incoming.empty:
        summary = GameLogUpsertSummary(
            existing_rows=len(existing),
            incoming_rows=0,
            unmatched_rows=unmatched_rows,
            output_rows=len(existing),
            inserted_rows=0,
            replaced_rows=0,
        )
        return existing.reset_index(drop=True), summary

    existing_keys = _canonical_upsert_keys(existing)
    incoming_keys = _canonical_upsert_keys(incoming)
    existing["_upsert_key"] = pd.MultiIndex.from_frame(existing_keys).to_flat_index()
    incoming["_upsert_key"] = pd.MultiIndex.from_frame(incoming_keys).to_flat_index()
    existing = existing.drop_duplicates("_upsert_key", keep="last")
    incoming = incoming.drop_duplicates("_upsert_key", keep="last")

    existing_game_ids = existing.set_index("_upsert_key")["game_id"]
    overlap_mask = incoming["_upsert_key"].isin(existing_game_ids.index)
    incoming.loc[overlap_mask, "game_id"] = incoming.loc[overlap_mask, "_upsert_key"].map(
        existing_game_ids
    )

    replaced_rows = int(overlap_mask.sum())
    inserted_rows = int(len(incoming) - replaced_rows)
    concat_frames = [frame.dropna(axis=1, how="all") for frame in [existing, incoming]]
    combined = pd.concat(concat_frames, ignore_index=True, sort=False)
    combined = combined.drop_duplicates("_upsert_key", keep="last").drop(columns="_upsert_key")
    combined = recompute_rest_days(combined)

    preferred_columns = list(clean_game_logs.columns)
    appended_columns = [column for column in combined.columns if column not in preferred_columns]
    combined = combined[preferred_columns + appended_columns]
    summary = GameLogUpsertSummary(
        existing_rows=len(clean_game_logs),
        incoming_rows=len(incoming),
        unmatched_rows=unmatched_rows,
        output_rows=len(combined),
        inserted_rows=inserted_rows,
        replaced_rows=replaced_rows,
    )
    return combined.reset_index(drop=True), summary


def rebuild_latest_inference_datasets(
    paths: DataPaths,
    game_logs: pd.DataFrame | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
    """Rebuild label-free Gold inference datasets from the complete Silver history."""
    resolved_logs = game_logs if game_logs is not None else load_player_game_logs(paths)
    players = load_players(paths)
    season_stats = load_player_season_stats(paths)

    outputs = {
        "short_term_inference_latest": build_performance_inference(resolved_logs),
        "long_term_player_forecast_inference_latest": build_long_term_inference(
            resolved_logs,
            players,
            season_stats,
        ),
    }
    output_paths = {
        "short_term_inference_latest": paths.gold_dir / SHORT_TERM_INFERENCE_FILE,
        "long_term_player_forecast_inference_latest": paths.gold_dir / LONG_TERM_INFERENCE_FILE,
    }
    for name, dataframe in outputs.items():
        _atomic_write_parquet(dataframe, output_paths[name])
    return outputs, output_paths


def run_balldontlie_incremental_pipeline(
    paths: DataPaths,
    end_date: date | str | pd.Timestamp | None = None,
    overlap_days: int = 2,
    player_id_overrides: dict[int, Any] | None = None,
    client: BallDontLieClient | None = None,
) -> IncrementalPipelineResult:
    """Fetch missing stats, update Silver, and rebuild current Gold inference data."""
    if overlap_days < 0:
        raise ValueError("overlap_days cannot be negative.")

    clean_game_logs = load_player_game_logs(paths)
    players = load_players(paths)
    checkpoint = load_balldontlie_checkpoint(paths)
    explicit_start = (
        checkpoint + timedelta(days=1 - overlap_days)
        if checkpoint is not None
        else None
    )
    fetch_result = fetch_missing_balldontlie_game_logs(
        clean_game_logs=clean_game_logs,
        players=players,
        start_date=explicit_start,
        end_date=end_date,
        overlap_days=overlap_days,
        player_id_overrides=player_id_overrides,
        client=client,
    )
    snapshot_path = save_balldontlie_bronze_snapshot(paths, fetch_result)

    updated_logs, upsert_summary = upsert_player_game_logs(
        clean_game_logs,
        fetch_result.game_logs,
    )
    silver_path = paths.silver_dir / SILVER_GAME_LOG_FILE
    if upsert_summary.incoming_rows > 0:
        _atomic_write_parquet(updated_logs, silver_path)

    unmatched_path: Path | None = None
    if not fetch_result.unmatched_players.empty:
        unmatched_path = (
            paths.bronze_dir
            / BALLDONTLIE_BRONZE_DIR
            / "unmatched_players_latest.parquet"
        )
        _atomic_write_parquet(fetch_result.unmatched_players, unmatched_path)

    _, gold_paths = rebuild_latest_inference_datasets(paths, game_logs=updated_logs)

    checkpoint_path = paths.bronze_dir / BALLDONTLIE_BRONZE_DIR / BALLDONTLIE_STATE_FILE
    if fetch_result.date_range is not None:
        _atomic_write_json(
            {
                "source": "balldontlie",
                "last_successful_end_date": fetch_result.date_range.end_date.isoformat(),
                "updated_at": _utc_timestamp().isoformat(),
                "raw_row_count": fetch_result.raw_row_count,
                "mapped_row_count": upsert_summary.incoming_rows,
                "unmatched_row_count": upsert_summary.unmatched_rows,
            },
            checkpoint_path,
        )

    return IncrementalPipelineResult(
        fetch=fetch_result,
        upsert=upsert_summary,
        bronze_snapshot_path=snapshot_path,
        unmatched_players_path=unmatched_path,
        silver_game_logs_path=silver_path,
        gold_output_paths=gold_paths,
        checkpoint_path=checkpoint_path,
    )
