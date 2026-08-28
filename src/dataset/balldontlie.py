from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import os
import time
from typing import Any, Mapping

import httpx
import numpy as np
import pandas as pd

from .cleaning import normalize_name_key, normalize_team_abbreviation


BALLDONTLIE_BASE_URL = "https://api.balldontlie.io/v1"
BALLDONTLIE_API_KEY_ENV = "BALLDONTLIE_API_KEY"

CANONICAL_GAME_LOG_COLUMNS = [
    "player_id",
    "player_name",
    "game_date",
    "game_id",
    "season",
    "season_end_year",
    "season_label",
    "team_id",
    "team_abbreviation",
    "opponent",
    "home_away",
    "minutes",
    "points",
    "assists",
    "rebounds",
    "offensive_rebounds",
    "defensive_rebounds",
    "steals",
    "blocks",
    "personal_fouls",
    "turnovers",
    "fgm",
    "fga",
    "fg3m",
    "fg3a",
    "ftm",
    "fta",
    "true_shooting_pct",
    "rest_days",
]

SOURCE_AUDIT_COLUMNS = [
    "source",
    "source_stat_id",
    "source_game_id",
    "source_player_id",
    "source_team_id",
    "source_status",
    "fetched_at",
]


@dataclass(frozen=True)
class BallDontLieConfig:
    """HTTP and retry settings for the BALLDONTLIE NBA API."""

    base_url: str = BALLDONTLIE_BASE_URL
    timeout_seconds: float = 30.0
    per_page: int = 100
    max_retries: int = 4
    retry_backoff_seconds: float = 1.0
    max_pages: int = 10_000

    def __post_init__(self) -> None:
        if not 1 <= self.per_page <= 100:
            raise ValueError("per_page must be between 1 and 100.")
        if self.max_retries < 1:
            raise ValueError("max_retries must be at least 1.")
        if self.max_pages < 1:
            raise ValueError("max_pages must be at least 1.")


@dataclass(frozen=True)
class MissingDateRange:
    """Inclusive date range absent from the current canonical game logs."""

    start_date: date
    end_date: date


@dataclass(frozen=True)
class BallDontLieFetchResult:
    """Normalized incremental game logs plus player rows requiring ID review."""

    date_range: MissingDateRange | None
    raw_stats: list[dict[str, Any]]
    game_logs: pd.DataFrame
    unmatched_players: pd.DataFrame
    raw_row_count: int


class BallDontLieClient:
    """Small synchronous client for cursor-paginated BALLDONTLIE endpoints."""

    def __init__(
        self,
        api_key: str | None = None,
        config: BallDontLieConfig | None = None,
        http_client: httpx.Client | None = None,
    ) -> None:
        resolved_key = (api_key or os.getenv(BALLDONTLIE_API_KEY_ENV, "")).strip()
        if not resolved_key:
            raise ValueError(
                f"Missing API key. Set {BALLDONTLIE_API_KEY_ENV} or pass api_key explicitly."
            )
        self.api_key = resolved_key
        self.config = config or BallDontLieConfig()
        self._owns_client = http_client is None
        self.http_client = http_client or httpx.Client(
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
        )

    def close(self) -> None:
        """Close the internally created HTTP connection pool."""
        if self._owns_client:
            self.http_client.close()

    def __enter__(self) -> BallDontLieClient:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _get_json(self, path: str, params: Mapping[str, Any]) -> dict[str, Any]:
        """Request one page and retry rate-limit or transient server failures."""
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                response = self.http_client.get(
                    path,
                    params=dict(params),
                    headers={"Authorization": self.api_key},
                )
                if response.status_code in {401, 403}:
                    raise PermissionError(
                        "BALLDONTLIE rejected the request. Check the API key and confirm "
                        "the account tier includes Game Player Stats."
                    )
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt + 1 == self.config.max_retries:
                        response.raise_for_status()
                    retry_after = response.headers.get("Retry-After")
                    wait_seconds = (
                        float(retry_after)
                        if retry_after and retry_after.replace(".", "", 1).isdigit()
                        else self.config.retry_backoff_seconds * (2**attempt)
                    )
                    time.sleep(wait_seconds)
                    continue
                response.raise_for_status()
                payload = response.json()
                if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
                    raise ValueError("BALLDONTLIE response must contain a list under 'data'.")
                return payload
            except (httpx.TimeoutException, httpx.NetworkError) as error:
                last_error = error
                if attempt + 1 == self.config.max_retries:
                    raise
                time.sleep(self.config.retry_backoff_seconds * (2**attempt))
        raise RuntimeError("BALLDONTLIE request failed after retries.") from last_error

    def fetch_paginated(
        self,
        path: str,
        params: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Fetch every cursor page for one list endpoint."""
        query = dict(params or {})
        query["per_page"] = self.config.per_page
        rows: list[dict[str, Any]] = []
        seen_cursors: set[str] = set()

        for _ in range(self.config.max_pages):
            payload = self._get_json(path, query)
            rows.extend(row for row in payload["data"] if isinstance(row, dict))
            next_cursor = (payload.get("meta") or {}).get("next_cursor")
            if next_cursor in {None, ""}:
                return rows
            cursor_key = str(next_cursor)
            if cursor_key in seen_cursors:
                raise RuntimeError(f"BALLDONTLIE returned a repeated cursor: {next_cursor}")
            seen_cursors.add(cursor_key)
            query["cursor"] = next_cursor

        raise RuntimeError(
            f"BALLDONTLIE pagination exceeded max_pages={self.config.max_pages}."
        )

    def fetch_player_game_stats(
        self,
        start_date: date,
        end_date: date,
        postseason: bool = False,
    ) -> list[dict[str, Any]]:
        """Fetch full-game player box scores for an inclusive date range."""
        if start_date > end_date:
            return []
        return self.fetch_paginated(
            "/stats",
            params={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "postseason": str(postseason).lower(),
                "period": 0,
            },
        )


def determine_missing_date_range(
    clean_game_logs: pd.DataFrame,
    end_date: date | str | pd.Timestamp | None = None,
    overlap_days: int = 0,
) -> MissingDateRange | None:
    """Find the API range after the latest date in the canonical clean game logs."""
    if overlap_days < 0:
        raise ValueError("overlap_days cannot be negative.")
    if "game_date" not in clean_game_logs.columns:
        raise KeyError("clean_game_logs is missing required column: game_date")

    parsed_dates = pd.to_datetime(clean_game_logs["game_date"], errors="coerce").dropna()
    if parsed_dates.empty:
        raise ValueError("clean_game_logs has no valid game_date values; pass a seeded dataset first.")

    resolved_end = (
        datetime.now(timezone.utc).date()
        if end_date is None
        else pd.Timestamp(end_date).date()
    )
    strict_start = parsed_dates.max().date() + timedelta(days=1)
    resolved_start = strict_start - timedelta(days=overlap_days)
    if resolved_start > resolved_end:
        return None
    return MissingDateRange(start_date=resolved_start, end_date=resolved_end)


def parse_balldontlie_minutes(value: object) -> float | None:
    """Convert BALLDONTLIE minute text such as '30:15' into decimal minutes."""
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if ":" not in text:
        try:
            return float(text)
        except ValueError:
            return None
    minute_text, second_text = text.split(":", 1)
    try:
        return float(minute_text) + float(second_text) / 60
    except ValueError:
        return None


def _canonical_player_lookup(
    clean_game_logs: pd.DataFrame,
    players: pd.DataFrame | None,
) -> dict[str, Any]:
    frames: list[pd.DataFrame] = []
    for frame in [players, clean_game_logs]:
        if frame is None or frame.empty:
            continue
        if {"player_id", "player_name"}.issubset(frame.columns):
            frames.append(frame[["player_id", "player_name"]].dropna().drop_duplicates())
    if not frames:
        return {}

    lookup = pd.concat(frames, ignore_index=True).drop_duplicates()
    lookup["player_name_key"] = lookup["player_name"].map(normalize_name_key)
    id_counts = lookup.groupby("player_name_key")["player_id"].nunique(dropna=True)
    unique_keys = set(id_counts[id_counts.eq(1)].index)
    lookup = lookup[lookup["player_name_key"].isin(unique_keys)]
    return (
        lookup.drop_duplicates("player_name_key")
        .set_index("player_name_key")["player_id"]
        .to_dict()
    )


def _api_player_name(player: Mapping[str, Any]) -> str:
    return " ".join(
        part.strip()
        for part in [str(player.get("first_name") or ""), str(player.get("last_name") or "")]
        if part.strip()
    )


def _season_label(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _source_team_lookup(raw_stats: list[dict[str, Any]]) -> dict[str, str]:
    """Collect BALLDONTLIE team IDs and canonical abbreviations from a response batch."""
    lookup: dict[str, str] = {}
    for row in raw_stats:
        containers = [row.get("team")]
        game = row.get("game") or {}
        if isinstance(game, Mapping):
            containers.extend([game.get("home_team"), game.get("visitor_team")])
        for team in containers:
            if not isinstance(team, Mapping):
                continue
            team_id = team.get("id")
            abbreviation = normalize_team_abbreviation(team.get("abbreviation"))
            if team_id is not None and abbreviation:
                lookup[str(team_id)] = abbreviation
    return lookup


def _normalize_stat_row(
    row: Mapping[str, Any],
    player_lookup: Mapping[str, Any],
    player_id_overrides: Mapping[int, Any],
    source_team_lookup: Mapping[str, str],
    fetched_at: pd.Timestamp,
) -> dict[str, Any] | None:
    player = row.get("player") or {}
    team = row.get("team") or {}
    game = row.get("game") or {}
    if not isinstance(player, Mapping) or not isinstance(team, Mapping) or not isinstance(game, Mapping):
        return None

    status_state = str(game.get("status_state") or "").lower()
    status = str(game.get("status") or "")
    if status_state and status_state != "final":
        return None
    if not status_state and status.lower() != "final":
        return None
    if bool(game.get("postseason", False)):
        return None

    player_name = _api_player_name(player)
    source_player_id = player.get("id")
    try:
        source_player_id_int = int(source_player_id)
    except (TypeError, ValueError):
        source_player_id_int = None
    canonical_player_id = player_id_overrides.get(source_player_id_int)
    if canonical_player_id is None:
        canonical_player_id = player_lookup.get(normalize_name_key(player_name))

    team_abbreviation = normalize_team_abbreviation(team.get("abbreviation"))
    source_team_id = team.get("id")
    home_team_id = game.get("home_team_id")
    visitor_team_id = game.get("visitor_team_id")
    is_home = str(source_team_id) == str(home_team_id)
    opponent_source_id = visitor_team_id if is_home else home_team_id

    home_abbreviation = normalize_team_abbreviation(
        (game.get("home_team") or {}).get("abbreviation")
        if isinstance(game.get("home_team"), Mapping)
        else None
    )
    visitor_abbreviation = normalize_team_abbreviation(
        (game.get("visitor_team") or {}).get("abbreviation")
        if isinstance(game.get("visitor_team"), Mapping)
        else None
    )
    opponent = visitor_abbreviation if is_home else home_abbreviation
    if opponent is None:
        opponent = source_team_lookup.get(str(opponent_source_id))

    game_date = pd.to_datetime(game.get("date"), errors="coerce")
    try:
        season_start_year = int(game.get("season"))
    except (TypeError, ValueError):
        season_start_year = None

    fga = pd.to_numeric(row.get("fga"), errors="coerce")
    fta = pd.to_numeric(row.get("fta"), errors="coerce")
    points = pd.to_numeric(row.get("pts"), errors="coerce")
    shooting_denominator = 2 * (fga + 0.44 * fta)
    true_shooting_pct = (
        float(points / shooting_denominator)
        if pd.notna(points) and pd.notna(shooting_denominator) and shooting_denominator > 0
        else np.nan
    )

    return {
        "player_id": canonical_player_id,
        "player_name": player_name,
        "game_date": game_date,
        "game_id": game.get("id"),
        "season": _season_label(season_start_year) if season_start_year is not None else None,
        "season_end_year": season_start_year + 1 if season_start_year is not None else None,
        "season_label": _season_label(season_start_year) if season_start_year is not None else None,
        "team_id": team_abbreviation,
        "team_abbreviation": team_abbreviation,
        "opponent": opponent,
        "home_away": "HOME" if is_home else "AWAY",
        "minutes": parse_balldontlie_minutes(row.get("min")),
        "points": points,
        "assists": row.get("ast"),
        "rebounds": row.get("reb"),
        "offensive_rebounds": row.get("oreb"),
        "defensive_rebounds": row.get("dreb"),
        "steals": row.get("stl"),
        "blocks": row.get("blk"),
        "personal_fouls": row.get("pf"),
        "turnovers": row.get("turnover"),
        "fgm": row.get("fgm"),
        "fga": fga,
        "fg3m": row.get("fg3m"),
        "fg3a": row.get("fg3a"),
        "ftm": row.get("ftm"),
        "fta": fta,
        "true_shooting_pct": true_shooting_pct,
        "rest_days": np.nan,
        "source": "balldontlie",
        "source_stat_id": row.get("id"),
        "source_game_id": game.get("id"),
        "source_player_id": source_player_id_int,
        "source_team_id": source_team_id,
        "source_status": status_state or status,
        "fetched_at": fetched_at,
    }


def _add_rest_days_from_history(
    new_game_logs: pd.DataFrame,
    clean_game_logs: pd.DataFrame,
) -> pd.DataFrame:
    result = new_game_logs.copy()
    mapped = result.dropna(subset=["player_id", "game_date"])
    if mapped.empty:
        return result

    history = clean_game_logs[["player_id", "game_date"]].copy()
    history["game_date"] = pd.to_datetime(history["game_date"], errors="coerce")
    current = mapped[["player_id", "game_date"]].copy()
    timeline = pd.concat([history, current], ignore_index=True).dropna().drop_duplicates()
    timeline = timeline.sort_values(["player_id", "game_date"])
    timeline["rest_days"] = timeline.groupby("player_id")["game_date"].diff().dt.days
    rest_lookup = timeline.set_index(["player_id", "game_date"])["rest_days"]
    keys = pd.MultiIndex.from_frame(result[["player_id", "game_date"]])
    result["rest_days"] = rest_lookup.reindex(keys).to_numpy(dtype="float64")
    return result


def normalize_balldontlie_game_stats(
    raw_stats: list[dict[str, Any]],
    clean_game_logs: pd.DataFrame,
    players: pd.DataFrame | None = None,
    player_id_overrides: Mapping[int, Any] | None = None,
    fetched_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Normalize BALLDONTLIE player stats into the canonical Silver game-log schema."""
    resolved_fetched_at = fetched_at or pd.Timestamp.now(tz="UTC")
    lookup = _canonical_player_lookup(clean_game_logs, players)
    overrides = dict(player_id_overrides or {})
    team_lookup = _source_team_lookup(raw_stats)
    normalized_rows = [
        normalized
        for row in raw_stats
        if (
            normalized := _normalize_stat_row(
                row,
                player_lookup=lookup,
                player_id_overrides=overrides,
                source_team_lookup=team_lookup,
                fetched_at=resolved_fetched_at,
            )
        )
        is not None
    ]
    if not normalized_rows:
        return pd.DataFrame(columns=CANONICAL_GAME_LOG_COLUMNS + SOURCE_AUDIT_COLUMNS)

    result = pd.DataFrame(normalized_rows)
    result = result.drop_duplicates("source_stat_id", keep="last")
    result["game_date"] = pd.to_datetime(result["game_date"], errors="coerce")
    result = result.dropna(subset=["player_name", "game_date", "game_id", "team_id"])
    result = _add_rest_days_from_history(result, clean_game_logs)

    integer_columns = [
        "player_id",
        "game_id",
        "season_end_year",
        "points",
        "assists",
        "rebounds",
        "offensive_rebounds",
        "defensive_rebounds",
        "steals",
        "blocks",
        "personal_fouls",
        "turnovers",
        "fgm",
        "fga",
        "fg3m",
        "fg3a",
        "ftm",
        "fta",
        "source_stat_id",
        "source_game_id",
        "source_player_id",
        "source_team_id",
    ]
    for column in integer_columns:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce").astype("Int64")
    result["minutes"] = pd.to_numeric(result["minutes"], errors="coerce")
    result["true_shooting_pct"] = pd.to_numeric(
        result["true_shooting_pct"], errors="coerce"
    )
    result["rest_days"] = pd.to_numeric(result["rest_days"], errors="coerce")
    return result[CANONICAL_GAME_LOG_COLUMNS + SOURCE_AUDIT_COLUMNS].sort_values(
        ["game_date", "game_id", "player_id"], na_position="last"
    ).reset_index(drop=True)


def unmatched_balldontlie_players(game_logs: pd.DataFrame) -> pd.DataFrame:
    """Return distinct API players that still need a canonical player ID mapping."""
    required = ["player_id", "source_player_id", "player_name"]
    if game_logs.empty or not set(required).issubset(game_logs.columns):
        return pd.DataFrame(columns=["source_player_id", "player_name"])
    return (
        game_logs.loc[game_logs["player_id"].isna(), ["source_player_id", "player_name"]]
        .drop_duplicates()
        .sort_values(["player_name", "source_player_id"])
        .reset_index(drop=True)
    )


def fetch_missing_balldontlie_game_logs(
    clean_game_logs: pd.DataFrame,
    players: pd.DataFrame | None = None,
    start_date: date | str | pd.Timestamp | None = None,
    end_date: date | str | pd.Timestamp | None = None,
    overlap_days: int = 0,
    player_id_overrides: Mapping[int, Any] | None = None,
    client: BallDontLieClient | None = None,
) -> BallDontLieFetchResult:
    """Fetch and normalize regular-season player stats missing after clean history."""
    if start_date is None:
        date_range = determine_missing_date_range(
            clean_game_logs,
            end_date=end_date,
            overlap_days=overlap_days,
        )
    else:
        resolved_start = pd.Timestamp(start_date).date()
        resolved_end = (
            datetime.now(timezone.utc).date()
            if end_date is None
            else pd.Timestamp(end_date).date()
        )
        date_range = (
            MissingDateRange(resolved_start, resolved_end)
            if resolved_start <= resolved_end
            else None
        )
    if date_range is None:
        empty = pd.DataFrame(columns=CANONICAL_GAME_LOG_COLUMNS + SOURCE_AUDIT_COLUMNS)
        return BallDontLieFetchResult(
            date_range=None,
            raw_stats=[],
            game_logs=empty,
            unmatched_players=unmatched_balldontlie_players(empty),
            raw_row_count=0,
        )

    owns_client = client is None
    resolved_client = client or BallDontLieClient()
    try:
        raw_stats = resolved_client.fetch_player_game_stats(
            start_date=date_range.start_date,
            end_date=date_range.end_date,
            postseason=False,
        )
    finally:
        if owns_client:
            resolved_client.close()

    normalized = normalize_balldontlie_game_stats(
        raw_stats,
        clean_game_logs=clean_game_logs,
        players=players,
        player_id_overrides=player_id_overrides,
    )
    return BallDontLieFetchResult(
        date_range=date_range,
        raw_stats=raw_stats,
        game_logs=normalized,
        unmatched_players=unmatched_balldontlie_players(normalized),
        raw_row_count=len(raw_stats),
    )


def fetch_missing_balldontlie_from_paths(
    paths,
    start_date: date | str | pd.Timestamp | None = None,
    end_date: date | str | pd.Timestamp | None = None,
    overlap_days: int = 0,
    player_id_overrides: Mapping[int, Any] | None = None,
    client: BallDontLieClient | None = None,
) -> BallDontLieFetchResult:
    """Load canonical Silver history and fetch only the subsequent API date range."""
    from .loaders import load_player_game_logs, load_players

    return fetch_missing_balldontlie_game_logs(
        clean_game_logs=load_player_game_logs(paths),
        players=load_players(paths),
        start_date=start_date,
        end_date=end_date,
        overlap_days=overlap_days,
        player_id_overrides=player_id_overrides,
        client=client,
    )
