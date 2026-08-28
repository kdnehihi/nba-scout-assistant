from __future__ import annotations

from datetime import date

import httpx
import pandas as pd

from src.dataset.balldontlie import (
    BallDontLieClient,
    BallDontLieConfig,
    determine_missing_date_range,
    normalize_balldontlie_game_stats,
    parse_balldontlie_minutes,
    unmatched_balldontlie_players,
)


def make_stat_row(
    *,
    stat_id: int = 9001,
    player_id: int = 70,
    first_name: str = "Jaylen",
    last_name: str = "Brown",
    game_date: str = "2025-04-15",
) -> dict[str, object]:
    return {
        "id": stat_id,
        "min": "30:30",
        "fgm": 7,
        "fga": 18,
        "fg3m": 5,
        "fg3a": 9,
        "ftm": 4,
        "fta": 4,
        "oreb": 2,
        "dreb": 5,
        "reb": 7,
        "ast": 1,
        "stl": 1,
        "blk": 0,
        "turnover": 1,
        "pf": 3,
        "pts": 23,
        "player": {
            "id": player_id,
            "first_name": first_name,
            "last_name": last_name,
        },
        "team": {"id": 2, "abbreviation": "BOS"},
        "game": {
            "id": 15907438,
            "date": game_date,
            "season": 2024,
            "status": "Final",
            "status_state": "final",
            "postseason": False,
            "home_team_id": 2,
            "visitor_team_id": 20,
            "home_team": {"id": 2, "abbreviation": "BOS"},
            "visitor_team": {"id": 20, "abbreviation": "NYK"},
        },
    }


def test_determine_missing_date_range_starts_after_clean_history():
    clean = pd.DataFrame({"game_date": pd.to_datetime(["2025-04-10", "2025-04-13"])})

    result = determine_missing_date_range(clean, end_date="2025-04-20")
    overlap = determine_missing_date_range(clean, end_date="2025-04-20", overlap_days=3)

    assert result is not None
    assert result.start_date == date(2025, 4, 14)
    assert result.end_date == date(2025, 4, 20)
    assert overlap is not None
    assert overlap.start_date == date(2025, 4, 11)


def test_balldontlie_client_follows_cursor_pagination():
    requested_cursors: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "secret"
        cursor = request.url.params.get("cursor")
        requested_cursors.append(cursor)
        if cursor is None:
            return httpx.Response(
                200,
                json={"data": [{"id": 1}], "meta": {"next_cursor": 25}},
            )
        return httpx.Response(200, json={"data": [{"id": 2}], "meta": {}})

    http_client = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url="https://api.balldontlie.io/v1",
    )
    client = BallDontLieClient(
        api_key="secret",
        config=BallDontLieConfig(max_retries=1),
        http_client=http_client,
    )

    rows = client.fetch_player_game_stats(date(2025, 4, 14), date(2025, 4, 15))

    assert [row["id"] for row in rows] == [1, 2]
    assert requested_cursors == [None, "25"]


def test_normalize_balldontlie_stats_maps_identity_and_box_score():
    clean = pd.DataFrame(
        {
            "player_id": [1627759],
            "player_name": ["Jaylen Brown"],
            "game_date": pd.to_datetime(["2025-04-13"]),
        }
    )

    opponent_row = make_stat_row(
        stat_id=9002,
        player_id=999,
        first_name="New",
        last_name="Player",
    )
    opponent_row["team"] = {"id": 20, "abbreviation": "NYK"}
    opponent_row["game"].pop("home_team")
    opponent_row["game"].pop("visitor_team")
    target_row = make_stat_row()
    target_row["game"].pop("home_team")
    target_row["game"].pop("visitor_team")

    normalized = normalize_balldontlie_game_stats(
        [target_row, opponent_row],
        clean_game_logs=clean,
        fetched_at=pd.Timestamp("2025-04-16T06:00:00Z"),
    )

    row = normalized[normalized["source_player_id"].eq(70)].iloc[0]
    assert row["player_id"] == 1627759
    assert row["source_player_id"] == 70
    assert row["season"] == "2024-25"
    assert row["team_id"] == "BOS"
    assert row["opponent"] == "NYK"
    assert row["home_away"] == "HOME"
    assert row["minutes"] == 30.5
    assert row["rest_days"] == 2
    assert row["true_shooting_pct"] == 23 / (2 * (18 + 0.44 * 4))


def test_unmatched_players_are_reported_without_reusing_source_id():
    clean = pd.DataFrame(
        columns=["player_id", "player_name", "game_date"],
    ).astype({"player_name": "object"})
    clean.loc[0] = [1, "Existing Player", pd.Timestamp("2025-04-13")]

    normalized = normalize_balldontlie_game_stats(
        [
            make_stat_row(
                player_id=999,
                first_name="New",
                last_name="Rookie",
            )
        ],
        clean_game_logs=clean,
    )
    unmatched = unmatched_balldontlie_players(normalized)

    assert pd.isna(normalized.iloc[0]["player_id"])
    assert normalized.iloc[0]["source_player_id"] == 999
    assert unmatched.to_dict("records") == [
        {"source_player_id": 999, "player_name": "New Rookie"}
    ]


def test_parse_balldontlie_minutes_handles_decimal_and_clock_text():
    assert parse_balldontlie_minutes("12") == 12.0
    assert parse_balldontlie_minutes("12:30") == 12.5
    assert parse_balldontlie_minutes("") is None
