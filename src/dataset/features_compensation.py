from __future__ import annotations

import pandas as pd

from .cleaning import normalize_name_key, normalize_team_abbreviation, parse_salary


def build_player_salary_history(
    salaries: pd.DataFrame,
    salary_cap: pd.DataFrame | None = None,
    players: pd.DataFrame | None = None,
) -> pd.DataFrame:
    # Normalize historical player salaries for reporting and candidate context.
    """Build clean player-season salary history without creating modeling targets."""
    history = salaries.copy()
    history["salary_usd"] = (
        history["salary_usd"].map(parse_salary)
        if history["salary_usd"].dtype == "object"
        else pd.to_numeric(history["salary_usd"], errors="coerce")
    )
    history["team"] = history.get("team", "UNK")
    history["team_id"] = history["team"].map(normalize_team_abbreviation).fillna("UNK")
    history["player_name_key"] = history["player_name"].map(normalize_name_key)

    if "season_start_year" in history.columns:
        history["season_start_year"] = pd.to_numeric(history["season_start_year"], errors="coerce").astype("Int64")
    if "season_end_year" in history.columns:
        history["season_end_year"] = pd.to_numeric(history["season_end_year"], errors="coerce").astype("Int64")

    if salary_cap is not None and not salary_cap.empty:
        caps = salary_cap.copy()
        caps["salary_cap_usd"] = pd.to_numeric(caps["salary_cap_usd"], errors="coerce")
        if "tax_level_usd" in caps.columns:
            caps["tax_level_usd"] = pd.to_numeric(caps["tax_level_usd"], errors="coerce")
        history = history.merge(
            caps.rename(columns={"season": "season_label"}),
            on="season_label",
            how="left",
        )
        history["salary_cap_share"] = history["salary_usd"] / history["salary_cap_usd"]

    if players is not None and not players.empty and "player_id" not in history.columns:
        player_lookup = players.copy()
        player_lookup["player_name_key"] = player_lookup["player_name"].map(normalize_name_key)
        player_cols = [column for column in ["player_name_key", "player_id"] if column in player_lookup.columns]
        if set(player_cols) == {"player_name_key", "player_id"}:
            history = history.merge(
                player_lookup[player_cols].drop_duplicates("player_name_key"),
                on="player_name_key",
                how="left",
            )

    output_cols = [
        "player_id",
        "player_name",
        "team",
        "team_id",
        "season_start_year",
        "season_end_year",
        "season_label",
        "salary_usd",
        "salary_cap_usd",
        "tax_level_usd",
        "salary_cap_share",
        "source",
        "source_file",
        "collected_at",
    ]
    existing_cols = [column for column in output_cols if column in history.columns]
    history = history[existing_cols].dropna(subset=["player_name", "season_label", "salary_usd"])
    return history.sort_values(["player_name", "season_start_year"]).reset_index(drop=True)
