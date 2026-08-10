from __future__ import annotations

import pandas as pd

from scouting.compensation import (
    build_player_compensation_context,
    normalize_contract_history,
    player_contract_history,
    player_salary_history,
)


def test_player_salary_history_selects_by_player_id_and_orders_seasons():
    salary_history = pd.DataFrame(
        {
            "player_id": [1, 1, 2],
            "player_name": ["Player One", "Player One", "Player Two"],
            "season_start_year": [2023, 2022, 2023],
            "season_label": ["2023-24", "2022-23", "2023-24"],
            "salary_usd": [12_000_000, 10_000_000, 8_000_000],
        }
    )

    selected = player_salary_history(salary_history, player_id=1)

    assert selected["season_label"].tolist() == ["2022-23", "2023-24"]


def test_normalize_contract_history_handles_external_column_names():
    contracts = pd.DataFrame(
        {
            "NAME": ["Player One"],
            "CONTRACT_START": [2023],
            "CONTRACT_END": [2026],
            "AVG_SALARY": ["$15,000,000"],
            "PTS": [18.5],
            "AST": [4.2],
            "TRB": [5.1],
        }
    )

    normalized = normalize_contract_history(contracts)

    assert normalized["average_salary_usd"].iloc[0] == 15_000_000
    assert {"points", "assists", "rebounds"}.issubset(normalized.columns)


def test_build_player_compensation_context_returns_salary_and_contract_rows():
    salary_history = pd.DataFrame(
        {
            "player_id": [1],
            "player_name": ["Player One"],
            "season_start_year": [2024],
            "season_label": ["2024-25"],
            "salary_usd": [20_000_000],
        }
    )
    contract_history = pd.DataFrame({"NAME": ["Player One"], "CONTRACT_START": [2024], "AVG_SALARY": ["$20,000,000"]})

    context = build_player_compensation_context(salary_history, player_id=1, contract_history=contract_history)

    assert context["latest_salary"]["salary_usd"] == 20_000_000
    assert not context["salary_history"].empty
    assert not context["contract_history"].empty


def test_player_contract_history_matches_normalized_name():
    contracts = pd.DataFrame({"NAME": ["Player One"], "AVG_SALARY": ["$10,000,000"]})

    selected = player_contract_history(contracts, "player-one")

    assert len(selected) == 1
