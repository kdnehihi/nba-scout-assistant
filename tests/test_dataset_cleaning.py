from __future__ import annotations

import pandas as pd

from dataset.cleaning import (
    normalize_name_key,
    normalize_season,
    normalize_team_abbreviation,
    parse_salary,
    percent_to_ratio,
    safe_per_36,
    to_snake_case,
)


def test_basic_cleaning_helpers():
    assert to_snake_case("Player Name!") == "player_name"
    assert normalize_name_key("LeBron James Jr.") == "lebronjamesjr"
    assert normalize_team_abbreviation("BRK") == "BKN"
    assert parse_salary("$12,345,678") == 12345678
    assert normalize_season("2023-24") == (2023, 2024, "2023-24")


def test_percent_to_ratio_and_safe_per_36():
    ratios = percent_to_ratio(pd.Series([55.0, 0.55, None]))
    assert ratios.iloc[0] == 0.55
    assert ratios.iloc[1] == 0.55
    assert pd.isna(ratios.iloc[2])

    per_36 = safe_per_36(pd.Series([10, 5]), pd.Series([20, 0]))
    assert per_36.iloc[0] == 18
    assert pd.isna(per_36.iloc[1])
