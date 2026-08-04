from __future__ import annotations

from .cleaning import season_label_to_start_year


SHORT_TERM_TRAIN_END_SEASON = "2022-23"
SHORT_TERM_VALIDATION_SEASONS = {"2023-24"}
SHORT_TERM_TEST_SEASONS = {"2024-25"}

SALARY_TRAIN_SEASONS = {
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
}
SALARY_VALIDATION_SEASONS = {"2023-24"}
SALARY_TEST_SEASONS = {"2024-25"}

LONG_TERM_TRAIN_END_SEASON = "2019-20"
LONG_TERM_VALIDATION_SEASONS = {"2020-21"}
LONG_TERM_TEST_SEASONS = {"2021-22"}


def assign_temporal_split(season: object) -> str:
    # Assign point-in-time splits for short-term forecasting.
    """Assign short-term performance split from an NBA season label."""
    if season in SHORT_TERM_VALIDATION_SEASONS:
        return "validation"
    if season in SHORT_TERM_TEST_SEASONS:
        return "test"
    start_year = season_label_to_start_year(season)
    train_end_year = season_label_to_start_year(SHORT_TERM_TRAIN_END_SEASON)
    if start_year is not None and train_end_year is not None and start_year <= train_end_year:
        return "train"
    return "ignore"


def assign_salary_temporal_split(season: object) -> str:
    # Assign modern salary valuation splits.
    """Assign salary modeling split from an NBA season label."""
    if season in SALARY_TRAIN_SEASONS:
        return "train"
    if season in SALARY_VALIDATION_SEASONS:
        return "validation"
    if season in SALARY_TEST_SEASONS:
        return "test"
    return "ignore"


def assign_long_term_temporal_split(season: object) -> str:
    # Assign shifted long-term splits with complete H3 labels.
    """Assign long-term anchor split from an NBA season label."""
    if season in LONG_TERM_VALIDATION_SEASONS:
        return "validation"
    if season in LONG_TERM_TEST_SEASONS:
        return "test"
    start_year = season_label_to_start_year(season)
    train_end_year = season_label_to_start_year(LONG_TERM_TRAIN_END_SEASON)
    if start_year is not None and train_end_year is not None and start_year <= train_end_year:
        return "train"
    return "ignore"

