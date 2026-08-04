from __future__ import annotations

from dataset.splits import (
    assign_long_term_temporal_split,
    assign_salary_temporal_split,
    assign_temporal_split,
)


def test_temporal_splits():
    assert assign_temporal_split("2022-23") == "train"
    assert assign_temporal_split("2023-24") == "validation"
    assert assign_temporal_split("2024-25") == "test"
    assert assign_temporal_split("2025-26") == "ignore"


def test_salary_and_long_term_splits():
    assert assign_salary_temporal_split("2016-17") == "train"
    assert assign_salary_temporal_split("2023-24") == "validation"
    assert assign_salary_temporal_split("2024-25") == "test"
    assert assign_long_term_temporal_split("2019-20") == "train"
    assert assign_long_term_temporal_split("2020-21") == "validation"
    assert assign_long_term_temporal_split("2021-22") == "test"
