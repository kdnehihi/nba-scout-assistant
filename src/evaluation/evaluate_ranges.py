from __future__ import annotations

import pandas as pd

from .metrics import range_metrics


def evaluate_short_term_ranges(
    signals: pd.DataFrame,
    stat_targets: dict[str, str],
    split_col: str = "split",
) -> pd.DataFrame:
    # Evaluate deterministic expected/floor/ceiling ranges by split and stat.
    """Return split-level range metrics for short-term scouting ranges."""
    rows: list[dict[str, object]] = []
    for split_name, split_df in signals.groupby(split_col, dropna=False):
        for stat, target_col in stat_targets.items():
            expected_col = f"expected_next_5_{stat}_avg"
            lower_col = f"floor_next_5_{stat}_avg"
            upper_col = f"ceiling_next_5_{stat}_avg"
            required = [target_col, expected_col, lower_col, upper_col]
            valid = split_df[required].dropna()
            if valid.empty:
                continue
            metrics = range_metrics(
                valid[target_col],
                valid[expected_col],
                valid[lower_col],
                valid[upper_col],
            )
            rows.append({"split": split_name, "stat": stat, "rows": len(valid), **metrics})
    return pd.DataFrame(rows).sort_values(["stat", "split"]).reset_index(drop=True)

