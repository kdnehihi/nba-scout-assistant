from __future__ import annotations

import numpy as np
import pandas as pd


def build_salary_value_signals(
    salary_df: pd.DataFrame,
    predicted_salary_usd_col: str = "predicted_salary_usd",
    actual_salary_usd_col: str = "salary_usd",
    fair_value_tolerance: float = 0.10,
) -> pd.DataFrame:
    # Convert salary predictions into underpaid/fair/overpaid labels.
    """Return deterministic salary value gap and value bucket signals."""
    result = salary_df.copy()
    result["salary_value_gap_usd"] = result[predicted_salary_usd_col] - result[actual_salary_usd_col]
    result["salary_value_ratio"] = result[predicted_salary_usd_col] / result[actual_salary_usd_col].replace(0, np.nan)
    result["salary_value_label"] = np.select(
        [
            result["salary_value_ratio"] >= 1 + fair_value_tolerance,
            result["salary_value_ratio"] <= 1 - fair_value_tolerance,
        ],
        ["underpaid", "overpaid"],
        default="fair_value",
    )
    return result

