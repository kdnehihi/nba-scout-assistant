from __future__ import annotations

import numpy as np
import pandas as pd

from .config import SalaryValueConfig


def build_salary_value_signals(
    salary_df: pd.DataFrame,
    predicted_salary_usd_col: str = "predicted_salary_usd",
    actual_salary_usd_col: str = "salary_usd",
    config: SalaryValueConfig = SalaryValueConfig(),
    fair_value_tolerance: float | None = None,
) -> pd.DataFrame:
    # Convert salary predictions into underpaid/fair/overpaid labels.
    """Return deterministic salary value gap and value bucket signals."""
    tolerance = config.fair_value_tolerance if fair_value_tolerance is None else fair_value_tolerance
    result = salary_df.copy()
    result["salary_value_gap_usd"] = result[predicted_salary_usd_col] - result[actual_salary_usd_col]
    result["salary_value_ratio"] = result[predicted_salary_usd_col] / result[actual_salary_usd_col].replace(0, np.nan)
    result["salary_value_label"] = np.select(
        [
            result["salary_value_ratio"] >= 1 + tolerance,
            result["salary_value_ratio"] <= 1 - tolerance,
        ],
        ["underpaid", "overpaid"],
        default="fair_value",
    )
    return result
