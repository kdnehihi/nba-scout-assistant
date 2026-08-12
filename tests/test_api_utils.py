from __future__ import annotations

import numpy as np
import pandas as pd

from src.api_utils import json_safe


def test_json_safe_converts_non_finite_and_pandas_values():
    payload = {
        "nan_value": np.nan,
        "nat_value": pd.NaT,
        "inf_value": float("inf"),
        "timestamp": pd.Timestamp("2024-01-01"),
        "array": np.asarray([1.0, np.nan]),
        "records": pd.DataFrame({"value": [1.0, np.nan]}),
    }

    result = json_safe(payload)

    assert result["nan_value"] is None
    assert result["nat_value"] is None
    assert result["inf_value"] is None
    assert result["timestamp"] == "2024-01-01T00:00:00"
    assert result["array"] == [1.0, None]
    assert result["records"] == [{"value": 1.0}, {"value": None}]
