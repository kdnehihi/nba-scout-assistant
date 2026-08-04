from __future__ import annotations

import pandas as pd


def similarity_diagnostics(candidates: pd.DataFrame) -> dict[str, float]:
    # Summarize replacement-candidate ranking outputs without supervised labels.
    """Return simple diagnostics for a similarity candidate table."""
    if candidates.empty:
        return {
            "rows": 0.0,
            "avg_similarity_score": float("nan"),
            "avg_similarity_distance": float("nan"),
            "avg_salary_cap_share_gap": float("nan"),
            "avg_age_gap": float("nan"),
        }
    return {
        "rows": float(len(candidates)),
        "avg_similarity_score": float(candidates["similarity_score"].mean()),
        "avg_similarity_distance": float(candidates["similarity_distance"].mean()),
        "avg_salary_cap_share_gap": float(candidates["salary_cap_share_gap"].mean()),
        "avg_age_gap": float(candidates["age_gap"].mean()),
    }

