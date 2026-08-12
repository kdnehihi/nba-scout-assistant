from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.dataset.loaders import resolve_data_paths
from src.scouting.pipeline import build_all_scouting_artifacts


def run_scouting_artifact_pipeline(
    data_dir: Path | str = "data",
    example_queries: list[dict[str, object]] | None = None,
) -> dict[str, pd.DataFrame]:
    # Build deterministic scouting artifacts from clean gold datasets.
    """Run the deterministic scouting artifact pipeline."""
    paths = resolve_data_paths(data_dir)
    return build_all_scouting_artifacts(paths, example_queries=example_queries)
