from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.dataset.loaders import DataPaths, resolve_data_paths
from src.dataset.incremental import IncrementalPipelineResult, run_balldontlie_incremental_pipeline
from src.dataset.pipeline import build_all_gold_datasets


def run_gold_data_pipeline(data_dir: Path | str = "data") -> dict[str, pd.DataFrame]:
    # Rebuild model-ready gold datasets from canonical raw and silver inputs.
    """Run the local data pipeline and return generated gold dataframes."""
    paths = resolve_data_paths(data_dir)
    return build_all_gold_datasets(paths)


def resolve_pipeline_data_paths(data_dir: Path | str = "data") -> DataPaths:
    # Centralize path resolution for application pipelines.
    """Return resolved project data paths for pipeline callers."""
    return resolve_data_paths(data_dir)


def run_online_data_update_pipeline(
    data_dir: Path | str = "data",
    **kwargs,
) -> IncrementalPipelineResult:
    # Update canonical history and current inference datasets from BALLDONTLIE.
    """Run the incremental online data pipeline for a resolved project data root."""
    return run_balldontlie_incremental_pipeline(
        paths=resolve_data_paths(data_dir),
        **kwargs,
    )
