from __future__ import annotations

import pandas as pd
import pytest

from dataset.loaders import (
    load_long_term_inference,
    load_short_term_inference,
    load_tabular_data,
    require_columns,
    resolve_data_paths,
)


def test_resolve_data_paths_builds_layer_paths(tmp_path):
    paths = resolve_data_paths(tmp_path)

    assert paths.data_dir == tmp_path.resolve()
    assert paths.raw_dir == tmp_path / "raw"
    assert paths.silver_dir == tmp_path / "silver"
    assert paths.gold_dir == tmp_path / "gold"


def test_load_tabular_data_reads_csv(tmp_path):
    path = tmp_path / "sample.csv"
    pd.DataFrame({"a": [1], "b": [2]}).to_csv(path, index=False)

    loaded = load_tabular_data(path)

    assert loaded.to_dict("records") == [{"a": 1, "b": 2}]


def test_load_tabular_data_rejects_unsupported_file(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("x")

    with pytest.raises(ValueError, match="Unsupported file extension"):
        load_tabular_data(path)


def test_require_columns_reports_missing_columns():
    df = pd.DataFrame({"a": [1]})

    with pytest.raises(KeyError, match="missing required columns"):
        require_columns(df, ["a", "b"], "sample")


def test_inference_loaders_prefer_latest_gold_outputs(tmp_path):
    paths = resolve_data_paths(tmp_path)
    paths.gold_dir.mkdir(parents=True)
    pd.DataFrame({"marker": ["short_latest"]}).to_parquet(
        paths.gold_dir / "short_term_inference_latest.parquet",
        index=False,
    )
    pd.DataFrame({"marker": ["long_latest"]}).to_parquet(
        paths.gold_dir / "long_term_player_forecast_inference_latest.parquet",
        index=False,
    )

    assert load_short_term_inference(paths)["marker"].iloc[0] == "short_latest"
    assert load_long_term_inference(paths)["marker"].iloc[0] == "long_latest"
