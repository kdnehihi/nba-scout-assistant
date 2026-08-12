from __future__ import annotations

from src.pipelines.reporting import build_player_scouting_report, build_service_metadata
from tests.test_pipeline_forecasting import make_long_term_artifact, make_long_term_rows, make_short_term_artifact, make_short_term_rows
from tests.test_pipelines import make_recommendation_pipeline_data


def test_build_player_scouting_report_returns_context_and_forecasts():
    recommendation_data = make_recommendation_pipeline_data()
    short_artifact = make_short_term_artifact("points")
    long_artifact = make_long_term_artifact()

    report = build_player_scouting_report(
        recommendation_data=recommendation_data,
        performance_df=make_short_term_rows("points"),
        long_term_df=make_long_term_rows(),
        short_term_models={"points": short_artifact},
        long_term_models={(long_artifact.task, long_artifact.horizon): long_artifact},
        player_id=1,
        season="2024-25",
        anchor_season="2021-22",
        short_term_tasks=["points"],
        long_term_tasks=["pts_per_36"],
        long_term_horizons=[1],
    )

    assert report["player"]["player_id"] == 1
    assert "compensation" in report
    assert "points" in report["short_term_forecast"]
    assert report["long_term_forecast"]["pts_per_36"][1]["prediction"] == 18.0


def test_build_service_metadata_reports_loaded_models_and_rows():
    recommendation_data = make_recommendation_pipeline_data()
    long_artifact = make_long_term_artifact()

    metadata = build_service_metadata(
        recommendation_data=recommendation_data,
        performance_df=make_short_term_rows("points"),
        long_term_df=make_long_term_rows(),
        short_term_models={"points": make_short_term_artifact("points")},
        long_term_models={(long_artifact.task, long_artifact.horizon): long_artifact},
    )

    assert metadata["recommendation_base_rows"] > 0
    assert metadata["short_term_models"] == ["points"]
    assert metadata["long_term_models"] == [{"task": "pts_per_36", "horizon": 1}]
