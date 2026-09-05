from __future__ import annotations

from dataclasses import replace

from fastapi.testclient import TestClient

import app.main as api_main
from app.services.resources import AppResources
from src.scouting.ranking import RecommendationRankerArtifact, SeasonFeaturePreprocessor
from tests.test_pipeline_forecasting import (
    make_long_term_artifact,
    make_long_term_rows,
    make_short_term_artifact,
    make_short_term_rows,
)
from tests.test_pipelines import make_recommendation_pipeline_data


def make_resources(with_ranker: bool) -> AppResources:
    recommendation_data = make_recommendation_pipeline_data()
    if with_ranker:
        preprocessor = SeasonFeaturePreprocessor().fit(recommendation_data.recommendation_base)
        recommendation_data = replace(
            recommendation_data,
            recommendation_base=preprocessor.transform(recommendation_data.recommendation_base),
            ranker_artifact=RecommendationRankerArtifact(
                algorithm="season_normalized_euclidean",
                preprocessor=preprocessor,
                version="api-test-v1",
            ),
        )
    long_artifact = make_long_term_artifact()
    return AppResources(
        recommendation_data=recommendation_data,
        short_term_data=make_short_term_rows("points"),
        long_term_data=make_long_term_rows(),
        short_term_models={"points": make_short_term_artifact("points")},
        long_term_models={(long_artifact.task, long_artifact.horizon): long_artifact},
    )


def test_recommendations_endpoint_is_backward_compatible_with_and_without_ranker(monkeypatch):
    for with_ranker in (False, True):
        monkeypatch.setattr(
            api_main,
            "load_app_resources",
            lambda with_ranker=with_ranker, **_: make_resources(with_ranker),
        )
        with TestClient(api_main.app) as client:
            response = client.post(
                "/recommendations",
                json={
                    "player_name": "Target Guard",
                    "season": "2024-25",
                    "top_n": 2,
                    "preset": "playing_profile",
                },
            )
        assert response.status_code == 200
        recommendation = response.json()["recommendations"][0]
        assert "recommendation_score" in recommendation
        assert "ranking_algorithm" in recommendation
        assert "ranker_version" in recommendation


def test_direct_player_analysis_supports_name_and_partial_forecasts(monkeypatch):
    monkeypatch.setattr(api_main, "load_app_resources", lambda **_: make_resources(False))
    with TestClient(api_main.app) as client:
        response = client.post(
            "/players/scouting-report",
            json={
                "player_name": "target guard",
                "season": "2024-25",
                "include_forecasts": True,
                "short_term_tasks": ["assists"],
                "long_term_tasks": ["ast_per_36"],
                "long_term_horizons": [1],
            },
        )

    assert response.status_code == 200
    report = response.json()
    assert report["player"]["player_name"] == "Target Guard"
    assert report["short_term_forecast"] == {}
    assert report["long_term_forecast"] == {}
    assert len(report["warnings"]) == 2


def test_frontend_exposes_find_similar_and_analyze_player_modes():
    html = api_main.STATIC_DIR.joinpath("index.html").read_text(encoding="utf-8")
    javascript = api_main.STATIC_DIR.joinpath("app.js").read_text(encoding="utf-8")

    assert 'data-mode="recommend"' in html
    assert 'data-mode="analyze"' in html
    assert 'postJson("/players/scouting-report"' in javascript
    assert "loadPlayerReport" in javascript
