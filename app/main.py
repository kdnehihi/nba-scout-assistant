from fastapi import FastAPI, HTTPException
from pydatic import BaseModel
from src.pipelines.recommendation import (
    load_recommendation_pipeline_data,
    recommend_similar_players,
    evaluate_recommendation_result,
)

app = FastAPI(title= "NBA Scout Assistant")


class RecommendationRequest(BaseModel):
    player_name: str
    season: str | None = None
    top_n: int = 5
    preset: str = "playing_profile"
    same_position_group: bool = True
    minutes_min: float | None = 500

@app.on_envent("startup")
def startup():
    app.state.recommendation_data = load_recommendation_pipeline_data("data")
    app.state.short_term_models_pts = {} 
    app.state.long_term_models = {}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommendation")
def recommendation(request:RecommendationRequest):
    try:
        recs = recommend_similar_players(
            pipeline_data = app.state.recommendation_data,
            player_name= request.player_name,
            season= request.top_n,
            preset= request.preset,
            same_position_group= request.same_position_group,
            minutes_min = request.minutes_min
        )
        diagnostics = evaluate_recommendation_result(
            pipeline_data = app.state.recommendation_data,
            recommendations= recs,
            top_n= request.top_n
        )
        return {
            "recommendations": recs.to_dict("records"),
            "diagnostics": diagnostics,
        }
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))

