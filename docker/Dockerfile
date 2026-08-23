FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-api.txt .

RUN pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.3,<3.0" \
    && pip install -r requirements-api.txt \
    && addgroup --system app \
    && adduser --system --ingroup app app

COPY --chown=app:app app ./app
COPY --chown=app:app src ./src
COPY --chown=app:app \
    data/gold/performance_training_clean.parquet \
    data/gold/player_role_features_clean.parquet \
    data/gold/player_salary_history_clean.parquet \
    data/gold/long_term_player_forecast_inference.parquet \
    ./data/gold/
COPY --chown=app:app \
    artifacts/short_term_lstm_points.pt \
    artifacts/short_term_lstm_assists.pt \
    artifacts/short_term_lstm_rebounds.pt \
    artifacts/long_term_active_probability_h1_random_forest.joblib \
    artifacts/long_term_active_probability_h2_random_forest.joblib \
    artifacts/long_term_active_probability_h3_random_forest.joblib \
    artifacts/long_term_pts_per_36_h1_random_forest.joblib \
    artifacts/long_term_pts_per_36_h2_random_forest.joblib \
    artifacts/long_term_pts_per_36_h3_mlp.pt \
    artifacts/long_term_pts_per_36_h3_mlp_preprocessor.joblib \
    artifacts/long_term_ast_per_36_h1_random_forest.joblib \
    artifacts/long_term_ast_per_36_h2_random_forest.joblib \
    artifacts/long_term_ast_per_36_h3_mlp.pt \
    artifacts/long_term_ast_per_36_h3_mlp_preprocessor.joblib \
    artifacts/long_term_reb_per_36_h1_random_forest.joblib \
    artifacts/long_term_reb_per_36_h2_random_forest.joblib \
    artifacts/long_term_reb_per_36_h3_mlp.pt \
    artifacts/long_term_reb_per_36_h3_mlp_preprocessor.joblib \
    ./artifacts/

USER app

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=3)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
