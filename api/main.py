"""
FastAPI service exposing the heart disease risk model.

Endpoints:
  GET  /health   -- liveness/readiness probe target for Docker/Kubernetes
  POST /predict  -- accepts patient clinical features, returns risk prediction
  GET  /metrics  -- Prometheus scrape target (request counts, latency, etc.)

Every request is also logged as a structured JSON line to logs/api.log
(and stdout, so `docker logs` / `kubectl logs` show it) satisfying the
assignment's "integrate logging of API requests" requirement.
"""

import logging
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from prometheus_fastapi_instrumentator import Instrumentator
from pythonjsonlogger import json as jsonlogger

from api.schemas import HealthResponse, PatientFeatures, PredictionResponse
from src.predict import get_model, predict_one

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("heart_disease_api")
logger.setLevel(logging.INFO)
_formatter = jsonlogger.JsonFormatter("%(asctime)s %(name)s %(levelname)s %(message)s")
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(_formatter)
_file_handler = logging.FileHandler(LOG_DIR / "api.log")
_file_handler.setFormatter(_formatter)
logger.addHandler(_stream_handler)
logger.addHandler(_file_handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_model()
        logger.info("model_loaded", extra={"status": "ok"})
    except FileNotFoundError as exc:
        logger.error("model_load_failed", extra={"error": str(exc)})
    yield


app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Serves a scikit-learn pipeline trained on the UCI Heart Disease dataset.",
    version="1.0.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    logger.info(
        "api_request",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
            "client": request.client.host if request.client else None,
        },
    )
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    try:
        get_model()
        return HealthResponse(status="ok", model_loaded=True)
    except FileNotFoundError:
        return HealthResponse(status="degraded", model_loaded=False)


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures) -> PredictionResponse:
    try:
        model = get_model()
        result = predict_one(features.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    model_name = type(model.named_steps["model"]).__name__
    logger.info(
        "prediction_made",
        extra={"prediction": result["prediction"], "confidence": result["confidence"]},
    )
    return PredictionResponse(**result, model_name=model_name)


@app.get("/")
def root():
    return {"message": "Heart Disease Risk Prediction API. See /docs for usage."}
