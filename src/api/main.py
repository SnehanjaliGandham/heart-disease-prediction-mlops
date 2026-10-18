"""Heart disease prediction API.

Run locally:      uvicorn src.api.main:app --port 8000
In Docker:        see Dockerfile (same command, as a non-root user, access log off -
                  every request is already logged as one JSON line)
Interactive docs: http://localhost:8000/docs

Endpoints
    GET  /               service information
    GET  /health         liveness - the process is up
    GET  /ready          readiness - the model is loaded (503 otherwise)
    GET  /model-info     which model is served and how it scored
    POST /predict        one patient   -> prediction + confidence
    POST /predict/batch  up to 100 patients
    GET  /metrics        Prometheus metrics

Configuration (environment variables)
    MODEL_PATH            path to the joblib model   (default: models/heart_disease_model.joblib)
    PREDICTION_THRESHOLD  decision threshold 0-1     (default: 0.5)
    LOG_LEVEL             DEBUG / INFO / WARNING     (default: INFO)
"""
from __future__ import annotations

import json
import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.routing import Match

from src import config
from src.api import observability as obs
from src.api.schemas import (
    BatchRequest,
    BatchResponse,
    HealthResponse,
    PatientFeatures,
    PredictionResponse,
    PredictionResult,
    ReadyResponse,
)
from src.predict import predict, version_mismatch

API_VERSION = "1.0.0"
REQUEST_ID_HEADER = "X-Request-ID"
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
QUIET_ROUTES = {"/health", "/ready", "/metrics"}


def _threshold() -> float:
    value = float(os.environ.get("PREDICTION_THRESHOLD", "0.5"))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"PREDICTION_THRESHOLD must be between 0 and 1, got {value}")
    return value


def _route_template(request: Request) -> str:
    """Route template for metrics/logs ("/predict"), never the raw URL, so that random
    paths (scanners, typos) cannot create unbounded metric labels."""
    route = request.scope.get("route")
    if route is not None:
        return route.path
    for candidate in request.app.router.routes:      # e.g. /docs, /openapi.json
        if candidate.matches(request.scope)[0] == Match.FULL:
            return candidate.path
    return "unmatched"


def _load(model_path: Path, log) -> dict:
    """Load the model and its metadata card. Never raises: failure = not ready."""
    state = {"model": None, "metadata": {}, "model_version": None}
    try:
        meta_path = model_path.parent / config.MODEL_METADATA_PATH.name
        if meta_path.exists():
            state["metadata"] = json.loads(meta_path.read_text())
        problem = version_mismatch(meta_path)
        if problem:
            log.warning("model/library version mismatch",
                        extra={"fields": {"event": "version_mismatch", "detail": problem}})
        state["model"] = joblib.load(model_path)
        meta = state["metadata"]
        # name + code version, falling back to the training date for older model cards
        stamp = meta.get("git_commit") or str(meta.get("trained_at_utc", "unknown"))[:10]
        state["model_version"] = f"{meta.get('model_name', 'model')}-{stamp}"
        obs.MODEL_INFO.labels(model_name=meta.get("model_name", "unknown"),
                              model_version=state["model_version"]).set(1)
        log.info("model loaded", extra={"fields": {
            "event": "model_loaded", "path": str(model_path),
            "model_version": state["model_version"]}})
    except Exception:  # noqa: BLE001 - report and stay up (readiness probe will fail)
        log.exception("model failed to load",
                      extra={"fields": {"event": "model_load_failed", "path": str(model_path)}})
    return state


def create_app(model_path: Optional[Path] = None) -> FastAPI:
    log = obs.get_logger(level=os.environ.get("LOG_LEVEL", "INFO"))
    path = Path(model_path or os.environ.get("MODEL_PATH", config.MODEL_PATH))
    threshold = _threshold()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.serving = _load(path, log)
        app.state.threshold = threshold
        yield

    app = FastAPI(
        title="Heart Disease Prediction API",
        version=API_VERSION,
        description="Predicts the presence of heart disease from 13 clinical attributes "
                    "(UCI Heart Disease, Cleveland). Academic project - not a medical device.",
        lifespan=lifespan,
    )

    # ------------------------------------------------------------ middleware
    @app.middleware("http")
    async def request_context(request: Request, call_next):
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        request_id = incoming if _SAFE_ID.match(incoming) else uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        with obs.Timer() as timer:
            try:
                response = await call_next(request)
            except Exception:  # noqa: BLE001
                log.exception("unhandled error", extra={"fields": {"request_id": request_id}})
                obs.ERRORS.labels(reason="internal").inc()
                response = JSONResponse(status_code=500, content={
                    "detail": "Internal server error", "request_id": request_id})
        route_path = _route_template(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        obs.record_request(request.method, route_path, response.status_code, timer.seconds)
        # probes and scrapes run every few seconds: keep them out of INFO-level logs
        level = logging.DEBUG if route_path in QUIET_ROUTES else logging.INFO
        log.log(level, "request", extra={"fields": {
                "event": "request", "request_id": request_id, "method": request.method,
                "route": route_path, "status": response.status_code,
                "duration_ms": round(timer.seconds * 1000, 2),
                "client": request.client.host if request.client else None}})
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        obs.ERRORS.labels(reason="validation").inc()
        errors = [{"field": ".".join(str(p) for p in e["loc"] if p != "body"),
                   "message": e["msg"], "type": e["type"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content={
            "detail": errors, "request_id": request.state.request_id})

    # ------------------------------------------------------------ helpers
    def serving(request: Request) -> dict:
        return request.app.state.serving

    def unavailable(request: Request) -> JSONResponse:
        obs.ERRORS.labels(reason="model_unavailable").inc()
        return JSONResponse(status_code=503, content={
            "detail": "Model not loaded", "request_id": request.state.request_id})

    def run(request: Request, patients: list) -> list:
        records = [p.model_dump() for p in patients]
        results = predict(records, model=serving(request)["model"],
                          threshold=request.app.state.threshold)
        for record, res in zip(records, results):
            obs.record_prediction(res["label"], res["probability_disease"], record)
        log.info("prediction", extra={"fields": {
            "event": "prediction", "request_id": request.state.request_id,
            "count": len(results),
            "classes": [r["prediction"] for r in results],
            "model_version": serving(request)["model_version"]}})
        return results

    # ------------------------------------------------------------ endpoints
    @app.get("/", tags=["info"])
    def root():
        return {"service": "heart-disease-prediction-api", "version": API_VERSION,
                "docs": "/docs", "health": "/health", "ready": "/ready",
                "predict": "/predict", "metrics": "/metrics"}

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health():
        return {"status": "ok"}

    @app.get("/ready", response_model=ReadyResponse, tags=["ops"],
             responses={503: {"description": "Model not loaded"}})
    def ready(request: Request):
        state = serving(request)
        body = {"status": "ready" if state["model"] is not None else "not ready",
                "model_loaded": state["model"] is not None,
                "model_version": state["model_version"]}
        return JSONResponse(status_code=200 if state["model"] is not None else 503,
                            content=body)

    @app.get("/model-info", tags=["info"])
    def model_info(request: Request):
        state = serving(request)
        meta = state["metadata"]
        return {
            "model_version": state["model_version"],
            "model_name": meta.get("model_name"),
            "model_class": meta.get("model_class"),
            "best_params": meta.get("best_params"),
            "cv_roc_auc_mean": meta.get("cv_metrics", {}).get("cv_roc_auc_mean"),
            "test_metrics": meta.get("test_metrics"),
            "trained_at_utc": meta.get("trained_at_utc"),
            "git_commit": meta.get("git_commit"),
            "sklearn_version": meta.get("sklearn_version"),
            "decision_threshold": request.app.state.threshold,
            "features": config.FEATURES,
            "served_by": obs.INSTANCE,
        }

    @app.post("/predict", response_model=PredictionResponse, tags=["prediction"],
              responses={422: {"description": "Invalid input"},
                         503: {"description": "Model not loaded"}})
    def predict_one(patient: PatientFeatures, request: Request):
        if serving(request)["model"] is None:
            return unavailable(request)
        result = run(request, [patient])[0]
        return {**result, "model_version": serving(request)["model_version"],
                "request_id": request.state.request_id}

    @app.post("/predict/batch", response_model=BatchResponse, tags=["prediction"],
              responses={422: {"description": "Invalid input"},
                         503: {"description": "Model not loaded"}})
    def predict_batch(batch: BatchRequest, request: Request):
        if serving(request)["model"] is None:
            return unavailable(request)
        results = run(request, batch.patients)
        return {"predictions": [PredictionResult(**r) for r in results],
                "count": len(results), "model_version": serving(request)["model_version"],
                "request_id": request.state.request_id}

    @app.get("/metrics", tags=["ops"], include_in_schema=False)
    def metrics():
        return Response(content=obs.metrics_payload(), media_type=obs.PROMETHEUS_CONTENT_TYPE)

    return app


app = create_app()
