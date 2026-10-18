"""Structured logging and Prometheus metrics for the prediction API.

Logging: one JSON object per line (easy to search and to ship to a log system).
Patient feature values are never logged - only request metadata and the outcome.

Metrics (exposed at /metrics, scraped by Prometheus in Step 8):
    http_requests_total{method,route,status}     request count
    http_request_duration_seconds{route}         latency histogram
    predictions_total{label}                     predictions by class
    prediction_probability                       histogram of P(disease)
    prediction_errors_total{reason}              validation / model_unavailable / internal
    input_feature_value{feature}                 summary of incoming values (drift signal)
    model_info{model_name,model_version}         which model is being served
"""
from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone

from prometheus_client import CONTENT_TYPE_LATEST as PROMETHEUS_CONTENT_TYPE  # noqa: F401
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Summary,
    generate_latest,
)

REGISTRY = CollectorRegistry()

HTTP_REQUESTS = Counter(
    "http_requests_total", "HTTP requests handled", ["method", "route", "status"],
    registry=REGISTRY)
HTTP_LATENCY = Histogram(
    "http_request_duration_seconds", "Request latency in seconds", ["route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5), registry=REGISTRY)
PREDICTIONS = Counter(
    "predictions_total", "Predictions returned, by predicted class", ["label"],
    registry=REGISTRY)
PROBABILITY = Histogram(
    "prediction_probability", "Predicted probability of heart disease",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0), registry=REGISTRY)
ERRORS = Counter(
    "prediction_errors_total", "Rejected or failed prediction requests", ["reason"],
    registry=REGISTRY)
FEATURES = Summary(
    "input_feature_value", "Values of numeric input features (mean = sum / count)",
    ["feature"], registry=REGISTRY)
MODEL_INFO = Gauge(
    "model_info", "Model currently served (value is always 1)",
    ["model_name", "model_version"], registry=REGISTRY)

MONITORED_FEATURES = ("age", "trestbps", "chol", "thalach", "oldpeak")


def metrics_payload() -> bytes:
    return generate_latest(REGISTRY)


def record_request(method: str, route: str, status: int, seconds: float) -> None:
    HTTP_REQUESTS.labels(method=method, route=route, status=str(status)).inc()
    HTTP_LATENCY.labels(route=route).observe(seconds)


def record_prediction(label: str, probability: float, features: dict) -> None:
    PREDICTIONS.labels(label=label).inc()
    PROBABILITY.observe(probability)
    for name in MONITORED_FEATURES:
        value = features.get(name)
        if value is not None:
            FEATURES.labels(feature=name).observe(float(value))


# ---------------------------------------------------------------- logging


class JsonFormatter(logging.Formatter):
    """Render a log record as one JSON line, including any `extra={"fields": {...}}`."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        entry.update(getattr(record, "fields", {}) or {})
        if record.exc_info:
            entry["error"] = self.formatException(record.exc_info).splitlines()[-1]
        return json.dumps(entry, default=str)


def get_logger(name: str = "heart_api", level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not any(getattr(h, "_heart_api", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        handler._heart_api = True  # mark so repeated app creation does not duplicate output
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = True   # lets pytest's caplog see the records
    return logger


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start
        return False
