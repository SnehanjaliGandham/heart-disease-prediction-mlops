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
    model_training_feature_mean{feature}         training-data mean  } read from the model's
    model_training_feature_std{feature}          training-data std   } own StandardScaler
    model_info{model_name,model_version}         which model is being served
    process_*                                    CPU and memory of each API process / pod

Drift = (mean of recent inputs - training mean) / training std, computed in Prometheus.
"""
from __future__ import annotations

import json
import logging
import socket
import sys
import time
from datetime import datetime, timezone

from prometheus_client import CONTENT_TYPE_LATEST as PROMETHEUS_CONTENT_TYPE  # noqa: F401
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    ProcessCollector,
    Summary,
    disable_created_metrics,
    generate_latest,
)

disable_created_metrics()        # drop the *_created timestamp series (not used, add noise)
REGISTRY = CollectorRegistry()
ProcessCollector(registry=REGISTRY)   # per-pod CPU and memory (process_* metrics, Linux)

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
TRAINING_MEAN = Gauge(
    "model_training_feature_mean", "Mean of the feature in the training data",
    ["feature"], registry=REGISTRY)
TRAINING_STD = Gauge(
    "model_training_feature_std", "Standard deviation of the feature in the training data",
    ["feature"], registry=REGISTRY)

MONITORED_FEATURES = ("age", "trestbps", "chol", "thalach", "oldpeak")

# Container / pod name: shows which replica handled a request when several run behind
# a Kubernetes Service.
INSTANCE = socket.gethostname()


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


def set_training_baseline(model) -> dict:
    """Publish the training mean/std of the monitored features as a drift baseline.

    The values come from the StandardScaler fitted inside the saved pipeline, so the
    baseline always matches the model being served. Returns {} if the pipeline does
    not have that shape (the API keeps working, only the drift panel stays empty).
    """
    try:
        numeric = model.named_steps["preprocess"].named_transformers_["num"]
        scaler = numeric.named_steps["scale"]
        columns = list(model.named_steps["preprocess"].transformers_[0][2])
    except (AttributeError, KeyError, IndexError, TypeError):
        return {}
    baseline = {}
    for name, mean, std in zip(columns, scaler.mean_, scaler.scale_):
        if name in MONITORED_FEATURES:
            TRAINING_MEAN.labels(feature=name).set(float(mean))
            TRAINING_STD.labels(feature=name).set(float(std))
            baseline[name] = (round(float(mean), 3), round(float(std), 3))
    return baseline


# ---------------------------------------------------------------- logging


class JsonFormatter(logging.Formatter):
    """Render a log record as one JSON line, including any `extra={"fields": {...}}`."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "instance": INSTANCE,
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
