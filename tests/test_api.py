"""Prediction API (src/api): contract, validation, errors, logging and metrics.

The app is started with a model trained during the test session (not the committed
file), so these tests run on every Python / scikit-learn version in the CI matrix.
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys

import joblib
import pytest
import sklearn
from fastapi.testclient import TestClient

from src import config
from src.api import observability as obs
from src.api.main import create_app
from src.api.schemas import MAX_BATCH, PatientFeatures
from tests.conftest import HEALTHY_PATIENT, SICK_PATIENT


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory, fitted_lr):
    folder = tmp_path_factory.mktemp("served_model")
    joblib.dump(fitted_lr, folder / "heart_disease_model.joblib")
    meta = json.loads(config.MODEL_METADATA_PATH.read_text())
    meta.update({"sklearn_version": sklearn.__version__, "git_commit": "test123"})
    (folder / config.MODEL_METADATA_PATH.name).write_text(json.dumps(meta))
    return folder


@pytest.fixture(scope="module")
def client(model_dir):
    app = create_app(model_path=model_dir / "heart_disease_model.joblib")
    with TestClient(app) as c:
        yield c


def metric_value(client, name, **labels):
    """Read one sample value from /metrics (0.0 if absent)."""
    want = ",".join(f'{k}="{v}"' for k, v in labels.items())
    prefix = f"{name}{{{want}}}" if labels else name
    for line in client.get("/metrics").text.splitlines():
        if line.startswith(prefix + " "):
            return float(line.split()[-1])
    return 0.0


# ------------------------------------------------------------------ service endpoints


def test_root_lists_the_endpoints(client):
    body = client.get("/").json()
    assert body["predict"] == "/predict" and body["docs"] == "/docs"


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_ready_reports_the_loaded_model(client):
    r = client.get("/ready")
    assert r.status_code == 200
    assert r.json() == {"status": "ready", "model_loaded": True,
                        "model_version": "logistic_regression-test123"}


def test_model_info(client):
    info = client.get("/model-info").json()
    assert info["model_name"] == "logistic_regression"
    assert info["decision_threshold"] == 0.5
    assert info["features"] == config.FEATURES
    assert info["test_metrics"]["test_roc_auc"] > 0.85
    assert info["served_by"] == obs.INSTANCE


def test_openapi_docs_are_published(client):
    spec = client.get("/openapi.json").json()
    assert {"/predict", "/predict/batch", "/health", "/ready"} <= set(spec["paths"])


# ------------------------------------------------------------------ predictions


def test_sick_patient_is_predicted_as_disease(client):
    r = client.post("/predict", json=SICK_PATIENT)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] == 1 and body["label"] == "Heart disease"
    assert body["probability_disease"] > 0.5
    assert body["confidence"] == body["probability_disease"]
    assert body["model_version"] == "logistic_regression-test123"


def test_healthy_patient_is_predicted_as_no_disease(client):
    body = client.post("/predict", json=HEALTHY_PATIENT).json()
    assert body["prediction"] == 0
    assert body["confidence"] == pytest.approx(1 - body["probability_disease"], abs=1e-4)


def test_missing_ca_and_thal_are_accepted(client):
    for patient in ({**SICK_PATIENT, "ca": None, "thal": None},
                    {k: v for k, v in SICK_PATIENT.items() if k not in ("ca", "thal")}):
        r = client.post("/predict", json=patient)
        assert r.status_code == 200, r.text


def test_api_matches_the_python_predict_function(client, fitted_lr):
    from src.predict import predict
    expected = predict(SICK_PATIENT, model=fitted_lr)[0]["probability_disease"]
    assert client.post("/predict", json=SICK_PATIENT).json()["probability_disease"] == expected


def test_batch_prediction(client):
    r = client.post("/predict/batch", json={"patients": [SICK_PATIENT, HEALTHY_PATIENT]})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert [p["prediction"] for p in body["predictions"]] == [1, 0]


# ------------------------------------------------------------------ input validation


@pytest.mark.parametrize("field,value", [
    ("age", 500), ("age", 0), ("chol", 5), ("oldpeak", -1), ("ca", 4),
    ("cp", 9), ("sex", 2), ("thal", 5), ("slope", 0), ("restecg", 3),
])
def test_out_of_range_or_unknown_code_is_rejected(client, field, value):
    r = client.post("/predict", json={**SICK_PATIENT, field: value})
    assert r.status_code == 422
    assert field in {e["field"] for e in r.json()["detail"]}


def test_missing_required_field_is_rejected(client):
    patient = {k: v for k, v in SICK_PATIENT.items() if k != "thalach"}
    r = client.post("/predict", json=patient)
    assert r.status_code == 422
    assert r.json()["detail"][0]["field"] == "thalach"


def test_unknown_field_is_rejected(client):
    r = client.post("/predict", json={**SICK_PATIENT, "patient_name": "x"})
    assert r.status_code == 422


def test_wrong_type_is_rejected(client):
    assert client.post("/predict", json={**SICK_PATIENT, "age": "old"}).status_code == 422


def test_malformed_json_is_rejected(client):
    r = client.post("/predict", content=b"{not json",
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 422


def test_batch_size_limits(client):
    assert client.post("/predict/batch", json={"patients": []}).status_code == 422
    too_many = {"patients": [SICK_PATIENT] * (MAX_BATCH + 1)}
    assert client.post("/predict/batch", json=too_many).status_code == 422


def test_error_response_has_no_stack_trace(client):
    text = client.post("/predict", json={**SICK_PATIENT, "cp": 9}).text
    assert "Traceback" not in text and "request_id" in text


def test_schema_limits_match_project_config():
    """API validation and data validation use the same clinical ranges and codes."""
    props = PatientFeatures.model_json_schema()["properties"]
    for feature, (low, high) in config.NUMERIC_RANGES.items():
        bounds = props[feature].get("anyOf", [props[feature]])[0]
        assert (bounds["minimum"], bounds["maximum"]) == (low, high), feature
    for feature, codes in config.CATEGORY_CODES.items():
        enum = props[feature].get("anyOf", [props[feature]])[0]["enum"]
        assert sorted(enum) == codes, feature


# ------------------------------------------------------------------ tracing, logging, metrics


def test_request_id_is_generated_or_propagated(client):
    assert len(client.get("/health").headers["X-Request-ID"]) == 16
    r = client.post("/predict", json=SICK_PATIENT, headers={"X-Request-ID": "abc-123"})
    assert r.headers["X-Request-ID"] == "abc-123" and r.json()["request_id"] == "abc-123"


def test_metrics_use_route_templates_not_raw_urls(client):
    client.get("/docs")
    client.get("/no-such-page-123")
    text = client.get("/metrics").text
    assert 'route="/docs"' in text
    assert 'route="unmatched",status="404"' in text and "no-such-page" not in text


def test_unsafe_request_id_is_replaced(client):
    r = client.get("/health", headers={"X-Request-ID": "bad id <script>"})
    assert r.headers["X-Request-ID"] != "bad id <script>"


def test_logs_are_json_and_contain_no_patient_data(client, caplog):
    caplog.set_level(logging.INFO, logger="heart_api")
    client.post("/predict", json={**SICK_PATIENT, "chol": 286.5},
                headers={"X-Request-ID": "log-check"})
    records = [r for r in caplog.records if getattr(r, "fields", {}).get("request_id")
               == "log-check"]
    assert {r.fields["event"] for r in records} == {"prediction", "request"}
    for record in records:
        line = obs.JsonFormatter().format(record)
        entry = json.loads(line)               # every line is valid JSON
        assert entry["instance"] == obs.INSTANCE
        assert "286.5" not in line and "chol" not in entry
    request_log = next(r for r in records if r.fields["event"] == "request")
    assert request_log.fields["status"] == 200 and request_log.fields["duration_ms"] >= 0


def test_metrics_count_predictions_errors_and_requests(client):
    before = metric_value(client, "predictions_total", label="Heart disease")
    errors_before = metric_value(client, "prediction_errors_total", reason="validation")
    client.post("/predict", json=SICK_PATIENT)
    client.post("/predict", json={**SICK_PATIENT, "cp": 9})
    assert metric_value(client, "predictions_total", label="Heart disease") == before + 1
    assert metric_value(client, "prediction_errors_total", reason="validation") == \
        errors_before + 1
    text = client.get("/metrics").text
    assert 'http_requests_total{method="POST",route="/predict",status="200"}' in text
    assert "http_request_duration_seconds_bucket" in text
    assert 'input_feature_value_count{feature="chol"}' in text
    assert 'model_info{model_name="logistic_regression"' in text


# ------------------------------------------------------------------ failure modes


def test_missing_model_means_not_ready_and_503(tmp_path):
    app = create_app(model_path=tmp_path / "missing.joblib")
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200           # process is alive
        assert c.get("/ready").status_code == 503            # but not ready for traffic
        r = c.post("/predict", json=SICK_PATIENT)
        assert r.status_code == 503 and r.json()["detail"] == "Model not loaded"
        assert c.post("/predict/batch", json={"patients": [SICK_PATIENT]}).status_code == 503


def test_internal_error_returns_clean_500(model_dir, monkeypatch):
    app = create_app(model_path=model_dir / "heart_disease_model.joblib")

    def boom(*args, **kwargs):
        raise RuntimeError("model exploded")

    monkeypatch.setattr("src.api.main.predict", boom)
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/predict", json=SICK_PATIENT)
    assert r.status_code == 500
    assert r.json()["detail"] == "Internal server error" and "exploded" not in r.text


def test_threshold_is_configurable(model_dir, monkeypatch):
    monkeypatch.setenv("PREDICTION_THRESHOLD", "0.99")
    app = create_app(model_path=model_dir / "heart_disease_model.joblib")
    with TestClient(app) as c:
        assert c.get("/model-info").json()["decision_threshold"] == 0.99
        body = c.post("/predict", json=HEALTHY_PATIENT).json()
        assert body["prediction"] == 0


def test_invalid_threshold_is_refused(monkeypatch):
    monkeypatch.setenv("PREDICTION_THRESHOLD", "1.5")
    with pytest.raises(ValueError, match="PREDICTION_THRESHOLD"):
        create_app()


def test_version_mismatch_is_logged_but_model_still_served(model_dir, tmp_path, caplog):
    joblib.dump(joblib.load(model_dir / "heart_disease_model.joblib"),
                tmp_path / "heart_disease_model.joblib")
    meta = json.loads((model_dir / config.MODEL_METADATA_PATH.name).read_text())
    meta["sklearn_version"] = "0.0.1"
    (tmp_path / config.MODEL_METADATA_PATH.name).write_text(json.dumps(meta))
    caplog.set_level(logging.WARNING, logger="heart_api")
    with TestClient(create_app(model_path=tmp_path / "heart_disease_model.joblib")) as c:
        assert c.get("/ready").status_code == 200
    assert any(getattr(r, "fields", {}).get("event") == "version_mismatch"
               for r in caplog.records)


def test_serving_code_does_not_import_the_training_stack():
    """Keeps the Docker image small: the API needs neither MLflow nor matplotlib."""
    code = ("import sys, src.api.main; "
            "print(','.join(m for m in ('mlflow', 'matplotlib', 'seaborn') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], cwd=config.ROOT_DIR,
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == ""


def test_probe_requests_are_logged_at_debug_level(client, caplog):
    caplog.set_level(logging.DEBUG, logger="heart_api")
    client.get("/ready", headers={"X-Request-ID": "probe-check"})
    record = next(r for r in caplog.records
                  if getattr(r, "fields", {}).get("request_id") == "probe-check")
    assert record.levelno == logging.DEBUG
