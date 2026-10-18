"""scripts/smoke_test.py: run its checks against the app in-process (no server needed)."""
from __future__ import annotations

import json
import urllib.error

import joblib
import pytest
import sklearn
from fastapi.testclient import TestClient

from scripts import smoke_test
from src import config
from src.api.main import create_app


@pytest.fixture()
def routed_to(tmp_path, monkeypatch):
    """Point the smoke test's HTTP helper at a TestClient for the given app."""

    def install(client):
        def call(url, method="GET", body=None, timeout=10):
            path = url.split("://", 1)[-1].split("/", 1)[-1]
            r = client.request(method, "/" + path, json=body,
                               headers={"X-Request-ID": "smoke-test"})
            return r.status_code, r.text, dict(r.headers)
        monkeypatch.setattr(smoke_test, "call", call)
        monkeypatch.setattr(smoke_test.time, "sleep", lambda s: None)
    return install


def make_app(folder, fitted_lr):
    joblib.dump(fitted_lr, folder / "heart_disease_model.joblib")
    meta = json.loads(config.MODEL_METADATA_PATH.read_text())
    meta["sklearn_version"] = sklearn.__version__
    (folder / config.MODEL_METADATA_PATH.name).write_text(json.dumps(meta))
    return create_app(model_path=folder / "heart_disease_model.joblib")


def test_smoke_test_passes_against_a_healthy_api(tmp_path, fitted_lr, routed_to, capsys):
    with TestClient(make_app(tmp_path, fitted_lr)) as client:
        routed_to(client)
        assert smoke_test.main(["--url", "http://api:8000/", "--wait", "2"]) == 0
    out = capsys.readouterr().out
    assert out.count("[PASS]") == 8 and "All checks passed." in out


def test_smoke_test_fails_when_the_model_is_not_loaded(tmp_path, routed_to, capsys):
    with TestClient(create_app(model_path=tmp_path / "missing.joblib")) as client:
        routed_to(client)
        assert smoke_test.main(["--wait", "0.2"]) == 1
    assert "did not return 200" in capsys.readouterr().out


def test_smoke_test_reports_failed_checks(tmp_path, fitted_lr, routed_to, monkeypatch, capsys):
    with TestClient(make_app(tmp_path, fitted_lr)) as client:
        routed_to(client)
        real_sample = smoke_test.sample
        # a "low-risk" request that is really high-risk must make that check fail
        monkeypatch.setattr(smoke_test, "sample", lambda name: real_sample(
            "high_risk.json" if name == "low_risk.json" else name))
        assert smoke_test.main(["--wait", "2"]) == 1
    out = capsys.readouterr().out
    assert "[FAIL] POST /predict (low-risk patient -> 0)" in out and "1 check(s) failed" in out


def test_unreachable_server_is_retried_until_timeout(monkeypatch):
    calls = []

    def refuse(*args, **kwargs):
        calls.append(1)
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr(smoke_test, "call", refuse)
    monkeypatch.setattr(smoke_test.time, "sleep", lambda s: None)
    clock = iter(range(100))
    monkeypatch.setattr(smoke_test.time, "time", lambda: next(clock))
    assert smoke_test.wait_until_ready("http://nowhere", 5) is False
    assert len(calls) >= 3


def test_call_handles_success_and_http_errors(monkeypatch):
    class Resp:
        status = 200
        headers = {"X-Request-ID": "smoke-test"}

        def read(self):
            return b'{"status": "ok"}'

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(smoke_test.urllib.request, "urlopen", lambda req, timeout: Resp())
    assert smoke_test.call("http://x/health")[0:2] == (200, '{"status": "ok"}')

    def http_error(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 422, "bad", {"a": "b"},
                                     __import__("io").BytesIO(b'{"detail": []}'))
    monkeypatch.setattr(smoke_test.urllib.request, "urlopen", http_error)
    assert smoke_test.call("http://x/predict", "POST", {"age": 1})[0] == 422
