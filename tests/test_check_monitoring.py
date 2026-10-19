"""scripts/check_monitoring.py against a fake Prometheus + Grafana HTTP server."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import pytest

from scripts import check_monitoring as cm

HEALTHY = {
    "/api/v1/targets": {"data": {"activeTargets": [
        {"labels": {"job": "heart-disease-api", "pod": "api-1"}, "health": "up",
         "scrapeUrl": "http://10.0.0.1:8000/metrics"},
        {"labels": {"job": "heart-disease-api", "pod": "api-2"}, "health": "up",
         "scrapeUrl": "http://10.0.0.2:8000/metrics"},
        {"labels": {"job": "prometheus"}, "health": "up", "scrapeUrl": "http://localhost:9090"},
    ]}},
    "/api/v1/rules": {"data": {"groups": [{"rules": [
        {"type": "recording", "name": "feature:drift_score:15m"},
        {"type": "alerting", "name": "HeartApiDown", "state": "inactive"},
        {"type": "alerting", "name": "HeartApiInputDrift", "state": "firing"},
    ]}]}},
    "/api/v1/query": {"status": "success", "data": {"result": [
        {"metric": {"feature": "age"}, "value": [0, "54.5"]}]}},
    "/api/health": {"database": "ok", "version": "12.4.12"},
    "/api/dashboards/uid/heart-disease-api": {"dashboard": {
        "title": "Heart Disease API", "panels": [{"type": "row"}, {"type": "stat"}]}},
    "/api/datasources/proxy/uid/prometheus/api/v1/query": {
        "status": "success", "data": {"result": [{"metric": {"pod": "api-1"}}]}},
}


class Fake(BaseHTTPRequestHandler):
    def do_GET(self):                                        # noqa: N802 - http.server API
        body = self.server.routes.get(urlparse(self.path).path)
        if body is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode())

    def log_message(self, *args):
        pass


@pytest.fixture()
def fake():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    httpd.routes = dict(HEALTHY)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd
    httpd.shutdown()


def args(httpd, *extra):
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    return ["--prometheus", url, "--grafana", url, "--wait", "0.5", *extra]


def test_healthy_stack_passes(fake, capsys):
    assert cm.main(args(fake, "--min-targets", "2")) == 0
    out = capsys.readouterr().out
    assert "2/2 targets up" in out and "alerts firing now: HeartApiInputDrift" in out
    assert "All monitoring checks passed." in out


def test_too_few_targets_fails(fake, capsys):
    assert cm.main(args(fake, "--min-targets", "3")) == 1
    assert "[FAIL] Prometheus scrapes the API" in capsys.readouterr().out


def test_missing_dashboard_fails(fake, capsys):
    del fake.routes["/api/dashboards/uid/heart-disease-api"]
    assert cm.main(args(fake)) == 1
    assert "[FAIL] Grafana reachable" in capsys.readouterr().out


def test_nothing_running_fails(capsys):
    assert cm.main(["--prometheus", "http://127.0.0.1:9", "--grafana", "http://127.0.0.1:9",
                    "--wait", "0.1"]) == 1
    out = capsys.readouterr().out
    assert "[FAIL] Prometheus API reachable" in out and "[FAIL] Grafana reachable" in out
