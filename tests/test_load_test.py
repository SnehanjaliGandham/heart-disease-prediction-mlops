"""scripts/load_test.py against a tiny local HTTP server (no Docker needed)."""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from scripts import load_test


class FakeApi(BaseHTTPRequestHandler):
    def do_POST(self):                                   # noqa: N802 - http.server API
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        status = 422 if body.get("age", 0) > 120 else self.server.status
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"prediction": 1}')

    def log_message(self, *args):
        pass


@pytest.fixture()
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), FakeApi)
    httpd.status = 200
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()


def url(httpd):
    return f"http://127.0.0.1:{httpd.server_address[1]}"


def test_healthy_api_passes(server, capsys):
    assert load_test.main(["--url", url(server), "--duration", "0.6", "--workers", "3"]) == 0
    out = capsys.readouterr().out
    result = json.loads(out[out.index("{"):])
    assert result["requests"] > 0 and result["connection_errors"] == 0
    assert set(result["status_codes"]) <= {"200", "422"}      # JSON keys
    assert result["latency_ms"]["p50"] <= result["latency_ms"]["p95"] <= result["latency_ms"]["p99"]


def test_invalid_requests_are_mixed_in(server):
    result = load_test.run(url(server), duration=0.6, workers=1)
    assert result["status_codes"].get(422, 0) >= 1          # every 20th request


def test_server_errors_fail_the_run(server, capsys):
    server.status = 500
    assert load_test.main(["--url", url(server), "--duration", "0.3", "--workers", "2"]) == 1
    assert "Unexpected responses" in capsys.readouterr().out


def test_unreachable_api_counts_connection_errors():
    status, seconds, _ = load_test.send("http://127.0.0.1:9", load_test.BODIES[0])
    assert status == 0 and seconds >= 0


def test_progress_is_reported(server, capsys):
    load_test.run(url(server), duration=0.5, workers=1, report_every=0.2)
    assert "requests sent" in capsys.readouterr().out


def test_summary_of_no_requests():
    from collections import Counter
    result = load_test.summarise(Counter(), [], 0)
    assert result["requests"] == 0 and result["requests_per_second"] == 0.0
