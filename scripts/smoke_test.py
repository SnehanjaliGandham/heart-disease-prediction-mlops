"""Smoke-test a running prediction API (Docker container, Kubernetes service or local).

Standard library only, so it runs anywhere Python does:
    python scripts/smoke_test.py                              # http://localhost:8000
    python scripts/smoke_test.py --url http://localhost:8080  # e.g. Kubernetes service
    python scripts/smoke_test.py --wait 60                    # wait up to 60 s for /ready

Checks liveness, readiness, a high-risk and a low-risk prediction, a request with
missing values, a batch, input validation (expects HTTP 422) and the /metrics
endpoint. Prints each response and exits with code 1 if any check fails.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


def call(url: str, method: str = "GET", body: dict | None = None, timeout: float = 10):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json",
                                          "X-Request-ID": "smoke-test"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode(), dict(resp.headers)
    except urllib.error.HTTPError as err:
        return err.code, err.read().decode(), dict(err.headers)


def wait_until_ready(base: str, seconds: float) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            if call(f"{base}/ready", timeout=3)[0] == 200:
                return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(1)
    return False


def sample(name: str) -> dict:
    return json.loads((SAMPLES / name).read_text())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the prediction API")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--wait", type=float, default=30, help="seconds to wait for /ready")
    args = parser.parse_args(argv)
    base = args.url.rstrip("/")

    print(f"Target: {base}")
    if not wait_until_ready(base, args.wait):
        print(f"FAIL: {base}/ready did not return 200 within {args.wait:.0f} s")
        return 1

    failures = []

    def check(name, ok, status, body):
        mark = "PASS" if ok else "FAIL"
        print(f"\n[{mark}] {name}  (HTTP {status})")
        print("       " + (body if len(body) < 600 else body[:600] + " ..."))
        if not ok:
            failures.append(name)

    status, body, headers = call(f"{base}/health")
    check("GET /health", status == 200 and json.loads(body)["status"] == "ok", status, body)

    status, body, _ = call(f"{base}/ready")
    check("GET /ready", status == 200 and json.loads(body)["model_loaded"], status, body)

    status, body, headers = call(f"{base}/predict", "POST", sample("high_risk.json"))
    res = json.loads(body) if status == 200 else {}
    check("POST /predict (high-risk patient -> 1)",
          status == 200 and res.get("prediction") == 1
          and headers.get("X-Request-ID", headers.get("x-request-id")) == "smoke-test",
          status, body)

    status, body, _ = call(f"{base}/predict", "POST", sample("low_risk.json"))
    res = json.loads(body) if status == 200 else {}
    check("POST /predict (low-risk patient -> 0)",
          status == 200 and res.get("prediction") == 0, status, body)

    status, body, _ = call(f"{base}/predict", "POST", sample("missing_values.json"))
    check("POST /predict (ca and thal missing -> imputed)", status == 200, status, body)

    status, body, _ = call(f"{base}/predict/batch", "POST", sample("batch.json"))
    res = json.loads(body) if status == 200 else {}
    check("POST /predict/batch (3 patients)", status == 200 and res.get("count") == 3,
          status, body)

    status, body, _ = call(f"{base}/predict", "POST", sample("invalid.json"))
    fields = {e["field"] for e in json.loads(body).get("detail", [])} if status == 422 else set()
    check("POST /predict (invalid input -> 422)",
          status == 422 and {"age", "cp", "thal"} <= fields, status, body)

    status, body, _ = call(f"{base}/metrics")
    check("GET /metrics", status == 200 and "predictions_total" in body, status,
          "\n       ".join(line for line in body.splitlines()
                           if line.startswith(("predictions_total", "prediction_errors_total",
                                               "model_info")))[:600])

    print("\n" + ("All checks passed." if not failures
                  else f"{len(failures)} check(s) failed: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
