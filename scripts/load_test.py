"""Send concurrent prediction requests to the API and report throughput and latency.

Standard library only. Used to demonstrate autoscaling (Kubernetes HPA) and to put
traffic on the monitoring dashboards:
    python scripts/load_test.py                                   # 60 s, 8 workers, :8000
    python scripts/load_test.py --url http://localhost:8080 --duration 180 --workers 16

Requests mix a high-risk, a low-risk and a missing-values patient, plus about 5 %
invalid requests so that error metrics are not empty.
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

SAMPLES = Path(__file__).resolve().parents[1] / "samples"
BODIES = [json.loads((SAMPLES / f"{n}.json").read_text())
          for n in ("high_risk", "low_risk", "missing_values")]
INVALID = json.loads((SAMPLES / "invalid.json").read_text())


def send(url: str, body: dict) -> tuple[int, float, str]:
    req = urllib.request.Request(f"{url}/predict", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            resp.read()
            status = resp.status
    except urllib.error.HTTPError as err:
        status = err.code
    except (urllib.error.URLError, OSError):
        status = 0                                   # connection error / timeout
    return status, time.perf_counter() - start, ""


def run(url: str, duration: float, workers: int, report_every: float = 10) -> dict:
    stop_at = time.time() + duration
    lock = threading.Lock()
    statuses: Counter = Counter()
    latencies: list[float] = []

    def worker(n: int):
        i = n
        while time.time() < stop_at:
            body = INVALID if i % 20 == 0 else BODIES[i % len(BODIES)]
            status, seconds, _ = send(url, body)
            with lock:
                statuses[status] += 1
                latencies.append(seconds)
            i += workers

    threads = [threading.Thread(target=worker, args=(n,), daemon=True) for n in range(workers)]
    started = time.time()
    for t in threads:
        t.start()
    next_report = started + report_every
    while any(t.is_alive() for t in threads):
        time.sleep(0.2)
        if time.time() >= next_report:
            with lock:
                done = sum(statuses.values())
            print(f"  {time.time() - started:5.0f} s  {done} requests sent")
            next_report += report_every
    elapsed = time.time() - started
    return summarise(statuses, latencies, elapsed)


def summarise(statuses: Counter, latencies: list[float], elapsed: float) -> dict:
    total = sum(statuses.values())
    ordered = sorted(latencies) or [0.0]

    def pct(p):
        return ordered[min(len(ordered) - 1, int(p / 100 * len(ordered)))] * 1000

    return {
        "requests": total,
        "seconds": round(elapsed, 1),
        "requests_per_second": round(total / elapsed, 1) if elapsed else 0.0,
        "status_codes": dict(sorted(statuses.items())),
        "latency_ms": {"mean": round(statistics.fmean(ordered) * 1000, 1),
                       "p50": round(pct(50), 1), "p95": round(pct(95), 1),
                       "p99": round(pct(99), 1)},
        "connection_errors": statuses.get(0, 0),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Load-test the prediction API")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--duration", type=float, default=60, help="seconds")
    parser.add_argument("--workers", type=int, default=8, help="concurrent clients")
    args = parser.parse_args(argv)
    url = args.url.rstrip("/")
    print(f"Load test: {url}/predict  {args.workers} workers for {args.duration:.0f} s")
    result = run(url, args.duration, args.workers)
    print(json.dumps(result, indent=2))
    # 422s are intentional; anything else that is not 200 counts as a failure
    unexpected = {s: n for s, n in result["status_codes"].items() if s not in (200, 422)}
    if unexpected:
        print(f"Unexpected responses: {unexpected}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
