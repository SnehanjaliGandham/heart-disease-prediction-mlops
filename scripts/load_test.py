"""Send concurrent prediction requests to the API and report throughput and latency.

Standard library only. Used to demonstrate autoscaling (Kubernetes HPA) and to put
traffic on the monitoring dashboards:
    python scripts/load_test.py                                   # 60 s, 8 workers, :8000
    python scripts/load_test.py --url http://localhost:8080 --duration 180 --workers 16
    python scripts/load_test.py --url http://localhost:8080 --profile drifted   # drift demo

Patients are real records from data/heart_disease_clean.csv, so the traffic has the
same distribution as the training data. `--profile drifted` makes them older, with
higher cholesterol and blood pressure and a lower maximum heart rate - the kind of
population change the drift panel and alert in Grafana/Prometheus should catch.
About 5 % of requests are deliberately invalid so that error metrics are not empty.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"
DATASET = ROOT / "data" / "heart_disease_clean.csv"
BODIES = [json.loads((SAMPLES / f"{n}.json").read_text())
          for n in ("high_risk", "low_risk", "missing_values")]
INVALID = json.loads((SAMPLES / "invalid.json").read_text())
INTEGER_FIELDS = ("age", "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal")
# drifted population: (shift, lower limit, upper limit) - limits keep requests valid
DRIFT = {"age": (15, 1, 120), "trestbps": (15, 50, 250), "chol": (60, 50, 700),
         "thalach": (-25, 50, 250), "oldpeak": (1.0, 0, 10)}


def load_patients(path: Path = DATASET) -> list[dict]:
    """Patient records (without the label) from the cleaned dataset, or the samples."""
    if not path.exists():
        return list(BODIES)
    patients = []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            row.pop("target", None)
            patients.append({k: (None if v == "" else
                                 int(float(v)) if k in INTEGER_FIELDS else float(v))
                             for k, v in row.items()})
    return patients


def drifted(patient: dict) -> dict:
    out = dict(patient)
    for field, (shift, low, high) in DRIFT.items():
        value = out[field] + shift
        out[field] = int(min(max(value, low), high)) if field == "age" else \
            round(min(max(value, low), high), 1)
    return out


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


def run(url: str, duration: float, workers: int, report_every: float = 10,
        profile: str = "normal", patients: list | None = None) -> dict:
    patients = patients or load_patients()
    if profile == "drifted":
        patients = [drifted(p) for p in patients]
    stop_at = time.time() + duration
    lock = threading.Lock()
    statuses: Counter = Counter()
    latencies: list[float] = []

    def worker(n: int):
        rng = random.Random(n)
        i = n
        while time.time() < stop_at:
            body = INVALID if i % 20 == 0 else rng.choice(patients)
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
    parser.add_argument("--profile", choices=["normal", "drifted"], default="normal",
                        help="normal = patients as in the training data; drifted = shifted")
    args = parser.parse_args(argv)
    url = args.url.rstrip("/")
    print(f"Load test: {url}/predict  {args.workers} workers for {args.duration:.0f} s"
          f"  ({args.profile} patients)")
    result = run(url, args.duration, args.workers, profile=args.profile)
    print(json.dumps(result, indent=2))
    # 422s are intentional; anything else that is not 200 counts as a failure
    unexpected = {s: n for s, n in result["status_codes"].items() if s not in (200, 422)}
    if unexpected:
        print(f"Unexpected responses: {unexpected}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
