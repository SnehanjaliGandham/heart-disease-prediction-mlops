"""Check that the monitoring stack is working: Prometheus scrapes the API and loaded the
alert rules, and Grafana is up with the dashboard provisioned.

Standard library only:
    python scripts/check_monitoring.py                                    # Kubernetes ports
    python scripts/check_monitoring.py --prometheus http://localhost:9091 \
                                       --grafana http://localhost:3001        # Compose
    python scripts/check_monitoring.py --wait 120 --min-targets 2         # used in CI

Exits with code 1 if any check fails.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

JOB = "heart-disease-api"
DASHBOARD_UID = "heart-disease-api"


def get_json(url: str, timeout: float = 5):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def query(prometheus: str, expr: str) -> list:
    url = f"{prometheus}/api/v1/query?" + urllib.parse.urlencode({"query": expr})
    return get_json(url)["data"]["result"]


def api_targets(prometheus: str) -> list[dict]:
    targets = get_json(f"{prometheus}/api/v1/targets")["data"]["activeTargets"]
    return [t for t in targets if t["labels"].get("job") == JOB]


def wait_for_targets(prometheus: str, minimum: int, seconds: float) -> list[dict]:
    deadline = time.time() + seconds
    targets: list[dict] = []
    while True:
        try:
            targets = api_targets(prometheus)
            if sum(t["health"] == "up" for t in targets) >= minimum:
                return targets
        except (urllib.error.URLError, OSError, KeyError, ValueError):
            pass
        if time.time() >= deadline:
            return targets
        time.sleep(2)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Check Prometheus and Grafana")
    parser.add_argument("--prometheus", default="http://localhost:9090")
    parser.add_argument("--grafana", default="http://localhost:3000")
    parser.add_argument("--min-targets", type=int, default=1,
                        help="API pods/containers that must be scraped successfully")
    parser.add_argument("--wait", type=float, default=30, help="seconds to wait for targets")
    args = parser.parse_args(argv)
    prom, graf = args.prometheus.rstrip("/"), args.grafana.rstrip("/")
    failures = []

    def check(name: str, ok: bool, detail: str = ""):
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  - {detail}" if detail else ""))
        if not ok:
            failures.append(name)

    # ---------------------------------------------------------------- Prometheus
    targets = wait_for_targets(prom, args.min_targets, args.wait)
    up = [t for t in targets if t["health"] == "up"]
    check(f"Prometheus scrapes the API ({len(up)}/{len(targets)} targets up)",
          len(up) >= args.min_targets,
          ", ".join(t["labels"].get("pod") or t["scrapeUrl"] for t in targets))
    try:
        groups = get_json(f"{prom}/api/v1/rules")["data"]["groups"]
        alerts = [r["name"] for g in groups for r in g["rules"] if r["type"] == "alerting"]
        records = [r["name"] for g in groups for r in g["rules"] if r["type"] == "recording"]
        check("Alert and recording rules loaded", len(alerts) > 0 and len(records) > 0,
              f"{len(alerts)} alerts, {len(records)} recording rules")
        firing = [r["name"] for g in groups for r in g["rules"]
                  if r["type"] == "alerting" and r.get("state") == "firing"]
        print(f"       alerts firing now: {', '.join(firing) if firing else 'none'}")
        baseline = query(prom, "model_training_feature_mean")
        check("Training baseline exported (drift reference)", len(baseline) > 0,
              ", ".join(sorted({b["metric"]["feature"] for b in baseline})))
    except (urllib.error.URLError, OSError, KeyError, ValueError) as err:
        check("Prometheus API reachable", False, str(err))

    # ---------------------------------------------------------------- Grafana
    try:
        health = get_json(f"{graf}/api/health")
        check("Grafana is up", health.get("database") == "ok",
              f"version {health.get('version', '?')}")
        dash = get_json(f"{graf}/api/dashboards/uid/{DASHBOARD_UID}")["dashboard"]
        panels = [p for p in dash["panels"] if p["type"] != "row"]
        check("Dashboard provisioned", len(panels) > 0, f"'{dash['title']}', {len(panels)} panels")
        # query Prometheus *through* Grafana: proves the data source connection works
        proxied = get_json(f"{graf}/api/datasources/proxy/uid/prometheus/api/v1/query?"
                           + urllib.parse.urlencode({"query": f'up{{job="{JOB}"}}'}))
        series = proxied["data"]["result"]
        check("Grafana can query Prometheus", proxied.get("status") == "success" and series,
              f"{len(series)} API target(s) visible from Grafana")
    except (urllib.error.URLError, OSError, KeyError, ValueError) as err:
        check("Grafana reachable", False, str(err))

    print("\n" + ("All monitoring checks passed." if not failures
                  else f"{len(failures)} check(s) failed: {failures}"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
