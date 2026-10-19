"""Monitoring configuration (deployment/monitoring): dashboard, alert rules, Prometheus.

The most common monitoring bug is silent: a dashboard panel or alert that queries a
metric name the application does not export, so it shows "No data" or never fires.
These tests check every metric used in the dashboard and the rules against what the
API really exposes. The alert logic itself is unit-tested with `promtool test rules`
(alerts_test.yml) in CI.
"""
from __future__ import annotations

import json
import re

import pytest
import yaml
from fastapi.testclient import TestClient

from src import config
from src.api import observability as obs
from src.api.main import create_app
from tests.conftest import SICK_PATIENT

MON = config.ROOT_DIR / "deployment" / "monitoring"
DASHBOARD = MON / "grafana" / "dashboards" / "heart-disease-api.json"
RULES = MON / "prometheus" / "alerts.yml"

PROMQL_WORDS = {
    "sum", "by", "rate", "increase", "max", "min", "count", "abs", "on", "and", "or",
    "histogram_quantile", "clamp_min", "vector", "absent", "le", "without", "round",
}
METRIC_NAME = re.compile(r"\b([a-zA-Z_:][a-zA-Z0-9_:]*)\s*(?=\{|\[|\)|$|\s)")


@pytest.fixture(scope="module")
def dashboard():
    return json.loads(DASHBOARD.read_text())


@pytest.fixture(scope="module")
def rules():
    return yaml.safe_load(RULES.read_text())


@pytest.fixture(scope="module")
def exported_metrics(tmp_path_factory, fitted_lr):
    """Every series name the API exposes after serving one prediction."""
    import joblib
    folder = tmp_path_factory.mktemp("mon_model")
    joblib.dump(fitted_lr, folder / "heart_disease_model.joblib")
    (folder / config.MODEL_METADATA_PATH.name).write_text(
        config.MODEL_METADATA_PATH.read_text())
    with TestClient(create_app(model_path=folder / "heart_disease_model.joblib")) as c:
        c.post("/predict", json=SICK_PATIENT)
        c.post("/predict", json={**SICK_PATIENT, "cp": 9})
        text = c.get("/metrics").text
    return {line.split("{")[0].split(" ")[0] for line in text.splitlines()
            if line and not line.startswith("#")}


def metric_names(expr: str) -> set[str]:
    expr = re.sub(r"\{[^}]*\}", "{}", expr)           # drop label matchers
    expr = re.sub(r'"[^"]*"', "", expr)               # and string literals
    expr = re.sub(r"\b(by|on|without|ignoring)\s*\([^)]*\)", "", expr)   # label lists
    names = set(METRIC_NAME.findall(expr))
    return {n for n in names if n not in PROMQL_WORDS and not n.replace(".", "").isdigit()}


def recording_rules(rules) -> set[str]:
    return {r["record"] for g in rules["groups"] for r in g["rules"] if "record" in r}


def all_expressions(dashboard, rules):
    for panel in dashboard["panels"]:
        for target in panel.get("targets", []):
            yield f"panel '{panel['title']}'", target["expr"]
    for group in rules["groups"]:
        for rule in group["rules"]:
            yield f"rule {rule.get('alert') or rule.get('record')}", rule["expr"]


def test_every_queried_metric_is_exported_by_the_api(dashboard, rules, exported_metrics):
    # process_* metrics come from /proc, so they exist in the (Linux) container but not
    # when the tests run on Windows or macOS
    linux_only = {"process_cpu_seconds_total", "process_resident_memory_bytes"}
    known = exported_metrics | recording_rules(rules) | linux_only | {"up", "ALERTS"}
    for where, expr in all_expressions(dashboard, rules):
        missing = metric_names(expr) - known
        assert not missing, f"{where} queries unknown metric(s) {missing}"


def test_process_metrics_are_exported_on_linux(exported_metrics):
    import sys
    if not sys.platform.startswith("linux"):
        pytest.skip("process metrics are read from /proc (Linux only)")
    assert {"process_cpu_seconds_total", "process_resident_memory_bytes"} <= exported_metrics


def test_training_baseline_matches_the_model(fitted_lr):
    baseline = obs.set_training_baseline(fitted_lr)
    assert set(baseline) == set(obs.MONITORED_FEATURES)
    age_mean, age_std = baseline["age"]
    assert 40 < age_mean < 70 and 5 < age_std < 15
    assert obs.set_training_baseline(object()) == {}     # unusual model -> no crash


def test_dashboard_is_well_formed(dashboard):
    assert dashboard["uid"] == "heart-disease-api"
    ids = [p["id"] for p in dashboard["panels"]]
    assert len(ids) == len(set(ids)), "panel ids must be unique"
    for panel in dashboard["panels"]:
        pos = panel["gridPos"]
        assert pos["x"] + pos["w"] <= 24, panel["title"]
        if panel["type"] != "row":
            assert panel["datasource"]["uid"] == "prometheus", panel["title"]
            assert panel["targets"], panel["title"]


def test_dashboard_panels_do_not_overlap(dashboard):
    cells = {}
    for panel in dashboard["panels"]:
        pos = panel["gridPos"]
        for x in range(pos["x"], pos["x"] + pos["w"]):
            for y in range(pos["y"], pos["y"] + pos["h"]):
                assert (x, y) not in cells, f"{panel['title']} overlaps {cells[(x, y)]}"
                cells[(x, y)] = panel["title"]


def test_dashboard_covers_service_and_model_monitoring(dashboard):
    text = json.dumps(dashboard)
    for metric in ("http_requests_total", "http_request_duration_seconds_bucket",
                   "predictions_total", "feature:drift_score:15m", "process_cpu_seconds_total",
                   "model_info", "ALERTS"):
        assert metric in text


def test_datasource_uid_matches_the_dashboard():
    ds = yaml.safe_load((MON / "grafana/provisioning/datasources/prometheus.yml").read_text())
    assert ds["datasources"][0]["uid"] == "prometheus"
    assert ds["datasources"][0]["url"] == "http://prometheus:9090"


def test_every_alert_has_severity_summary_and_description(rules):
    alerts = [r for g in rules["groups"] for r in g["rules"] if "alert" in r]
    assert len(alerts) >= 6
    for alert in alerts:
        assert alert["labels"]["severity"] in {"critical", "warning", "info"}
        assert alert["annotations"]["summary"] and alert["annotations"]["description"]


def test_every_alert_is_unit_tested_or_simple(rules):
    tests = yaml.safe_load((MON / "prometheus" / "alerts_test.yml").read_text())
    tested = {t["alertname"] for case in tests["tests"] for t in case.get("alert_rule_test", [])}
    alerts = {r["alert"] for g in rules["groups"] for r in g["rules"] if "alert" in r}
    assert {"HeartApiDown", "HeartApiHighErrorRate", "HeartApiInputDrift",
            "HeartApiPredictionShift", "HeartApiNoTargets"} <= tested <= alerts


def test_both_prometheus_configs_scrape_the_api_with_the_same_job_and_rules():
    k8s = yaml.safe_load((MON / "prometheus" / "prometheus-k8s.yml").read_text())
    docker = yaml.safe_load((MON / "prometheus" / "prometheus-docker.yml").read_text())
    for cfg in (k8s, docker):
        assert cfg["rule_files"] == ["/etc/prometheus/rules/*.yml"]
        assert "heart-disease-api" in [j["job_name"] for j in cfg["scrape_configs"]]
    compose = yaml.safe_load((config.ROOT_DIR / "docker-compose.yml").read_text())
    target = docker["scrape_configs"][0]["static_configs"][0]["targets"][0]
    assert target.split(":")[0] in compose["services"]


def test_kubernetes_discovery_uses_the_pod_annotations():
    k8s = yaml.safe_load((MON / "prometheus" / "prometheus-k8s.yml").read_text())
    job = k8s["scrape_configs"][0]
    assert job["kubernetes_sd_configs"][0]["namespaces"]["names"] == ["heart-disease"]
    deployment = yaml.safe_load(
        (config.ROOT_DIR / "deployment" / "k8s" / "deployment.yaml").read_text())
    annotations = deployment["spec"]["template"]["metadata"]["annotations"]
    used = " ".join(str(r.get("source_labels")) for r in job["relabel_configs"])
    for key in annotations:
        meta = "__meta_kubernetes_pod_annotation_" + re.sub(r"[./-]", "_", key)
        assert meta in used


def test_monitoring_kustomization_files_exist():
    kustomization = yaml.safe_load((MON / "kustomization.yaml").read_text())
    for resource in kustomization["resources"]:
        assert (MON / resource).exists()
    for generator in kustomization["configMapGenerator"]:
        for entry in generator["files"]:
            assert (MON / entry.split("=")[-1]).exists(), entry


def test_prometheus_may_only_read_pods_in_the_api_namespace():
    docs = list(yaml.safe_load_all((MON / "prometheus.yaml").read_text()))
    roles = [d for d in docs if d["kind"] in ("Role", "ClusterRole")]
    assert [r["kind"] for r in roles] == ["Role"]
    assert roles[0]["metadata"]["namespace"] == "heart-disease"
    assert roles[0]["rules"][0]["verbs"] == ["get", "list", "watch"]
