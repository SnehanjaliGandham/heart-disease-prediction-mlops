"""Kubernetes manifests (deployment/k8s): consistency with the app and production settings.

These checks catch mistakes that `kubectl apply` would accept but that break or weaken
the deployment: a probe pointing at a route the API does not have, a port mismatch,
labels that do not select the pods, a missing resource limit or security setting.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from src import config
from src.api.main import create_app

K8S = config.ROOT_DIR / "deployment" / "k8s"


def load_all() -> dict:
    docs = {}
    for path in sorted(K8S.glob("*.yaml")):
        for doc in yaml.safe_load_all(path.read_text()):
            if doc:
                docs[(doc["kind"], doc.get("metadata", {}).get("name"))] = doc
    return docs


@pytest.fixture(scope="module")
def docs():
    return load_all()


@pytest.fixture(scope="module")
def deployment(docs):
    return docs[("Deployment", "heart-disease-api")]


@pytest.fixture(scope="module")
def container(deployment):
    return deployment["spec"]["template"]["spec"]["containers"][0]


def test_kustomization_lists_every_manifest(docs):
    kustomization = docs[("Kustomization", None)]
    on_disk = {p.name for p in K8S.glob("*.yaml")} - {"kustomization.yaml"}
    assert set(kustomization["resources"]) == on_disk


def test_everything_is_in_one_namespace(docs):
    namespaced = [d for (kind, _), d in docs.items() if kind not in ("Namespace", "Kustomization")]
    assert {d["metadata"]["namespace"] for d in namespaced} == {"heart-disease"}
    assert ("Namespace", "heart-disease") in docs


def test_selectors_match_pod_labels(docs, deployment):
    pod_labels = deployment["spec"]["template"]["metadata"]["labels"]
    pdb = docs[("PodDisruptionBudget", "heart-disease-api")]
    selectors = [deployment["spec"]["selector"]["matchLabels"],
                 docs[("Service", "heart-disease-api")]["spec"]["selector"],
                 pdb["spec"]["selector"]["matchLabels"]]
    for selector in selectors:
        assert selector.items() <= pod_labels.items()


def test_ports_match_the_container_image(docs, container):
    dockerfile = (config.ROOT_DIR / "Dockerfile").read_text()
    exposed = int(re.search(r"^EXPOSE (\d+)", dockerfile, re.M).group(1))
    assert container["ports"][0]["containerPort"] == exposed
    service = docs[("Service", "heart-disease-api")]["spec"]
    assert service["type"] == "LoadBalancer"
    assert service["ports"][0]["targetPort"] == container["ports"][0]["name"]


def test_probes_use_routes_that_exist_in_the_api(container):
    routes = {r.path for r in create_app().routes}
    assert container["readinessProbe"]["httpGet"]["path"] == "/ready"
    assert container["livenessProbe"]["httpGet"]["path"] == "/health"
    for probe in ("startupProbe", "readinessProbe", "livenessProbe"):
        assert container[probe]["httpGet"]["path"] in routes


def test_prometheus_annotations_point_at_the_metrics_route(deployment, container):
    notes = deployment["spec"]["template"]["metadata"]["annotations"]
    assert notes["prometheus.io/scrape"] == "true"
    assert notes["prometheus.io/path"] == "/metrics"
    assert int(notes["prometheus.io/port"]) == container["ports"][0]["containerPort"]


def test_configmap_only_sets_variables_the_app_reads(docs):
    main_py = (config.ROOT_DIR / "src" / "api" / "main.py").read_text()
    settings = docs[("ConfigMap", "heart-disease-api-config")]["data"]
    for key in settings:
        assert f'"{key}"' in main_py, f"{key} is not read by the API"
    assert 0 <= float(settings["PREDICTION_THRESHOLD"]) <= 1


def test_resources_are_requested_and_limited(container):
    resources = container["resources"]
    for kind in ("requests", "limits"):
        assert {"cpu", "memory"} <= set(resources[kind])


def test_pods_run_locked_down(deployment, container):
    pod = deployment["spec"]["template"]["spec"]
    assert pod["securityContext"]["runAsNonRoot"] is True
    assert pod["securityContext"]["runAsUser"] == 10001       # same uid as the Dockerfile
    assert pod["automountServiceAccountToken"] is False
    sc = container["securityContext"]
    assert sc["allowPrivilegeEscalation"] is False
    assert sc["readOnlyRootFilesystem"] is True
    assert sc["capabilities"]["drop"] == ["ALL"]


def test_rolling_update_never_reduces_capacity(deployment, container):
    strategy = deployment["spec"]["strategy"]
    assert strategy["type"] == "RollingUpdate"
    assert strategy["rollingUpdate"]["maxUnavailable"] == 0
    assert container["lifecycle"]["preStop"]["sleep"]["seconds"] > 0


def test_autoscaler_targets_the_deployment(docs, deployment):
    hpa = docs[("HorizontalPodAutoscaler", "heart-disease-api")]["spec"]
    assert hpa["scaleTargetRef"]["name"] == deployment["metadata"]["name"]
    assert hpa["minReplicas"] == deployment["spec"]["replicas"] >= 2
    assert hpa["maxReplicas"] > hpa["minReplicas"]


def test_image_matches_docker_compose(container):
    compose = yaml.safe_load((config.ROOT_DIR / "docker-compose.yml").read_text())
    assert container["image"] == compose["services"]["api"]["image"]
    assert container["imagePullPolicy"] == "IfNotPresent"


def test_manifests_contain_no_secrets(docs):
    assert "Secret" not in {kind for kind, _ in docs}
    text = "".join(p.read_text() for p in Path(K8S).glob("*.yaml")).lower()
    assert not re.search(r"password|api[_-]?key|secret", text)


def test_optional_ingress_routes_to_the_api_service(docs):
    ingress = yaml.safe_load(
        (config.ROOT_DIR / "deployment" / "k8s-optional" / "ingress.yaml").read_text())
    assert ingress["metadata"]["namespace"] == "heart-disease"
    backend = ingress["spec"]["rules"][0]["http"]["paths"][0]["backend"]["service"]
    service = docs[("Service", backend["name"])]
    assert backend["port"]["name"] in {p["name"] for p in service["spec"]["ports"]}
