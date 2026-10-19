# Heart Disease Prediction – End-to-End MLOps Pipeline

BITS Pilani WILP · M.Tech AIML · MLOps (AIMLCZG523) Assignment 01

[![CI/CD pipeline](https://github.com/SnehanjaliGandham/heart-disease-prediction-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/SnehanjaliGandham/heart-disease-prediction-mlops/actions/workflows/ci.yml)

End-to-end, reproducible ML pipeline for predicting heart disease from the
UCI Heart Disease (Cleveland) dataset: data acquisition → EDA → feature
engineering → model training & tuning → MLflow tracking → packaged model →
automated tests and CI/CD → containerised prediction API → Kubernetes deployment with autoscaling
→ monitoring with Prometheus and Grafana, including data-drift detection.

## Project structure

```
data/                      raw + cleaned CSV, UCI variable descriptions, SHA-256 manifest
scripts/download_data.py   downloads the dataset from UCI and verifies it against the manifest
scripts/check_env.py       checks a machine (e.g. Prayogshala) can run the project
scripts/quality_gate.py    blocks a model that falls below minimum metric standards
scripts/run_pipeline.sh    whole pipeline in one command (run_pipeline.ps1 on Windows)
scripts/smoke_test.py      end-to-end checks against a running API (Docker / Kubernetes)
scripts/load_test.py       traffic generator (real patient records, optional drifted population)
scripts/check_monitoring.py checks Prometheus scrapes the API and Grafana has the dashboard
src/
  config.py                paths, feature lists, constants (single source of truth)
  data.py                  load, validate, clean, split
  features.py              engineered feature (ClinicalFeatures) + preprocessing ColumnTransformer
  models.py                candidate models + hyper-parameter grids
  evaluate.py              metrics and evaluation plots
  train.py                 tuning, model selection, MLflow logging, packaging
  predict.py               load packaged model and score patient records
  api/                     FastAPI service: schemas, endpoints, JSON logging, Prometheus metrics
notebooks/
  01_data_acquisition_eda.ipynb   Task 1 – data acquisition and EDA
  02_model_development.ipynb      Tasks 2–4 – features, models, MLflow, packaging
  03_inference.ipynb              Tasks 4 and 6 – packaged model in Python and through the API
models/                    packaged champion model (+ MLflow format, metadata)
reports/                   model comparison table and figures for the report
samples/                   example request bodies (high/low risk, missing values, invalid, batch)
tests/                     pytest unit and API tests (Task 5)
.github/workflows/ci.yml   CI/CD pipeline: lint -> tests (Py 3.10-3.13) -> train + quality gate -> Docker -> Kubernetes
deployment/k8s/            Kubernetes manifests (Deployment, LoadBalancer Service, HPA, PDB, ConfigMap) (Task 7)
deployment/k8s-optional/   optional Ingress (needs an Ingress controller)
deployment/monitoring/     Prometheus (config, alert rules + rule tests) and Grafana (dashboard) (Task 8)
Dockerfile                 multi-stage, non-root image for the API (Task 6)
docker-compose.yml         API + Prometheus + Grafana with one command (no Kubernetes needed)
requirements-serve.txt     the only packages the API image installs
screenshots/               evidence for the report
```

## Setup

Works on **Python 3.9 – 3.13**, on Linux (e.g. the BITS Prayogshala lab), Windows and macOS.
Do **not** copy the `.venv` folder between machines; create a fresh one on each machine.

**Linux / Prayogshala**
```bash
python3 scripts/check_env.py          # stdlib only: checks Python, pip, network, docker, kubectl
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt       # newest versions your Python supports
python scripts/check_env.py           # re-run: every package should now be PASS
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`requirements-lock.txt` holds the exact versions used to produce the submitted results
(needs Python ≥ 3.12); `requirements.txt` gives compatible minimum versions.

## Run the pipeline

One command runs everything (environment check → data → cleaning → training + MLflow → quality gate →
sample prediction → lint + tests) and stops at the first failure:

```bash
bash scripts/run_pipeline.sh            # Linux / Prayogshala      (add --quick for small grids)
.\scripts\run_pipeline.ps1              # Windows PowerShell       (add -Quick)
```

Or step by step:

```bash
python scripts/download_data.py      # 1. get the data (skipped if already present)
python -m src.data                   # 2. raw -> data/heart_disease_clean.csv
python -m src.train                  # 3. tune + compare models, log to MLflow, save best model
python -m src.predict                # 4. score a sample patient with the saved model
mlflow ui --backend-store-uri sqlite:///mlflow.db   # 5. browse runs at http://127.0.0.1:5000
```

`python -m src.train --quick` runs with small grids (handy for quick local checks).

## Feature engineering and model selection

* **Preprocessing** inside each model `Pipeline` (re-fitted on every CV fold, so no leakage): median imputation +
  standard scaling for numeric features, most-frequent imputation + one-hot encoding for coded features.
* **Engineered feature** `hr_pct_predicted = thalach / (220 − age)`: peak heart rate as a share of the
  age-predicted maximum (a recognised marker of heart disease). It is added by a stateless transformer inside the
  pipeline, so the API still takes the 13 raw values, and it is a **hyper-parameter** (`features__add_hr_ratio`) of
  every model rather than an assumption. Result (`reports/feature_ablation.csv`): it separates the classes on its
  own (mean 0.85 with disease vs 0.95 without) but adds nothing once `thalach` and `age` are in the model, so the
  grid search leaves it out for all three models.
* **Selection rule, fixed before training** (`src/config.py`): rank models by mean 5-fold CV ROC-AUC; models within
  0.01 of the best count as equal and the simplest is chosen (LR → RF → GB). The test set is never used for selection.
* **Over-fitting check**: train-fold minus validation-fold ROC-AUC inside cross-validation.

## Results

| Model | CV ROC-AUC (5-fold) | Over-fitting gap | Test accuracy | Test recall | Test ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression (selected)** | **0.911 ± 0.018** | **0.015** | 0.885 | 0.929 | 0.961 |
| Random Forest | 0.900 ± 0.032 | 0.057 | 0.885 | 0.929 | 0.955 |
| Gradient Boosting | 0.884 ± 0.027 | 0.077 | 0.902 | 0.929 | 0.960 |
| Majority baseline | 0.500 | 0.000 | 0.541 | 0.000 | 0.500 |

Logistic Regression has the best and most stable CV score and over-fits least; the held-out test set (61 patients)
is used only for reporting.

## Testing and CI/CD

```bash
ruff check .                                   # lint
python -m pytest --cov                         # all tests, coverage report
python scripts/quality_gate.py                 # check the trained model meets minimum standards
```

The tests run offline in under a minute. They cover data validation and cleaning, the
preprocessing (including a leakage guard), every hyper-parameter grid, metrics and plots,
an end-to-end training run with checks on what MLflow recorded and on the model registry,
inference, the helper scripts and the prediction API (contract, input validation, error
handling, request IDs, JSON logs and metrics).

Every push and pull request to `main` runs the GitHub Actions pipeline:

| Job | What it does | Fails when |
|---|---|---|
| 1 · Lint | `ruff` static checks; `promtool` validates the Prometheus config and unit-tests the alert rules | style or likely-bug findings, invalid config, an alert that does not fire as specified |
| 2 · Unit tests | pytest on Python 3.10, 3.11, 3.12 and 3.13 with coverage | any test fails or coverage < 90 % |
| 3 · Train | verify data checksum → clean → train + MLflow → quality gate → inference smoke test → packaged-model tests → execute notebooks | any step errors or the model misses the quality gate |
| 4 · Docker | build the image with the model trained in job 3 → start the container → smoke test → wait for HEALTHCHECK = healthy → check it runs as non-root → push to GitHub Container Registry (pushes to `main` only) | build fails, any smoke check fails or the container is unhealthy |
| 5 · Kubernetes | create a temporary cluster (kind) → `kubectl apply -k deployment/k8s` → wait for the rollout → smoke test through the Service → delete a pod and check it is replaced → rolling update → deploy Prometheus + Grafana and check they monitor the API | a manifest is invalid, pods never become ready, a smoke check fails, self-healing / rollout fails or monitoring cannot see the API |

Each job uploads its results as artifacts (lint report, JUnit/coverage reports, packaged
model, MLflow store, logs, executed notebooks) and writes a summary to the run page.

## Prediction API (Docker)

The packaged model is served by a FastAPI application (`src/api/`) in a Docker image.

```bash
docker build -t heart-disease-api:1.0.0 .
docker run -d -p 8000:8000 --name heart-api heart-disease-api:1.0.0
python scripts/smoke_test.py                 # 8 end-to-end checks, exit code 1 on failure
```

or simply `docker compose up -d --build`. Interactive documentation: http://localhost:8000/docs

| Endpoint | Purpose |
|---|---|
| `POST /predict` | one patient (JSON) → `prediction`, `label`, `probability_disease`, `confidence`, `model_version`, `request_id` |
| `POST /predict/batch` | up to 100 patients in one call |
| `GET /health` | liveness – the process is running |
| `GET /ready` | readiness – the model is loaded (HTTP 503 otherwise) |
| `GET /model-info` | which model is served, its parameters and test metrics |
| `GET /metrics` | Prometheus metrics (requests, latency, predictions by class, errors, input drift) |

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @samples/high_risk.json
# PowerShell: Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType "application/json" -InFile samples/high_risk.json
```

Design points:

* **Input validation** – every field is checked against clinical ranges and valid UCI codes
  (the same values as `src/config.py`); unknown fields are refused. Errors return HTTP 422 with
  one message per field and never a stack trace. `ca` and `thal` may be null (imputed by the pipeline).
* **Observability** – one JSON log line per request with a request ID (taken from the
  `X-Request-ID` header or generated); patient values are never logged. Health-probe requests
  log at DEBUG so they do not flood the log.
* **Image** – multi-stage build, only the serving packages (no MLflow, Jupyter or plotting),
  runs as non-root user 10001 with read-only code and model, Docker `HEALTHCHECK` on `/ready`.
* **Configuration** – `PREDICTION_THRESHOLD` (default 0.5), `LOG_LEVEL`, `MODEL_PATH`.

## Kubernetes deployment

`deployment/k8s/` deploys the same image to any Kubernetes cluster. Tested on Docker Desktop's
built-in Kubernetes, and on a temporary kind cluster in CI (job 5).

```bash
docker build -t heart-disease-api:1.0.0 .         # Docker Desktop's Kubernetes uses this image directly
kubectl apply -k deployment/k8s                    # namespace, config, deployment, service, autoscaler, PDB
kubectl -n heart-disease rollout status deployment/heart-disease-api
kubectl -n heart-disease get all
python scripts/smoke_test.py --url http://localhost:8080
```

| Object | Purpose |
|---|---|
| `Deployment` | 2 replicas; rolling updates that never reduce capacity (`maxSurge 1`, `maxUnavailable 0`) and a 5 s `preStop` delay so no request is dropped while a pod shuts down |
| Probes | *startup* and *readiness* on `/ready` (traffic only reaches pods with a loaded model), *liveness* on `/health` (a hung process is restarted) |
| `Service` (LoadBalancer) | one address for all ready pods – `http://localhost:8080` on Docker Desktop, a public IP on a cloud cluster |
| `HorizontalPodAutoscaler` | 2–5 pods, adds pods above 60 % CPU (needs metrics-server) |
| `PodDisruptionBudget` | node maintenance can never take the last pod down |
| `ConfigMap` | decision threshold and log level, changeable without rebuilding the image |
| Security | non-root user 10001, read-only root filesystem, all Linux capabilities dropped, no privilege escalation, seccomp `RuntimeDefault`, no service-account token |
| Resources | requests 100m CPU / 192 Mi, limits 1 CPU / 512 Mi (idle use ≈ 125 Mi) |

The autoscaler needs metrics-server, which Docker Desktop does not include:
`kubectl apply -k deployment/metrics-server` (then `kubectl top pods -n heart-disease` works).

Operations demonstrated (see the report for screenshots):

```bash
kubectl -n heart-disease delete pod <pod-name>                      # self-healing: a replacement starts
python scripts/load_test.py --url http://localhost:8080 --duration 180 --workers 16
kubectl -n heart-disease get hpa -w                                 # autoscaling 2 -> 5 pods under load
kubectl -n heart-disease rollout restart deployment/heart-disease-api   # zero-downtime rolling update
kubectl -n heart-disease set image deployment/heart-disease-api api=heart-disease-api:broken
kubectl -n heart-disease rollout undo deployment/heart-disease-api       # failed release -> rollback
```

To run the published image instead of a local build:
`kubectl -n heart-disease set image deployment/heart-disease-api api=ghcr.io/snehanjaligandham/heart-disease-api:latest`.

## Monitoring (Prometheus + Grafana)

```bash
kubectl apply -k deployment/monitoring          # Kubernetes: Prometheus :9090, Grafana :3000
# or, without Kubernetes:  docker compose up -d --build   (API :8000, Prometheus :9091, Grafana :3001)
python scripts/check_monitoring.py               # targets up, rules loaded, dashboard provisioned
python scripts/load_test.py --url http://localhost:8080 --duration 300            # normal traffic
python scripts/load_test.py --url http://localhost:8080 --duration 600 --profile drifted   # drift demo
```

* **Metrics** (`/metrics` on every pod): requests by route/status, latency histogram, predictions by
  class, predicted-probability histogram, validation errors, the values of incoming features,
  the model version, CPU/memory, and the **training mean and standard deviation of each feature**,
  read from the model's own `StandardScaler` so the drift baseline always matches the model served.
* **Prometheus** finds every API pod by itself (Kubernetes service discovery + the `prometheus.io/*`
  pod annotations), so pods added by the autoscaler are monitored automatically. It may only read
  pods in the API's namespace (namespaced `Role`).
* **Grafana dashboard** (provisioned from `deployment/monitoring/grafana/dashboards/`, no manual
  setup): pods up, request rate, server-error ratio, p50/p95/p99 latency, traffic per pod (load
  balancing), CPU/memory per pod, predictions per class, positive-prediction share, mean predicted
  probability, **input drift per feature**, recent-vs-training means, rejected requests and active alerts.
* **Data drift** = (mean of the last 15 min of inputs − training mean) / training standard deviation.
  With real patient records it stays near 0; with the drifted population it exceeds the ±0.5 band.
* **Alerts** (`prometheus/alerts.yml`, unit-tested with `promtool test rules`):

| Alert | Fires when | Severity |
|---|---|---|
| HeartApiDown | a pod's `/metrics` fails for 1 min | critical |
| HeartApiNoTargets | no API pod found for 2 min | critical |
| HeartApiHighErrorRate | > 5 % of requests return 5xx for 5 min | critical |
| HeartApiSlowPredictions | p95 latency of `/predict` > 500 ms for 10 min | warning |
| HeartApiInputDrift | a feature's drift score is beyond ±0.5 for 5 min (with ≥ 50 requests) | warning |
| HeartApiPredictionShift | positive predictions < 20 % or > 80 % for 5 min (training ≈ 46 %) | warning |
| HeartApiManyInvalidRequests | > 20 % of prediction requests rejected for 10 min | warning |

Grafana shows dashboards without logging in; log in as `admin` / `admin` (you are asked to change it)
to edit. This is a development set-up – on a shared cluster the admin password would come from a
Kubernetes Secret.
