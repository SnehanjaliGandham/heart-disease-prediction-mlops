# Heart Disease Prediction – End-to-End MLOps Pipeline

BITS Pilani WILP · M.Tech AIML · MLOps (AIMLCZG523) Assignment 01

[![CI/CD pipeline](https://github.com/SnehanjaliGandham/heart-disease-prediction-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/SnehanjaliGandham/heart-disease-prediction-mlops/actions/workflows/ci.yml)

End-to-end, reproducible ML pipeline for predicting heart disease from the
UCI Heart Disease (Cleveland) dataset: data acquisition → EDA → feature
engineering → model training & tuning → MLflow tracking → packaged model →
automated tests and CI/CD → containerised prediction API (Kubernetes and monitoring in later stages).

## Project structure

```
data/                      raw + cleaned CSV, UCI variable descriptions, SHA-256 manifest
scripts/download_data.py   downloads the dataset from UCI and verifies it against the manifest
scripts/check_env.py       checks a machine (e.g. Prayogshala) can run the project
scripts/quality_gate.py    blocks a model that falls below minimum metric standards
scripts/smoke_test.py      end-to-end checks against a running API (Docker / Kubernetes)
src/
  config.py                paths, feature lists, constants (single source of truth)
  data.py                  load, validate, clean, split
  features.py              preprocessing ColumnTransformer
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
.github/workflows/ci.yml   CI/CD pipeline: lint -> tests (Py 3.10-3.13) -> train + quality gate -> Docker
Dockerfile                 multi-stage, non-root image for the API (Task 6)
docker-compose.yml         one-command local run of the API
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

```bash
python scripts/download_data.py      # 1. get the data (skipped if already present)
python -m src.data                   # 2. raw -> data/heart_disease_clean.csv
python -m src.train                  # 3. tune + compare models, log to MLflow, save best model
python -m src.predict                # 4. score a sample patient with the saved model
mlflow ui --backend-store-uri sqlite:///mlflow.db   # 5. browse runs at http://127.0.0.1:5000
```

`python -m src.train --quick` runs with small grids (handy for quick local checks).

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
| 1 · Lint | `ruff` static checks | style or likely-bug findings |
| 2 · Unit tests | pytest on Python 3.10, 3.11, 3.12 and 3.13 with coverage | any test fails or coverage < 90 % |
| 3 · Train | verify data checksum → clean → train + MLflow → quality gate → inference smoke test → packaged-model tests → execute notebooks | any step errors or the model misses the quality gate |
| 4 · Docker | build the image with the model trained in job 3 → start the container → smoke test → wait for HEALTHCHECK = healthy → check it runs as non-root → push to GitHub Container Registry (pushes to `main` only) | build fails, any smoke check fails or the container is unhealthy |

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

## Results so far

| Model | CV ROC-AUC (5-fold) | Test accuracy | Test recall | Test ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression (selected)** | **0.911 ± 0.018** | 0.885 | 0.929 | 0.961 |
| Random Forest | 0.900 ± 0.032 | 0.885 | 0.929 | 0.955 |
| Gradient Boosting | 0.884 ± 0.027 | 0.902 | 0.929 | 0.960 |
| Majority baseline | 0.500 | 0.541 | 0.000 | 0.500 |

The model is selected on cross-validated ROC-AUC; the held-out test set is used only for reporting.
