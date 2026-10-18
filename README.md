# Heart Disease Prediction – End-to-End MLOps Pipeline

BITS Pilani WILP · M.Tech AIML · MLOps (AIMLCZG523) Assignment 01

[![CI/CD pipeline](https://github.com/SnehanjaliGandham/heart-disease-prediction-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/SnehanjaliGandham/heart-disease-prediction-mlops/actions/workflows/ci.yml)

End-to-end, reproducible ML pipeline for predicting heart disease from the
UCI Heart Disease (Cleveland) dataset: data acquisition → EDA → feature
engineering → model training & tuning → MLflow tracking → packaged model →
automated tests and CI/CD (Docker, Kubernetes and monitoring in later stages).

## Project structure

```
data/                      raw + cleaned CSV, UCI variable descriptions, SHA-256 manifest
scripts/download_data.py   downloads the dataset from UCI and verifies it against the manifest
scripts/check_env.py       checks a machine (e.g. Prayogshala) can run the project
scripts/quality_gate.py    blocks a model that falls below minimum metric standards
src/
  config.py                paths, feature lists, constants (single source of truth)
  data.py                  load, validate, clean, split
  features.py              preprocessing ColumnTransformer
  models.py                candidate models + hyper-parameter grids
  evaluate.py              metrics and evaluation plots
  train.py                 tuning, model selection, MLflow logging, packaging
  predict.py               load packaged model and score patient records
notebooks/
  01_data_acquisition_eda.ipynb   Task 1 – data acquisition and EDA
  02_model_development.ipynb      Tasks 2–4 – features, models, MLflow, packaging
models/                    packaged champion model (+ MLflow format, metadata)
reports/                   model comparison table and figures for the report
tests/                     95 pytest unit tests, 98 % coverage (Task 5)
.github/workflows/ci.yml   CI/CD pipeline: lint -> tests (Py 3.10-3.13) -> train + quality gate
deployment/                Docker / Kubernetes files (Tasks 6–7)
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
python -m pytest --cov                         # 95 tests, coverage report
python scripts/quality_gate.py                 # check the trained model meets minimum standards
```

The tests run offline in under a minute. They cover data validation and cleaning, the
preprocessing (including a leakage guard), every hyper-parameter grid, metrics and plots,
an end-to-end training run with checks on what MLflow recorded and on the model registry,
inference, and the helper scripts.

Every push and pull request to `main` runs the GitHub Actions pipeline:

| Job | What it does | Fails when |
|---|---|---|
| 1 · Lint | `ruff` static checks | style or likely-bug findings |
| 2 · Unit tests | pytest on Python 3.10, 3.11, 3.12 and 3.13 with coverage | any test fails or coverage < 90 % |
| 3 · Train | verify data checksum → clean → train + MLflow → quality gate → inference smoke test → packaged-model tests → execute notebooks | any step errors or the model misses the quality gate |

Each job uploads its results as artifacts (lint report, JUnit/coverage reports, packaged
model, MLflow store, logs, executed notebooks) and writes a summary to the run page.

## Results so far

| Model | CV ROC-AUC (5-fold) | Test accuracy | Test recall | Test ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression (selected)** | **0.911 ± 0.018** | 0.885 | 0.929 | 0.961 |
| Random Forest | 0.900 ± 0.032 | 0.885 | 0.929 | 0.955 |
| Gradient Boosting | 0.884 ± 0.027 | 0.902 | 0.929 | 0.960 |
| Majority baseline | 0.500 | 0.541 | 0.000 | 0.500 |

The model is selected on cross-validated ROC-AUC; the held-out test set is used only for reporting.
