# Heart Disease Prediction – End-to-End MLOps Pipeline

BITS Pilani WILP · M.Tech AIML · MLOps (AIMLCZG523) Assignment 01

End-to-end, reproducible ML pipeline for predicting heart disease from the
UCI Heart Disease (Cleveland) dataset: data acquisition → EDA → feature
engineering → model training & tuning → MLflow tracking → packaged model →
(CI/CD, Docker, Kubernetes and monitoring in later stages).

## Project structure

```
data/                      raw + cleaned CSV, UCI variable descriptions
scripts/download_data.py   downloads the dataset from the UCI repository
scripts/check_env.py       checks a machine (e.g. Prayogshala) can run the project
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
tests/                     unit tests (Task 5)
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

`python -m src.train --quick` runs with small grids (used in CI).

## Results so far

| Model | CV ROC-AUC (5-fold) | Test accuracy | Test recall | Test ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression (selected)** | **0.911 ± 0.018** | 0.885 | 0.929 | 0.961 |
| Random Forest | 0.900 ± 0.032 | 0.885 | 0.929 | 0.955 |
| Gradient Boosting | 0.884 ± 0.027 | 0.902 | 0.929 | 0.960 |
| Majority baseline | 0.500 | 0.541 | 0.000 | 0.500 |

The model is selected on cross-validated ROC-AUC; the held-out test set is used only for reporting.
