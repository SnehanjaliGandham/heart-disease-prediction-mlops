"""Central configuration: paths, feature lists and constants shared by every stage.

Keeping these in one place means the notebook, the training script, the tests
and the API all agree on column names and file locations.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATA_PATH = DATA_DIR / "heart_disease_raw.csv"
CLEAN_DATA_PATH = DATA_DIR / "heart_disease_clean.csv"

MODEL_PATH = MODELS_DIR / "heart_disease_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
MLFLOW_MODEL_DIR = MODELS_DIR / "mlflow_model"

# ---------------------------------------------------------------- MLflow
# A local SQLite backend supports the Model Registry; artifacts go to ./mlruns.
MLFLOW_TRACKING_URI = f"sqlite:///{(ROOT_DIR / 'mlflow.db').as_posix()}"
MLFLOW_ARTIFACT_ROOT = (ROOT_DIR / "mlruns").as_uri()
MLFLOW_EXPERIMENT = "heart-disease-classification"
REGISTERED_MODEL_NAME = "heart-disease-classifier"

# ---------------------------------------------------------------- data schema
RAW_TARGET = "num"          # original 0-4 severity label from UCI
TARGET = "target"           # binary: 1 = heart disease present (num > 0)

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Allowed codes for categorical features (from the UCI data dictionary).
CATEGORY_CODES = {
    "sex": [0, 1],
    "cp": [1, 2, 3, 4],
    "fbs": [0, 1],
    "restecg": [0, 1, 2],
    "exang": [0, 1],
    "slope": [1, 2, 3],
    "thal": [3, 6, 7],
}

# Plausible clinical ranges, used to validate data and API input.
NUMERIC_RANGES = {
    "age": (1, 120),
    "trestbps": (50, 250),
    "chol": (50, 700),
    "thalach": (50, 250),
    "oldpeak": (0.0, 10.0),
    "ca": (0, 3),
}

# ---------------------------------------------------------------- modelling
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
SELECTION_METRIC = "roc_auc"   # best model is chosen on CV ROC-AUC, never on the test set
