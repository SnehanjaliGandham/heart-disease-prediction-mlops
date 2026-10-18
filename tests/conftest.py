"""Shared pytest fixtures.

All tests run offline: they use the committed UCI CSV plus small synthetic frames,
and training tests write to a temporary folder so the real models/, reports/ and
MLflow store are never touched.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from src import config  # noqa: E402
from src.data import clean, load_raw, split  # noqa: E402
from src.models import get_candidates, make_pipeline  # noqa: E402

HEALTHY_PATIENT = {
    "age": 41, "sex": 0, "cp": 2, "trestbps": 130, "chol": 204, "fbs": 0, "restecg": 2,
    "thalach": 172, "exang": 0, "oldpeak": 1.4, "slope": 1, "ca": 0, "thal": 3,
}
SICK_PATIENT = {
    "age": 67, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286, "fbs": 0, "restecg": 2,
    "thalach": 108, "exang": 1, "oldpeak": 1.5, "slope": 2, "ca": 3, "thal": 3,
}


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """The real UCI Cleveland file (303 rows, 5-class `num`)."""
    return load_raw()


@pytest.fixture(scope="session")
def clean_df(raw_df) -> pd.DataFrame:
    return clean(raw_df)


@pytest.fixture(scope="session")
def split_data(clean_df):
    return split(clean_df)


@pytest.fixture
def synthetic_raw() -> pd.DataFrame:
    """60 valid rows in raw UCI format, with gaps in ca/thal and one exact duplicate."""
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "age": rng.integers(30, 77, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(1, 5, n),
        "trestbps": rng.integers(95, 200, n),
        "chol": rng.integers(130, 560, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.integers(71, 202, n),
        "exang": rng.integers(0, 2, n),
        "oldpeak": rng.uniform(0, 6, n).round(1),
        "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n),
        "num": rng.integers(0, 5, n),
    })
    df.loc[[3, 10], "ca"] = np.nan
    df.loc[[5], "thal"] = np.nan
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)   # row 60 duplicates row 0


@pytest.fixture(scope="session")
def fitted_lr(split_data):
    """A Logistic Regression pipeline fitted on the real training split."""
    X_train, _, y_train, _ = split_data
    estimator, _ = get_candidates()["logistic_regression"]
    return make_pipeline(estimator).fit(X_train, y_train)


@pytest.fixture(scope="module")
def isolated_paths(tmp_path_factory):
    """Point every output path in src.config at a temporary project folder."""
    root = tmp_path_factory.mktemp("project")
    patches = {
        "ROOT_DIR": root,
        "MODELS_DIR": root / "models",
        "MODEL_PATH": root / "models" / "heart_disease_model.joblib",
        "MODEL_METADATA_PATH": root / "models" / "model_metadata.json",
        "MLFLOW_MODEL_DIR": root / "models" / "mlflow_model",
        "REPORTS_DIR": root / "reports",
        "FIGURES_DIR": root / "reports" / "figures",
        "MLFLOW_TRACKING_URI": f"sqlite:///{(root / 'mlflow.db').as_posix()}",
        "MLFLOW_ARTIFACT_ROOT": (root / "mlruns").as_uri(),
    }
    with pytest.MonkeyPatch.context() as mp:
        for name, value in patches.items():
            mp.setattr(config, name, value)
        yield root
