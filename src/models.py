"""Candidate models and their hyper-parameter search spaces."""
from __future__ import annotations

import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config
from src.features import build_preprocessor

# scikit-learn 1.8 replaced LogisticRegression(penalty=...) with l1_ratio alone.
# Detect the version so the same grid search runs on old and new installs.
_SK_VERSION = tuple(int(x) for x in sklearn.__version__.split(".")[:2])
NEW_LOGREG_API = _SK_VERSION >= (1, 8)


def make_pipeline(estimator) -> Pipeline:
    """preprocess -> estimator, as one object."""
    return Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])


def get_candidates(quick: bool = False) -> dict:
    """name -> (estimator, param_grid). `quick=True` gives tiny grids for CI."""
    rs = config.RANDOM_STATE
    candidates = {
        # Baseline: always predicts the majority class. Any real model must beat it.
        "baseline_majority": (
            DummyClassifier(strategy="most_frequent"),
            {},
        ),
        # Linear, interpretable. l1_ratio: 0 = L2 (ridge), 1 = L1 (lasso), 0.5 = elastic net.
        "logistic_regression": (
            LogisticRegression(solver="saga", max_iter=5000, random_state=rs),
            {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
                **({"model__l1_ratio": [0.0, 0.5, 1.0]} if NEW_LOGREG_API else
                   {"model__penalty": ["elasticnet"], "model__l1_ratio": [0.0, 0.5, 1.0]}),
            },
        ),
        # Non-linear bagging ensemble; robust to outliers and feature scale.
        "random_forest": (
            RandomForestClassifier(random_state=rs, n_jobs=-1),
            {
                "model__n_estimators": [200, 500],
                "model__max_depth": [None, 4, 8],
                "model__min_samples_leaf": [1, 3, 5],
                "model__max_features": ["sqrt", 0.5],
            },
        ),
        # Boosting ensemble; included as a third, stronger non-linear comparison.
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=rs),
            {
                "model__n_estimators": [100, 200],
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [2, 3],
                "model__subsample": [0.8, 1.0],
            },
        ),
    }
    if quick:
        candidates["logistic_regression"] = (
            candidates["logistic_regression"][0], {"model__C": [0.1, 1.0]})
        candidates["random_forest"] = (
            candidates["random_forest"][0],
            {"model__n_estimators": [100], "model__max_depth": [None, 4]})
        candidates["gradient_boosting"] = (
            candidates["gradient_boosting"][0], {"model__n_estimators": [100]})
    return candidates
