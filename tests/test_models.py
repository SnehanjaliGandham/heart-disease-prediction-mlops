"""Candidate models and hyper-parameter grids (src/models.py)."""
from __future__ import annotations

import numpy as np
import pytest
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import ParameterGrid

from src import models
from src.models import get_candidates, make_pipeline

NAMES = {"baseline_majority", "logistic_regression", "random_forest", "gradient_boosting"}


def test_candidate_set():
    assert set(get_candidates()) == NAMES
    assert set(get_candidates(quick=True)) == NAMES


@pytest.mark.parametrize("name", sorted(NAMES))
def test_every_grid_point_is_a_valid_parameter_setting(name):
    estimator, grid = get_candidates()[name]
    pipe = make_pipeline(estimator)
    for params in ParameterGrid(grid or [{}]):
        clone(pipe).set_params(**params)      # raises if a name or value is invalid


def test_quick_grids_are_no_larger_than_full_grids():
    full, quick = get_candidates(), get_candidates(quick=True)
    for name in NAMES:
        n_quick = len(ParameterGrid(quick[name][1] or [{}]))
        assert n_quick <= len(ParameterGrid(full[name][1] or [{}]))


def test_pipeline_structure():
    estimator, _ = get_candidates()["random_forest"]
    pipe = make_pipeline(estimator)
    assert [step for step, _ in pipe.steps] == ["features", "preprocess", "model"]


def test_baseline_is_majority_class():
    estimator, grid = get_candidates()["baseline_majority"]
    assert isinstance(estimator, DummyClassifier) and estimator.strategy == "most_frequent"
    assert grid == {}


def test_logistic_regression_api_detection():
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    assert models.NEW_LOGREG_API == ((major, minor) >= (1, 8))


@pytest.mark.parametrize("name", sorted(NAMES))
def test_each_candidate_fits_and_returns_probabilities(name, split_data):
    X_train, X_test, y_train, _ = split_data
    estimator, _ = get_candidates(quick=True)[name]
    proba = make_pipeline(estimator).fit(X_train, y_train).predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)


def test_models_are_seeded_and_reproducible(split_data):
    X_train, X_test, y_train, _ = split_data
    estimator, _ = get_candidates()["random_forest"]
    p1 = make_pipeline(clone(estimator)).fit(X_train, y_train).predict_proba(X_test)
    p2 = make_pipeline(clone(estimator)).fit(X_train, y_train).predict_proba(X_test)
    np.testing.assert_array_equal(p1, p2)
