"""Metrics and plots (src/evaluate.py)."""
from __future__ import annotations

import numpy as np
import pytest

from src.evaluate import (
    classification_metrics,
    feature_importance,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curves,
)
from src.models import get_candidates, make_pipeline


def test_metrics_on_a_known_case():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    y_proba = [0.1, 0.6, 0.7, 0.9]
    m = classification_metrics(y_true, y_pred, y_proba)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(0.8)
    assert m["roc_auc"] == pytest.approx(1.0)


def test_metrics_do_not_fail_when_nothing_is_predicted_positive():
    m = classification_metrics([0, 1, 1], [0, 0, 0], [0.2, 0.3, 0.4])
    assert m["precision"] == 0 and m["recall"] == 0 and m["f1"] == 0


def test_logistic_regression_importance(fitted_lr):
    imp = feature_importance(fitted_lr)
    assert len(imp) == 25 and (imp >= 0).all()
    assert imp.is_monotonic_decreasing
    assert "ca" in imp.index[:5]          # strongest numeric predictor in the EDA


def test_random_forest_importance_sums_to_one(split_data):
    X_train, _, y_train, _ = split_data
    estimator, _ = get_candidates(quick=True)["random_forest"]
    rf = make_pipeline(estimator).fit(X_train, y_train)
    assert np.isclose(feature_importance(rf).sum(), 1.0)


def test_baseline_has_no_importance(split_data, tmp_path):
    X_train, _, y_train, _ = split_data
    estimator, _ = get_candidates()["baseline_majority"]
    dummy = make_pipeline(estimator).fit(X_train, y_train)
    assert feature_importance(dummy) is None
    assert plot_feature_importance(dummy, "x", tmp_path / "x.png") is None


def test_plots_are_written(fitted_lr, split_data, tmp_path):
    _, X_test, _, y_test = split_data
    paths = [
        plot_confusion_matrix(fitted_lr, X_test, y_test, "cm", tmp_path / "cm.png"),
        plot_roc_curves({"lr": fitted_lr}, X_test, y_test, tmp_path / "roc.png"),
        plot_feature_importance(fitted_lr, "imp", tmp_path / "imp.png"),
    ]
    for p in paths:
        assert p.exists() and p.stat().st_size > 1000
        assert p.read_bytes()[:4] == b"\x89PNG"
