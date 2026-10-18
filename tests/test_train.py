"""End-to-end training stage with MLflow (src/train.py).

One quick training run is executed into a temporary project folder and a temporary
MLflow store; the tests then check what was produced and what MLflow recorded.
"""
from __future__ import annotations

import json

import joblib
import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from src import config, train
from src.data import file_sha256

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def run(isolated_paths):
    out = train.main(quick=True, do_register=True, experiment="pytest-experiment")
    client = MlflowClient(tracking_uri=config.MLFLOW_TRACKING_URI)
    exp = client.get_experiment_by_name("pytest-experiment")
    runs = client.search_runs([exp.experiment_id])
    return {**out, "client": client, "runs": runs, "root": isolated_paths}


def _children(runs):
    return [r for r in runs if "mlflow.parentRunId" in r.data.tags]


def _parent(runs):
    return next(r for r in runs if "mlflow.parentRunId" not in r.data.tags)


# ------------------------------------------------------------------ results


def test_all_candidates_are_compared_and_sorted(run):
    results = run["results"]
    assert set(results["model"]) == {"baseline_majority", "logistic_regression",
                                     "random_forest", "gradient_boosting"}
    assert results["cv_roc_auc_mean"].is_monotonic_decreasing


def test_baseline_is_never_selected_and_is_beaten(run):
    results = run["results"].set_index("model")
    assert run["best_name"] != "baseline_majority"
    assert results.loc["baseline_majority", "cv_roc_auc_mean"] == pytest.approx(0.5)
    assert results.loc[run["best_name"], "cv_roc_auc_mean"] > 0.8


def test_selection_uses_cv_score_not_test_score(run):
    real = run["results"][run["results"]["model"] != "baseline_majority"]
    assert run["best_name"] == real.sort_values("cv_roc_auc_mean").iloc[-1]["model"]


def test_cv_summary_contains_mean_and_std_for_every_metric(run):
    row = run["results"].iloc[0]
    for metric in train.METRICS:
        assert f"cv_{metric}_mean" in row and f"cv_{metric}_std" in row


# ------------------------------------------------------------------ packaged outputs


def test_model_and_metadata_are_written(run):
    assert config.MODEL_PATH.exists() and (config.MLFLOW_MODEL_DIR / "MLmodel").exists()
    meta = json.loads(config.MODEL_METADATA_PATH.read_text())
    for key in ["model_name", "best_params", "cv_metrics", "test_metrics", "features",
                "mlflow_run_id", "training_data_sha256", "git_commit", "sklearn_version"]:
        assert key in meta
    assert meta["model_name"] == run["best_name"]
    assert meta["training_data_sha256"] == file_sha256(config.CLEAN_DATA_PATH)


def test_report_artifacts_are_written(run):
    assert (config.REPORTS_DIR / "model_comparison.csv").exists()
    for name in ["roc_curves.png", "confusion_matrix_best.png", "feature_importance_best.png"]:
        assert (config.FIGURES_DIR / name).exists()
    table = pd.read_csv(config.REPORTS_DIR / "model_comparison.csv")
    assert len(table) == 4


def test_saved_model_reproduces_trained_model(run):
    _, X_test, _, _ = run["split"]
    saved = joblib.load(config.MODEL_PATH)
    np.testing.assert_allclose(saved.predict_proba(X_test),
                               run["fitted"][run["best_name"]].predict_proba(X_test))


def test_mlflow_format_model_loads(run):
    _, X_test, _, _ = run["split"]
    pyfunc = mlflow.pyfunc.load_model(str(config.MLFLOW_MODEL_DIR))
    np.testing.assert_array_equal(pyfunc.predict(X_test),
                                  run["fitted"][run["best_name"]].predict(X_test))


def test_round_trip_check_detects_a_wrong_file(run, tmp_path, monkeypatch):
    _, X_test, _, _ = run["split"]
    other = run["fitted"]["baseline_majority"]
    wrong = tmp_path / "wrong.joblib"
    joblib.dump(other, wrong)
    monkeypatch.setattr(config, "MODEL_PATH", wrong)
    with pytest.raises(RuntimeError, match="does not reproduce"):
        train.check_round_trip(run["fitted"][run["best_name"]], X_test)


# ------------------------------------------------------------------ MLflow records


def test_one_parent_run_with_one_child_per_model(run):
    runs = run["runs"]
    parent = _parent(runs)
    children = _children(runs)
    assert len(runs) == 5 and len(children) == 4
    assert {c.data.tags["mlflow.parentRunId"] for c in children} == {parent.info.run_id}
    assert parent.data.tags["selected_model"] == run["best_name"]
    assert "git_commit" in parent.data.tags


def test_logged_metrics_equal_returned_results(run):
    """What MLflow shows must be exactly what the code computed."""
    results = run["results"].set_index("model")
    for child in _children(run["runs"]):
        name = child.data.tags["mlflow.runName"]
        for metric in ["cv_roc_auc_mean", "cv_accuracy_mean", "test_roc_auc", "test_recall"]:
            assert child.data.metrics[metric] == pytest.approx(results.loc[name, metric])


def test_child_runs_log_params_artifacts_and_model(run):
    client = run["client"]
    child = next(c for c in _children(run["runs"])
                 if c.data.tags["mlflow.runName"] == "logistic_regression")
    assert child.data.params["model_type"] == "LogisticRegression"
    assert "C" in child.data.params
    folders = {a.path for a in client.list_artifacts(child.info.run_id)}
    assert {"plots", "tables"} <= folders


def test_champion_alias_points_to_selected_run(run):
    client = run["client"]
    version = client.get_model_version_by_alias(config.REGISTERED_MODEL_NAME, "champion")
    selected = next(c for c in _children(run["runs"])
                    if c.data.tags["mlflow.runName"] == run["best_name"])
    assert version.run_id == selected.info.run_id


def test_git_commit_helper_returns_text():
    assert isinstance(train.git_commit(), str) and train.git_commit()
