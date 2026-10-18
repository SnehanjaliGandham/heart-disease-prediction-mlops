"""Train, tune, compare and package the heart disease models, tracked with MLflow.

Usage (from the repo root):
    python -m src.train                 # full grid search for every candidate
    python -m src.train --quick         # tiny grids (used by CI)
    python -m src.train --no-register   # skip the MLflow Model Registry step

What one run does
1. Loads the cleaned data and makes a stratified 80/20 split.
2. For each candidate model, runs GridSearchCV (5-fold stratified CV) on the
   training split only, scoring accuracy / precision / recall / F1 / ROC-AUC.
3. Logs each model as a nested MLflow run: best params, CV metrics (mean and std),
   test metrics, plots, CV table and the fitted pipeline.
4. Picks the best model by CV ROC-AUC (the test set is only used for reporting).
5. Saves the winner to models/ (joblib + MLflow format + metadata JSON) and
   registers it in the MLflow Model Registry with the alias "champion".
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import platform
import shutil
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.metrics import f1_score, make_scorer, precision_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src import config
from src.data import file_sha256, load_clean, split
from src.evaluate import (
    classification_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curves,
)
from src.models import get_candidates, make_pipeline

# Own handler (not basicConfig): some MLflow versions reconfigure the root logger
# while creating the tracking database, which would otherwise hide these messages.
log = logging.getLogger("heart.train")
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(_handler)
log.setLevel(logging.INFO)
log.propagate = False

METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
# zero_division=0 keeps the majority-class baseline (which never predicts
# "disease") from raising warnings: its precision is reported as 0.
SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": make_scorer(f1_score, zero_division=0),
    "roc_auc": "roc_auc",
}
# MLflow 3 stores sklearn models with skops (a safer format than pickle), which
# only loads types that are explicitly trusted. We trust these two because the
# models are created by this script, never downloaded from elsewhere.
SKOPS_TRUSTED_TYPES = ["numpy.dtype", "sklearn.tree._tree.Tree"]
# Older MLflow 3.x releases pickle models and have no such argument.
_SAVE_KWARGS = (
    {"skops_trusted_types": SKOPS_TRUSTED_TYPES}
    if "skops_trusted_types" in inspect.signature(mlflow.sklearn.save_model).parameters
    else {}
)


def setup_mlflow(experiment_name: str) -> None:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=config.MLFLOW_ARTIFACT_ROOT)
    mlflow.set_experiment(experiment_name)


def tune(name, estimator, grid, X_train, y_train) -> GridSearchCV:
    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True,
                         random_state=config.RANDOM_STATE)
    search = GridSearchCV(
        make_pipeline(estimator), param_grid=grid or [{}], cv=cv,
        scoring=SCORING, refit=config.SELECTION_METRIC, n_jobs=-1,
    )
    search.fit(X_train, y_train)
    log.info("%-20s best CV %s = %.4f", name, config.SELECTION_METRIC, search.best_score_)
    return search


def cv_summary(search: GridSearchCV) -> dict:
    """Mean and std of every CV metric for the best parameter set."""
    i = search.best_index_
    out = {}
    for metric in METRICS:
        out[f"cv_{metric}_mean"] = float(search.cv_results_[f"mean_test_{metric}"][i])
        out[f"cv_{metric}_std"] = float(search.cv_results_[f"std_test_{metric}"][i])
    return out


def log_candidate(name, search, X_train, X_test, y_test, tmp: Path) -> dict:
    """Log one candidate as a nested MLflow run and return its summary row."""
    model = search.best_estimator_
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    test = {f"test_{k}": float(v) for k, v in
            classification_metrics(y_test, y_pred, y_proba).items()}
    cv = cv_summary(search)

    with mlflow.start_run(run_name=name, nested=True) as run:
        mlflow.set_tag("model_family", name)
        mlflow.log_params({"model_type": type(model.named_steps["model"]).__name__,
                           "n_param_combinations": len(search.cv_results_["params"])})
        mlflow.log_params({k.replace("model__", ""): v for k, v in search.best_params_.items()})
        mlflow.log_metrics({**cv, **test})

        cv_table = pd.DataFrame(search.cv_results_).sort_values("rank_test_roc_auc")
        cv_path = tmp / f"{name}_cv_results.csv"
        cv_table.to_csv(cv_path, index=False)
        mlflow.log_artifact(cv_path, "tables")

        mlflow.log_artifact(plot_confusion_matrix(
            model, X_test, y_test, f"{name} (test set)", tmp / f"{name}_confusion.png"), "plots")
        imp = plot_feature_importance(model, f"{name}: top features",
                                      tmp / f"{name}_importance.png")
        if imp:
            mlflow.log_artifact(imp, "plots")

        signature = infer_signature(X_train, model.predict_proba(X_train)[:, 1])
        info = mlflow.sklearn.log_model(
            model, name="model", signature=signature, input_example=X_train.head(3),
            **_SAVE_KWARGS,
        )
        run_id = run.info.run_id

    return {"model": name, "run_id": run_id, "model_uri": info.model_uri,
            "best_params": search.best_params_, **cv, **test}


def save_final_model(name, model, row, data_hash, tmp_plots: dict) -> None:
    """Write the champion to models/ in two formats plus a metadata card."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)

    if config.MLFLOW_MODEL_DIR.exists():
        shutil.rmtree(config.MLFLOW_MODEL_DIR)
    mlflow.sklearn.save_model(model, config.MLFLOW_MODEL_DIR, **_SAVE_KWARGS)

    metadata = {
        "model_name": name,
        "model_class": type(model.named_steps["model"]).__name__,
        "best_params": {k.replace("model__", ""): v for k, v in row["best_params"].items()},
        "selection_metric": f"cv_{config.SELECTION_METRIC}_mean",
        "cv_metrics": {k: round(v, 4) for k, v in row.items() if k.startswith("cv_")},
        "test_metrics": {k: round(v, 4) for k, v in row.items() if k.startswith("test_")},
        "features": {"numeric": config.NUMERIC_FEATURES,
                     "categorical": config.CATEGORICAL_FEATURES},
        "target": "1 = heart disease present (num > 0), 0 = absent",
        "mlflow_run_id": row["run_id"],
        "training_data_sha256": data_hash,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "figures": tmp_plots,
    }
    config.MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str))
    log.info("Saved %s and %s", config.MODEL_PATH.name, config.MODEL_METADATA_PATH.name)


def register(model_uri: str) -> None:
    version = mlflow.register_model(model_uri, config.REGISTERED_MODEL_NAME)
    MlflowClient().set_registered_model_alias(
        config.REGISTERED_MODEL_NAME, "champion", version.version)
    log.info("Registered %s v%s as @champion", config.REGISTERED_MODEL_NAME, version.version)


def main(quick: bool = False, do_register: bool = True,
         experiment: str = config.MLFLOW_EXPERIMENT) -> dict:
    """Run the whole training stage. Returns results, fitted models and the split."""
    setup_mlflow(experiment)
    log.disabled = False   # MLflow's DB migration may disable existing loggers
    df = load_clean()
    X_train, X_test, y_train, y_test = split(df)
    data_hash = file_sha256(config.CLEAN_DATA_PATH)
    log.info("Train %s | Test %s", X_train.shape, X_test.shape)

    run_name = f"training-{datetime.now():%Y%m%d-%H%M%S}"
    with mlflow.start_run(run_name=run_name), tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        mlflow.set_tags({"stage": "model_selection", "quick_mode": str(quick)})
        mlflow.log_params({
            "test_size": config.TEST_SIZE, "cv_folds": config.CV_FOLDS,
            "random_state": config.RANDOM_STATE, "selection_metric": config.SELECTION_METRIC,
            "n_train": len(X_train), "n_test": len(X_test), "data_sha256": data_hash[:12],
        })
        with warnings.catch_warnings():   # MLflow warns about an ambiguous local source
            warnings.simplefilter("ignore", UserWarning)
            mlflow.log_input(mlflow.data.from_pandas(
                df, source=str(config.CLEAN_DATA_PATH), name="heart_disease_clean",
                targets=config.TARGET), context="training")

        rows, fitted = [], {}
        for name, (estimator, grid) in get_candidates(quick).items():
            search = tune(name, estimator, grid, X_train, y_train)
            rows.append(log_candidate(name, search, X_train, X_test, y_test, tmp))
            fitted[name] = search.best_estimator_

        results = pd.DataFrame(rows).sort_values(
            f"cv_{config.SELECTION_METRIC}_mean", ascending=False)
        real = results[results["model"] != "baseline_majority"]
        best = real.iloc[0]
        best_name = best["model"]
        log.info("Selected model: %s", best_name)

        # Comparison artifacts on the parent run (and copies for the report).
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        cols = ["model"] + [c for c in results.columns if c.startswith(("cv_", "test_"))]
        table = results[cols].round(4)
        table_path = config.REPORTS_DIR / "model_comparison.csv"
        table.to_csv(table_path, index=False)
        roc_path = plot_roc_curves(
            {k: v for k, v in fitted.items() if k != "baseline_majority"},
            X_test, y_test, config.FIGURES_DIR / "roc_curves.png")
        cm_path = plot_confusion_matrix(fitted[best_name], X_test, y_test,
                                        f"Selected: {best_name}",
                                        config.FIGURES_DIR / "confusion_matrix_best.png")
        imp_path = plot_feature_importance(fitted[best_name], f"Selected: {best_name}",
                                           config.FIGURES_DIR / "feature_importance_best.png")
        for p in (table_path, roc_path, cm_path, imp_path):
            if p:
                mlflow.log_artifact(p, "comparison")

        mlflow.set_tag("selected_model", best_name)
        mlflow.log_metrics({
            "best_cv_roc_auc": best[f"cv_{config.SELECTION_METRIC}_mean"],
            "best_test_roc_auc": best["test_roc_auc"],
            "best_test_accuracy": best["test_accuracy"],
        })

        save_final_model(best_name, fitted[best_name], best.to_dict(), data_hash, {
            "roc": str(roc_path.relative_to(config.ROOT_DIR)),
            "confusion_matrix": str(cm_path.relative_to(config.ROOT_DIR)),
        })
        if do_register:
            register(best["model_uri"])

    print("\nModel comparison (sorted by CV ROC-AUC):")
    print(table.to_string(index=False))
    return {"results": results, "fitted": fitted, "best_name": best_name,
            "split": (X_train, X_test, y_train, y_test)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="small grids, for CI")
    parser.add_argument("--no-register", action="store_true", help="skip Model Registry")
    parser.add_argument("--experiment", default=config.MLFLOW_EXPERIMENT)
    args = parser.parse_args()
    main(quick=args.quick, do_register=not args.no_register, experiment=args.experiment)
