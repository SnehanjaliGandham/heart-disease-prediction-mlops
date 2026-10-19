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
import subprocess
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
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
SKOPS_TRUSTED_TYPES = ["numpy.dtype", "sklearn.tree._tree.Tree",
                       "src.features.ClinicalFeatures",              # our own transformer
                       "sklearn.compose._column_transformer.make_column_selector"]
# Older MLflow 3.x releases pickle models and have no such argument.
_SAVE_KWARGS = (
    {"skops_trusted_types": SKOPS_TRUSTED_TYPES}
    if "skops_trusted_types" in inspect.signature(mlflow.sklearn.save_model).parameters
    else {}
)
# The pipeline contains our own ClinicalFeatures class, so the MLflow model carries the
# few modules it needs and loads anywhere (e.g. models:/heart-disease-classifier@champion).
MODEL_CODE = ["__init__.py", "config.py", "features.py"]


def model_code_dir() -> str:
    """A minimal copy of the `src` package with only what the saved pipeline imports."""
    target = Path(tempfile.mkdtemp(prefix="model_code_")) / "src"
    target.mkdir()
    for name in MODEL_CODE:
        shutil.copy2(Path(__file__).parent / name, target / name)
    return str(target)


def clean_params(params: dict) -> dict:
    """{'model__C': 1, 'features__add_hr_ratio': True} -> {'C': 1, 'add_hr_ratio': True}"""
    return {k.split("__", 1)[-1]: v for k, v in params.items()}


def select_model(results: pd.DataFrame) -> tuple[str, str]:
    """Apply the selection rule fixed in config; return (model name, explanation).

    1. Rank the real models by mean CV score (the test set is never used).
    2. Every model within SELECTION_TOLERANCE of the best is considered equally good.
    3. Among those, take the simplest one (config.SIMPLICITY_ORDER).
    """
    metric = f"cv_{config.SELECTION_METRIC}_mean"
    real = results[results["model"] != "baseline_majority"]
    best_score = real[metric].max()
    tied = real[real[metric] >= best_score - config.SELECTION_TOLERANCE]["model"].tolist()
    order = {name: i for i, name in enumerate(config.SIMPLICITY_ORDER)}
    chosen = min(tied, key=lambda m: order.get(m, len(order)))
    top = real.loc[real[metric].idxmax(), "model"]
    if chosen == top and len(tied) == 1:
        reason = f"{chosen} has the highest CV {config.SELECTION_METRIC} ({best_score:.4f})"
    elif chosen == top:
        reason = (f"{chosen} has the highest CV {config.SELECTION_METRIC} ({best_score:.4f}) "
                  f"and is the simplest of {tied}")
    else:
        reason = (f"{tied} are within {config.SELECTION_TOLERANCE} of the best CV "
                  f"{config.SELECTION_METRIC} ({best_score:.4f}); {chosen} is the simplest")
    return chosen, reason


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
        return_train_score=True,       # to measure over-fitting (train vs validation gap)
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
    sel = config.SELECTION_METRIC
    out[f"cv_train_{sel}_mean"] = float(search.cv_results_[f"mean_train_{sel}"][i])
    # How much better the model scores on the folds it was fitted on than on the
    # held-out folds: a large gap means over-fitting.
    out["cv_overfit_gap"] = out[f"cv_train_{sel}_mean"] - out[f"cv_{sel}_mean"]
    return out


def feature_ablation(search: GridSearchCV) -> dict:
    """Best CV score with and without the engineered feature (other settings tuned freely).

    Shows whether hr_pct_predicted actually helps this model - the evidence behind keeping
    or dropping it, instead of assuming that a clinically sensible feature must help.
    """
    params = search.cv_results_["params"]
    scores = search.cv_results_[f"mean_test_{config.SELECTION_METRIC}"]
    out = {}
    for flag, label in ((False, "without"), (True, "with")):
        chosen = [s for p, s in zip(params, scores) if p.get("features__add_hr_ratio") is flag]
        if chosen:
            out[f"cv_{config.SELECTION_METRIC}_{label}_hr_ratio"] = float(max(chosen))
    return out


def log_candidate(name, search, X_train, X_test, y_test, tmp: Path) -> dict:
    """Log one candidate as a nested MLflow run and return its summary row."""
    model = search.best_estimator_
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    test = {f"test_{k}": float(v) for k, v in
            classification_metrics(y_test, y_pred, y_proba).items()}
    cv = {**cv_summary(search), **feature_ablation(search)}

    with mlflow.start_run(run_name=name, nested=True) as run:
        mlflow.set_tag("model_family", name)
        mlflow.log_params({"model_type": type(model.named_steps["model"]).__name__,
                           "n_param_combinations": len(search.cv_results_["params"])})
        mlflow.log_params(clean_params(search.best_params_))
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
            code_paths=[model_code_dir()], **_SAVE_KWARGS,
        )
        run_id = run.info.run_id

    return {"model": name, "run_id": run_id, "model_uri": info.model_uri,
            "best_params": search.best_params_, **cv, **test}


def git_commit() -> str:
    """Short hash of the code version being trained (``unknown`` outside a git repo)."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=config.ROOT_DIR,
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def check_round_trip(model, X_sample) -> None:
    """Reload the saved file and make sure it predicts exactly like the in-memory model."""
    reloaded = joblib.load(config.MODEL_PATH)
    if not np.allclose(reloaded.predict_proba(X_sample), model.predict_proba(X_sample)):
        raise RuntimeError("Saved model does not reproduce the trained model's predictions")
    log.info("Round-trip check passed: reloaded model gives identical probabilities")


def save_final_model(name, model, row, data_hash, tmp_plots: dict, X_sample=None) -> None:
    """Write the champion to models/ in two formats plus a metadata card."""
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.MODEL_PATH)
    if X_sample is not None:
        check_round_trip(model, X_sample)

    if config.MLFLOW_MODEL_DIR.exists():
        shutil.rmtree(config.MLFLOW_MODEL_DIR)
    mlflow.sklearn.save_model(model, config.MLFLOW_MODEL_DIR,
                              code_paths=[model_code_dir()], **_SAVE_KWARGS)

    metadata = {
        "model_name": name,
        "model_class": type(model.named_steps["model"]).__name__,
        "best_params": clean_params(row["best_params"]),
        "selection_metric": f"cv_{config.SELECTION_METRIC}_mean",
        "cv_metrics": {k: round(v, 4) for k, v in row.items() if k.startswith("cv_")},
        "test_metrics": {k: round(v, 4) for k, v in row.items() if k.startswith("test_")},
        "features": {"numeric": config.NUMERIC_FEATURES,
                     "categorical": config.CATEGORICAL_FEATURES,
                     "engineered": (config.ENGINEERED_FEATURES
                                    if row["best_params"].get("features__add_hr_ratio")
                                    else [])},
        "target": "1 = heart disease present (num > 0), 0 = absent",
        "mlflow_run_id": row["run_id"],
        "training_data_sha256": data_hash,
        "git_commit": git_commit(),
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
        mlflow.set_tags({"stage": "model_selection", "quick_mode": str(quick),
                         "git_commit": git_commit(), "python_version": platform.python_version(),
                         "sklearn_version": sklearn.__version__})
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
        best_name, reason = select_model(results)
        best = results[results["model"] == best_name].iloc[0]
        log.info("Selected model: %s - %s", best_name, reason)

        # Comparison artifacts on the parent run (and copies for the report).
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        cols = ["model"] + [c for c in results.columns if c.startswith(("cv_", "test_"))]
        table = results[cols].round(4)
        table_path = config.REPORTS_DIR / "model_comparison.csv"
        table.to_csv(table_path, index=False)
        ablation_cols = [c for c in results.columns if c.endswith("_hr_ratio")]
        ablation = results[results["model"] != "baseline_majority"][["model"] + ablation_cols]
        ablation = ablation.assign(feature_kept=[
            bool(clean_params(p).get("add_hr_ratio")) for p in
            results[results["model"] != "baseline_majority"]["best_params"]]).round(4)
        ablation_path = config.REPORTS_DIR / "feature_ablation.csv"
        ablation.to_csv(ablation_path, index=False)
        roc_path = plot_roc_curves(
            {k: v for k, v in fitted.items() if k != "baseline_majority"},
            X_test, y_test, config.FIGURES_DIR / "roc_curves.png")
        cm_path = plot_confusion_matrix(fitted[best_name], X_test, y_test,
                                        f"Selected: {best_name}",
                                        config.FIGURES_DIR / "confusion_matrix_best.png")
        imp_path = plot_feature_importance(fitted[best_name], f"Selected: {best_name}",
                                           config.FIGURES_DIR / "feature_importance_best.png")
        for p in (table_path, ablation_path, roc_path, cm_path, imp_path):
            if p:
                mlflow.log_artifact(p, "comparison")

        mlflow.set_tags({"selected_model": best_name, "selection_reason": reason})
        mlflow.log_params({"selection_tolerance": config.SELECTION_TOLERANCE,
                           "simplicity_order": ",".join(config.SIMPLICITY_ORDER)})
        mlflow.log_metrics({
            "best_cv_roc_auc": best[f"cv_{config.SELECTION_METRIC}_mean"],
            "best_test_roc_auc": best["test_roc_auc"],
            "best_test_accuracy": best["test_accuracy"],
        })

        save_final_model(best_name, fitted[best_name], best.to_dict(), data_hash, {
            "roc": str(roc_path.relative_to(config.ROOT_DIR)),
            "confusion_matrix": str(cm_path.relative_to(config.ROOT_DIR)),
        }, X_sample=X_test)
        if do_register:
            register(best["model_uri"])

    print("\nModel comparison (sorted by CV ROC-AUC):")
    print(table.to_string(index=False))
    print("\nEngineered feature hr_pct_predicted - best CV ROC-AUC without / with it:")
    print(ablation.to_string(index=False))
    print(f"\nSelected: {best_name} - {reason}")
    return {"results": results, "fitted": fitted, "best_name": best_name, "reason": reason,
            "split": (X_train, X_test, y_train, y_test)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="small grids, for CI")
    parser.add_argument("--no-register", action="store_true", help="skip Model Registry")
    parser.add_argument("--experiment", default=config.MLFLOW_EXPERIMENT)
    args = parser.parse_args()
    main(quick=args.quick, do_register=not args.no_register, experiment=args.experiment)
