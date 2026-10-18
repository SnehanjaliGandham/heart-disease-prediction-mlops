"""Metrics and evaluation plots shared by the training script and notebook 02."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: works in CI and Docker
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

CLASS_NAMES = ["No disease", "Disease"]


def classification_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def plot_confusion_matrix(model, X, y, title: str, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X, y, display_labels=CLASS_NAMES, cmap="Blues", colorbar=False, ax=ax
    )
    ax.set_title(title)
    ax.grid(False)
    return _save(fig, path)


def plot_roc_curves(models: dict, X, y, path: Path) -> Path:
    """One ROC curve per fitted model on the same held-out data."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, model in models.items():
        proba = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y, proba):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate (recall)")
    ax.legend(loc="lower right")
    ax.set_title("ROC curves on the held-out test set")
    return _save(fig, path)


def feature_importance(pipeline) -> pd.Series | None:
    """Importance per encoded feature: |coef| for linear models, impurity for trees."""
    model = pipeline.named_steps["model"]
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    if hasattr(model, "coef_"):
        values = np.abs(model.coef_).ravel()
    elif hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    else:
        return None
    clean_names = [n.split("__", 1)[-1] for n in names]
    return pd.Series(values, index=clean_names).sort_values(ascending=False)


def plot_feature_importance(pipeline, title: str, path: Path, top: int = 15) -> Path | None:
    importance = feature_importance(pipeline)
    if importance is None:
        return None
    fig, ax = plt.subplots(figsize=(6, 5))
    importance.head(top)[::-1].plot.barh(ax=ax, color="#4C78A8")
    ax.set_title(title)
    ax.set_xlabel("importance")
    return _save(fig, path)


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
