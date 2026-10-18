"""Inference helpers: load the packaged model and score patient records.

Used by the API (Step 6) and runnable on its own:
    python -m src.predict                      # scores a built-in sample patient
    python -m src.predict --json patient.json  # scores records from a file
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src import config


@lru_cache(maxsize=1)
def load_model(path: str = str(config.MODEL_PATH)):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Train first: python -m src.train")
    return joblib.load(path)


def to_frame(records) -> pd.DataFrame:
    """Accept one dict or a list of dicts; return a frame in training column order."""
    if isinstance(records, dict):
        records = [records]
    df = pd.DataFrame(records)
    missing = set(config.FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")
    return df[config.FEATURES].astype("float64")


def predict(records, model=None, threshold: float = 0.5) -> list[dict]:
    """Return prediction, label and probability (confidence) for each record."""
    model = model or load_model()
    proba = model.predict_proba(to_frame(records))[:, 1]
    results = []
    for p in proba:
        pred = int(p >= threshold)
        results.append({
            "prediction": pred,
            "label": "Heart disease" if pred else "No heart disease",
            "probability_disease": round(float(p), 4),
            "confidence": round(float(p if pred else 1 - p), 4),
        })
    return results


SAMPLE_PATIENT = {
    "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
    "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
    "ca": 0, "thal": 6,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, help="file with one record or a list")
    args = parser.parse_args()
    data = json.loads(args.json.read_text()) if args.json else SAMPLE_PATIENT
    print(json.dumps(predict(data), indent=2))
