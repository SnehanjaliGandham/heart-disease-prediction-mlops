"""Data loading, validation, cleaning and splitting.

The same cleaning rules as notebook 01 (section 2.10), but as plain functions so
CI, the training script and the tests can run them without Jupyter.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def load_raw(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw UCI CSV (13 features + 5-class `num`)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    return pd.read_csv(path)


def validate_schema(df: pd.DataFrame, target: str) -> None:
    """Fail loudly if columns are missing or values fall outside known codes."""
    expected = set(config.FEATURES) | {target}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    for col, codes in config.CATEGORY_CODES.items():
        bad = set(df[col].dropna().unique()) - set(codes)
        if bad:
            raise ValueError(f"Unexpected codes in '{col}': {sorted(bad)}")

    for col, (low, high) in config.NUMERIC_RANGES.items():
        values = df[col].dropna()
        if ((values < low) | (values > high)).any():
            raise ValueError(f"Values of '{col}' outside [{low}, {high}]")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Binarise the target and drop exact duplicates.

    Missing values in `ca` / `thal` are deliberately kept: they are imputed inside
    the model pipeline so imputation is learned on training folds only.
    """
    validate_schema(df, config.RAW_TARGET)
    out = df.drop_duplicates().copy()
    out[config.TARGET] = (out[config.RAW_TARGET] > 0).astype(int)
    out = out.drop(columns=[config.RAW_TARGET])
    return out.reset_index(drop=True)   # original UCI column order, target last


def build_clean_dataset(
    raw_path: Path = config.RAW_DATA_PATH, clean_path: Path = config.CLEAN_DATA_PATH
) -> pd.DataFrame:
    """Raw CSV -> cleaned CSV. Returns the cleaned frame."""
    cleaned = clean(load_raw(raw_path))
    Path(clean_path).parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(clean_path, index=False)
    return cleaned


def load_clean(path: Path = config.CLEAN_DATA_PATH) -> pd.DataFrame:
    """Load the cleaned dataset, creating it from the raw file if needed."""
    path = Path(path)
    df = pd.read_csv(path) if path.exists() else build_clean_dataset(clean_path=path)
    validate_schema(df, config.TARGET)
    return df


def split(df: pd.DataFrame, test_size: float = config.TEST_SIZE,
          random_state: int = config.RANDOM_STATE):
    """Stratified train/test split -> X_train, X_test, y_train, y_test."""
    # All features as float64: a model input schema of doubles accepts missing
    # values (NaN) and matches what the API will send.
    X = df[config.FEATURES].astype("float64")
    y = df[config.TARGET]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )


def file_sha256(path: Path) -> str:
    """Fingerprint of a data file, logged with every run for traceability."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == "__main__":
    frame = build_clean_dataset()
    print(f"Saved {config.CLEAN_DATA_PATH.relative_to(config.ROOT_DIR)} {frame.shape}")
