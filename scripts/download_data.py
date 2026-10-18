"""Download the UCI Heart Disease (Cleveland) dataset.

Usage (from the repo root):
    python scripts/download_data.py            # download only if the CSV is missing
    python scripts/download_data.py --force    # re-download

Writes to <repo>/data/:
    heart_disease_raw.csv         13 features + original 5-class target `num`
    heart_disease_variables.csv   UCI variable descriptions
    heart_disease_metadata.txt    UCI dataset metadata
and prints the SHA-256 of the raw CSV so a re-download can be verified.
"""
import hashlib
import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_NAME = "heart_disease_raw.csv"
UCI_ID = 45  # Heart Disease (Cleveland subset used here: 303 rows)


def download(data_dir=DATA_DIR, force=False):
    """Return the path of the raw CSV, downloading it from UCI if needed."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_path = data_dir / RAW_NAME

    if raw_path.exists() and not force:
        print(f"Using existing file: data/{RAW_NAME}")
        return raw_path

    # Imported here so that offline runs (CSV already committed) still work.
    import pandas as pd
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=UCI_ID)
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    assert df.shape == (303, 14), f"Unexpected shape {df.shape}"

    df.to_csv(raw_path, index=False)
    dataset.variables.to_csv(data_dir / "heart_disease_variables.csv", index=False)
    (data_dir / "heart_disease_metadata.txt").write_text(
        str(dataset.metadata), encoding="utf-8"
    )

    digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    print(f"Saved data/{RAW_NAME} {df.shape}  sha256={digest}")
    return raw_path


if __name__ == "__main__":
    download(force="--force" in sys.argv)
