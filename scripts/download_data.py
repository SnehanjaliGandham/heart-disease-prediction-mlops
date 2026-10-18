"""Download the UCI Heart Disease (Cleveland) dataset.

Usage (from the repo root):
    python scripts/download_data.py            # download only if the CSV is missing
    python scripts/download_data.py --force    # re-download
    python scripts/download_data.py --verify   # only check the CSV against the manifest

Writes to <repo>/data/:
    heart_disease_raw.csv         13 features + original 5-class target `num`
    heart_disease_variables.csv   UCI variable descriptions
    heart_disease_metadata.txt    UCI dataset metadata
    heart_disease_manifest.json   source, shape and SHA-256 of the raw CSV

The manifest makes the data step verifiable: every later run (locally, in
Prayogshala or in CI) recomputes the SHA-256 and stops if the file has changed.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RAW_NAME = "heart_disease_raw.csv"
MANIFEST_NAME = "heart_disease_manifest.json"
UCI_ID = 45  # Heart Disease (Cleveland subset used here: 303 rows)
UCI_URL = "https://archive.ics.uci.edu/dataset/45/heart+disease"
EXPECTED_SHAPE = (303, 14)


def normalised_bytes(path: Path) -> bytes:
    """File content with Windows line endings (CRLF) turned into Unix ones (LF).

    Git checks text files out with CRLF on Windows and LF on Linux, so the raw bytes
    differ between a laptop, Prayogshala and CI even though the data is identical.
    Hashing the normalised content gives one fingerprint on every platform.
    """
    return Path(path).read_bytes().replace(b"\r\n", b"\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(normalised_bytes(path)).hexdigest()


def write_manifest(raw_path: Path) -> dict:
    """Record where the data came from and its fingerprint."""
    raw_path = Path(raw_path)
    lines = raw_path.read_text(encoding="utf-8").strip().splitlines()
    manifest = {
        "source": UCI_URL,
        "uci_id": UCI_ID,
        "file": raw_path.name,
        "rows": len(lines) - 1,
        "columns": len(lines[0].split(",")),
        "bytes": len(normalised_bytes(raw_path)),     # LF line endings
        "sha256": sha256(raw_path),
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (raw_path.parent / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify(raw_path: Path) -> bool:
    """True if the CSV matches its manifest. Raises if they disagree."""
    raw_path = Path(raw_path)
    manifest_path = raw_path.parent / MANIFEST_NAME
    if not manifest_path.exists():
        print(f"No manifest yet - creating {MANIFEST_NAME}")
        write_manifest(raw_path)
        return True
    expected = json.loads(manifest_path.read_text())["sha256"]
    actual = sha256(raw_path)
    if actual != expected:
        raise ValueError(
            f"{raw_path.name} changed: sha256 {actual[:12]}... != manifest {expected[:12]}... "
            "Re-download with --force if the change is intended."
        )
    print(f"Verified data/{raw_path.name} sha256={actual[:12]}...")
    return True


def download(data_dir=DATA_DIR, force=False):
    """Return the path of the raw CSV, downloading it from UCI if needed."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_path = data_dir / RAW_NAME

    if raw_path.exists() and not force:
        print(f"Using existing file: data/{RAW_NAME}")
        verify(raw_path)
        return raw_path

    # Imported here so that offline runs (CSV already committed) still work.
    import pandas as pd
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=UCI_ID)
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    if df.shape != EXPECTED_SHAPE:
        raise ValueError(f"Unexpected shape {df.shape}, expected {EXPECTED_SHAPE}")

    df.to_csv(raw_path, index=False)
    dataset.variables.to_csv(data_dir / "heart_disease_variables.csv", index=False)
    (data_dir / "heart_disease_metadata.txt").write_text(
        str(dataset.metadata), encoding="utf-8"
    )
    manifest = write_manifest(raw_path)
    print(f"Saved data/{RAW_NAME} {df.shape}  sha256={manifest['sha256']}")
    return raw_path


if __name__ == "__main__":
    if "--verify" in sys.argv:
        verify(DATA_DIR / RAW_NAME)
    else:
        download(force="--force" in sys.argv)
