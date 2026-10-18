"""Model quality gate: stop the pipeline if a newly trained model is not good enough.

Run after `python -m src.train` (CI does this automatically):
    python scripts/quality_gate.py

It reads models/model_metadata.json and reports/model_comparison.csv, checks the
selected model against minimum standards, prints a table (also written to the
GitHub Actions job summary when available) and exits with code 1 on failure, so a
weaker model can never move on to packaging and deployment.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402

# Minimum standards. Recall matters most: a missed disease case is the costly error.
THRESHOLDS = {
    "cv_roc_auc_mean": 0.85,
    "test_roc_auc": 0.85,
    "test_recall": 0.80,
    "test_accuracy": 0.75,
}
MIN_GAIN_OVER_BASELINE = 0.20   # CV ROC-AUC above the majority-class baseline


def evaluate_gate(metadata: dict, comparison: pd.DataFrame) -> list[dict]:
    metrics = {**metadata["cv_metrics"], **metadata["test_metrics"]}
    checks = [
        {"check": name, "value": metrics[name], "required": f">= {limit}",
         "passed": metrics[name] >= limit}
        for name, limit in THRESHOLDS.items()
    ]
    baseline = comparison.set_index("model").loc["baseline_majority", "cv_roc_auc_mean"]
    gain = metrics["cv_roc_auc_mean"] - baseline
    checks.append({"check": "gain over baseline (CV ROC-AUC)", "value": round(gain, 4),
                   "required": f">= {MIN_GAIN_OVER_BASELINE}",
                   "passed": gain >= MIN_GAIN_OVER_BASELINE})
    return checks


def to_markdown(model_name: str, checks: list[dict]) -> str:
    lines = [f"### Model quality gate - `{model_name}`", "",
             "| Check | Value | Required | Result |", "|---|---|---|---|"]
    for c in checks:
        lines.append(f"| {c['check']} | {c['value']:.4f} | {c['required']} | "
                     f"{'PASS' if c['passed'] else 'FAIL'} |")
    return "\n".join(lines) + "\n"


def main(metadata_path: Path = config.MODEL_METADATA_PATH,
         comparison_path: Path = config.REPORTS_DIR / "model_comparison.csv") -> int:
    metadata = json.loads(Path(metadata_path).read_text())
    checks = evaluate_gate(metadata, pd.read_csv(comparison_path))
    report = to_markdown(metadata["model_name"], checks)
    print(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(report)
    failed = [c["check"] for c in checks if not c["passed"]]
    if failed:
        print(f"QUALITY GATE FAILED: {', '.join(failed)}")
        return 1
    print("Quality gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
