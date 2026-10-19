#!/usr/bin/env bash
# Run the whole ML pipeline from a clean checkout (Linux / macOS / BITS Prayogshala):
#   bash scripts/run_pipeline.sh            # full grid search (about 5 minutes)
#   bash scripts/run_pipeline.sh --quick    # small grids (about 1 minute)
# Steps: environment check -> data (checksum-verified) -> cleaning -> training + MLflow
#        -> quality gate -> sample prediction -> lint + tests. Stops at the first failure.
set -euo pipefail
cd "$(dirname "$0")/.."
QUICK=""
[ "${1:-}" = "--quick" ] && QUICK="--quick"
PY=${PYTHON:-python}

step() { printf '\n========== %s ==========\n' "$1"; }

step "1/7 Environment check";        $PY scripts/check_env.py
step "2/7 Data (download + verify)"; $PY scripts/download_data.py
step "3/7 Validate and clean";       $PY -m src.data
step "4/7 Train, tune, track";       $PY -m src.train $QUICK
step "5/7 Quality gate";             $PY scripts/quality_gate.py
step "6/7 Sample prediction";        $PY -m src.predict
# (two separate commands: with `set -e`, a failure on the left of `&&` would not stop the script)
step "7/7 Lint and tests";           ruff check .; $PY -m pytest -q

printf '\nPipeline finished. Browse runs with:  mlflow ui --backend-store-uri sqlite:///mlflow.db\n'
