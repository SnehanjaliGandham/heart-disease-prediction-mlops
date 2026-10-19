# Run the whole ML pipeline from a clean checkout (Windows PowerShell):
#   .\scripts\run_pipeline.ps1            # full grid search (about 5 minutes)
#   .\scripts\run_pipeline.ps1 -Quick     # small grids (about 1 minute)
# Steps: environment check -> data (checksum-verified) -> cleaning -> training + MLflow
#        -> quality gate -> sample prediction -> lint + tests. Stops at the first failure.
param([switch]$Quick)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

function Step($title, [scriptblock]$cmd) {
    Write-Host "`n========== $title ==========" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED: $title" -ForegroundColor Red; exit $LASTEXITCODE }
}

$trainArgs = @()
if ($Quick) { $trainArgs += "--quick" }

Step "1/7 Environment check"        { python scripts/check_env.py }
Step "2/7 Data (download + verify)" { python scripts/download_data.py }
Step "3/7 Validate and clean"       { python -m src.data }
Step "4/7 Train, tune, track"       { python -m src.train @trainArgs }
Step "5/7 Quality gate"             { python scripts/quality_gate.py }
Step "6/7 Sample prediction"        { python -m src.predict }
Step "7/7 Lint and tests"           { ruff check .; if ($LASTEXITCODE -eq 0) { python -m pytest -q } }

Write-Host "`nPipeline finished. Browse runs with:  mlflow ui --backend-store-uri sqlite:///mlflow.db" -ForegroundColor Green
