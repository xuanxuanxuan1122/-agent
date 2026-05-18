$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$startIngest = Join-Path $scriptRoot "start_ingest.ps1"

if (-not (Test-Path $startIngest)) {
    throw "Required script not found: $startIngest"
}

& $startIngest --skip-slicing --drop-if-exists --reupsert-existing --preview-top-k 0 @args
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    Write-Host "[FAILED] Existing chunk/vector sync failed. ExitCode=$exitCode" -ForegroundColor Red
    throw "Existing chunk/vector sync failed with exit code $exitCode"
}

Write-Host "[DONE] Existing chunks and embeddings have been synced to Qdrant." -ForegroundColor Green
