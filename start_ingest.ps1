$ErrorActionPreference = "Stop"

$pipelineRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceRoot = Split-Path -Parent $pipelineRoot
$pythonExe = Join-Path $workspaceRoot ".venv\Scripts\python.exe"

function Import-DotEnvFile {
    param(
        [string[]]$Paths
    )

    foreach ($path in $Paths) {
        if (-not (Test-Path $path)) {
            continue
        }
        foreach ($rawLine in Get-Content -Path $path -Encoding UTF8) {
            $line = $rawLine.Trim()
            if ([string]::IsNullOrWhiteSpace($line) -or $line.StartsWith("#")) {
                continue
            }
            if ($line.StartsWith("export ")) {
                $line = $line.Substring(7).Trim()
            }
            $equalsIndex = $line.IndexOf("=")
            if ($equalsIndex -le 0) {
                continue
            }
            $key = $line.Substring(0, $equalsIndex).Trim()
            $value = $line.Substring($equalsIndex + 1).Trim()
            if ($value.Length -ge 2) {
                if (($value.StartsWith('"') -and $value.EndsWith('"')) -or ($value.StartsWith("'") -and $value.EndsWith("'"))) {
                    $value = $value.Substring(1, $value.Length - 2)
                }
            }
            $existing = [Environment]::GetEnvironmentVariable($key, "Process")
            if ([string]::IsNullOrWhiteSpace($existing)) {
                Set-Item -Path "Env:$key" -Value $value
            }
        }
    }
}

if (-not (Test-Path $pythonExe)) {
    throw "Python executable not found: $pythonExe"
}

Set-Location $pipelineRoot
Import-DotEnvFile -Paths @(
    (Join-Path $pipelineRoot ".env")
)

if ([string]::IsNullOrWhiteSpace($env:QDRANT_URL)) {
    try {
        $null = Invoke-WebRequest -UseBasicParsing http://127.0.0.1:6333/collections -TimeoutSec 2
        $env:QDRANT_URL = "http://127.0.0.1:6333"
    } catch {
        $env:QDRANT_URL = ""
    }
}

$env:QDRANT_COLLECTION_NAME = $env:QDRANT_COLLECTION_NAME
if ([string]::IsNullOrWhiteSpace($env:QDRANT_COLLECTION_NAME)) {
    $env:QDRANT_COLLECTION_NAME = "rag_local_chunks"
}

if ([string]::IsNullOrWhiteSpace($env:QDRANT_PREFER_GRPC)) {
    $env:QDRANT_PREFER_GRPC = "0"
}

if ([string]::IsNullOrWhiteSpace($env:QDRANT_URL)) {
    Write-Host "[INFO] QDRANT_URL=(auto-detect disabled, script fallback will decide)"
} else {
    Write-Host "[INFO] QDRANT_URL=$env:QDRANT_URL"
}
Write-Host "[INFO] QDRANT_COLLECTION_NAME=$env:QDRANT_COLLECTION_NAME"

& $pythonExe -m rag_pipeline.pipelines.ingest_pipeline @args
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    Write-Host "[FAILED] Slicing/vector storage pipeline failed. ExitCode=$exitCode" -ForegroundColor Red
    throw "Slicing/vector storage pipeline failed with exit code $exitCode"
}

Write-Host "[DONE] Slicing/vector storage pipeline completed successfully." -ForegroundColor Green
