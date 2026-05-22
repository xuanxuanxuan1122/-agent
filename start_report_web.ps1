$ErrorActionPreference = "Stop"

$pipelineRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceRoot = Split-Path -Parent $pipelineRoot
$pythonExe = Join-Path $workspaceRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "Python executable not found: $pythonExe"
}

Set-Location $pipelineRoot
& $pythonExe -m report_web_app.main
