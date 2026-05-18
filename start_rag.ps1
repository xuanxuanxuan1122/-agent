$ErrorActionPreference = "Stop"

$pipelineRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceRoot = Split-Path -Parent $pipelineRoot
$pythonExe = Join-Path $workspaceRoot ".venv\Scripts\python.exe"
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = $utf8NoBom
[Console]::OutputEncoding = $utf8NoBom
$OutputEncoding = $utf8NoBom
$env:PYTHONUTF8 = "1"
if ([string]::IsNullOrWhiteSpace($env:PYTHONIOENCODING)) {
    $env:PYTHONIOENCODING = "utf-8"
}
if ([string]::IsNullOrWhiteSpace($env:TQDM_DISABLE)) {
    $env:TQDM_DISABLE = "1"
}
if ([string]::IsNullOrWhiteSpace($env:HF_HUB_DISABLE_PROGRESS_BARS)) {
    $env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
}
if ([string]::IsNullOrWhiteSpace($env:TRANSFORMERS_NO_ADVISORY_WARNINGS)) {
    $env:TRANSFORMERS_NO_ADVISORY_WARNINGS = "1"
}
if ([string]::IsNullOrWhiteSpace($env:TOKENIZERS_PARALLELISM)) {
    $env:TOKENIZERS_PARALLELISM = "false"
}

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

function Invoke-SearchCommand {
    param(
        [object[]]$RawSearchArgs
    )

    $searchArgs = @($RawSearchArgs)
    if ($searchArgs.Count -gt 0 -and $searchArgs[0] -eq "search") {
        if ($searchArgs.Count -gt 1) {
            $searchArgs = $searchArgs[1..($searchArgs.Count - 1)]
        } else {
            $searchArgs = @()
        }
    }

    if ($searchArgs.Count -eq 0) {
        $queryText = Read-Host "Enter query"
        if ([string]::IsNullOrWhiteSpace($queryText)) {
            throw "Query cannot be empty."
        }
        $searchArgs = @("--query", $queryText)
    }

    $hasJson = $searchArgs -contains "--json"
    $hasAnswerOnly = $searchArgs -contains "--answer-only"
    $hasSynthesisSwitch = ($searchArgs -contains "--enable-llm-synthesis") -or ($searchArgs -contains "--disable-llm-synthesis")
    $hasReviewSwitch = ($searchArgs -contains "--enable-llm-answer-review") -or ($searchArgs -contains "--disable-llm-answer-review")

    if (-not $hasSynthesisSwitch) {
        $searchArgs = @("--enable-llm-synthesis") + $searchArgs
    }
    if (-not $hasReviewSwitch -and $env:RAG_ENABLE_LLM_ANSWER_REVIEW -eq "1") {
        $searchArgs = @("--enable-llm-answer-review") + $searchArgs
    }
    if (-not $hasJson -and -not $hasAnswerOnly) {
        $searchArgs = @("--answer-only") + $searchArgs
        $hasAnswerOnly = $true
    }

    if ($hasAnswerOnly -and -not $hasJson) {
        $stderrFile = [System.IO.Path]::GetTempFileName()
        $previousErrorActionPreference = $ErrorActionPreference
        try {
            $ErrorActionPreference = "Continue"
            & $pythonExe -m rag_pipeline.agents.brain_agent @searchArgs 2> $stderrFile
            $exitCode = $LASTEXITCODE
            $ErrorActionPreference = $previousErrorActionPreference
            if ($exitCode -ne 0) {
                $stderrText = Get-Content -Path $stderrFile -Raw -ErrorAction SilentlyContinue
                if (-not [string]::IsNullOrWhiteSpace($stderrText)) {
                    Write-Error $stderrText
                }
                throw "Brain agent failed with exit code $exitCode"
            }
        } finally {
            $ErrorActionPreference = $previousErrorActionPreference
            Remove-Item -LiteralPath $stderrFile -Force -ErrorAction SilentlyContinue
        }
        return
    }

    & $pythonExe -m rag_pipeline.search.engine @searchArgs
}

function Invoke-FullReportCommand {
    param(
        [object[]]$RawReportArgs,
        [bool]$ForceSelectLlm = $false
    )

    $reportArgs = @($RawReportArgs)
    $noReportArgs = $reportArgs.Count -eq 0
    $hasLlmProfile = $reportArgs -contains "--llm-profile"
    $hasSelectLlm = $reportArgs -contains "--select-llm"
    $envSelectLlm = $env:REPORT_SELECT_LLM_PROFILE -eq "true" -or $env:REPORT_SELECT_LLM_PROFILE -eq "1"
    $shouldSelectLlm = ($ForceSelectLlm -or $envSelectLlm -or $noReportArgs) -and -not $hasLlmProfile -and -not $hasSelectLlm
    if ($shouldSelectLlm) {
        $reportArgs = @("--select-llm") + $reportArgs
        $hasSelectLlm = $true
    }

    if ($reportArgs.Count -eq 0) {
        $queryText = Read-Host "请输入报告问题/主题"
        if ([string]::IsNullOrWhiteSpace($queryText)) {
            throw "Query cannot be empty."
        }
        $reportArgs = @("--query", $queryText)
    }

    & $pythonExe (Join-Path $pipelineRoot "run_full_report.py") @reportArgs
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

$showStartupInfo = $env:RAG_SHOW_STARTUP_INFO -eq "1"
if ($args.Count -gt 0 -and @("ingest", "sync", "benchmark") -contains $args[0]) {
    $showStartupInfo = $true
}
if ($showStartupInfo) {
    if ([string]::IsNullOrWhiteSpace($env:QDRANT_URL)) {
        Write-Host "[INFO] QDRANT_URL=(auto-detect disabled, script fallback will decide)"
    } else {
        Write-Host "[INFO] QDRANT_URL=$env:QDRANT_URL"
    }
    Write-Host "[INFO] QDRANT_COLLECTION_NAME=$env:QDRANT_COLLECTION_NAME"
}

if ($args.Count -gt 0 -and @("brain", "agent") -contains $args[0]) {
    $brainArgs = @()
    if ($args.Count -gt 1) {
        $brainArgs = $args[1..($args.Count - 1)]
    }
    $stderrFile = [System.IO.Path]::GetTempFileName()
    $previousErrorActionPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & $pythonExe -m rag_pipeline.agents.brain_agent @brainArgs 2> $stderrFile
        $exitCode = $LASTEXITCODE
        $ErrorActionPreference = $previousErrorActionPreference
        if ($exitCode -ne 0) {
            $stderrText = Get-Content -Path $stderrFile -Raw -ErrorAction SilentlyContinue
            if (-not [string]::IsNullOrWhiteSpace($stderrText)) {
                Write-Error $stderrText
            }
            exit $exitCode
        }
    } finally {
        $ErrorActionPreference = $previousErrorActionPreference
        Remove-Item -LiteralPath $stderrFile -Force -ErrorAction SilentlyContinue
    }
} elseif ($args.Count -gt 0 -and @("web", "online") -contains $args[0]) {
    $webArgs = @()
    if ($args.Count -gt 1) {
        $webArgs = $args[1..($args.Count - 1)]
    }
    & $pythonExe -m rag_pipeline.agents.web_analysis_agent @webArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
} elseif ($args.Count -gt 0 -and @("web-search-api", "iqs-api") -contains $args[0]) {
    $apiArgs = @()
    if ($args.Count -gt 1) {
        $apiArgs = $args[1..($args.Count - 1)]
    }
    & $pythonExe -m rag_pipeline.agents.web_search_server @apiArgs
} elseif ($args.Count -gt 0 -and $args[0] -eq "serve") {
    $serveArgs = @()
    if ($args.Count -gt 1) {
        $serveArgs = $args[1..($args.Count - 1)]
    }
    & $pythonExe -m rag_pipeline.search.agent_server @serveArgs
} elseif ($args.Count -gt 0 -and $args[0] -eq "ingest") {
    $ingestArgs = @()
    if ($args.Count -gt 1) {
        $ingestArgs = $args[1..($args.Count - 1)]
    }
    & $pythonExe -m rag_pipeline.pipelines.ingest_pipeline @ingestArgs
} elseif ($args.Count -gt 0 -and $args[0] -eq "sync") {
    $syncUrl = $env:QDRANT_URL
    if ([string]::IsNullOrWhiteSpace($syncUrl)) {
        $syncUrl = "http://127.0.0.1:6333"
    }
    $syncInputPath = $env:RAG_EMBED_OUTPUT_DIR
    if ([string]::IsNullOrWhiteSpace($syncInputPath)) {
        $syncInputPath = Join-Path $workspaceRoot "rag_chunks_store"
    }
    $syncArgs = @(
        "--input-path", $syncInputPath,
        "--output-dir", $syncInputPath,
        "--url", $syncUrl,
        "--collection", $env:QDRANT_COLLECTION_NAME,
        "--reupsert-existing",
        "--no-write-json"
    )
    if ($env:QDRANT_PREFER_GRPC -eq "1") {
        $syncArgs += "--prefer-grpc"
    }
    if ($args.Count -gt 1) {
        $syncArgs += $args[1..($args.Count - 1)]
    }
    & $pythonExe -m rag_pipeline.ingest.embedding_qdrant @syncArgs
} elseif ($args.Count -gt 0 -and $args[0] -eq "benchmark") {
    $benchmarkArgs = @()
    if ($args.Count -gt 1) {
        $benchmarkArgs = $args[1..($args.Count - 1)]
    }
    & $pythonExe -m rag_pipeline.tools.benchmark @benchmarkArgs
} elseif ($args.Count -gt 0 -and @("report", "full-report", "research") -contains $args[0]) {
    $reportArgs = @()
    if ($args.Count -gt 1) {
        $reportArgs = $args[1..($args.Count - 1)]
    }
    Invoke-FullReportCommand -RawReportArgs $reportArgs
} elseif ($args.Count -gt 0 -and @("report-select", "full-report-select", "research-select") -contains $args[0]) {
    $reportArgs = @()
    if ($args.Count -gt 1) {
        $reportArgs = $args[1..($args.Count - 1)]
    }
    Invoke-FullReportCommand -RawReportArgs $reportArgs -ForceSelectLlm $true
} elseif ($args.Count -eq 0) {
    Invoke-FullReportCommand -RawReportArgs @() -ForceSelectLlm $true
} else {
    Invoke-SearchCommand -RawSearchArgs $args
}
