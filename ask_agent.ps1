$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[Console]::InputEncoding = $utf8NoBom
[Console]::OutputEncoding = $utf8NoBom
$OutputEncoding = $utf8NoBom

function Import-DotEnvFile {
    param(
        [string]$Path
    )

    if (-not (Test-Path $Path)) {
        return
    }
    foreach ($rawLine in Get-Content -Path $Path -Encoding UTF8) {
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

Import-DotEnvFile -Path (Join-Path $scriptRoot ".env")

$agentUrl = $env:RAG_AGENT_URL
if ([string]::IsNullOrWhiteSpace($agentUrl)) {
    $agentUrl = "http://127.0.0.1:7860"
}

$queryParts = @($args)
if ($queryParts.Count -gt 0 -and $queryParts[0] -eq "--query") {
    if ($queryParts.Count -gt 1) {
        $queryParts = $queryParts[1..($queryParts.Count - 1)]
    } else {
        $queryParts = @()
    }
}

if ($queryParts.Count -eq 0) {
    $queryText = Read-Host "Enter query"
} else {
    $queryText = ($queryParts -join " ").Trim()
}

if ([string]::IsNullOrWhiteSpace($queryText)) {
    throw "Query cannot be empty."
}

$body = @{
    query = $queryText
    answer_only = $true
    show_evidence = $true
} | ConvertTo-Json -Depth 5

$response = Invoke-RestMethod -Method Post -Uri "$agentUrl/ask" -Body $body -ContentType "application/json; charset=utf-8"
$response.answer_text
