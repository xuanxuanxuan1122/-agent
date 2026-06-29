$ErrorActionPreference = "Stop"

$env:PYTHONUTF8 = "1"
$env:REPORT_EVIDENCE_MODE = "advisory_weight"
$env:REPORT_QUALITY_MODE = "high"
$env:REPORT_ENABLE_TABLES = "false"
$env:REPORT_TARGET_BODY_CHARS = "12000"
$env:REPORT_TARGET_BODY_CHARS_BLOCKING = "false"
$env:STAGE_PROBE_ENABLED = "true"
$env:MODULE_PROBE_ENABLED = "true"
$env:RUN_TRACE_ENABLED = "true"
$env:TOPIC_BUNDLE_CACHE_ENABLED = "false"
$env:TOPIC_BUNDLE_CACHE_ALLOW_SKIP_SEARCH = "false"
$env:TOPIC_BUNDLE_CACHE_REUSE_ANALYSIS = "false"
$env:IQS_SEARCH_CACHE_ENABLED = "false"
$env:EVIDENCE_CACHE_ENABLED = "false"
$env:BRAIN_LLM_ANALYSIS_CACHE_ENABLED = "false"

$sessionId = "advisory_quality_smoke_" + (Get-Date -Format "yyyyMMdd_HHmmss")
$outputDir = Join-Path "output/full_reports" $sessionId

python -X utf8 -m rag_pipeline.flows.report.full_report `
  --query "中国低空经济商业化落地机会与风险" `
  --route web `
  --output-dir $outputDir `
  --session-id $sessionId `
  --supervisor-max-loops 1 `
  --supervisor-max-followup-queries 8 `
  --no-progress-bar `
  --no-interactive-input

