<#
.SYNOPSIS
    Start vLLM model containers and enable them in the API registry.

.DESCRIPTION
    Starts one or more vLLM containers, sets the matching ENABLED flag in .env,
    and restarts the API so the model appears in the UI model selector.

.PARAMETER Model
    Which model(s) to start: qwen | llama | mistral | all
    Defaults to "all" (starts every GPU-profile container).

.EXAMPLE
    .\scripts\start_gpu.ps1 qwen       # start Qwen3-4B only
    .\scripts\start_gpu.ps1 llama      # start Llama-3.2-3B only
    .\scripts\start_gpu.ps1 mistral    # start Mistral-7B only
    .\scripts\start_gpu.ps1 all        # start all three
#>
param(
    [ValidateSet("qwen","llama","mistral","all")]
    [string]$Model = "all"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Root = Split-Path $PSScriptRoot -Parent
$EnvFile = Join-Path $Root ".env"

function Enable-VllmFlag([string]$key) {
    $content = Get-Content $EnvFile -Raw
    $content = $content -replace "(?m)^(${key}_ENABLED\s*=\s*).*$", "`${1}true"
    Set-Content $EnvFile $content -NoNewline
    Write-Host "  .env: ${key}_ENABLED=true"
}

$models = if ($Model -eq "all") { @("VLLM_QWEN","VLLM_LLAMA","VLLM_MISTRAL") }
          else                  { @("VLLM_$($Model.ToUpper())") }

Write-Host "`nEnabling model(s) in .env..."
foreach ($m in $models) { Enable-VllmFlag $m }

Write-Host "`nStarting vLLM container(s) (profile: $Model)..."
$profile = if ($Model -eq "all") { "gpu" } else { $Model }
& docker compose --profile $profile up -d

Write-Host "`nRestarting API to refresh model registry..."
& docker compose restart api

Write-Host "`nDone. Open http://localhost:8000/ and check the model selector."
Write-Host "Note: first startup downloads model weights from HuggingFace (~minutes)."
Write-Host "      Monitor progress:  docker logs rag_vllm_$($Model -replace 'all','qwen') -f"
