# SPDX-License-Identifier: MIT
param([int]$Port = 8765)
$ErrorActionPreference = 'Stop'
$ProjectRoot = $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python)) {
    throw 'Run setup.ps1 first to create the project environment.'
}
Push-Location -LiteralPath $ProjectRoot
try {
    Write-Host "FedBrain demo: http://127.0.0.1:$Port" -ForegroundColor Cyan
    & $Python -m fedbrain.server --port $Port
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
