# SPDX-License-Identifier: MIT
param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
try {
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        & $Python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Could not create the virtual environment. Pass a real Python 3.10+ path with -Python.' }
    }
    & '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    Write-Host 'Environment ready. Download data: .\.venv\Scripts\python.exe -m fedbrain.runner download' -ForegroundColor Green
} finally {
    Pop-Location
}
