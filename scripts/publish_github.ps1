# Push this repository to GitHub without storing the token in .git/config.
#   $env:GITHUB_TOKEN = '<fine-grained token with Contents: Read and write>'
#   .\scripts\publish_github.ps1 -Repo 'Cybermatics-prog/fedbrain-federated-brain-network'
param(
    [Parameter(Mandatory = $true)][string]$Repo,
    [string]$Branch = 'main',
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
if ($Repo -notmatch '^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$') {
    throw 'Repo must look like owner/name.'
}
$token = $env:GITHUB_TOKEN
if (-not $token) {
    throw 'Set $env:GITHUB_TOKEN to a token with Contents: Read and write before running this script.'
}
$env:GIT_TERMINAL_PROMPT = '0'
Push-Location -LiteralPath (Join-Path $PSScriptRoot '..')
try {
    if (-not (Test-Path -LiteralPath '.git')) { throw 'Run this from a cloned repository root.' }
    $existing = (git remote 2>$null) -contains 'origin'
    if ($existing) { git remote remove origin }
    git remote add origin "https://github.com/$Repo.git"
    Write-Host "Remote: https://github.com/$Repo.git (token supplied per command, not saved)" -ForegroundColor Cyan
    if ($DryRun) {
        Write-Host 'Dry run: skipping push.' -ForegroundColor Yellow
        return
    }
    # Authorization travels as a one-shot header so no credential lands in .git/config.
    $pair = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("x-access-token:$token"))
    git -c "http.extraheader=AUTHORIZATION: basic $pair" push -u origin $Branch
    if ($LASTEXITCODE -ne 0) { throw "git push failed with exit code $LASTEXITCODE" }
    Write-Host "Pushed. Verify: https://github.com/$Repo" -ForegroundColor Green
} finally {
    Pop-Location
}
