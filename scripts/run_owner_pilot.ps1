[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($env:OS -ne "Windows_NT") {
    throw "The owner-pilot launcher is intended for Windows."
}

$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$configPath = Join-Path $projectRoot ".photomatch-owner-config.xml"
$modelsPath = Join-Path $projectRoot "models"
$weightsPath = Join-Path $modelsPath ".deepface\weights"

function ConvertFrom-PhotoMatchSecureString {
    param([Security.SecureString]$Value)
    $pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Value)
    try {
        return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer)
    }
    finally {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer)
    }
}

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "PhotoMatch is not installed. Run Setup-Owner-Pilot.cmd first."
}
if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw "The private owner configuration is missing. Run Setup-Owner-Pilot.cmd first."
}
$requiredWeight = Join-Path $weightsPath "facenet512_weights.h5"
if (
    -not (Test-Path -LiteralPath $requiredWeight -PathType Leaf) -or
    (Get-Item -LiteralPath $requiredWeight).Length -le 0
) {
    throw "The offline face-model files are missing. Run Setup-Owner-Pilot.cmd while online."
}

$ownerConfig = Import-Clixml -LiteralPath $configPath
if ($ownerConfig.Version -ne 1) {
    throw "The owner configuration format is not supported. Run setup again."
}

$plainPassword = ConvertFrom-PhotoMatchSecureString $ownerConfig.Password
$plainSecret = ConvertFrom-PhotoMatchSecureString $ownerConfig.SecretKey

try {
    $env:PHOTOMATCH_ENV = "development"
    $env:PHOTOMATCH_ADMIN_USERNAME = [string]$ownerConfig.Username
    $env:PHOTOMATCH_ADMIN_PASSWORD = $plainPassword
    $env:PHOTOMATCH_SECRET_KEY = $plainSecret
    $env:PHOTOMATCH_SECURE_COOKIE = "0"
    $env:PHOTOMATCH_BACKGROUND_WORKER = "1"
    $env:PHOTOMATCH_COMPUTE_DEVICE = "auto"
    $env:PHOTOMATCH_TRUST_PROXY = "0"
    $env:PHOTOMATCH_ALLOWED_NETWORKS = "127.0.0.1/32,::1/128"
    $env:PHOTOMATCH_MODELS_DIR = $modelsPath

    Write-Host "Starting the private PhotoMatch owner pilot..."
    Write-Host "Open http://127.0.0.1:5000 in a browser on this PC."
    Write-Host "Press Ctrl+C in this window to stop PhotoMatch."
    Push-Location $projectRoot
    try {
        & $pythonPath -m src.web
        if ($LASTEXITCODE -ne 0) {
            throw "PhotoMatch stopped with exit code $LASTEXITCODE."
        }
    }
    finally {
        Pop-Location
    }
}
finally {
    $plainPassword = $null
    $plainSecret = $null
    foreach ($name in @(
        "PHOTOMATCH_ENV",
        "PHOTOMATCH_ADMIN_USERNAME",
        "PHOTOMATCH_ADMIN_PASSWORD",
        "PHOTOMATCH_SECRET_KEY",
        "PHOTOMATCH_SECURE_COOKIE",
        "PHOTOMATCH_BACKGROUND_WORKER",
        "PHOTOMATCH_COMPUTE_DEVICE",
        "PHOTOMATCH_TRUST_PROXY",
        "PHOTOMATCH_ALLOWED_NETWORKS",
        "PHOTOMATCH_MODELS_DIR"
    )) {
        Remove-Item -LiteralPath "Env:\$name" -ErrorAction SilentlyContinue
    }
}
