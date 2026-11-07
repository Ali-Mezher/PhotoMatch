[CmdletBinding()]
param(
    [string]$OwnerUsername,
    [switch]$SkipDependencyInstall,
    [switch]$SkipModelPreparation
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($env:OS -ne "Windows_NT") {
    throw "The owner-pilot setup is intended for Windows."
}

$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$venvPath = Join-Path $projectRoot ".venv"
$pythonPath = Join-Path $venvPath "Scripts\python.exe"
$configPath = Join-Path $projectRoot ".photomatch-owner-config.xml"
$modelsPath = Join-Path $projectRoot "models"

function Assert-LastCommandSucceeded {
    param([string]$Action)
    if ($LASTEXITCODE -ne 0) {
        throw "$Action failed with exit code $LASTEXITCODE."
    }
}

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
    $pyLauncher = Get-Command "py.exe" -ErrorAction SilentlyContinue
    $pythonCommand = Get-Command "python.exe" -ErrorAction SilentlyContinue
    if ($null -ne $pyLauncher) {
        & $pyLauncher.Source -3 -m venv $venvPath
    }
    elseif ($null -ne $pythonCommand) {
        & $pythonCommand.Source -m venv $venvPath
    }
    else {
        throw "Python 3 is not installed. Install Python 3.11, then run setup again."
    }
    Assert-LastCommandSucceeded "Creating the private Python environment"
}

if (-not $SkipDependencyInstall) {
    Write-Host "Installing PhotoMatch packages. Internet is required for this setup step..."
    & $pythonPath -m pip install --upgrade pip
    Assert-LastCommandSucceeded "Updating pip"
    & $pythonPath -m pip install -r (Join-Path $projectRoot "requirements.txt")
    Assert-LastCommandSucceeded "Installing PhotoMatch packages"
}

if ([string]::IsNullOrWhiteSpace($OwnerUsername)) {
    $OwnerUsername = Read-Host "Choose the owner's username"
}
if ($OwnerUsername -notmatch '^[A-Za-z0-9._-]{3,64}$') {
    throw "The username must be 3-64 characters using letters, numbers, dot, underscore, or hyphen."
}

Write-Host "Choose a password with at least 12 characters. It will not be displayed."
$password = Read-Host "Owner password" -AsSecureString
$passwordConfirmation = Read-Host "Confirm owner password" -AsSecureString
$passwordText = ConvertFrom-PhotoMatchSecureString $password
$confirmationText = ConvertFrom-PhotoMatchSecureString $passwordConfirmation
try {
    if ($passwordText.Length -lt 12) {
        throw "The owner password must contain at least 12 characters."
    }
    if ($passwordText -cne $confirmationText) {
        throw "The two passwords did not match. Run setup again."
    }
}
finally {
    $passwordText = $null
    $confirmationText = $null
    $passwordConfirmation.Dispose()
}

$secretBytes = New-Object byte[] 48
$random = [Security.Cryptography.RandomNumberGenerator]::Create()
try {
    $random.GetBytes($secretBytes)
}
finally {
    $random.Dispose()
}
$secretText = [Convert]::ToBase64String($secretBytes)
$secret = ConvertTo-SecureString $secretText -AsPlainText -Force
$secretText = $null

$ownerConfig = [PSCustomObject]@{
    Version = 1
    Username = $OwnerUsername
    Password = $password
    SecretKey = $secret
    CreatedAt = (Get-Date).ToString("o")
}
$ownerConfig | Export-Clixml -Path $configPath -Force
Write-Host "Owner credentials were encrypted for this Windows account."

$env:PHOTOMATCH_MODELS_DIR = $modelsPath
if (-not $SkipModelPreparation) {
    Write-Host "Preparing face models. Missing weights may be downloaded now..."
    & $pythonPath (Join-Path $projectRoot "scripts\prepare_offline_models.py") --prepare
    Assert-LastCommandSucceeded "Preparing face models"

    Write-Host "Starting a fresh process with network sockets blocked..."
    & $pythonPath (Join-Path $projectRoot "scripts\prepare_offline_models.py") --verify-offline
    Assert-LastCommandSucceeded "Offline model verification"
}

Write-Host ""
Write-Host "Owner pilot setup is complete."
Write-Host "Disconnect the internet, then double-click Start-PhotoMatch.cmd."
Write-Host "The private site will be available only on this PC at http://127.0.0.1:5000"
