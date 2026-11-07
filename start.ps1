<#
  start.ps1 - Lance le bot Hook sur n'importe quel PC Windows apres un git clone.

  1. Trouve Python 3.10+ (ou l'installe avec winget s'il est absent).
  2. Cree l'environnement .venv et installe requirements.txt (a nouveau si le
     fichier change, ou si .venv est casse ou vient d'un autre PC).
  3. Se relance en administrateur si besoin : le jeu tourne en administrateur,
     et Windows bloque les touches envoyees par un programme qui ne l'est pas.
  4. Demarre hook.py avec les options passees (ex. -v --duration 2h).

  Utilisation (depuis le dossier du projet) :  .\start.bat [options du bot]
  Options du script : -NoAdmin (sans elevation), -SetupOnly (prepare puis quitte).

  Fichier volontairement en ASCII : Windows PowerShell 5.1 lit mal les accents
  d'un script sans BOM.
#>
param(
    [switch]$NoAdmin,
    [switch]$SetupOnly
)
$BotArgs = $args
$ErrorActionPreference = 'Continue'
Set-Location -LiteralPath $PSScriptRoot

function Write-Step($text) { Write-Host "==> $text" -ForegroundColor Cyan }
function Fail($text) {
    Write-Host "ERREUR : $text" -ForegroundColor Red
    exit 1
}

function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    return ([Security.Principal.WindowsPrincipal]$id).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Python 3.10+ utilisable (ignore l'alias "python" du Microsoft Store).
function Find-Python {
    $candidates = @()
    if (Get-Command py -ErrorAction SilentlyContinue) { $candidates += , @('py', '-3') }
    if (Get-Command python -ErrorAction SilentlyContinue) { $candidates += , @('python') }
    $guess = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312\python.exe'
    if (Test-Path $guess) { $candidates += , @($guess) }
    foreach ($c in $candidates) {
        $exe = $c[0]
        $rest = @($c | Select-Object -Skip 1)
        $out = & $exe @rest -c "import sys; print(sys.version_info >= (3, 10)); print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $out -and $out[0] -eq 'True') { return $out[1] }
    }
    return $null
}

# --- 1. Droits administrateur -----------------------------------------------
if (-not $NoAdmin -and -not $SetupOnly -and -not (Test-Admin)) {
    Write-Step "Relance en administrateur (le jeu tourne en administrateur)..."
    $argList = @('-NoProfile', '-NoExit', '-ExecutionPolicy', 'Bypass', '-File', "`"$PSCommandPath`"")
    foreach ($a in $BotArgs) { $argList += "`"$a`"" }
    try {
        Start-Process powershell -Verb RunAs -WorkingDirectory $PSScriptRoot -ArgumentList $argList
    } catch {
        Fail "droits administrateur refuses. Relancez et acceptez la demande de Windows."
    }
    exit 0
}

# --- 2. Python ----------------------------------------------------------------
Write-Step "Recherche de Python 3.10+..."
$python = Find-Python
if (-not $python) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Fail "Python 3.10+ introuvable et winget absent. Installez Python depuis https://www.python.org/downloads/ (cochez 'Add python.exe to PATH') puis relancez."
    }
    Write-Step "Python absent : installation de Python 3.12 avec winget..."
    winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
    $env:Path = [Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' + [Environment]::GetEnvironmentVariable('Path', 'User')
    $python = Find-Python
    if (-not $python) { Fail "l'installation de Python a echoue. Installez-le depuis https://www.python.org/downloads/ puis relancez." }
}
Write-Host "    $python"

# --- 3. Environnement virtuel ------------------------------------------------
$venvPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (Test-Path $venvPython) {
    & $venvPython -c "import sys" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Step ".venv inutilisable (copie depuis un autre PC ?) : recreation..."
        Remove-Item -Recurse -Force (Join-Path $PSScriptRoot '.venv')
    }
}
if (-not (Test-Path $venvPython)) {
    Write-Step "Creation de l'environnement .venv..."
    & $python -m venv (Join-Path $PSScriptRoot '.venv')
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $venvPython)) { Fail "creation de .venv impossible." }
}

# --- 4. Dependances ------------------------------------------------------------
$stamp = Join-Path $PSScriptRoot '.venv\requirements.sha256'
# Empreinte via .NET (Get-FileHash peut manquer quand PowerShell 5.1 est lance depuis pwsh 7).
$sha = [Security.Cryptography.SHA256]::Create()
$bytes = [IO.File]::ReadAllBytes((Join-Path $PSScriptRoot 'requirements.txt'))
$hash = [BitConverter]::ToString($sha.ComputeHash($bytes)).Replace('-', '')
$installed = (Test-Path $stamp) -and ((Get-Content $stamp -Raw).Trim() -eq $hash)
if ($installed) {
    & $venvPython -c "import cv2, mss, numpy, pyautogui" 2>$null
    $installed = ($LASTEXITCODE -eq 0)
}
if (-not $installed) {
    Write-Step "Installation des dependances (premiere fois : 1 a 2 minutes)..."
    & $venvPython -m pip install --disable-pip-version-check --quiet --upgrade pip
    & $venvPython -m pip install --disable-pip-version-check --quiet -r (Join-Path $PSScriptRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { Fail "installation des dependances impossible (connexion Internet ?)." }
    & $venvPython -c "import cv2, mss, numpy, pyautogui"
    if ($LASTEXITCODE -ne 0) { Fail "dependances installees mais non importables." }
    Set-Content -Path $stamp -Value $hash
}
Write-Host "    environnement pret."

if ($SetupOnly) { exit 0 }

# --- 5. Lancement --------------------------------------------------------------
Write-Step "Demarrage du bot (F8 : pause, Echap : arret)..."
& $venvPython (Join-Path $PSScriptRoot 'hook.py') @BotArgs
exit $LASTEXITCODE
