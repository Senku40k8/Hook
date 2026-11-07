@echo off
rem Lance le bot Hook (prepare Python et les dependances au premier lancement).
rem Utilisation : start.bat [options du bot], par ex. start.bat -v --duration 2h
rem PSModulePath vide : evite que PowerShell 5.1 herite des modules de PowerShell 7.
setlocal
set "PSModulePath="
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
exit /b %ERRORLEVEL%
