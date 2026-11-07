@echo off
rem Lance le bot Hook (prepare Python et les dependances au premier lancement).
rem Utilisation : start.bat [options du bot], par ex. start.bat -v --duration 2h
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
