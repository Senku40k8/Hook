@echo off
rem Mode frenzy : F sur l'invite "Fishing", "Start Fishing", puis clics au centre.
rem Utilisation : frenzy.bat [options], par ex. frenzy.bat -r 1920x1080
setlocal
set "PSModulePath="
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" --frenzy %*
exit /b %ERRORLEVEL%
