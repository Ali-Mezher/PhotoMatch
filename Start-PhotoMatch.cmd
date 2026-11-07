@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run_owner_pilot.ps1"
if errorlevel 1 (
  echo.
  echo PhotoMatch could not start. Read the error above.
)
echo.
pause
