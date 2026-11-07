@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_owner_pilot.ps1"
if errorlevel 1 (
  echo.
  echo Setup did not finish. Read the error above, then try again.
  pause
  exit /b 1
)
echo.
pause
