@echo off
REM Start the VitaScan backend and frontend together (Windows).
REM All arguments are passed through: start.bat --prod, start.bat --setup
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if not errorlevel 1 (
  py -3 start.py %*
  exit /b %errorlevel%
)

where python >nul 2>nul
if not errorlevel 1 (
  python start.py %*
  exit /b %errorlevel%
)

echo error ^| Python not found. Install Python 3.12 (x64) and re-run start.bat 1>&2
exit /b 1
