@echo off
echo Universal Game Audio Radar - Setup
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found.
    echo Install Python 3.11 or newer from https://python.org
    echo Make sure to check "Add Python to PATH" during install.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    python -m venv .venv
)

echo Installing dependencies (first run takes a few minutes)...
.venv\Scripts\python.exe -m pip install -e .

echo.
echo Setup complete. Double-click run.bat to launch the radar.
pause
