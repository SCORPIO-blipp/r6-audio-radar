@echo off
if not exist ".venv\Scripts\pythonw.exe" (
    echo Setup not complete. Run install.bat first.
    pause
    exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -m universal_game_audio_radar.gui
