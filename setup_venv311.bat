@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    echo Python Launcher was not found. Install Python 3.11 and enable the py launcher.
    exit /b 1
)

if not exist ".venv311\Scripts\python.exe" (
    py -3.11 -m venv .venv311
    if errorlevel 1 exit /b 1
)

".venv311\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)"
if errorlevel 1 (
    echo .venv311 exists but does not use Python 3.11. Remove it manually and rerun this script.
    exit /b 1
)

".venv311\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 exit /b 1

".venv311\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo Environment is ready. Activate it with .venv311\Scripts\activate.bat
