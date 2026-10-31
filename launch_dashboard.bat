@echo off
setlocal
chcp 65001 >nul 2>&1
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
cd /d "%~dp0"

echo =====================================================================
echo   AUTONOMOUS THREAT DEFENSE CONSOLE - LAUNCHER
echo   Unified Multi-Agent Intrusion Detection System
echo =====================================================================
echo.

where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python was not found in PATH.
    echo Please install Python and ensure it is added to your PATH environment variable.
    echo.
    exit /b 1
)

echo [1/2] Verifying Python environment...
python -c "import streamlit" >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [NOTICE] Streamlit is not installed in the active environment.
    echo Installing dependencies from requirements.txt...
    python -m pip install -r requirements.txt
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to install required packages.
        echo.
        exit /b 1
    )
)

echo [2/2] Launching Streamlit dashboard...
echo The console will start on http://localhost:8501
echo Press Ctrl+C in this terminal window to stop the console.
echo.
python -m streamlit run app/streamlit_app.py %*

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Dashboard exited with error code %ERRORLEVEL%.
    exit /b %ERRORLEVEL%
)

endlocal
