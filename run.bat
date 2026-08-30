@echo off
title CarnetLM
cd /d "%~dp0"

:: Add common Python and UV paths to PATH if missing
set "PATH=%LOCALAPPDATA%\Programs\Python\Python313\Scripts;%LOCALAPPDATA%\Programs\Python\Python313;%USERPROFILE%\.cargo\bin;%LOCALAPPDATA%\bin;%PATH%"

echo.
echo  ========================================
echo             CarnetLM Server
echo  ========================================
echo.

:: Check if uv is available
where uv >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] 'uv' is not installed or not in PATH.
    echo Install it from: https://docs.astral.sh/uv/
    echo.
    pause
    exit /b 1
)

:: Kill any existing server on port 8000
for /f "tokens=5" %%a in ('netstat -aon ^| findstr /r ":8000.*LISTENING" 2^>nul') do (
    if not "%%a"=="0" taskkill /PID %%a /F >nul 2>&1
)

:: Create .env if missing
if not exist ".env" (
    if exist ".env.example" (
        echo Creating .env from .env.example ...
        copy ".env.example" ".env" >nul
    )
)

:: Sync/Install dependencies if .venv is missing or requested
if not exist ".venv" (
    echo Setting up environment and dependencies...
    uv sync
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] 'uv sync' failed. Check pyproject.toml for issues.
        pause
        exit /b 1
    )
    echo.
)

echo Starting CarnetLM on http://localhost:8000
echo Waiting for server to become ready...
echo Press Ctrl+C to stop.
echo.

:: Automatically open browser only AFTER server is fully ready and responding
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "$url='http://127.0.0.1:8000/api/health'; for ($i=0; $i -lt 120; $i++) { try { $res = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 1; if ($res.StatusCode -eq 200) { Start-Process 'http://localhost:8000'; break } } catch { Start-Sleep -Milliseconds 250 } }"

:: Start the server in the foreground
uv run uvicorn backend.main:app --host 0.0.0.0 --port 8000

echo.
echo [CarnetLM stopped]
echo.
pause

