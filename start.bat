@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
title PARK-PILOT Demo

set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"

where uv >nul 2>nul
if errorlevel 1 (
  echo [Setup] Installing uv package manager ^(one-time^)...
  powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
  set "PATH=%USERPROFILE%\.local\bin;%USERPROFILE%\.cargo\bin;%PATH%"
)

echo.
echo ============================================================
echo   PARK-PILOT Demo
echo   First run installs Python + packages (about 3-5 min).
echo   Please wait. The browser opens automatically when ready.
echo   URL:  http://localhost:8000/pilot
echo   To stop the server: just close this window.
echo ============================================================
echo.

REM Wait until the server answers, then open the browser automatically.
start "" powershell -WindowStyle Hidden -Command "$u='http://localhost:8000/pilot'; for($i=0;$i -lt 400;$i++){ try{ $r=Invoke-WebRequest -Uri $u -UseBasicParsing -TimeoutSec 3; if($r.StatusCode -eq 200){ Start-Process $u; break } }catch{}; Start-Sleep -Seconds 3 }"

uv run uvicorn demo.app:app --host 127.0.0.1 --port 8000

echo.
echo Server stopped.
pause
