@echo off
setlocal
title rdna3-kawpow  -  benchmark
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found on PATH. Install Python 3.9+ and try again.
    pause
    exit /b 1
)
python -c "import vulkan" >nul 2>nul
if errorlevel 1 (
    echo Installing Python dependencies ^(first run only^)...
    python -m pip install -r requirements.txt
)

echo Measuring KawPow hashrate (no pool, watchdog-safe)...
python -m rdna3_kawpow --benchmark --bench-seconds 10
echo.
pause
