@echo off
setlocal
title rdna3-kawpow  -  KawPow on Vulkan (RDNA3)

REM ============================================================
REM  Just double-click this file to start mining.
REM  Edit the 5 values below for your pool / wallet / worker.
REM ============================================================
set "ALGO=kawpow"
set "POOL=stratum+tcp://pool.gaelium.io:3638"
set "WALLET=GQqbuovw1Zziaz2Q6EEsMuw2cHwZzofM6u"
set "WORKER=rx7900xt"
set "PASS=c=GAEL,mc=GAEL"
REM ============================================================

REM Run from this script's own folder (where the rdna3_kawpow package lives).
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found on PATH. Install Python 3.9+ and try again.
    pause
    exit /b 1
)

REM First run: install the Python dependencies if the Vulkan binding is missing.
python -c "import vulkan" >nul 2>nul
if errorlevel 1 (
    echo Installing Python dependencies ^(first run only^)...
    python -m pip install -r requirements.txt
)

:mine
echo.
echo ============================================================
echo  Mining %ALGO%  -^>  %POOL%
echo  Wallet : %WALLET%
echo  Worker : %WORKER%
echo  Close this window to stop.
echo ============================================================
python -m rdna3_kawpow -a %ALGO% -o "%POOL%" -u "%WALLET%.%WORKER%" -p "%PASS%"

echo.
echo Miner exited (code %errorlevel%). Auto-restarting in 5 seconds...
echo Press Ctrl+C to stop instead.
timeout /t 5 /nobreak >nul
goto mine
