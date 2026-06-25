@echo off
setlocal
title rdna3-kawpow (Rust)  -  KawPow on Vulkan (RDNA3)

REM ============================================================
REM  Rust build of the miner. Same pool config as mine.bat.
REM  Edit the 5 values below for your pool / wallet / worker.
REM ============================================================
set "ALGO=kawpow"
set "POOL=stratum+tcp://pool.gaelium.io:3638"
set "WALLET=GQqbuovw1Zziaz2Q6EEsMuw2cHwZzofM6u"
set "WORKER=rx7900xt"
set "PASS=c=GAEL,mc=GAEL"
set "API=127.0.0.1:4068"
REM ============================================================

cd /d "%~dp0"
set "EXE=rust\target\release\rdna3-kawpow.exe"

if not exist "%EXE%" (
    echo Building the Rust miner ^(first run only^)...
    cargo build --release --manifest-path rust\Cargo.toml || (echo build failed & pause & exit /b 1)
)

:mine
echo.
echo ============================================================
echo  Mining %ALGO%  -^>  %POOL%
echo  Wallet : %WALLET%
echo  Worker : %WORKER%
echo  Stats  : http://%API%
echo  Close this window to stop.
echo ============================================================
"%EXE%" -a %ALGO% -o "%POOL%" -u "%WALLET%.%WORKER%" -p "%PASS%" --api-bind %API%

echo.
echo Miner exited (code %errorlevel%). Auto-restarting in 5 seconds...
timeout /t 5 /nobreak >nul
goto mine
