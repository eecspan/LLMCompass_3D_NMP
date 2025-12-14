@echo off
setlocal

REM One-click runner for BatchedMatmul in none and booksim NoC modes.
REM You can override BS/M/N/K via arguments: run_matmul_hbm_tests.bat [BS] [M] [N] [K]
REM Example: run_matmul_hbm_tests.bat 8 4096 4096 8192

REM Default values
set BS=8
set M=4096
set N=4096
set K=4096

if not "%~1"=="" set BS=%~1
if not "%~2"=="" set M=%~2
if not "%~3"=="" set N=%~3
if not "%~4"=="" set K=%~4

REM Ensure we run from repo root (script lives in change\test)
pushd "%~dp0\..\.."

@REM echo Running none-mode NoC simulation...
@REM python -m change.test.run_matmul_hbm --bs %BS% --M %M% --N %N% --K %K% --mode 3D_stacked --noc-model none
@REM if errorlevel 1 goto :fail

echo.
echo Running estimate-mode NoC simulation...
python -m change.test.run_matmul_hbm --bs %BS% --M %M% --N %N% --K %K% --mode 3D_stacked --noc-model estimate
if errorlevel 1 goto :fail

echo.
echo All runs completed.
goto :eof

:fail
echo.
echo Script aborted due to error.
exit /b 1
