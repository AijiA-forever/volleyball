@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHON=C:\miniconda\envs\volleyball_analytics\python.exe
if not exist "%PYTHON%" set PYTHON=python
"%PYTHON%" -m uvicorn main:app --host 127.0.0.1 --port 8000
pause