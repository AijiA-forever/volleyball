@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHON=C:\miniconda\envs\volleyball_analytics\python.exe
if not exist "%PYTHON%" set PYTHON=python
echo ============================================
echo  排球训练智能分析系统 - 局域网访问模式
echo  端口：8000
echo  本机 IPv4 地址（其他电脑用这个地址访问）：
ipconfig | findstr /i "IPv4"
echo ============================================
echo  如果其他电脑无法访问，请用管理员 PowerShell 执行：
echo  Set-NetFirewallRule -DisplayName "排球训练智能分析系统 8000" -Profile Private,Public
echo ============================================
"%PYTHON%" -m uvicorn main:app --host 0.0.0.0 --port 8000
pause