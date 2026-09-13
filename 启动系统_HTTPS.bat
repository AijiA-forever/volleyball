@echo off
chcp 65001 >nul
cd /d "%~dp0"
set PYTHON=C:\miniconda\envs\volleyball_analytics\python.exe
if not exist "%PYTHON%" set PYTHON=python
if not exist "certs\server.crt" (
  echo 未检测到证书，正在生成...
  "%PYTHON%" make_cert.py
)
echo ============================================
echo  HTTPS 模式启动（浏览器允许摄像头）
echo  本机访问：https://127.0.0.1:8000
echo  局域网访问：https://本机IP:8000
echo  首次访问需信任 certs\ca.crt（见说明文档）
echo ============================================
ipconfig | findstr /i "IPv4"
"%PYTHON%" -m uvicorn main:app --host 0.0.0.0 --port 8000 --ssl-keyfile certs\server.key --ssl-certfile certs\server.crt
pause