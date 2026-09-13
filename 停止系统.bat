@echo off
chcp 65001 >nul
echo 正在停止 8000 端口上的系统服务...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { Stop-Process -Id $_ -Force }"
echo 服务已停止，可以安全修改代码或数据库。
pause