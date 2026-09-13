# -*- coding: utf-8 -*-
"""一致性备份 SQLite 数据库（服务运行中也可安全执行）。"""
import sqlite3
from datetime import datetime
from pathlib import Path

root = Path(__file__).resolve().parent
src = root / "data" / "volleyball.db"
out_dir = root / "data" / "backups"
out_dir.mkdir(parents=True, exist_ok=True)
dst = out_dir / f"volleyball_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"

if not src.exists():
    raise SystemExit(f"数据库不存在: {src}")

src_conn = sqlite3.connect(str(src))
dst_conn = sqlite3.connect(str(dst))
try:
    src_conn.backup(dst_conn)
finally:
    dst_conn.close()
    src_conn.close()
print(f"备份完成: {dst}")