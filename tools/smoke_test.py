# -*- coding: utf-8 -*-
"""一键冒烟测试：health / 登录 / 标准库 / 训练记录 / 实时 WebSocket 二进制链路。

用法：
    python tools/smoke_test.py
退出码 0 表示全部通过，1 表示有失败项。
"""
import asyncio
import json
import ssl
import sqlite3
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import websockets

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://127.0.0.1:8000"
CTX = ssl.create_default_context()
CTX.check_hostname = False
CTX.verify_mode = ssl.CERT_NONE
TEST_STUDENT = "冒烟测试"


def call(path, data=None, token=None):
    body = urllib.parse.urlencode(data).encode() if data is not None else None
    headers = {}
    if data is not None:
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(BASE + path, data=body, headers=headers,
                                 method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, context=CTX, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as exc:
        return exc.code, None


def sample_frame():
    video = ROOT / "onepeople.mp4"
    if video.exists():
        cap = cv2.VideoCapture(str(video))
        ok, frame = cap.read()
        cap.release()
        if ok:
            return cv2.resize(frame, (640, 480))
    return np.zeros((480, 640, 3), dtype=np.uint8)


async def ws_check(token):
    ok, jpg = cv2.imencode(".jpg", sample_frame(), [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    payload = jpg.tobytes()
    async with websockets.connect("wss://127.0.0.1:8000/ws/camera?token=" + token,
                                  ssl=CTX, max_size=None) as ws:
        await ws.send(json.dumps({"action_type": "dig", "student": TEST_STUDENT}))
        first = json.loads(await asyncio.wait_for(ws.recv(), timeout=30))
        if first.get("type") != "started":
            return False, f"未收到 started：{first}"
        await ws.send(payload)
        meta = images = 0
        for _ in range(2):
            msg = await asyncio.wait_for(ws.recv(), timeout=60)
            if isinstance(msg, (bytes, bytearray)):
                images += 1
            elif json.loads(msg).get("type") == "frame":
                meta += 1
        await ws.send(json.dumps({"stop": True}))
        while True:
            msg = await asyncio.wait_for(ws.recv(), timeout=60)
            if isinstance(msg, (bytes, bytearray)):
                continue
            data = json.loads(msg)
            if data.get("type") == "summary":
                summary = data.get("summary") or {}
                detail = (f"meta={meta} images={images} frames={summary.get('total_frames')} "
                          f"dropped={summary.get('dropped_frames')}")
                return (meta >= 1 and images >= 1), detail


def cleanup():
    db = ROOT / "data" / "volleyball.db"
    if not db.exists():
        return
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT id, out_path FROM training_sessions WHERE student=?", (TEST_STUDENT,)).fetchall()
    for row in rows:
        if row["out_path"]:
            path = Path(row["out_path"])
            if path.exists():
                path.unlink()
        conn.execute("DELETE FROM session_attempts WHERE session_id=?", (row["id"],))
        conn.execute("DELETE FROM training_sessions WHERE id=?", (row["id"],))
    conn.commit()
    conn.close()


def main():
    results = []
    status, health = call("/api/health")
    results.append(("健康检查", status == 200, f"status={status} {health}"))

    status, login = call("/api/login", {"username": "coach", "password": "coach123"})
    if status != 200:
        results.append(("教练登录", False, f"status={status}（维护模式下会失败，请先退出维护模式）"))
        token = None
    else:
        token = login["token"]
        results.append(("教练登录", True, f"user={login['user']['username']}"))

    if token:
        status, standards = call("/api/standards", token=token)
        results.append(("标准动作库", status == 200, f"status={status} count={len(standards or [])}"))
        status, sessions = call("/api/sessions", token=token)
        results.append(("训练记录", status == 200, f"status={status} count={len(sessions or [])}"))
        ok, detail = asyncio.run(ws_check(token))
        results.append(("实时 WebSocket（二进制）", ok, detail))

    cleanup()
    print("=" * 56)
    for name, ok, detail in results:
        print(("PASS  " if ok else "FAIL  ") + name + "  " + str(detail))
    print("=" * 56)
    failed = [r for r in results if not r[1]]
    print(f"结果：{len(results) - len(failed)}/{len(results)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())