# -*- coding: utf-8 -*-
"""
排球训练智能分析系统（Web 整合版）后端

- 账号：注册 / 密码登录 / 单点登录（同账号只能一处在线）/ 角色权限
- 分析：图片 / 视频 / 摄像头实时
- 数据：标准动作库、训练记录、报表、视角标定（SQLite）
"""
from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
import threading
import os
import secrets
import uuid
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import storage as store
from analysis_service import AnalysisService

ROOT = Path(__file__).resolve().parent
INDEX_HTML = ROOT / "index.html"
RESULTS_DIR = ROOT / "results"
UPLOAD_DIR = ROOT / "data" / "uploads"
STANDARD_DIR = ROOT / "data" / "standards"
STANDARD_VIDEO_DIR = STANDARD_DIR / "videos"
STANDARD_COVER_DIR = STANDARD_DIR / "covers"
DATASET_DIR = ROOT / "data" / "datasets"
TRAIN_DIR = ROOT / "models" / "trained"

MAINTENANCE_FLAG = ROOT / "data" / "maintenance.lock"
MAINTENANCE_ALLOW = {
    "/api/health",
    "/api/login",
    "/api/me",
    "/api/logout",
    "/api/maintenance/status",
    "/api/maintenance/enable",
    "/api/maintenance/disable",
}


def in_maintenance() -> bool:
    return MAINTENANCE_FLAG.exists()


COACH_INVITE_CODE = os.environ.get("VB_COACH_CODE", "gdut2026")
DEFAULT_COACH_PASSWORD = os.environ.get("VB_COACH_PASSWORD", "coach123")
DEFAULT_STUDENT_PASSWORD = os.environ.get("VB_STUDENT_PASSWORD", "123456")

for folder in (
    RESULTS_DIR / "videos",
    RESULTS_DIR / "images",
    RESULTS_DIR / "realtime_videos",
    UPLOAD_DIR,
    STANDARD_VIDEO_DIR,
    STANDARD_COVER_DIR,
    ROOT / "data",
    DATASET_DIR,
    TRAIN_DIR,
):
    folder.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="排球训练智能分析系统（Web 整合版）")
@app.middleware("http")
async def maintenance_middleware(request, call_next):
    path = request.url.path
    if in_maintenance() and path.startswith("/api/") and path not in MAINTENANCE_ALLOW:
        return JSONResponse({"detail": "系统维护中，请稍后再试"}, status_code=503)
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store.init_db()
service = AnalysisService()


def _seed_accounts() -> None:
    """首次启动自动创建教练账号，并为已有训练记录的学生补建账号。"""
    if store.get_user_by_username("coach") is None:
        ok, _msg, _user = store.create_user("coach", DEFAULT_COACH_PASSWORD, "主教练", "coach")
        if ok:
            print(f"[账号] 已创建默认教练账号 coach / {DEFAULT_COACH_PASSWORD}，请尽快修改密码")
    names = set()
    for row in store.list_sessions(limit=100000):
        names.add(row.get("owner_username") or row.get("student"))
    for name in sorted(n for n in names if n):
        if store.get_user_by_username(name) is None:
            ok, _msg, user = store.create_user(name, DEFAULT_STUDENT_PASSWORD, name, "student")
            if ok:
                print(f"[账号] 已为训练记录 {name} 创建学生账号，默认密码 {DEFAULT_STUDENT_PASSWORD}")
    # 兼容早期无密码账号：补默认密码
    for user in store.list_users():
        if not store.has_password(user["username"]):
            pwd = DEFAULT_COACH_PASSWORD if user.get("role") == "coach" else DEFAULT_STUDENT_PASSWORD
            store.reset_password(user["username"], pwd)
            print(f"[账号] 旧账号 {user['username']} 已补默认密码：{pwd}")


_seed_accounts()

app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")


@app.get("/media/{media_path:path}")
def serve_media(media_path: str):
    base = (ROOT / "data").resolve()
    target = (base / media_path).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=404, detail="文件不存在")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(str(target))


# ----------------------------- 鉴权 -----------------------------


def _bearer_token(authorization: Optional[str]) -> Optional[str]:
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


def current_user(authorization: Optional[str] = Header(None)):
    user = store.get_user_by_token(_bearer_token(authorization))
    if user is None:
        raise HTTPException(status_code=401, detail="未登录或账号已在其他设备登录，请重新登录")
    return user


def require_coach(user=Depends(current_user)):
    if user.get("role") != "coach":
        raise HTTPException(status_code=403, detail="该操作需要教练权限")
    return user


@app.post("/api/register")
def register(
    username: str = Form(...),
    password: str = Form(...),
    display_name: str = Form(""),
    role: str = Form("student"),
    coach_code: str = Form(""),
):
    if role == "coach" and coach_code != COACH_INVITE_CODE:
        raise HTTPException(status_code=403, detail="教练注册码不正确")
    ok, message, user = store.create_user(username, password, display_name, role)
    if not ok:
        raise HTTPException(status_code=400, detail=message)
    return {"ok": True, "user": user}


@app.post("/api/login")
def login(username: str = Form(...), password: str = Form(...)):
    user = store.verify_user(username, password)
    if user is None:
        raise HTTPException(status_code=401, detail="账号或密码错误")
    token = secrets.token_urlsafe(32)
    store.set_session_token(user["id"], token)
    return {"token": token, "user": user}


@app.get("/api/me")
def me(user=Depends(current_user)):
    return user


@app.post("/api/logout")
def logout(user=Depends(current_user)):
    store.clear_session_token(user["id"])
    return {"ok": True}


@app.post("/api/change_password")
def change_password(old_password: str = Form(...), new_password: str = Form(...), user=Depends(current_user)):
    ok, message = store.change_password(user["username"], old_password, new_password)
    if not ok:
        raise HTTPException(status_code=400, detail=message)
    return {"ok": True}


@app.post("/api/users/{username}/reset_password")
def reset_password(username: str, new_password: str = Form("123456"), user=Depends(require_coach)):
    ok, message = store.reset_password(username, new_password)
    if not ok:
        raise HTTPException(status_code=400, detail=message)
    return {"ok": True, "username": username}


@app.get("/api/users")
def users(role: Optional[str] = None, user=Depends(require_coach)):
    return store.list_users(role=role)


@app.get("/api/maintenance/status")
def maintenance_status(user=Depends(current_user)):
    return {"maintenance": in_maintenance()}


@app.post("/api/maintenance/enable")
def maintenance_enable(user=Depends(require_coach)):
    MAINTENANCE_FLAG.parent.mkdir(parents=True, exist_ok=True)
    MAINTENANCE_FLAG.write_text("maintenance", encoding="utf-8")
    return {"maintenance": True, "message": "已进入维护模式，其他用户的分析与写入将被拒绝"}


@app.post("/api/maintenance/disable")
def maintenance_disable(user=Depends(require_coach)):
    if MAINTENANCE_FLAG.exists():
        MAINTENANCE_FLAG.unlink()
    return {"maintenance": False, "message": "已退出维护模式"}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "standard_count": len(store.list_standards()),
        "student_count": len(store.list_students()),
        "coach_ready": store.get_user_by_username("coach") is not None,
    }


@app.get("/")
def home() -> FileResponse:
    return FileResponse(str(INDEX_HTML))


@app.get("/dataset_features.js")
def dataset_js():
    return FileResponse(str(ROOT / "dataset_features.js"))


@app.get("/auth_features.js")
def auth_js():
    return FileResponse(str(ROOT / "auth_features.js"))


@app.get("/extra_features.js")
def extra_js():
    return FileResponse(str(ROOT / "extra_features.js"))


# ----------------------------- 标准动作库 -----------------------------


def _save_upload(file: Optional[UploadFile], folder: Path) -> Optional[str]:
    if file is None or not file.filename:
        return None
    ext = Path(file.filename).suffix.lower()
    if ext not in (".mp4", ".mov", ".avi", ".mkv", ".jpg", ".jpeg", ".png", ".webp"):
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")
    name = f"{uuid.uuid4().hex}{ext}"
    target = folder / name
    with target.open("wb") as fh:
        fh.write(file.file.read())
    return name


@app.get("/api/standards")
def list_standards(action_type: Optional[str] = None, search: str = "", user=Depends(current_user)):
    return store.list_standards(action_type=action_type or None, search=search or "")


@app.get("/api/standards/{standard_id}")
def get_standard(standard_id: int, user=Depends(current_user)):
    item = store.get_standard(standard_id)
    if item is None:
        raise HTTPException(status_code=404, detail="标准动作不存在")
    return item


@app.post("/api/standards", status_code=201)
def create_standard(
    name: str = Form(...),
    action_type: str = Form(...),
    description: str = Form(""),
    params_json: str = Form("{}"),
    note: str = Form(""),
    video: Optional[UploadFile] = File(None),
    cover: Optional[UploadFile] = File(None),
    user=Depends(require_coach),
):
    if action_type not in store.ACTION_LABELS:
        raise HTTPException(status_code=400, detail="不支持的动作类型")
    try:
        params = json.loads(params_json or "{}")
    except Exception:
        raise HTTPException(status_code=400, detail="params_json 不是合法 JSON")
    video_rel = None
    cover_rel = None
    if video:
        fn = _save_upload(video, STANDARD_VIDEO_DIR)
        if fn:
            video_rel = f"standards/videos/{fn}"
    if cover:
        fn = _save_upload(cover, STANDARD_COVER_DIR)
        if fn:
            cover_rel = f"standards/covers/{fn}"
    std_id = store.create_standard(
        name=name, action_type=action_type, description=description,
        params=params, note=note, video_path=video_rel, cover_path=cover_rel,
    )
    return store.get_standard(std_id)


@app.put("/api/standards/{standard_id}")
def update_standard(
    standard_id: int,
    name: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    action_type: Optional[str] = Form(None),
    params_json: Optional[str] = Form(None),
    note: Optional[str] = Form(None),
    video: Optional[UploadFile] = File(None),
    cover: Optional[UploadFile] = File(None),
    user=Depends(require_coach),
):
    item = store.get_standard(standard_id)
    if item is None:
        raise HTTPException(status_code=404, detail="标准动作不存在")
    params = None
    if params_json is not None:
        try:
            params = json.loads(params_json)
        except Exception:
            raise HTTPException(status_code=400, detail="params_json 不是合法 JSON")
    video_rel = None
    cover_rel = None
    if video is not None:
        fn = _save_upload(video, STANDARD_VIDEO_DIR)
        if fn:
            video_rel = f"standards/videos/{fn}"
    if cover is not None:
        fn = _save_upload(cover, STANDARD_COVER_DIR)
        if fn:
            cover_rel = f"standards/covers/{fn}"
    store.update_standard(
        standard_id,
        name=name, description=description, action_type=action_type,
        params=params, note=note,
        video_path=video_rel if video_rel else item.get("video_path"),
        cover_path=cover_rel if cover_rel else item.get("cover_path"),
    )
    return store.get_standard(standard_id)


@app.delete("/api/standards/{standard_id}")
def delete_standard(standard_id: int, user=Depends(require_coach)):
    if not store.delete_standard(standard_id):
        raise HTTPException(status_code=404, detail="标准动作不存在")
    return {"ok": True}


# ----------------------------- 训练记录与报表 -----------------------------


@app.get("/api/sessions")
def sessions(student: Optional[str] = None, action_type: Optional[str] = None, limit: int = 200, user=Depends(current_user)):
    owner = user["username"] if user.get("role") == "student" else None
    return store.list_sessions(student=student or None, action_type=action_type or None, limit=limit, owner=owner)


@app.get("/api/sessions/{session_id}")
def session_detail(session_id: int, user=Depends(current_user)):
    item = store.get_session(session_id)
    if item is None:
        raise HTTPException(status_code=404, detail="训练记录不存在")
    if user.get("role") == "student" and item.get("owner_username") != user["username"]:
        raise HTTPException(status_code=403, detail="无权查看他人训练记录")
    return item


@app.get("/api/students")
def students(user=Depends(require_coach)):
    return store.list_students()


@app.get("/api/reports/overview")
def report_overview(user=Depends(require_coach)):
    return store.report_overview()


@app.get("/api/reports/student/{student}")
def report_student(student: str, user=Depends(require_coach)):
    rows = store.list_sessions(student=student, limit=100000)
    total = sum(int(r["action_count"] or 0) for r in rows)
    std = sum(int(r["standard_count"] or 0) for r in rows)
    return {
        "student": student,
        "session_count": len(rows),
        "action_total": total,
        "standard_total": std,
        "normative_rate": round(std / total * 100, 1) if total else 0.0,
        "avg_score": round(sum(float(r["avg_score"] or 0) for r in rows) / len(rows), 1) if rows else 0.0,
        "sessions": rows,
    }


# ----------------------------- 图片 / 视频分析 -----------------------------


def _save_input(file: UploadFile) -> Path:
    ext = Path(file.filename).suffix.lower()
    if ext not in (".mp4", ".mov", ".avi", ".mkv", ".jpg", ".jpeg", ".png", ".webp"):
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {ext}")
    name = f"{uuid.uuid4().hex}{ext}"
    target = UPLOAD_DIR / name
    with target.open("wb") as fh:
        fh.write(file.file.read())
    return target


@app.post("/api/analyze/video")
def analyze_video(
    file: UploadFile = File(...),
    student: str = Form(...),
    action_type: str = Form("dig"),
    standard_id: Optional[int] = Form(None),
    calibration_label: Optional[str] = Form(None),
    user=Depends(current_user),
):
    if action_type not in store.ACTION_LABELS:
        raise HTTPException(status_code=400, detail="不支持的动作类型")
    if user.get("role") == "student":
        student = user["username"]
    path = _save_input(file)
    try:
        return service.analyze_video(path, action_type, standard_id, student, calibration_label)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"视频分析失败: {exc}")


@app.post("/api/analyze/image")
def analyze_image(
    file: UploadFile = File(...),
    student: str = Form(...),
    action_type: str = Form("dig"),
    standard_id: Optional[int] = Form(None),
    calibration_label: Optional[str] = Form(None),
    user=Depends(current_user),
):
    if action_type not in store.ACTION_LABELS:
        raise HTTPException(status_code=400, detail="不支持的动作类型")
    if user.get("role") == "student":
        student = user["username"]
    path = _save_input(file)
    try:
        return service.analyze_image(path, action_type, standard_id, student, calibration_label)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"图片分析失败: {exc}")


@app.post("/api/analyze/dual")
def analyze_dual(
    file: UploadFile = File(...),
    file2: UploadFile = File(...),
    student: str = Form(...),
    action_type: str = Form("dig"),
    standard_id: Optional[int] = Form(None),
    calibration_a: Optional[str] = Form(None),
    calibration_b: Optional[str] = Form(None),
    user=Depends(current_user),
):
    """双视角融合：两路视频分别分析后按动作序号融合分数与反馈。"""
    if action_type not in store.ACTION_LABELS:
        raise HTTPException(status_code=400, detail="不支持的动作类型")
    if user.get("role") == "student":
        student = user["username"]
    path_a = _save_input(file)
    path_b = _save_input(file2)
    try:
        result_a = service.analyze_video(path_a, action_type, standard_id, student, calibration_a, save=False)
        result_b = service.analyze_video(path_b, action_type, standard_id, student, calibration_b, save=False)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"双视角分析失败: {exc}")
    fused = service.fuse_attempts(result_a.get("attempts"), result_b.get("attempts"))
    scores = [a["avg_score"] for a in fused]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    best_score = max(scores) if scores else 0.0
    standard_count = sum(1 for a in fused if a["is_standard"])
    summary = dict(result_a.get("summary") or {})
    summary.update({
        "mode": "dual_view_fusion",
        "view_a": result_a.get("summary"),
        "view_b": result_b.get("summary"),
        "calibration_a": calibration_a,
        "calibration_b": calibration_b,
    })
    session_id = store.create_session(
        student=student,
        action_type=action_type,
        standard_id=standard_id,
        src_path=f"{path_a}|{path_b}",
        out_path=result_a.get("output_path"),
        total_frames=int((result_a.get("summary") or {}).get("total_frames", 0)),
        action_count=len(fused),
        standard_count=standard_count,
        avg_score=avg_score,
        best_score=best_score,
        summary=summary,
        attempts=fused,
        owner_username=student,
    )
    return {
        "session_id": session_id,
        "kind": "dual",
        "video_url": result_a.get("video_url"),
        "video_url_b": result_b.get("video_url"),
        "summary": {
            "total_frames": (result_a.get("summary") or {}).get("total_frames", 0),
            "action_count": len(fused),
            "standard_count": standard_count,
            "average_score": round(avg_score, 2),
            "best_score": round(best_score, 2),
            "mode": "dual_view_fusion",
            "feedback": fused[-1]["feedback"] if fused else [],
        },
        "attempts": fused,
    }


# ----------------------------- 摄像头实时分析 -----------------------------


@app.websocket("/ws/camera")
async def websocket_camera(ws: WebSocket, token: str = ""):
    if in_maintenance():
        await ws.close(code=4403)
        return
    user = store.get_user_by_token(token)
    if user is None:
        await ws.close(code=4401)
        return
    await ws.accept()
    action_type = "dig"
    standard_id = None
    student = user["username"] if user.get("role") == "student" else "实时训练"
    started = False
    try:
        first = json.loads(await ws.receive_text())
        action_type = first.get("action_type", "dig")
        standard_id = first.get("standard_id")
        calibration_label = first.get("calibration_label")
        if user.get("role") != "student":
            student = first.get("student") or "实时训练"
        service.start_realtime(student, action_type, standard_id, calibration_label)
        started = True
        await ws.send_json({"type": "started", "student": student, "action_type": action_type})
        while True:
            message = await ws.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                # 二进制帧：收到的就是 JPEG 字节流，省掉 base64 解码
                payload = message["bytes"]
            else:
                data = json.loads(message.get("text") or "{}")
                if data.get("stop"):
                    break
                frame_b64 = data.get("frame")
                if not frame_b64:
                    continue
                payload = base64.b64decode(frame_b64)
            if not payload:
                continue
            buf = np.frombuffer(payload, np.uint8)
            frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            # 逐帧推理是 CPU 密集的同步调用，放线程池执行，避免阻塞事件循环
            result = await run_in_threadpool(service.handle_realtime_frame, frame, action_type, standard_id)
            annotated = result["annotated"]
            judgment = result["judgment"]
            ok, jpg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            await ws.send_json({
                "type": "frame",
                "frame": base64.b64encode(jpg).decode("utf-8"),
                "pose_judgment": judgment or {},
                "live": result["live"],
                "session_info": result["session_info"],
            })
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await ws.send_json({"type": "error", "error": str(exc)})
        except Exception:
            pass
    finally:
        if started:
            try:
                final = service.stop_realtime()
                if final is not None:
                    await ws.send_json({"type": "summary", **final})
            except Exception:
                pass


# ----------------------------- 数据集与模型训练 -----------------------------


def _run_training_job(job_id: int, dataset_dir: Path, data_yaml: Path, task: str,
                      model: str, epochs: int, imgsz: int, batch: int) -> None:
    log_path = dataset_dir / f"train_job_{job_id}.log"
    store.update_training_job(job_id, status="running", log_path=str(log_path))
    try:
        from ultralytics import YOLO
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"task={task} model={model} epochs={epochs} imgsz={imgsz} batch={batch}\n")
            log.flush()
            yolo = YOLO(model)
            results = yolo.train(
                data=str(data_yaml), epochs=int(epochs), imgsz=int(imgsz), batch=int(batch),
                project=str(TRAIN_DIR), name=f"job_{job_id}", exist_ok=True,
            )
            save_dir = Path(getattr(results, "save_dir", TRAIN_DIR / f"job_{job_id}"))
            best = save_dir / "weights" / "best.pt"
            if best.exists():
                target = TRAIN_DIR / f"job_{job_id}" / "best.pt"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(best), str(target))
                store.update_training_job(job_id, status="success", weight_path=str(target))
                log.write(f"\n训练完成，权重：{target}\n")
            else:
                store.update_training_job(job_id, status="finished_no_weight")
                log.write("\n训练结束，但未找到 best.pt\n")
    except Exception as exc:
        try:
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"\n训练失败：{exc}\n")
        except Exception:
            pass
        store.update_training_job(job_id, status=f"failed: {exc}")


@app.get("/api/datasets")
def datasets(user=Depends(require_coach)):
    return store.list_datasets()


@app.post("/api/datasets")
def create_dataset(name: str = Form(...), task: str = Form("detect"), description: str = Form(""), user=Depends(require_coach)):
    dataset_id = store.create_dataset(name, task, description)
    (DATASET_DIR / str(dataset_id) / "images").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / str(dataset_id) / "labels").mkdir(parents=True, exist_ok=True)
    return store.get_dataset(dataset_id)


@app.delete("/api/datasets/{dataset_id}")
def delete_dataset(dataset_id: int, user=Depends(require_coach)):
    if not store.delete_dataset(dataset_id):
        raise HTTPException(status_code=404, detail="数据集不存在")
    return {"ok": True}


@app.post("/api/datasets/{dataset_id}/upload")
def upload_dataset_files(dataset_id: int, files: List[UploadFile] = File(...), user=Depends(require_coach)):
    dataset = store.get_dataset(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="数据集不存在")
    base = DATASET_DIR / str(dataset_id)
    (base / "images").mkdir(parents=True, exist_ok=True)
    (base / "labels").mkdir(parents=True, exist_ok=True)
    saved = []
    for file in files:
        ext = Path(file.filename or "").suffix.lower()
        if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            name = f"{uuid.uuid4().hex}{ext}"
            target = base / "images" / name
            target.write_bytes(file.file.read())
            store.add_dataset_item(dataset_id, file.filename, str(target), None, "image")
            saved.append(name)
        elif ext == ".txt":
            name = Path(file.filename).name
            target = base / "labels" / name
            target.write_bytes(file.file.read())
            saved.append(name)
    return {"ok": True, "saved": saved, "items": len(store.list_dataset_items(dataset_id))}


@app.get("/api/datasets/{dataset_id}/items")
def dataset_items(dataset_id: int, user=Depends(require_coach)):
    return store.list_dataset_items(dataset_id)


@app.post("/api/datasets/{dataset_id}/train")
def start_training(
    dataset_id: int,
    task: str = Form("detect"),
    model: str = Form("yolov8n.pt"),
    epochs: int = Form(100),
    imgsz: int = Form(640),
    batch: int = Form(8),
    user=Depends(require_coach),
):
    dataset = store.get_dataset(dataset_id)
    if dataset is None:
        raise HTTPException(status_code=404, detail="数据集不存在")
    base = DATASET_DIR / str(dataset_id)
    data_yaml = base / "data.yaml"
    names_line = "names:\n  0: person\n"
    content = f"path: {base.as_posix()}\ntrain: images\nval: images\n{names_line}"
    if task == "pose":
        content += "kpt_shape: [17, 3]\n"
    data_yaml.write_text(content, encoding="utf-8")
    job_id = store.create_training_job(dataset_id, task, model, epochs, imgsz, batch, {"data_yaml": str(data_yaml)})
    thread = threading.Thread(
        target=_run_training_job,
        args=(job_id, base, data_yaml, task, model, epochs, imgsz, batch),
        daemon=True,
    )
    thread.start()
    return {"ok": True, "job_id": job_id, "status": "running"}


@app.get("/api/training_jobs")
def training_jobs(user=Depends(require_coach)):
    return store.list_training_jobs()


@app.get("/api/training_jobs/{job_id}")
def training_job(job_id: int, user=Depends(require_coach)):
    job = store.get_training_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    return job


@app.get("/api/training_jobs/{job_id}/log")
def training_log(job_id: int, user=Depends(require_coach)):
    job = store.get_training_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    log_path = job.get("log_path")
    text = ""
    if log_path and Path(log_path).exists():
        text = Path(log_path).read_text(encoding="utf-8", errors="ignore")[-20000:]
    return {"job_id": job_id, "status": job.get("status"), "log": text}


@app.post("/api/training_jobs/{job_id}/activate")
def activate_model(job_id: int, user=Depends(require_coach)):
    job = store.get_training_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="训练任务不存在")
    weight = job.get("weight_path")
    if not weight or not Path(weight).exists():
        raise HTTPException(status_code=400, detail="该任务没有可用权重")
    target_name = "pose_best.pt" if job.get("task") == "pose" else "best.pt"
    target = ROOT / target_name
    backup = ROOT / f"{target.stem}_before_{job_id}.pt"
    if target.exists() and not backup.exists():
        shutil.copy(str(target), str(backup))
    shutil.copy(weight, str(target))
    return {"ok": True, "target": target_name, "backup": backup.name}


# ----------------------------- 视角标定 -----------------------------


@app.post("/api/calibration")
def calibration_save(payload: dict, user=Depends(require_coach)):
    store.save_calibration(
        label=str(payload.get("label", "default")),
        points=payload.get("points", {}) or {},
        frame_index=int(payload.get("frame_index", 0) or 0),
    )
    return {"ok": True}


@app.get("/api/calibrations")
def calibrations(user=Depends(require_coach)):
    return store.list_calibrations()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
