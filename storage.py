# -*- coding: utf-8 -*-
"""SQLite 数据层：标准动作库 / 训练记录 / 用户 / 视角标定。"""
from __future__ import annotations
import hashlib
import json
import re
import secrets
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "volleyball.db"
_lock = threading.Lock()
ACTION_LABELS = {"dig": "垫球", "serve": "发球", "set": "传球", "spike": "扣球"}
DEFAULT_CRITERIA: Dict[str, Dict[str, Any]] = {
    "dig": {
        "left_elbow": [160, 180],
        "right_elbow": [160, 180],
        "left_knee": [100, 150],
        "right_knee": [100, 150],
        "hip_height_ratio": [0.3, 0.7],
        "weights": {"elbow": 0.4, "knee": 0.4, "hip": 0.2},
    },
    "serve": {
        "right_elbow": [100, 160],
        "right_wrist_shoulder": [1.05, 2.0],
        "right_knee": [120, 170],
        "weights": {"elbow": 0.5, "wrist": 0.3, "knee": 0.2},
    },
    "set": {
        "left_elbow": [70, 140],
        "right_elbow": [70, 140],
        "left_wrist_shoulder": [0.7, 1.1],
        "right_wrist_shoulder": [0.7, 1.1],
        "weights": {"elbow": 0.6, "wrist": 0.4},
    },
    "spike": {
        "right_elbow": [130, 180],
        "right_knee": [70, 130],
        "left_knee": [120, 170],
        "hip_shoulder_ratio": [1.0, 1.4],
        "weights": {"elbow": 0.5, "knee": 0.4, "hip": 0.1},
    },
}
def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    path = db_path or DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn
def _safe_filename(name: str) -> str:
    name = Path(name).name
    name = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_.-]", "_", name)
    return name or "file"
def _ensure_columns(conn) -> None:
    """轻量迁移：为旧数据库补充账号与数据归属字段。"""
    wanted = {
        "users": [
            ("password_hash", "TEXT"),
            ("salt", "TEXT"),
            ("session_token", "TEXT"),
            ("token_created_at", "TEXT"),
            ("last_login", "TEXT"),
        ],
        "training_sessions": [
            ("owner_username", "TEXT"),
        ],
    }
    for table, cols in wanted.items():
        try:
            existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        except Exception:
            continue
        for col, typ in cols:
            if col not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {typ}")


def init_db(db_path: Optional[Path] = None) -> None:
    """建表 + 默认标准动作。启动时调用一次。"""
    with _lock:
        conn = _connect(db_path)
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    display_name TEXT,
                    role TEXT NOT NULL CHECK(role IN ('student','coach')),
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS standards (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    video_path TEXT,
                    cover_path TEXT,
                    params_json TEXT,
                    note TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS training_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student TEXT NOT NULL,
                    action_type TEXT NOT NULL,
                    standard_id INTEGER,
                    src_path TEXT,
                    out_path TEXT,
                    total_frames INTEGER DEFAULT 0,
                    action_count INTEGER DEFAULT 0,
                    standard_count INTEGER DEFAULT 0,
                    avg_score REAL DEFAULT 0,
                    best_score REAL DEFAULT 0,
                    summary_json TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS session_attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    seq INTEGER DEFAULT 0,
                    avg_score REAL DEFAULT 0,
                    is_standard INTEGER DEFAULT 0,
                    frames INTEGER DEFAULT 0,
                    feedback_json TEXT,
                    details_json TEXT
                );
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    task TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS dataset_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    path TEXT NOT NULL,
                    label_path TEXT,
                    kind TEXT DEFAULT 'image',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS training_jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    dataset_id INTEGER,
                    task TEXT NOT NULL,
                    model TEXT NOT NULL,
                    epochs INTEGER DEFAULT 100,
                    imgsz INTEGER DEFAULT 640,
                    batch INTEGER DEFAULT 8,
                    status TEXT DEFAULT 'pending',
                    log_path TEXT,
                    weight_path TEXT,
                    params_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS calibrations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    label TEXT NOT NULL UNIQUE,
                    frame_index INTEGER DEFAULT 0,
                    points_json TEXT,
                    updated_at TEXT NOT NULL
                );
            """)
            _ensure_columns(conn)
            conn.commit()
            conn.execute("UPDATE training_sessions SET owner_username=student WHERE owner_username IS NULL")
            conn.commit()
            cnt = conn.execute("SELECT COUNT(*) AS c FROM standards").fetchone()["c"]
            if cnt == 0:
                now = _now()
                for action_type, label in ACTION_LABELS.items():
                    conn.execute(
                        "INSERT INTO standards(name, action_type, description, params_json, note, created_at, updated_at) VALUES(?,?,?,?,?,?,?)",
                        (
                            f"{label}标准动作参考",
                            action_type,
                            "默认标准动作，教练可上传示范视频并微调参数。",
                            json.dumps(DEFAULT_CRITERIA[action_type], ensure_ascii=False),
                            "系统自动生成",
                            now,
                            now,
                        ),
                    )
                conn.commit()
        finally:
            conn.close()
def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), 120000).hex()


def create_user(username: str, password: str, display_name: str, role: str, db_path=None):
    """创建带密码的账号。返回 (ok, message, user)。"""
    username = (username or "").strip()
    if not username or not password:
        return False, "账号和密码不能为空", None
    if role not in ("student", "coach"):
        return False, "角色只能是 student 或 coach", None
    if len(password) < 4:
        return False, "密码至少 4 位", None
    salt = secrets.token_hex(16)
    pwd_hash = _hash_password(password, salt)
    with _lock:
        conn = _connect(db_path)
        try:
            exists = conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
            if exists:
                return False, "该账号已存在，请直接登录", None
            conn.execute(
                "INSERT INTO users(username, display_name, role, created_at, password_hash, salt) VALUES(?,?,?,?,?,?)",
                (username, (display_name or username).strip(), role, _now(), pwd_hash, salt),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
            return True, "注册成功", _public_user(dict(row))
        finally:
            conn.close()


def verify_user(username: str, password: str, db_path=None):
    """校验账号密码，成功返回公开用户信息，失败返回 None。"""
    with _lock:
        conn = _connect(db_path)
        try:
            row = conn.execute("SELECT * FROM users WHERE username=?", ((username or "").strip(),)).fetchone()
            if row is None or not row["password_hash"] or not row["salt"]:
                return None
            calc = _hash_password(password or "", row["salt"])
            if calc != row["password_hash"]:
                return None
            conn.execute("UPDATE users SET last_login=? WHERE id=?", (_now(), row["id"]))
            conn.commit()
            data = dict(row)
            data["last_login"] = _now()
            return _public_user(data)
        finally:
            conn.close()


def _public_user(row):
    return {
        "id": row["id"],
        "username": row["username"],
        "display_name": row.get("display_name") or row["username"],
        "role": row["role"],
        "created_at": row.get("created_at"),
        "last_login": row.get("last_login"),
    }


def set_session_token(user_id: int, token: str, db_path=None) -> None:
    """单点登录核心：新登录覆盖旧 token，旧设备随即失效。"""
    with _lock:
        conn = _connect(db_path)
        try:
            conn.execute("UPDATE users SET session_token=?, token_created_at=? WHERE id=?", (token, _now(), user_id))
            conn.commit()
        finally:
            conn.close()


def get_user_by_token(token: str, db_path=None):
    if not token:
        return None
    with _lock:
        conn = _connect(db_path)
        try:
            row = conn.execute("SELECT * FROM users WHERE session_token=?", (token,)).fetchone()
            return _public_user(dict(row)) if row else None
        finally:
            conn.close()


def get_user_by_username(username: str, db_path=None):
    with _lock:
        conn = _connect(db_path)
        try:
            row = conn.execute("SELECT * FROM users WHERE username=?", ((username or "").strip(),)).fetchone()
            return _public_user(dict(row)) if row else None
        finally:
            conn.close()


def clear_session_token(user_id: int, db_path=None) -> None:
    with _lock:
        conn = _connect(db_path)
        try:
            conn.execute("UPDATE users SET session_token=NULL WHERE id=?", (user_id,))
            conn.commit()
        finally:
            conn.close()


def has_password(username: str, db_path=None) -> bool:
    with _lock:
        conn = _connect(db_path)
        try:
            row = conn.execute("SELECT password_hash FROM users WHERE username=?", ((username or "").strip(),)).fetchone()
            return bool(row and row["password_hash"])
        finally:
            conn.close()


def change_password(username: str, old_password: str, new_password: str, db_path=None):
    """用户自助修改密码。"""
    if len(new_password or "") < 4:
        return False, "新密码至少 4 位"
    user = verify_user(username, old_password, db_path=db_path)
    if user is None:
        return False, "原密码不正确"
    salt = secrets.token_hex(16)
    with _lock:
        conn = _connect(db_path)
        try:
            conn.execute(
                "UPDATE users SET password_hash=?, salt=? WHERE username=?",
                (_hash_password(new_password, salt), salt, username),
            )
            conn.commit()
            return True, "密码已修改"
        finally:
            conn.close()


def reset_password(username: str, new_password: str = "123456", db_path=None):
    """教练重置学生密码。"""
    if len(new_password or "") < 4:
        return False, "新密码至少 4 位"
    with _lock:
        conn = _connect(db_path)
        try:
            row = conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
            if row is None:
                return False, "账号不存在"
            salt = secrets.token_hex(16)
            conn.execute(
                "UPDATE users SET password_hash=?, salt=?, session_token=NULL WHERE username=?",
                (_hash_password(new_password, salt), salt, username),
            )
            conn.commit()
            return True, "密码已重置"
        finally:
            conn.close()


def create_dataset(name: str, task: str, description: str = "", db_path=None) -> int:
    with _lock:
        conn = _connect(db_path)
        try:
            cur = conn.execute(
                "INSERT INTO datasets(name, task, description, created_at) VALUES(?,?,?,?)",
                (name.strip(), task, (description or "").strip(), _now()),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()


def list_datasets(db_path=None) -> List[Dict[str, Any]]:
    with _lock:
        conn = _connect(db_path)
        try:
            rows = conn.execute(
                "SELECT d.*, (SELECT COUNT(*) FROM dataset_items i WHERE i.dataset_id=d.id) AS item_count "
                "FROM datasets d ORDER BY d.id DESC"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def get_dataset(dataset_id: int, db_path=None):
    with _lock:
        conn = _connect(db_path)
        try:
            row = conn.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


def delete_dataset(dataset_id: int, db_path=None) -> bool:
    with _lock:
        conn = _connect(db_path)
        try:
            conn.execute("DELETE FROM dataset_items WHERE dataset_id=?", (dataset_id,))
            cur = conn.execute("DELETE FROM datasets WHERE id=?", (dataset_id,))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()


def add_dataset_item(dataset_id: int, filename: str, path: str, label_path=None, kind: str = "image", db_path=None) -> int:
    with _lock:
        conn = _connect(db_path)
        try:
            cur = conn.execute(
                "INSERT INTO dataset_items(dataset_id, filename, path, label_path, kind, created_at) VALUES(?,?,?,?,?,?)",
                (dataset_id, filename, path, label_path, kind, _now()),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()


def list_dataset_items(dataset_id: int, db_path=None) -> List[Dict[str, Any]]:
    with _lock:
        conn = _connect(db_path)
        try:
            rows = conn.execute(
                "SELECT * FROM dataset_items WHERE dataset_id=? ORDER BY id DESC", (dataset_id,)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def create_training_job(dataset_id, task: str, model: str, epochs: int, imgsz: int, batch: int,
                        params: Optional[Dict[str, Any]] = None, db_path=None) -> int:
    with _lock:
        conn = _connect(db_path)
        try:
            now = _now()
            cur = conn.execute(
                "INSERT INTO training_jobs(dataset_id, task, model, epochs, imgsz, batch, status, params_json, created_at, updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (dataset_id, task, model, int(epochs), int(imgsz), int(batch), "pending",
                 json.dumps(params or {}, ensure_ascii=False), now, now),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()


def update_training_job(job_id: int, status=None, log_path=None, weight_path=None, db_path=None) -> None:
    with _lock:
        conn = _connect(db_path)
        try:
            updates = ["updated_at=?"]
            args: List[Any] = [_now()]
            if status is not None:
                updates.append("status=?"); args.append(status)
            if log_path is not None:
                updates.append("log_path=?"); args.append(log_path)
            if weight_path is not None:
                updates.append("weight_path=?"); args.append(weight_path)
            args.append(job_id)
            conn.execute(f"UPDATE training_jobs SET {', '.join(updates)} WHERE id=?", args)
            conn.commit()
        finally:
            conn.close()


def list_training_jobs(limit: int = 50, db_path=None) -> List[Dict[str, Any]]:
    with _lock:
        conn = _connect(db_path)
        try:
            rows = conn.execute(
                "SELECT * FROM training_jobs ORDER BY id DESC LIMIT ?", (int(limit),)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


def get_training_job(job_id: int, db_path=None):
    with _lock:
        conn = _connect(db_path)
        try:
            row = conn.execute("SELECT * FROM training_jobs WHERE id=?", (job_id,)).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


def upsert_user(username: str, display_name: str, role: str, db_path: Optional[Path] = None) -> Dict[str, Any]:
    with _lock:
        conn = _connect(db_path)
        try:
            name = username.strip()
            conn.execute(
                "INSERT INTO users(username, display_name, role, created_at) VALUES(?,?,?,?) "
                "ON CONFLICT(username) DO UPDATE SET display_name=excluded.display_name, role=excluded.role",
                (name, (display_name or name).strip(), role, _now()),
            )
            conn.commit()
            return dict(conn.execute("SELECT * FROM users WHERE username=?", (name,)).fetchone())
        finally:
            conn.close()
def list_users(role: Optional[str] = None, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    with _lock:
        conn = _connect(db_path)
        try:
            if role:
                rows = conn.execute("SELECT * FROM users WHERE role=? ORDER BY username", (role,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM users ORDER BY username").fetchall()
            return [_public_user(dict(r)) for r in rows]
        finally:
            conn.close()
def list_standards(action_type: Optional[str] = None, search: str = "", db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    with _lock:
        conn = _connect(db_path)
        try:
            sql = "SELECT * FROM standards WHERE 1=1"
            args: List[Any] = []
            if action_type:
                sql += " AND action_type=?"
                args.append(action_type)
            if search:
                sql += " AND (name LIKE ? OR description LIKE ? OR note LIKE ?)"
                kw = f"%{search}%"
                args += [kw, kw, kw]
            sql += " ORDER BY action_type, updated_at DESC"
            data = [dict(r) for r in conn.execute(sql, args).fetchall()]
            for item in data:
                item["action_label"] = ACTION_LABELS.get(item["action_type"], item["action_type"])
                try:
                    item["params"] = json.loads(item["params_json"] or "{}")
                except Exception:
                    item["params"] = {}
            return data
        finally:
            conn.close()
def get_standard(standard_id: int, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    with _lock:
        conn = _connect(db_path)
        try:
            row = conn.execute("SELECT * FROM standards WHERE id=?", (standard_id,)).fetchone()
            if row is None:
                return None
            item = dict(row)
            try:
                item["params"] = json.loads(item["params_json"] or "{}")
            except Exception:
                item["params"] = {}
            return item
        finally:
            conn.close()
def create_standard(name, action_type, description="", params=None, note="", video_path=None, cover_path=None, db_path=None) -> int:
    with _lock:
        conn = _connect(db_path)
        try:
            now = _now()
            cur = conn.execute(
                "INSERT INTO standards(name, action_type, description, video_path, cover_path, params_json, note, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (
                    name.strip(),
                    action_type,
                    (description or "").strip(),
                    video_path,
                    cover_path,
                    json.dumps(params or DEFAULT_CRITERIA.get(action_type, {}), ensure_ascii=False),
                    (note or "").strip(),
                    now,
                    now,
                ),
            )
            conn.commit()
            return int(cur.lastrowid)
        finally:
            conn.close()
def update_standard(standard_id: int, *, name=None, description=None, action_type=None, params=None, note=None, video_path=None, cover_path=None, db_path=None) -> bool:
    with _lock:
        conn = _connect(db_path)
        try:
            if conn.execute("SELECT id FROM standards WHERE id=?", (standard_id,)).fetchone() is None:
                return False
            updates, args = [], []
            if name is not None:
                updates.append("name=?"); args.append(name.strip())
            if description is not None:
                updates.append("description=?"); args.append((description or "").strip())
            if action_type is not None:
                updates.append("action_type=?"); args.append(action_type)
            if params is not None:
                updates.append("params_json=?"); args.append(json.dumps(params, ensure_ascii=False))
            if note is not None:
                updates.append("note=?"); args.append((note or "").strip())
            if video_path is not None:
                updates.append("video_path=?"); args.append(video_path)
            if cover_path is not None:
                updates.append("cover_path=?"); args.append(cover_path)
            updates.append("updated_at=?"); args.append(_now())
            args.append(standard_id)
            conn.execute(f"UPDATE standards SET {', '.join(updates)} WHERE id=?", args)
            conn.commit()
            return True
        finally:
            conn.close()
def delete_standard(standard_id: int, db_path: Optional[Path] = None) -> bool:
    with _lock:
        conn = _connect(db_path)
        try:
            cur = conn.execute("DELETE FROM standards WHERE id=?", (standard_id,))
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()
def find_standard_by_action(action_type: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    with _lock:
        conn = _connect(db_path)
        try:
            row = conn.execute('SELECT * FROM standards WHERE action_type=? ORDER BY id LIMIT 1', (action_type,)).fetchone()
            if row is None:
                return None
            item = dict(row)
            try:
                item['params'] = json.loads(item['params_json'] or '{}')
            except Exception:
                item['params'] = {}
            return item
        finally:
            conn.close()

def create_session(*, student, action_type, standard_id=None, src_path=None, out_path=None, total_frames=0, action_count=0, standard_count=0, avg_score=0.0, best_score=0.0, summary=None, attempts=None, owner_username=None, db_path=None) -> int:
    with _lock:
        conn = _connect(db_path)
        try:
            cur = conn.execute(
                "INSERT INTO training_sessions(student, action_type, standard_id, src_path, out_path, total_frames, action_count, standard_count, avg_score, best_score, summary_json, created_at, owner_username) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    student.strip(),
                    action_type,
                    standard_id,
                    src_path,
                    out_path,
                    int(total_frames or 0),
                    int(action_count or 0),
                    int(standard_count or 0),
                    float(avg_score or 0),
                    float(best_score or 0),
                    json.dumps(summary or {}, ensure_ascii=False),
                    _now(),
                    owner_username or student,
                ),
            )
            session_id = int(cur.lastrowid)
            for i, att in enumerate(attempts or [], start=1):
                conn.execute(
                    "INSERT INTO session_attempts(session_id, seq, avg_score, is_standard, frames, feedback_json, details_json) VALUES(?,?,?,?,?,?,?)",
                    (
                        session_id,
                        att.get("seq", i),
                        float(att.get("avg_score", 0) or 0),
                        1 if att.get("is_standard") else 0,
                        int(att.get("frames", 0) or 0),
                        json.dumps(att.get("feedback", []), ensure_ascii=False),
                        json.dumps(att.get("details", {}), ensure_ascii=False),
                    ),
                )
            conn.commit()
            return session_id
        finally:
            conn.close()
def list_sessions(student: Optional[str] = None, action_type: Optional[str] = None, limit: int = 200, owner: Optional[str] = None, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    with _lock:
        conn = _connect(db_path)
        try:
            sql = "SELECT * FROM training_sessions WHERE 1=1"
            args: List[Any] = []
            if owner:
                sql += " AND owner_username=?"; args.append(owner)
            if student:
                sql += " AND student=?"; args.append(student)
            if action_type:
                sql += " AND action_type=?"; args.append(action_type)
            sql += " ORDER BY id DESC LIMIT ?"; args.append(int(limit))
            data = [dict(r) for r in conn.execute(sql, args).fetchall()]
            for item in data:
                try:
                    item["summary"] = json.loads(item["summary_json"] or "{}")
                except Exception:
                    item["summary"] = {}
            return data
        finally:
            conn.close()
def get_session(session_id: int, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    with _lock:
        conn = _connect(db_path)
        try:
            row = conn.execute("SELECT * FROM training_sessions WHERE id=?", (session_id,)).fetchone()
            if row is None:
                return None
            item = dict(row)
            try:
                item["summary"] = json.loads(item["summary_json"] or "{}")
            except Exception:
                item["summary"] = {}
            atts = conn.execute("SELECT * FROM session_attempts WHERE session_id=? ORDER BY seq", (session_id,)).fetchall()
            item["attempts"] = []
            for a in atts:
                d = dict(a)
                for key, default in (("feedback_json", "[]"), ("details_json", "{}")):
                    try:
                        d[key.replace("_json", "")] = json.loads(d.get(key) or default)
                    except Exception:
                        d[key.replace("_json", "")] = json.loads(default)
                    d.pop(key, None)
                item["attempts"].append(d)
            return item
        finally:
            conn.close()
def list_students(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    with _lock:
        conn = _connect(db_path)
        try:
            rows = conn.execute(
                "SELECT student AS name, COUNT(*) AS session_count, COALESCE(SUM(action_count),0) AS action_total, "
                "COALESCE(SUM(standard_count),0) AS standard_total, ROUND(COALESCE(AVG(avg_score),0),1) AS avg_score "
                "FROM training_sessions GROUP BY student ORDER BY name"
            ).fetchall()
            data = [dict(r) for r in rows]
            for item in data:
                total = int(item.get("action_total") or 0)
                std = int(item.get("standard_total") or 0)
                item["normative_rate"] = round(std / total * 100, 1) if total else 0.0
            return data
        finally:
            conn.close()
def report_overview(db_path: Optional[Path] = None) -> Dict[str, Any]:
    students = list_students(db_path=db_path)
    sessions = list_sessions(limit=100000, db_path=db_path)
    total_actions = sum(int(s["action_count"] or 0) for s in sessions)
    total_standard = sum(int(s["standard_count"] or 0) for s in sessions)
    common: Dict[str, int] = {}
    with _lock:
        conn = _connect(db_path)
        try:
            rows = conn.execute("SELECT feedback_json FROM session_attempts WHERE feedback_json IS NOT NULL").fetchall()
        finally:
            conn.close()
    for row in rows:
        try:
            for item in json.loads(row["feedback_json"]):
                common[str(item)] = common.get(str(item), 0) + 1
        except Exception:
            continue
    return {
        "student_count": len(students),
        "session_count": len(sessions),
        "action_total": total_actions,
        "standard_total": total_standard,
        "normative_rate": round(total_standard / total_actions * 100, 1) if total_actions else 0.0,
        "avg_score": round(sum(float(s["avg_score"] or 0) for s in sessions) / len(sessions), 1) if sessions else 0.0,
        "top_feedback": [{"text": k, "count": v} for k, v in sorted(common.items(), key=lambda x: -x[1])[:8]],
    }
def save_calibration(label: str, points: Dict[str, Any], frame_index: int = 0, db_path: Optional[Path] = None) -> None:
    with _lock:
        conn = _connect(db_path)
        try:
            conn.execute(
                "INSERT INTO calibrations(label, frame_index, points_json, updated_at) VALUES(?,?,?,?) "
                "ON CONFLICT(label) DO UPDATE SET frame_index=excluded.frame_index, points_json=excluded.points_json, updated_at=excluded.updated_at",
                (label, int(frame_index), json.dumps(points, ensure_ascii=False), _now()),
            )
            conn.commit()
        finally:
            conn.close()
def get_calibration(label: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    with _lock:
        conn = _connect(db_path)
        try:
            row = conn.execute("SELECT * FROM calibrations WHERE label=?", (label,)).fetchone()
            if row is None:
                return None
            item = dict(row)
            try:
                item["points"] = json.loads(item["points_json"] or "{}")
            except Exception:
                item["points"] = {}
            return item
        finally:
            conn.close()
def list_calibrations(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    with _lock:
        conn = _connect(db_path)
        try:
            rows = conn.execute("SELECT * FROM calibrations ORDER BY updated_at DESC").fetchall()
            data = [dict(r) for r in rows]
            for item in data:
                try:
                    item["points"] = json.loads(item["points_json"] or "{}")
                except Exception:
                    item["points"] = {}
            return data
        finally:
            conn.close()
