# -*- coding: utf-8 -*-
"""整合版分析服务：把旧半成品的视觉分析引擎与 SQLite 记录、标准动作库对接。"""
from __future__ import annotations
import re
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from analyzer import VolleyballActionAnalyzer, ActionSession
from storage import create_session, find_standard_by_action, get_standard, get_calibration

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"


def _safe_dir(name: str) -> str:
    name = (name or "unknown").strip() or "unknown"
    return re.sub(r'[^0-9A-Za-z_\u4e00-\u9fff-]', "_", name)


class AnalysisService:
    """进程内单例：模型只加载一次，分析请求串行执行，避免并发争用全局会话状态。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rt_lock = threading.Lock()
        self._rt: Optional[Dict[str, Any]] = None
        self._analyzer: Optional[VolleyballActionAnalyzer] = None

    @staticmethod
    def resolve_calibration(label):
        if not label:
            return None
        try:
            item = get_calibration(str(label))
        except Exception:
            return None
        if not item:
            return None
        return {"label": item.get("label"), "points": item.get("points") or {}}

    @staticmethod
    def fuse_attempts(attempts_a, attempts_b):
        """双视角融合：按动作序号对齐，融合分数与反馈。"""
        fused = []
        n = max(len(attempts_a or []), len(attempts_b or []))
        for i in range(n):
            a = (attempts_a or [])[i] if i < len(attempts_a or []) else None
            b = (attempts_b or [])[i] if i < len(attempts_b or []) else None
            if a and b:
                score = (float(a.get("avg_score", 0)) + float(b.get("avg_score", 0))) / 2.0
                feedback = list(dict.fromkeys((a.get("feedback") or []) + (b.get("feedback") or [])))
                frames = max(int(a.get("frames", 0)), int(b.get("frames", 0)))
                person = a.get("person_id") if a.get("person_id") is not None else b.get("person_id")
            else:
                item = a or b
                score = float(item.get("avg_score", 0))
                feedback = item.get("feedback") or []
                frames = int(item.get("frames", 0))
                person = item.get("person_id")
            fused.append({
                "seq": i + 1,
                "avg_score": round(score, 2),
                "is_standard": score >= 80,
                "frames": frames,
                "feedback": feedback,
                "person_id": person,
                "details": {},
            })
        return fused

    def _get_analyzer(self) -> VolleyballActionAnalyzer:
        if self._analyzer is None:
            self._analyzer = VolleyballActionAnalyzer(detect_volleyball=True)
        return self._analyzer

    @staticmethod
    def resolve_criteria(action_type: str, standard_id: Optional[int]) -> Optional[Dict[str, Any]]:
        std = None
        if standard_id:
            std = get_standard(int(standard_id))
        if std is None:
            std = find_standard_by_action(action_type)
        if not std:
            return None
        params = std.get("params") or {}
        if isinstance(params, dict) and "weights" in params:
            return params
        return None

    @staticmethod
    def _out_name(prefix: str, ext: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:10]}.{ext}"

    def _new_realtime_writer(self, frame, owner: str = "unknown"):
        folder = RESULTS_DIR / "realtime_videos" / _safe_dir(owner)
        folder.mkdir(parents=True, exist_ok=True)
        h, w = frame.shape[:2]
        out_w, out_h = w, h
        if w > 640:
            scale = 640.0 / w
            out_w = int(w * scale) // 2 * 2
            out_h = int(h * scale) // 2 * 2
        for codec in ("VP80", "VP90"):
            candidate = folder / self._out_name("realtime", "webm")
            writer = cv2.VideoWriter(str(candidate), cv2.VideoWriter_fourcc(*codec), 15.0, (out_w, out_h))
            if writer.isOpened():
                return writer, candidate, (out_w, out_h)
            writer.release()
        candidate = folder / self._out_name("realtime", "mp4")
        writer = cv2.VideoWriter(str(candidate), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (out_w, out_h))
        if writer.isOpened():
            return writer, candidate, (out_w, out_h)
        raise RuntimeError("无法创建实时分析视频文件")

    @staticmethod
    def _attempts_from_summary(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
        scores = summary.get("action_scores") or []
        feedbacks = summary.get("action_feedbacks") or []
        attempts = []
        for idx, item in enumerate(scores):
            attempts.append({
                "seq": idx + 1,
                "avg_score": float(item.get("score", 0) or 0),
                "is_standard": bool(item.get("is_standard", False)),
                "frames": int(item.get("frame_count", 0) or 0),
                "feedback": feedbacks[idx] if idx < len(feedbacks) else [],
                "details": {},
                "person_id": item.get("person_id"),
            })
        return attempts

    def start_realtime(self, student: str, action_type: str, standard_id: Optional[int] = None, calibration_label: Optional[str] = None) -> Dict[str, Any]:
        if self._lock.locked():
            raise RuntimeError("视频分析正在进行，请稍后再开始实时分析")
        with self._rt_lock:
            self._finalize_realtime_locked()
            analyzer = self._get_analyzer()
            analyzer.action_session = ActionSession(action_type)
            analyzer.tracker.reset()
            analyzer.frame_person_counts = []
            analyzer.primary_id_history = []
            self._rt = {
                "owner": student or "实时训练",
                "calibration": self.resolve_calibration(calibration_label),
                "calibration_label": calibration_label,
                "student": student or "实时训练",
                "action_type": action_type,
                "standard_id": standard_id,
                "criteria": self.resolve_criteria(action_type, standard_id),
                "writer": None,
                "path": None,
                "size": None,
                "frame_count": 0,
                "scores": [],
                "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            return {"ok": True, "student": self._rt["student"], "action_type": action_type}

    def handle_realtime_frame(self, frame, action_type: str, standard_id: Optional[int] = None) -> Dict[str, Any]:
        with self._rt_lock:
            rt = self._rt
            if rt is None:
                raise RuntimeError("实时分析尚未开始")
            analyzer = self._get_analyzer()
            if rt["action_type"] != action_type:
                rt["action_type"] = action_type
                rt["standard_id"] = standard_id
                rt["criteria"] = self.resolve_criteria(action_type, standard_id)
                analyzer.action_session = ActionSession(action_type)
            annotated, judgment, _balls, _kpts, session_info = analyzer.process_frame(
                frame, rt["action_type"], rt["criteria"], rt.get("calibration")
            )
            if rt["writer"] is None:
                rt["writer"], rt["path"], rt["size"] = self._new_realtime_writer(frame, rt["owner"])
            output = annotated
            if rt["size"] != (annotated.shape[1], annotated.shape[0]):
                output = cv2.resize(annotated, rt["size"], interpolation=cv2.INTER_AREA)
            rt["writer"].write(output)
            rt["frame_count"] += 1
            score = float(judgment.get("total_score", 0) or 0) if judgment else 0.0
            rt["scores"].append(score)
            summary = analyzer.action_session.get_summary()
            valid = [v for v in rt["scores"] if v > 0]
            action_scores = summary.get("action_scores") or []
            if action_scores:
                avg_score = float(summary.get("average_score") or 0)
                best_score = max(float(v.get("score", 0) or 0) for v in action_scores)
            elif valid:
                avg_score = sum(valid) / len(valid)
                best_score = max(valid)
            else:
                avg_score = 0.0
                best_score = 0.0
            return {
                "annotated": annotated,
                "judgment": judgment,
                "session_info": session_info,
                "live": {
                    "current_score": round(score, 1),
                    "action_count": int(summary.get("action_count") or 0),
                    "standard_count": int(summary.get("standard_count") or 0),
                    "average_score": round(avg_score, 2),
                    "best_score": round(best_score, 2),
                    "feedback": (judgment or {}).get("feedback", []),
                },
            }

    def _finalize_realtime_locked(self) -> Optional[Dict[str, Any]]:
        rt = self._rt
        if rt is None:
            return None
        if rt["writer"] is not None:
            rt["writer"].release()
        analyzer = self._get_analyzer()
        summary = analyzer.action_session.get_summary()
        attempts = self._attempts_from_summary(summary)
        scores = rt["scores"]
        valid = [v for v in scores if v > 0]
        action_scores = summary.get("action_scores") or []
        if action_scores:
            avg_score = float(summary.get("average_score") or 0)
            best_score = max(float(v.get("score", 0) or 0) for v in action_scores)
        elif valid:
            avg_score = sum(valid) / len(valid)
            best_score = max(valid)
        else:
            avg_score = 0.0
            best_score = 0.0
        video_url = None
        out_path = rt["path"]
        if out_path is not None:
            video_url = f"/results/realtime_videos/{_safe_dir(rt['owner'])}/{Path(out_path).name}"
        summary = dict(summary)
        summary.update(analyzer.tracking_summary())
        summary.update({
            "mode": "realtime",
            "started_at": rt["started_at"],
            "duration_frames": rt["frame_count"],
            "score_curve": [round(float(v), 1) for v in scores],
        })
        session_id = None
        if rt["frame_count"] > 0:
            session_id = create_session(
                student=rt["student"],
                action_type=rt["action_type"],
                standard_id=rt["standard_id"],
                src_path=None,
                out_path=str(out_path) if out_path is not None else None,
                total_frames=rt["frame_count"],
                action_count=int(summary.get("action_count") or 0),
                standard_count=int(summary.get("standard_count") or 0),
                avg_score=avg_score,
                best_score=best_score,
                summary=summary,
                attempts=attempts,
                owner_username=rt["owner"],
            )
        result = {
            "session_id": session_id,
            "video_url": video_url,
            "summary": {
                "total_frames": rt["frame_count"],
                "action_count": int(summary.get("action_count") or 0),
                "standard_count": int(summary.get("standard_count") or 0),
                "average_score": round(avg_score, 2),
                "best_score": round(best_score, 2),
                "score_curve": [round(float(v), 1) for v in scores],
                "max_person_count": summary.get("max_person_count", 1),
                "avg_person_count": summary.get("avg_person_count", 1),
                "multi_person_frames": summary.get("multi_person_frames", 0),
                "primary_ids": summary.get("primary_ids", []),
                "calibration_label": rt.get("calibration_label"),
                "feedback": attempts[-1]["feedback"] if attempts else [],
            },
            "attempts": attempts,
        }
        self._rt = None
        return result

    def stop_realtime(self) -> Optional[Dict[str, Any]]:
        with self._rt_lock:
            return self._finalize_realtime_locked()

    def analyze_video(self, input_path: Path, action_type: str,
                      standard_id: Optional[int], student: str,
                      calibration_label: Optional[str] = None, save: bool = True) -> Dict[str, Any]:
        if self._rt is not None:
            raise RuntimeError("实时分析正在进行，请先停止实时分析")
        with self._lock:
            analyzer = self._get_analyzer()
            analyzer.action_session = ActionSession(action_type)
            analyzer.tracker.reset()
            analyzer.frame_person_counts = []
            analyzer.primary_id_history = []
            criteria = self.resolve_criteria(action_type, standard_id)
            calibration = self.resolve_calibration(calibration_label)
            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                raise RuntimeError(f"无法打开视频: {input_path}")
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            RESULTS_DIR.joinpath("videos").mkdir(parents=True, exist_ok=True)
            writer = None
            out_path = None
            out_width, out_height = width, height
            if width > 960:
                scale = 960.0 / width
                out_width = int(width * scale) // 2 * 2
                out_height = int(height * scale) // 2 * 2
            for codec in ("VP80", "VP90", "mp4v"):
                ext = "webm" if codec != "mp4v" else "mp4"
                candidate = RESULTS_DIR / "videos" / self._out_name("analyzed", ext)
                w = cv2.VideoWriter(str(candidate), cv2.VideoWriter_fourcc(*codec), fps, (out_width, out_height))
                if w.isOpened():
                    writer = w
                    out_path = candidate
                    break
                w.release()
            if writer is None:
                raise RuntimeError("当前环境没有可用的视频编码器")
            out_name = out_path.name
            frame_count = 0
            frame_scores: List[float] = []
            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    annotated, judgment, _balls, _kpts, _info = analyzer.process_frame(
                        frame, action_type, criteria, calibration
                    )
                    frame_scores.append(float(judgment.get("total_score", 0) or 0) if judgment else 0.0)
                    writer.write(cv2.resize(annotated, (out_width, out_height), interpolation=cv2.INTER_AREA))
            finally:
                cap.release()
                writer.release()
            summary = analyzer.action_session.get_summary()
            summary.update(analyzer.tracking_summary())
            summary["calibration_label"] = calibration_label
            action_scores = summary.get("action_scores") or []
            action_feedbacks = summary.get("action_feedbacks") or []
            attempts: List[Dict[str, Any]] = []
            for idx, score_item in enumerate(action_scores):
                attempts.append({
                    "seq": idx + 1,
                    "avg_score": float(score_item.get("score", 0) or 0),
                    "is_standard": bool(score_item.get("is_standard", False)),
                    "frames": int(score_item.get("frame_count", 0) or 0),
                    "feedback": action_feedbacks[idx] if idx < len(action_feedbacks) else [],
                    "details": {},
                    "person_id": score_item.get("person_id"),
                })
            valid_frame_scores = [s for s in frame_scores if s and s > 0]
            if action_scores:
                avg_score = float(summary.get("average_score") or 0)
                best_score = max(float(s.get("score", 0) or 0) for s in action_scores)
            elif valid_frame_scores:
                avg_score = sum(valid_frame_scores) / len(valid_frame_scores)
                best_score = max(valid_frame_scores)
            else:
                avg_score = 0.0
                best_score = 0.0
            action_count = int(summary.get("action_count") or 0)
            standard_count = int(summary.get("standard_count") or 0)
            rel_out = f"results/videos/{out_name}"
            session_id = None
            if save:
                session_id = create_session(
                student=student,
                action_type=action_type,
                standard_id=standard_id,
                src_path=str(input_path),
                out_path=str(out_path),
                total_frames=frame_count,
                action_count=action_count,
                standard_count=standard_count,
                avg_score=avg_score,
                best_score=best_score,
                    summary=summary,
                    attempts=attempts,
                    owner_username=student,
                )
            return {
                "session_id": session_id,
                "kind": "video",
                "video_url": f"/{rel_out.replace(chr(92), '/')}",
                "output_path": str(out_path),
                "summary": {
                    "total_frames": frame_count,
                    "action_count": action_count,
                    "standard_count": standard_count,
                    "average_score": round(avg_score, 2),
                    "best_score": round(best_score, 2),
                    "dig_count": summary.get("dig_count", action_count),
                    "score_curve": [round(float(s), 1) for s in frame_scores],
                    "max_person_count": summary.get("max_person_count", 1),
                    "avg_person_count": summary.get("avg_person_count", 1),
                    "multi_person_frames": summary.get("multi_person_frames", 0),
                    "primary_ids": summary.get("primary_ids", []),
                    "calibration_label": calibration_label,
                    "feedback": attempts[-1]["feedback"] if attempts else [],
                },
                "attempts": attempts,
                "criteria": criteria,
            }

    def analyze_image(self, input_path: Path, action_type: str,
                      standard_id: Optional[int], student: str,
                      calibration_label: Optional[str] = None, save: bool = True) -> Dict[str, Any]:
        with self._lock:
            analyzer = self._get_analyzer()
            analyzer.action_session = ActionSession(action_type)
            analyzer.tracker.reset()
            analyzer.frame_person_counts = []
            analyzer.primary_id_history = []
            criteria = self.resolve_criteria(action_type, standard_id)
            calibration = self.resolve_calibration(calibration_label)
            img = cv2.imread(str(input_path))
            if img is None:
                raise RuntimeError("无法读取图片")
            annotated, judgment, _balls, _kpts, _info = analyzer.process_frame(img, action_type, criteria, calibration)
            RESULTS_DIR.joinpath("images").mkdir(parents=True, exist_ok=True)
            out_name = self._out_name("analyzed", "jpg")
            out_path = RESULTS_DIR / "images" / out_name
            cv2.imwrite(str(out_path), annotated)
            attempts: List[Dict[str, Any]] = []
            if judgment:
                attempts.append({
                    "seq": 1,
                    "avg_score": float(judgment.get("total_score", 0) or 0),
                    "is_standard": bool(judgment.get("is_correct", False)),
                    "frames": 1,
                    "feedback": judgment.get("feedback", []),
                    "details": judgment.get("score_details", {}),
                })
            if judgment:
                attempts[0]["person_id"] = judgment.get("person_id")
            score = attempts[0]["avg_score"] if attempts else 0.0
            is_std = attempts[0]["is_standard"] if attempts else False
            session_id = None
            if save:
                session_id = create_session(
                student=student,
                action_type=action_type,
                standard_id=standard_id,
                src_path=str(input_path),
                out_path=str(out_path),
                total_frames=1,
                action_count=1 if attempts else 0,
                standard_count=1 if is_std else 0,
                avg_score=score,
                best_score=score,
                summary={"image": True, "feedback": attempts[0]["feedback"] if attempts else []},
                attempts=attempts,
                owner_username=student,
            )
            rel_out = f"results/images/{out_name}"
            return {
                "session_id": session_id,
                "kind": "image",
                "image_url": f"/{rel_out}",
                "output_path": str(out_path),
                "summary": {
                    "total_frames": 1,
                    "action_count": 1 if attempts else 0,
                    "standard_count": 1 if is_std else 0,
                    "average_score": round(score, 2),
                    "judgment": judgment,
                    "max_person_count": (judgment or {}).get("person_count", 1),
                    "primary_ids": [(judgment or {}).get("person_id")] if (judgment or {}).get("person_id") is not None else [],
                    "calibration_label": calibration_label,
                    "feedback": attempts[0]["feedback"] if attempts else [],
                },
                "attempts": attempts,
                "criteria": criteria,
            }