# 排球动作分析器核心类
# 输入：视频帧 | 输出：标注帧 + 评分结果
# 功能：姿态检测、动作评分、排球追踪、动作状态管理

from body import (
    extract_valid_keypoints,
    extract_persons,
    draw_pose_results,
    draw_track_labels,
    tensor_to_numpy,
    calculate_angle,
)
from tracking import SimpleTracker
from metrics import compute_frame_metrics, summarize as summarize_biomechanics
from geometry import homography_from_points, correct_keypoints
from pose_judge import judge_pose
from volleyball_detect import (
    load_volleyball_model,
    detect_volleyball,
    draw_volleyball_results
)
from ultralytics import YOLO
from pathlib import Path
import os
import torch
import numpy as np


def _env_int(name, default):
    """读取整数环境变量，非法值回退默认值。"""
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name, default):
    """读取浮点环境变量，非法值回退默认值。"""
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# 实时性能相关配置：默认值与原行为一致（640 / 0.3 / 0.5），可用环境变量覆盖
POSE_IMGSZ = _env_int("VB_POSE_IMGSZ", 640)
POSE_CONF = _env_float("VB_POSE_CONF", 0.3)
BALL_IMGSZ = _env_int("VB_BALL_IMGSZ", 640)
BALL_CONF = _env_float("VB_BALL_CONF", 0.5)
# 每 N 帧执行一次排球检测。默认 1（每帧检测），保证动作计数准确。
# 实测：在 4 段不同素材上，N=2 会把动作数从 [3,5,3,4] 降为 [3,1,1,1]，
# 因为跳帧期间球轨迹靠外推，动作区间容易被提前截断。
# 机器性能不足时可用环境变量 VB_BALL_INTERVAL=2 换取速度，但要接受计数偏差。
BALL_INTERVAL = max(1, _env_int("VB_BALL_INTERVAL", 1))
DEBUG_LOG = os.environ.get("VB_DEBUG_LOG") == "1"        # 逐帧/区间调试打印开关


def _select_device():
    """默认优先使用 GPU：torch 有可用的 CUDA 时用 0 号卡，否则回退 CPU。

    可用 VB_DEVICE 显式指定（cpu / 0 / cuda:0），便于排障与对比测试。
    """
    override = os.environ.get("VB_DEVICE", "").strip()
    if override:
        lowered = override.lower()
        device = 0 if lowered == "gpu" else (int(override) if override.isdigit() else override)
    else:
        device = 0 if torch.cuda.is_available() else "cpu"
    if device != "cpu" and not torch.cuda.is_available():
        print("[设备] 指定的 GPU 不可用（当前 torch 没有可用的 CUDA），回退 CPU")
        return "cpu"
    return device


class VolleyballActionAnalyzer:
    """
    排球动作分析器构造函数
    :param detect_volleyball: 是否启用排球检测（默认True）
    """
    def __init__(self, detect_volleyball=True):
        # 一次性加载模型（UI里只加载一次，不重复加载）
        self.pose_model = YOLO(str(Path(__file__).resolve().parent / "pose_best.pt"))
        self.volleyball_model = None
        self.detect_volleyball_flag = detect_volleyball
        self.device = _select_device()
        self.half = bool(torch.cuda.is_available() and self.device != 'cpu')
        if self.device == 'cpu':
            print(f"[设备] CPU 推理（torch {torch.__version__}，未检测到可用的 CUDA 版 torch）", flush=True)
        else:
            print(f"[设备] GPU 推理：{torch.cuda.get_device_name(0)} "
                  f"（torch {torch.__version__}，CUDA {torch.version.cuda}，FP16={self.half}）", flush=True)
        if self.half:
            try:
                self.pose_model.model.half()
            except Exception as exc:
                print(f"[设备] FP16 设置失败，回退 FP32：{exc}", flush=True)
                self.half = False
        self.ball_frame_index = 0  # 排球检测抽帧计数
        self.ball_interval = BALL_INTERVAL  # 每 N 帧检测一次排球，可由 AnalysisService 按模式覆盖
        self.last_valid_kpts = None
        self.tracker = SimpleTracker()
        self.calibration_H = None
        self.calibration_label = None
        self.last_primary_id = None
        self.frame_person_counts = []
        self.primary_id_history = []

        # 排球追踪相关参数
        self.volleyball_history = []  # 排球历史坐标列表（最近10帧）
        self.volleyball_patience = 5  # 容忍丢帧数
        self.missed_frames_counter = 0  # 连续丢帧计数
        self.last_velocity = None  # 排球速度 (vx, vy)
        self.max_history = 10  # 最大历史记录数

        if detect_volleyball:
            try:
                self.volleyball_model = load_volleyball_model()
                print(f"Volleyball detection enabled: {self.detect_volleyball_flag}")
            except Exception as e:
                self.detect_volleyball_flag = False
                print(f"Volleyball detection disabled: {str(e)}")

        # 初始化动作会话（整合 ActionSession）
        self.action_session = ActionSession(action_type="dig")

    def _update_volleyball_history(self, center):
        """更新排球历史坐标列表"""
        if center is not None:
            self.volleyball_history.append({
                'center': center,
                'frame_idx': len(self.volleyball_history)
            })
            # 保持历史记录在最大长度内
            if len(self.volleyball_history) > self.max_history:
                self.volleyball_history.pop(0)

    def _calculate_velocity(self):
        """根据历史轨迹计算排球速度"""
        if len(self.volleyball_history) < 2:
            return None

        # 使用最近的两帧计算速度
        latest = self.volleyball_history[-1]['center']
        previous = self.volleyball_history[-2]['center']
        vx = latest[0] - previous[0]
        vy = latest[1] - previous[1]
        return (float(vx), float(vy))

    def _predict_volleyball_position(self):
        """基于速度预测排球位置"""
        if self.last_velocity is None or len(self.volleyball_history) == 0:
            return None

        latest = self.volleyball_history[-1]['center']
        vx, vy = self.last_velocity

        # 基于速度预测当前位置
        predicted_x = latest[0] + vx
        predicted_y = latest[1] + vy

        return (int(predicted_x), int(predicted_y))

    def _interpolate_volleyball(self, vb_dets, detected=True):
        """
        时序插值与动量保留
        :param detected: 本帧是否真正执行了检测；抽帧跳过的帧传 False，
                         不计入丢帧容忍预算（否则抽帧会截断球的连续轨迹）
        返回: (interpolated_det, is_interpolated)
        """
        if vb_dets and len(vb_dets) > 0:
            # 当前帧检测到排球
            best_det = max(vb_dets, key=lambda x: x['confidence'])
            center = best_det['center']

            # 更新历史和计数
            self._update_volleyball_history(center)
            self.last_velocity = self._calculate_velocity()
            self.missed_frames_counter = 0

            return best_det, False

        else:
            # 当前帧未检测到排球
            if detected:
                self.missed_frames_counter += 1

            if self.missed_frames_counter <= self.volleyball_patience and len(self.volleyball_history) > 0:
                # 在容忍期内，使用预测位置
                predicted_center = self._predict_volleyball_position()

                if predicted_center is not None:
                    # 创建预测的检测结果
                    interpolated_det = {
                        'bbox': [predicted_center[0] - 30, predicted_center[1] - 30,
                                 predicted_center[0] + 30, predicted_center[1] + 30],
                        'confidence': 0.5,  # 降低置信度因为是预测的
                        'class_id': 0,
                        'center': predicted_center,
                        'is_predicted': True  # 标记为预测结果
                    }

                    # 同时更新历史（使用预测位置）
                    self._update_volleyball_history(predicted_center)
                    self.last_velocity = self._calculate_velocity()

                    return interpolated_det, True

            # 超出容忍期或无法预测，返回空
            return None, False

    def get_body_height(self, valid_kpts):
        """从关键点估算身高（肩部到脚踝的垂直距离）"""
        if valid_kpts and 'left_shoulder' in valid_kpts and 'left_ankle' in valid_kpts:
            return abs(valid_kpts['left_shoulder'][1] - valid_kpts['left_ankle'][1])
        return 500  # 默认值

    def get_wrist_y(self, valid_kpts):
        """获取手腕Y坐标（取左右手平均）"""
        wrist_ys = []
        if valid_kpts:
            if 'left_wrist' in valid_kpts:
                wrist_ys.append(valid_kpts['left_wrist'][1])
            if 'right_wrist' in valid_kpts:
                wrist_ys.append(valid_kpts['right_wrist'][1])
        return sum(wrist_ys) / len(wrist_ys) if wrist_ys else None

    def biomechanics_summary(self, fps: float = 30.0):
        """动作级生物力学指标汇总（见 metrics.py 与文献对照表）。"""
        try:
            return summarize_biomechanics(self.metric_records, fps=fps)
        except Exception as exc:
            return {"error": str(exc), "total_frames": len(self.metric_records)}

    def reset_tracking(self):
        """重置所有跨帧状态（换视频/换会话前必须调用，避免上一段素材的轨迹污染下一段）。"""
        self.tracker.reset()
        self.frame_person_counts = []
        self.primary_id_history = []
        self.last_primary_id = None
        self.last_valid_kpts = None
        self.volleyball_history = []
        self.missed_frames_counter = 0
        self.last_velocity = None
        self.ball_frame_index = 0
        self.metric_records = []

    def set_calibration(self, calibration):
        """calibration 可以是 {label, points} 或直接 points 字典。"""
        if not calibration:
            self.calibration_H = None
            self.calibration_label = None
            return
        if isinstance(calibration, dict) and "points" in calibration:
            points = calibration.get("points") or {}
            self.calibration_label = calibration.get("label")
        else:
            points = calibration
        self.calibration_H = homography_from_points(points)

    def _select_primary(self, persons, ball_center=None):
        """主分析对象：优先延续上一帧 ID，其次离球最近，最后取面积最大者。"""
        if not persons:
            return None
        if self.last_primary_id is not None:
            for person in persons:
                if person.get("track_id") == self.last_primary_id:
                    return person
        if ball_center is not None:
            def distance(person):
                x1, y1, x2, y2 = person["bbox"]
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                return ((cx - ball_center[0]) ** 2 + (cy - ball_center[1]) ** 2) ** 0.5
            return min(persons, key=distance)
        return persons[0]

    def tracking_summary(self):
        counts = self.frame_person_counts or [0]
        return {
            "max_person_count": int(max(counts)),
            "avg_person_count": round(sum(counts) / len(counts), 2),
            "multi_person_frames": int(sum(1 for c in counts if c > 1)),
            "primary_ids": sorted(set(self.primary_id_history)),
        }

    # 给UI调用的核心函数：输入一帧画面 → 返回标注好的画面 + 评分结果
    def process_frame(self, frame, action_type="dig", criteria=None, calibration=None):
        """多人跟踪 + 排球检测 + 主目标选择 + 几何校正 + 动作评分。"""
        if calibration is not None:
            self.set_calibration(calibration)

        # verbose=False：关闭 ultralytics 的逐帧推理摘要打印（原先每帧写一次控制台）
        # 注意：不再传已废弃的 half 参数；FP16 在 __init__ 里通过 model.half() 设置
        results = self.pose_model(frame, imgsz=POSE_IMGSZ, conf=POSE_CONF,
                                  device=self.device, verbose=False)
        persons = extract_persons(results)
        track_ids = self.tracker.update([p["bbox"] for p in persons]) if persons else []
        for person, tid in zip(persons, track_ids):
            person["track_id"] = tid

        raw_vb_dets = []
        ball_detected = True
        if self.detect_volleyball_flag and self.volleyball_model is not None:
            # 抽帧执行：跳过的帧由 _interpolate_volleyball 的轨迹预测补位（容忍 5 帧丢失）
            ball_detected = self.ball_frame_index % self.ball_interval == 0
            if ball_detected:
                raw_vb_dets = detect_volleyball(self.volleyball_model, frame,
                                                conf=BALL_CONF, imgsz=BALL_IMGSZ)
            self.ball_frame_index += 1
        vb_dets, is_interpolated = self._interpolate_volleyball(raw_vb_dets, ball_detected)
        if vb_dets is not None:
            vb_dets = [vb_dets]
        ball_center = vb_dets[0]["center"] if vb_dets else None

        primary = self._select_primary(persons, ball_center)
        person_id = primary.get("track_id") if primary else None
        valid_kpts = primary.get("keypoints") if primary else None
        if valid_kpts is not None and self.calibration_H is not None:
            valid_kpts = correct_keypoints(valid_kpts, self.calibration_H, frame.shape)
        if valid_kpts is None and self.last_valid_kpts is not None:
            valid_kpts = self.last_valid_kpts
        elif valid_kpts is not None:
            self.last_valid_kpts = valid_kpts
        if person_id is None:
            person_id = self.last_primary_id
        if person_id is not None:
            self.last_primary_id = person_id
        self.frame_person_counts.append(len(persons))
        if person_id is not None:
            self.primary_id_history.append(person_id)
        try:
            self.metric_records.append(compute_frame_metrics(valid_kpts) if valid_kpts else {})
        except Exception:
            self.metric_records.append({})

        pose_judgment = None
        if valid_kpts:
            try:
                pose_judgment = judge_pose(valid_kpts, action_type, criteria)
                if pose_judgment and 'score_details' in pose_judgment:
                    for kpt_name, details in pose_judgment['score_details'].items():
                        base_kpt_name = kpt_name.replace('_angle', '').replace('_ratio', '').replace('_shoulder', '')
                        if base_kpt_name in valid_kpts:
                            details['pt'] = valid_kpts[base_kpt_name]
            except Exception:
                pose_judgment = None

        if pose_judgment is not None:
            pose_judgment["person_id"] = person_id
            pose_judgment["person_count"] = len(persons)
            pose_judgment["track_ids"] = list(track_ids)
            pose_judgment["calibration_applied"] = self.calibration_H is not None
            pose_judgment["calibration_label"] = self.calibration_label

        if (vb_dets or (not vb_dets and is_interpolated)) and pose_judgment and valid_kpts:
            det_to_use = vb_dets[0] if vb_dets else None
            if det_to_use:
                vx, vy = det_to_use['center']

                def is_valid_pt(pt):
                    return pt and len(pt) == 2 and pt[0] > 0 and pt[1] > 0

                distances = []
                if 'right_wrist' in valid_kpts and 'right_elbow' in valid_kpts:
                    wrist_pt = valid_kpts['right_wrist']
                    elbow_pt = valid_kpts['right_elbow']
                    if is_valid_pt(wrist_pt) and is_valid_pt(elbow_pt):
                        distances.append(abs(vy - (wrist_pt[1] + elbow_pt[1]) / 2))
                if 'left_wrist' in valid_kpts and 'left_elbow' in valid_kpts:
                    wrist_pt = valid_kpts['left_wrist']
                    elbow_pt = valid_kpts['left_elbow']
                    if is_valid_pt(wrist_pt) and is_valid_pt(elbow_pt):
                        distances.append(abs(vy - (wrist_pt[1] + elbow_pt[1]) / 2))
                if distances:
                    pose_judgment['volleyball_arm_distance'] = min(distances)
                    pose_judgment['distance_threshold'] = frame.shape[0] * 0.35
                    pose_judgment['volleyball_interpolated'] = is_interpolated

        annotated_frame = draw_pose_results(frame, results, pose_judgment)
        if persons:
            annotated_frame = draw_track_labels(annotated_frame, persons, track_ids, person_id)
        if vb_dets:
            annotated_frame = draw_volleyball_results(annotated_frame, vb_dets)

        session_info = self.action_session.update(pose_judgment, vb_dets if vb_dets else [], frame.shape[0], valid_kpts)

        return annotated_frame, pose_judgment, vb_dets if vb_dets else [], valid_kpts, session_info


from pose_judge import generate_feedback

class ActionSession:
    def __init__(self, action_type="dig"):
        self.action_type = action_type
        self.is_active = False          # 动作是否正在进行
        self.current_period_data = []   # 存储当前动作区间的详情
        self.action_scores = []         # 存储所有完成动作的得分
        self.action_feedbacks = []      # 存储所有完成动作的反馈
        self.dig_count = 0             # 垫球次数

        # 区间记录相关
        self.interval_records = []     # 当前区间的记录列表 [{'distance': ..., 'volleyball_y': ..., 'total_score': ..., 'score_details': ...}, ...]
        self.entry_threshold_multiplier = 1.3  # 进入阈值（身高倍数）
        self.rebound_threshold_multiplier = 0.5  # 折返判定阈值（身高倍数）

    def _get_body_height(self, valid_kpts):
        """从关键点估算身高"""
        if valid_kpts and 'left_shoulder' in valid_kpts and 'left_ankle' in valid_kpts:
            return abs(valid_kpts['left_shoulder'][1] - valid_kpts['left_ankle'][1])
        return 500  # 默认值

    def _get_wrist_y(self, valid_kpts):
        """获取手腕Y坐标（取左右手平均）"""
        wrist_ys = []
        if valid_kpts:
            if 'left_wrist' in valid_kpts:
                wrist_ys.append(valid_kpts['left_wrist'][1])
            if 'right_wrist' in valid_kpts:
                wrist_ys.append(valid_kpts['right_wrist'][1])
        return sum(wrist_ys) / len(wrist_ys) if wrist_ys else None

    def _check_rebound(self, records, body_height):
        """
        宏观落差法检测折返
        records: [{'distance': ..., 'volleyball_y': ..., 'total_score': ..., 'score_details': ...}, ...]
        body_height: 身高用于计算阈值
        返回: True表示有折返（是有效的垫球动作）
        """
        if len(records) < 3:
            return False

        # 提取距离序列（过滤掉None值）
        distances = [r['distance'] for r in records if r.get('distance') is not None]
        volleyball_ys = [r['volleyball_y'] for r in records if r.get('volleyball_y') is not None]

        # 确保有足够的数据
        if len(distances) < 3 or len(volleyball_ys) < 3:
            return False

        first_dist = distances[0]
        last_dist = distances[-1]
        min_dist_idx = distances.index(min(distances))
        min_dist = distances[min_dist_idx]

        rebound_threshold = body_height * self.rebound_threshold_multiplier

        # 1. 距离折返：必须同时满足"进来时落差大" AND "出去时落差大"
        is_not_at_ends = (min_dist_idx != 0 and min_dist_idx != len(distances) - 1)
        drop_from_first = first_dist - min_dist
        drop_from_last = last_dist - min_dist
        has_significant_drop = (drop_from_first > rebound_threshold and drop_from_last > rebound_threshold)

        # 2. 轨迹折返：物理最低点（即 Y 的最大值）应该在区间中间
        max_y_idx = volleyball_ys.index(max(volleyball_ys))
        y_drop_is_valid = (max_y_idx > 0 and max_y_idx < len(volleyball_ys) - 1)

        return is_not_at_ends and has_significant_drop and y_drop_is_valid

    def update(self, pose_judgment, volleyballs, frame_height, valid_kpts=None):
        """
        每帧调用一次，使用"宽进严出"的区间结算逻辑
        """
        # 计算身高和手腕Y坐标
        body_height = self._get_body_height(valid_kpts) if valid_kpts else 500
        wrist_y = self._get_wrist_y(valid_kpts)

        # 计算排球距离和位置
        distance = None
        volleyball_y = None
        if volleyballs and valid_kpts:
            # 拿到置信度最高的球
            best_ball = max(volleyballs, key=lambda x: x['confidence'])
            vx, vy = best_ball['center']
            volleyball_y = vy

            # 检查坐标有效性的辅助函数
            def is_valid_pt(pt):
                return pt and len(pt) == 2 and pt[0] > 0 and pt[1] > 0

            # 存储左右手的距离
            distances = []

            # 计算右手距离
            if 'right_wrist' in valid_kpts and 'right_elbow' in valid_kpts:
                wrist_pt = valid_kpts['right_wrist']
                elbow_pt = valid_kpts['right_elbow']
                if is_valid_pt(wrist_pt) and is_valid_pt(elbow_pt):
                    arm_center_y = (wrist_pt[1] + elbow_pt[1]) / 2
                    dist = abs(vy - arm_center_y)
                    distances.append(dist)

            # 计算左手距离
            if 'left_wrist' in valid_kpts and 'left_elbow' in valid_kpts:
                wrist_pt = valid_kpts['left_wrist']
                elbow_pt = valid_kpts['left_elbow']
                if is_valid_pt(wrist_pt) and is_valid_pt(elbow_pt):
                    arm_center_y = (wrist_pt[1] + elbow_pt[1]) / 2
                    dist = abs(vy - arm_center_y)
                    distances.append(dist)

            if distances:
                distance = min(distances)

        # 进入阈值：身高的1.3倍
        entry_threshold = body_height * self.entry_threshold_multiplier

        # 判断是否在范围内且排球在手腕上方
        in_range = distance is not None and distance < entry_threshold
        above_wrist = volleyball_y is not None and wrist_y is not None and volleyball_y < wrist_y

        should_be_active = in_range and above_wrist

        # 状态切换逻辑
        if should_be_active:
            if not self.is_active:
                # 动作开始 - 开启新区间
                self.is_active = True
                self.interval_records = []
                if DEBUG_LOG:
                    print("--- 动作区间开始 ---")

            # 记录当前帧数据
            if pose_judgment:
                self.interval_records.append({
                    'distance': distance,
                    'volleyball_y': volleyball_y,
                    'total_score': pose_judgment.get('total_score', 0),
                    'score_details': pose_judgment.get('score_details', {}),
                    'person_id': pose_judgment.get('person_id')
                })
        else:
            if self.is_active:
                # 动作结束 - 结算区间
                self.is_active = False
                if DEBUG_LOG:
                    print(f"--- 动作区间结束，记录帧数: {len(self.interval_records)} ---")

                # 使用宏观落差法检测折返
                if len(self.interval_records) >= 4 or self._check_rebound(self.interval_records, body_height):
                    self.dig_count += 1
                    if DEBUG_LOG:
                        print(f"检测到动作！次数: {self.dig_count}")

                    # 计算区间平均分
                    avg_score = sum(r['total_score'] for r in self.interval_records) / len(self.interval_records)

                    # 聚合详情并生成反馈
                    combined_details = self._get_avg_details([r['score_details'] for r in self.interval_records])

                    if combined_details:
                        feedback = generate_feedback(combined_details, self.action_type)
                        self.action_feedbacks.append(feedback)

                    # 判断是否标准
                    is_standard = avg_score >= 80
                    self.action_scores.append({
                        'score': avg_score,
                        'is_standard': is_standard,
                        'frame_count': len(self.interval_records),
                        'person_id': self.interval_records[-1].get('person_id')
                    })
                    if DEBUG_LOG:
                        print(f"区间得分: {avg_score:.2f}, 标准: {is_standard}")
                else:
                    if DEBUG_LOG:
                        print("无折返，不计入垫球次数")

                self.interval_records = []

        return {
            'is_active': self.is_active,
            'in_range': in_range,
            'above_wrist': above_wrist,
            'distance': distance,
            'entry_threshold': entry_threshold,
            'current_score': self.interval_records[-1].get('total_score', 0) if self.interval_records else 0,
            'dig_count': self.dig_count
        }

    def _get_avg_details(self, details_list):
        """聚合多帧详情的辅助函数"""
        avg_details = {}
        for detail in details_list:
            for k, v in detail.items():
                # 只处理值为字典且包含score字段的键
                if isinstance(v, dict) and "score" in v:
                    if k not in avg_details:
                        avg_details[k] = {"score": 0, "count": 0}
                    avg_details[k]["score"] += v.get("score", 0)
                    avg_details[k]["count"] += 1
        return {k: {"score": v["score"]/v["count"]} for k, v in avg_details.items()}

    def get_summary(self):
        """获取动作分析总结"""
        action_count = len(self.action_scores)
        standard_count = sum(1 for s in self.action_scores if s.get('is_standard', False))
        average_score = sum(s['score'] for s in self.action_scores) / action_count if action_count > 0 else 0

        return {
            "dig_count": self.dig_count,
            "action_count": action_count,
            "standard_count": standard_count,
            "action_scores": self.action_scores,
            "action_feedbacks": self.action_feedbacks,
            "average_score": round(average_score, 2)
        }

    def reset(self):
        """重置会话状态"""
        self.is_active = False
        self.current_period_data = []
        self.action_scores = []
        self.action_feedbacks = []
        self.dig_count = 0
        self.interval_records = []
