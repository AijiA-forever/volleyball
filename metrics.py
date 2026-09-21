# -*- coding: utf-8 -*-
"""排球生物力学指标计算（初版，矢状面优先）。

设计依据见 docs/排球动作生物力学量化指标研究.md
原则：
1) 只实现 2D 姿态可靠可测的指标（矢状面角度/角速度、重心速度、时序）；
2) 肩关节等 2D 效度差的指标只记录原始值，不参与评分；
3) 速度类指标统一用"身高/秒(bh/s)"归一化，便于不同身高/距离比较；
4) 所有指标都附带文献来源（LITERATURE），便于报告自动引用。

主要参考文献：
- Reeser 等 2010, Sports Health, DOI 10.1177/1941738110374624
- Sarvestan 等 2020, J Sports Sci, DOI 10.1080/02640414.2020.1782008
- De Bleecker 等 2024, Gait & Posture, DOI 10.1016/j.gaitpost.2024.07.001
- De Bleecker 等 2025, J Sports Sci, DOI 10.1080/02640414.2025.2486795
- Baldinger 等 2025, Sensors, DOI 10.3390/s25030799
- Ozawa 等 2021/2022, Sports Biomechanics
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

SIDES = ("left", "right")


def joint_angle(a, b, c) -> float:
    """以 b 为顶点的夹角（度）。三点均为 (x, y)。"""
    if not (a and b and c):
        return float("nan")
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    cos_v = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
    return math.degrees(math.acos(cos_v))


def _mid(p, q):
    if p and q:
        return ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)
    return None


def _angle_from_vertical(p_bottom, p_top) -> float:
    """向量(下→上)与竖直方向的夹角（度），0 表示完全竖直。"""
    if not p_bottom or not p_top:
        return float("nan")
    dx = p_top[0] - p_bottom[0]
    dy = p_top[1] - p_bottom[1]
    if abs(dy) < 1e-6 and abs(dx) < 1e-6:
        return float("nan")
    # 图像坐标 y 向下，故竖直向上向量为 (0, -1)
    cos_v = max(-1.0, min(1.0, (-dy) / math.hypot(dx, dy)))
    return math.degrees(math.acos(cos_v))


def body_height_px(kpts) -> Optional[float]:
    """身高估计（肩中点到踝中点的垂直距离，像素）。"""
    sh = _mid(kpts.get("left_shoulder"), kpts.get("right_shoulder"))
    an = _mid(kpts.get("left_ankle"), kpts.get("right_ankle"))
    if sh and an:
        h = abs(an[1] - sh[1])
        return h if h > 10 else None
    return None


def compute_frame_metrics(kpts: Dict[str, tuple]) -> Dict[str, float]:
    """单帧指标：关节角度 + 平台角度 + 重心/腕部位置。"""
    out: Dict[str, float] = {}
    if not kpts:
        return out
    for side in SIDES:
        s, e, w = kpts.get(f"{side}_shoulder"), kpts.get(f"{side}_elbow"), kpts.get(f"{side}_wrist")
        h, k, a = kpts.get(f"{side}_hip"), kpts.get(f"{side}_knee"), kpts.get(f"{side}_ankle")
        if s and e and w:
            out[f"{side}_elbow_angle"] = joint_angle(s, e, w)
            out[f"{side}_shoulder_angle"] = joint_angle(h or k, s, e)  # 上臂相对躯干抬升
        if h and k and a:
            out[f"{side}_knee_angle"] = joint_angle(h, k, a)
        if s and h and k:
            out[f"{side}_hip_angle"] = joint_angle(s, h, k)

    sh = _mid(kpts.get("left_shoulder"), kpts.get("right_shoulder"))
    hip = _mid(kpts.get("left_hip"), kpts.get("right_hip"))
    an = _mid(kpts.get("left_ankle"), kpts.get("right_ankle"))
    lw, rw = kpts.get("left_wrist"), kpts.get("right_wrist")
    if sh and hip:
        out["trunk_inclination"] = _angle_from_vertical(hip, sh)
    if lw and rw:
        # 前臂平台与水平线的夹角（垫球平台指向）
        dx = rw[0] - lw[0]
        dy = rw[1] - lw[1]
        out["platform_angle"] = math.degrees(math.atan2(dy, dx))
    bh = body_height_px(kpts)
    if bh:
        out["body_height_px"] = bh
    if hip:
        out["hip_y_px"] = hip[1]
    if sh:
        out["shoulder_y_px"] = sh[1]
    if lw and rw:
        out["wrist_y_px"] = (lw[1] + rw[1]) / 2.0
    return out


def _clean(series: List[float]) -> List[float]:
    return [x for x in series if x is not None and not math.isnan(x)]


def _peak_abs(values: List[float]) -> float:
    vals = _clean(values)
    return max((abs(v) for v in vals), default=float("nan"))


def _diff_series(values: List[float], fps: float) -> List[float]:
    """中心差分求一阶导（单位/秒）。"""
    n = len(values)
    out = [float("nan")] * n
    if n < 2 or not fps:
        return out
    for i in range(n):
        if i == 0:
            out[i] = (values[1] - values[0]) * fps if not math.isnan(values[0]) and not math.isnan(values[1]) else float("nan")
        elif i == n - 1:
            out[i] = (values[-1] - values[-2]) * fps if not math.isnan(values[-1]) and not math.isnan(values[-2]) else float("nan")
        else:
            if math.isnan(values[i - 1]) or math.isnan(values[i + 1]):
                out[i] = float("nan")
            else:
                out[i] = (values[i + 1] - values[i - 1]) / 2.0 * fps
    return out


LITERATURE = {
    "peak_knee_angular_velocity": "Sarvestan 2020, J Sports Sci (DOI 10.1080/02640414.2020.1782008)：成功扣球膝角速度高约 12.4%",
    "peak_hip_angular_velocity": "Sarvestan 2020：成功扣球髋角速度高约 13.3%",
    "takeoff_com_velocity": "Sarvestan 2020：成功扣球起跳重心垂直速度高约 6.5%",
    "landing_knee_flexion_angle": "De Bleecker 2024, Gait & Posture：矢状面膝角 ICC 0.61-0.89；落地中段窗口信度最好",
    "landing_knee_flexion_velocity": "De Bleecker 2025, J Sports Sci：膝关节功与髌腱病风险相关（HR 1.20-1.32），落地缓冲速度可作代理指标",
    "shoulder_abduction": "Reeser 2010, Sports Health：击球瞬间肩外展约 130°；单视角 2D 效度差，仅作参考",
    "camera_view": "Baldinger 2025, Sensors：后视机位 OpenPose 效度最好",
}


def _median(values):
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None
    vals = sorted(vals)
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0


def _moving_average(series, window):
    """滑动平均（忽略 NaN），用于运动学信号低通滤波。

    文献中运动学数据通常需要低通滤波后再求导（本实现用滑动平均近似）。
    """
    if window < 3:
        return list(series)
    half = window // 2
    out = []
    for i in range(len(series)):
        seg = [series[j] for j in range(max(0, i - half), min(len(series), i + half + 1))
               if not math.isnan(series[j])]
        out.append(sum(seg) / len(seg) if seg else float("nan"))
    return out


def _winsorize(series, k=3.0):
    """按中位数 ± k*MAD 裁剪极值，抑制关键点抖动造成的离群点。"""
    vals = [v for v in series if v is not None and not math.isnan(v)]
    if len(vals) < 5:
        return list(series)
    med = _median(vals)
    mad = _median([abs(v - med) for v in vals]) or 1e-6
    low, high = med - k * 1.4826 * mad, med + k * 1.4826 * mad
    return [float("nan") if (v is None or math.isnan(v)) else max(low, min(high, v)) for v in series]


def _sanitize_records(records):
    """剔除明显异常帧：身高估计异常、角度超出人体范围、归一化位置越界。"""
    heights = [r.get("body_height_px") for r in records if r and r.get("body_height_px")]
    med = _median(heights)
    valid = 0
    for r in records:
        if not r:
            continue
        bh = r.get("body_height_px")
        if med and (not bh or bh < 0.5 * med or bh > 2.0 * med):
            r.clear()
            continue
        for key in ("hip_y_norm", "wrist_y_norm", "shoulder_y_norm"):
            v = r.get(key)
            if v is not None and (math.isnan(v) or v < -1.0 or v > 3.0):
                r.pop(key, None)
        for key in list(r.keys()):
            if key.endswith("_angle"):
                v = r[key]
                if v is None or math.isnan(v) or v < 0.0 or v > 180.0:
                    r.pop(key, None)
        valid += 1
    return valid


def summarize(records: List[Dict[str, float]], fps: float = 30.0) -> Dict[str, object]:
    """把逐帧指标汇总为动作级生物力学指标。

    records: compute_frame_metrics 的结果列表（按帧顺序，缺失帧可为空 dict）
    """
    if not records:
        return {"valid_frames": 0, "phases_detected": False}
    n = len(records)
    valid = _sanitize_records(records)

    window = max(3, (int(round(fps / 6.0)) // 2) * 2 + 1)      # 角度：约 6 Hz 低通
    pos_window = max(5, (int(round(fps / 4.0)) // 2) * 2 + 1)   # 位置：约 4 Hz 低通（更严格）

    ref_height = _median([r.get("body_height_px") for r in records if r and r.get("body_height_px")])

    def series(key, smooth=True, win=None):
        raw = [r.get(key, float("nan")) if r else float("nan") for r in records]
        raw = _winsorize(raw)
        return _moving_average(raw, win or window) if smooth else raw

    def pos_series(key):
        if not ref_height:
            return [float("nan")] * n
        raw = [r.get(key, float("nan")) / ref_height if (r and r.get(key) is not None) else float("nan")
               for r in records]
        return _moving_average(_winsorize(raw), pos_window)

    knee = series("right_knee_angle")
    hip = series("right_hip_angle")
    elbow = series("right_elbow_angle")
    hip_y = pos_series("hip_y_px")
    wrist_y = pos_series("wrist_y_px")
    trunk = series("trunk_inclination")
    platform = series("platform_angle", smooth=False)

    knee_vel = _diff_series(knee, fps)          # 角速度 deg/s
    hip_vel = _diff_series(hip, fps)
    elbow_vel = _diff_series(elbow, fps)
    hip_vel_up = [-v for v in _diff_series(hip_y, fps)] if any(not math.isnan(v) for v in hip_y) else [float("nan")] * n
    wrist_vel_up = [-v for v in _diff_series(wrist_y, fps)] if any(not math.isnan(v) for v in wrist_y) else [float("nan")] * n

    # 相位检测：以髋部高度序列为准（y 越小越高）
    peak_idx = None
    if any(not math.isnan(v) for v in hip_y):
        candidates = [(v, i) for i, v in enumerate(hip_y) if not math.isnan(v)]
        peak_idx = min(candidates)[1]
    takeoff_idx = None
    landing_idx = None
    if peak_idx is not None:
        for i in range(peak_idx, -1, -1):
            if not math.isnan(hip_vel_up[i]) and hip_vel_up[i] > 0.5:
                takeoff_idx = i
                break
        for i in range(peak_idx, n):
            if not math.isnan(hip_vel_up[i]) and hip_vel_up[i] < -0.5:
                landing_idx = i
                break

    # 击球帧：起跳后腕部速度最大的帧
    contact_idx = None
    window = range(takeoff_idx if takeoff_idx is not None else 0, (peak_idx + 6) if peak_idx is not None else n)
    best = float("-inf")
    for i in window:
        if i < n and not math.isnan(wrist_vel_up[i]) and wrist_vel_up[i] > best:
            best = wrist_vel_up[i]
            contact_idx = i

    # 落地缓冲：落地后 0.5 秒内膝关节最小角（最大屈曲）
    landing_knee_angle = float("nan")
    landing_knee_vel = float("nan")
    if landing_idx is not None:
        end = min(n, landing_idx + max(1, int(0.5 * fps)))
        seg = [(knee[i], i) for i in range(landing_idx, end) if not math.isnan(knee[i])]
        if seg:
            landing_knee_angle = min(seg)[0]
            landing_knee_vel = knee_vel[min(seg)[1]]

    jump_height = float("nan")
    if any(not math.isnan(v) for v in hip_y):
        vals = _clean(hip_y)
        jump_height = max(vals) - min(vals)

    result = {
        "valid_frames": valid,
        "total_frames": n,
        "valid_ratio": round(valid / n, 3) if n else 0.0,
        "fps": round(float(fps), 2),
        "reference_body_height_px": round(ref_height, 1) if ref_height else None,
        "phases_detected": takeoff_idx is not None and landing_idx is not None,
        "takeoff_frame": takeoff_idx,
        "contact_frame": contact_idx,
        "landing_frame": landing_idx,
        "peak_knee_angular_velocity": round(_peak_abs(knee_vel), 1),
        "peak_hip_angular_velocity": round(_peak_abs(hip_vel), 1),
        "peak_elbow_angular_velocity": round(_peak_abs(elbow_vel), 1),
        "peak_wrist_speed_bh": round(_peak_abs(wrist_vel_up), 2),
        "takeoff_com_velocity_bh": round(max(_clean(hip_vel_up), default=float("nan")), 2),
        "jump_height_bh": round(jump_height, 3) if not math.isnan(jump_height) else float("nan"),
        "landing_knee_flexion_angle": round(landing_knee_angle, 1) if not math.isnan(landing_knee_angle) else float("nan"),
        "landing_knee_flexion_velocity": round(abs(landing_knee_vel), 1) if not math.isnan(landing_knee_vel) else float("nan"),
        "min_knee_angle": round(min(_clean(knee)), 1) if _clean(knee) else float("nan"),
        "max_trunk_inclination": round(max(_clean(trunk)), 1) if _clean(trunk) else float("nan"),
        "platform_angle_mean": round(sum(_clean(platform)) / len(_clean(platform)), 1) if _clean(platform) else float("nan"),
        "platform_angle_std": round(_std(_clean(platform)), 1) if _clean(platform) else float("nan"),
        "literature": LITERATURE,
    }
    # JSON 安全：NaN/Inf 一律转成 None，避免前端 JSON.parse 失败
    for key, value in list(result.items()):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            result[key] = None
    return result


def _std(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))