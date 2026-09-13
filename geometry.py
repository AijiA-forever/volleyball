# -*- coding: utf-8 -*-
"""几何校正：基于场地四角点单应，补偿透视导致的垂直比例失真。"""
import numpy as np
import cv2

COURT_LENGTH = 18.0
COURT_WIDTH = 9.0

ALIASES = {
    "far_left": {"far_left", "far_end_line_left"},
    "far_right": {"far_right", "far_end_line_right"},
    "near_left": {"near_left", "near_end_line_left"},
    "near_right": {"near_right", "near_end_line_right"},
}


def _pick(points, key):
    for alias in ALIASES[key]:
        if alias in points:
            value = points[alias]
            if isinstance(value, dict):
                return [float(value.get("x", 0)), float(value.get("y", 0))]
            return [float(value[0]), float(value[1])]
    return None


def homography_from_points(points):
    """输入四角点，返回图像→场地平面(米)的单应矩阵，失败返回 None。"""
    if not points:
        return None
    far_left = _pick(points, "far_left")
    far_right = _pick(points, "far_right")
    near_left = _pick(points, "near_left")
    near_right = _pick(points, "near_right")
    if None in (far_left, far_right, near_left, near_right):
        return None
    src = np.array([far_left, far_right, near_right, near_left], dtype=np.float32)
    dst = np.array([
        [0.0, COURT_LENGTH],
        [COURT_WIDTH, COURT_LENGTH],
        [COURT_WIDTH, 0.0],
        [0.0, 0.0],
    ], dtype=np.float32)
    try:
        return cv2.getPerspectiveTransform(src, dst)
    except Exception:
        return None


def transform_point(H, point):
    if H is None or point is None:
        return None
    arr = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
    out = cv2.perspectiveTransform(arr, H)[0][0]
    return float(out[0]), float(out[1])


def local_scale(H, point):
    """估计该图像位置处 1 像素对应的场地距离（米/像素）>"""
    p0 = transform_point(H, point)
    p1 = transform_point(H, (point[0] + 1.0, point[1]))
    p2 = transform_point(H, (point[0], point[1] + 1.0))
    if p0 is None or p1 is None or p2 is None:
        return None
    sx = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5
    sy = ((p2[0] - p0[0]) ** 2 + (p2[1] - p0[1]) ** 2) ** 0.5
    return (sx + sy) / 2.0


def correct_keypoints(keypoints, H, frame_shape, min_factor=0.5, max_factor=2.0):
    """按局部尺度对垂直方向做近似透视补偿，使远近选手的角度可比。"""
    if H is None or not keypoints:
        return keypoints
    h, w = frame_shape[:2]
    ankles = [keypoints[k] for k in ("left_ankle", "right_ankle") if k in keypoints]
    if ankles:
        ground_x = sum(p[0] for p in ankles) / len(ankles)
        ground_y = max(p[1] for p in ankles)
    else:
        points = list(keypoints.values())
        ground_x = sum(p[0] for p in points) / len(points)
        ground_y = max(p[1] for p in points)
    scale_local = local_scale(H, (ground_x, ground_y))
    scale_ref = local_scale(H, (w / 2.0, float(h - 1)))
    if not scale_local or not scale_ref or scale_ref <= 0:
        return keypoints
    factor = max(min_factor, min(max_factor, scale_local / scale_ref))
    corrected = {}
    for name, (x, y) in keypoints.items():
        corrected[name] = (float(x), float(ground_y - (ground_y - float(y)) * factor))
    return corrected