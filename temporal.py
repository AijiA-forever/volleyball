# -*- coding: utf-8 -*-
"""时序动作对比（最小闭环）：逐帧序列采集 + DTW 相似度。

设计说明
--------
1) 特征复用 metrics.compute_frame_metrics 的关节角度，并补两个以身高归一化的
   垂直位置（髋、腕相对踝中点），用来抵消拍摄距离造成的尺度差异；
2) 不做定长重采样：时长差异交给 DTW 的弯曲对齐处理，时长比单独作为"节奏差异"输出；
3) 相似度 = 100 * exp(-归一化DTW距离 / tau)。tau 默认 1.0，需要用教练评分校准
   （见 tools/validate_metrics.py 的一致性实验），原始距离同时输出，便于回调；
4) 标准模板取"中位数轨迹"（medoid，到其余样本 DTW 距离之和最小的一条），
   不引入 DTW 重心等更重的做法；
5) 序列采集以 analyzer 每帧返回的 session_info 为准：is_active 由真变假表示
   一个动作区间结束，dig_count 增加说明该区间被计为有效动作。

已知边界
--------
- 切分仍依赖现有 ActionSession 的排球轨迹判据，漏检会漏掉动作；
- 未做左右镜像归一，左撇子学生会与右手模板产生系统性差异；
- 单目 2D 的角度在肩关节等部位效度有限，结论需限定条件。

用法
----
    python temporal.py --self-test
    python temporal.py --build-template --videos <标准动作目录> --action dig --out templates/dig.npz
    python temporal.py --video <学生视频> --template templates/dig.npz
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from metrics import compute_frame_metrics

ANGLE_KEYS = [
    "left_elbow_angle", "right_elbow_angle",
    "left_knee_angle", "right_knee_angle",
    "left_hip_angle", "right_hip_angle",
    "trunk_inclination",
]
DERIVED_KEYS = ["platform_tilt"]
POS_KEYS = ["hip_y_rel", "wrist_y_rel"]
FEATURE_KEYS = ANGLE_KEYS + DERIVED_KEYS + POS_KEYS

# 各维度在算距离前的缩放：角度按 90°、位置按 0.25 倍身高换算到同一量级
SCALES = [90.0] * (len(ANGLE_KEYS) + len(DERIVED_KEYS)) + [0.25] * len(POS_KEYS)
MIN_CLIP_FRAMES = 4


def _mid(p, q):
    if p is None or q is None:
        return None
    return ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0)


def extract_frame_features(kpts) -> dict:
    """单帧时序特征：8 个关节角 + 2 个身高归一化垂直位置。"""
    if not kpts:
        return {}
    base = compute_frame_metrics(kpts)
    feats = {}
    for key in ANGLE_KEYS:
        value = base.get(key)
        if value is not None and not math.isnan(value):
            feats[key] = float(value)
    # 前臂平台倾角由 metrics 折叠到 [0,90]（无向线段，左右手顺序互换等价于旋转 180°），
    # 与生物力学汇总共用同一定义，避免两处各写一份折叠逻辑。
    tilt = base.get("platform_tilt")
    if tilt is not None and not math.isnan(tilt):
        feats["platform_tilt"] = float(tilt)
    bh = base.get("body_height_px")
    ankle = _mid(kpts.get("left_ankle"), kpts.get("right_ankle"))
    hip = _mid(kpts.get("left_hip"), kpts.get("right_hip"))
    wrists = _mid(kpts.get("left_wrist"), kpts.get("right_wrist"))
    if bh and ankle and hip:
        feats["hip_y_rel"] = (ankle[1] - hip[1]) / bh  # 图像 y 向下，取负后向上为正
    if bh and ankle and wrists:
        feats["wrist_y_rel"] = (ankle[1] - wrists[1]) / bh
    return feats


def clip_matrix(frames, smooth: int = 3) -> np.ndarray:
    """逐帧特征字典列表 -> T×D 矩阵。缺失值按列线性插值，两端用最近有效值。"""
    n = len(frames)
    if n == 0:
        return np.zeros((0, len(FEATURE_KEYS)))
    mat = np.full((n, len(FEATURE_KEYS)), np.nan)
    for i, frame in enumerate(frames):
        if not frame:
            continue
        for j, key in enumerate(FEATURE_KEYS):
            value = frame.get(key)
            if value is not None:
                mat[i, j] = float(value)
    idx = np.arange(n)
    for j in range(mat.shape[1]):
        col = mat[:, j]
        valid = ~np.isnan(col)
        if not valid.any():
            col[:] = 0.0
        elif valid.sum() < n:
            col[:] = np.interp(idx, idx[valid], col[valid])
    if smooth and smooth >= 3 and n >= smooth:
        kernel = np.ones(smooth) / smooth
        for j in range(mat.shape[1]):
            mat[:, j] = np.convolve(mat[:, j], kernel, mode="same")
    return mat


class SequenceCollector:
    """按动作区间采集逐帧特征序列（由调用方逐帧喂入 session_info 与关键点）。"""

    def __init__(self, min_frames: int = MIN_CLIP_FRAMES):
        self.min_frames = min_frames
        self.settings: dict = {}  # 本次采集使用的推理配置，用于模板一致性校验
        self.reset()

    def reset(self) -> None:
        self._buffer: list = []
        self._active = False
        self._last_count = 0
        self.clips: list = []  # 每个元素为逐帧特征字典列表
        self.rejected = 0      # 未被计为动作的区间数

    def update(self, session_info, valid_kpts) -> None:
        info = session_info or {}
        active = bool(info.get("is_active"))
        count = int(info.get("dig_count") or 0)
        if active and not self._active:
            self._buffer = []
        if active:
            self._buffer.append(extract_frame_features(valid_kpts))
        if self._active and not active:
            if count > self._last_count and len(self._buffer) >= self.min_frames:
                self.clips.append(self._buffer)
            elif count > self._last_count:
                self.rejected += 1
            self._buffer = []
        self._active = active
        self._last_count = count

    def finish(self) -> None:
        """视频结束时若仍在区间内，按同一规则结算。"""
        if self._active and len(self._buffer) >= self.min_frames:
            self.clips.append(self._buffer)
        self._buffer = []
        self._active = False


def dtw(a: np.ndarray, b: np.ndarray, band_ratio: float = 0.25):
    """多维 DTW。返回 (按 (n+m) 归一化的距离, 对齐路径)。带 Sakoe-Chiba 弯曲带。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        raise ValueError("序列不能为空")
    band = max(abs(n - m), int(round(band_ratio * max(n, m))))
    inf = float("inf")
    cost = np.full((n + 1, m + 1), inf)
    back = np.zeros((n + 1, m + 1), dtype=np.int8)  # 0 对角 1 上 2 左
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        lo = max(1, i - band)
        hi = min(m, i + band)
        for j in range(lo, hi + 1):
            step = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            best, direction = cost[i - 1, j - 1], 0
            if cost[i - 1, j] < best:
                best, direction = cost[i - 1, j], 1
            if cost[i, j - 1] < best:
                best, direction = cost[i, j - 1], 2
            cost[i, j] = step + best
            back[i, j] = direction
    if not np.isfinite(cost[n, m]):
        raise RuntimeError("DTW 弯曲带内没有可行路径")
    path = []
    i, j = n, m
    while i > 0 or j > 0:
        path.append((i - 1, j - 1))
        direction = back[i, j]
        if direction == 0:
            i, j = i - 1, j - 1
        elif direction == 1:
            i -= 1
        else:
            j -= 1
    path.reverse()
    return cost[n, m] / (n + m), path


def compare_matrices(student: np.ndarray, template: np.ndarray, tau: float = 1.0,
                     band_ratio: float = 0.25) -> dict:
    """比较两条原始单位的特征矩阵，返回距离、相似度、时长比与逐维偏差。"""
    if len(student) < 2 or len(template) < 2:
        raise ValueError("序列至少需要 2 帧")
    scales = np.asarray(SCALES, dtype=float)
    dist, path = dtw(student / scales, template / scales, band_ratio)
    sim = 100.0 * math.exp(-dist / tau) if tau > 0 else 0.0
    deviations = {}
    for j, key in enumerate(FEATURE_KEYS):
        deltas = [abs(student[i, j] - template[k, j]) for i, k in path]
        deviations[key] = round(sum(deltas) / len(deltas), 2) if deltas else None
    return {
        "dtw_distance": round(float(dist), 4),
        "similarity": round(float(sim), 1),
        "student_frames": int(len(student)),
        "template_frames": int(len(template)),
        "length_ratio": round(len(student) / len(template), 2),
        "deviations": deviations,
    }


def build_template(matrices: list):
    """取中位数轨迹（到其余样本 DTW 距离之和最小的一条）作为模板。"""
    candidates = [m for m in matrices if len(m) >= 2]
    if not candidates:
        raise ValueError("没有可用于建立模板的动作序列")
    if len(candidates) == 1:
        return candidates[0], {"n_clips": 1, "total_distances": [0.0]}
    scales = np.asarray(SCALES, dtype=float)
    totals = []
    for i, a in enumerate(candidates):
        total = 0.0
        for j, b in enumerate(candidates):
            if i == j:
                continue
            total += dtw(a / scales, b / scales)[0]
        totals.append(total)
    best = int(np.argmin(totals))
    return candidates[best], {"n_clips": len(candidates), "total_distances": [round(t, 4) for t in totals],
                              "medoid_index": best}


def save_template(path: Path, matrix: np.ndarray, action_type: str, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, matrix=matrix, feature_keys=np.array(FEATURE_KEYS), action_type=action_type,
             meta=json.dumps(meta, ensure_ascii=False))


def load_template(path: Path):
    data = np.load(path, allow_pickle=False)
    keys = [str(k) for k in data["feature_keys"]]
    if keys != FEATURE_KEYS:
        raise ValueError(f"模板特征维度与当前版本不一致：{keys}")
    return data["matrix"], str(data["action_type"]), json.loads(str(data["meta"]))


def settings_mismatch(template_settings, current_settings) -> list:
    """比较建模板与对比时用的推理配置，返回不一致的项描述。

    关键点会随推理尺寸/置信度阈值变化，进而改变切分边界和特征数值，
    两边配置不同时相似度不可比（实测同一段视频会从 100 分掉到 65 分）。
    """
    if not template_settings:
        return []
    return [f"{key}: 模板 {template_settings.get(key)} vs 当前 {current_settings.get(key)}"
            for key in template_settings if current_settings.get(key) != template_settings[key]]


def collect_clips(video: Path, action_type: str = "dig", min_frames: int = MIN_CLIP_FRAMES):
    """用现有分析器跑一遍视频并采集动作序列（不写结果视频、不落库）。"""
    import cv2

    from analyzer import (
        BALL_CONF, BALL_IMGSZ, POSE_CONF, POSE_IMGSZ,
        ActionSession, VolleyballActionAnalyzer,
    )

    analyzer = VolleyballActionAnalyzer(detect_volleyball=True)
    analyzer.action_session = ActionSession(action_type)
    analyzer.reset_tracking()
    collector = SequenceCollector(min_frames=min_frames)
    collector.settings = {
        "pose_imgsz": POSE_IMGSZ, "pose_conf": POSE_CONF,
        "ball_imgsz": BALL_IMGSZ, "ball_conf": BALL_CONF,
    }
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        _annotated, _judgment, _balls, kpts, info = analyzer.process_frame(frame, action_type)
        collector.update(info, kpts)
    cap.release()
    collector.finish()
    return collector, float(fps)


def _top_deviations(deviation: dict, limit: int = 3):
    items = [(k, v) for k, v in (deviation or {}).items() if v is not None]
    items.sort(key=lambda kv: kv[1], reverse=True)
    out = []
    for key, value in items[:limit]:
        unit = "°" if key.endswith("_angle") or key in ("trunk_inclination", "platform_tilt") else " bh"
        out.append(f"{key} {value}{unit}")
    return out


def _synthetic(n: int = 20, noise_deg: float = 0.0, knee_bias: float = 0.0, rng=None) -> np.ndarray:
    """构造量级接近真实数据的合成序列：角度约 90±40°，位置约 0.5 倍身高。"""
    t = np.linspace(0, math.pi, n)
    mat = np.zeros((n, len(FEATURE_KEYS)))
    angle_block = len(ANGLE_KEYS) + len(DERIVED_KEYS)
    for j in range(len(ANGLE_KEYS)):
        mat[:, j] = 90.0 + 40.0 * np.sin(t + j * 0.1)
    mat[:, len(ANGLE_KEYS)] = 40.0 + 20.0 * np.sin(t)
    mat[:, angle_block] = 0.50 + 0.15 * np.sin(t)
    mat[:, angle_block + 1] = 0.60 + 0.30 * np.sin(t + 0.3)
    if knee_bias:
        for key in ("left_knee_angle", "right_knee_angle"):
            mat[:, FEATURE_KEYS.index(key)] += knee_bias
    if noise_deg:
        noise = np.zeros_like(mat)
        noise[:, :angle_block] = rng.normal(0, noise_deg, (n, angle_block))
        noise[:, angle_block:] = rng.normal(0, noise_deg / 90.0 * 0.25, (n, len(POS_KEYS)))
        mat = mat + noise
    return mat


def self_test() -> int:
    """DTW 与模板逻辑的自检，不依赖模型。"""
    rng = np.random.default_rng(0)
    failures = []
    base = _synthetic()

    same = compare_matrices(base, base)
    ok = same["dtw_distance"] < 1e-9 and same["similarity"] > 99.9
    print(f"[{'PASS' if ok else 'FAIL'}] 同一序列 -> 距离 {same['dtw_distance']}，相似度 {same['similarity']}")
    failures += [] if ok else ["同一序列不为满分"]

    warped = base[np.linspace(0, len(base) - 1, 34).round().astype(int)]
    slow = compare_matrices(warped, base)
    ok = slow["similarity"] > 85 and 1.5 < slow["length_ratio"] < 1.8
    print(f"[{'PASS' if ok else 'FAIL'}] 拉伸 1.7 倍 -> 相似度 {slow['similarity']}，时长比 {slow['length_ratio']}")
    failures += [] if ok else ["时间拉伸后相似度偏低"]

    noise = compare_matrices(_synthetic(noise_deg=3, rng=rng), base)
    mild = compare_matrices(_synthetic(knee_bias=15), base)
    bias = compare_matrices(_synthetic(knee_bias=30), base)
    ok = noise["similarity"] > 90 and bias["similarity"] < mild["similarity"] < noise["similarity"] and 60 < bias["similarity"] < 90
    print(f"[{'PASS' if ok else 'FAIL'}] 逐级劣化 -> 3°噪声 {noise['similarity']} > 屈膝差 15° {mild['similarity']} > 30° {bias['similarity']}")
    failures += [] if ok else ["相似度未随动作偏差单调下降"]

    tpl, meta = build_template([base, warped, _synthetic(noise_deg=3, rng=rng)])
    ok = tpl.shape[1] == len(FEATURE_KEYS) and "medoid_index" in meta
    print(f"[{'PASS' if ok else 'FAIL'}] 模板选取 -> 形状 {tpl.shape}，样本数 {meta['n_clips']}")
    failures += [] if ok else ["模板构建异常"]

    single = clip_matrix([])
    ok = single.shape == (0, len(FEATURE_KEYS))
    print(f"[{'PASS' if ok else 'FAIL'}] 空序列 -> 形状 {single.shape}")
    failures += [] if ok else ["空序列处理异常"]

    filled = clip_matrix([{}, {"right_knee_angle": 120.0}, {}])
    ok = filled.shape == (3, len(FEATURE_KEYS)) and np.isfinite(filled).all()
    print(f"[{'PASS' if ok else 'FAIL'}] 缺失帧插值 -> 形状 {filled.shape}，无 NaN/Inf: {bool(np.isfinite(filled).all())}")
    failures += [] if ok else ["缺失值填充后仍有 NaN/Inf"]

    print("自检结果:", "全部通过" if not failures else f"{len(failures)} 项失败：{failures}")
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="时序动作对比（DTW 最小闭环）")
    parser.add_argument("--self-test", action="store_true", help="运行 DTW 自检后退出")
    parser.add_argument("--video", help="待评估的视频")
    parser.add_argument("--videos", help="用于建立模板的视频目录")
    parser.add_argument("--template", help="模板 .npz 路径")
    parser.add_argument("--build-template", action="store_true", help="用 --videos 建立模板")
    parser.add_argument("--action", default=None, help="动作类型：dig/serve/set/spike（对比时默认沿用模板）")
    parser.add_argument("--out", default="templates/dig.npz", help="模板输出路径")
    parser.add_argument("--tau", type=float, default=1.0, help="相似度映射尺度，需用教练评分校准")
    parser.add_argument("--band-ratio", type=float, default=0.25, help="DTW 弯曲带宽度比例")
    parser.add_argument("--min-frames", type=int, default=MIN_CLIP_FRAMES,
                        help=f"动作区间的最少帧数，低于该值不计入（默认 {MIN_CLIP_FRAMES}）")
    args = parser.parse_args()

    if args.self_test:
        return self_test()

    video_exts = (".mp4", ".mov", ".avi", ".mkv", ".webm")

    if args.build_template:
        if not args.videos:
            parser.error("--build-template 需要 --videos 目录")
        action = args.action or "dig"
        videos = sorted(p for p in Path(args.videos).iterdir() if p.suffix.lower() in video_exts)
        if not videos:
            print("目录里没有视频文件:", args.videos)
            return 1
        matrices, sources, settings_seen = [], [], {}
        for video in videos:
            collector, fps = collect_clips(video, action, min_frames=args.min_frames)
            settings_seen = collector.settings
            for clip in collector.clips:
                matrices.append(clip_matrix(clip))
                sources.append(video.name)
            print(f"  {video.name}: 采集 {len(collector.clips)} 个动作（{fps:.1f} fps）")
        if not matrices:
            print("\n没有采集到任何动作区间，无法建立模板。")
            print("最常见原因：视频里没有排球，或球太小/被遮挡导致检测不到。")
            print("现有切分依赖排球轨迹（进入区间要求“球与手臂距离 < 1.3 倍身高，且球在手腕上方”），")
            print("画面里没有球时状态机不会启动，一个区间也切不出来。")
            print("请依次检查：1) 示范时是否带球；2) --action 是否选对；3) 视频是否太短、人物是否完整入镜。")
            return 1
        matrix, meta = build_template(matrices)
        meta["sources"] = sources
        meta["action_type"] = action
        meta["settings"] = settings_seen
        save_template(Path(args.out), matrix, action, meta)
        print(f"模板已保存: {args.out}（{matrix.shape[0]} 帧 × {matrix.shape[1]} 维，来自 {meta['n_clips']} 个动作）")
        return 0

    if not args.video or not args.template:
        parser.error("需要 --video 与 --template（或用 --build-template / --self-test）")
    template, action_type, meta = load_template(Path(args.template))
    collector, fps = collect_clips(Path(args.video), args.action or action_type, min_frames=args.min_frames)
    mismatches = settings_mismatch(meta.get("settings"), collector.settings)
    if mismatches:
        print("注意：当前推理配置与建模板时不一致，相似度不可直接比较（关键点会变，切分边界也会变）：")
        for item in mismatches:
            print("  -", item)
    if not collector.clips:
        print("没有采集到有效动作序列。")
        print("最常见原因：视频里没有排球，或球检测不到——现有切分依赖球与手臂的距离和球的折返轨迹。")
        print("也可能是 --action 选错，或动作未完整入镜。")
        return 1
    print(f"模板: {args.template}（{template.shape[0]} 帧，来自 {meta.get('n_clips', '?')} 个动作）")
    print(f"视频: {args.video}（{fps:.1f} fps，采集 {len(collector.clips)} 个动作）")
    scores = []
    for i, clip in enumerate(collector.clips, 1):
        result = compare_matrices(clip_matrix(clip), template, tau=args.tau, band_ratio=args.band_ratio)
        scores.append(result["similarity"])
        print(f"  动作 {i}: 相似度 {result['similarity']}，距离 {result['dtw_distance']}，"
              f"时长比 {result['length_ratio']}（{result['student_frames']}/{result['template_frames']} 帧）")
        print(f"          偏差最大项: {'; '.join(_top_deviations(result['deviations']))}")
    print(f"平均相似度: {round(sum(scores) / len(scores), 1)}（{len(scores)} 个动作）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
