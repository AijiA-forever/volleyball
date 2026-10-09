# -*- coding: utf-8 -*-
"""绘制球-手臂距离随帧变化的折线图，用于分析动作分区行为。

输出两张图：
1. ball_distance_curves.png —— 每段视频一条折线（距离按身高归一化），标出进入阈值、
   现有状态机的活跃区间、外推补位帧，以及局部极小值（候选触球时刻）；
2. ball_distance_distribution.png —— 六段视频的距离分布箱线图，用于说明
   "同一个绝对阈值在不同视频里意味着完全不同的行为"。

用法：
    python tools/plot_ball_distance.py
    python tools/plot_ball_distance.py --videos data/templates_src/dig_side --out results/ball_distance
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from analyzer import BALL_IMGSZ, POSE_IMGSZ, ActionSession, VolleyballActionAnalyzer

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
ENTRY_THRESHOLD = 1.3     # 与 ActionSession.entry_threshold_multiplier 保持一致
MIN_GAP_FRAMES = 12       # 0.4 秒 @30fps：两次触球之间的最小间隔


def setup_font() -> bool:
    """设置中文字体；找不到时返回 False，图注回退英文，避免显示成方块。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "DengXian"):
        if name in available:
            matplotlib.rcParams["font.sans-serif"] = [name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    return False


ZH = {
    "title": "球-手臂距离（按身高归一化）随帧变化",
    "xlabel": "帧序号",
    "ylabel": "球-手臂距离 / 身高",
    "threshold": "进入阈值 1.30",
    "curve": "球-手臂距离",
    "predicted": "外推补位帧",
    "active": "现有状态机活跃区间",
    "minima": "局部极小值（候选触球）",
    "frames": "帧",
    "clips": "现有区间",
    "minima_n": "极小值",
    "dist_fig": "六段视频的球-手臂距离分布对比",
    "dist_ylabel": "球-手臂距离 / 身高",
    "subtitle": "推理配置：姿态 {pose} / 排球 {ball}；进入阈值 = {thr:g} 倍身高",
}
EN = {
    "title": "Ball-to-arm distance (normalized by body height)",
    "xlabel": "Frame index",
    "ylabel": "distance / body height",
    "threshold": "entry threshold 1.30",
    "curve": "distance",
    "predicted": "extrapolated frames",
    "active": "active intervals (current state machine)",
    "minima": "local minima (candidate contacts)",
    "frames": "frames",
    "clips": "clips",
    "minima_n": "minima",
    "dist_fig": "Distance distribution across the six videos",
    "dist_ylabel": "distance / body height",
    "subtitle": "config: pose {pose} / ball {ball}; entry threshold = {thr:g} x body height",
}


def local_minima(series: np.ndarray, min_gap: int = MIN_GAP_FRAMES, smooth: int = 5):
    """平滑后取局部极小值，贪心保留每个最小间隔窗口内最深的一个。"""
    if len(series) < 3:
        return []
    window = min(smooth, len(series))
    smoothed = np.convolve(series, np.ones(window) / window, mode="same")
    candidates = [i for i in range(1, len(smoothed) - 1)
                  if smoothed[i] <= smoothed[i - 1] and smoothed[i] < smoothed[i + 1]]
    kept: list = []
    for idx in candidates:
        if kept and idx - kept[-1] < min_gap:
            if smoothed[idx] < smoothed[kept[-1]]:
                kept[-1] = idx
        else:
            kept.append(idx)
    return kept


def fill_gaps(series: np.ndarray) -> np.ndarray:
    """缺失帧线性插值，保证极小值检测不被 NaN 打断。"""
    valid = np.isfinite(series)
    if valid.sum() < 2:
        return np.array([])
    x = np.arange(len(series))
    return np.interp(x, x[valid], series[valid])


def collect_series(analyzer, video: Path, action: str) -> dict:
    """跑一遍视频，记录逐帧的球-手臂距离、活跃状态与检测来源。"""
    import cv2

    analyzer.action_session = ActionSession(action)
    analyzer.reset_tracking()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    distances, active, predicted = [], [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        _annotated, _judgment, balls, kpts, info = analyzer.process_frame(frame, action)
        body_height = analyzer.get_body_height(kpts) if kpts else None
        distance = info.get("distance")
        distances.append(distance / body_height if (distance is not None and body_height) else float("nan"))
        active.append(bool(info.get("is_active")))
        predicted.append(bool(balls[0].get("is_predicted")) if balls else False)
    cap.release()
    return {
        "name": video.name,
        "fps": float(fps),
        "distance": np.array(distances),
        "active": np.array(active),
        "predicted": np.array(predicted),
        "clips": len(analyzer.action_session.action_scores),
        "minima": local_minima(fill_gaps(np.array(distances))),
    }


def plot_curves(data: list, out_png: Path, text: dict, subtitle: str) -> None:
    cols = 3
    rows = (len(data) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.6 * cols, 3.4 * rows), squeeze=False)
    peaks = [np.nanmax(d["distance"]) for d in data if np.isfinite(d["distance"]).any()]
    ymax = max(3.0, (max(peaks) if peaks else 3.0) * 1.05)

    for ax, item in zip(axes.flat, data):
        series = item["distance"]
        x = np.arange(len(series))
        valid = np.isfinite(series)

        active = item["active"] & valid
        if active.any():
            ax.fill_between(x, 0, ymax, where=active, color="#2ca02c", alpha=0.10,
                            step="mid", label=text["active"])
        ax.plot(x, series, color="#1f77b4", lw=1.2, label=text["curve"])
        if item["predicted"].any():
            mask = item["predicted"] & valid
            ax.scatter(x[mask], series[mask], s=7, c="#ff7f0e", marker=".",
                       label=text["predicted"], zorder=4)
        if item["minima"]:
            idx = np.array(item["minima"])
            ax.scatter(x[idx], series[idx], marker="v", s=42, c="#d62728",
                       edgecolors="white", linewidths=0.6, zorder=5, label=text["minima"])
        ax.axhline(ENTRY_THRESHOLD, ls="--", c="k", lw=1.0, label=text["threshold"])
        ax.set_ylim(0, ymax)
        ax.set_xlim(0, max(1, len(series)))
        ax.set_title(f"{item['name']}\n{len(series)} {text['frames']} · {item['clips']} {text['clips']} "
                     f"· {len(item['minima'])} {text['minima_n']}", fontsize=10)
        ax.set_xlabel(text["xlabel"], fontsize=9)
        ax.set_ylabel(text["ylabel"], fontsize=9)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.tick_params(labelsize=8)

    for ax in axes.flat[len(data):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=9, frameon=False)
    fig.suptitle(f"{text['title']}\n{subtitle}", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def plot_distribution(data: list, out_png: Path, text: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.4))
    series = [d["distance"][np.isfinite(d["distance"])] for d in data]
    labels = [d["name"].replace("dig_coach_side_", "").replace(".mp4", "") for d in data]
    bp = ax.boxplot(series, labels=labels, showmeans=True, patch_artist=True, widths=0.55)
    for patch in bp["boxes"]:
        patch.set_facecolor("#9ecae1")
        patch.set_alpha(0.85)
    ax.axhline(ENTRY_THRESHOLD, ls="--", c="k", lw=1.2, label=text["threshold"])
    ax.set_ylabel(text["dist_ylabel"], fontsize=10)
    ax.set_title(text["dist_fig"], fontsize=12)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="绘制球-手臂距离折线图")
    parser.add_argument("--videos", default=str(ROOT / "data" / "templates_src" / "dig_side"),
                        help="视频目录")
    parser.add_argument("--action", default="dig", help="动作类型")
    parser.add_argument("--out", default=str(ROOT / "results" / "ball_distance"), help="输出目录")
    parser.add_argument("--min-gap", type=int, default=MIN_GAP_FRAMES, help="两次极小值的最小间隔帧数")
    args = parser.parse_args()

    text = ZH if setup_font() else EN
    if text is EN:
        print("未找到中文字体，图注改用英文")

    video_dir = Path(args.videos)
    if not video_dir.is_dir():
        print("视频目录不存在:", video_dir)
        return 1
    videos = sorted(p for p in video_dir.iterdir() if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        print("目录里没有视频文件:", video_dir)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    analyzer = VolleyballActionAnalyzer(detect_volleyball=True)
    data = []
    for video in videos:
        print(f"分析 {video.name} ...", flush=True)
        item = collect_series(analyzer, video, args.action)
        item["minima"] = local_minima(fill_gaps(item["distance"]), min_gap=args.min_gap)
        data.append(item)

    subtitle = text["subtitle"].format(pose=POSE_IMGSZ, ball=BALL_IMGSZ, thr=ENTRY_THRESHOLD)
    curves_png = out_dir / "ball_distance_curves.png"
    dist_png = out_dir / "ball_distance_distribution.png"
    plot_curves(data, curves_png, text, subtitle)
    plot_distribution(data, dist_png, text)

    print(f"\n{'视频':<26}{'帧数':>6}{'有效帧':>7}{'最小':>7}{'中位':>7}{'最大':>7}{'现有区间':>9}{'极小值':>7}")
    for item in data:
        d = item["distance"][np.isfinite(item["distance"])]
        print(f"{item['name']:<26}{len(item['distance']):>6}{len(d):>7}{d.min():>7.2f}"
              f"{statistics.median(d):>7.2f}{d.max():>7.2f}{item['clips']:>9}{len(item['minima']):>7}")
    print("\n折线图:", curves_png)
    print("分布图:", dist_png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
