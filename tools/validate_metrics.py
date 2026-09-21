# -*- coding: utf-8 -*-
"""生物力学指标验证工具：系统评分/指标 vs 教练评分。

用法：
    # 1) 生成评分模板
    python tools/validate_metrics.py --template ratings.csv

    # 2) 填好模板（video 列写视频文件名，coach_score 写教练评分 1-10），再运行
    python tools/validate_metrics.py --videos "C:\\path\\to\\videos" --ratings ratings.csv --action dig

输出：
    validation_report.md   统计报告（Pearson / Spearman / ICC(2,1) / Bland-Altman）
    validation_metrics.csv 每个视频的系统评分与生物力学指标
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def pearson(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def rank(values):
    order = np.argsort(np.asarray(values, dtype=float))
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(1, len(values) + 1, dtype=float)
    return ranks


def spearman(x, y):
    return pearson(rank(x), rank(y))


def icc_2_1(matrix):
    """ICC(2,1)：双向随机效应、绝对一致性、单次测量。

    matrix: n 个受试者 × k 个评分者（这里 k=2：系统分、教练分）
    """
    data = np.asarray(matrix, dtype=float)
    n, k = data.shape
    if n < 2 or k < 2:
        return float("nan")
    grand = data.mean()
    row_means = data.mean(axis=1)
    col_means = data.mean(axis=0)
    ss_total = ((data - grand) ** 2).sum()
    ss_row = k * ((row_means - grand) ** 2).sum()
    ss_col = n * ((col_means - grand) ** 2).sum()
    ss_error = ss_total - ss_row - ss_col
    df_row, df_col, df_error = n - 1, k - 1, (n - 1) * (k - 1)
    if df_error <= 0:
        return float("nan")
    msr = ss_row / df_row
    msc = ss_col / df_col
    mse = ss_error / df_error
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    if denom == 0:
        return float("nan")
    return float((msr - mse) / denom)


def bland_altman(system, coach):
    diff = np.asarray(system, dtype=float) - np.asarray(coach, dtype=float)
    if len(diff) < 2:
        return float("nan"), float("nan"), float("nan")
    bias = float(diff.mean())
    sd = float(diff.std(ddof=1))
    return bias, bias - 1.96 * sd, bias + 1.96 * sd


def write_template(path: Path):
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh)
        writer.writerow(["video", "coach_score", "action"])
        writer.writerow(["example_dig_01.mp4", "8", "dig"])
    print("模板已生成:", path)


def main():
    parser = argparse.ArgumentParser(description="生物力学指标验证工具")
    parser.add_argument("--videos", help="视频目录")
    parser.add_argument("--ratings", help="教练评分 CSV")
    parser.add_argument("--action", default="dig", help="动作类型：dig/serve/set/spike")
    parser.add_argument("--template", help="生成评分模板 CSV 后退出")
    parser.add_argument("--out", default=str(ROOT / "validation"), help="输出目录")
    parser.add_argument("--limit", type=int, default=0, help="最多处理多少个视频（0=全部）")
    parser.add_argument("--system-max", type=float, default=100.0, help="系统评分满分（默认 100）")
    parser.add_argument("--coach-max", type=float, default=10.0, help="教练评分满分（默认 10）")
    args = parser.parse_args()

    if args.template:
        write_template(Path(args.template))
        return 0
    if not args.videos or not args.ratings:
        parser.error("需要 --videos 和 --ratings（或使用 --template 生成模板）")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with Path(args.ratings).open(encoding="utf-8-sig") as fh:
        for item in csv.DictReader(fh):
            if item.get("video") and item.get("coach_score"):
                rows.append(item)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("评分 CSV 没有有效行")
        return 1

    from analysis_service import AnalysisService
    service = AnalysisService()
    results = []
    for i, row in enumerate(rows, 1):
        video = Path(args.videos) / row["video"]
        if not video.exists():
            print(f"[{i}/{len(rows)}] 跳过（文件不存在）: {video}")
            continue
        print(f"[{i}/{len(rows)}] 分析 {video.name} ...", flush=True)
        try:
            res = service.analyze_video(video, row.get("action") or args.action, None,
                                        "指标验证", save=False, write_video=False)
        except Exception as exc:
            print("  失败:", exc)
            continue
        summary = res.get("summary") or {}
        bio = summary.get("biomechanics") or {}
        results.append({
            "video": row["video"],
            "coach_score": float(row["coach_score"]),
            "system_score": summary.get("average_score"),
            "action_count": summary.get("action_count"),
            "peak_knee_angular_velocity": bio.get("peak_knee_angular_velocity"),
            "peak_hip_angular_velocity": bio.get("peak_hip_angular_velocity"),
            "takeoff_com_velocity_bh": bio.get("takeoff_com_velocity_bh"),
            "jump_height_bh": bio.get("jump_height_bh"),
            "landing_knee_flexion_angle": bio.get("landing_knee_flexion_angle"),
            "min_knee_angle": bio.get("min_knee_angle"),
            "platform_angle_std": bio.get("platform_angle_std"),
        })

    if len(results) < 2:
        print("有效样本不足，无法统计")
        return 1

    system = [r["system_score"] or 0 for r in results]
    coach = [r["coach_score"] for r in results]
    # 系统分与教练分满分不同，ICC/Bland-Altman 前先换算到同一量纲
    system_scaled = [v / args.system_max * args.coach_max for v in system]
    report = []
    report.append("# 生物力学指标验证报告\n")
    report.append(f"- 样本量：{len(results)}\n- 动作：{args.action}\n")
    report.append(f"- 尺度换算：系统分 {args.system_max:g} 分制 → 教练 {args.coach_max:g} 分制\n")
    report.append(f"- 系统分 vs 教练分 Pearson r = **{pearson(system_scaled, coach):.3f}**\n")
    report.append(f"- 系统分 vs 教练分 Spearman ρ = **{spearman(system_scaled, coach):.3f}**\n")
    report.append(f"- ICC(2,1) 绝对一致性 = **{icc_2_1(np.column_stack([system_scaled, coach])):.3f}**\n")
    bias, loa_low, loa_high = bland_altman(system_scaled, coach)
    report.append(f"- Bland-Altman：偏差 {bias:.2f}（教练分制），一致性界限 [{loa_low:.2f}, {loa_high:.2f}]\n")
    report.append("\n## 各指标与教练评分的相关性\n\n")
    report.append("| 指标 | Pearson r |\n| --- | --- |\n")
    for key in results[0].keys():
        if key in ("video", "coach_score", "system_score"):
            continue
        values = [r[key] for r in results]
        pairs = [(v, c) for v, c in zip(values, coach) if v is not None]
        if len(pairs) >= 3:
            r = pearson([p[0] for p in pairs], [p[1] for p in pairs])
            report.append(f"| {key} | {r:.3f} |\n")
    report.append("\n## 逐样本数据\n\n")
    report.append("| 视频 | 教练分 | 系统分 | 峰值膝角速度 | 起跳重心速度 | 落地膝角 |\n| --- | --- | --- | --- | --- | --- |\n")
    for r in results:
        report.append(f"| {r['video']} | {r['coach_score']} | {r['system_score']} | "
                      f"{r['peak_knee_angular_velocity']} | {r['takeoff_com_velocity_bh']} | "
                      f"{r['landing_knee_flexion_angle']} |\n")

    (out_dir / "validation_report.md").write_text("".join(report), encoding="utf-8")
    with (out_dir / "validation_metrics.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print("\n报告:", out_dir / "validation_report.md")
    print("数据:", out_dir / "validation_metrics.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())