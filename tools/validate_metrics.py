# -*- coding: utf-8 -*-
"""生物力学指标验证工具：系统评分/指标 vs 教练评分。

用法：
    python tools/validate_metrics.py --template ratings.csv
    python tools/validate_metrics.py --videos "D:\训练视频" --ratings ratings.csv --action dig

评分 CSV 支持：
- 每行一个"视频 × 教练"的评分；
- 总分列名可以是 overall_score 或 coach_score；
- 可选分项列：prep_score / contact_score / follow_score /
  knee_flexion_score / trunk_lean_score / arm_extension_score / landing_buffer_score；
- 同一视频多名教练时，系统会自动取平均，并计算教练之间的 ICC（inter-rater）。

输出：
    validation/validation_report.md
    validation/validation_metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SUB_KEYS = [
    "prep_score", "contact_score", "follow_score",
    "knee_flexion_score", "trunk_lean_score", "arm_extension_score", "landing_buffer_score",
]
BIO_KEYS = [
    "peak_knee_angular_velocity", "peak_hip_angular_velocity", "peak_elbow_angular_velocity",
    "takeoff_com_velocity_bh", "jump_height_bh",
    "landing_knee_flexion_angle", "landing_knee_flexion_velocity",
    "min_knee_angle", "max_trunk_inclination", "platform_angle_std",
]
TEMPLATE_HEADER = ["video", "student", "action", "cam_view", "coach_id", "reps", "overall_score",
                   "standard_flag"] + SUB_KEYS + ["notes"]


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
    """ICC(2,1)：双向随机效应、绝对一致性、单次测量。matrix = n 个受试者 × k 个评分者。"""
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
    msr, msc, mse = ss_row / df_row, ss_col / df_col, ss_error / df_error
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / denom) if denom else float("nan")


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
        writer.writerow(TEMPLATE_HEADER)
        writer.writerow(["dig_S01_side_01.mp4", "S01", "dig", "side", "coachA", "3", "8", "1",
                         "4", "4", "4", "3", "4", "4", "4", "准备稍慢"])
        writer.writerow(["dig_S01_side_01.mp4", "S01", "dig", "side", "coachB", "3", "7", "1",
                         "4", "4", "3", "3", "4", "4", "4", "手臂可以更稳"])
    print("模板已生成:", path)


def load_ratings(path: Path):
    """按视频聚合：多名教练取平均，分项分取平均。"""
    groups = {}
    with path.open(encoding="utf-8-sig") as fh:
        for item in csv.DictReader(fh):
            video = (item.get("video") or "").strip()
            score = item.get("overall_score") or item.get("coach_score")
            if not video or score in (None, ""):
                continue
            group = groups.setdefault(video, {
                "action": (item.get("action") or "").strip(),
                "scores": [],
                "coach_ids": [],
                "sub": {k: [] for k in SUB_KEYS},
            })
            try:
                group["scores"].append(float(score))
            except ValueError:
                continue
            group["coach_ids"].append((item.get("coach_id") or "coach").strip())
            for key in SUB_KEYS:
                value = item.get(key)
                if value not in (None, ""):
                    try:
                        group["sub"][key].append(float(value))
                    except ValueError:
                        pass
    return groups


def mean_or_none(values):
    return round(sum(values) / len(values), 2) if values else None


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

    groups = load_ratings(Path(args.ratings))
    if args.limit:
        groups = dict(list(groups.items())[: args.limit])
    if not groups:
        print("评分 CSV 没有有效行")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from analysis_service import AnalysisService
    service = AnalysisService()
    results, inter_pairs = [], []
    for i, (video_name, group) in enumerate(groups.items(), 1):
        video = Path(args.videos) / video_name
        if not video.exists():
            print(f"[{i}/{len(groups)}] 跳过（文件不存在）: {video}")
            continue
        print(f"[{i}/{len(groups)}] 分析 {video_name} ...", flush=True)
        try:
            res = service.analyze_video(video, group["action"] or args.action, None,
                                        "指标验证", save=False, write_video=False)
        except Exception as exc:
            print("  失败:", exc)
            continue
        summary = res.get("summary") or {}
        bio = summary.get("biomechanics") or {}
        row = {
            "video": video_name,
            "coach_score": mean_or_none(group["scores"]),
            "coach_n": len(group["scores"]),
            "system_score": summary.get("average_score"),
            "action_count": summary.get("action_count"),
        }
        for key in SUB_KEYS:
            row[key] = mean_or_none(group["sub"][key])
        for key in BIO_KEYS:
            row[key] = bio.get(key)
        results.append(row)
        if len(group["scores"]) >= 2:
            inter_pairs.append(group["scores"][:2])

    if len(results) < 2:
        print("有效样本不足，无法统计")
        return 1

    system = [r["system_score"] or 0 for r in results]
    coach = [r["coach_score"] or 0 for r in results]
    system_scaled = [v / args.system_max * args.coach_max for v in system]

    report = ["# 生物力学指标验证报告\n\n"]
    report.append(f"- 样本量：{len(results)}\n- 动作：{args.action}\n")
    report.append(f"- 尺度换算：系统分 {args.system_max:g} 分制 → 教练 {args.coach_max:g} 分制\n")
    report.append(f"- 系统分 vs 教练分 Pearson r = **{pearson(system_scaled, coach):.3f}**\n")
    report.append(f"- 系统分 vs 教练分 Spearman ρ = **{spearman(system_scaled, coach):.3f}**\n")
    report.append(f"- ICC(2,1) 绝对一致性 = **{icc_2_1(np.column_stack([system_scaled, coach])):.3f}**\n")
    bias, loa_low, loa_high = bland_altman(system_scaled, coach)
    report.append(f"- Bland-Altman：偏差 {bias:.2f}（教练分制），一致性界限 [{loa_low:.2f}, {loa_high:.2f}]\n")
    if len(inter_pairs) >= 3:
        report.append(f"- 教练之间 ICC(2,1)（{len(inter_pairs)} 个视频 × 2 名教练）= **{icc_2_1(np.array(inter_pairs, dtype=float)):.3f}**\n")

    report.append("\n## 各指标与教练评分的相关性\n\n| 指标 | Pearson r |\n| --- | --- |\n")
    for key in BIO_KEYS + SUB_KEYS:
        values = [r.get(key) for r in results]
        pairs = [(v, c) for v, c in zip(values, coach) if v is not None]
        if len(pairs) >= 3:
            report.append(f"| {key} | {pearson([p[0] for p in pairs], [p[1] for p in pairs]):.3f} |\n")

    report.append("\n## 逐样本数据\n\n| 视频 | 教练分 | 系统分 | 峰值膝角速度 | 起跳重心速度 | 落地膝角 |\n| --- | --- | --- | --- | --- | --- |\n")
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