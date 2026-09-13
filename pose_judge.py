# 排球动作姿势判断模块
from body import calculate_angle
import numpy as np
import copy

# 动作角度阈值配置
# 垫球: 双臂伸直、膝盖弯曲、重心前倾
# 发球: 发球臂弯曲、手腕高于肩部、腿部微屈
# 传球: 双手举过头顶、肘部弯曲
# 扣球: 击球臂伸展、抬膝、身体伸展
ACTION_CRITERIA = {
    "dig": {
        "left_elbow": (160, 180),
        "right_elbow": (160, 180),
        "left_knee": (100, 150),
        "right_knee": (100, 150),
        "hip_height_ratio": (0.3, 0.7),  # 髋部高度/身高 比例（前倾特征）
        "weights": {"elbow": 0.4, "knee": 0.4, "hip": 0.2}  # 评分权重
    },
    # 发球：发球臂肘部100-160°、手腕高于肩部、腿部微屈
    "serve": {
        "right_elbow": (100, 160),  # 假设右手发球
        "right_wrist_shoulder": (1.05, 2.0),  # 手腕y坐标/肩部y坐标 < 1（更高）
        "right_knee": (120, 170),
        "weights": {"elbow": 0.5, "wrist": 0.3, "knee": 0.2}
    },
    # 传球：双手举过头顶（手腕高于肩部）、肘部弯曲70-140°
    "set": {
        "left_elbow": (70, 140),
        "right_elbow": (70, 140),
        "left_wrist_shoulder": (0.7, 1.1),  # 手腕更高
        "right_wrist_shoulder": (0.7, 1.1),
        "weights": {"elbow": 0.6, "wrist": 0.4}
    },
    # 扣球：击球臂肘部130-180°、抬膝（膝盖70-130°）、身体伸展
    "spike": {
        "right_elbow": (130, 180),  # 假设右手扣球
        "right_knee": (70, 130),
        "left_knee": (120, 170),
        "hip_shoulder_ratio": (1.0, 1.4),  # 髋部/肩部高度（身体伸展）
        "weights": {"elbow": 0.5, "knee": 0.4, "hip": 0.1}
    }
}

def get_criteria(action_type, override=None):
    """合并标准动作库参数。override 为空时使用内置默认值。"""
    base = ACTION_CRITERIA.get(action_type)
    if base is None:
        raise ValueError(f"不支持的动作类型：{action_type}")
    if not override:
        return copy.deepcopy(base)
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged

def calculate_height_ratio(kpt1, kpt2):
    """计算两个关键点的高度比例（y坐标比值）"""
    if kpt2[1] == 0:
        return 1.0
    return kpt1[1] / kpt2[1]

def calculate_horizontal_distance(kpt1, kpt2):
    """计算两个关键点的水平距离（x坐标差值）"""
    return abs(kpt1[0] - kpt2[0])

def calculate_euclidean_distance(kpt1, kpt2):
    """计算两个关键点的欧几里得距离"""
    return ((kpt1[0] - kpt2[0]) ** 2 + (kpt1[1] - kpt2[1]) ** 2) ** 0.5

def get_body_height(keypoints):
    """估算身高（肩部到脚踝的垂直距离）"""
    if "left_shoulder" in keypoints and "left_ankle" in keypoints:
        return abs(keypoints["left_shoulder"][1] - keypoints["left_ankle"][1])
    return 1.0

def is_side_position(keypoints):
    """判断是否为侧身姿势"""
    # 计算两侧手臂的肘关节和腕关节水平距离
    # 使用手腕到手肘的距离除以 sin(36.9°)（≈ 0.6），标准更高
    # 即 side_threshold = distance(wrist, elbow) / 0.6

    def check_side_for_arm(elbow, wrist):
        if elbow and wrist:
            arm_length = calculate_euclidean_distance(elbow, wrist)
            side_threshold = arm_length / 0.6  # sin(36.9°) ≈ 0.6，标准更高
            horizontal_distance = calculate_horizontal_distance(elbow, wrist)
            return horizontal_distance < side_threshold
        return False

    # 检查右侧手臂
    if "right_elbow" in keypoints and "right_wrist" in keypoints:
        if check_side_for_arm(keypoints["right_elbow"], keypoints["right_wrist"]):
            return True

    # 检查左侧手臂
    if "left_elbow" in keypoints and "left_wrist" in keypoints:
        if check_side_for_arm(keypoints["left_elbow"], keypoints["left_wrist"]):
            return True

    return False

# ===================== 2.1 指导建议生成函数 =====================
def generate_feedback(score_details, action_type):
    """
    根据评分详情生成简洁的指导建议
    :param score_details: 评分详情字典
    :param action_type: 动作类型
    :return: 指导建议列表
    """
    feedback = []

    if not score_details:
        feedback.append("动作基本标准，继续保持！")
        return feedback

    if action_type == "dig":
        elbow_scores = []
        knee_scores = []
        if "left_elbow_angle" in score_details and "score" in score_details["left_elbow_angle"]:
            elbow_scores.append(score_details["left_elbow_angle"]["score"])
        if "right_elbow_angle" in score_details and "score" in score_details["right_elbow_angle"]:
            elbow_scores.append(score_details["right_elbow_angle"]["score"])
        if "left_knee_angle" in score_details and "score" in score_details["left_knee_angle"]:
            knee_scores.append(score_details["left_knee_angle"]["score"])
        if "right_knee_angle" in score_details and "score" in score_details["right_knee_angle"]:
            knee_scores.append(score_details["right_knee_angle"]["score"])

        if elbow_scores and sum(elbow_scores) / len(elbow_scores) < 80:
            feedback.append("保持双臂伸直，肘部不要弯曲")
        if knee_scores and sum(knee_scores) / len(knee_scores) < 80:
            feedback.append("膝盖适当弯曲，降低重心")
        if "hip_height_ratio" in score_details and "score" in score_details["hip_height_ratio"]:
            if score_details["hip_height_ratio"]["score"] < 80:
                feedback.append("身体前倾，保持重心稳定")

    elif action_type == "serve":
        if "right_elbow_angle" in score_details and "score" in score_details["right_elbow_angle"]:
            if score_details["right_elbow_angle"]["score"] < 80:
                feedback.append("发球臂肘部角度保持在100-160度之间")
        if "right_wrist_shoulder" in score_details and "score" in score_details["right_wrist_shoulder"]:
            if score_details["right_wrist_shoulder"]["score"] < 80:
                feedback.append("确保手腕高于肩部")
        if "right_knee_angle" in score_details and "score" in score_details["right_knee_angle"]:
            if score_details["right_knee_angle"]["score"] < 80:
                feedback.append("腿部微屈，保持身体平衡")

    elif action_type == "set":
        elbow_scores = []
        wrist_scores = []
        if "left_elbow_angle" in score_details and "score" in score_details["left_elbow_angle"]:
            elbow_scores.append(score_details["left_elbow_angle"]["score"])
        if "right_elbow_angle" in score_details and "score" in score_details["right_elbow_angle"]:
            elbow_scores.append(score_details["right_elbow_angle"]["score"])
        if "left_wrist_shoulder" in score_details and "score" in score_details["left_wrist_shoulder"]:
            wrist_scores.append(score_details["left_wrist_shoulder"]["score"])
        if "right_wrist_shoulder" in score_details and "score" in score_details["right_wrist_shoulder"]:
            wrist_scores.append(score_details["right_wrist_shoulder"]["score"])

        if elbow_scores and sum(elbow_scores) / len(elbow_scores) < 80:
            feedback.append("肘部弯曲角度保持在70-140度之间")
        if wrist_scores and sum(wrist_scores) / len(wrist_scores) < 80:
            feedback.append("双手举过头顶，手腕高于肩部")

    elif action_type == "spike":
        if "right_elbow_angle" in score_details and "score" in score_details["right_elbow_angle"]:
            if score_details["right_elbow_angle"]["score"] < 80:
                feedback.append("击球臂肘部角度保持在130-180度之间")
        if "right_knee_angle" in score_details and "score" in score_details["right_knee_angle"]:
            if score_details["right_knee_angle"]["score"] < 80:
                feedback.append("右侧膝盖适当弯曲，做好起跳准备")
        if "hip_shoulder_ratio" in score_details and "score" in score_details["hip_shoulder_ratio"]:
            if score_details["hip_shoulder_ratio"]["score"] < 80:
                feedback.append("身体充分伸展，提高击球点")

    # 如果没有具体建议，给出通用建议
    if not feedback:
        feedback.append("动作基本标准，继续保持！")

    return feedback

# ===================== 3. 核心姿势判断函数 =====================
def judge_pose(keypoints, action_type, criteria=None):
    """
    判断指定动作的姿势是否标准
    :param keypoints: 人体关键点字典 {kpt_name: (x, y)}
    :param action_type: 动作类型（dig/serve/set/spike）
    :return: 评分（0-100）、判断结果（True/False）、详细分析
    """
    if action_type not in ACTION_CRITERIA:
        raise ValueError(f"不支持的动作类型：{action_type}，可选：dig/serve/set/spike")

    # 判断是否为侧身姿势
    is_side = is_side_position(keypoints)
    print(f"Position: {'Side' if is_side else 'Front'}")

    criteria = get_criteria(action_type, criteria)
    weights = criteria["weights"]
    score_details = {}
    total_score = 0.0

    # 1. 垫球判断
    if action_type == "dig":
        # 肘部角度评分
        for side in ["left", "right"]:
            angle = calculate_angle(
                keypoints[f"{side}_shoulder"],
                keypoints[f"{side}_elbow"],
                keypoints[f"{side}_wrist"]
            )
            # 根据正侧身使用不同的角度阈值
            if is_side:
                # 侧身时的肘部角度阈值（更宽松）
                min_angle, max_angle = (120, 180)
            else:
                # 正面时的肘部角度阈值
                min_angle, max_angle = criteria[f"{side}_elbow"]
            score = 100 if (min_angle <= angle <= max_angle) else max(0, 100 - abs(angle - (min_angle+max_angle)/2)*1)
            score_details[f"{side}_elbow_angle"] = {"angle": float(angle), "score": float(score)}

        # 膝盖角度评分
        for side in ["left", "right"]:
            angle = calculate_angle(
                keypoints[f"{side}_hip"],
                keypoints[f"{side}_knee"],
                keypoints[f"{side}_ankle"]
            )
            # 根据正侧身使用不同的角度阈值
            if is_side:
                # 侧身时的膝盖角度阈值（更宽松）
                min_angle, max_angle = (70, 160)
            else:
                # 正面时的膝盖角度阈值
                min_angle, max_angle = criteria[f"{side}_knee"]
            score = 100 if (min_angle <= angle <= max_angle) else max(0, 100 - abs(angle - (min_angle+max_angle)/2)*1)
            score_details[f"{side}_knee_angle"] = {"angle": float(angle), "score": float(score)}

        # 髋部高度比例（前倾）
        # 前倾判断：计算髋部中点相对于肩部中点的垂直偏移量，除以身高得到比例
        # 在图像坐标系中，y值越大表示位置越低
        # 前倾时髋部应该在肩部下方，即 hip_y > shoulder_y
        # hip_shoulder_diff > 0 表示髋部在肩部下方
        shoulder_center_y = (keypoints["left_shoulder"][1] + keypoints["right_shoulder"][1]) / 2
        hip_center_y = (keypoints["left_hip"][1] + keypoints["right_hip"][1]) / 2
        ankle_center_y = (keypoints["left_ankle"][1] + keypoints["right_ankle"][1]) / 2

        # 计算躯干长度（肩到髋）
        torso_length = abs(hip_center_y - shoulder_center_y)
        # 计算腿长（髋到踝）
        leg_length = abs(ankle_center_y - hip_center_y)
        # 计算身体总高度（躯干 + 腿）
        body_height = torso_length + leg_length if leg_length > 0 else 1.0

        # 前倾比例：躯干长度 / 总高度，表示身体前倾的程度
        # 值越大表示躯干越长，身体越前倾
        if body_height > 0:
            hip_height_ratio = torso_length / body_height
        else:
            hip_height_ratio = 0.5

        # 计算髋部相对于肩部的垂直偏移（躯干占身体的比例）
        # 前倾时躯干占比应该在 0.3 到 0.55 之间（垫球姿势）
        # 躯干太短（占比小）说明直立，躯干太长（占比大）说明过度前倾
        if is_side:
            # 侧身时的阈值（更宽松），侧身姿势前倾判断不太准确
            min_ratio, max_ratio = (0.25, 0.6)
        else:
            # 正面时的髋部高度比例阈值
            min_ratio, max_ratio = criteria["hip_height_ratio"]

        # 计算得分
        if min_ratio <= hip_height_ratio <= max_ratio:
            score = 100
        else:
            # 偏离区间越远，得分越低
            ideal_ratio = (min_ratio + max_ratio) / 2
            deviation = abs(hip_height_ratio - ideal_ratio)
            score = max(0, 100 - deviation * 300)

        score_details["hip_height_ratio"] = {"ratio": float(hip_height_ratio), "score": float(score)}

        # 添加侧身判断结果
        score_details["position"] = "side" if is_side else "front"

        # 加权总分
        elbow_avg = (score_details["left_elbow_angle"]["score"] + score_details["right_elbow_angle"]["score"]) / 2
        knee_avg = (score_details["left_knee_angle"]["score"] + score_details["right_knee_angle"]["score"]) / 2
        total_score = elbow_avg * weights["elbow"] + knee_avg * weights["knee"] + score_details["hip_height_ratio"]["score"] * weights["hip"]

    # 2. 发球判断（右手发球为例）
    elif action_type == "serve":
        # 发球臂肘部角度
        elbow_angle = calculate_angle(
            keypoints["right_shoulder"],
            keypoints["right_elbow"],
            keypoints["right_wrist"]
        )
        # 根据正侧身使用不同的角度阈值
        if is_side:
            # 侧身时的肘部角度阈值（更宽松）
            min_angle, max_angle = (90, 170)
        else:
            # 正面时的肘部角度阈值
            min_angle, max_angle = criteria["right_elbow"]
        elbow_score = 100 if (min_angle <= elbow_angle <= max_angle) else max(0, 100 - abs(elbow_angle - (min_angle+max_angle)/2)*1)
        score_details["right_elbow_angle"] = {"angle": float(elbow_angle), "score": float(elbow_score)}

        # 手腕高于肩部
        wrist_shoulder_ratio = calculate_height_ratio(keypoints["right_wrist"], keypoints["right_shoulder"])
        min_ratio, max_ratio = criteria["right_wrist_shoulder"]
        wrist_score = 100 if (wrist_shoulder_ratio <= min_ratio) else max(0, 100 - abs(wrist_shoulder_ratio - min_ratio)*50)
        score_details["right_wrist_shoulder"] = {"ratio": float(wrist_shoulder_ratio), "score": float(wrist_score)}

        # 腿部微屈
        knee_angle = calculate_angle(
            keypoints["right_hip"],
            keypoints["right_knee"],
            keypoints["right_ankle"]
        )
        # 根据正侧身使用不同的角度阈值
        if is_side:
            # 侧身时的膝盖角度阈值（更宽松）
            min_angle, max_angle = (110, 180)
        else:
            # 正面时的膝盖角度阈值
            min_angle, max_angle = criteria["right_knee"]
        knee_score = 100 if (min_angle <= knee_angle <= max_angle) else max(0, 100 - abs(knee_angle - (min_angle+max_angle)/2)*1)
        score_details["right_knee_angle"] = {"angle": float(knee_angle), "score": float(knee_score)}

        # 添加侧身判断结果
        score_details["position"] = "side" if is_side else "front"

        # 加权总分
        total_score = elbow_score * weights["elbow"] + wrist_score * weights["wrist"] + knee_score * weights["knee"]

    # 3. 传球判断
    elif action_type == "set":
        # 肘部角度
        elbow_scores = []
        for side in ["left", "right"]:
            angle = calculate_angle(
                keypoints[f"{side}_shoulder"],
                keypoints[f"{side}_elbow"],
                keypoints[f"{side}_wrist"]
            )
            # 根据正侧身使用不同的角度阈值
            if is_side:
                # 侧身时的肘部角度阈值（更宽松）
                min_angle, max_angle = (60, 150)
            else:
                # 正面时的肘部角度阈值
                min_angle, max_angle = criteria[f"{side}_elbow"]
            score = 100 if (min_angle <= angle <= max_angle) else max(0, 100 - abs(angle - (min_angle+max_angle)/2)*1)
            score_details[f"{side}_elbow_angle"] = {"angle": float(angle), "score": float(score)}
            elbow_scores.append(score)
        elbow_avg = sum(elbow_scores) / len(elbow_scores)

        # 手腕高度
        wrist_scores = []
        for side in ["left", "right"]:
            ratio = calculate_height_ratio(keypoints[f"{side}_wrist"], keypoints[f"{side}_shoulder"])
            min_ratio, max_ratio = criteria[f"{side}_wrist_shoulder"]
            score = 100 if (ratio <= max_ratio) else max(0, 100 - abs(ratio - max_ratio)*50)
            score_details[f"{side}_wrist_shoulder"] = {"ratio": float(ratio), "score": float(score)}
            wrist_scores.append(score)
        wrist_avg = sum(wrist_scores) / len(wrist_scores)

        # 添加侧身判断结果
        score_details["position"] = "side" if is_side else "front"

        # 加权总分
        total_score = elbow_avg * weights["elbow"] + wrist_avg * weights["wrist"]

    # 4. 扣球判断
    elif action_type == "spike":
        # 击球臂肘部角度
        elbow_angle = calculate_angle(
            keypoints["right_shoulder"],
            keypoints["right_elbow"],
            keypoints["right_wrist"]
        )
        # 根据正侧身使用不同的角度阈值
        if is_side:
            # 侧身时的肘部角度阈值（更宽松）
            min_angle, max_angle = (120, 190)
        else:
            # 正面时的肘部角度阈值
            min_angle, max_angle = criteria["right_elbow"]
        elbow_score = 100 if (min_angle <= elbow_angle <= max_angle) else max(0, 100 - abs(elbow_angle - (min_angle+max_angle)/2)*1)
        score_details["right_elbow_angle"] = {"angle": float(elbow_angle), "score": float(elbow_score)}

        # 膝盖角度
        right_knee_angle = calculate_angle(
            keypoints["right_hip"],
            keypoints["right_knee"],
            keypoints["right_ankle"]
        )
        left_knee_angle = calculate_angle(
            keypoints["left_hip"],
            keypoints["left_knee"],
            keypoints["left_ankle"]
        )
        # 根据正侧身使用不同的角度阈值
        if is_side:
            # 侧身时的膝盖角度阈值（更宽松）
            min_r, max_r = (60, 140)
            min_l, max_l = (110, 180)
        else:
            # 正面时的膝盖角度阈值
            min_r, max_r = criteria["right_knee"]
            min_l, max_l = criteria["left_knee"]
        right_knee_score = 100 if (min_r <= right_knee_angle <= max_r) else max(0, 100 - abs(right_knee_angle - (min_r+max_r)/2)*1)
        left_knee_score = 100 if (min_l <= left_knee_angle <= max_l) else max(0, 100 - abs(left_knee_angle - (min_l+max_l)/2)*1)
        score_details["right_knee_angle"] = {"angle": float(right_knee_angle), "score": float(right_knee_score)}
        score_details["left_knee_angle"] = {"angle": float(left_knee_angle), "score": float(left_knee_score)}
        knee_avg = (right_knee_score + left_knee_score) / 2

        # 身体伸展（髋肩比例）
        hip_shoulder_ratio = calculate_height_ratio(keypoints["right_hip"], keypoints["right_shoulder"])
        # 根据正侧身使用不同的比例阈值
        if is_side:
            # 侧身时的比例阈值（更宽松）
            min_ratio, max_ratio = (0.9, 1.5)
        else:
            # 正面时的比例阈值
            min_ratio, max_ratio = criteria["hip_shoulder_ratio"]
        hip_score = 100 if (min_ratio <= hip_shoulder_ratio <= max_ratio) else max(0, 100 - abs(hip_shoulder_ratio - (min_ratio+max_ratio)/2)*100)
        score_details["hip_shoulder_ratio"] = {"ratio": float(hip_shoulder_ratio), "score": float(hip_score)}

        # 添加侧身判断结果
        score_details["position"] = "side" if is_side else "front"

        # 加权总分
        total_score = elbow_score * weights["elbow"] + knee_avg * weights["knee"] + hip_score * weights["hip"]

    # 最终结果
    is_correct = total_score >= 80  # 80分以上判定为标准姿势
    feedback = generate_feedback(score_details, action_type)
    return {
        "action_type": action_type,
        "total_score": float(total_score),
        "is_correct": is_correct,
        "score_details": score_details,
        "feedback": feedback,
        "judgment": "Standard" if is_correct else "Not Standard"
    }