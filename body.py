from ultralytics import YOLO
import cv2
import numpy as np
import warnings
import os
import torch
from PIL import Image, ImageDraw, ImageFont
warnings.filterwarnings('ignore')

# 加载中文字体
def get_chinese_font():
    """
    获取中文字体路径
    """
    # 尝试不同的中文字体路径
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",  # 黑体
        "C:/Windows/Fonts/simsun.ttc",  # 宋体
        "C:/Windows/Fonts/msyh.ttf",   # 微软雅黑
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # Linux
        "/Library/Fonts/Songti.ttc"  # macOS
    ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            return font_path
    return None

CHINESE_FONT = get_chinese_font()

def put_chinese_texts(img, items):
    """一次 PIL 往返绘制多条中文文本。

    :param items: [(text, position, font_size, color), ...]
    逐条调用 put_chinese_text 会对整帧做多次 BGR↔RGB 转换，逐帧开销明显；
    把同一帧的所有中文合并成一次转换可省掉这部分重复开销。
    """
    if not items:
        return img
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    for text, position, font_size, color in items:
        if CHINESE_FONT:
            try:
                font = ImageFont.truetype(CHINESE_FONT, font_size)
            except Exception:
                font = ImageFont.load_default()
        else:
            font = ImageFont.load_default()
        draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def put_chinese_text(img, text, position, font_size=12, color=(255, 255, 0)):
    """
    在图像上绘制中文字符
    :param img: OpenCV图像
    :param text: 要绘制的文字
    :param position: 文字位置 (x, y)
    :param font_size: 字体大小
    :param color: 文字颜色 (B, G, R)
    :return: 绘制后的图像
    """
    return put_chinese_texts(img, [(text, position, font_size, color)])

# ===================== 核心配置 =====================
# YOLOv8-pose关键点名称与索引映射（固定顺序）
POSE_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]
# 排球动作分析核心关键点索引
CORE_KEYPOINTS = {
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16
}

# ===================== 工具函数 =====================
def tensor_to_numpy(tensor):
    """将 torch.Tensor / tuple / list 转换为 numpy 数组"""
    if isinstance(tensor, np.ndarray):
        return tensor
    if isinstance(tensor, (tuple, list)):
        return np.asarray(tensor, dtype=np.float32)
    if hasattr(tensor, "cpu") and hasattr(tensor, "numpy"):
        return tensor.cpu().numpy() if getattr(tensor, "is_cuda", False) else tensor.numpy()
    return np.asarray(tensor)

def calculate_angle(p1, p2, p3):
    """计算三个关键点形成的夹角（p2为顶点），适配肢体角度分析"""
    p1 = tensor_to_numpy(p1)
    p2 = tensor_to_numpy(p2)
    p3 = tensor_to_numpy(p3)
    
    v1 = p1 - p2
    v2 = p3 - p2
    
    # 计算余弦值并限制范围（避免数值误差）
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cos_angle))
    return angle

def setup_display_window(window_name, width, height, scale=1.0):
    """Create a resizable OpenCV window and apply a scale factor."""
    scale = max(float(scale), 0.1)
    disp_w = max(1, int(width * scale))
    disp_h = max(1, int(height * scale))
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, disp_w, disp_h)

def draw_pose_results(image, results, pose_judgment=None):
    """姿态估计结果可视化：新增腿部角度+统一字体样式+姿势判断结果显示"""
    img_copy = image.copy()
    h, w = img_copy.shape[:2]
    
    # 自定义单位像素值：基于帧高度的缩放因子（以960p为基准）
    scale_factor = h / 960.0

    # 骨骼连线（完整上下身）
    skeleton_pairs = [
        # 上身
        (5, 7), (7, 9),    # left shoulder-elbow-wrist
        (6, 8), (8, 10),   # right shoulder-elbow-wrist
        (5, 6),            # shoulders
        (5, 11), (6, 12),  # shoulder-hip
        (11, 12),          # hips
        # 下身
        (11, 13), (13, 15),# left hip-knee-ankle
        (12, 14), (14, 16) # right hip-knee-ankle
    ]

    if results[0].keypoints is None:
        return img_copy

    def to_pixel(x, y):
        # 归一化坐标转像素坐标
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            return int(x * w), int(y * h)
        return int(x), int(y)

    # 存储需要显示的文本（按人分开）
    person_texts = {}  # key: person_idx, value: list of texts
    # 统一字体样式配置（使用缩放因子）
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7 * scale_factor       # 与角度显示一致
    font_thickness = max(1, int(2 * scale_factor))     # 与角度显示一致
    text_color = (0, 0, 255)  # 红色
    text_line_height = int(30 * scale_factor)  # 行高（与角度显示间隔一致）
    
    # ========== 按左肩x坐标从左到右排序人物 ==========
    # 左肩索引为5（在POSE_KEYPOINTS中）
    left_shoulder_idx = 5
    keypoints_data = results[0].keypoints.data
    
    # 创建带左肩x坐标的列表
    persons_with_x = []
    for person_idx, keypoints in enumerate(keypoints_data):
        kpts = tensor_to_numpy(keypoints)
        if len(kpts) > left_shoulder_idx:
            x, y, conf = kpts[left_shoulder_idx]
            if conf > 0.3:
                # 使用归一化x坐标进行排序
                persons_with_x.append((float(x), person_idx, keypoints))
            else:
                # 如果左肩检测不可靠，使用其他可见点的平均x坐标
                valid_x = []
                for i, (kx, ky, kconf) in enumerate(kpts):
                    if kconf > 0.3:
                        valid_x.append(float(kx))
                if valid_x:
                    avg_x = sum(valid_x) / len(valid_x)
                    persons_with_x.append((avg_x, person_idx, keypoints))
                else:
                    persons_with_x.append((float('inf'), person_idx, keypoints))
        else:
            persons_with_x.append((float('inf'), person_idx, keypoints))
    
    # 按左肩x坐标从小到大排序（从左到右）
    persons_with_x.sort(key=lambda x: x[0])
    
    # 遍历排序后的人物
    for sorted_idx, (shoulder_x, original_idx, keypoints) in enumerate(persons_with_x):
        keypoints = tensor_to_numpy(keypoints)
        person_label = f"person{sorted_idx + 1}"  # 按排序后的顺序标注person1/person2...
        person_texts[sorted_idx] = []  # 初始化当前人的文本列表

        valid_kpts = {}
        # 提取有效点位置
        for kpt_name, idx in CORE_KEYPOINTS.items():
            x, y, conf = keypoints[idx]
            if conf > 0.3:
                px, py = to_pixel(float(x), float(y))
                valid_kpts[kpt_name] = (px, py)
                # 按格式记录关键点位置（加入当前人的显示列表）
                person_texts[sorted_idx].append(f"{person_label} {kpt_name}: ({px}, {py})")

        # 绘制关键点（使用缩放因子）
        kpt_radius = max(2, int(6 * scale_factor))
        kpt_thickness = max(1, int(1 * scale_factor))
        for x, y, conf in keypoints:
            if conf > 0.3:
                px, py = to_pixel(float(x), float(y))
                cv2.circle(img_copy, (px, py), kpt_radius, (0, 255, 0), -1)
                cv2.circle(img_copy, (px, py), kpt_radius, (0, 0, 0), kpt_thickness)

        # 绘制骨骼连线（使用缩放因子）
        line_thickness = max(1, int(4 * scale_factor))
        for a, b in skeleton_pairs:
            if a < len(keypoints) and b < len(keypoints):
                x1, y1, c1 = keypoints[a]
                x2, y2, c2 = keypoints[b]
                if c1 > 0.3 and c2 > 0.3:
                    p1 = to_pixel(float(x1), float(y1))
                    p2 = to_pixel(float(x2), float(y2))
                    cv2.line(img_copy, p1, p2, (0, 215, 255), line_thickness)

        # ========== 1. 手臂角度计算（原有） ==========
        if all(k in valid_kpts for k in ["right_shoulder", "right_elbow", "right_wrist"]):
            r_arm_angle = calculate_angle(
                valid_kpts["right_shoulder"],
                valid_kpts["right_elbow"],
                valid_kpts["right_wrist"]
            )
            arm_text = f"{person_label} Right arm: {r_arm_angle:.0f}"
            person_texts[sorted_idx].insert(0, arm_text)  # 角度显示在关键点上方

        if all(k in valid_kpts for k in ["left_shoulder", "left_elbow", "left_wrist"]):
            l_arm_angle = calculate_angle(
                valid_kpts["left_shoulder"],
                valid_kpts["left_elbow"],
                valid_kpts["left_wrist"]
            )
            arm_text = f"{person_label} Left arm: {l_arm_angle:.0f}"
            person_texts[sorted_idx].insert(1, arm_text)

        # ========== 2. 新增：腿部角度计算 ==========
        if all(k in valid_kpts for k in ["right_hip", "right_knee", "right_ankle"]):
            r_leg_angle = calculate_angle(
                valid_kpts["right_hip"],
                valid_kpts["right_knee"],
                valid_kpts["right_ankle"]
            )
            leg_text = f"{person_label} Right leg: {r_leg_angle:.0f}"
            person_texts[sorted_idx].insert(2, leg_text)

        if all(k in valid_kpts for k in ["left_hip", "left_knee", "left_ankle"]):
            l_leg_angle = calculate_angle(
                valid_kpts["left_hip"],
                valid_kpts["left_knee"],
                valid_kpts["left_ankle"]
            )
            leg_text = f"{person_label} Left leg: {l_leg_angle:.0f}"
            person_texts[sorted_idx].insert(3, leg_text)

    # ========== 分别显示：person1在左上角，person2在右上角 ==========
    margin = int(10 * scale_factor)
    
    # person1（索引0）显示在左上角
    if 0 in person_texts:
        text_y = int(20 * scale_factor)
        for text in person_texts[0]:
            cv2.putText(img_copy,
                        text,
                        (margin, text_y),
                        font,
                        font_scale,
                        text_color,
                        font_thickness)
            text_y += text_line_height
    
    # person2（索引1）显示在右上角
    if 1 in person_texts:
        text_y = int(20 * scale_factor)
        for text in person_texts[1]:
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = w - text_size[0] - margin
            cv2.putText(img_copy,
                        text,
                        (text_x, text_y),
                        font,
                        font_scale,
                        text_color,
                        font_thickness)
            text_y += text_line_height

    # ========== 新增：绘制姿势判断结果（左下角） ==========
    if pose_judgment:
        # 配置判断结果的字体样式（与角度/关键点保持视觉统一）
        judge_font = cv2.FONT_HERSHEY_SIMPLEX
        judge_font_scale = 0.7 * scale_factor
        judge_font_thickness = max(1, int(2 * scale_factor))
        # 颜色：姿势标准（绿色），不标准（红色）
        judge_text_color = (0, 255, 0) if pose_judgment['is_correct'] else (0, 0, 255)
        
        # 拼接判断结果文本
        action_name_map = {
            "dig": "Dig",
            "serve": "Serve",
            "set": "Set",
            "spike": "Spike"
        }
        action_en = action_name_map.get(pose_judgment['action_type'], pose_judgment['action_type'].upper())
        judge_text = f"{action_en} Pose: {pose_judgment['judgment']} (Score: {pose_judgment['total_score']})"
        
        # 绘制判断结果（左下角，保留20像素边距）（使用缩放因子）
        cv2.putText(
            img_copy,
            judge_text,
            (int(20 * scale_factor), h - int(50 * scale_factor)),  # 左下角坐标
            judge_font,
            judge_font_scale,
            judge_text_color,
            judge_font_thickness
        )
        
        # 中文文本先收集，函数末尾一次性绘制（避免逐条对整帧做色彩空间转换）
        chinese_items = []
        cn_font_size = int(30 * judge_font_scale)

        # 绘制指导建议（在判断结果上方）
        if 'feedback' in pose_judgment and pose_judgment['feedback']:
            feedback_y = h - int(50 * scale_factor) - int(50 * scale_factor)
            feedback_line_height = int(35 * scale_factor)
            # 使用与姿势判断相同的字体格式
            for i, advice in enumerate(reversed(pose_judgment['feedback']), 1):
                feedback_text = f"建议 {i}: {advice}"
                # 字体大小与姿势判断保持一致
                chinese_items.append((feedback_text, (int(20 * scale_factor), feedback_y),
                                      cn_font_size, (255, 255, 0)))
                feedback_y -= feedback_line_height  # 每行间隔，与其他元素保持一致
        else:
            feedback_y = h - int(50 * scale_factor) - int(50 * scale_factor)
        
        # 绘制排球与手臂距离（在指导建议上方）
        if 'volleyball_arm_distance' in pose_judgment:
            distance_text = f"球臂距离: {pose_judgment['volleyball_arm_distance']:.1f}px"
            distance_y = feedback_y - int(35 * scale_factor)  # 调整间距，与其他元素保持一致

            chinese_items.append((distance_text, (int(20 * scale_factor), distance_y),
                                  cn_font_size, (255, 255, 0)))
            
            # 判断距离是否超过阈值并显示相关文字
            if 'distance_threshold' in pose_judgment:
                threshold = pose_judgment['distance_threshold']
                current_distance = pose_judgment['volleyball_arm_distance']
                
                if current_distance < threshold:
                    status_text = "状态: 动作进行中"
                    status_color = (0, 255, 0)  # 绿色文字
                else:
                    status_text = "状态: 动作结束"
                    status_color = (0, 0, 255)  # 红色文字
                
                status_y = distance_y - int(35 * scale_factor)  # 调整状态文字的位置，避免重叠
                # 与姿势判断使用相同的字体大小
                chinese_items.append((status_text, (int(20 * scale_factor), status_y),
                                      cn_font_size, status_color))

        img_copy = put_chinese_texts(img_copy, chinese_items)

    return img_copy

def extract_valid_keypoints(results, confidence_threshold=0.3):
    """
    从YOLOv8-pose推理结果中提取有效的关键点
    :param results: YOLOv8-pose的推理结果
    :param confidence_threshold: 置信度阈值
    :return: 有效的关键点字典 {kpt_name: (x, y)}
    """
    if not results or not results[0].keypoints:
        return None
    
    # 获取第一个人的关键点数据
    keypoints_data = results[0].keypoints.data[0]
    keypoints_data = tensor_to_numpy(keypoints_data)
    
    valid_kpts = {}
    for kpt_name, idx in CORE_KEYPOINTS.items():
        if idx < len(keypoints_data):
            x, y, conf = keypoints_data[idx]
            if conf > confidence_threshold:
                valid_kpts[kpt_name] = (float(x), float(y))
    
    # 确保所有必要的关键点都存在
    required_kpts = set()
    for kpt in CORE_KEYPOINTS.keys():
        required_kpts.add(kpt)
    
    if len(valid_kpts) >= len(required_kpts) * 0.8:  # 至少80%的关键点有效
        return valid_kpts
    return None

# ===================== 多人提取与跟踪标注 =====================
def extract_persons(results, confidence_threshold=0.3):
    """从 YOLO-Pose 结果中提取所有有效人物（关键点 + 边界框）。"""
    if not results or results[0].keypoints is None:
        return []
    keypoints_data = results[0].keypoints.data
    boxes = results[0].boxes
    persons = []
    for i, kps in enumerate(keypoints_data):
        arr = tensor_to_numpy(kps)
        valid = {}
        confs = []
        for name, idx in CORE_KEYPOINTS.items():
            if idx < len(arr):
                x, y, conf = arr[idx]
                if conf > confidence_threshold:
                    valid[name] = (float(x), float(y))
                    confs.append(float(conf))
        if len(valid) < len(CORE_KEYPOINTS) * 0.6:
            continue
        if boxes is not None and len(boxes) > i:
            xyxy = tensor_to_numpy(boxes.xyxy[i])
            bbox = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))
        else:
            xs = [pt[0] for pt in valid.values()]
            ys = [pt[1] for pt in valid.values()]
            bbox = (min(xs), min(ys), max(xs), max(ys))
        persons.append({
            "bbox": bbox,
            "keypoints": valid,
            "confidence": sum(confs) / len(confs) if confs else 0.0,
        })
    persons.sort(key=lambda item: (item["bbox"][2] - item["bbox"][0]) * (item["bbox"][3] - item["bbox"][1]), reverse=True)
    return persons


def draw_track_labels(frame, persons, track_ids, primary_id=None):
    """在画面上标注每个人物的跟踪 ID，主分析对象用黄色标记。"""
    img = frame.copy()
    for person, tid in zip(persons, track_ids):
        x1, y1, x2, y2 = [int(v) for v in person["bbox"]]
        is_primary = (tid == primary_id)
        color = (0, 255, 255) if is_primary else (255, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{tid}" + (" (主)" if is_primary else "")
        cv2.putText(img, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img


# ===================== 核心姿态估计函数 =====================
def pose_estimation_demo(input_path, is_video=False, save_result=True, window_scale=1.0):
    """
    姿态估计主函数
    :param input_path: 图片/视频路径，或摄像头索引（0为默认摄像头）
    :param is_video: 是否为视频/摄像头输入
    :param save_result: 是否保存结果
    """
    # 加载YOLOv8-pose轻量化模型（nano版，适配电脑/手机）
    pose_model = YOLO("pose_best.pt")
    
    # 创建结果保存目录
    os.makedirs("pose_results", exist_ok=True)
    
    # 改用ultralytics自带的设备检测（兼容所有环境）
    device = 0 if torch.cuda.is_available() else 'cpu'
    print(f"使用设备：{'GPU (CUDA)' if device == 0 else 'CPU'}")
    window_name = "Pose Estimation"
    
    # 处理视频/摄像头输入
    if is_video:
        if isinstance(input_path, int):  # 摄像头
            cap = cv2.VideoCapture(input_path)
        else:  # 视频文件
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                print(f"错误：无法打开视频文件 {input_path}")
                return
        
        # 获取视频参数（用于保存结果）
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter("pose_results/pose_video_result.mp4", fourcc, fps, (width, height))
        setup_display_window(window_name, width, height, window_scale)
        
        print("开始视频姿态估计（按q退出）...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 姿态推理（使用torch检测的device）
            results = pose_model(frame, imgsz=640, conf=0.3, device=device)
            # 可视化结果
            annotated_frame = draw_pose_results(frame, results)
            
            # 显示结果
            cv2.imshow(window_name, annotated_frame)
            # 保存帧到视频
            if save_result:
                out.write(annotated_frame)
            
            # 按q退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # 释放资源
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"视频姿态估计完成，结果保存至：pose_results/pose_video_result.mp4")
    
    # 处理图片输入
    else:
        # 读取图片
        img = cv2.imread(input_path)
        if img is None:
            print(f"错误：无法读取图片 {input_path}")
            return
        
        # 姿态推理（使用torch检测的device）
        setup_display_window(window_name, img.shape[1], img.shape[0], window_scale)

        results = pose_model(img, imgsz=640, conf=0.3, device=device)
        
        # 解析结果（打印关键点信息）
        print("\n=== 姿态估计结果 ===")
        person_count = len(results[0].keypoints.data) if results[0].keypoints is not None else 0
        print(f"检测到的人数：{person_count}")
        
        for person_idx, keypoints in enumerate(results[0].keypoints.data):
            # 修复：张量转numpy
            keypoints = tensor_to_numpy(keypoints)
            print(f"\n第 {person_idx+1} 个人的核心关键点：")
            for kpt_name, idx in CORE_KEYPOINTS.items():
                x, y, conf = keypoints[idx]
                if conf > 0.3:
                    print(f"  {kpt_name}: (x={x:.2f}, y={y:.2f}) 置信度={conf:.2f}")
        
        # 可视化并保存结果
        annotated_img = draw_pose_results(img, results)
        save_path = "pose_results/pose_image_result.jpg"
        cv2.imwrite(save_path, annotated_img)
        print(f"\n图片姿态估计完成，结果保存至：{save_path}")
        
        # 显示结果
        cv2.imshow(window_name, annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# ===================== 主函数 =====================
if __name__ == "__main__":
    # 示例：视频姿态估计（替换为你的视频路径）
    pose_estimation_demo("onepeople.mp4", is_video=True)
    
    # 若要测试摄像头，取消下面注释：
    # pose_estimation_demo(0, is_video=True)
    
    # 若要测试图片，取消下面注释并替换路径：
    # pose_estimation_demo("volleyball_train.jpg", is_video=False)
