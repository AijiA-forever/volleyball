# volleyball_detect.py - 排球检测模块，使用训练好的模型检测视频和图像中的排球
from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import numpy as np


MODEL_PATH = str(Path(__file__).resolve().parent / "best.pt")


def load_volleyball_model():
    """加载训练好的排球检测模型"""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    print(f"Volleyball detection model loaded from: {MODEL_PATH}")
    return model


def detect_volleyball(model, frame, conf=0.5, imgsz=640):
    """
    检测视频帧中的排球
    :param model: 排球检测模型
    :param frame: 输入图像帧
    :param conf: 置信度阈值
    :param imgsz: 输入图像大小
    :return: 检测结果列表
    """
    results = model(frame, imgsz=imgsz, conf=conf, verbose=False)
    detections = []

    if results and len(results) > 0:
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf_score = float(box.conf[0])
                cls_id = int(box.cls[0])
                detections.append({
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': conf_score,
                    'class_id': cls_id,
                    'center': ((int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2)
                })

    return detections


def draw_volleyball_results(frame, detections, show_conf=True):
    """
    在图像上绘制排球检测结果
    :param frame: 输入图像帧
    :param detections: 检测结果列表
    :param show_conf: 是否显示置信度
    :return: 标注后的图像
    """
    img_copy = frame.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det['bbox']]
        conf = float(det['confidence'])

        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if show_conf:
            label = f"Volleyball: {conf:.2f}"
        else:
            label = "Volleyball"

        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(img_copy, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(img_copy, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    return img_copy


def volleyball_detection_demo(input_path, is_video=False, conf=0.5, save_result=True):
    """
    排球检测演示函数
    :param input_path: 输入视频/图片路径
    :param is_video: 是否为视频输入
    :param conf: 置信度阈值
    :param save_result: 是否保存结果
    """
    model = load_volleyball_model()
    os.makedirs("volleyball_results", exist_ok=True)

    # 处理视频输入
    if is_video:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {input_path}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter("volleyball_results/volleyball_detection_result.mp4", fourcc, fps, (width, height))

        frame_count = 0
        total_detections = 0

        print(f"Starting volleyball detection (press 'q' to exit)...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            detections = detect_volleyball(model, frame, conf=conf)
            total_detections += len(detections)
            frame_count += 1

            annotated_frame = draw_volleyball_results(frame, detections)

            cv2.imshow("Volleyball Detection", annotated_frame)
            if save_result:
                out.write(annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"Detection complete! Total frames: {frame_count}, Total volleyballs detected: {total_detections}")
        print(f"Results saved to: volleyball_results/volleyball_detection_result.mp4")

    # 处理图片输入
    else:
        img = cv2.imread(input_path)
        if img is None:
            print(f"Error: Cannot read image {input_path}")
            return

        detections = detect_volleyball(model, img, conf=conf)

        print(f"\n=== Volleyball Detection Results ===")
        print(f"Number of volleyballs detected: {len(detections)}")
        for i, det in enumerate(detections):
            print(f"  Volleyball {i+1}: Confidence={det['confidence']:.2f}, BBox={det['bbox']}, Center={det['center']}")

        annotated_img = draw_volleyball_results(img, detections)
        save_path = "volleyball_results/volleyball_detection_result.jpg"
        cv2.imwrite(save_path, annotated_img)

        cv2.imshow("Volleyball Detection", annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(f"\nResults saved to: {save_path}")


if __name__ == "__main__":
    volleyball_detection_demo("onepeople.mp4", is_video=True)

    # Test with image:
    # volleyball_detection_demo("test.jpg", is_video=False)
