import os
import cv2
import time
import json
import numpy as np
from siam_one_frame import process_frame

# ------- 打分函数 -------
def calculate_time_score(time_in_ms):
    if time_in_ms <= 0.9:
        return 100
    elif time_in_ms >= 5:
        return 0
    return 60 + (100 - 60) * (5 - time_in_ms) / (5 - 0.9)


def calculate_acc_score(pixels):
    if pixels <= 1:
        return 100
    elif pixels >= 10:
        return 0
    return 60 + (100 - 60) * (10 - pixels) / (10 - 1)


def read_normalized_bbox(label_file, img_shape):
    """
    从 YOLO 归一化标签读取 x_center_norm, y_center_norm, w_norm, h_norm
    转换为像素坐标的 init_bbox: [x, y, w, h]
    返回 init_bbox 和 GT 中心坐标
    """
    h_img, w_img = img_shape[:2]
    with open(label_file, 'r') as f:
        line = f.readline().strip()
        if not line:
            raise ValueError(f"Ground truth file {label_file} is empty.")
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Invalid label format in {label_file}")
        _, x_c_norm, y_c_norm, w_norm, h_norm = map(float, parts[:5])
    x_c = x_c_norm * w_img
    y_c = y_c_norm * h_img
    w_box = w_norm * w_img
    h_box = h_norm * h_img
    x = x_c - w_box / 2
    y = y_c - h_box / 2
    return [x, y, w_box, h_box], (x_c, y_c)


def calculate_pixel_difference(pred_center, gt_center):
    return np.linalg.norm(np.array(pred_center) - np.array(gt_center))


def main():
    # --------- 配置路径 ---------
    image_folder  = r"H:\pycharm\SiamMask-master\data\20250319_j162036_uav1\images"
    gt_folder     = r"H:\pycharm\SiamMask-master\data\20250319_j162036_uav1\labels"
    center_folder = r"H:\pycharm\SiamMask-master\data\20250319_j162036_uav1\center_out"
    log_path      = r"H:\pycharm\SiamMask-master\data\20250319_j162036_uav1\score_log.json"

    os.makedirs(center_folder, exist_ok=True)

    results = []
    sum_time_score = 0.0
    sum_acc_score  = 0.0
    sum_score      = 0.0
    count = 0

    # 遍历并按序号区分是否初始化
    image_files = sorted([f for f in os.listdir(image_folder)
                          if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tiff'))])
    for idx, fname in enumerate(image_files, start=1):
        img_path = os.path.join(image_folder, fname)
        gt_path  = os.path.join(gt_folder, os.path.splitext(fname)[0] + '.txt')
        if not os.path.exists(gt_path):
            print(f"GT not found for {fname}, skip.")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to read {fname}")
            continue

        # 读取 init_bbox 和 GT 中心
        init_bbox, gt_center = read_normalized_bbox(gt_path, img.shape)

        # 是否初始化帧
        is_init = (idx == 1)
        # 调用 process_frame 并计时
        t0 = time.time()
        pred_center = process_frame(img, init_bbox=init_bbox if is_init else None, is_init=is_init)
        elapsed = (time.time() - t0) * 1000  # ms

        if pred_center is None:
            print(f"No mask for {fname}")
            continue

        # 预测中心
        out_path = os.path.join(center_folder, os.path.splitext(fname)[0] + '.txt')
        with open(out_path, 'w') as f:
            f.write(f"{pred_center[0]:.2f} {pred_center[1]:.2f}\n")

        # 计算分数
        time_score = calculate_time_score(elapsed)
        pixel_diff = calculate_pixel_difference(pred_center, gt_center)
        acc_score  = calculate_acc_score(pixel_diff)
        total_score = time_score * acc_score / 10000

        results.append({
            'filename': fname,
            'time_ms': elapsed,
            'time_score': time_score,
            'pixel_diff': pixel_diff,
            'acc_score': acc_score,
            'score': total_score
        })

        sum_time_score += time_score
        sum_acc_score  += acc_score
        sum_score      += total_score
        count += 1

        fps = 1000 / elapsed if elapsed > 0 else 0
        print(f"{fname}: time_score={time_score:.1f}, acc_score={acc_score:.1f}, score={total_score:.4f}, "
              f"pixel_diff={pixel_diff:.2f}, fps={fps:.2f}")

        # print(f"{fname}: time_score={time_score:.1f}, acc_score={acc_score:.1f}, score={total_score:.4f}")

    # 平均分
    if count > 0:
        print(f"Average time_score: {sum_time_score/count:.2f}")
        print(f"Average acc_score: {sum_acc_score/count:.2f}")
        print(f"Average total_score: {sum_score/count:.4f}")

    # 保存日志
    with open(log_path, 'w') as jf:
        json.dump(results, jf, indent=4)

if __name__ == '__main__':
    main()
