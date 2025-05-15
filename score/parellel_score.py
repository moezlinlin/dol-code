# -*- coding: utf-8 -*-
import os
import cv2
import time
import json
import numpy as np
from siam_one_frame import process_frame

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
    返回：
     - init_bbox: [x, y, w_box, h_box]
     - gt_center: (x_c, y_c)
     - w_box, h_box: 目标的像素宽和高
    """
    h_img, w_img = img_shape[:2]
    with open(label_file, 'r') as f:
        line = f.readline().strip()
        if not line:
            raise ValueError(f"Empty label file: {label_file}")
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Invalid label: {label_file}")
        _, x_c_norm, y_c_norm, w_norm, h_norm = map(float, parts[:5])
    x_c = x_c_norm * w_img
    y_c = y_c_norm * h_img
    w_box = w_norm * w_img
    h_box = h_norm * h_img
    x = x_c - w_box / 2
    y = y_c - h_box / 2
    return [x, y, w_box, h_box], (x_c, y_c), (w_box, h_box)

def calculate_pixel_difference(pred_center, gt_center):
    return np.linalg.norm(np.array(pred_center) - np.array(gt_center))

def process_single_folder(folder_path):
    folder_name = os.path.basename(folder_path)
    print(f"\n= 正在处理文件夹: {folder_name} =")
    center_folder = os.path.join(folder_path, "center_out")
    os.makedirs(center_folder, exist_ok=True)
    log_path = os.path.join(folder_path, "score_log.json")

    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
    ])

    results = []
    sum_time_score = sum_acc_score = sum_score = sum_fps = sum_pixel_diff = 0.0
    count = 0

    for idx, fname in enumerate(image_files, start=1):
        img_path = os.path.join(folder_path, fname)
        label_path = os.path.join(folder_path, os.path.splitext(fname)[0] + '.txt')
        if not os.path.exists(label_path):
            print(f"标签缺失，跳过：{fname}")
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f"图像读取失败，跳过：{fname}")
            continue

        try:
            init_bbox, gt_center, (w_box, h_box) = read_normalized_bbox(label_path, img.shape)
        except Exception as e:
            print(str(e))
            continue

        is_init = (idx == 1)
        t0 = time.time()
        pred_center = process_frame(img, init_bbox=init_bbox if is_init else None, is_init=is_init)
        elapsed = (time.time() - t0) * 1000  # ms

        if pred_center is None:
            print(f"未返回预测中心，跳过：{fname}")
            continue


        with open(os.path.join(center_folder, os.path.splitext(fname)[0] + '.txt'), 'w') as f:
            f.write(f"{pred_center[0]:.2f} {pred_center[1]:.2f}\n")


        time_score = calculate_time_score(elapsed)
        pixel_diff = calculate_pixel_difference(pred_center, gt_center)
        acc_score = calculate_acc_score(pixel_diff)
        total_score = time_score * acc_score / 10000
        fps = 1000 / elapsed if elapsed > 0 else 0


        results.append({
            'filename': fname,
            'time_ms': elapsed,
            'fps': fps,
            'time_score': time_score,
            'pixel_diff': pixel_diff,
            'acc_score': acc_score,
            'width': w_box,
            'height': h_box,
            'score': total_score
        })

        sum_time_score += time_score
        sum_acc_score += acc_score
        sum_score += total_score
        sum_fps += fps
        sum_pixel_diff += pixel_diff
        count += 1

        print(f"{fname}: total_score={total_score:.4f}, acc_score={acc_score:.1f}, pixel_diff={pixel_diff:.2f}, "
              f"time_score={time_score:.1f}, fps={fps:.1f}, "
              f"w={w_box:.1f}, h={h_box:.1f}")

    if count > 0:
        avg_time_score = sum_time_score / count
        avg_acc_score  = sum_acc_score / count
        avg_score      = sum_score / count
        avg_fps        = sum_fps / count
        avg_pixel_diff = sum_pixel_diff / count

        print(f"\n--- {folder_name} 处理完成 ---")
        print(f"平均 time_score: {avg_time_score:.2f}")
        print(f"平均 acc_score : {avg_acc_score:.2f}")
        print(f"平均总分       : {avg_score:.4f}")
        print(f"平均 像素差     : {avg_pixel_diff:.2f}")
        print(f"平均 FPS       : {avg_fps:.2f}\n")

    with open(log_path, 'w', encoding='utf-8') as jf:
        json.dump(results, jf, indent=2, ensure_ascii=False)

def batch_process_all_subfolders(root_dir):
    for subfolder in sorted(os.listdir(root_dir)):
        sub_path = os.path.join(root_dir, subfolder)
        if os.path.isdir(sub_path):
            process_single_folder(sub_path)

if __name__ == '__main__':
    dataset_root = './datasets/1'
    batch_process_all_subfolders(dataset_root)
