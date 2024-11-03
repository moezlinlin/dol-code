import cv2
import numpy as np
import time
import math
import os

def my_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def show_img(frame, start, algorithm_name):
    end = time.time()
    ms_double = (end - start) * 1000
    fps = 1000 / ms_double if ms_double > 0 else 0
    print(f"Processing took {ms_double:.2f} ms, FPS: {fps:.2f}")

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Algorithm: {algorithm_name}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.namedWindow("result", 0)
    cv2.resizeWindow("result", 640, 512)
    cv2.imshow("result", frame)
    cv2.waitKey(1)

def process_image(frame, threshold=30, erode_kernel_size=(4, 4)):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.blur(gray_frame, (3, 3))
    erode_kernel = np.ones(erode_kernel_size, np.uint8)
    gray_frame = cv2.erode(gray_frame, erode_kernel)

    roi_frame = gray_frame[gray_frame.shape[0] // 4: 3 * gray_frame.shape[0] // 4,
                           gray_frame.shape[1] // 4: 3 * gray_frame.shape[1] // 4]

    scharr_grad_x = cv2.Scharr(roi_frame, cv2.CV_16S, 1, 0)
    scharr_grad_y = cv2.Scharr(roi_frame, cv2.CV_16S, 0, 1)
    abs_grad_x = cv2.convertScaleAbs(scharr_grad_x)
    abs_grad_y = cv2.convertScaleAbs(scharr_grad_y)
    scharr_image = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    _, binary_img = cv2.threshold(scharr_image, threshold, 255, cv2.THRESH_BINARY)

    y_indices, x_indices = np.where(binary_img > 0)
    if len(x_indices) > 0:
        weights = binary_img[y_indices, x_indices].astype(float)
        total_weight = np.sum(weights)
        total_weight_x = np.sum(x_indices * weights)
        total_weight_y = np.sum(y_indices * weights)

        center_x = total_weight_x / total_weight + frame.shape[1] / 4
        center_y = total_weight_y / total_weight + frame.shape[0] / 4

        return int(center_x), int(center_y)

    return frame.shape[1] // 2, frame.shape[0] // 2

def draw_true_bbox(frame, label_file_path):
    try:
        with open(label_file_path, 'r') as f:
            for line in f:
                _, x, y, w, h = map(float, line.strip().split())
                centroid = int(x + w / 2), int(y + h / 2)
                cv2.circle(frame, centroid, 5, (0, 255, 0), -1)
                cv2.rectangle(frame, (max(centroid[0] - 30, 0), max(centroid[1] - 30, 0)),
                              (min(centroid[0] + 30, frame.shape[1]), min(centroid[1] + 30, frame.shape[0])),
                              (0, 255, 0), 2)
                return centroid
    except FileNotFoundError:
        print(f"Label file not found: {label_file_path}")

def main():
    folder_path = r"H:\dataprocessing\sot-all"  # 测试图片文件夹
    label_folder_path = r"H:\dataprocessing\sot_label"  # 真值标签的txt文件夹路径
    output_txt_path = r"H:\dataprocessing\output.txt"  # 输出图像名称的TXT文件路径
    count_output_txt_path = r"H:\dataprocessing\count_output.txt"  # 超过10个255值的图像名称TXT文件路径
    supported_formats = (".tif", ".tiff", ".bmp")

    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(supported_formats)]
    if not image_files:
        print(f"No supported images found in folder: {folder_path}")
        return -1

    image_files.sort()

    with open(output_txt_path, 'w') as output_file, open(count_output_txt_path, 'w') as count_output_file:  # 打开输出文件
        for image_file in image_files:
            frame = cv2.imread(os.path.join(folder_path, image_file))
            if frame is None:
                print(f"Failed to read image: {image_file}")
                continue

            start = time.time()

            # 处理图像，获取检测到的质心坐标
            detected_centroid = process_image(frame)
            if detected_centroid:
                x_detected, y_detected = detected_centroid
                cv2.circle(frame, detected_centroid, 5, (0, 0, 255), -1)
                cv2.rectangle(frame, (max(x_detected - 30, 0), max(y_detected - 30, 0)),
                              (min(x_detected + 30, frame.shape[1]), min(y_detected + 30, frame.shape[0])),
                              (0, 0, 255), 2)

            # 获取与图像同名的标签文件路径
            label_file_path = os.path.join(label_folder_path, os.path.splitext(image_file)[0] + ".txt")
            true_centroid = draw_true_bbox(frame, label_file_path)  # 这里可以返回计算出的质心坐标

            # 检查质心距离
            if true_centroid and detected_centroid:
                distance = my_distance(detected_centroid, true_centroid)
                if distance > 20:
                    print(f"Image {image_file} centroid distance exceeds 15 pixels.")
                    output_file.write(image_file + "\n")  # 写入图像名称

            # 检查原图像素均值是否大于 190
            if np.mean(frame) > 120:
                count_output_file.write(image_file + "\n")  # 写入均值大于190的图像名称
            #
            # # 检查原图中值为255的元素数量
            # if np.sum(frame == 255) > 100:
            #     count_output_file.write(image_file + "\n")  # 写入超过10个255值的图像名称

            show_img(frame, start, algorithm_name="sot")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
