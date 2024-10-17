import cv2
import numpy as np
import time
import math

initialPoint = None
pointSelected = False

def my_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def show_img(frame, start):
    end = time.time()
    ms_double = (end - start) * 1000
    print(f"it took {ms_double:.2f} ms")

    cv2.namedWindow("result", 0)
    cv2.resizeWindow("result", 640, 512)
    cv2.imshow("result", frame)

    cv2.waitKey(1)

# 鼠标事件回调函数：选择跟踪目标
def on_mouse(event, x, y, flags, param):
    global initialPoint, pointSelected
    if event == cv2.EVENT_LBUTTONDOWN:
        initialPoint = (x, y)
        pointSelected = True

def main():
    video_path = r"I:\dolphin_dataset\处理后\原始的\track-train-1\video.mp4"
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"Failed to open video: {video_path}")
        return -1

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取帧数
    FPS = video.get(cv2.CAP_PROP_FPS)  # 获取FPS
    lightFlag = True
    num = 0

    cv2.namedWindow("00", 0)
    cv2.resizeWindow("00", 640, 512)

    try:
        for i in range(frame_count):
            ret, frame = video.read()
            if not ret:
                print("Failed to read frame!")
                break

            cv2.imshow("00", frame)

            # 转换为灰度图像并模糊处理
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.blur(gray_frame, (3, 3))

            # 腐蚀操作
            b2 = np.ones((4, 4), np.uint8)
            gray_frame = cv2.erode(gray_frame, b2)

            start = time.time()
            print(f"min= {num // 20 // 60}, sec= {num // 20 % 60}")
            num += 1

            if not lightFlag:
                gray_frame = 255 - gray_frame

            # 裁剪ROI区域
            roi_frame = gray_frame[gray_frame.shape[0] // 4: 3 * gray_frame.shape[0] // 4,
                                   gray_frame.shape[1] // 4: 3 * gray_frame.shape[1] // 4]

            # 计算Scharr梯度
            scharr_grad_x = cv2.Scharr(roi_frame, cv2.CV_16S, 1, 0)
            scharr_grad_y = cv2.Scharr(roi_frame, cv2.CV_16S, 0, 1)
            abs_grad_x = cv2.convertScaleAbs(scharr_grad_x)
            abs_grad_y = cv2.convertScaleAbs(scharr_grad_y)
            scharr_image = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

            cv2.namedWindow("11", 0)
            cv2.resizeWindow("11", 640, 512)
            cv2.imshow("11", scharr_image)

            # 二值化处理
            _, binary_img = cv2.threshold(scharr_image, 30, 255, cv2.THRESH_BINARY)

            cv2.namedWindow("binaryImg", 0)
            cv2.resizeWindow("binaryImg", 640, 512)
            cv2.imshow("binaryImg", binary_img)

            # 计算质心位置，使用NumPy进行向量化操作
            y_indices, x_indices = np.where(binary_img > 0)
            if len(x_indices) > 0:
                weights = binary_img[y_indices, x_indices].astype(float)
                total_weight = np.sum(weights)
                total_weight_x = np.sum(x_indices * weights)
                total_weight_y = np.sum(y_indices * weights)

                # 计算加权平均位置
                center_x = total_weight_x / total_weight + frame.shape[1] / 4
                center_y = total_weight_y / total_weight + frame.shape[0] / 4

                cv2.rectangle(frame,
                              (max(int(center_x - 30), 0), max(int(center_y - 30), 0)),
                              (min(int(center_x + 30), frame.shape[1]), min(int(center_y + 30), frame.shape[0])),
                              (0, 0, 255), 2)
                print(f"全图质心位置：({center_x:.2f}, {center_y:.2f})")

            # 显示效果图窗口
            show_img(frame, start)

    finally:
        # 确保在任何情况下都能释放资源
        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
