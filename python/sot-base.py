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
    # video = cv2.VideoCapture("I:/dolphin_dataset/J20240711fire.mp4")
    video = cv2.VideoCapture(r"I:\dolphin_dataset\处理后\原始的\track-train-1\video.mp4")

    if not video.isOpened():
        return -1

    frameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) # 获取帧数
    FPS = video.get(cv2.CAP_PROP_FPS) # 获取FPS
    frame = None # 存储帧
    lightFlag = True
    num = 0

    cv2.namedWindow("00", 0)
    cv2.resizeWindow("00", 640, 512)

    for i in range(frameCount):
        ret, frame = video.read()
        if not ret:
            print("Frame is empty!")
            break

        cv2.imshow("00", frame)

        # 转换为灰度图像并模糊处理
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.blur(frame, (3, 3))
        # frame = cv2.medianBlur(frame, 3)

        # 腐蚀操作
        b2 = np.ones((4, 4), np.uint8)
        frame = cv2.erode(frame, b2)

        start = time.time()
        print(f"min= {num // 20 // 60}, sec= {num // 20 % 60}")
        num += 1

        if not lightFlag:
            frame = 255 - frame

        # 裁剪ROI区域
        roi_frame = frame[frame.shape[0] // 4: 3 * frame.shape[0] // 4,
                          frame.shape[1] // 4: 3 * frame.shape[1] // 4]
        frame_little = roi_frame.copy()

        # 计算Scharr梯度
        g_scharrGradient_X = cv2.Scharr(frame_little, cv2.CV_16S, 1, 0)
        g_scharrGradient_Y = cv2.Scharr(frame_little, cv2.CV_16S, 0, 1)
        g_scharrAbsGradient_X = cv2.convertScaleAbs(g_scharrGradient_X)
        g_scharrAbsGradient_Y = cv2.convertScaleAbs(g_scharrGradient_Y)
        scharrImage = cv2.addWeighted(g_scharrAbsGradient_X, 0.5, g_scharrAbsGradient_Y, 0.5, 0)

        cv2.namedWindow("11", 0)
        cv2.resizeWindow("11", 640, 512)
        cv2.imshow("11", scharrImage)

        # 二值化处理
        _, binaryImg = cv2.threshold(scharrImage, 30, 255, cv2.THRESH_BINARY)

        cv2.namedWindow("binaryImg", 0)
        cv2.resizeWindow("binaryImg", 640, 512)
        cv2.imshow("binaryImg", binaryImg)

        # 计算质心位置
        totalWeightX = 0
        totalWeightY = 0
        totalWeight = 0
        step = 2

        for y in range(0, binaryImg.shape[0], step):
            for x in range(0, binaryImg.shape[1], step):
                weight = float(binaryImg[y, x])
                totalWeightX += x * weight
                totalWeightY += y * weight
                totalWeight += weight

        print(f"total weight = {totalWeight}")

        # 计算加权平均位置
        if totalWeight > 0:
            centerX = totalWeightX / totalWeight + frame.shape[1] / 4
            centerY = totalWeightY / totalWeight + frame.shape[0] / 4
            cv2.rectangle(frame, (max(int(centerX - 30), 0), max(int(centerY - 30), 0)),
                          (min(int(centerX + 30), frame.shape[1]), min(int(centerY + 30), frame.shape[0])),
                          (0, 0, 255), 2)
            print(f"全图质心位置：({centerX:.2f}, {centerY:.2f})")

        # 显示效果图窗口
        show_img(frame, start)

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()




# # **优化措施总结**：
# # 1. 降低显示窗口大小，减少了视觉处理的开销。
# # 2. 将帧率降低，每隔两帧进行处理，减少处理频率。
# # 3. 调整图像分辨率，处理前缩小到 640x360。
# # 4. 在质心计算时增大遍历步长（从 2 改为 4），减少计算量。
# #
# # 这些改动能显著提高代码的运行速度，同时保留目标位置检测的主要功能。
# import cv2
# import numpy as np
# import time
# import math
#
# initialPoint = None
# pointSelected = False
#
# def my_distance(p1, p2):
#     return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
#
# def show_img(frame, start):
#     end = time.time()
#     ms_double = (end - start) * 1000
#     print(f"it took {ms_double:.2f} ms")
#     cv2.imshow("result", frame)
#     cv2.waitKey(1)
#
# # 鼠标事件回调函数：选择跟踪目标
# def on_mouse(event, x, y, flags, param):
#     global initialPoint, pointSelected
#     if event == cv2.EVENT_LBUTTONDOWN:
#         initialPoint = (x, y)
#         pointSelected = True
#
# def main():
#     # video = cv2.VideoCapture("I:/dolphin_dataset/J20240711fire.mp4")
#     video = cv2.VideoCapture(r"I:\dolphin_dataset\处理后\原始的\track-train-1\video-label.mp4")
#
#     if not video.isOpened():
#         return -1
#
#     frameCount = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) # 获取帧数
#     FPS = video.get(cv2.CAP_PROP_FPS) # 获取FPS
#     frame = None # 存储帧
#     lightFlag = True
#     num = 0
#
#     # 设置窗口
#     cv2.namedWindow("00", 0)
#     cv2.resizeWindow("00", 480, 360)  # 将窗口调小一点
#     cv2.namedWindow("result", 0)
#     cv2.resizeWindow("result", 480, 360)
#
#     # 读取并处理帧
#     for i in range(0, frameCount, 1):  # 每隔两帧处理一次
#         ret, frame = video.read()
#         if not ret:
#             print("Frame is empty!")
#             break
#
#         # frame = cv2.resize(frame, (640, 360))  # 降低分辨率来加快处理速度
#         cv2.imshow("00", frame)
#
#         # 转换为灰度图像并模糊处理
#         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         frame = cv2.blur(frame, (3, 3))
#
#         # 腐蚀操作
#         b2 = np.ones((4, 4), np.uint8)
#         frame = cv2.erode(frame, b2)
#
#         start = time.time()
#         print(f"min= {num // 20 // 60}, sec= {num // 20 % 60}")
#         num += 1
#
#         if not lightFlag:
#             frame = 255 - frame
#
#         # 裁剪ROI区域
#         roi_frame = frame[frame.shape[0] // 4: 3 * frame.shape[0] // 4,
#                           frame.shape[1] // 4: 3 * frame.shape[1] // 4]
#         frame_little = roi_frame.copy()
#
#         # 计算Scharr梯度
#         g_scharrGradient_X = cv2.Scharr(frame_little, cv2.CV_16S, 1, 0)
#         g_scharrGradient_Y = cv2.Scharr(frame_little, cv2.CV_16S, 0, 1)
#         g_scharrAbsGradient_X = cv2.convertScaleAbs(g_scharrGradient_X)
#         g_scharrAbsGradient_Y = cv2.convertScaleAbs(g_scharrGradient_Y)
#         scharrImage = cv2.addWeighted(g_scharrAbsGradient_X, 0.5, g_scharrAbsGradient_Y, 0.5, 0)
#
#         # 二值化处理
#         _, binaryImg = cv2.threshold(scharrImage, 30, 255, cv2.THRESH_BINARY)
#
#         # 计算质心位置
#         totalWeightX = 0
#         totalWeightY = 0
#         totalWeight = 0
#         step = 4  # 增大步长来减少计算量
#
#         for y in range(0, binaryImg.shape[0], step):
#             for x in range(0, binaryImg.shape[1], step):
#                 weight = float(binaryImg[y, x])
#                 totalWeightX += x * weight
#                 totalWeightY += y * weight
#                 totalWeight += weight
#
#         print(f"total weight = {totalWeight}")
#
#         # 计算加权平均位置
#         if totalWeight > 0:
#             centerX = totalWeightX / totalWeight + frame.shape[1] / 4
#             centerY = totalWeightY / totalWeight + frame.shape[0] / 4
#             cv2.rectangle(frame, (max(int(centerX - 30), 0), max(int(centerY - 30), 0)),
#                           (min(int(centerX + 30), frame.shape[1]), min(int(centerY + 30), frame.shape[0])),
#                           (0, 0, 255), 2)
#             print(f"全图质心位置：({centerX:.2f}, {centerY:.2f})")
#
#         # 显示效果图窗口
#         show_img(frame, start)
#
#     video.release()
#     cv2.destroyAllWindows()
#
# if __name__ == "__main__":
#     main()
