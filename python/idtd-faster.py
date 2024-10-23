import cv2
import numpy as np
import time

def newRingStrel(ro, ri):
    d = 2 * ro + 1
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (d, d))
    start_index = ro + 1 - ri
    end_index = ro + 1 + ri
    se[start_index:end_index, start_index:end_index] = 0  # 设置内部区域为零
    return se

def MNWTH(img, delta_b, bb):
    img_d = cv2.dilate(img, delta_b)  # 执行膨胀
    img_e = cv2.erode(img_d, bb)  # 执行腐蚀
    out = cv2.subtract(img, img_e)  # 从原始图像减去腐蚀后的图像
    out[out < 0] = 0  # 将负值设置为零
    return out

def moveDetect(frame, delta_b, bb):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
    return MNWTH(gray, delta_b, bb)  # 使用顶帽变换处理

def showImg(frame, start_time):
    elapsed_time = (time.time() - start_time) * 1000  # 计算时间（毫秒）
    print(f"处理时间: {elapsed_time:.2f} ms")
    cv2.imshow("frame", frame)  # 显示处理后的图像
    cv2.waitKey(1)

def main():
    video = cv2.VideoCapture("I:/dolphin_dataset/104658.mp4")
    if not video.isOpened():
        print("无法打开视频文件")
        return

    # 参数设置
    ro = 11
    ri = 10
    delta_b = newRingStrel(ro, ri)
    bb = np.ones((2 * ri + 1, 2 * ri + 1), dtype=np.uint8)

    while True:
        ret, frame = video.read()  # 捕获帧
        if not ret:  # 检查视频结束
            break

        start_time = time.time()  # 开始计时

        result = moveDetect(frame, delta_b, bb)  # 检测运动

        # 显示结果
        cv2.namedWindow("nobinary,frame", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("nobinary,frame", 640, 480)
        cv2.imshow("nobinary,frame", result)

        _, binaryImg = cv2.threshold(result, 30, 255, cv2.THRESH_BINARY)  # 二值化
        cv2.namedWindow("binaryImg", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("binaryImg", 640, 480)
        cv2.imshow("binaryImg", binaryImg)

        # 获取连通组件
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binaryImg, connectivity=4)
        currentPoints = []
        currentScale = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] // 2
            y = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] // 2
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            currentPoints.append((x, y))
            currentScale.append((w, h))

        # 获取最大值位置并在帧上绘制矩形
        if len(currentPoints) > 0:
            max_point = currentPoints[0]
            for point in currentPoints:
                if point[0] > max_point[0]:  # 找到最右侧的点
                    max_point = point
            x, y = max_point
            w, h = 20, 20

            cv2.rectangle(frame, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 2)  # 绘制矩形框

        showImg(frame, start_time)  # 显示帧图像

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
