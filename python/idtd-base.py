import cv2
import numpy as np
import time
import math

def new_ring_strel(ro, ri):
    d = 2 * ro + 1
    se = np.ones((d, d), dtype=np.uint8)
    start_index = ro + 1 - ri
    end_index = ro + 1 + ri
    se[start_index:end_index, start_index:end_index] = 0
    return se

def mnwth(img, delta_b, bb):
    img_f = img.copy()
    _, binary_img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

    img_d = cv2.dilate(img, delta_b)
    img_e = cv2.erode(img_d, bb)

    out = cv2.subtract(img, img_e)
    out[out < 0] = 0
    return out

def move_detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    result = mnwth(gray, delta_b, bb)
    return result

def my_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def show_img(frame, start_time):
    elapsed_time = (time.time() - start_time) * 1000
    cv2.imshow("frame", frame)
    cv2.waitKey(1)

if __name__ == "__main__":
    video = cv2.VideoCapture("I:/dolphin_dataset/104658.mp4")
    if not video.isOpened():
        print("Failed to open video.")
        exit()

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)

    ro = 11
    ri = 10
    delta_b = new_ring_strel(ro, ri)
    bb = np.ones((2 * ri + 1, 2 * ri + 1), dtype=np.uint8)

    num = 0

    while True:
        ret, frame = video.read()
        if not ret:
            print("Frame is empty!")
            break

        start_time = time.time()
        num += 1

        result = move_detect(frame)

        cv2.namedWindow("nobinary,frame", 0)
        cv2.resizeWindow("nobinary,frame", 640, 480)
        cv2.imshow("nobinary,frame", result)

        _, binary_img = cv2.threshold(result, 30, 255, cv2.THRESH_BINARY)
        cv2.namedWindow("binaryImg", 0)
        cv2.resizeWindow("binaryImg", 640, 480)
        cv2.imshow("binaryImg", binary_img)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_img, connectivity=4)
        current_points = []
        current_scale = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            x = stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] // 2
            y = stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] // 2
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            current_points.append((x, y))
            current_scale.append((w, h))

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        x, y = max_loc
        w, h = 20, 20
        cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2)
        show_img(frame, start_time)

    video.release()
    cv2.destroyAllWindows()
