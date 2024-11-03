import cv2
import numpy as np
import os
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


def process_image(image):
    ro = 11
    ri = 10
    delta_b = newRingStrel(ro, ri)
    bb = np.ones((2 * ri + 1, 2 * ri + 1), dtype=np.uint8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换为灰度图
    result = MNWTH(gray, delta_b, bb)  # 检测运动

    _, binaryImg = cv2.threshold(result, 30, 255, cv2.THRESH_BINARY)  # 二值化

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binaryImg, connectivity=4)

    if num_labels > 1:
        largest_component_index = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # 找到最大的连通区域
        centroid = centroids[largest_component_index]  # 获取质心坐标
        return (int(centroid[0]), int(centroid[1])), num_labels - 1  # 返回质心坐标 (x, y) 和连通组件数量

    return None, num_labels - 1  # 如果没有检测到对象，返回 None



def draw_boxes(image, label_path, predicted_box):
    with open(label_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        values = line.strip().split()
        if len(values) >= 5:
            _, x, y, w, h = map(float, values)
            x, y, w, h = int(x), int(y), int(w), int(h)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 真值框为红色

    if predicted_box:
        pred_x, pred_y, pred_w, pred_h = predicted_box
        cv2.rectangle(image, (pred_x, pred_y), (pred_x + pred_w, pred_y + pred_h), (0, 255, 0), 2)  # 预测框为绿色


def save_image_if_distance_exceeds(image_name, true_centroid, predicted_centroid, output_directory, threshold=10):
    distance = np.linalg.norm(np.array(true_centroid) - np.array(predicted_centroid))
    print(f"真实中心: {true_centroid}, 预测中心: {predicted_centroid}, 距离: {distance}")  # 打印距离
    if distance > threshold:
        with open(output_directory, 'a') as f:
            f.write(image_name + '\n')
            print("写入成功")


def process_directory(image_dir, label_dir, output_directory):
    frame_count = 0
    total_time = 0.0

    for image_name in os.listdir(image_dir):
        if image_name.endswith(('.bmp', '.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, image_name)
            label_path = os.path.join(label_dir, image_name.replace(image_name.split('.')[-1], 'txt'))

            image = cv2.imread(image_path)
            if image is not None:
                start_time = time.time()  # 记录开始时间

                # 读取真值中心
                true_center = None
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        # 读取第一个框的坐标
                        values = lines[0].strip().split()
                        if len(values) >= 5:
                            _, x, y, w, h = map(float, values)
                            true_center = (int(x + w / 2), int(y + h / 2))  # 计算中心坐标

                centroid, num_components = process_image(image)
                predicted_box = (centroid[0] - 20, centroid[1] - 20, 40, 40) if centroid else None

                draw_boxes(image, label_path, predicted_box)

                if centroid:
                    predicted_center = (predicted_box[0] + predicted_box[2] // 2, predicted_box[1] + predicted_box[3] // 2)
                    save_image_if_distance_exceeds(image_name, true_center, predicted_center, output_directory)

                else:
                    print(f"没有检测到对象: {image_name}")

                end_time = time.time()  # 记录结束时间
                process_time = end_time - start_time
                total_time += process_time
                frame_count += 1

                fps = frame_count / total_time if total_time > 0 else 0
                print(f"处理时间: {process_time:.4f}秒, 当前 FPS: {fps:.2f}")

                cv2.imshow("Detected Boxes", image)
                key = cv2.waitKey(100)
                if key == 27:
                    break
            else:
                print(f"无法读取图像: {image_path}")


if __name__ == "__main__":
    image_directory = r"H:\dataprocessing\idtd"  # 替换为你的图像文件夹路径
    label_directory = r"H:\dataprocessing\idtd_label"  # 替换为你的标签文件夹路径
    output_directory = r"H:\dataprocessing\output\mismatched_images.txt"  # 用户指定的输出路径
    process_directory(image_directory, label_directory, output_directory)

    cv2.destroyAllWindows()







