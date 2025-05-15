# 评分程序使用说明

## 1，替换90行的process\_frame，函数要求如下

    def process_frame(image, init_bbox=None, is_init=False):
        """
        参数:
            image: BGR 格式 ndarray
            init_bbox: [x, y, w, h]，仅在 is_init=True 时必需
            is_init: bool, 是否初始化帧

        返回值:
            (centerX, centerY) 或 None
        """

## 2，替换55至58行数据路径

        # --------- 配置路径 ---------
        image_folder  = r"H:\pycharm\SiamMask-master\data\20250319_j162036_uav1\images"
        txt_gt_folder     = r"H:\pycharm\SiamMask-master\data\20250319_j162036_uav1\labels"
        center_folder = r"H:\pycharm\SiamMask-master\data\20250319_j162036_uav1\center_out"
        log_path      = r"H:\pycharm\SiamMask-master\data\20250319_j162036_uav1\score_log.json"

## 3，评分标准

***

1\. 时间评分 `time_score`（范围：0\~100）

*   若推理时间 ≤ 0.9 ms，得满分 100；
*   若 ≥ 5 ms，得 0 分；
*   中间线性插值。

2\. 精度评分 `acc_score`（范围：0\~100）

*   若预测点距离 GT 中心 ≤ 1 像素，得满分 100；
*   若 ≥ 10 像素，得 0 分；
*   中间线性插值。

## 4，示例输出

    9411.jpg: time_score=0.0, acc_score=73.5, score=0.0000, pixel_diff=6.96, fps=8.00
    9501.jpg: time_score=0.0, acc_score=86.4, score=0.0000, pixel_diff=4.05, fps=7.84
    9591.jpg: time_score=0.0, acc_score=79.3, score=0.0000, pixel_diff=5.65, fps=7.50
    9681.jpg: time_score=0.0, acc_score=0.0, score=0.0000, pixel_diff=15.29, fps=7.47
    9771.jpg: time_score=0.0, acc_score=75.9, score=0.0000, pixel_diff=6.42, fps=7.89
    9861.jpg: time_score=0.0, acc_score=76.3, score=0.0000, pixel_diff=6.32, fps=7.39
    9951.jpg: time_score=0.0, acc_score=0.0, score=0.0000, pixel_diff=459.22, fps=7.61
    Average time_score: 0.00
    Average acc_score: 32.98
    Average total_score: 0.0000



