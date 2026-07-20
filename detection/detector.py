# -*- coding: utf-8 -*-
"""视觉检测（胶条外形检测算法，按用户定义实现）。

处理流水线：
  Step1   轻度加亮：gamma 提亮暗部，拉大胶条与暗背景灰度差
  Step2   ROI 裁剪：锁定车顶胶条固定区域，剔除无关车身背景
  Step3   小波滤波：db4 多尺度分解去噪，保留胶条边缘
  Step4   OTSU + 阈值扣减 + 面积过滤：过滤暗背景误判，保留大面积真实胶条
  Step4.5 形态学开闭运算：闭运算填充胶条内部反光孔洞，开运算平滑边缘

所有参数集中在文件顶部，便于现场调参。接口保持
detect(car_model, images) -> DetectionResult 不变。
"""
import cv2
import numpy as np
from common.interfaces import DetectionResult, Defect

# ===================== 算法参数（来自用户定义） =====================
# Step1 轻度加亮
BRIGHT_MODE = 'gamma'        # 'gamma' 或 'linear'
GAMMA_VALUE = 0.95           # 轻度提亮暗部
LINEAR_ALPHA = 1.0
LINEAR_BETA = 5

# Step2 ROI：(x左上, y左上, 宽度w, 高度h)
# 实测相机分辨率为 1920x1200，ROI 宽度设 1920 覆盖整幅；
# _detect_one 内部用 min() 把 ROI clamp 到图像边界，更窄图像(如 1440)也自适应。
ROI = (0, 405, 1920, 259)

# Step3 小波滤波
WAVELET_TYPE = 'db4'

# Step4 OTSU 参数
OTSU_INVERTED = True         # 灰度<=阈值→白色（胶条）
OTSU_THRESHOLD_DELTA = 120   # 实际阈值 = OTSU自动阈值 - 120，过滤暗背景误判
OTSU_MAX_THRESH = 220        # 阈值上限保护
OTSU_MIN_THRESH = 40         # 阈值下限保护
MIN_CONTOUR_AREA = 300        # 面积过滤：滤除 17x15 级小噪声碎片，仅保留大面积胶条

# Step4.5 形态学开闭运算
MORPHOLOGY_ENABLE = True
CLOSE_KERNEL_SIZE = (21, 3)  # 闭运算核：横向长适配水平胶条，高度小避免变形
OPEN_KERNEL_SIZE = (3, 3)    # 开运算核：去除边缘毛刺
MORPHOLOGY_ITERATIONS = 1
# ================================================================


def _brighten(gray: np.ndarray) -> np.ndarray:
    """Step1：轻度加亮暗部，不碰暗背景。"""
    if BRIGHT_MODE == 'gamma':
        inv = 1.0 / GAMMA_VALUE
        table = np.array([((i / 255.0) ** inv) * 255
                           for i in range(256)]).astype(np.uint8)
        return cv2.LUT(gray, table)
    return cv2.convertScaleAbs(gray, alpha=LINEAR_ALPHA, beta=LINEAR_BETA)


def _wavelet_denoise(gray: np.ndarray) -> np.ndarray:
    """Step3：db4 小波去噪，保留边缘；pywt 不可用时降级为中值滤波。"""
    try:
        import pywt
        coeffs = pywt.wavedec2(gray, WAVELET_TYPE, level=2)
        # 用 BayeShrink 思路做软阈值
        detail = coeffs[-1]
        sigma = np.median(np.abs(detail)) / 0.6745 if detail.size else 1.0
        uthresh = sigma * np.sqrt(2 * np.log(gray.size))
        new_coeffs = [coeffs[0]]
        for c in coeffs[1:]:
            new_coeffs.append(tuple(pywt.threshold(d, uthresh, mode='soft')
                                    for d in c))
        denoised = pywt.waverec2(new_coeffs, WAVELET_TYPE)
        denoised = np.clip(denoised, 0, 255).astype(np.uint8)
        return denoised[:gray.shape[0], :gray.shape[1]]
    except Exception:
        # 降级：中值滤波去噪（不增强边缘，但保证可运行）
        return cv2.medianBlur(gray, 5)


def _detect_one(path: str):
    img = cv2.imread(path)
    if img is None:
        return [Defect(0, 0, 0, 0, "missing", 0.9)]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step1 加亮
    gray = _brighten(gray)

    # Step2 ROI 裁剪（clamp 到图像边界，防止尺寸不符越界）
    x, y, w, h = ROI
    x2 = min(x + w, gray.shape[1])
    y2 = min(y + h, gray.shape[0])
    if x2 <= x or y2 <= y:
        # ROI 超出图像范围，退化为全图处理
        roi = gray
    else:
        roi = gray[y:y2, x:x2]

    # Step3 小波去噪
    roi = _wavelet_denoise(roi)

    # Step4 OTSU + 阈值扣减 + 面积过滤
    if OTSU_INVERTED:
        roi_inv = cv2.bitwise_not(roi)
        otsu_val, _ = cv2.threshold(roi_inv, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        actual = int(otsu_val) - OTSU_THRESHOLD_DELTA
    else:
        otsu_val, _ = cv2.threshold(roi, 0, 255,
                                    cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        actual = int(otsu_val) - OTSU_THRESHOLD_DELTA
    actual = max(OTSU_MIN_THRESH, min(OTSU_MAX_THRESH, actual))

    if OTSU_INVERTED:
        mask = (roi <= actual).astype(np.uint8) * 255   # 灰度<=阈值 → 胶条(白)
    else:
        mask = (roi >= actual).astype(np.uint8) * 255

    # Step4.5 形态学开闭运算
    if MORPHOLOGY_ENABLE:
        close_k = cv2.getStructuringElement(cv2.MORPH_CROSS, CLOSE_KERNEL_SIZE)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k,
                                iterations=MORPHOLOGY_ITERATIONS)
        open_k = cv2.getStructuringElement(cv2.MORPH_CROSS, OPEN_KERNEL_SIZE)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k,
                                iterations=MORPHOLOGY_ITERATIONS)

    # 面积过滤 + 轮廓提取
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    defects = []
    for c in contours:
        if cv2.contourArea(c) < MIN_CONTOUR_AREA:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        defects.append(Defect(x + bx, y + by, bw, bh, "seal", 0.85))

    if not defects:
        return [Defect(0, 0, 0, 0, "missing", 0.9)]
    return defects


class SealDetector:
    def detect(self, car_model: str, images: list) -> DetectionResult:
        all_defects = []
        for path in images:
            all_defects.extend(_detect_one(path))

        missing = [d for d in all_defects if d.label == "missing"]
        ok = len(missing) == 0
        conf = round(float(np.mean([d.confidence for d in all_defects]))
                    if all_defects else 0.9, 3)

        return DetectionResult(
            car_model=car_model,
            ok=ok,
            defects=all_defects,
            confidence=conf,
            message=f"检测 {len(images)} 张图，发现 {len(all_defects)} 处胶条区域（缺失 {len(missing)}）",
        )
