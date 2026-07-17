# -*- coding: utf-8 -*-
"""视觉检测（占位实现）：用传统 CV 做胶条定位与 OK / NG 判定。

这是算法模块的"最小可用占位"，目的是先让整条流程跑通。
后续可无缝替换为深度学习模型（YOLO / 分割网络等），只需保持
detect(car_model, images) -> DetectionResult 的接口不变。

当前占位逻辑：对每张图阈值分割出明显亮于背景的条带作为胶条；
若某张图未找到足够长的胶条，则记为一处 "missing" 缺陷，
存在 missing 即整体判定 NG。
"""
import os
import cv2
import numpy as np
from common.interfaces import DetectionResult, Defect


class SealDetector:
    def detect(self, car_model: str, images: list) -> DetectionResult:
        all_defects = []
        for path in images:
            all_defects.extend(self._detect_one(path))

        missing = [d for d in all_defects if d.label == "missing"]
        ok = len(missing) == 0
        conf = round(float(np.mean([d.confidence for d in all_defects])) if all_defects else 0.9, 3)

        return DetectionResult(
            car_model=car_model,
            ok=ok,
            defects=all_defects,
            confidence=conf,
            message=f"检测 {len(images)} 张图，发现 {len(all_defects)} 处异常（缺失 {len(missing)}）",
        )

    def _detect_one(self, path: str):
        img = cv2.imread(path)
        if img is None:
            return [Defect(0, 0, 0, 0, "missing", 0.95)]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 占位：找明显亮于背景的条带作为胶条（针对模拟图的高亮胶条）
        mask = gray > 180
        ys, xs = np.where(mask)
        if len(xs) < 500:
            return [Defect(0, 0, 0, 0, "missing", 0.9)]
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        return [Defect(x0, y0, x1 - x0, y1 - y0, "seal", 0.85)]
