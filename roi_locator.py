# -*- coding: utf-8 -*-
"""模块① ROI 定位：从图像给出胶条紧 ROI (x,y,w,h)。

当前实现用红框解析（48 张标注样本）；未来 YOLO 训练好后，只需新增一个
YOLOLocator 实现同一 locate() 接口，下游②③不变——这正是用户定的架构：
YOLO 框 = 替代红框的紧 ROI，喂给 crop_and_segment。
"""
from __future__ import annotations

import cv2
import numpy as np
import os
import sys


class Locator:
    """接口：locate(bgr) -> List[(x, y, w, h)]（可能为空列表）。

    一张图可能含多段胶条（如左右两段），因此返回所有有效 ROI，
    而非单一最大框。YOLO 到位后 YOLOLocator 同样返回列表。
    """

    def locate(self, bgr: np.ndarray):
        raise NotImplementedError


class RedBoxLocator(Locator):
    """从烧录了红框的标注图解析红色 GAD 框（最大范围框）。

    返回该图所有有效红框（按 x 升序：左段在前、右段在后）；
    过滤掉面积过小或高度过低、明显是红点/杂点的伪框。
    """

    def __init__(self, min_area: int = 2000, min_h: int = 50):
        self.min_area = min_area
        self.min_h = min_h

    def locate(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        R = rgb[:, :, 0].astype(int)
        G = rgb[:, :, 1].astype(int)
        B = rgb[:, :, 2].astype(int)
        red = ((R - np.maximum(G, B)) > 40) & (R > 70)
        cnts, _ = cv2.findContours(red.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [
            (x, y, w, h)
            for (x, y, w, h) in (cv2.boundingRect(c) for c in cnts)
            if w * h >= self.min_area and h >= self.min_h
        ]
        if not boxes:
            return []
        # 按 x 升序：左段在前、右段在后
        boxes.sort(key=lambda b: b[0])
        return boxes


class YOLOLocator(Locator):
    """YOLO 检测红框（GAD_strip 类，YOLOv11n），替代 RedBoxLocator。

    实现直接复用 detect_and_segment/detect_gad_strips 的成熟链路：
        make_detector(weights) -> predict_boxes(gray, det) -> snap_and_pad_boxes(...)
    该链路是生产定版——喂**灰度**图（模型按灰度训练）、对预测框做「贴边吸附 + 外扩」
    （pad_x=14 / pad_y=10 / snap=40，2026-09-29 实测调出的甜点）：既保证 ROI 包住胶条、
    又不过度外扩压住精修。输出格式与 RedBoxLocator 完全一致：
    返回该图所有红框 (x,y,w,h) 的列表（按 x 升序）；无框则返回空列表。
    """

    # detect_and_segment 子包目录（与本项目根同级），用于导入 detect_gad_strips
    _DET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "detect_and_segment")

    def __init__(self, weights=None, imgsz=896, min_conf=0.25):
        if weights is None:
            raise ValueError(
                "YOLOLocator 需要权重路径：YOLOLocator(weights='models/weights/gad_yolo11n.pt')；"
                "若尚未就绪请改用 RedBoxLocator()。"
            )
        self.weights = weights
        self.imgsz = imgsz
        self.min_conf = min_conf
        # 延迟导入：未装 ultralytics/torch、或 detect_and_segment 不可用时不影响 RedBoxLocator
        if self._DET_DIR not in sys.path:
            sys.path.insert(0, self._DET_DIR)
        import detect_gad_strips as D

        self.D = D
        # make_detector 会打印框宽容量自检（reg_max / max_stride / 是否上限会咬人）
        self.detector = D.make_detector(weights, conf=min_conf, imgsz=imgsz)

    def locate(self, bgr: np.ndarray):
        # 模型按灰度训练，喂灰度最贴合训练分布（彩色图先转灰）
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        D = self.D
        raw = D.predict_boxes(gray, self.detector)
        if not raw:
            return []
        # snap_and_pad：贴边吸附 + 外扩（pad_x/pad_y/snap 用 detect_gad_strips 的定版常数）。
        # 这套参数是「包得住胶条且不偏大」的甜点（取代此前自建的 inset 收缩）。
        rois = D.snap_and_pad_boxes(
            raw, gray.shape,
            pad_x=D.DET_ROI_PAD_X, pad_y=D.DET_ROI_PAD_Y,
            snap=D.DET_SNAP_TOL, wide_fill=D.DET_WIDE_FILL,
        )
        if not rois:
            return []
        # 按 x 升序：左段在前、右段在后（与 RedBoxLocator / 下游 L/R 标记一致）
        rois.sort(key=lambda bb: bb[0])
        return rois


def make_locator(prefer_yolo: bool = True, yolo_weights: str | None = None):
    """工厂：默认优先返回 YOLOLocator（若权重存在且可加载），否则退回 RedBoxLocator。

    下游只需调用 make_locator()，无需关心当前用哪一套定位；YOLO 不可用时自动降级，
    保证 run_metrics_48 等批量脚本在无权重/无依赖环境下仍能跑。

    默认权重指向 models/weights/gad_yolo11n.pt（detect_and_segment 的正式交付位模型）。
    """
    if prefer_yolo:
        wp = yolo_weights or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "models", "weights", "gad_yolo11n.pt",
        )
        if os.path.exists(wp):
            try:
                return YOLOLocator(weights=wp)
            except Exception as e:  # 权重损坏 / 依赖缺失等，降级到红框解析
                print(f"[make_locator] YOLO 加载失败，退回 RedBoxLocator: {e}")
    return RedBoxLocator()
