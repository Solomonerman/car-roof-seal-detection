# -*- coding: utf-8 -*-
"""模块② 胶条分割：封装 crop_and_segment，ROI -> mask。

算法一行不动，只做调用并把返回归一化为 (mask, notes)。
失败返回 (None, [err])，便于上层记录而非崩。
"""
from __future__ import annotations

import numpy as np
import crop_and_segment as C


def segment(roi_gray: np.ndarray, roi_origin, image_shape):
    """在红框 ROI 内分割胶条。

    roi_gray   : 灰度 ROI（已由模块①裁出）
    roi_origin : (x, y) ROI 在原图坐标（用于画面边规则）
    image_shape: 原图 (H, W)
    返回 (mask, notes)；失败 (None, [err])
    """
    mask0 = C.segment_dark_strip(roi_gray)
    if mask0 is None:
        return None, ["segment_dark_strip 未分割到深色胶条"]
    mask, notes = C.refine_strip_mask(
        roi_gray, mask0, tuple(roi_origin), tuple(image_shape)
    )
    if mask is None or not mask.any():
        return None, (notes + ["refine_strip_mask 结果为空"])
    return mask, notes
