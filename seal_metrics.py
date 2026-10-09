# -*- coding: utf-8 -*-
"""模块③ 形态处理 + 长宽度计算 + 骨架中心线。

输入 mask（来自模块②），输出：
  - 去毛刺后的 mask（形态学开闭 + 丢弃小连通块）
  - 逐列上下边界 (top, bot)
  - 逐列宽度(mm) 与统计（median/mean/std/P5/P95/极值/公差带内占比）
  - 胶条长度(mm)
  - 中心线（逐列中点，胶条基本水平时足够稳）
"""
from __future__ import annotations

import cv2
import numpy as np

MM_PER_PX = 0.30466  # 左相机像素当量；后续按相机标定替换


def clean_mask(mask: np.ndarray, open_ks: int = 3, open_h: int = 5, min_area_px: int = 50,
               trim_end_bumps: bool = True, end_frac: float = 0.10,
               bot_thresh_px: int = 8, width_thresh_mm: float = 1.0) -> np.ndarray:
    """形态学去毛刺：开运算去细突出 + 闭运算补洞 + 丢弃小连通块，保留最大块。

    - open_ks：对称开/闭核边长(默认 3×3)，去掉单像素毛刺、补小洞；
    - open_h：额外“横向开运算”核宽(1×open_h)，专门吃掉水平方向窄于该值的
      竖向细突刺(如算法端头检测在胶条端部产生的向下小凸起)。胶条主体宽度
      远大于 open_h 时完全不受影响。设为 0/1 则跳过该步(退化为仅对称开闭)。
    - trim_end_bumps：是否在形态学后再做一次端部鼓包截断。端部（左右各 end_frac）
      若同时满足"下沿超出主体趋势 bot_thresh_px 像素"且"宽度超出主体中位数
      width_thresh_mm 毫米"，则把该列下沿拉回主体趋势线。此条件可去掉形态学
      吃不掉的宽平台式端部鼓包，同时保护正常收尾。
    """
    m = (mask > 0).astype(np.uint8)
    if open_ks > 1:
        k = np.ones((open_ks, open_ks), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if open_h and open_h > 1:
        kh = np.ones((1, open_h), np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kh)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        m_out = m
    else:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = [i + 1 for i in range(len(areas)) if areas[i] >= min_area_px]
        if not keep:
            keep = [int(np.argmax(areas)) + 1]
        m_out = np.isin(labels, keep).astype(np.uint8)

    if trim_end_bumps:
        m_out = _trim_end_bumps(m_out, end_frac=end_frac,
                                bot_thresh_px=bot_thresh_px,
                                width_thresh_mm=width_thresh_mm)
    return m_out


def _trim_end_bumps(mask: np.ndarray, end_frac: float = 0.10,
                    bot_thresh_px: int = 8, width_thresh_mm: float = 1.0) -> np.ndarray:
    """端部鼓包截断：仅处理左右各 end_frac 的列，要求同时满足下沿超趋势+宽度超主体。

    用中间 80% 列拟合下沿线性趋势并计算主体宽度中位数；端部列若同时满足
    bot - trend > bot_thresh_px 且 width - body_median > width_thresh_mm，
    则把该列下沿拉回趋势预测位置。保护正常单调收尾不误截。
    """
    top, bot = col_bounds(mask)
    xs = np.where(np.isfinite(top) & np.isfinite(bot))[0]
    if len(xs) < 20:
        return mask
    n = len(xs)
    trim = max(1, int(n * 0.1))
    widths = per_column_width_mm(top, bot)
    body_mm = float(np.median(widths[xs[trim:-trim]]))
    core_x = xs[trim:-trim].astype(float)
    core_y = bot[xs[trim:-trim]]
    valid = np.isfinite(core_y)
    if valid.sum() < 5:
        return mask
    A = np.vstack([core_x[valid], np.ones(valid.sum())]).T
    a, b = np.linalg.lstsq(A, core_y[valid], rcond=None)[0]
    end_n = max(1, int(n * end_frac))
    bot_new = bot.copy()
    for side in (0, 1):
        idxs = xs[:end_n] if side == 0 else xs[-end_n:]
        for c in idxs:
            pred = a * c + b
            over_bot = bot[c] - pred
            over_w = widths[c] - body_mm
            if over_bot > bot_thresh_px and over_w > width_thresh_mm:
                bot_new[c] = max(int(top[c] + 1), int(round(pred)))
    m = mask.copy()
    for c in xs:
        t = int(top[c])
        b = int(bot_new[c])
        m[:, c] = 0
        if b >= t:
            m[t:b+1, c] = 1
    return m


def col_bounds(mask: np.ndarray):
    """逐列上下边界；无掩膜像素列置 NaN。返回 (top, bot)。"""
    ys, xs = np.nonzero(mask)
    top = np.full(mask.shape[1], np.nan)
    bot = np.full(mask.shape[1], np.nan)
    if xs.size == 0:
        return top, bot
    for x in range(mask.shape[1]):
        col = ys[xs == x]
        if col.size:
            top[x] = col.min()
            bot[x] = col.max()
    return top, bot


def per_column_width_mm(top, bot, mm_per_px: float = MM_PER_PX) -> np.ndarray:
    """逐列宽度(mm)，含 NaN。"""
    return (bot - top) * mm_per_px


def width_stats(widths_mm: np.ndarray, spec=(15.0, 30.0)) -> dict:
    """宽度统计：中位数/均值/标准差/P5/P95/极值/公差带内占比。

    spec 默认 20±5mm；in_spec_pct = 落在 [spec_low, spec_high] 的列占比(%)。
    """
    v = widths_mm[np.isfinite(widths_mm)]
    if v.size == 0:
        return {"n": 0}
    p5, p50, p95 = np.percentile(v, [5, 50, 95])
    in_spec = float(np.sum((v >= spec[0]) & (v <= spec[1])) / v.size * 100.0)
    return {
        "n": int(v.size),
        "median_mm": round(float(p50), 2),
        "mean_mm": round(float(v.mean()), 2),
        "std_mm": round(float(v.std()), 2),
        "p5_mm": round(float(p5), 2),
        "p95_mm": round(float(p95), 2),
        "min_mm": round(float(v.min()), 2),
        "max_mm": round(float(v.max()), 2),
        "in_spec_pct": round(in_spec, 1),
        "spec_mm": spec,
    }


def length_mm(mask: np.ndarray, mm_per_px: float = MM_PER_PX):
    """胶条长度 = 有掩膜列的像素跨度 × mm。返回 (length_mm, n_cols)。"""
    n_cols = int(np.count_nonzero(mask.sum(axis=0)))
    return round(n_cols * mm_per_px, 2), n_cols


def skeleton_centerline(top, bot):
    """逐列中点作为中心线（胶条基本水平，足够稳）。返回 (xs, mid_y)。"""
    xs = np.arange(len(top))
    valid = np.isfinite(top) & np.isfinite(bot)
    mid = np.where(valid, (top + bot) / 2.0, np.nan)
    return xs[valid].astype(int), mid[valid]
