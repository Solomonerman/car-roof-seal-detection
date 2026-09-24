# -*- coding: utf-8 -*-
"""模块④ 偏移判定 v15：提取胶条下方白色参考线，算与中心线距离判偏移。

用户确认要点：
- 白线不是一根全局硬直线：允许极微小圆弧（曲率约束二次拟合，弧度非常小）。
- 偏移基准 = 白线 → 胶条中心线（非右侧金属边）。
  offset(x) = (白线y(x) − 中心线y(x)) × MM_PER_PX ；白线为全局线，中心线每段各一。
- 白线可能缺失 -> 先判存在（白线点 < 阈值即报 no_white_pts），不误报偏移。

流程（每张图）：
  红框定位 boxes -> 逐列在胶条底沿下方窗口 [sb+1, sb+100] 取最亮行 argmax -> 白线点 ->
  曲率约束稳健二次拟合（离群点迭代剔除；弧度过大或内点率不足则降级 RANSAC 直线） ->
  每段胶条最小二乘中心线 -> 在段中心列取 白线y − 中心线y 得该段 offset。

注意：segment / clean_mask 由调用方传入（seal_segment.segment / seal_metrics.clean_mask），
避免循环 import，也复用主流程已验证的分割逻辑。
"""
from __future__ import annotations

import numpy as np
import cv2

MM_PER_PX = 0.30466

# ---------- v15 默认参数 ----------
WHITE_SEARCH_H = 100       # 白线搜索窗口高度(px)：从胶条底沿 sb 往下找 100 行
WHITE_MIN_PEAK = 150       # 列窗口最亮值下限，过滤过暗/无效列（噪点）
CURVE_LAM_A = 1e6          # 二次拟合二次项系数 L2 正则：把弧度压到极小
CURVE_MAX_ABS_A = 5e-5     # 二次项系数上限：超过=弧度过大 -> 降级直线
CURVE_OUTLIER_PX = 8.0     # 离群点残差阈值(px)：用于迭代剔除
CURVE_MIN_INLIER = 0.4     # 内点率下限：低于=离群太多 -> 降级 RANSAC
MIN_WHITE_PTS = 10         # 白线点最少数量，少于则判白线缺失


# ---------- 基础拟合 ----------
def fit_poly1(xs, ys):
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    A = np.vstack([xs, np.ones_like(xs)]).T
    (a, b), _, _, _ = np.linalg.lstsq(A, ys, rcond=None)
    return float(a), float(b)


def fit_poly2_reg(xs, ys, lam_a=CURVE_LAM_A):
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    if len(np.unique(xs)) < 3:
        a, b = fit_poly1(xs, ys)
        return 0.0, a, b
    A = np.vstack([xs ** 2, xs, np.ones_like(xs)]).T
    AtA = A.T @ A
    AtA[0, 0] += lam_a          # L2 正则锁定二次项系数 -> 仅允许极微小圆弧
    theta = np.linalg.solve(AtA, A.T @ ys)
    return float(theta[0]), float(theta[1]), float(theta[2])


def ransac_line(xs, ys, max_iter=400, thresh_px=6.0, min_ratio=0.5):
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    n = len(xs)
    if n < 6:
        a, b = fit_poly1(xs, ys)
        return a, b, np.arange(n)
    best = None
    best_inliers = None
    for _ in range(max_iter):
        i, j = np.random.choice(n, 2, replace=False)
        x1, y1 = xs[i], ys[i]
        x2, y2 = xs[j], ys[j]
        if abs(x2 - x1) < 1:
            continue
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1
        resid = np.abs(ys - (a * xs + b))
        inliers = np.where(resid < thresh_px)[0]
        if best is None or len(inliers) > len(best_inliers):
            best = (a, b)
            best_inliers = inliers
        if len(inliers) >= min_ratio * n:
            break
    if best_inliers is not None and len(best_inliers) >= 4:
        a, b = fit_poly1(xs[best_inliers], ys[best_inliers])
        return a, b, best_inliers
    a, b = fit_poly1(xs, ys)
    return a, b, np.arange(n)


def robust_curve(xs, ys, outlier_thresh_px=CURVE_OUTLIER_PX, max_iter=5,
                  lam_a=CURVE_LAM_A, max_abs_a=CURVE_MAX_ABS_A, min_inlier_ratio=CURVE_MIN_INLIER):
    """曲率约束稳健拟合：先二次(迭代剔除离群点)，不稳则降级 RANSAC 直线。
    返回 (mode, params, mask, x_min, x_max)。
    """
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    mask = np.ones(len(xs), dtype=bool)
    for _ in range(max_iter):
        if mask.sum() < 6:
            break
        a, b, c = fit_poly2_reg(xs[mask], ys[mask], lam_a=lam_a)
        pred = a * xs ** 2 + b * xs + c
        resid = np.abs(ys - pred)
        new_mask = resid < outlier_thresh_px
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
    a, b, c = fit_poly2_reg(xs[mask], ys[mask], lam_a=lam_a)
    inlier_ratio = mask.sum() / len(xs)
    use_quad = (abs(a) <= max_abs_a) and (inlier_ratio >= min_inlier_ratio)
    if use_quad:
        x_min = float(xs[mask].min())
        x_max = float(xs[mask].max())
        return "quad", (a, b, c), mask, x_min, x_max
    # 降级：RANSAC 直线，抗离群更稳
    a_l, b_l, inliers = ransac_line(xs, ys, thresh_px=outlier_thresh_px, min_ratio=min_inlier_ratio)
    mask_line = np.zeros(len(xs), dtype=bool)
    mask_line[inliers] = True
    return "line", (a_l, b_l), mask_line, float(xs[mask_line].min()), float(xs[mask_line].max())


# ---------- 白线点检测 ----------
def detect_white_pts(gray, boxes, search_h=WHITE_SEARCH_H, min_peak=WHITE_MIN_PEAK,
                     segment=None, clean_mask=None):
    """逐列在胶条底沿下方窗口取最亮行 -> 白线点 (x, y)。
    返回 (pts ndarray(N,2) 或 None, total_cols)。
    segment / clean_mask 必传，避免重复分割与循环 import。
    """
    if segment is None or clean_mask is None:
        raise RuntimeError("detect_white_pts 需传入 segment=seal_segment.segment, clean_mask=seal_metrics.clean_mask")
    H, W = gray.shape
    pts = []
    total_cols = 0
    for (x, y, w, h) in boxes:
        roi = gray[y:y + h, x:x + w]
        mask, _ = segment(roi, (x, y), gray.shape)
        cm = clean_mask(mask)
        cols = np.any(cm > 0, axis=0)
        for cx in range(w):
            if not cols[cx]:
                continue
            total_cols += 1
            rows = np.nonzero(cm[:, cx])[0]
            sb = y + int(rows.max())                 # 胶条底沿
            y0 = sb + 1
            y1 = min(sb + search_h, H)
            if y1 <= y0:
                continue
            win = gray[y0:y1, x + cx]
            if int(win.max()) < min_peak:           # 过暗列丢弃
                continue
            yp = y0 + int(np.argmax(win))           # 窗口内最亮行 = 白线点
            pts.append((x + cx, yp))
    if len(pts) >= MIN_WHITE_PTS:
        return np.array(pts, dtype=float), total_cols
    return None, total_cols


def centerline_fit(gray, box, segment=None, clean_mask=None):
    """胶条段最小二乘中心线 (a, b)：y = a*x + b。失败返回 None。"""
    if segment is None or clean_mask is None:
        raise RuntimeError("centerline_fit 需传入 segment / clean_mask")
    x, y, w, h = box
    roi = gray[y:y + h, x:x + w]
    mask, _ = segment(roi, (x, y), gray.shape)
    cm = clean_mask(mask)
    cols = np.any(cm > 0, axis=0)
    pts = []
    for cx in range(w):
        if not cols[cx]:
            continue
        rows = np.nonzero(cm[:, cx])[0]
        if rows.size == 0:
            continue
        cy = y + int((rows.min() + rows.max()) / 2)
        pts.append((x + cx, cy))
    if len(pts) < 5:
        return None
    P = np.array(pts)
    A = np.vstack([P[:, 0], np.ones_like(P[:, 0])]).T
    (a, b), _, _, _ = np.linalg.lstsq(A, P[:, 1], rcond=None)
    return float(a), float(b)


class WhiteLineModel:
    """拟合出的白线，可按 x 求 y。区间内用拟合；区间外 clamp 端点（不外推防崩溃）。"""

    def __init__(self, mode, params, x_min, x_max):
        self.mode = mode
        self.params = params
        self.x_min = x_min
        self.x_max = x_max

    def y_at(self, x):
        x = float(x)
        if self.mode == "quad":
            a, b, c = self.params
            if self.x_min <= x <= self.x_max:
                return a * x ** 2 + b * x + c
            if x < self.x_min:
                y0 = a * self.x_min ** 2 + b * self.x_min + c
                dy = 2 * a * self.x_min + b
                return y0 + dy * (x - self.x_min)
            y1 = a * self.x_max ** 2 + b * self.x_max + c
            dy = 2 * a * self.x_max + b
            return y1 + dy * (x - self.x_max)
        else:  # line
            a, b = self.params
            if x < self.x_min:
                return a * self.x_min + b
            if x > self.x_max:
                return a * self.x_max + b
            return a * x + b


# ---------- 对外主接口 ----------
def compute_offsets(gray, boxes, mm_per_px=MM_PER_PX, segment=None, clean_mask=None):
    """对每张图的各胶条段算 白线->中心线 偏移。
    返回 list[dict]: seg, side, offset_mm, fit_mode, white_pts, inlier_pts, coverage, status。
    status: ok / no_white_pts / no_centerline。
    白线为全局一条（所有 boxes 的点一起拟合），中心线每段各一。
    """
    if segment is None or clean_mask is None:
        raise RuntimeError("compute_offsets 需传入 segment=seal_segment.segment, clean_mask=seal_metrics.clean_mask")
    H, W = gray.shape
    n = len(boxes)
    wp, total_cols = detect_white_pts(gray, boxes, segment=segment, clean_mask=clean_mask)
    if wp is None:
        return [dict(seg=i, side=_side(n, i), offset_mm=None, fit_mode=None,
                     white_pts=0, inlier_pts=0, coverage=0.0, status="no_white_pts") for i in range(n)]
    mode, params, mask, x_min, x_max = robust_curve(wp[:, 0], wp[:, 1])
    wmodel = WhiteLineModel(mode, params, x_min, x_max)
    n_inlier = int(mask.sum())
    n_total = len(wp)
    cov = round(n_total / max(1, total_cols), 3)
    out = []
    for i, box in enumerate(boxes):
        cf = centerline_fit(gray, box, segment=segment, clean_mask=clean_mask)
        if cf is None:
            out.append(dict(seg=i, side=_side(n, i), offset_mm=None, fit_mode=mode,
                            white_pts=n_total, inlier_pts=n_inlier, coverage=cov, status="no_centerline"))
            continue
        a_c, b_c = cf
        x, y, w, h = box
        xc = x + w // 2                       # 段中心列
        wy = wmodel.y_at(xc)                  # 白线 y
        cy = a_c * xc + b_c                  # 中心线 y
        off = round((wy - cy) * mm_per_px, 2)
        out.append(dict(seg=i, side=_side(n, i), offset_mm=off, fit_mode=mode,
                        white_pts=n_total, inlier_pts=n_inlier, coverage=cov, status="ok"))
    return out


def _side(n, i):
    if n == 2:
        return "L" if i == 0 else "R"
    return f"S{i}"
