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

RANSAC_SEED = 12345         # 固定种子：让 RANSAC 直线降级可复现（否则每次随机抽中不同内点，偏移乱跳）

# ---------- 偏移公差带与偏离判定（用户拍板）----------
OFFSET_SPEC_LO = 15.0      # 偏移下限(mm)：低于=胶条离白线过近
OFFSET_SPEC_HI = 30.0      # 偏移上限(mm)：高于=胶条离白线过远
OFFSET_MIN_RUN_COLS = 15   # 连续超差列数阈值：≥此值才判真实偏离；单点/少数噪点=误报


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


def ransac_line(xs, ys, max_iter=400, thresh_px=6.0, min_ratio=0.5, seed=RANSAC_SEED):
    np.random.seed(seed)        # 固定种子：结果可复现
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
    # 降级直线：RANSAC（固定种子，可复现）收敛到点数最多的内点簇，
    # 对双峰离群（如 idx47 的杂亮带）能稳定选中真实白线簇，偏移不再随机跳。
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


def centerline_points(gray, box, segment=None, clean_mask=None):
    """胶条段逐列中心线点 (x, y)：每列取上下边缘中点。保留真实 tilt（不拟合直线）。
    返回 (xs ndarray, ys ndarray) 或 None。供逐点测距离用。
    """
    if segment is None or clean_mask is None:
        raise RuntimeError("centerline_points 需传入 segment / clean_mask")
    x, y, w, h = box
    roi = gray[y:y + h, x:x + w]
    mask, _ = segment(roi, (x, y), gray.shape)
    cm = clean_mask(mask)
    cols = np.any(cm > 0, axis=0)
    xs = []
    ys = []
    for cx in range(w):
        if not cols[cx]:
            continue
        rows = np.nonzero(cm[:, cx])[0]
        if rows.size == 0:
            continue
        cy = y + int((rows.min() + rows.max()) / 2)
        xs.append(x + cx)
        ys.append(cy)
    if len(xs) < 5:
        return None
    return np.array(xs, dtype=float), np.array(ys, dtype=float)


def find_dev_runs(xs, d_mm, out_of_spec, min_run=OFFSET_MIN_RUN_COLS):
    """在逐列 out_of_spec 布尔序列中找连续超差段；连续长度 ≥ min_run 才判真实偏离。
    返回 (runs, deviated)。runs 每项:
      {start_col,end_col,start_x,end_x,lo_mm,hi_mm,direction,run_cols}。
    direction: 'low'(全段<下限) / 'high'(全段>上限) / 'both'。
    说明：连续按"逐列测量点"计（胶条列基本连续，≈连续列）；单点/少数噪点不构成 run。
    """
    runs = []
    n = len(out_of_spec)
    i = 0
    while i < n:
        if out_of_spec[i]:
            j = i
            while j + 1 < n and out_of_spec[j + 1]:
                j += 1
            run_len = j - i + 1
            if run_len >= min_run:
                seg_xs = xs[i:j + 1]
                seg_d = d_mm[i:j + 1]
                if seg_d.max() < OFFSET_SPEC_LO:
                    direction = "low"
                elif seg_d.min() > OFFSET_SPEC_HI:
                    direction = "high"
                else:
                    direction = "both"
                runs.append(dict(
                    start_col=int(i), end_col=int(j),
                    start_x=int(round(float(seg_xs[0]))), end_x=int(round(float(seg_xs[-1]))),
                    lo_mm=float(seg_d.min()), hi_mm=float(seg_d.max()),
                    direction=direction, run_cols=run_len,
                ))
            i = j + 1
        else:
            i += 1
    return runs, len(runs) > 0


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
    """对每张图的各胶条段算 白线->中心线 偏移，并逐点检测 Z 向旋转偏离。

    返回 list[dict]:
      seg, side, offset_mm(段中心列代表值), fit_mode, white_pts, inlier_pts, coverage, status,
      per_col({xs, cl_y, white_y, d_mm} 或 None), deviated(bool), dev_runs(list of region dict)。
    status: ok / no_white_pts / no_centerline。
    白线为全局一条（所有 boxes 的点一起拟合），中心线每段各一。

    偏离判定（用户需求）：
      逐列算 距离(x) = (白线y − 中心线y) × MM_PER_PX；
      落在区间 [OFFSET_SPEC_LO, OFFSET_SPEC_HI] 外 = 该点超差；
      仅当连续超差列数 ≥ OFFSET_MIN_RUN_COLS 才判真实偏离(deviated=True)；
      单点或少数噪点超差视为误报，不触发。
    """
    if segment is None or clean_mask is None:
        raise RuntimeError("compute_offsets 需传入 segment=seal_segment.segment, clean_mask=seal_metrics.clean_mask")
    H, W = gray.shape
    n = len(boxes)
    wp, total_cols = detect_white_pts(gray, boxes, segment=segment, clean_mask=clean_mask)
    if wp is None:
        return [dict(seg=i, side=_side(n, i), offset_mm=None, fit_mode=None,
                     white_pts=0, inlier_pts=0, coverage=0.0, status="no_white_pts",
                     per_col=None, deviated=False, dev_runs=[]) for i in range(n)]
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
                            white_pts=n_total, inlier_pts=n_inlier, coverage=cov, status="no_centerline",
                            per_col=None, deviated=False, dev_runs=[]))
            continue
        a_c, b_c = cf
        x, y, w, h = box
        xc = x + w // 2                       # 段中心列
        wy = wmodel.y_at(xc)                  # 白线 y
        cy = a_c * xc + b_c                  # 中心线 y
        off = round((wy - cy) * mm_per_px, 2)

        # ---- 逐列距离 + 偏离判定 ----
        cpts = centerline_points(gray, box, segment=segment, clean_mask=clean_mask)
        if cpts is None:
            per_col = None
            deviated = False
            dev_runs = []
        else:
            cxs, cys = cpts
            order = np.argsort(cxs)
            cxs = cxs[order]
            cys = cys[order]
            wys = np.array([wmodel.y_at(float(v)) for v in cxs])
            d_mm = (wys - cys) * mm_per_px
            out_of_spec = (d_mm < OFFSET_SPEC_LO) | (d_mm > OFFSET_SPEC_HI)
            dev_runs, deviated = find_dev_runs(cxs, d_mm, out_of_spec,
                                               min_run=OFFSET_MIN_RUN_COLS)
            per_col = dict(xs=cxs.tolist(), cl_y=cys.tolist(),
                           white_y=wys.tolist(), d_mm=d_mm.tolist())
        out.append(dict(seg=i, side=_side(n, i), offset_mm=off, fit_mode=mode,
                        white_pts=n_total, inlier_pts=n_inlier, coverage=cov, status="ok",
                        per_col=per_col, deviated=deviated, dev_runs=dev_runs))
    return out


def draw_offset_vis(bgr, boxes, off_recs, mm_per_px=MM_PER_PX):
    """偏移可视化：原图叠加 白线(红点) + 逐列中心线(绿点) + 超差段橙色高亮 + 段中心距离黄线。
    返回 BGR 图。off_recs 为 compute_offsets 返回（需含 per_col / dev_runs），与 boxes 同序。
    """
    vis = bgr.copy()
    for (x, y, w, h), r in zip(boxes, off_recs):
        pc = r.get("per_col")
        if not pc:
            continue
        xs = np.array(pc["xs"], float)
        cl_y = np.array(pc["cl_y"], float)
        w_y = np.array(pc["white_y"], float)
        d_mm = np.array(pc["d_mm"], float)
        # 绿点 = 逐列中心线（保留 tilt）；红点 = 白线
        for xi, yi in zip(xs, cl_y):
            vis[int(round(yi)), int(round(xi))] = (0, 255, 0)
        for xi, yi in zip(xs, w_y):
            vis[int(round(yi)), int(round(xi))] = (0, 0, 255)
        # 橙色高亮：超差列画 中心线->白线 竖线（黄变橙）
        oos = (d_mm < OFFSET_SPEC_LO) | (d_mm > OFFSET_SPEC_HI)
        for xi, yi_c, yi_w in zip(xs[oos], cl_y[oos], w_y[oos]):
            cv2.line(vis, (int(round(xi)), int(round(yi_c))),
                     (int(round(xi)), int(round(yi_w))), (0, 165, 255), 2)
        # 段中心列黄线（代表值，便于对照上一版）
        xc = x + w // 2
        if r.get("offset_mm") is not None and len(xs):
            k = int(np.argmin(np.abs(xs - xc)))
            cv2.line(vis, (int(round(xs[k])), int(round(cl_y[k]))),
                     (int(round(xs[k])), int(round(w_y[k]))), (0, 255, 255), 2)
        # 偏离段文字标注
        if r.get("deviated"):
            yy = int(round(min(cl_y.min(), w_y.min()))) - 10
            cv2.putText(vis, "DEVIATED", (int(round(xs[0])), max(0, yy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    return vis


def _side(n, i):
    if n == 2:
        return "L" if i == 0 else "R"
    return f"S{i}"
