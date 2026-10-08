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
CURVE_MAX_ABS_A = 1e-5     # v6：收紧（原5e-5）。紫线实测为完美直线(1阶残差0.77px)，放宽二次项会在长段两端发散(06段曾偏35px)
CURVE_OUTLIER_PX = 8.0     # 离群点残差阈值(px)：用于迭代剔除
CURVE_MIN_INLIER = 0.4     # 内点率下限：低于=离群太多 -> 降级 RANSAC
MIN_WHITE_PTS = 10         # 白线点最少数量，少于则判白线缺失

# ---------- v4 稳定性边界法参数（针对下移白线）----------
WHITE_PROFILE_LEN = 200    # 从胶条底沿往下取的剖面长度(px)
WHITE_STAB_WIN = 40        # 稳定性判定窗口(px)：往下扫时看后续多少行
WHITE_STAB_STD = 8.0       # 稳定性判定标准差阈值：小于此值=进入稳定淡灰车身区
WHITE_DEPTH_MIN = 10       # 扫描起始深度(px)：从紧贴胶条底沿处开始扫
WHITE_DEPTH_MAX = 150      # 扫描终止深度(px)
WHITE_DEPTH_LO = 15        # v6：有效白线深度下限(px)。稳定点出现过浅(贴胶条底沿)=该列不可靠，丢弃
WHITE_DEPTH_HI = 110       # v6：有效白线深度上限(px)。稳定点出现过深=误抓远处车身亮带，丢弃

RANSAC_SEED = 12345         # 固定种子：让 RANSAC 直线降级可复现（否则每次随机抽中不同内点，偏移乱跳）

# ---------- 偏移公差带与偏离判定（用户拍板）----------
OFFSET_SPEC_LO = 32.0      # v7c：偏移下限(mm) 由 30.0 上调为 32.0（用户拍板：距离<32 或 >40mm 判偏离）
OFFSET_SPEC_HI = 40.0      # v7：偏移上限(mm) 由 30.0 上调为 40.0（同上，公差带整体 +10mm）
OFFSET_MIN_RUN_COLS = 15   # 连续超差列数阈值：≥此值才计为「有效偏离段」；单点/少数噪点=误报
OFFSET_DEV_FRAC = 0.50     # v7d：整段偏移占比门槛。有效偏离段总长 / 整段评估列数 > 此值 才判整段偏移(deviated=True)；否则视为局部波动、整段正常

# ---------- 偏移报警分级（基于段代表偏移 offset_mm，单位 mm）----------
# 分级区间（用户 2026-10-08 拍板，按"偏移距离"分 4 级）：
#   正常     : [33.00, 35.00)                  -> level 0
#   轻微偏移 : [30.00, 33.00) ∪ [35.00, 38.00) -> level 1（可接受，不报警）
#   中度偏移 : [28.00, 30.00) ∪ [38.00, 40.00) -> level 2（报警提醒工人注意胶条状态）
#   漏水风险 : < 28.00 或 >= 40.00             -> level 3（需工人返工）
OFFSET_LV_NORMAL_LO = 33.00
OFFSET_LV_NORMAL_HI = 35.00
OFFSET_LV_SLIGHT_LO = 30.00
OFFSET_LV_SLIGHT_HI = 38.00
OFFSET_LV_MODERATE_LO = 28.00
OFFSET_LV_MODERATE_HI = 40.00
OFFSET_LEVELS = {0: "正常", 1: "轻微偏移", 2: "中度偏移", 3: "漏水风险"}
OFFSET_LEVEL_UNKNOWN = -1
OFFSET_LEVEL_UNKNOWN_NAME = "未检出"


def classify_offset_level(x):
    """按段代表偏移 x(mm) 返回 (level:int, name:str)。x 为 None 时返回未检出。"""
    if x is None:
        return OFFSET_LEVEL_UNKNOWN, OFFSET_LEVEL_UNKNOWN_NAME
    if x < OFFSET_LV_MODERATE_LO:           # < 28.00
        return 3, OFFSET_LEVELS[3]
    if x < OFFSET_LV_SLIGHT_LO:             # [28.00, 30.00)
        return 2, OFFSET_LEVELS[2]
    if x < OFFSET_LV_NORMAL_LO:             # [30.00, 33.00)
        return 1, OFFSET_LEVELS[1]
    if x < OFFSET_LV_NORMAL_HI:             # [33.00, 35.00)
        return 0, OFFSET_LEVELS[0]
    if x < OFFSET_LV_SLIGHT_HI:             # [35.00, 38.00)
        return 1, OFFSET_LEVELS[1]
    if x < OFFSET_LV_MODERATE_HI:           # [38.00, 40.00)
        return 2, OFFSET_LEVELS[2]
    return 3, OFFSET_LEVELS[3]              # >= 40.00
SEAL_END_TRIM = 0.10       # 距离检测去掉每段首/尾 10%（两端偏差大，不计入中心线距离）


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


# ---------- 白线点检测 (v4：稳定性边界法) ----------
def detect_white_pts(gray, boxes, search_h=WHITE_SEARCH_H, min_peak=WHITE_MIN_PEAK,
                     segment=None, clean_mask=None,
                     stab_win=WHITE_STAB_WIN, stab_std=WHITE_STAB_STD):
    """逐列找白线：上方"反光活跃区"与下方"稳定淡灰车身区"的分界。

    背景：这批图白线整体下移到胶条底沿下方 30~70px。白线上方存在比白线更亮的
    反光(197~255)，故 argmax 会抓错；而白线下方是一大片车身淡灰区(125~135)，
    灰度非常稳定。因此用「从胶条底沿往下扫，第一个之后 stab_win 长度内标准差
    < stab_std 的位置」作为白线，可绕开反光干扰。

    返回 (pts ndarray(N,2) 或 None, total_cols)。
    """
    if segment is None or clean_mask is None:
        raise RuntimeError("detect_white_pts 需传入 segment=seal_segment.segment, clean_mask=seal_metrics.clean_mask")
    H, W = gray.shape
    pts = []
    total_cols = 0
    d_min, d_max = WHITE_DEPTH_MIN, WHITE_DEPTH_MAX

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
            y1 = min(sb + 1 + WHITE_PROFILE_LEN, H)
            if y1 - y0 < 80:
                continue
            prof = gray[y0:y1, x + cx].astype(float)
            # 主策略：稳定性边界（优化：向量化滑动标准差，等价原逐深度 Python 循环，快几十倍）
            # 原实现每列都跑一个「逐深度」Python 循环（~110 次 np.std/列），是全程序最大性能热点；
            # 这里一次性对整条剖面算滑动窗口标准差，取「首个 < stab_std 的深度」，与原逻辑逐字节一致。
            L = prof.shape[0]
            max_dd = min(d_max, L - stab_win)
            d = -1
            if max_dd > d_min:
                win = np.lib.stride_tricks.sliding_window_view(prof, stab_win)  # (L-stab_win+1, stab_win)
                stds = win.std(axis=1)
                dd_range = np.arange(d_min, max_dd)
                ok = stds[dd_range] < stab_std
                if ok.any():
                    d = int(dd_range[ok][0])          # 首个稳定深度（等价原循环 `break`）
            if d < 0:
                # v6：深度窗口内找不到稳定边界 -> 丢弃该列（不 fallback argmax，避免抓到反光）
                continue
            # v6 核心：稳定点必须落在有效深度区间内，否则该列不可靠 -> 丢弃。
            # 过浅(depth<15)：紧贴胶条底沿就"稳定"，多半是胶条本体/暗区，不是白线；
            # 过深(depth>110)：误抓到远处车身亮带。实测这条过滤让 06 段误差 7.0px -> 2.1px。
            depth = d + 1
            if not (WHITE_DEPTH_LO <= depth <= WHITE_DEPTH_HI):
                continue
            yp = y0 + d
            pts.append((x + cx, yp))
    if len(pts) >= MIN_WHITE_PTS:
        return np.array(pts, dtype=float), total_cols
    return None, total_cols






def find_dev_runs(xs, d_mm, out_of_spec, min_run=OFFSET_MIN_RUN_COLS, dev_frac=OFFSET_DEV_FRAC):
    """在逐列 out_of_spec 布尔序列中找连续超差段；连续长度 ≥ min_run 才计为「有效偏离段」。

    整段是否判偏移：所有有效偏离段总长 / 整段评估列数(n) > dev_frac(默认0.50) 才判
    deviated=True；否则视为「局部波动、整段正常」。
    返回 (runs, deviated)。runs 每项:
      {start_col,end_col,start_x,end_x,lo_mm,hi_mm,direction,run_cols}。
    direction: 'low'(全段<下限) / 'high'(全段>上限) / 'both'。
    说明：
      - 连续按"逐列测量点"计（胶条列基本连续，≈连续列）；
      - min_run 仅过滤单点/少数噪点，不进入 dev_len 统计；
      - 真正判整段偏移的是 dev_frac（占比阈值）；n = 评估用到的列数
        （segment_offset 为去首尾10%后的中段80%；compute_offsets 为逐列中心线全长）。
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
    dev_len = sum(r["run_cols"] for r in runs)
    deviated = n > 0 and (dev_len / n) > dev_frac
    return runs, deviated


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
def fit_white_line(gray, boxes, segment=None, clean_mask=None):
    """拟合全局白线（所有 boxes 的白线点一起拟合）。返回 dict 或 None(白线缺失)。
    返回: mode, params, x_min, x_max, white_pts, inlier_pts, coverage。
    白线点检测用原图灰度 gray(与V15一致)，避免 inpaint 把标注框填补后亮度偏移误抓。
    """
    if segment is None or clean_mask is None:
        raise RuntimeError("fit_white_line 需传入 segment=seal_segment.segment, clean_mask=seal_metrics.clean_mask")
    wp, total_cols = detect_white_pts(gray, boxes, segment=segment, clean_mask=clean_mask)
    if wp is None:
        return None
    mode, params, mask, x_min, x_max = robust_curve(wp[:, 0], wp[:, 1])
    return dict(mode=mode, params=params, x_min=x_min, x_max=x_max,
                white_pts=len(wp), inlier_pts=int(mask.sum()),
                coverage=round(len(wp) / max(1, total_cols), 3))


def fit_white_lines_perbox(gray, boxes, segment=None, clean_mask=None):
    """v5 核心改动：每个红框段单独拟合一条白线（不再用全局一条曲线覆盖左右两段）。

    背景（取证依据）：白线是斜线，左右两段的斜率/高度不同，用一条全局曲线拟合会把
    段扯偏（实测 05/07 图右段点准但曲线偏 -13~-12px）。逐段拟合实测：
      全局曲线  中位绝对误差 2.8px(0.86mm) P90=13.0px
      逐段拟合  中位绝对误差 1.7px(0.51mm) P90= 3.2px

    返回 list[dict|None]，与 boxes 同序。某段白线点不足时，回退用全局曲线兜底。
    """
    if segment is None or clean_mask is None:
        raise RuntimeError("fit_white_lines_perbox 需传入 segment=seal_segment.segment, clean_mask=seal_metrics.clean_mask")
    wp, total_cols = detect_white_pts(gray, boxes, segment=segment, clean_mask=clean_mask)
    if wp is None:
        return [None] * len(boxes)
    # 全局曲线：仅作点数不足时的兜底
    g_mode, g_params, g_mask, g_xmin, g_xmax = robust_curve(wp[:, 0], wp[:, 1])
    global_wl = dict(mode=g_mode, params=g_params, x_min=g_xmin, x_max=g_xmax,
                     white_pts=len(wp), inlier_pts=int(g_mask.sum()),
                     coverage=round(len(wp) / max(1, total_cols), 3))
    out = []
    for (x, y, w, h) in boxes:
        sub = wp[(wp[:, 0] >= x) & (wp[:, 0] < x + w)]
        if len(sub) < MIN_WHITE_PTS:
            out.append(dict(global_wl, seg_fit="fallback_global"))
            continue
        mode, params, mask, x_min, x_max = robust_curve(sub[:, 0], sub[:, 1])
        out.append(dict(mode=mode, params=params, x_min=x_min, x_max=x_max,
                        white_pts=len(sub), inlier_pts=int(mask.sum()),
                        coverage=round(len(sub) / max(1, w), 3),
                        seg_fit="perbox"))
    return out


def segment_offset(box, wl, skeleton_pts, n, seg_idx, mm_per_px=MM_PER_PX, end_trim=SEAL_END_TRIM):
    """算单段 白线->中心线 距离(偏移) + 逐点偏离判定。

    wl: fit_white_line 结果(dict) 或 None；
    skeleton_pts: 宽度模块算好的统一骨架 (sxs_global, smid_global) 或 None。
    返回 dict（与 compute_offsets 单段同结构）：
      seg, side, offset_mm(中段80%中位数), fit_mode, white_pts, inlier_pts, coverage,
      status(ok/no_white_pts/no_centerline), cl_source, per_col, offset_trim, deviated, dev_runs。
    """
    if wl is None:
        return dict(seg=seg_idx, side=_side(n, seg_idx), offset_mm=None, fit_mode=None,
                    white_pts=0, inlier_pts=0, coverage=0.0, status="no_white_pts",
                    cl_source=None, per_col=None, offset_trim=None, deviated=False, dev_runs=[],
                    offset_level=OFFSET_LEVEL_UNKNOWN, offset_level_name=OFFSET_LEVEL_UNKNOWN_NAME)
    if skeleton_pts is None:
        return dict(seg=seg_idx, side=_side(n, seg_idx), offset_mm=None, fit_mode=wl["mode"],
                    white_pts=wl["white_pts"], inlier_pts=wl["inlier_pts"], coverage=wl["coverage"],
                    status="no_centerline", cl_source=None, per_col=None, offset_trim=None,
                    deviated=False, dev_runs=[])
    sxs, smid = skeleton_pts
    sxs = np.asarray(sxs, float)
    smid = np.asarray(smid, float)
    if len(sxs) < 5:
        return dict(seg=seg_idx, side=_side(n, seg_idx), offset_mm=None, fit_mode=wl["mode"],
                    white_pts=wl["white_pts"], inlier_pts=wl["inlier_pts"], coverage=wl["coverage"],
                    status="no_centerline", cl_source=None, per_col=None, offset_trim=None,
                    deviated=False, dev_runs=[])
    order = np.argsort(sxs)
    sxs = sxs[order]
    smid = smid[order]
    wmodel = WhiteLineModel(wl["mode"], wl["params"], wl["x_min"], wl["x_max"])
    wys = np.array([wmodel.y_at(float(v)) for v in sxs])
    d_mm = (wys - smid) * mm_per_px
    # 全量逐列（用于画图）；代表值与偏离判定仅用中段 80%（去掉首尾 10%）
    per_col = dict(xs=sxs.tolist(), cl_y=smid.tolist(),
                   white_y=wys.tolist(), d_mm=d_mm.tolist())
    N = len(sxs)
    lo = int(round(N * end_trim))
    hi = int(round(N * (1 - end_trim)))
    if hi - lo < 5:
        lo, hi = 0, N
    rm = slice(lo, hi)
    d_t = d_mm[rm]
    xs_t = sxs[rm]
    offset_median = round(float(np.median(d_t)), 2)
    out_of_spec = (d_t < OFFSET_SPEC_LO) | (d_t > OFFSET_SPEC_HI)
    dev_runs, deviated = find_dev_runs(xs_t, d_t, out_of_spec, min_run=OFFSET_MIN_RUN_COLS)
    lv, ln = classify_offset_level(offset_median)
    return dict(seg=seg_idx, side=_side(n, seg_idx), offset_mm=offset_median, fit_mode=wl["mode"],
                white_pts=wl["white_pts"], inlier_pts=wl["inlier_pts"], coverage=wl["coverage"],
                status="ok", cl_source="skeleton", per_col=per_col,
                offset_trim=(lo, hi), deviated=deviated, dev_runs=dev_runs,
                offset_level=lv, offset_level_name=ln)




# 偏移报警分级配色（BGR）：正常绿 / 轻微黄 / 中度橙 / 漏水风险红 / 未检出灰
_LEVEL_COLOR = {
    0: (0, 255, 0),
    1: (0, 255, 255),
    2: (0, 165, 255),
    3: (0, 0, 255),
}
_LEVEL_LABEL = {
    0: "正常",
    1: "轻微偏移",
    2: "中度偏移!",
    3: "漏水风险!",
}
_UNKNOWN_COLOR = (128, 128, 128)
_UNKNOWN_LABEL = "未检出"


# 偏移区域填充色（BGR）：与"中度偏移"报警橙完全一致 (0,165,255)
OFFSET_FILL_COLOR = (0, 165, 255)
OFFSET_FILL_ALPHA = 0.40   # 半透明：让红/绿点透出，区域本身醒目
# 白线标记点：淡紫（区别于报警色），半径加粗约2倍
WHITE_LINE_COLOR = (190, 90, 230)
WHITE_LINE_RADIUS = 3
# 偏移代表线（段中心列）：青色，避开与"轻微偏移"黄色框撞色
REP_LINE_COLOR = (255, 255, 0)
REP_LINE_THICK = 2
# 文字字号：历史 0.7/0.6 → 第一轮放大1.25倍(0.875/0.75) → 本轮再放大1.25倍
FONT_LEVEL = 1.094    # 0.875 * 1.25
FONT_WIDTH = 0.938    # 0.75 * 1.25
FONT_OFFSET = 0.938   # 0.75 * 1.25
# 胶条检测外轮廓（亮蓝，加粗）：保留算法检出的胶条范围，区别于报警框线/橙带；
# 亮蓝(255,0,0) 与现有 绿/黄/橙/红框线、淡紫白线、青代表线、橙带均不重合
SEAL_CONTOUR_COLOR = (255, 128, 0)
SEAL_CONTOUR_THICK = 3


def _put_text_plate(vis, text, x, y, scale=0.6, color=(255, 255, 255), thickness=2):
    """带黑色底板的文字：保证在任何背景上都清晰，且压在最上层不被覆盖。原地修改 vis。"""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad = 4
    x0 = int(x)
    y0 = int(y) - th - pad
    if x0 < 0:
        x0 = 0
    if y0 < 0:
        y0 = 0
    if x0 + tw + 2 * pad > vis.shape[1]:
        x0 = max(0, vis.shape[1] - (tw + 2 * pad))
    if y0 + th + 2 * pad > vis.shape[0]:
        y0 = max(0, vis.shape[0] - (th + 2 * pad))
    cv2.rectangle(vis, (x0, y0), (x0 + tw + 2 * pad, y0 + th + 2 * pad), (0, 0, 0), -1)
    cv2.putText(vis, text, (x0 + pad, y0 + th + pad - 1),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_offset_vis(bgr, boxes, off_recs, mm_per_px=MM_PER_PX, seg_arts=None):
    """偏移可视化（v优化）：
    Pass1 底层：把「中心线↔白线」之间的整条偏移区域用半透明橙色填充（与报警橙同色），
        再叠加 白线(淡紫圆点,加粗) + 逐列中心线(绿点) + 段中心偏移代表线(青色)；
    Pass2 顶层：框线按报警级别变色，并标注 级别文字 / 偏移值(mm) / 胶条宽度(中位数mm+规格内占比+宽异)；
        文字带黑色底板、压在最上层，永不被标记覆盖。
    注意：调用方应传入已去标注框的干净底图（如 run_metrics_48 的 clean 转 3 通道），
        否则样本图自带的红/蓝/黄标注边框会残留。
    返回 BGR 图。off_recs 需含 per_col / level 信息 / 宽度信息，与 boxes 同序。
    """
    vis = bgr.copy()
    H = vis.shape[0]
    # ---- 胶条检测外轮廓（亮蓝，加粗）：算法检出的胶条上下沿折线，保留"检测范围" ----
    if seg_arts:
        for (x, y, w, h), art in zip(boxes, seg_arts):
            if not art:
                continue
            atop = np.asarray(art.get("top"), float)
            abot = np.asarray(art.get("bot"), float)
            axs = np.arange(len(atop), dtype=float)  # 列索引，与 top/bot 同长（避免 xs 子集错位）
            fin = np.isfinite(atop) & np.isfinite(abot)
            if not fin.any():
                continue
            axs = axs[fin] + x
            atop = atop[fin] + y
            abot = abot[fin] + y
            pts_t = np.stack([axs, atop], axis=1).astype(np.int32)
            pts_b = np.stack([axs, abot], axis=1).astype(np.int32)
            cv2.polylines(vis, [pts_t], False, SEAL_CONTOUR_COLOR, SEAL_CONTOUR_THICK)
            cv2.polylines(vis, [pts_b], False, SEAL_CONTOUR_COLOR, SEAL_CONTOUR_THICK)
    # ---------- Pass 1：偏移区域橙色填充 + 标记点 + 代表线（最底层）----------
    for (x, y, w, h), r in zip(boxes, off_recs):
        pc = r.get("per_col")
        if not pc:
            continue
        xs = np.array(pc["xs"], float)
        cl_y = np.array(pc["cl_y"], float)
        w_y = np.array(pc["white_y"], float)
        if len(xs) >= 2:
            # 整条偏移带：中心线 与 白线 之间的区域，全部橙色半透明填充
            top = np.minimum(cl_y, w_y)
            bot = np.maximum(cl_y, w_y)
            poly = np.concatenate([
                np.stack([xs, top], axis=1),
                np.stack([xs[::-1], bot[::-1]], axis=1),
            ]).astype(np.int32)
            overlay = vis.copy()
            cv2.fillPoly(overlay, [poly], OFFSET_FILL_COLOR)
            cv2.addWeighted(overlay, OFFSET_FILL_ALPHA, vis, 1 - OFFSET_FILL_ALPHA, 0, vis)
        # 绿点 = 逐列中心线；淡紫圆点 = 白线（加粗约2倍，透出橙色带之上）
        for xi, yi in zip(xs, cl_y):
            vis[int(round(yi)), int(round(xi))] = (0, 255, 0)
        for xi, yi in zip(xs, w_y):
            cv2.circle(vis, (int(round(xi)), int(round(yi))), WHITE_LINE_RADIUS,
                       WHITE_LINE_COLOR, -1)
        # 段中心列偏移代表线（青色，避开与轻微偏移黄框撞色）
        xc = x + w // 2
        if r.get("offset_mm") is not None and len(xs):
            k = int(np.argmin(np.abs(xs - xc)))
            cv2.line(vis, (int(round(xs[k])), int(round(cl_y[k]))),
                     (int(round(xs[k])), int(round(w_y[k]))), REP_LINE_COLOR, REP_LINE_THICK)
    # ---------- Pass 2：框线 + 级别 + 偏移值 + 宽度文字（最顶层，确保不被覆盖）----------
    for (x, y, w, h), r in zip(boxes, off_recs):
        # ---- 框线 + 报警级别（颜色随级别变化）----
        lv = r.get("offset_level")
        if lv in _LEVEL_COLOR:
            col = _LEVEL_COLOR[lv]
            lab = _LEVEL_LABEL.get(lv, "")
        else:
            col = _UNKNOWN_COLOR
            lab = _UNKNOWN_LABEL
        cv2.rectangle(vis, (x, y), (x + w, y + h), col, 4)
        if lab:
            _put_text_plate(vis, lab, x, y - 8, scale=FONT_LEVEL, color=col, thickness=2)
        # ---- 偏移值（mm）文字：紧贴代表线，青色，带底板 ----
        pc = r.get("per_col")
        if pc and r.get("offset_mm") is not None:
            xs2 = np.array(pc["xs"], float)
            cl_y2 = np.array(pc["cl_y"], float)
            w_y2 = np.array(pc["white_y"], float)
            xc = x + w // 2
            k = int(np.argmin(np.abs(xs2 - xc)))
            tx = int(round(xs2[k])) + 6
            tyv = int(round((cl_y2[k] + w_y2[k]) / 2))
            _put_text_plate(vis, f"{r['offset_mm']:.1f}mm", tx, tyv,
                            scale=FONT_OFFSET, color=REP_LINE_COLOR, thickness=2)
        # ---- 宽度信息文字（框下方，带底板，永不被覆盖）----
        wm = r.get("width_median_mm")
        if wm is not None:
            txt = f"W={wm:.1f}mm"
            wip = r.get("width_in_spec_pct")
            if wip is not None:
                txt += f" {wip:.0f}%"
            if r.get("width_problem"):
                txt += " 宽异"
            ty = y + h + 18
            if ty + 6 > H:          # 框贴底：文字改到框上方，避免被裁切/被盖
                ty = y - 30
            _put_text_plate(vis, txt, x, ty, scale=FONT_WIDTH, color=(255, 255, 255), thickness=2)
    return vis


def _side(n, i):
    # 单相机单侧场景：仅段序号代号，不用 L/R（避免误读为左右侧）
    return f"S{i}"
