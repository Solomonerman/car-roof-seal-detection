# -*- coding: utf-8 -*-
"""视觉检测（胶条外形检测算法，按用户定义实现）。

处理流水线：
  Step1   轻度加亮：gamma 提亮暗部，拉大胶条与暗背景灰度差
  Step2   ROI 裁剪：锁定车顶胶条固定区域，剔除无关车身背景
  Step3   小波滤波：db4 多尺度分解去噪，保留胶条边缘
  Step4   OTSU + 阈值扣减 + 面积过滤：过滤暗背景误判，保留大面积真实胶条
  Step4.5 形态学开闭运算：闭运算填充胶条内部反光孔洞，开运算平滑边缘

缺陷判定（用户定义）：
  - 宽度 width   : 每段胶条横向厚度，mm；超出 20±5mm 报警
  - 缺失 missing : ROI 内整段无胶条
  - 断胶 break   : 胶条内部纵向空洞，或段间间隙显著大于本图常态（离群）
  - 过喷 overspray: 主胶条附近的离散暗斑（非连续的额外喷胶）

过程数据：
  每处理一张图，把"真实检测到的胶条像素(mask)"叠加回原图生成验证图，
  连同纯 mask 一起保存到 data/process_data/，供人工核对检测是否准确
  （绿框是外接矩形，远大于真实胶条，无法直接判断，必须看 mask）。

所有参数集中在文件顶部，便于现场调参。接口保持
detect(car_model, images, process_dir=None) -> DetectionResult 不变。
"""
import os
import json
import cv2
import numpy as np
from common.interfaces import DetectionResult, Defect

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ===================== 算法参数（来自用户定义） =====================
# Step1 加亮：gamma<1 提亮暗部，值越小越亮（用户要求再加亮）
BRIGHT_MODE = 'gamma'        # 'gamma' 或 'linear'
GAMMA_VALUE = 0.82           # 提亮暗部，增强胶条与背景对比
LINEAR_ALPHA = 1.0
LINEAR_BETA = 8              # linear 模式下的额外提亮偏移

# Step1.5 反光 / 非均匀光照抑制
# 模型：图像 = 光照(低频) × 反射率(高频)。反光是局部高光照，胶条是低反射率(暗)。
# 'retinex'：用大核高斯估计光照并除掉，反光被抵消、胶条因低反射率仍保持暗 → 分离二者
# 'none'   ：关闭
REFLECTION_CORRECTION = 'retinex'
RETINEX_SIGMA = 31           # 光照估计的高斯模糊半径(px)，越大越平滑、去越大块反光

# Step2 ROI：(x左上, y左上, 宽度w, 高度h)
# 实测相机分辨率为 1920x1200，ROI 宽度设 1920 覆盖整幅；
# _detect_one 内部用 min() 把 ROI clamp 到图像边界，更窄图像(如 1440)也自适应。
ROI = (0, 405, 1920, 259)

# Step3 小波滤波
WAVELET_TYPE = 'db4'

# Step4 OTSU 参数
OTSU_INVERTED = True         # 灰度<=阈值→白色（胶条）
OTSU_THRESHOLD_DELTA = 135   # 实际阈值 = OTSU自动阈值 - delta；delta 越大判定越严（去误检）
OTSU_MAX_THRESH = 220        # 阈值上限保护
OTSU_MIN_THRESH = 40         # 阈值下限保护
MIN_CONTOUR_AREA = 300        # 面积过滤：滤除 17x15 级小噪声碎片，仅保留大面积胶条

# Step4.5 形态学开闭运算
MORPHOLOGY_ENABLE = True
CLOSE_KERNEL_SIZE = (21, 3)  # 闭运算核：横向长适配水平胶条，高度小避免变形
OPEN_KERNEL_SIZE = (3, 3)    # 开运算核：去除边缘毛刺
MORPHOLOGY_ITERATIONS = 1
# ================================================================

# ===================== 标定 / 尺寸换算 =====================
CALIBRATION_PATH = os.path.join(ROOT, "config", "calibration.json")
DEFAULT_MM_PER_PIXEL = 0.30466   # 兜底：标定文件缺失时使用（左侧相机标定值）

SEAL_WIDTH_NOMINAL_MM = 20.0
SEAL_WIDTH_TOL_MM = 5.0
SEAL_WIDTH_MIN_MM = SEAL_WIDTH_NOMINAL_MM - SEAL_WIDTH_TOL_MM   # 15
SEAL_WIDTH_MAX_MM = SEAL_WIDTH_NOMINAL_MM + SEAL_WIDTH_TOL_MM   # 25

# ===================== 缺陷判定参数 =====================
MAIN_LINE_MIN_WIDTH_PX = 40     # 横向跨度 > 此值（px）视为主胶条段，否则可能是过喷/噪声
MAIN_LINE_MIN_ASPECT = 2.0      # 主胶条段长宽比 >= 此值（胶条是细长水平对象）
HOLE_MIN_AREA_PX = 150          # 胶条内部纵向空洞面积阈值（px）→ 断胶
BREAK_MIN_GAP_PX = 120          # 段间间隙离群判定的下限（px，≈36mm）；小于此值不报断胶
GAP_OUTLIER_FACTOR = 2.5        # 间隙 > 本图常态间隙的中位数 × 此系数 → 断胶（离群）
EDGE_IGNORE_PX = 30             # 距 ROI 左右边界此范围内不报断胶（防裁切误报）

OVERSPRAY_BAND_FRAC = 2.0       # 在主胶条上下 N×胶条宽(px) 范围内才算过喷候选
OVERSPRAY_MIN_AREA = 80         # 过喷最小面积（px）
OVERSPRAY_MAX_AREA = 12000      # 过喷最大面积（px），超过则视为另一段胶条而非过喷

# ===================== 过程数据 =====================
PROCESS_DIR = os.path.join(ROOT, "data", "process_data")

# 缺陷标签集合（这些标签出现 → 整体判定 NG）
NG_LABELS = {"missing", "break", "overspray", "width"}

# 颜色（BGR）
_COLOR_GREEN = (0, 200, 0)
_COLOR_RED = (0, 0, 220)
_COLOR_CYAN = (255, 255, 0)
_COLOR_YELLOW = (0, 255, 255)


def _load_mm_per_pixel() -> float:
    """读取标定文件获取 毫米/像素；缺失则用兜底值并告警。"""
    try:
        if os.path.isfile(CALIBRATION_PATH):
            with open(CALIBRATION_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            v = float(data.get("mm_per_pixel", DEFAULT_MM_PER_PIXEL))
            return v
    except Exception as e:
        print(f"[标定] 读取 {CALIBRATION_PATH} 失败：{e}，使用兜底值")
    return DEFAULT_MM_PER_PIXEL


MM_PER_PIXEL = _load_mm_per_pixel()


# ===================== 图像预处理 =====================
def _brighten(gray: np.ndarray) -> np.ndarray:
    """Step1：轻度加亮暗部，不碰暗背景。"""
    if BRIGHT_MODE == 'gamma':
        inv = 1.0 / GAMMA_VALUE
        table = np.array([((i / 255.0) ** inv) * 255
                           for i in range(256)]).astype(np.uint8)
        return cv2.LUT(gray, table)
    return cv2.convertScaleAbs(gray, alpha=LINEAR_ALPHA, beta=LINEAR_BETA)


def _illumination_normalize(gray: np.ndarray, sigma: int = 31) -> np.ndarray:
    """Step1.5 Retinex 光照归一化：估计并除掉低频光照(含反光)，保留反射率。

    图像 = 光照(低频) × 反射率(高频)。用大核高斯估计光照分量，
    在 log 域相减后还原：反光(高光照)被抵消，胶条(低反射率)仍保持暗。
    """
    roi = gray.astype(np.float32) + 1.0
    illum = cv2.GaussianBlur(roi, (0, 0), sigma)
    illum = np.clip(illum, 1.0, None)
    r = np.log(roi) - np.log(illum)              # 反射率（log 域）
    r = (r - np.min(r)) / (np.ptp(r) + 1e-6)     # 归一化到 0~1
    return (r * 255).astype(np.uint8)


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


def _build_mask(gray: np.ndarray):
    """Step1~4.5：输入灰度 ROI，返回 (mask_closed, mask_open, steps)。

    mask_open  : 仅做开运算，保留真实断裂（用于连续性/断胶判定）
    mask_closed: 闭+开，胶条更连贯平滑（用于显示与测宽）
    steps      : dict{名称: 灰度图/二值图}，用于逐步过程可视化
                0_roi / 1_bright / 2_wavelet / 3_otsu / 4_open / 5_close
    """
    out = {}
    # Step1 加亮（增强胶条与背景对比）
    roi = _brighten(gray)
    out['1_bright'] = roi.copy()

    # Step1.5 反光/非均匀光照抑制：Retinex 光照归一化
    if REFLECTION_CORRECTION == 'retinex':
        roi = _illumination_normalize(roi, RETINEX_SIGMA)
        out['1b_reflect'] = roi.copy()
    else:
        out['1b_reflect'] = roi.copy()

    # Step3 小波去噪
    roi = _wavelet_denoise(roi)
    out['2_wavelet'] = roi.copy()

    # Step4 OTSU + 阈值扣减
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
    out['_actual_thresh'] = actual  # 仅记录，不存图

    if OTSU_INVERTED:
        mask = (roi <= actual).astype(np.uint8) * 255   # 灰度<=阈值 → 胶条(白)
    else:
        mask = (roi >= actual).astype(np.uint8) * 255
    out['3_otsu'] = mask.copy()

    if not MORPHOLOGY_ENABLE:
        out['5_close'] = mask.copy()
        return mask, mask, out

    open_k = cv2.getStructuringElement(cv2.MORPH_CROSS, OPEN_KERNEL_SIZE)
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_k,
                                 iterations=MORPHOLOGY_ITERATIONS)
    out['4_open'] = mask_open.copy()
    close_k = cv2.getStructuringElement(cv2.MORPH_CROSS, CLOSE_KERNEL_SIZE)
    mask_closed = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, close_k,
                                   iterations=MORPHOLOGY_ITERATIONS)
    out['5_close'] = mask_closed.copy()
    return mask_closed, mask_open, out


# ===================== 几何测量 =====================
def _segment_width_px(seg_mask: np.ndarray) -> float:
    """计算单段胶条的横向厚度(px)：取每列竖直跨度的中位数。

    胶条近似水平，横向厚度≈竖直方向跨度；用中位数避免波浪导致的高估。
    """
    ys, xs = np.where(seg_mask > 0)
    if xs.size == 0:
        return 0.0
    w = seg_mask.shape[1]
    col_min = np.full(w, seg_mask.shape[0], dtype=np.int32)
    col_max = np.full(w, -1, dtype=np.int32)
    np.minimum.at(col_min, xs, ys)
    np.maximum.at(col_max, xs, ys)
    valid = col_max >= 0
    span = (col_max[valid] - col_min[valid] + 1).astype(np.float64)
    return float(np.median(span)) if span.size else 0.0


def _detect_holes(seg_mask: np.ndarray) -> list:
    """检测胶条内部纵向空洞（背景被胶条包围）。返回各空洞面积(px)列表。

    先对 seg_mask padding 一圈背景，保证外部背景在 flood fill 时连通，
    避免胶条横跨整幅/贴边把上下背景隔断而造成的误判。
    """
    h, w = seg_mask.shape
    if h < 3 or w < 3:
        return []
    pad = np.pad(seg_mask, 1, mode="constant", constant_values=0)
    inv = cv2.bitwise_not(pad)
    ff = inv.copy()
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 0)
    # ff 中剩余白色 = 未与边界连通的背景 = 内部空洞
    num, lbl, stats, _ = cv2.connectedComponentsWithStats(ff, 8)
    holes = []
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= HOLE_MIN_AREA_PX:
            holes.append(area)
    return holes


# ===================== 单图检测 =====================
def _detect_one(path: str, process_dir: str = None):
    """检测单张图，返回 Defect 列表（含 meta: width_mm 等）。

    过程数据：若 process_dir 给定，保存 mask 验证图与纯 mask 到该目录。
    """
    img = cv2.imread(path)
    if img is None:
        return [Defect(0, 0, 0, 0, "missing", 0.9,
                       meta={"reason": "图像读取失败"})]

    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step2 ROI 裁剪（clamp 到图像边界，防止尺寸不符越界）
    x0, y0, rw, rh = ROI
    x1 = min(x0 + rw, gray_full.shape[1])
    y1 = min(y0 + rh, gray_full.shape[0])
    if x1 <= x0 or y1 <= y0:
        roi_gray = gray_full
        x0, y0, x1, y1 = 0, 0, gray_full.shape[1], gray_full.shape[0]
    else:
        roi_gray = gray_full[y0:y1, x0:x1]

    # 加亮 + 建 mask（含每一步中间图，供逐步调参确认）
    mask_closed, mask_open, steps = _build_mask(roi_gray)
    if process_dir:
        print(f"[过程] {os.path.basename(path)} OTSU 实际阈值 = "
              f"{steps.get('_actual_thresh', 0)}")

    defects: list = []

    # 轮廓提取（用 closed 图做主体展示与测宽）
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area < MIN_CONTOUR_AREA:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        cx = bx + bw / 2
        cy = by + bh / 2
        candidates.append({
            "idx": i, "contour": c, "area": area,
            "bx": bx, "by": by, "bw": bw, "bh": bh,
            "cx": cx, "cy": cy,
        })

    if not candidates:
        # 整段无胶条 → 缺失
        defects.append(Defect(x0, y0, x1 - x0, y1 - y0, "missing", 0.9,
                              meta={"reason": "ROI内未检出胶条"}))
        if process_dir:
            _save_process(img, (x0, y0, x1, y1), mask_closed, defects,
                          path, process_dir, steps, roi_gray)
        return defects

    # 主胶条段 vs 过喷候选
    main_segs = [s for s in candidates
                 if s["bw"] >= MAIN_LINE_MIN_WIDTH_PX
                 and s["bw"] >= s["bh"] * MAIN_LINE_MIN_ASPECT]
    main_idx_set = {s["idx"] for s in main_segs}
    main_cy_list = [s["cy"] for s in main_segs]
    main_cy = float(np.median(main_cy_list)) if main_cy_list else \
        float(np.median([s["cy"] for s in candidates]))

    # 测宽 + 宽度缺陷 + 空洞断胶
    for s in main_segs:
        seg_mask_full = np.zeros(mask_closed.shape, np.uint8)
        cv2.drawContours(seg_mask_full, [s["contour"]], -1, 255, -1)
        # 裁剪到外接框，缩小处理范围并避免跨整幅误判
        sub = seg_mask_full[s["by"]:s["by"] + s["bh"],
                            s["bx"]:s["bx"] + s["bw"]]
        width_px = _segment_width_px(sub)
        width_mm = width_px * MM_PER_PIXEL
        meta = {"width_mm": round(width_mm, 2),
                "width_px": round(width_px, 1)}
        # 全长坐标
        gx, gy, gw, gh = x0 + s["bx"], y0 + s["by"], s["bw"], s["bh"]
        if width_mm < SEAL_WIDTH_MIN_MM or width_mm > SEAL_WIDTH_MAX_MM:
            defects.append(Defect(gx, gy, gw, gh, "width", 0.85, meta=meta))
        else:
            defects.append(Defect(gx, gy, gw, gh, "seal", 0.85, meta=meta))

        # 纵向空洞 → 断胶
        holes = _detect_holes(sub)
        if holes:
            biggest = max(holes)
            defects.append(Defect(gx, gy, gw, gh, "break", 0.8,
                                  meta={"type": "纵向空洞",
                                        "hole_area_px": biggest,
                                        "width_mm": round(width_mm, 2)}))

    # 段间间隙断胶（离群判定）+ 边缘忽略
    if len(main_segs) >= 2:
        segs_sorted = sorted(main_segs, key=lambda s: s["bx"])
        gaps = []
        for a, b in zip(segs_sorted, segs_sorted[1:]):
            gap = (b["bx"]) - (a["bx"] + a["bw"])
            # 同一竖直带才比较（否则是不同行的两条胶）
            if abs(a["cy"] - b["cy"]) > max(a["bh"], b["bh"]) * 1.5:
                continue
            gaps.append(gap)
        if len(gaps) <= 2:
            typical_gap = float(min(gaps)) if gaps else 0.0  # 样本少时以最小间隙为常态
        else:
            typical_gap = float(np.median(gaps))
        threshold = max(BREAK_MIN_GAP_PX, typical_gap * GAP_OUTLIER_FACTOR)
        for a, b in zip(segs_sorted, segs_sorted[1:]):
            if abs(a["cy"] - b["cy"]) > max(a["bh"], b["bh"]) * 1.5:
                continue
            gap = (b["bx"]) - (a["bx"] + a["bw"])
            if gap <= threshold:
                continue
            # 边缘忽略：任一段贴近 ROI 左右边界 → 可能是图像裁切，不报
            if (a["bx"] + a["bw"] >= (x1 - x0) - EDGE_IGNORE_PX) or \
               (b["bx"] <= EDGE_IGNORE_PX):
                continue
            gap_mm = gap * MM_PER_PIXEL
            gx = x0 + a["bx"] + a["bw"]
            gy = y0 + int(min(a["cy"], b["cy"]))
            gw = gap
            gh = int(abs(a["cy"] - b["cy"])) + max(a["bh"], b["bh"]) // 2
            defects.append(Defect(gx, gy, gw, gh, "break", 0.8,
                                  meta={"type": "段间间隙",
                                        "gap_mm": round(gap_mm, 1),
                                        "typical_gap_mm": round(typical_gap * MM_PER_PIXEL, 1)}))

    # 过喷：非主段、靠近主胶条竖直带、面积适中
    band_px = max(OVERSPRAY_BAND_FRAC * (SEAL_WIDTH_NOMINAL_MM / MM_PER_PIXEL), 30)
    for s in candidates:
        if s["idx"] in main_idx_set:
            continue
        if abs(s["cy"] - main_cy) > band_px:
            continue
        if not (OVERSPRAY_MIN_AREA <= s["area"] <= OVERSPRAY_MAX_AREA):
            continue
        gx, gy, gw, gh = x0 + s["bx"], y0 + s["by"], s["bw"], s["bh"]
        defects.append(Defect(gx, gy, gw, gh, "overspray", 0.7,
                              meta={"area_px": int(s["area"])}))

    if process_dir:
        _save_process(img, (x0, y0, x1, y1), mask_closed, defects,
                      path, process_dir, steps, roi_gray)
    return defects


def _save_process(img, roi_box, mask, defects, src_path, process_dir,
                  steps=None, roi_gray=None):
    """保存过程数据：每一步中间图 + 纯 mask + mask 叠加验证图。"""
    os.makedirs(process_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(src_path))[0]
    x0, y0, x1, y1 = roi_box
    h, w = img.shape[:2]

    # 逐步过程图（灰度/二值）
    if steps:
        # 0_roi：ROI 裁剪后的原始灰度（与 1_bright 对比看加亮效果）
        if roi_gray is not None:
            cv2.imwrite(os.path.join(process_dir, f"{base}_0_roi.png"), roi_gray)
        for name in ("1_bright", "1b_reflect", "2_wavelet",
                     "3_otsu", "4_open", "5_close"):
            if name in steps:
                cv2.imwrite(os.path.join(process_dir, f"{base}_{name}.png"),
                            steps[name])

    # 纯 mask（白=胶条，黑=背景）
    mask_full = np.zeros((h, w), np.uint8)
    mask_full[y0:y1, x0:x1] = mask
    mask_path = os.path.join(process_dir, base + "_mask.png")
    cv2.imwrite(mask_path, mask_full)

    # 叠加验证图：把真实检测到的胶条像素染成青色，并画框
    vis = img.copy()
    roi_vis = vis[y0:y1, x0:x1]
    mask3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    blend = cv2.addWeighted(roi_vis, 0.55, mask3, 0.45, 0)
    vis[y0:y1, x0:x1] = blend

    for d in defects:
        color = _COLOR_RED if d.label in NG_LABELS else _COLOR_GREEN
        cv2.rectangle(vis, (d.x, d.y), (d.x + d.w, d.y + d.h), color, 2)
        text = d.label
        if d.meta and "width_mm" in d.meta:
            text += f" {d.meta['width_mm']:.1f}mm"
        elif d.meta and "gap_mm" in d.meta:
            text += f" {d.meta['gap_mm']:.0f}mm"
        cv2.putText(vis, text, (max(0, d.x), max(12, d.y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # ROI 框（青色细线）标明检测区域
    cv2.rectangle(vis, (x0, y0), (x1, y1), _COLOR_CYAN, 1)

    overlay_path = os.path.join(process_dir, base + "_overlay.png")
    cv2.imwrite(overlay_path, vis)


class SealDetector:
    def detect(self, car_model: str, images: list, process_dir: str = None) \
            -> DetectionResult:
        all_defects = []
        for path in images:
            all_defects.extend(_detect_one(path, process_dir))

        ng = [d for d in all_defects if d.label in NG_LABELS]
        ok = len(ng) == 0
        conf = round(float(np.mean([d.confidence for d in all_defects]))
                     if all_defects else 0.9, 3)

        # 汇总每条信息
        seals = [d for d in all_defects if d.label == "seal"]
        widths = [d.meta.get("width_mm") for d in seals if d.meta and "width_mm" in d.meta]
        wmsg = ""
        if widths:
            wmsg = " 宽度(mm)=" + "/".join(f"{v:.1f}" for v in widths)
        msg = (f"检测 {len(images)} 张图，胶条段={len(seals)} "
               f"缺陷(NG)={len(ng)}{wmsg}")

        return DetectionResult(
            car_model=car_model,
            ok=ok,
            defects=all_defects,
            confidence=conf,
            message=msg,
        )
