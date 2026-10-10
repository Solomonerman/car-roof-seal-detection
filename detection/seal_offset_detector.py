# -*- coding: utf-8 -*-
"""胶条在线检测（偏移 + 宽度）：与离线验证 run_metrics_48 同一套算法。

复用模块（均已在仓库内、随代码交付）：
  - roi_locator    : 红框 / YOLO 定位 ROI（make_locator 自动 YOLO→红框回退）
  - seal_segment   : 胶条分割
  - seal_metrics   : 宽度 / 中心线 / 长度度量
  - offset_check_v7: 白线拟合 + 偏移分级 + 橙色带可视化（与 0921 验证一致）

接口对齐 detection.detector.SealDetector：
  detect(car_model, images, process_dir=None) -> DetectionResult
过程图（mask overlay + 橙色带偏移图）存到 process_dir，供 web_capture / ui 读取。

本文件不改动 detector.py 的现有实现；detector.py 的 SealDetector 保留作回退/对照。
"""
from __future__ import annotations
import os
import sys

import cv2
import numpy as np
from common.interfaces import DetectionResult, Defect

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)          # detection/ 的上级 = 仓库根
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import roi_locator as L
import seal_segment as S
import seal_metrics as M
import offset_check_v7 as OC

# ===================== 量算参数（与 run_metrics_48 一致） =====================
MM = 0.30466
SPEC = (15.0, 30.0)                        # 宽度规格 20±5mm
WIDTH_SPEC_LO = 15.0
WIDTH_SPEC_HI = 30.0
WIDTH_MIN_RUN_COLS = 15                    # 宽度连续超差列数阈值（抗单列噪点）
# 偏移报警：仅『漏水风险』(level=3) 才判 offset 缺陷并计 NG；中度/轻微仅展示、不判NG
OFFSET_NG_LEVEL = 3

NG_LABELS = {"missing", "break", "overspray", "width", "offset"}


def pick_worst_offset(off_recs, boxes):
    """从一张照片的多个胶条段里挑『偏移最严重』(offset_level 最大) 的一段。

    一张照片可能有多段胶条；其外框颜色 / 表格偏移列应以最严重段为准，而非首段。
    无胶条(missing)/丢失(status!=ok) 的段不计入；全部未知时返回 None。
    """
    worst_idx = None
    worst_lv = OC.OFFSET_LEVEL_UNKNOWN          # -1 起始，任何有效段都能覆盖
    for i in range(len(boxes)):
        orec = off_recs[i] if i < len(off_recs) else None
        if not orec or orec.get("status") != "ok":
            continue
        lv = orec.get("offset_level", OC.OFFSET_LEVEL_UNKNOWN)
        if lv is None or lv <= worst_lv:
            continue
        worst_lv = lv
        worst_idx = i
    return off_recs[worst_idx] if worst_idx is not None else None


def find_width_runs(wp, lo=WIDTH_SPEC_LO, hi=WIDTH_SPEC_HI, min_run=WIDTH_MIN_RUN_COLS):
    """逐列宽度超差连续段检测：≥min_run 列超差才判（抗单列噪点，与偏移偏离同思路）。

    返回 (runs, flagged)。runs 每项 {start_col,end_col,run_cols,lo_mm,hi_mm,direction}。
    direction: 'low'(持续偏窄<下限) / 'high'(持续偏宽>上限) / 'both'。
    """
    oos = np.isfinite(wp) & ((wp < lo) | (wp > hi))
    runs = []
    n = len(oos)
    i = 0
    while i < n:
        if oos[i]:
            j = i
            while j + 1 < n and oos[j + 1]:
                j += 1
            run_len = j - i + 1
            if run_len >= min_run:
                seg = wp[i:j + 1]
                sv = seg[np.isfinite(seg)]
                if sv.size == 0:
                    i = j + 1
                    continue
                lo_seg = float(sv.min())
                hi_seg = float(sv.max())
                if hi_seg > hi and lo_seg >= lo:
                    direction = "high"
                elif lo_seg < lo and hi_seg <= hi:
                    direction = "low"
                else:
                    direction = "both"
                runs.append(dict(start_col=int(i), end_col=int(j), run_cols=run_len,
                                 lo_mm=lo_seg, hi_mm=hi_seg, direction=direction))
            i = j + 1
        else:
            i += 1
    return runs, len(runs) > 0


def _inpaint_frames(bgr):
    """去标注框（红/蓝/黄），返回灰度；现场无标注直接返回灰度。"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    R = rgb[:, :, 0].astype(int)
    G = rgb[:, :, 1].astype(int)
    B = rgb[:, :, 2].astype(int)
    red = ((R - np.maximum(G, B)) > 40) & (R > 70)
    blue = ((B - np.maximum(R, G)) > 40) & (B > 70)
    yellow = ((R > 120) & (G > 120) & (B < 90))
    fmask = (red | blue | yellow).astype(np.uint8) * 255
    if int(fmask.sum()) == 0:
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clean = cv2.inpaint(bgr, fmask, 3, cv2.INPAINT_TELEA)
    return cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)


class SealOffsetDetector:
    """胶条检测（偏移 + 宽度）：现场在线版，与 0921 离线验证同一套算法。"""

    def __init__(self):
        self._loc = L.make_locator()      # 自动 YOLO/红框回退

    def detect(self, car_model: str, images: list, process_dir: str = None,
               roi: tuple = None) -> DetectionResult:
        all_defects = []
        per_photo = []
        for path in images:
            try:
                defects, summ = self._detect_one(path, process_dir)
            except Exception as e:
                print(f"[SealOffset] 处理异常 {os.path.basename(path)}: {e}")
                defects = [Defect(0, 0, 0, 0, "missing", 0.9,
                                 meta={"reason": f"处理异常:{e}"})]
                summ = {"name": os.path.splitext(os.path.basename(path))[0],
                        "width_mm": None, "width_in_spec": False, "offset_mm": None,
                        "offset_level": OC.OFFSET_LEVEL_UNKNOWN,
                        "offset_level_name": OC.OFFSET_LEVEL_UNKNOWN_NAME,
                        "deviated": False, "status": "error", "ok": False}
            all_defects.extend(defects)
            per_photo.append(summ)

        ng = [d for d in all_defects if d.label in NG_LABELS]
        ok = len(ng) == 0
        conf = round(float(np.mean([d.confidence for d in all_defects]))
                     if all_defects else 0.9, 3)
        # 有效检出段数：成功分割出胶条(mask 非空)的 ROI 段，
        # 不论宽度/偏移是否合格（与 missing 区分开，避免误读成"没检出胶条"）
        n_seg = sum(1 for d in all_defects if d.label != "missing")
        widths = [d.meta.get("width_mm") for d in all_defects
                  if d.label == "width" and d.meta]
        offs = [d.meta.get("offset_mm") for d in all_defects
                if d.meta and d.meta.get("offset_mm") is not None]
        wmsg = " 宽度(mm)=" + "/".join(f"{v:.1f}" for v in widths) if widths else ""
        omsg = " 偏移(mm)=" + "/".join(f"{v:.1f}" for v in offs) if offs else ""
        msg = (f"检测 {len(images)} 张图，胶条段={n_seg} "
               f"缺陷(NG)={len(ng)}{wmsg}{omsg}")

        # 逐图汇总 JSON（供看板页底部数据表格读取；仅写 proc_dir，不改库、不影响已有过程图）
        if process_dir:
            try:
                import json as _json
                _sp = os.path.join(process_dir, "summary.json")
                with open(_sp, "w", encoding="utf-8") as _f:
                    _json.dump({"per_photo": per_photo, "ok": ok, "message": msg},
                               _f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[SealOffset] 逐图汇总写入失败 {process_dir}: {e}")

        return DetectionResult(
            car_model=car_model,
            ok=ok,
            defects=all_defects,
            confidence=conf,
            message=msg,
        )

    def _detect_one(self, path: str, process_dir: str = None):
        defects = []
        bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            d = Defect(0, 0, 0, 0, "missing", 0.9, meta={"reason": "图像读取失败"})
            summary = {"name": os.path.splitext(os.path.basename(path))[0],
                       "width_mm": None, "width_in_spec": False, "offset_mm": None,
                       "offset_level": OC.OFFSET_LEVEL_UNKNOWN,
                       "offset_level_name": OC.OFFSET_LEVEL_UNKNOWN_NAME,
                       "deviated": False, "status": "no_image", "ok": False}
            return [d], summary

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        boxes = self._loc.locate(bgr)
        if not boxes:
            # 无定位框：整图兜底为单一 ROI（现场无标注框 / 定位失败时不致整图漏检）
            h, w = gray.shape[:2]
            boxes = [(0, 0, w, h)]

        clean = _inpaint_frames(bgr)
        try:
            wl = OC.fit_white_line(gray, boxes, segment=S.segment,
                                   clean_mask=M.clean_mask)
        except Exception as e:
            print(f"[SealOffset] fit_white_line 失败: {e}")
            wl = None

        n = len(boxes)
        off_recs = []
        box_art = {}
        for idx, box in enumerate(boxes):
            x, y, w, h = box
            side = f"S{idx}"
            roi = clean[y:y + h, x:x + w]
            mask, notes = S.segment(roi, (x, y), gray.shape)
            if mask is None:
                defects.append(Defect(x, y, w, h, "missing", 0.9,
                                     meta={"reason": "no_strip"}))
                orec = OC.segment_offset(box, wl, None, n, idx)
                off_recs.append(orec)
                continue

            cmask = M.clean_mask(mask)
            top, bot = M.col_bounds(cmask)
            wp = M.per_column_width_mm(top, bot, MM)
            ws = M.width_stats(wp, SPEC)
            length, ncols = M.length_mm(cmask, MM)
            xs, mid = M.skeleton_centerline(top, bot)

            # 偏移（白线 → 中心线）
            sxs_g = (xs + x).astype(float)
            smid_g = (mid + y).astype(float)
            orec = OC.segment_offset(box, wl, (sxs_g, smid_g), n, idx)
            off_recs.append(orec)

            box_art[idx] = dict(cmask=cmask, top=top, bot=bot, xs=xs, mid=mid,
                                x=x, y=y, w=w, h=h, gt=None)

            # 宽度缺陷：仅当『代表宽度(中值)超出规格 15~30mm』才判 NG（按需求：区间内值如 26.8 不报NG）
            wmedian = ws.get("median_mm")
            width_out = (wmedian is not None) and (
                wmedian < WIDTH_SPEC_LO or wmedian > WIDTH_SPEC_HI)
            width_problem = bool(width_out)
            if width_problem:
                defects.append(Defect(x, y, w, h, "width", 0.85,
                                     meta={"width_mm": wmedian,
                                           "in_spec_pct": ws.get("in_spec_pct")}))
            else:
                defects.append(Defect(x, y, w, h, "seal", 0.85,
                                     meta={"width_mm": wmedian}))

            # 把宽度信息挂到 box_art，供结果图画宽度标注（与本地 detector.py 对齐）
            box_art[idx]["width_mm"] = wmedian
            box_art[idx]["width_in_spec"] = (not width_problem)

            # 偏移缺陷（核心漏水风险指标）：仅『漏水风险』(level=3) 计 NG；中度/轻微仅展示不判NG
            lv = orec.get("offset_level", OC.OFFSET_LEVEL_UNKNOWN)
            dev = bool(orec.get("deviated"))
            if orec.get("status") == "ok" and lv >= OFFSET_NG_LEVEL:
                defects.append(Defect(x, y, w, h, "offset", 0.9,
                                     meta={"offset_mm": orec.get("offset_mm"),
                                           "offset_level": lv,
                                           "offset_level_name": orec.get("offset_level_name"),
                                           "deviated": dev}))

        # —— 过程图（存到 web_capture 传入的 process_dir）——
        if process_dir:
            try:
                os.makedirs(process_dir, exist_ok=True)
                base = os.path.splitext(os.path.basename(path))[0]
                # mask overlay：亮青半透明=算法 mask(可透出原图)，红框=检测区；宽度文字标注（与本地 detector.py 对齐）
                vis = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)
                SEAL_CYAN_BGR = (255, 255, 0)    # 亮青（BGR：B=255 G=255 R=0）
                SEAL_ALPHA = 0.25                  # 半透明：0=全透 1=不透明（越小越透；25% 透明度）
                for art in box_art.values():
                    cx, cy, cw, ch = art["x"], art["y"], art["w"], art["h"]
                    # 亮青半透明：把青色按 alpha 混到原图(灰度转BGR)上，保留底层亮度，看着更舒服
                    _reg = vis[cy:cy + ch, cx:cx + cw]
                    _mpx = art["cmask"] > 0
                    _reg[_mpx] = (_reg[_mpx].astype(np.float32) * (1 - SEAL_ALPHA)
                                  + np.array(SEAL_CYAN_BGR, np.float32) * SEAL_ALPHA).astype(np.uint8)
                    # 外框：红色，加粗一号
                    cv2.rectangle(vis, (cx, cy), (cx + cw, cy + ch), (0, 0, 255), 5)
                    # 宽度信息：中值宽度(mm)，统一白色（与 _offset.png 一致，不区分合格/超差）
                    wmm = art.get("width_mm")
                    if wmm is not None:
                        OC._put_text_plate(vis, f"W={wmm:.1f}mm", cx, max(20, cy - 8),
                                          scale=OC.FONT_WIDTH, color=(255, 255, 255), thickness=2)
                cv2.imwrite(os.path.join(process_dir, base + "_mask.png"), vis)
                # 橙色带偏移图（与 0921 验证一致）
                if off_recs:
                    seg_arts = [box_art.get(i) for i in range(len(boxes))]
                    off_vis = OC.draw_offset_vis(
                        cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR),
                        boxes, off_recs, seg_arts=seg_arts)
                    cv2.imwrite(os.path.join(process_dir, base + "_offset.png"), off_vis)
            except Exception as e:
                print(f"[SealOffset] 过程图生成失败 {os.path.basename(path)}: {e}")

        # ---- 逐图汇总（供看板页底部数据表格；不入库）----
        summary = {
            "name": os.path.splitext(os.path.basename(path))[0],
            "width_mm": None,
            "width_in_spec": True,
            "offset_mm": None,
            "offset_level": OC.OFFSET_LEVEL_UNKNOWN,
            "offset_level_name": OC.OFFSET_LEVEL_UNKNOWN_NAME,
            "deviated": False,
            "status": "ok",
            "ok": not any(d.label in NG_LABELS for d in defects),
        }
        present = [i for i in range(len(boxes)) if i in box_art]
        if present:
            summary["width_mm"] = box_art[present[0]].get("width_mm")
            summary["width_in_spec"] = all(box_art[i].get("width_in_spec", True)
                                           for i in present)
        # 偏移代表：取『偏移最严重』(offset_level 最大) 的一段作为该照片外框/表格依据
        # （一张照片可能有多段胶条，外框颜色按最严重段计，而非首段）
        rep_off = pick_worst_offset(off_recs, boxes)
        if rep_off is not None:
            summary["offset_mm"] = rep_off.get("offset_mm")
            summary["offset_level"] = rep_off.get("offset_level", OC.OFFSET_LEVEL_UNKNOWN)
            summary["offset_level_name"] = rep_off.get("offset_level_name",
                                                      OC.OFFSET_LEVEL_UNKNOWN_NAME)
            summary["deviated"] = rep_off.get("deviated", False)
            summary["status"] = rep_off.get("status", "ok")
        return defects, summary
