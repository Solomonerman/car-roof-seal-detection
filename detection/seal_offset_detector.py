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
SPEC = (16.0, 32.0)                        # 宽度规格 24±8mm
WIDTH_SPEC_LO = 16.0
WIDTH_SPEC_HI = 32.0
WIDTH_MIN_RUN_COLS = 15                    # 宽度连续超差列数阈值（抗单列噪点）
SMALL_BOX_AREA_RATIO = 0.02                 # 小框面积占比阈值: 面积 < 此值*整图面积 且 框内无胶条 → 视为误检丢弃(按OK), 避免角落假框制造假NG
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
                        "deviated": False, "status": "error", "ok": False,
                        "loc_ok": False, "wl_ok": False, "reliable": False}
            all_defects.extend(defects)
            per_photo.append(summ)

        ng = [d for d in all_defects if d.label in NG_LABELS]
        ok = len(ng) == 0
        loc_ok = not any(d.label == "loc_warn" for d in all_defects)  # 全部帧定位/白线均成功
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
                    _json.dump({"per_photo": per_photo, "ok": ok,
                                "loc_ok": loc_ok, "message": msg},
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
                       "deviated": False, "status": "no_image", "ok": False,
                       "loc_ok": False, "wl_ok": False, "reliable": False}
            return [d], summary

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        ih, iw = gray.shape[:2]
        boxes = self._loc.locate(bgr)
        loc_failed = (len(boxes) == 0)     # YOLO/红框未定位到任何胶条框
        if loc_failed:
            # 无定位框：整图兜底为单一 ROI（现场无标注框 / 定位失败时不致整图漏检）
            # 注意：整图兜底结果不可靠，下面不再据此自动判 NG，仅打标+日志交由人工复核
            print(f"[SealOffset][WARN] 帧 {os.path.basename(path)} "
                  f"未定位到胶条框，退回整图兜底(结果不可靠, 不判NG)")
            boxes = [(0, 0, iw, ih)]

        clean = _inpaint_frames(bgr)
        # 双模型: 全局白线(含安全锚定, 治第一类浅带误报) + 逐段白线(治第二类右段钳位)
        try:
            wl_global = OC.fit_white_line(gray, boxes, segment=S.segment,
                                          clean_mask=M.clean_mask)
        except Exception as e:
            print(f"[SealOffset] fit_white_line 失败: {e}")
            wl_global = None
        try:
            wls_perbox = OC.fit_white_lines_perbox(gray, boxes, segment=S.segment,
                                                  clean_mask=M.clean_mask)
        except Exception as e:
            print(f"[SealOffset] fit_white_lines_perbox 失败: {e}")
            wls_perbox = [None] * len(boxes)
        wl_failed = (wl_global is None) and all(w is None for w in wls_perbox)
        if wl_failed:
            print(f"[SealOffset][WARN] 帧 {os.path.basename(path)} "
                  f"白线拟合失败, 偏移量不可靠(不判offset NG)")

        n = len(boxes)
        off_recs = []
        box_art = {}
        photo_name = os.path.splitext(os.path.basename(path))[0]   # 照片名(用于总览区分每段胶条)
        for idx, box in enumerate(boxes):
            x, y, w, h = box
            # 白线模型选择:
            #  - 第一类(安全锚定激活): 全局已借可靠段修正浅带误报, 两段都用全局(保持原行为, 防退化)
            #  - 第二类(非锚定): 框落在全局拟合范围内用全局; 框在全局范围外(右段贴图像右缘被钳位)用逐段拟合
            xr1 = x + w
            if (wl_global is not None and wl_global.get("safe_anchor")):
                wl_i = wl_global
            elif (wl_global is not None and x >= wl_global.get("x_min", -1)
                  and xr1 <= wl_global.get("x_max", -1)):
                wl_i = wl_global
            else:
                wl_i = wls_perbox[idx] if (idx < len(wls_perbox) and wls_perbox[idx] is not None) else wl_global
            side = f"S{idx}"
            roi = clean[y:y + h, x:x + w]
            mask, notes = S.segment(roi, (x, y), gray.shape)
            if mask is None:
                # 小框(面积占比过小)里没胶条 → 判定为误检, 直接放弃该框, 不记 missing(按 OK 处理)
                if w * h < SMALL_BOX_AREA_RATIO * iw * ih:
                    print(f"[SealOffset][WARN] 帧 {photo_name} 小框(面积 {w*h}px < "
                          f"{int(SMALL_BOX_AREA_RATIO * iw * ih)}px) 无胶条, 视为误检丢弃, 不判NG")
                    continue
                orec = OC.segment_offset(box, wl_i, None, n, idx)
                off_recs.append(orec)
                _lv = orec.get("offset_level", OC.OFFSET_LEVEL_UNKNOWN)
                defects.append(Defect(x, y, w, h, "missing", 0.9,
                                     meta={"reason": "no_strip",
                                           "photo": photo_name, "seg": side,
                                           "offset_mm": orec.get("offset_mm"),
                                           "offset_level": _lv,
                                           "offset_level_name": orec.get("offset_level_name", OC.OFFSET_LEVEL_UNKNOWN_NAME)}))
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
            orec = OC.segment_offset(box, wl_i, (sxs_g, smid_g), n, idx)
            off_recs.append(orec)

            # 偏移代表量（每段都记录，供总览『每段胶条偏移』展示；与结果图等级一致）
            lv = orec.get("offset_level", OC.OFFSET_LEVEL_UNKNOWN)
            dev = bool(orec.get("deviated"))
            off_mm = orec.get("offset_mm")
            off_name = orec.get("offset_level_name", OC.OFFSET_LEVEL_UNKNOWN_NAME)
            seg_meta = {"photo": photo_name, "seg": side,
                        "offset_mm": off_mm, "offset_level": lv,
                        "offset_level_name": off_name, "deviated": dev}

            box_art[idx] = dict(cmask=cmask, top=top, bot=bot, xs=xs, mid=mid,
                                x=x, y=y, w=w, h=h, gt=None)

            # 宽度缺陷：仅当『代表宽度(中值)超出规格 16~32mm』才判 NG（按需求：区间内值如 26.8 不报NG）
            # 若本帧定位失败(整图兜底)，宽度量无意义，不据此判 NG（交由人工复核）
            wmedian = ws.get("median_mm")
            width_out = (wmedian is not None) and (
                wmedian < WIDTH_SPEC_LO or wmedian > WIDTH_SPEC_HI)
            width_problem = bool(width_out) and (not loc_failed)
            if width_problem:
                defects.append(Defect(x, y, w, h, "width", 0.85,
                                     meta={"width_mm": wmedian,
                                           "in_spec_pct": ws.get("in_spec_pct"),
                                           **seg_meta}))
            else:
                defects.append(Defect(x, y, w, h, "seal", 0.85,
                                     meta={"width_mm": wmedian,
                                           "loc_unreliable": loc_failed,
                                           **seg_meta}))

            # 把宽度信息挂到 box_art，供结果图画宽度标注（与本地 detector.py 对齐）
            box_art[idx]["width_mm"] = wmedian
            # 定位失败(整图兜底)时宽度不可信，过程图不标红(避免误导)，靠 loc_warn 提示不可靠
            box_art[idx]["width_in_spec"] = (not width_problem) and (not loc_failed)

            # 偏移缺陷（核心漏水风险指标）：仅『漏水风险』(level=3) 才单独置 NG；
            # 中度/轻微仅展示（已挂在上面 seal/width 的 offset_* 字段），不判NG。
            # 该独立 offset 缺陷仅用于驱动整记录/整图 NG（总览显示时跳过，避免与上面重复）。
            # 若本帧定位失败或白线拟合失败，偏移量不可靠，不据此判 NG。
            if (not loc_failed and not wl_failed
                    and orec.get("status") == "ok" and lv >= OFFSET_NG_LEVEL):
                defects.append(Defect(x, y, w, h, "offset", 0.9,
                                     meta={"offset_mm": off_mm,
                                           "offset_level": lv,
                                           "offset_level_name": off_name,
                                           "deviated": dev,
                                           "photo": photo_name, "seg": side}))

        # 定位失败/白线拟合失败：追加『不可靠』标记(非NG)，供看板/日志提示需人工复核
        if loc_failed or wl_failed:
            defects.append(Defect(0, 0, iw, ih, "loc_warn", 0.95,
                                 meta={"reason": "定位失败" if loc_failed else "白线拟合失败",
                                       "loc_failed": loc_failed, "wl_failed": wl_failed}))

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
                    # 宽度信息：中值宽度(mm)；16~32mm 规格内白色，超出规格红色
                    wmm = art.get("width_mm")
                    if wmm is not None:
                        _w_in = 16.0 <= wmm <= 32.0
                        _w_color = (255, 255, 255) if _w_in else (0, 0, 255)
                        _w_txt = f"W={wmm:.1f}mm"
                        _w_y = max(20, cy - 8)
                        if not _w_in:
                            # 紧邻上方标注『宽度异常』(红，字体=宽度字体50%)
                            (_w_tw, _w_th), _ = cv2.getTextSize(
                                _w_txt, cv2.FONT_HERSHEY_SIMPLEX, OC.FONT_WIDTH, 2)
                            OC._put_text_plate(
                                vis, "宽度异常", cx, _w_y - _w_th - 6,
                                scale=OC.FONT_WIDTH * 0.5, color=(0, 0, 255), thickness=2)
                        OC._put_text_plate(vis, _w_txt, cx, _w_y,
                                          scale=OC.FONT_WIDTH, color=_w_color, thickness=2)
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
            "loc_ok": not loc_failed,
            "wl_ok": not wl_failed,
            "reliable": not (loc_failed or wl_failed),
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
        # 所有段宽度/偏移列表：供看板表格『宽度/偏移』列逗号展示全部胶条数据
        summary["widths_all_mm"] = [box_art[i].get("width_mm")
                                   for i in present
                                   if box_art[i].get("width_mm") is not None]
        summary["offsets_all_mm"] = [r.get("offset_mm")
                                    for r in off_recs
                                    if r.get("offset_mm") is not None]
        return defects, summary
