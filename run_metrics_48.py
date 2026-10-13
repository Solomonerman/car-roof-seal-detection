# -*- coding: utf-8 -*-
"""48 张黄线标注样本 · 多段胶条（左/右）端到端量算 + 黄框 GT 验证。

流程（每张图 → 每段红框各做一次）：
  红框定位所有 ROI -> 逐段 crop_and_segment 分割 -> 形态学去毛刺(3x3) ->
  逐列宽度/长度/骨架中心线 -> 每段各自的黄线框当真值，算逐列宽度一致性与列间 IoU。

输出：data/detection_results/2026-09-29_segoffset_v1/   # 版本号：生产路径锁定 segment_offset（中段80%中位数+统一骨架中心线）
  - metrics.json  ：每张图 -> segments 列表（每段独立记录）
  - metrics.csv   ：每段一行（左段/右段各一条记录）
  - metrics.md    ：全段汇总
  - 每图可视化 jpg（默认不输出；设 SAVE_IMAGES=True 才生成）：青色=算法 mask、红线=骨架中心线、黄点=黄框真值、红框=各段 ROI
"""
from __future__ import annotations

import os
import sys
import json
import csv
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import roi_locator as L
import seal_segment as S
import seal_metrics as M
import offset_check_v7 as OC  # 模块④ 白线偏移（v7：稳定性边界法 + 25~40mm 公差带，适配白线下移）

ROOT = os.environ.get("V37_IN", os.path.join(HERE, "data", "field_samples", "8X_已标注_20260922"))  # 输入目录可由 V37_IN 覆盖；不设则仍用 48 张基线目录
OUT = os.environ.get("V37_OUT", os.path.join(HERE, "data", "detection_results", "2026-09-29_segoffset_v1"))
# 输出自动分三个子文件夹，避免全部平铺在一个目录里
SUB_OFFSET = os.path.join(OUT, "offset_images")
SUB_METRIC = os.path.join(OUT, "metric_images")
# 是否输出每图可视化 jpg；默认 True（与历史一致）。设 SAVE_IMAGES=0 可跳过（省约 30% 耗时，仅出 metrics 数据表）。
# 并行进程数：默认 1（单进程，输出与历史逐字节一致）；设 WORKERS=N 可对每图并行（结果仍按文件名排序，输出不变）。
SAVE_IMAGES = os.environ.get("SAVE_IMAGES", "1").strip().lower() not in ("0", "false", "no", "")
WORKERS = int(os.environ.get("WORKERS", "1") or "1")
if WORKERS < 1:
    WORKERS = 1
_LOC = None  # 进程内 YOLO 定位器（worker 初始化时创建）
SUB_REPORTS = os.path.join(OUT, "reports")
MM = 0.30466
SPEC = (16.0, 32.0)  # 24±8mm
WIDTH_SPEC_LO = 16.0  # 宽度下限(mm)：段内存在 <此值 即宽度异常
WIDTH_SPEC_HI = 32.0  # 宽度上限(mm)：段内存在 >此值 即宽度异常
WIDTH_MIN_RUN_COLS = 15  # 宽度连续超差列数阈值：≥此值才判真实宽度异常(抗单列噪点，与偏移偏离同思路)


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


def inpaint_frames(bgr):
    """验证专用：去掉烧录的红/蓝/黄标注框，模拟无标注的现场图。

    生产现场由 YOLO 给干净 ROI、图里无标注线，不需要这步；仅 48 标注样本验证时用。
    """
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


def parse_yellow_in_roi(bgr, box):
    """解析某段红框 ROI 内的黄线框为逐列上下边界（ROI 坐标）。缺失返回 None。"""
    x, y, w, h = box
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    R = rgb[:, :, 0].astype(int)
    G = rgb[:, :, 1].astype(int)
    B = rgb[:, :, 2].astype(int)
    yel = ((R > 120) & (G > 120) & (B < 90))
    sub = yel[y : y + h, x : x + w]
    if int(sub.sum()) < 50:
        return None
    ys, xs = np.nonzero(sub)
    top = np.full(w, np.nan)
    bot = np.full(w, np.nan)
    for cx in range(w):
        col = ys[xs == cx]
        if col.size:
            top[cx] = col.min()
            bot[cx] = col.max()
    return top, bot


def _init_worker():
    """多进程 worker 初始化：每个子进程自建 YOLO 定位器（避免跨进程传递模型对象）。"""
    global _LOC
    _LOC = L.make_locator()


def compute_image(p, dp):
    """单图全流程（与历史 main 循环逐行一致）：定位→分割→宽度/中心线→白线偏移→黄线GT→可视化。
    返回 (name, per_img_entry, seg_rows_list)。可视化在 worker/主进程内直接落盘（按 rel 子目录）。
    """
    name = os.path.basename(p)
    rel = os.path.relpath(dp, ROOT)
    bgr = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return name, {"status": "decode_fail"}, [dict(file=name, status="decode_fail")]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    boxes = _LOC.locate(bgr)  # List[(x,y,w,h)]
    if not boxes:
        return name, {"status": "no_red_box"}, [dict(file=name, status="no_red_box")]

    # ---- 统一骨架：偏移模块复用宽度模块「去标注框灰度 + 骨架中心线」，避免两套中心线分歧 ----
    clean = inpaint_frames(bgr)  # 验证专用：去标注框；偏移/宽度都基于同一张图
    try:
        wl = OC.fit_white_line(gray, boxes, segment=S.segment, clean_mask=M.clean_mask)  # 白线点检测用原图(与V15一致), 不用inpaint图, 避免框被填补后亮度偏移误抓
    except Exception as _e:
        print("offset_check fit error:", _e)
        wl = None
    off_recs = []
    off_by_seg = {}
    n = len(boxes)
    segs = []
    seg_rows = []
    box_art = {}  # 缓存每段已算好的 mask/中心线/黄线，供可视化复用（避免重复 segment）

    for idx, box in enumerate(boxes):
        x, y, w, h = box
        side = f"S{idx}"  # 单相机单侧场景：仅段序号代号，不用 L/R（避免误读为左右侧；且双框多为同一条胶条被YOLO误分段）
        roi = clean[y : y + h, x : x + w]
        mask, notes = S.segment(roi, (x, y), gray.shape)
        if mask is None:
            segs.append(dict(seg=idx, side=side, status="no_strip", notes=";".join(notes)))
            seg_rows.append(dict(file=name, seg=idx, side=side, status="no_strip"))
            orec = OC.segment_offset(box, wl, None, n, idx)
            off_recs.append(orec)
            off_by_seg[idx] = orec
            continue
        cmask = M.clean_mask(mask)
        top, bot = M.col_bounds(cmask)
        wp = M.per_column_width_mm(top, bot, MM)
        ws = M.width_stats(wp, SPEC)
        length, ncols = M.length_mm(cmask, MM)
        xs, mid = M.skeleton_centerline(top, bot)

        rec = dict(seg=idx, side=side, status="ok", box=box, width=ws,
                   length_mm=length, n_cols=ncols, notes=notes)

        # ---- 模块④ 白线偏移（统一骨架）：用宽度模块算好的骨架算距离 ----
        sxs_g = (xs + x).astype(float)
        smid_g = (mid + y).astype(float)
        orec = OC.segment_offset(box, wl, (sxs_g, smid_g), n, idx)
        off_recs.append(orec)
        off_by_seg[idx] = orec
        # 给偏移可视化注入宽度信息（offset 图也要展示宽度）
        orec["width_median_mm"] = ws.get("median_mm")
        orec["width_in_spec_pct"] = ws.get("in_spec_pct")
        rec["offset_mm"] = orec["offset_mm"]
        rec["offset_fit_mode"] = orec["fit_mode"]
        rec["offset_status"] = orec["status"]
        rec["offset_white_pts"] = orec.get("white_pts")
        rec["offset_inlier_pts"] = orec.get("inlier_pts")
        rec["offset_cl_source"] = orec.get("cl_source")
        rec["offset_deviated"] = bool(orec["deviated"])
        rec["offset_dev_runs"] = orec.get("dev_runs")
        rec["offset_trim"] = orec.get("offset_trim")

        # ---- 宽度问题判定：A(主体分布 P5/P95 超差) + B(连续≥N列超差) 结合，抗单列噪点 ----
        # A：看主体分布第5/95百分位（自动忽略两端极端噪声列），P5<下限 或 P95>上限 即主体出规格；
        # B：逐列连续超差（≥WIDTH_MIN_RUN_COLS 列才算），抓局部持续偏窄/偏宽缺陷。
        # 二者任一触发即判宽度异常（双保险：既看整体主体、又看局部持续段）。
        body_out = bool((ws.get("p5_mm", 999) < WIDTH_SPEC_LO) or (ws.get("p95_mm", -1) > WIDTH_SPEC_HI))
        width_runs, run_out = find_width_runs(wp, WIDTH_SPEC_LO, WIDTH_SPEC_HI, WIDTH_MIN_RUN_COLS)
        rec["width_problem"] = bool(body_out or run_out)
        rec["width_problem_body"] = body_out
        rec["width_problem_runs"] = width_runs
        rec["width_p5_mm"] = ws.get("p5_mm")
        rec["width_p95_mm"] = ws.get("p95_mm")
        rec["width_min_mm"] = ws.get("min_mm")
        rec["width_max_mm"] = ws.get("max_mm")
        orec["width_problem"] = rec["width_problem"]

        # ---- 该段黄线 GT 验证（ROI 坐标直接比对）----
        gt = parse_yellow_in_roi(bgr, box)
        if gt is not None:
            gtop, gbot = gt
            gwp = (gbot - gtop) * MM
            gws = M.width_stats(gwp, SPEC)
            ov = np.isfinite(top) & np.isfinite(bot) & np.isfinite(gtop) & np.isfinite(gbot)
            if ov.any():
                aw = (bot - top)[ov]
                gw = (gbot - gtop)[ov]
                med_col_diff = round(float(np.median(np.abs(aw - gw) * MM)), 2)
                inter = np.maximum(0.0, np.minimum(bot[ov], gbot[ov]) - np.maximum(top[ov], gtop[ov]))
                union = aw + gw - inter
                col_iou = float(np.nanmean(np.where(union > 0, inter / union, 0.0)))
            else:
                med_col_diff = None
                col_iou = None
            rec["gt_width"] = gws
            rec["med_col_diff_mm"] = med_col_diff
            rec["col_iou_vs_yellow"] = round(col_iou, 4) if col_iou is not None else None
        else:
            rec["gt_width"] = None
            rec["med_col_diff_mm"] = None
            rec["col_iou_vs_yellow"] = None
        # 缓存本段已算好的产物，可视化块直接复用（不再重算 segment / 黄线解析）
        box_art[idx] = dict(cmask=cmask, top=top, bot=bot, xs=xs, mid=mid, gt=gt)

        segs.append(rec)
        seg_rows.append(
            dict(
                file=name, seg=idx, side=side, status="ok",
                median_mm=ws.get("median_mm"), mean_mm=ws.get("mean_mm"),
                p5_mm=ws.get("p5_mm"), p95_mm=ws.get("p95_mm"),
                min_mm=ws.get("min_mm"), max_mm=ws.get("max_mm"),
                in_spec_pct=ws.get("in_spec_pct"), length_mm=length,
                gt_median_mm=(rec["gt_width"] or {}).get("median_mm"),
                col_iou=rec.get("col_iou_vs_yellow"),
                med_col_diff_mm=rec.get("med_col_diff_mm"),
                offset_mm=rec.get("offset_mm"),
                offset_fit_mode=rec.get("offset_fit_mode"),
                offset_status=rec.get("offset_status"),
                offset_cl_source=rec.get("offset_cl_source"),
                offset_deviated=rec.get("offset_deviated"),
                offset_dev_n=len(rec.get("offset_dev_runs") or []),
                offset_dev_hi_mm=(max((dr["hi_mm"] for dr in (rec.get("offset_dev_runs") or [])), default=None)
                                  if rec.get("offset_dev_runs") else None),
                offset_dev_lo_mm=(min((dr["lo_mm"] for dr in (rec.get("offset_dev_runs") or [])), default=None)
                                  if rec.get("offset_dev_runs") else None),
                offset_trim=(f'{rec.get("offset_trim")}' if rec.get("offset_trim") else None),
                offset_level=orec.get("offset_level"),
                offset_level_name=orec.get("offset_level_name"),
                width_problem=rec.get("width_problem"),
                width_problem_body=rec.get("width_problem_body"),
                width_problem_run_n=len(rec.get("width_problem_runs") or []),
                width_p5_mm=rec.get("width_p5_mm"),
                width_p95_mm=rec.get("width_p95_mm"),
                width_min_mm=rec.get("width_min_mm"),
                width_max_mm=rec.get("width_max_mm"),
                box=f"{box}",
            )
        )

    per_entry = {"n_seg": n, "segments": segs}

    # ---- 可视化（复用测量阶段已缓存的 mask/中心线/黄线，避免重复 segment）----
    if SAVE_IMAGES:
        vis = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)
        for idx, box in enumerate(boxes):
            art = box_art.get(idx)
            if art is None:
                continue
            x, y, w, h = box
            cmask = art["cmask"]
            top, bot = art["top"], art["bot"]
            xs, mid = art["xs"], art["mid"]
            vis[y : y + h, x : x + w][cmask > 0] = (255, 255, 0)  # 青色 = 算法 mask
            cv = np.isfinite(mid)
            vis[np.round(mid[cv]).astype(int) + y, xs[cv].astype(int) + x] = (0, 0, 255)  # 红 = 中心线
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红框 ROI
            gt = art["gt"]
            if gt is not None:
                gtop, gbot = gt
                gx = np.where(np.isfinite(gtop))[0]
                vis[np.round(gtop[np.isfinite(gtop)]).astype(int) + y, gx + x] = (0, 255, 255)
                vis[np.round(gbot[np.isfinite(gbot)]).astype(int) + y, np.where(np.isfinite(gbot))[0] + x] = (0, 255, 255)
        _m_dir = os.path.join(SUB_METRIC, rel)
        os.makedirs(_m_dir, exist_ok=True)
        cv2.imencode(".jpg", vis)[1].tofile(os.path.join(_m_dir, os.path.splitext(name)[0] + "_metric.jpg"))

    # ---- 模块④ 偏移可视化：白线(红点)+逐列中心线(绿点)+超差段橙色高亮（SAVE_IMAGES 才生成）----
    if SAVE_IMAGES and off_recs:
        try:
            seg_arts = [box_art.get(i) for i in range(len(boxes))]
            off_vis = OC.draw_offset_vis(cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR), boxes, off_recs, seg_arts=seg_arts)
            _o_dir = os.path.join(SUB_OFFSET, rel)
            os.makedirs(_o_dir, exist_ok=True)
            cv2.imencode(".jpg", off_vis)[1].tofile(
                os.path.join(_o_dir, os.path.splitext(name)[0] + "_offset.jpg"))
        except Exception as _e:
            print("offset_vis error:", _e)

    return name, per_entry, seg_rows


def process_image(item):
    """ProcessPoolExecutor 入口（单参数，便于 map）。"""
    p, dp = item
    return compute_image(p, dp)


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(SUB_OFFSET, exist_ok=True)
    os.makedirs(SUB_METRIC, exist_ok=True)
    os.makedirs(SUB_REPORTS, exist_ok=True)
    # 定位器延迟创建：顺序路径在下方 _init_worker() 创建；并行路径由各 worker 的 initializer 创建（避免跨进程传模型）
    rows = []
    per_img = {}

    files = []
    for dp, _, fn in os.walk(ROOT):
        for f in fn:
            if f.lower().endswith((".png", ".bmp")):  # 兼容 .bmp（0922 现场样本为 bmp 格式）
                # 记录 (绝对路径, 父目录) 便于输出时按原文件夹分组
                files.append((os.path.join(dp, f), dp))
    files.sort(key=lambda t: t[0])
    print("found", len(files), "images")

    print("workers=", WORKERS, "save_images=", SAVE_IMAGES)
    if WORKERS > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_worker) as _ex:
            for name, per_entry, seg_rows in _ex.map(process_image, files):
                per_img[name] = per_entry
                rows.extend(seg_rows)
    else:
        _init_worker()  # 主进程定位器（顺序路径；与历史单进程行为一致）
        for p, dp in files:
            name, per_entry, seg_rows = compute_image(p, dp)
            per_img[name] = per_entry
            rows.extend(seg_rows)

    # ---- 汇总输出（reports/ 子文件夹）----
    with open(os.path.join(SUB_REPORTS, "metrics.json"), "w", encoding="utf-8") as fo:
        json.dump(per_img, fo, ensure_ascii=False, indent=2, default=str)

    # ---- 单文件数据（上位机人工查看用，按图归类、预留 car_id 供按车归类）----
    data = {
        "meta": {
            "mm_per_px": MM,
            "offset_spec_mm": [OC.OFFSET_SPEC_LO, OC.OFFSET_SPEC_HI],
            "width_spec_mm": [WIDTH_SPEC_LO, WIDTH_SPEC_HI],
            "end_trim": OC.SEAL_END_TRIM,
            "min_run_cols": OC.OFFSET_MIN_RUN_COLS,
            "n_images": len(per_img),
            "n_segments": len(rows),
        },
        "cars": {},  # 上位机按车归类用：{car_id: {files:[...]}}；当前样本无车号，留空由下一步填充
        "images": [
            {"file": fn, "car_id": None, "n_seg": v.get("n_seg"), "segments": v.get("segments")}
            for fn, v in per_img.items()
        ],
    }
    with open(os.path.join(OUT, "detection_data.json"), "w", encoding="utf-8") as fo:
        json.dump(data, fo, ensure_ascii=False, indent=2, default=str)

    flds = [
        "file", "seg", "side", "status", "median_mm", "mean_mm", "p5_mm", "p95_mm",
        "min_mm", "max_mm", "in_spec_pct", "width_problem", "width_problem_body", "width_problem_run_n",
        "width_p5_mm", "width_p95_mm", "width_min_mm", "width_max_mm",
        "length_mm", "gt_median_mm",
        "col_iou", "med_col_diff_mm", "offset_mm", "offset_fit_mode", "offset_status",
        "offset_level", "offset_level_name",
        "offset_cl_source", "offset_deviated", "offset_dev_n", "offset_dev_hi_mm", "offset_dev_lo_mm",
        "offset_trim", "box",
    ]
    with open(os.path.join(SUB_REPORTS, "metrics.csv"), "w", newline="", encoding="utf-8") as fo:
        w = csv.DictWriter(fo, fieldnames=flds)
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r.get("status") == "ok"]
    lines = []
    lines.append(f"# 48 张样本 · 多段胶条量算结果（黄框 GT 验证）\n")
    lines.append(f"- 总记录 {len(rows)} 段，成功 {len(ok)} 段（含左/右段）；原图 {len(per_img)} 张")
    n2 = sum(1 for v in per_img.values() if isinstance(v, dict) and v.get("n_seg") == 2)
    n1 = sum(1 for v in per_img.values() if isinstance(v, dict) and v.get("n_seg") == 1)
    lines.append(f"- 含 2 段图 {n2} 张；含 1 段图 {n1} 张")
    if ok:
        meds = [r["median_mm"] for r in ok if r.get("median_mm") is not None]
        ins = [r["in_spec_pct"] for r in ok if r.get("in_spec_pct") is not None]
        ious = [r["col_iou"] for r in ok if r.get("col_iou") is not None]
        cdiff = [r["med_col_diff_mm"] for r in ok if r.get("med_col_diff_mm") is not None]
        lines.append(f"- 算法宽度中位数：均值 {np.mean(meds):.2f} / 最小 {min(meds):.2f} / 最大 {max(meds):.2f} mm")
        lines.append(f"- 公差带内占比均值：{np.mean(ins):.1f}%")
        if ious:
            lines.append(f"- 与黄框逐列 IoU 均值：{np.mean(ious):.4f}")
        if cdiff:
            lines.append(f"- 与黄框逐列宽度中位偏差：{np.mean(cdiff):.2f} mm")
        wpn = sum(1 for r in rows if r.get("width_problem"))
        wpn_body = sum(1 for r in rows if r.get("width_problem_body"))
        wpn_run = sum(1 for r in rows if (r.get("width_problem_run_n") or 0) > 0)
        devn = sum(1 for r in rows if r.get("offset_deviated"))
        lines.append(f"- 宽度异常段(A主体P5/P95 或 B连续超差)：{wpn} 段（其中 A主体出规格 {wpn_body} 段 / B连续超差 {wpn_run} 段）")
        lines.append(f"- 偏移偏离段(有效超差段总长>整段{OC.OFFSET_DEV_FRAC:.0%} 且 落[{OC.OFFSET_SPEC_LO:.0f},{OC.OFFSET_SPEC_HI:.0f}]mm)：{devn} 段")
        lv_names = [r.get("offset_level_name") for r in rows if r.get("offset_level_name")]
        lv_cnt = {}
        for nm in lv_names:
            lv_cnt[nm] = lv_cnt.get(nm, 0) + 1
        lines.append(f"- 偏移报警分级：正常 {lv_cnt.get('正常',0)} / 轻微偏移 {lv_cnt.get('轻微偏移',0)} "
                      f"/ 中度偏移 {lv_cnt.get('中度偏移',0)} / 漏水风险 {lv_cnt.get('漏水风险',0)} 段")
    with open(os.path.join(SUB_REPORTS, "metrics.md"), "w", encoding="utf-8") as fo:
        fo.write("\n".join(lines) + "\n")

    print("OK done. segments=", len(rows), "ok=", len(ok))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
