# -*- coding: utf-8 -*-
"""48 张黄线标注样本 · 多段胶条（左/右）端到端量算 + 黄框 GT 验证。

流程（每张图 → 每段红框各做一次）：
  红框定位所有 ROI -> 逐段 crop_and_segment 分割 -> 形态学去毛刺(3x3) ->
  逐列宽度/长度/骨架中心线 -> 每段各自的黄线框当真值，算逐列宽度一致性与列间 IoU。

输出：data/detection_results/v37_seal_metrics_seg/
  - metrics.json  ：每张图 -> segments 列表（每段独立记录）
  - metrics.csv   ：每段一行（左段/右段各一条记录）
  - metrics.md    ：全段汇总
  - 每图可视化 jpg：青色=算法 mask、红线=骨架中心线、黄点=黄框真值、红框=各段 ROI
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
import offset_check as OC  # 模块④ 白线偏移（v15）

ROOT = os.path.join(HERE, "data", "field_samples", "8X_已标注_20260922")
OUT = os.environ.get("V37_OUT", os.path.join(HERE, "data", "detection_results", "v37_seal_metrics_seg"))
MM = 0.30466
SPEC = (15.0, 30.0)  # 20±5mm


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


def main():
    os.makedirs(OUT, exist_ok=True)
    loc = L.RedBoxLocator()
    rows = []
    per_img = {}

    files = []
    for dp, _, fn in os.walk(ROOT):
        for f in fn:
            if f.lower().endswith(".png"):
                files.append(os.path.join(dp, f))
    files.sort()
    print("found", len(files), "png")

    for p in files:
        name = os.path.basename(p)
        bgr = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            per_img[name] = {"status": "decode_fail"}
            rows.append(dict(file=name, status="decode_fail"))
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        boxes = loc.locate(bgr)  # List[(x,y,w,h)]
        if not boxes:
            per_img[name] = {"status": "no_red_box"}
            rows.append(dict(file=name, status="no_red_box"))
            continue

        # ---- 模块④ 白线偏移（v15）：原始灰度 + 红框，拟合全局白线算各段偏移 + 逐列偏离判定 ----
        off_recs = []
        off_by_seg = {}
        try:
            off_recs = OC.compute_offsets(gray, boxes, segment=S.segment, clean_mask=M.clean_mask)
            off_by_seg = {r["seg"]: r for r in off_recs}
        except Exception as _e:
            print("offset_check error:", _e)
            off_by_seg = {}

        clean = inpaint_frames(bgr)  # 验证专用：去标注框
        n = len(boxes)
        segs = []
        seg_rows = []

        for idx, box in enumerate(boxes):
            x, y, w, h = box
            side = "L" if n == 2 and idx == 0 else ("R" if n == 2 and idx == 1 else f"S{idx}")
            roi = clean[y : y + h, x : x + w]
            mask, notes = S.segment(roi, (x, y), gray.shape)
            if mask is None:
                segs.append(dict(seg=idx, side=side, status="no_strip", notes=";".join(notes)))
                seg_rows.append(dict(file=name, seg=idx, side=side, status="no_strip"))
                continue
            cmask = M.clean_mask(mask)
            top, bot = M.col_bounds(cmask)
            wp = M.per_column_width_mm(top, bot, MM)
            ws = M.width_stats(wp, SPEC)
            length, ncols = M.length_mm(cmask, MM)
            xs, mid = M.skeleton_centerline(top, bot)

            rec = dict(seg=idx, side=side, status="ok", box=box, width=ws,
                       length_mm=length, n_cols=ncols, notes=notes)

            # ---- 模块④ 白线偏移（v15）结果 ----
            orec = off_by_seg.get(idx)
            rec["offset_mm"] = orec["offset_mm"] if orec else None
            rec["offset_fit_mode"] = orec["fit_mode"] if orec else None
            rec["offset_status"] = orec["status"] if orec else "error"
            rec["offset_white_pts"] = orec.get("white_pts") if orec else None
            rec["offset_inlier_pts"] = orec.get("inlier_pts") if orec else None
            rec["offset_deviated"] = bool(orec["deviated"]) if orec else False
            rec["offset_dev_runs"] = orec.get("dev_runs") if orec else []

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
                    offset_deviated=rec.get("offset_deviated"),
                    offset_dev_n=len(rec.get("offset_dev_runs") or []),
                    offset_dev_hi_mm=(max((dr["hi_mm"] for dr in (rec.get("offset_dev_runs") or [])), default=None)
                                      if rec.get("offset_dev_runs") else None),
                    offset_dev_lo_mm=(min((dr["lo_mm"] for dr in (rec.get("offset_dev_runs") or [])), default=None)
                                      if rec.get("offset_dev_runs") else None),
                    box=f"{box}",
                )
            )

        per_img[name] = {"n_seg": n, "segments": segs}
        rows.extend(seg_rows)

        # ---- 可视化（整图叠加所有段）----
        vis = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)
        for idx, box in enumerate(boxes):
            x, y, w, h = box
            roi = clean[y : y + h, x : x + w]
            mask, _ = S.segment(roi, (x, y), gray.shape)
            if mask is None:
                continue
            cmask = M.clean_mask(mask)
            top, bot = M.col_bounds(cmask)
            vis[y : y + h, x : x + w][cmask > 0] = (255, 255, 0)  # 青色 = 算法 mask
            xs, mid = M.skeleton_centerline(top, bot)
            cv = np.isfinite(mid)
            vis[np.round(mid[cv]).astype(int) + y, xs[cv].astype(int) + x] = (0, 0, 255)  # 红 = 中心线
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 红框 ROI
            gt = parse_yellow_in_roi(bgr, box)
            if gt is not None:
                gtop, gbot = gt
                gx = np.where(np.isfinite(gtop))[0]
                vis[np.round(gtop[np.isfinite(gtop)]).astype(int) + y, gx + x] = (0, 255, 255)
                vis[np.round(gbot[np.isfinite(gbot)]).astype(int) + y, np.where(np.isfinite(gbot))[0] + x] = (0, 255, 255)
        cv2.imencode(".jpg", vis)[1].tofile(os.path.join(OUT, os.path.splitext(name)[0] + "_metric.jpg"))

        # ---- 模块④ 偏移可视化：白线(红点)+逐列中心线(绿点)+超差段橙色高亮 ----
        if off_recs:
            try:
                off_vis = OC.draw_offset_vis(bgr, boxes, off_recs)
                cv2.imencode(".jpg", off_vis)[1].tofile(
                    os.path.join(OUT, os.path.splitext(name)[0] + "_offset.jpg"))
            except Exception as _e:
                print("offset_vis error:", _e)

    # ---- 汇总输出 ----
    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as fo:
        json.dump(per_img, fo, ensure_ascii=False, indent=2, default=str)

    flds = [
        "file", "seg", "side", "status", "median_mm", "mean_mm", "p5_mm", "p95_mm",
        "min_mm", "max_mm", "in_spec_pct", "length_mm", "gt_median_mm",
        "col_iou", "med_col_diff_mm", "offset_mm", "offset_fit_mode", "offset_status",
        "offset_deviated", "offset_dev_n", "offset_dev_hi_mm", "offset_dev_lo_mm", "box",
    ]
    with open(os.path.join(OUT, "metrics.csv"), "w", newline="", encoding="utf-8") as fo:
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
    with open(os.path.join(OUT, "metrics.md"), "w", encoding="utf-8") as fo:
        fo.write("\n".join(lines) + "\n")

    print("OK done. segments=", len(rows), "ok=", len(ok))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
