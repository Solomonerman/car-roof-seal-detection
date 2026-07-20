# -*- coding: utf-8 -*-
"""相机标定工具：用棋盘图计算 像素↔毫米 比例尺。

用法：
  1) 把棋盘标定图放到 data/calibration/ （任意文件名，建议左侧/右侧分开）
  2) python tools/calibrate.py
  3) 脚本找出棋盘内角点，算平均格距，保存 config/calibration.json

约定：棋盘为 N×M 个方格，OpenCV 内角点 = (N-1, M-1)。
用户给的是 20×15 方格 → 内角点 (19,14)。
车身弧度未校正（用户确认影响很小）；后续如需再做畸变校正。
"""
import os
import sys
import glob
import json
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CAL_DIR = os.path.join(ROOT, "data", "calibration")
OUT = os.path.join(ROOT, "config", "calibration.json")

# 候选内角点尺寸（宽,高）；优先用户给的 20x15 方格 -> (19,14)
CANDIDATES = [(19, 14), (14, 19)]

SQUARE_MM = 10.0  # 每个方格 1cm = 10mm


def find_corners(gray):
    for pat in CANDIDATES:
        ok, corners = cv2.findChessboardCorners(
            gray, pat,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), crit)
            return pat, corners
    return None, None


def spacing_px(corners, pat):
    c = corners.reshape(-1, 2)
    rows, cols = pat[1], pat[0]
    c = c.reshape(rows, cols, 2)
    spacings = []
    for r in range(rows):                       # 水平相邻
        for col in range(cols - 1):
            spacings.append(np.linalg.norm(c[r, col + 1] - c[r, col]))
    for col in range(cols):                     # 垂直相邻
        for r in range(rows - 1):
            spacings.append(np.linalg.norm(c[r + 1, col] - c[r, col]))
    return float(np.mean(spacings))


def main():
    if not os.path.isdir(CAL_DIR):
        print(f"[错误] 找不到 {CAL_DIR}，请先建文件夹并放入标定图")
        return
    files = [f for f in sorted(glob.glob(os.path.join(CAL_DIR, "*")))
             if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
    if not files:
        print(f"[错误] {CAL_DIR} 下没有图片")
        return

    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pat, corners = find_corners(gray)
        if corners is not None:
            sp = spacing_px(corners, pat)
            mm_per_px = SQUARE_MM / sp
            res = {
                "mm_per_pixel": round(mm_per_px, 5),
                "pixel_per_mm": round(sp / SQUARE_MM, 3),
                "square_spacing_px": round(sp, 3),
                "pattern_inner_corners": list(pat),
                "source_image": os.path.basename(f),
                "camera_side": "left",
                "note": "1cm方格标定；车身弧度未校正，必要时后续做畸变校正",
            }
            os.makedirs(os.path.dirname(OUT), exist_ok=True)
            with open(OUT, "w", encoding="utf-8") as fp:
                json.dump(res, fp, indent=2, ensure_ascii=False)
            # 角点可视化（留在 calibration 目录，便于肉眼核对）
            vis = img.copy()
            cv2.drawChessboardCorners(vis, pat, corners, True)
            vis_path = os.path.join(CAL_DIR, "calib_corners.png")
            cv2.imwrite(vis_path, vis)
            print("=== 标定成功 ===")
            print(f"来源          : {os.path.basename(f)}")
            print(f"内角点        : {pat}  (20x15方格 -> 19x14内角点)")
            print(f"平均格距      : {sp:.2f} px  = 10 mm")
            print(f"比例尺        : {mm_per_px:.4f} mm/px")
            print(f"推导 胶条宽20mm ≈ {20 / mm_per_px:.1f} px")
            print(f"推导 偏移  3cm ≈ {30 / mm_per_px:.1f} px")
            print(f"已保存        : {OUT}")
            print(f"角点可视化    : {vis_path}")
            return

    print("[失败] 所有图片都未找到棋盘角点。请确认：")
    print("  - 图片是清晰棋盘图，20x15 方格完整入镜")
    print("  - 光照均匀、无强反光遮挡棋盘")
    print("  - 若实际方格数不同，告诉我行列数，我改 CANDIDATES")


if __name__ == "__main__":
    main()
