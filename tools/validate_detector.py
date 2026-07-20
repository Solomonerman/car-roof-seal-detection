# -*- coding: utf-8 -*-
"""胶条检测算法离线与仿真验证工具。

用途：
  1) 在没有现场真图时，生成"仿真车顶图"把整条检测管线跑通，确认
     ROI 坐标适配、pywt 可用、检测逻辑正确（能检出胶条、不误报缺失）。
  2) 现场联调时，把本脚本指向 data/raw_images（或任意目录）即可对真图批量验证。

生成两类仿真图（尺寸默认 1440x1080，匹配 ROI=(0,405,1439,259)）：
  - sim_real.png   : 浅灰车顶 + ROI 带内一条暗色胶条（应检出，不报 missing）
  - sim_missing.png: 仅浅灰车顶，无胶条（应报 missing）

运行：
  python tools/validate_detector.py                # 生成仿真图并验证
  python tools/validate_detector.py --src 目录      # 验证目录下所有真图
  python tools/validate_detector.py --w 1920 --h 1080  # 自定义仿真图尺寸

输出：
  tools/results/ 下保存带标注的可视化结果与文本报告。
"""
import os
import sys
import glob
import argparse
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from detection.detector import SealDetector, ROI
from common.interfaces import Defect


# ----------------------- 仿真图生成 -----------------------
def make_roof(width, height, with_seal=True, seed=0):
    """生成一张仿真车顶图。

    with_seal=True 时，在 ROI 带内画一条暗色胶条；否则只画车身底噪。
    """
    rng = np.random.default_rng(seed)
    # 浅灰车顶金属面：轻微竖向渐变 + 高斯噪声
    yy = np.linspace(0, 1, height)[:, None]
    base = 205 - 25 * yy                       # 顶部亮、底部略暗
    img = np.repeat(base, width, axis=1).astype(np.float32)
    img += rng.normal(0, 8, (height, width))    # 表面噪声
    img = np.clip(img, 0, 255).astype(np.uint8)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if with_seal:
        x, y, w, h = ROI
        # 胶条落在 ROI 带中部，留一点上下边距；胶条横向铺满整幅图宽(width)，非 ROI 宽
        sy = y + int(h * 0.35)
        sh = int(h * 0.30)
        sx, sw = 0, width
        # 暗色胶条主体：模拟接近黑色的橡胶密封条（灰度~30）
        seal = np.full((sh, sw, 3), 30.0, dtype=np.float32)
        # 胶条上的反光高光（局部提亮，但不改变"暗"的本质）
        hl = rng.normal(0, 6, (sh, sw)).astype(np.float32)[..., None]
        seal = np.clip(seal + hl, 0, 255).astype(np.uint8)
        vis[sy:sy + sh, sx:sx + sw] = seal
        # 加一点轻微波浪 + 两个小缺口，更贴近真实
        for i in range(sw):
            dy = int(3 * np.sin(i / 90.0))
            if 0 <= sy + dy and sy + dy + sh <= height:
                col = vis[sy + dy:sy + dy + sh, i]
                vis[sy:sy + sh, i] = 0  # 先清
                vis[sy + dy:sy + dy + sh, i] = col
        # 两个小缺口（模拟不连续，但整体仍连续）
        for gap in (int(sw * 0.45), int(sw * 0.7)):
            vis[sy - 2:sy + sh + 2, gap:gap + 12] = img[sy - 2:sy + sh + 2, gap:gap + 12][..., None].repeat(3, 2)

    return vis


def annotate(src_path, result, out_path):
    """把检测结果画回原图：胶条绿框、缺失红框。"""
    img = cv2.imread(src_path)
    if img is None:
        return
    for d in result.defects:
        color = (0, 200, 0) if d.label == "seal" else (0, 0, 220)
        cv2.rectangle(img, (d.x, d.y), (d.x + d.w, d.y + d.h), color, 2)
        cv2.putText(img, f"{d.label} {d.confidence:.2f}",
                    (max(0, d.x), max(0, d.y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.imwrite(out_path, img)


# ----------------------- 验证主流程 -----------------------
def validate_paths(paths, out_dir):
    det = SealDetector()
    report_lines = []
    for p in paths:
        res = det.detect("TEST", [p])
        out_path = os.path.join(out_dir, "annot_" + os.path.basename(p))
        annotate(p, res, out_path)
        seals = [d for d in res.defects if d.label == "seal"]
        miss = [d for d in res.defects if d.label == "missing"]
        line = (f"{os.path.basename(p):20s} -> {'OK' if res.ok else 'NG'} | "
                f"胶条={len(seals)} 缺失={len(miss)} 置信={res.confidence:.3f}")
        print(line)
        report_lines.append(line)
    return report_lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=None, help="验证该目录下所有图片（默认用仿真图）")
    ap.add_argument("--w", type=int, default=1440, help="仿真图宽")
    ap.add_argument("--h", type=int, default=1080, help="仿真图高")
    args = ap.parse_args()

    out_dir = os.path.join(ROOT, "tools", "results")
    os.makedirs(out_dir, exist_ok=True)

    if args.src:
        paths = sorted(glob.glob(os.path.join(args.src, "*")))
        paths = [p for p in paths if p.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))]
        if not paths:
            print(f"[警告] {args.src} 下没有图片")
            return
        print(f"=== 验证真图：{len(paths)} 张 ===")
        validate_paths(paths, out_dir)
        return

    # 默认：生成仿真图并验证
    print(f"=== 生成仿真图 {args.w}x{args.h}（ROI={ROI}） ===")
    sim_real = os.path.join(out_dir, "sim_real.png")
    sim_miss = os.path.join(out_dir, "sim_missing.png")
    cv2.imwrite(sim_real, make_roof(args.w, args.h, with_seal=True, seed=1))
    cv2.imwrite(sim_miss, make_roof(args.w, args.h, with_seal=False, seed=2))
    print(f"  已生成: {sim_real}")
    print(f"  已生成: {sim_miss}")
    print("=== 运行检测 ===")
    lines = validate_paths([sim_real, sim_miss], out_dir)
    rep = os.path.join(out_dir, "report.txt")
    with open(rep, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"=== 报告已保存: {rep} ===")
    print(f"=== 可视化结果在: {out_dir} ===")


if __name__ == "__main__":
    main()
