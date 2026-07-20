# -*- coding: utf-8 -*-
"""胶条检测算法离线与仿真验证工具。

用途：
  1) 在没有现场真图时，生成"仿真车顶图"把整条检测管线跑通，确认
     ROI 坐标适配、pywt 可用、检测逻辑正确（能检出胶条、不误报缺失）。
  2) 现场联调时，把本脚本指向 data/raw_images（或任意目录）即可对真图批量验证。

生成两类仿真图（尺寸默认 1440x1080，匹配 ROI=(0,405,1920,259)）：
  - sim_real.png   : 浅灰车顶 + ROI 带内一条暗色胶条（应检出，不报 missing）
  - sim_missing.png: 仅浅灰车顶，无胶条（应报 missing）

运行：
  python tools/validate_detector.py                # 生成仿真图并验证
  python tools/validate_detector.py --src 目录      # 验证目录下所有真图
  python tools/validate_detector.py --w 1920 --h 1080  # 自定义仿真图尺寸

输出：
  tools/results/      下保存带标注的可视化结果与 HTML 报告（含过程 mask 图）
  data/process_data/  下保存真实检测到的胶条像素 mask（过程数据，供排查）
"""
import os
import sys
import glob
import shutil
import subprocess
import argparse
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from detection.detector import SealDetector, ROI, PROCESS_DIR, NG_LABELS
from common.interfaces import Defect

_COLOR_GREEN = (0, 200, 0)
_COLOR_RED = (0, 0, 220)


# ----------------------- 仿真图生成 -----------------------
def make_roof(width, height, with_seal=True, seed=0):
    """生成一张仿真车顶图。

    with_seal=True 时，在 ROI 带内画一条暗色胶条；否则只画车身底噪。
    """
    rng = np.random.default_rng(seed)
    yy = np.linspace(0, 1, height)[:, None]
    base = 205 - 25 * yy
    img = np.repeat(base, width, axis=1).astype(np.float32)
    img += rng.normal(0, 8, (height, width))
    img = np.clip(img, 0, 255).astype(np.uint8)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    if with_seal:
        x, y, w, h = ROI
        sy = y + int(h * 0.35)
        sh = int(h * 0.30)
        sx, sw = 0, width
        seal = np.full((sh, sw, 3), 30.0, dtype=np.float32)
        hl = rng.normal(0, 6, (sh, sw)).astype(np.float32)[..., None]
        seal = np.clip(seal + hl, 0, 255).astype(np.uint8)
        vis[sy:sy + sh, sx:sx + sw] = seal
        for i in range(sw):
            dy = int(3 * np.sin(i / 90.0))
            if 0 <= sy + dy and sy + dy + sh <= height:
                col = vis[sy + dy:sy + dy + sh, i]
                vis[sy:sy + sh, i] = 0
                vis[sy + dy:sy + dy + sh, i] = col
        for gap in (int(sw * 0.45), int(sw * 0.7)):
            vis[sy - 2:sy + sh + 2, gap:gap + 12] = img[sy - 2:sy + sh + 2, gap:gap + 12][..., None].repeat(3, 2)
    return vis


def annotate(src_path, result, out_path):
    """把检测结果（外接框）画回原图：正常段绿框、缺陷红框、宽度标注。"""
    img = cv2.imread(src_path)
    if img is None:
        return
    for d in result.defects:
        color = _COLOR_RED if d.label in NG_LABELS else _COLOR_GREEN
        cv2.rectangle(img, (d.x, d.y), (d.x + d.w, d.y + d.h), color, 2)
        text = d.label
        if d.meta and "width_mm" in d.meta:
            text += f" {d.meta['width_mm']:.1f}mm"
        elif d.meta and "gap_mm" in d.meta:
            text += f" {d.meta['gap_mm']:.0f}mm"
        cv2.putText(img, text, (max(0, d.x), max(12, d.y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.imwrite(out_path, img)


def _open_file(path):
    """用系统默认程序打开文件（主要用于自动展示 HTML 报告）。"""
    try:
        if sys.platform == "win32":
            os.startfile(path)
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
    except Exception as e:
        print(f"[提示] 无法自动打开 {path}：{e}")


def _write_html(out_dir, rows, title="胶条检测结果"):
    parts = [f"<html><head><meta charset='utf-8'>"
             f"<title>{title}</title></head><body>"
             f"<h1>{title}</h1>"
             f"<p>绿框=正常胶条段（外接矩形，大于真实胶条）；红框=缺陷。"
             f"要看<strong>真实检测到的胶条像素</strong>，请核对下方的"
             f"<em>mask 叠加验证图</em>（青色区域即算法判定为胶条的像素）。</p>"]
    for name, ok, n_seal, n_ng, conf, annot, mask_ov in rows:
        color = "#1a7f1a" if ok else "#c01414"
        verdict = "OK" if ok else "NG"
        parts.append(
            f"<h3 style='color:{color}'>{name} → {verdict} "
            f"(胶条段={n_seal} 缺陷={n_ng} 置信={conf:.3f})</h3>"
            f"<img src='{annot}' style='max-width:95%;border:1px solid #ccc;"
            f"margin-bottom:8px;'>"
            f"<div style='color:#0a6;font-weight:bold;'>▼ 过程数据：真实检测 mask 叠加图"
            f"（青色=胶条像素，请确认是否贴合真实胶条）</div>"
            f"<img src='{mask_ov}' style='max-width:95%;border:1px solid #0a6;"
            f"margin-bottom:16px;'><hr>")
    parts.append("</body></html>")
    html_path = os.path.join(out_dir, "report.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    return html_path


def _copy_process_img(process_dir, base, suffix, out_dir):
    """把 data/process_data 下的过程图复制到 out_dir 供 HTML 本地引用。"""
    src = os.path.join(process_dir, base + suffix)
    if os.path.isfile(src):
        dst = os.path.join(out_dir, base + suffix)
        shutil.copy(src, dst)
        return os.path.basename(dst)
    return None


# ----------------------- 验证主流程 -----------------------
def validate_paths(paths, out_dir, process_dir):
    os.makedirs(process_dir, exist_ok=True)
    det = SealDetector()
    rows = []
    for p in paths:
        res = det.detect("TEST", [p], process_dir=process_dir)
        out_path = os.path.join(out_dir, "annot_" + os.path.basename(p))
        annotate(p, res, out_path)
        seals = [d for d in res.defects if d.label == "seal"]
        ng = [d for d in res.defects if d.label in NG_LABELS]
        widths = [d.meta["width_mm"] for d in seals
                  if d.meta and "width_mm" in d.meta]
        wstr = (" 宽度(mm)=" + "/".join(f"{v:.1f}" for v in widths)) if widths else ""
        line = (f"{os.path.basename(p):24s} -> {'OK' if res.ok else 'NG'} | "
                f"胶条段={len(seals)} 缺陷={len(ng)} 置信={res.confidence:.3f}{wstr}")
        print(line)
        # 复制过程 mask 图到 out_dir 供 HTML 引用
        base = os.path.splitext(os.path.basename(p))[0]
        mask_ov = _copy_process_img(process_dir, base, "_overlay.png", out_dir)
        mask_ov = mask_ov or ""
        rows.append((os.path.basename(p), res.ok, len(seals), len(ng),
                     res.confidence, os.path.basename(out_path), mask_ov))
        # 列出缺陷明细
        for d in ng:
            extra = ""
            if d.meta:
                extra = " " + ", ".join(f"{k}={v}" for k, v in d.meta.items())
            print(f"    [缺陷] {d.label} @({d.x},{d.y},{d.w},{d.h}){extra}")
    return rows


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
        print(f"=== 过程数据将保存到: {PROCESS_DIR} ===")
        rows = validate_paths(paths, out_dir, PROCESS_DIR)
        html_path = _write_html(out_dir, rows,
                                title=f"胶条检测结果（{len(paths)} 张真图）")
        print(f"=== 报告已生成: {html_path} ===")
        print(f"=== 过程 mask 已保存: {PROCESS_DIR}（请打开该文件夹核对检测是否准确） ===")
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
    rows = validate_paths([sim_real, sim_miss], out_dir, PROCESS_DIR)
    rep = os.path.join(out_dir, "report.txt")
    with open(rep, "w", encoding="utf-8") as f:
        f.write("\n".join(str(r) for r in rows))
    print(f"=== 报告已保存: {rep} ===")
    print(f"=== 可视化结果在: {out_dir} ===")
    print(f"=== 过程 mask 在: {PROCESS_DIR} ===")


if __name__ == "__main__":
    main()
