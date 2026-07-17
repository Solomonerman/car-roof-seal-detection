# -*- coding: utf-8 -*-
"""生成尺寸匹配真实 ROI 的 mock 演示图（1440x1080，ROI 带内有胶条）。

早期 mock 图是 640x480，与 ROI=(0,405,1439,259) 不匹配，导致 main.py 演示
全部 NG。本脚本重新生成一批正确尺寸的演示图，使 demo 与真实产线一致。

运行：
  python tools/make_mock_images.py          # 生成 8 张到 mock/sample_images
  python tools/make_mock_images.py --n 12   # 指定张数
"""
import os
import argparse
import numpy as np
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys = __import__("sys")
sys.path.insert(0, ROOT)
from detection.detector import ROI


def make_roof(width, height, seed=0):
    rng = np.random.default_rng(seed)
    yy = np.linspace(0, 1, height)[:, None]
    base = 205 - 25 * yy
    img = np.repeat(base, width, axis=1).astype(np.float32)
    img += rng.normal(0, 8, (height, width))
    img = np.clip(img, 0, 255).astype(np.uint8)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    x, y, w, h = ROI
    sy = y + int(h * 0.35)
    sh = int(h * 0.30)
    # 暗色胶条：模拟接近黑色的橡胶密封条（灰度~30），匹配算法阈值下限40
    seal = np.full((sh, w, 3), 30.0, dtype=np.float32)
    hl = rng.normal(0, 6, (sh, w)).astype(np.float32)[..., None]
    seal = np.clip(seal + hl, 0, 255).astype(np.uint8)
    vis[sy:sy + sh, 0:w] = seal
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    W, H = 1440, 1080
    out_dir = os.path.join(ROOT, "mock", "sample_images")
    os.makedirs(out_dir, exist_ok=True)
    for i in range(args.n):
        im = make_roof(W, H, seed=100 + i)
        p = os.path.join(out_dir, f"seal_{i:02d}.jpg")
        cv2.imwrite(p, im, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"已生成 {args.n} 张 {W}x{H} 演示图到 {out_dir}")
    print("提示：data/stored_images 为运行时副本，运行 main.py 后会自动刷新。")


if __name__ == "__main__":
    main()
