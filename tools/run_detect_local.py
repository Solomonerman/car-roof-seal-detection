#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地检测验证工具（算法优化迭代用）
================================
把任意本地图片文件夹喂给胶条检测器，输出过程图(_mask.png / _offset.png)与汇总，
方便在本地(本机 .venv，需 torch/ultralytics/cv2)快速验证「算法」改动。

与 web_capture.py 的区别：
  - 纯离线：不拍照、不写数据库、不改任何命名/看板；
  - 只做「读图 -> 跑 detector.detect -> 写过程图 + summary.json」；
  - 因此本工具产出的结果图可以用于肉眼对比「优化前/优化后」差异。

用法：
  # 按车分文件夹：src/车A/*.bmp, src/车B/*.bmp
  python tools/run_detect_local.py --src "D:/误报照片" --key 8X

  # 散图（直接放 .bmp/.png 在 src 下）
  python tools/run_detect_local.py --src "D:/误报照片/散图" --key 8X

  # 只验证某几台车（车名或文件名含该子串，加速迭代）
  python tools/run_detect_local.py --src "D:/误报照片" --only 8421

  # 强制重跑（忽略已存在的 _mask.png，用于换算法后复测）
  python tools/run_detect_local.py --src "D:/误报照片" --force

输出结构：
  <out>/<车名>/<图名>_mask.png      青色胶条叠加 + 红框 + 宽度标注(规格内白/超范围红+宽度异常)
  <out>/<车名>/<图名>_offset.png    橙色偏移带图
  <out>/<车名>/summary.json         逐图宽度/偏移/ok/loc_ok
  <out>/检测汇总.json               全部车的 OK/NG 总览

典型迭代节奏（算法优化期间）：
  1) 工控机把误报照片(整文件夹或散图)下载到本地某目录；
  2) 在 _pushrepo 改 detection/ 算法 -> 提交(不推送)；
  3) 本机 pull(或手动同步)到 PycharmProjects/dev，用 dev 的 .venv 跑本工具；
  4) 把 <out> 下的 _mask/_offset 图 + 检测汇总.json 贴回聊天窗口给我看；
  5) 我据结果继续调 -> 重复 2~4；
  6) 算法稳定后，只把 detection/ 相关提交推送给工控机 pull 生效。
"""
import argparse
import json
import os
import sys
import time

# 让脚本无论从哪个工作目录运行都能 import 项目模块
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

IMG_EXTS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")


def collect_cars(src):
    """返回 [(车名, [图片绝对路径...]), ...]。

    识别三种布局：
      (a) 按车分文件夹：src/车A/*.bmp, src/车B/*.bmp
      (b) 散图：src/*.bmp
      (c) 其他：递归扫描所有图片，统一归入 <散图>
    """
    entries = sorted(os.listdir(src))
    subdirs = [e for e in entries if os.path.isdir(os.path.join(src, e))]
    direct_imgs = [e for e in entries
                   if e.lower().endswith(IMG_EXTS)
                   and os.path.isfile(os.path.join(src, e))]
    cars = []
    # 子目录里含图片的，视为「按车模式」；同层散图另归一台，避免漏图
    subdirs_with_imgs = []
    for sd in subdirs:
        d = os.path.join(src, sd)
        sub_imgs = sorted([os.path.join(d, f) for f in os.listdir(d)
                           if f.lower().endswith(IMG_EXTS)
                           and os.path.isfile(os.path.join(d, f))])
        if sub_imgs:
            subdirs_with_imgs.append((sd, sub_imgs))
    if subdirs_with_imgs:                        # (a) 按车模式
        cars.extend(subdirs_with_imgs)
        if direct_imgs:
            cars.append(("<散图>", [os.path.join(src, f) for f in direct_imgs]))
    elif direct_imgs:                            # (b) 纯散图模式
        cars.append(("<散图>", [os.path.join(src, f) for f in direct_imgs]))
    else:                                       # (c) 递归兜底
        all_imgs = []
        for r, _, files in os.walk(src):
            for f in files:
                if f.lower().endswith(IMG_EXTS):
                    all_imgs.append(os.path.join(r, f))
        if all_imgs:
            cars.append(("<散图>", sorted(all_imgs)))
    return cars


def already_done(out_dir, imgs):
    """若所有图都已有 _mask.png，视为已检测（可跳过）。"""
    return all(os.path.exists(
        os.path.join(out_dir, os.path.splitext(os.path.basename(p))[0] + "_mask.png"))
        for p in imgs)


def load_summary(out_dir):
    sp = os.path.join(out_dir, "summary.json")
    if os.path.exists(sp):
        try:
            return json.load(open(sp, encoding="utf-8"))
        except Exception:
            return None
    return None


def main():
    ap = argparse.ArgumentParser(description="本地检测验证工具(算法迭代用)")
    ap.add_argument("--src", required=True, help="图片/文件夹路径（按车分文件夹或散图）")
    ap.add_argument("--out", default=None, help="输出根目录（默认 <src>/_detect_out）")
    ap.add_argument("--key", default="8X", help="车型 key（8X/NM/9X 等，默认 8X）")
    ap.add_argument("--only", default=None, help="仅处理车名/文件名含该子串的（加速验证）")
    ap.add_argument("--force", action="store_true",
                    help="强制重跑（忽略已存在的 _mask.png）")
    args = ap.parse_args()

    if not os.path.isdir(args.src):
        print("[错误] src 不是目录: %s" % args.src)
        sys.exit(2)

    out_root = args.out or os.path.join(args.src, "_detect_out")
    os.makedirs(out_root, exist_ok=True)

    # 载入检测器（优先用生产 router，失败则退化到 SealOffsetDetector）
    detector = None
    try:
        from detection.router import get_detector
        detector = get_detector(args.key)
    except Exception as e:
        print("[提示] get_detector(%s) 失败: %s，退化用 SealOffsetDetector" % (args.key, e))
    if detector is None:
        try:
            from detection.seal_offset_detector import SealOffsetDetector
            detector = SealOffsetDetector()
        except Exception as e:
            print("[错误] 无法载入检测器: %s" % e)
            sys.exit(3)

    cars = collect_cars(args.src)
    if args.only:
        cars = [(n, imgs) for n, imgs in cars
                if args.only in n
                or any(args.only in os.path.basename(p) for p in imgs)]
    if not cars:
        print("[警告] 未在 %s 找到任何图片" % args.src)
        sys.exit(0)

    print("=" * 64)
    print("源: %s" % args.src)
    print("车型 key: %s | 车数: %d | 仅筛选: %s | 强制: %s"
          % (args.key, len(cars), args.only, args.force))
    print("输出: %s" % out_root)
    print("=" * 64)

    summary_all = []
    t0 = time.time()
    for ci, (car_name, imgs) in enumerate(cars, 1):
        out_dir = os.path.join(out_root, car_name)
        if not args.force and already_done(out_dir, imgs):
            print("[跳过] %s 已检测(%d 张，--force 可重跑)" % (car_name, len(imgs)))
            s = load_summary(out_dir)
            if s is not None:
                summary_all.append({"car": car_name, **s})
            continue
        os.makedirs(out_dir, exist_ok=True)
        tc = time.time()
        try:
            det = detector.detect(car_model=args.key, images=imgs, process_dir=out_dir)
            ok, msg = det.ok, det.message
        except Exception as e:
            ok, msg = False, "检测异常:%s" % e
            print("  [异常] %s: %s" % (car_name, e))
        dt = time.time() - tc
        print("[%d/%d] %s: %s (%.1fs)"
              % (ci, len(cars), car_name, "OK" if ok else "NG", dt))
        s = load_summary(out_dir)
        if s is not None:
            summary_all.append({"car": car_name, **s})

    total = len(summary_all)
    ng = sum(1 for s in summary_all if not s.get("ok", True))
    overview = {
        "src": args.src, "key": args.key, "cars": total, "ng": ng,
        "ok": total - ng, "elapsed_sec": round(time.time() - t0, 1),
        "details": summary_all,
    }
    ov_path = os.path.join(out_root, "检测汇总.json")
    with open(ov_path, "w", encoding="utf-8") as f:
        json.dump(overview, f, ensure_ascii=False, indent=2)
    print("=" * 64)
    print("完成：共 %d 台车，NG %d / OK %d，耗时 %.1fs"
          % (total, ng, total - ng, overview["elapsed_sec"]))
    print("总览: %s" % ov_path)


if __name__ == "__main__":
    main()
