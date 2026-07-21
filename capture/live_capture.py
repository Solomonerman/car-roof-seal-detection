# -*- coding: utf-8 -*-
"""现场相机实时采集工具。

用法：
  1) 确认上位机已装 pypylon： pip install pypylon opencv-python
  2) python capture/live_capture.py
  3) 预览窗口出现，按【空格】触发一轮连拍，按【q】退出。

连拍参数（集中在顶部，按现场情况调整）：
  - 拍摄帧率 / 持续时长 / 总张数
  - 相机 IP / 序列号（二选一直连）
  - 保存目录 / 文件名格式

不依赖 PLC，纯手动触发。拍到的照片直接落入 data/raw_images/，
后续可直接用 tools/validate_detector.py --src data/raw_images 批量检测。
"""
import os
import sys
import time
import datetime
import argparse
import cv2

try:
    import pypylon.pylon as py
    HAS_PYPYLON = True
except ImportError:
    HAS_PYPYLON = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ===================== 连拍参数（现场可按需修改） =====================
FPS = 3                # 连拍帧率（张/秒）
DURATION_SEC = 5       # 连拍持续时长（秒）
# 总张数 = FPS × DURATION_SEC，当前 3×5 = 15 张

# ===================== 相机连接参数 =====================
# 二选一：填 IP 或序列号（都填则以 IP 优先）
CAMERA_IP = "172.30.173.249"   # 现场左侧 Basler aca1920-48gm
CAMERA_SERIAL = ""             # 例如 "12345678"
# 如果两个都空，脚本自动搜索第一台可用的 Basler 相机

# ===================== 相机采集参数 =====================
EXPOSURE_TIME_US = 2000      # 曝光时间（微秒）
# 增益：None = 沿用相机当前值（不修改）。现场暗光、曝光锁 2000µs 时，
# 你在 pylon Viewer 设的 Gain Raw 136 会被保留。想固定值就填 dB，例如 6.0。
GAIN_DB = None
GAMMA = 1.0                  # Gamma，默认 1.0
PIXEL_FORMAT = "Mono8"       # 像素格式：aca1920-48gm 是黑白相机，Mono8 = 8bit 灰度

# ===================== 存储参数 =====================
SAVE_DIR = os.path.join(ROOT, "data", "raw_images")
SAVE_EXT = ".bmp"      # BMP 无损，适合算法检测；也可改 ".jpg"

# ===================== 预览参数 =====================
PREVIEW_ENABLE = True
PREVIEW_WIDTH = 960    # 预览窗口宽度（等比缩放，不改变原始采集分辨率）

# ================================================================


def _find_camera():
    """发现并连接 Basler 相机。返回 (camera, info_dict)。"""
    if not HAS_PYPYLON:
        print("[错误] 未安装 pypylon，请先在上位机执行：pip install pypylon")
        sys.exit(1)

    factory = py.TlFactory.GetInstance()

    if CAMERA_IP:
        info = py.DeviceInfo()
        info.SetPropertyValue("IpAddress", CAMERA_IP)
        try:
            cam = py.InstantCamera(factory.CreateDevice(info))
        except Exception:
            print(f"[错误] 无法连接 IP={CAMERA_IP} 的相机。请检查：")
            print("  1) 相机是否已通电、网线是否连接")
            print("  2) 上位机 IP 是否与相机在同一网段")
            print("  3) 防火墙是否阻止了 GigE 通信")
            print(f"  4) 用 Basler pylon IP Configurator 确认相机 IP 是否为 {CAMERA_IP}")
            sys.exit(1)
        cam.Open()
        return cam, {"ip": CAMERA_IP, "serial": cam.GetDeviceInfo().GetSerialNumber()}

    if CAMERA_SERIAL:
        info = py.DeviceInfo()
        info.SetPropertyValue("SerialNumber", CAMERA_SERIAL)
        cam = py.InstantCamera(factory.CreateDevice(info))
        cam.Open()
        return cam, {"ip": "auto", "serial": CAMERA_SERIAL}

    # 自动发现第一台
    devices = factory.EnumerateDevices()
    if not devices:
        print("[错误] 未发现任何 Basler 相机。请检查：")
        print("  1) 相机是否已通电")
        print("  2) 网线是否连接")
        print("  3) 防火墙是否阻止了 GigE 通信")
        sys.exit(1)

    cam = py.InstantCamera(factory.CreateDevice(devices[0]))
    cam.Open()
    info = {
        "ip": devices[0].GetIpAddress(),
        "serial": devices[0].GetSerialNumber(),
    }
    return cam, info


def _configure_camera(cam):
    """设置相机采集参数：曝光、增益、Gamma、像素格式。"""
    nodemap = cam.GetNodeMap()
    # 顺序注意：先配增益/Gamma/曝光，最后再改像素格式。
    # 改 PixelFormat 会触发相机重配置，可能让曝光节点暂时不可写。
    # 增益（None = 沿用相机当前值，不修改）
    if GAIN_DB is None:
        print("[采集] 增益沿用相机当前值（不修改）")
    else:
        try:
            gain_node = nodemap.GetNode("Gain")
            gain_node.SetValue(GAIN_DB)
            print(f"[采集] 增益已设为 {GAIN_DB} dB")
        except Exception:
            print("[采集] 增益设置失败")
    # Gamma
    try:
        gamma_node = nodemap.GetNode("Gamma")
        gamma_node.SetValue(GAMMA)
        print(f"[采集] Gamma 已设为 {GAMMA}")
    except Exception:
        print("[采集] Gamma 设置失败")
    # 曝光：自动模式先关掉，否则手动曝光值写不进去
    try:
        auto_node = nodemap.GetNode("ExposureAuto")
        if auto_node:
            auto_node.SetValue("Off")
            print("[采集] 曝光自动已设为 Off")
    except Exception as e:
        print(f"[采集] 曝光自动设置异常: {e}")
    # 曝光模式设为 Timed（部分相机默认非 Timed，手动曝光会写不进）
    try:
        mode_node = nodemap.GetNode("ExposureMode")
        if mode_node:
            mode_node.SetValue("Timed")
            print("[采集] 曝光模式已设为 Timed")
    except Exception as e:
        print(f"[采集] 曝光模式设置异常: {e}")
    # 曝光时间：依次尝试 ExposureTime / ExposureTimeAbs
    # 注：aca1920-48gm 上 ExposureTime 是占位节点(not available)，
    #     真正可写的是 ExposureTimeAbs；占位失败属预期，静默跳过避免误报。
    exp_ok = False
    for name in ("ExposureTime", "ExposureTimeAbs"):
        try:
            node = nodemap.GetNode(name)
            if node is None:
                continue
            node.SetValue(EXPOSURE_TIME_US)
            print(f"[采集] 曝光时间已设为 {EXPOSURE_TIME_US} µs ({name})")
            exp_ok = True
            break
        except Exception as e:
            msg = str(e)
            if "not available" in msg.lower() or "placeholder" in msg.lower():
                continue
            print(f"[采集] {name} 设置失败: {msg}")
    if not exp_ok:
        print("[采集] 曝光未写入，沿用相机当前曝光值")
    # 采集帧率：锁成 FPS，保证连拍时序确定（避免 RetrieveResult 阻塞导致时长漂移）
    try:
        fr_en = nodemap.GetNode("AcquisitionFrameRateEnable")
        if fr_en is not None:
            fr_en.SetValue(True)
        fr = nodemap.GetNode("AcquisitionFrameRate")
        if fr is not None:
            fr.SetValue(FPS)
            print(f"[采集] 采集帧率已设为 {FPS}fps")
    except Exception as e:
        print(f"[采集] 采集帧率设置异常: {e}")
    # 像素格式放最后
    try:
        fmt_node = nodemap.GetNode("PixelFormat")
        fmt_node.SetValue(PIXEL_FORMAT)
        print(f"[采集] 像素格式已设为 {PIXEL_FORMAT}")
    except Exception:
        print(f"[采集] 像素格式设置失败，将使用相机当前值")


def _get_pixel_format_name(grab_result):
    """从抓取结果获取像素格式的友好名称。"""
    try:
        return grab_result.GetPixelType()
    except Exception:
        return "未知"


def _format_filename(prefix="Image"):
    """生成带时间戳的文件名，与现有 raw_images 命名一致。"""
    ts = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M-%S-%f")[:-3]
    return f"{prefix}__{ts}{SAVE_EXT}"


def main():
    parser = argparse.ArgumentParser(description="现场相机实时采集")
    parser.add_argument("--no-preview", action="store_true", help="关闭预览窗口")
    parser.add_argument("--fps", type=float, default=FPS, help="连拍帧率")
    parser.add_argument("--duration", type=float, default=DURATION_SEC, help="连拍持续秒数")
    parser.add_argument("--out", default=SAVE_DIR, help="保存目录")
    args = parser.parse_args()

    fps = args.fps
    duration = args.duration
    total = int(fps * duration)
    save_dir = args.out
    preview = PREVIEW_ENABLE and not args.no_preview

    # 连接相机
    print("[采集] 正在连接 Basler 相机 ...")
    cam, info = _find_camera()
    print(f"[采集] 已连接：型号={cam.GetDeviceInfo().GetModelName()}"
          f"  序列号={info['serial']}  IP={info.get('ip', 'N/A')}")

    # 配置相机
    _configure_camera(cam)
    cam.StartGrabbing(py.GrabStrategy_LatestImageOnly)
    # 获取一张以确认分辨率和像素格式
    grab = cam.RetrieveResult(5000, py.TimeoutHandling_ThrowException)
    if not grab.GrabSucceeded():
        print("[错误] 相机取图失败，请检查连接和配置")
        cam.StopGrabbing()
        cam.Close()
        sys.exit(1)

    img = grab.Array
    h, w = img.shape[:2]
    is_color = len(img.shape) == 3
    actual_fmt = _get_pixel_format_name(grab)
    print(f"[采集] 分辨率={w}×{h}  色彩={'彩色' if is_color else '黑白'}  像素格式={actual_fmt}")
    grab.Release()

    os.makedirs(save_dir, exist_ok=True)
    print(f"[采集] 保存目录：{save_dir}")
    print(f"[采集] 连拍参数：{fps} 张/秒 × {duration} 秒 = {total} 张")
    print(f"[采集] 按 [空格] 开始连拍，按 [q] 退出")
    print("=" * 55)

    frame_interval = 1.0 / fps
    cv2.namedWindow("Preview", cv2.WINDOW_NORMAL) if preview else None

    try:
        while True:
            # 取一帧用于预览
            grab = cam.RetrieveResult(100, py.TimeoutHandling_Return)
            if grab and grab.GrabSucceeded():
                frame = grab.Array.copy()
                grab.Release()
            else:
                if grab:
                    grab.Release()
                continue

            if preview:
                # 等比缩放预览
                scale = PREVIEW_WIDTH / w
                preview_frame = cv2.resize(frame, (PREVIEW_WIDTH, int(h * scale)))
                if not is_color:
                    preview_frame = cv2.cvtColor(preview_frame, cv2.COLOR_GRAY2BGR)
                cv2.imshow("Preview", preview_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("\n[采集] 用户退出")
                break

            if key == ord(' '):  # 空格
                print(f"\n[采集] 开始连拍：{total} 张，间隔 {frame_interval:.2f}s ...")
                batch = []
                t0 = time.perf_counter()

                for i in range(total):
                    t_start = time.perf_counter()
                    grab = cam.RetrieveResult(3000, py.TimeoutHandling_ThrowException)
                    if not grab.GrabSucceeded():
                        print(f"  [警告] 第 {i + 1}/{total} 张取图失败")
                        grab.Release()
                        continue

                    img_save = grab.Array.copy()
                    grab.Release()
                    fname = _format_filename()
                    fpath = os.path.join(save_dir, fname)
                    cv2.imwrite(fpath, img_save)
                    batch.append(fpath)
                    elapsed = time.perf_counter() - t_start
                    wait_ms = max(0, int((frame_interval - elapsed) * 1000))
                    if wait_ms > 0 and i < total - 1:
                        key_wait = cv2.waitKey(wait_ms) & 0xFF
                        if key_wait == ord('q'):
                            print("  [中断] 用户中止连拍")
                            break

                total_elapsed = time.perf_counter() - t0
                actual_fps = len(batch) / total_elapsed if total_elapsed > 0 else 0
                print(f"[采集] 完成！实际拍 {len(batch)}/{total} 张"
                      f"  耗时 {total_elapsed:.1f}s  实际帧率 {actual_fps:.1f} fps")
                for p in batch:
                    print(f"  {os.path.basename(p)}")
                print(f"[采集] 按 [空格] 再拍一轮，按 [q] 退出")
                print("=" * 55)

    except KeyboardInterrupt:
        print("\n[采集] Ctrl+C 退出")
    finally:
        if preview:
            cv2.destroyAllWindows()
        cam.StopGrabbing()
        cam.Close()
        print("[采集] 相机已释放")


if __name__ == "__main__":
    main()
