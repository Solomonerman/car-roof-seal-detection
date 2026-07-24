# -*- coding: utf-8 -*-
"""现场相机 · 网页版实时取景 + 远程触发采集 + PLC 自动触发检测。

为什么用这个：
  你在工位远程操作上位机，看不到现场车子。pylon Viewer 的实时窗口在
  上位机本地，你隔着远程桌面才能看到，且它独占相机、会和本脚本冲突。
  这个工具把相机的实时画面通过浏览器（局域网/公网）推给你，你在网页上
  就能看到车有没有到位，点一下按钮就触发连拍，无需远程桌面、无需 pylon Viewer。

自动触发（--plc-auto）：
  后台启一个【只读】PLC 监控线程，监测 DB130.DBX0.1（出车信号）上升沿；
  上升沿时按"先锁存后打印"取到车号上下文（滑橇/PIN/车型/NO_Paint），
  记录全部车（含免检），并对(9X/8X 且 NO_Paint=0)的车触发预触发拍照；
  全程仅 read_area，绝不写 PLC（结果回写按现场要求暂未实现）。

用法：
  1) 上位机安装依赖： pip install pypylon opencv-python flask
  2) 关闭 pylon Viewer（避免相机被占用）
  3) 手动模式： python capture/web_capture.py
     自动模式： python capture/web_capture.py --plc-auto
  4) 浏览器打开 http://<上位机IP>:5000 看实时画面、点按钮或看自动记录

连拍 / 相机参数 / PLC 参数都集中在顶部，按现场情况调整。
依赖：Flask（网页服务） + pypylon（相机） + opencv（图像） + python-snap7（仅自动模式）
"""
import os
import sys
import time
import json
import datetime
import threading
import argparse
import collections
import concurrent.futures as _cf

import cv2

try:
    import pypylon.pylon as py
    HAS_PYPYLON = True
except ImportError:
    HAS_PYPYLON = False

try:
    from flask import Flask, Response, request, jsonify
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# ===================== 网页服务参数 =====================
WEB_HOST = "0.0.0.0"     # 0.0.0.0 = 允许任意网卡访问（远程可见）；仅本机可改 127.0.0.1
WEB_PORT = 5000          # 浏览器访问端口（上位机防火墙需放行此端口）

# ===================== 连拍参数（现场可按需修改） =====================
FPS = 3                # 连拍帧率（张/秒）
DURATION_SEC = 7       # 连拍持续时长（秒）
# 总张数 = FPS × DURATION_SEC，当前 3×7 = 21 张
# 触发方式：
#   "pre"  = 预触发环形缓冲（推荐）：后台一直缓存最近 DURATION_SEC 秒的帧，
#            点按钮/出车信号立即把缓冲里已有的帧存盘 → 拍的是信号那一刻及之前的画面，
#            对运动物体无延迟（解决"点的时候是车头、拍到的是车身"问题）。
#   "post" = 点击/信号后才开始连拍 DURATION_SEC 秒（旧逻辑，运动物体会有明显延迟）。
CAPTURE_MODE = "pre"

# ===================== 相机连接参数 =====================
# 现场相机 IP 列表（多相机就绪：左右相机共用同一出车信号，分别把 IP 追加进来即可，
# 其余代码无需改动；当前仅左相机一台）。第二台接入后，自动模式会同时触发全部相机。
CAMERA_IPS = ["172.30.173.249"]   # 现场左侧 Basler aca1920-48gm
CAMERA_SERIAL = ""             # 留空则用上面列表的 IP；也可填序列号直连（单相机场景）

# ===================== 相机采集参数 =====================
EXPOSURE_TIME_US = 2000      # 曝光时间（微秒）
# 增益：设为 None = 沿用相机【当前值】、不修改。
#   你在 pylon Viewer 里已设 Gain Raw 136（现场暗光、曝光锁 2000µs 的可用档），
#   关掉 pylon 跑本程序时会保留该设置，拍出的图不会变暗。
#   若想强制指定固定值，填 dB 数字，例如 6.0。
#   ⚠️ 注意：相机断电/复位后会恢复默认（通常 0 dB），届时需要重设或在此填固定值。
GAIN_DB = None
GAMMA = 1.0                  # Gamma，默认 1.0（保持线性，利于检测）
PIXEL_FORMAT = "Mono8"       # 黑白相机 8bit 灰度

# ===================== 存储参数 =====================
SAVE_DIR = os.path.join(ROOT, "data", "raw_images")      # 预触发临时缓冲(拍完即被移走)
INSPECTION_ROOT = os.path.join(ROOT, "data", "inspection")  # 最终存储根(按 日期/PIN 分层)
SAVE_EXT = ".bmp"            # BMP 无损，适合算法检测；也可改 ".jpg"
RECORD_DIR = os.path.join(ROOT, "data", "records")   # 自动触发追溯 sidecar JSON(非拍照车兜底)

# ===================== 预览参数 =====================
STREAM_WIDTH = 960           # 网页实时流宽度（等比缩放，不改原始采集分辨率）
JPEG_QUALITY = 80            # 实时流 JPEG 质量（1-100，越小越流畅但越糊）

# ===================== CPU 占用控制（与机器人站共用上位机，必须省）=====================
# 相机自由运行在 ~48fps，但本程序不上满：预览 + 预触发只要几 fps 就够。
#  - GRAB_FPS  ：后台取流节拍上限（默认 6）。超出部分由相机/驱动直接丢弃，不进 Python。
#  - STREAM_FPS：网页 MJPEG 推送上限（默认 12），避免无谓的 JPEG 编码拖 CPU。
#  - 预触发环形缓冲仍按 FPS(3) 节流写入，保证 21 张铺满 7 秒（见 CameraStreamer._loop）。
# 三项叠加，相比"48fps 裸取 + 不限速编码"，CPU 与 GigE 带宽下降一个数量级。
GRAB_FPS = 6
STREAM_FPS = 12

# ===================== PLC 自动触发参数（--plc-auto 启用）=====================
PLC_IP = "172.30.173.6"
PLC_RACK = 0
PLC_SLOT = 2
PLC_POLL_MS = 20             # 出车信号轮询间隔（毫秒，原10；共用上位机降到20省CPU，仍足够抓车）
PLC_CTX_MS = 200             # DB230 上下文采样间隔（毫秒）

# 缺陷标签集合（出现 → 整体 NG）；与 detection.detector.NG_LABELS 保持一致
NG_LABELS = {"missing", "break", "overspray", "width"}

# ================================================================


class CameraStreamer:
    """后台线程：持续取流 → 存最新帧；按需执行连拍。单台相机。"""

    def __init__(self, camera_ip=None):
        self.camera_ip = camera_ip
        self.cam = None
        self.running = False
        self.is_color = False
        self.width = 0
        self.height = 0
        self.actual_fmt = "未知"
        self.camera_info = {}

        self._latest = None
        self._lock = threading.Lock()

        self._capture_req = threading.Event()   # 触发连拍
        self._capture_done = threading.Event()  # 连拍完成信号
        self._last_result = None                # 最近一次连拍结果

        # 预触发环形缓冲：始终保留最近 total 帧（点击即存，避免运动物体延迟）
        self._ring = collections.deque(maxlen=int(FPS * DURATION_SEC) + 5)

        self._thread = None
        self._error = None

    # ---------- 连接与配置 ----------
    def connect(self):
        if not HAS_PYPYLON:
            raise RuntimeError("未安装 pypylon，请先在上位机执行：pip install pypylon")

        factory = py.TlFactory.GetInstance()
        ip = self.camera_ip
        if ip:
            info = py.DeviceInfo()
            info.SetPropertyValue("IpAddress", ip)
            try:
                self.cam = py.InstantCamera(factory.CreateDevice(info))
            except Exception:
                raise RuntimeError(
                    f"无法连接 IP={ip} 的相机。检查：通电/网线/同网段/防火墙/"
                    f"用 pylon IP Configurator 确认 IP。若 pylon Viewer 开着请先关闭。"
                )
        elif CAMERA_SERIAL:
            info = py.DeviceInfo()
            info.SetPropertyValue("SerialNumber", CAMERA_SERIAL)
            self.cam = py.InstantCamera(factory.CreateDevice(info))
        else:
            devices = factory.EnumerateDevices()
            if not devices:
                raise RuntimeError("未发现任何 Basler 相机。检查通电/网线/防火墙。")
            self.cam = py.InstantCamera(factory.CreateDevice(devices[0]))

        self.cam.Open()
        self.camera_info = {
            "model": self.cam.GetDeviceInfo().GetModelName(),
            "serial": self.cam.GetDeviceInfo().GetSerialNumber(),
            "ip": ip or (self.cam.GetDeviceInfo().GetIpAddress()
                         if hasattr(self.cam.GetDeviceInfo(), "GetIpAddress") else "N/A"),
        }
        self._configure()

    def _configure(self):
        nodemap = self.cam.GetNodeMap()
        conf_log = []
        # 顺序注意：先配增益/Gamma/曝光，最后再改像素格式。
        # 改 PixelFormat 会触发相机重配置，可能让曝光节点暂时不可写。
        # 增益（None = 沿用相机当前值，不修改）
        if GAIN_DB is None:
            conf_log.append("增益=沿用相机当前值(不修改)")
        else:
            try:
                nodemap.GetNode("Gain").SetValue(GAIN_DB)
                conf_log.append(f"增益={GAIN_DB}dB")
            except Exception as e:
                conf_log.append(f"增益设置异常: {e}")
        # Gamma
        try:
            nodemap.GetNode("Gamma").SetValue(GAMMA)
            conf_log.append(f"Gamma={GAMMA}")
        except Exception as e:
            conf_log.append(f"Gamma设置异常: {e}")
        # 曝光：自动模式必须先关掉，否则手动曝光值写不进去
        try:
            auto_node = nodemap.GetNode("ExposureAuto")
            if auto_node:
                auto_node.SetValue("Off")
                conf_log.append("曝光自动=Off")
        except Exception as e:
            conf_log.append(f"曝光自动设置异常: {e}")
        # 曝光模式设为 Timed（部分相机默认非 Timed，手动曝光会写不进）
        try:
            mode_node = nodemap.GetNode("ExposureMode")
            if mode_node:
                mode_node.SetValue("Timed")
                conf_log.append("曝光模式=Timed")
        except Exception as e:
            conf_log.append(f"曝光模式设置异常: {e}")
        # 曝光：依次尝试 ExposureTime / ExposureTimeAbs
        # 注：aca1920-48gm 上 ExposureTime 是占位节点(not available)，
        #     真正可写的是 ExposureTimeAbs；占位失败属预期，静默跳过避免误报。
        exp_ok = False
        for name in ("ExposureTime", "ExposureTimeAbs"):
            try:
                node = nodemap.GetNode(name)
                if node is None:
                    continue
                node.SetValue(EXPOSURE_TIME_US)
                conf_log.append(f"曝光={EXPOSURE_TIME_US}µs ({name})")
                exp_ok = True
                break
            except Exception as e:
                msg = str(e)
                if "not available" in msg.lower() or "placeholder" in msg.lower():
                    continue
                conf_log.append(f"{name} 设置失败: {msg}")
        if not exp_ok:
            conf_log.append("曝光未写入，沿用相机当前曝光值")
        # 采集帧率：本意锁成 FPS 以保证时序确定。但 aca1920-48gm 上
        # AcquisitionFrameRate 是占位节点(not available)，相机端锁定会失败。
        # 【关键修复】仅当“成功写入帧率值”时才打开帧率限制开关。
        #   若只打开 AcquisitionFrameRateEnable 却没写进有效帧率值，相机会进入
        #   异常节流/冻结状态——预触发环形缓冲在 7s 内采到的 21 帧全是同一张
        #   （现场复现：21 张同一照片）。时序确定性完全由 _loop 的软件节流保证，
        #   因此相机保持自由运行即可，绝不强行打开锁定开关。
        try:
            fr = nodemap.GetNode("AcquisitionFrameRate")
            if fr is not None:
                fr.SetValue(FPS)
                fr_en = nodemap.GetNode("AcquisitionFrameRateEnable")
                if fr_en is not None:
                    fr_en.SetValue(True)
                conf_log.append(f"采集帧率=相机锁定 {FPS}fps")
            else:
                conf_log.append(f"采集帧率=相机未开放锁定(占位)，改由软件节流 {FPS}fps")
        except Exception as e:
            msg = str(e)
            if "not available" in msg.lower() or "placeholder" in msg.lower():
                # 相机未开放帧率锁定节点：预期内，由软件节流兜底（绝不打开锁定开关）
                conf_log.append(f"采集帧率=相机未开放锁定(占位)，改由软件节流 {FPS}fps")
            else:
                conf_log.append(f"采集帧率设置异常(已忽略): {msg}")
        # 像素格式放最后
        try:
            nodemap.GetNode("PixelFormat").SetValue(PIXEL_FORMAT)
            conf_log.append(f"像素格式={PIXEL_FORMAT}")
        except Exception as e:
            conf_log.append(f"像素格式设置异常(沿用当前值): {e}")
        self._conf_log = conf_log

    def start(self):
        # 抓取策略：用 LatestImageOnly（与现场验证可用的 live_capture.py 一致）。
        # 关键：后台取流线程轮询仅 GRAB_FPS=6，远慢于相机产出 48fps。
        #   OneByOne 会把帧按顺序塞进有限缓冲池，慢轮询下缓冲池耗尽 → 相机触发
        #   背压、停采 → RetrieveResult 反复返回同一冻结帧 → 预触发 21 帧全相同
        #   （现场复现：同一张照片）。LatestImageOnly 始终返回最新帧、自动丢弃旧帧，
        #   慢轮询也能拿到各不相同的新鲜帧；且轮询(6fps)慢于产出(48fps)，不会重复。
        #   时序确定性仍由 _loop 的 3fps 软件节流保证，不受策略影响。
        self.cam.StartGrabbing(py.GrabStrategy_LatestImageOnly)
        # 取一张确认分辨率
        grab = self.cam.RetrieveResult(5000, py.TimeoutHandling_ThrowException)
        if not grab.GrabSucceeded():
            self.cam.StopGrabbing()
            self.cam.Close()
            raise RuntimeError("相机取图失败，请检查连接与配置。")
        img = grab.Array
        self.height, self.width = img.shape[:2]
        self.is_color = len(img.shape) == 3
        try:
            self.actual_fmt = grab.GetPixelType()
        except Exception:
            self.actual_fmt = "未知"
        grab.Release()

        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        """后台取流线程。

        - 直接按 FPS(3) 节拍取图并写入环形缓冲,避免“6fps 取流 + 3fps 写 ring”两层
          节流的精度误差导致一帧被连续写两次(现场复现:01/02、03/04…成对相同)。
        - 每帧都用 TimeoutHandling_ThrowException + 3000ms 取最新帧,取不到就等下一拍。
        - 21 张 × 1/3s = 7s,时序完全由软件节拍保证(aca1920-48gm 帧率节点不可用)。
        """
        frame_interval = 1.0 / FPS
        next_frame_clock = time.perf_counter()
        while self.running:
            try:
                # 睡到下一帧该来的时刻,然后取一帧、写一次 ring。不取 6fps 中间帧,
                # 彻底消除同一帧被 throttle 判定通过两次的问题。
                wait = next_frame_clock - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                next_frame_clock = time.perf_counter() + frame_interval

                try:
                    grab = self.cam.RetrieveResult(3000, py.TimeoutHandling_ThrowException)
                except Exception:
                    time.sleep(0.05)
                    continue
                if grab and grab.GrabSucceeded():
                    frame = grab.Array.copy()
                    grab.Release()
                    ts = time.time()                     # 命名/计时用真实采集时刻
                    with self._lock:
                        self._latest = frame            # 预览：每帧都更新
                    self._ring.append((ts, frame))      # 每拍只写一次,无重复判定
                    if self._capture_req.is_set():
                        self._flush_buffer()
                elif grab:
                    grab.Release()
            except Exception as e:
                # 单帧异常不应杀死取流线程，否则相机静默停拍且无人察觉
                self._error = f"取流异常: {e}"
                print(f"[相机] 取流异常(已忽略，继续): {e}")
                time.sleep(0.1)

    def _flush_buffer(self):
        """预触发：把环形缓冲里最近的 total 帧立即存盘。

        拍的是信号那一刻及之前的画面（缓冲已缓存最近 DURATION_SEC 秒），
        因此对运动物体没有"信号后才开始抓"的延迟。
        """
        self._capture_req.clear()
        total = int(FPS * DURATION_SEC)
        try:
            frames = list(self._ring)
            if len(frames) > total:
                frames = frames[-total:]
            os.makedirs(SAVE_DIR, exist_ok=True)
            batch = []
            t0 = time.perf_counter()
            for ts, frame in frames:
                fname = self._format_filename_ts(ts)
                fpath = os.path.join(SAVE_DIR, fname)
                cv2.imwrite(fpath, frame)
                batch.append(os.path.basename(fpath))
            elapsed = time.perf_counter() - t0
            span = (frames[-1][0] - frames[0][0]) if len(frames) > 1 else 0.0
            self._last_result = {
                "ok": True,
                "saved": len(batch),
                "total": total,
                "elapsed": round(elapsed, 2),
                "span_sec": round(span, 2),
                "actual_fps": round(len(batch) / span, 1) if span > 0 else 0,
                "mode": f"预触发缓冲(最近{DURATION_SEC}s)",
                "files": batch,
                "ts": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as e:
            # 存盘失败（磁盘满等）不能让等待方卡 12s，也不能杀线程
            self._error = f"连拍存盘失败: {e}"
            print(f"[相机] 连拍存盘失败: {e}")
            self._last_result = {
                "ok": False,
                "error": str(e),
                "saved": 0,
                "total": total,
                "mode": f"预触发缓冲(最近{DURATION_SEC}s)",
                "files": [],
                "ts": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        finally:
            self._capture_done.set()

    def _format_filename(self, prefix="Image"):
        ts = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M-%S-%f")[:-3]
        return f"{prefix}__{ts}{SAVE_EXT}"

    def _format_filename_ts(self, ts):
        """用帧的实际采集时间戳（来自缓冲）命名，保证顺序正确。"""
        dt = datetime.datetime.fromtimestamp(ts)
        s = dt.strftime("%Y-%m-%d__%H-%M-%S-%f")[:-3]
        return f"Image__{s}{SAVE_EXT}"

    def request_capture(self):
        """外部触发连拍，阻塞直到完成，返回结果 dict。"""
        if not self.running:
            return {"ok": False, "error": "相机未运行"}
        self._capture_done.clear()
        self._capture_req.set()
        # 等待完成（最多 DURATION_SEC + 5s 余量）
        timeout = DURATION_SEC + 5.0
        if self._capture_done.wait(timeout=timeout):
            return self._last_result or {"ok": False, "error": "连拍无结果"}
        return {"ok": False, "error": "连拍超时"}

    def get_latest_jpeg(self):
        """返回最新帧的 JPEG 字节，或 None。"""
        with self._lock:
            frame = self._latest
        if frame is None:
            return None
        # 缩放
        scale = STREAM_WIDTH / self.width if self.width else 1.0
        disp = cv2.resize(frame, (STREAM_WIDTH, int(self.height * scale)))
        if not self.is_color:
            disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
        ok, buf = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        return buf.tobytes() if ok else None

    def status(self):
        return {
            "running": self.running,
            "camera": self.camera_info,
            "resolution": f"{self.width}x{self.height}",
            "color": "彩色" if self.is_color else "黑白",
            "pixel_format": self.actual_fmt,
            "config": getattr(self, "_conf_log", []),
            "params": {
                "fps": FPS, "duration_sec": DURATION_SEC,
                "total": int(FPS * DURATION_SEC),
                "exposure_us": EXPOSURE_TIME_US,
                "gain_db": GAIN_DB,
                "gain_display": "相机当前值" if GAIN_DB is None else f"{GAIN_DB} dB",
                "gamma": GAMMA,
            },
            "save_dir": INSPECTION_ROOT,
            "last_result": self._last_result,
            "error": self._error,       # 取流/存盘异常（兜底后仍记录，便于排查）
        }

    def stop(self):
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        try:
            self.cam.StopGrabbing()
            self.cam.Close()
        except Exception:
            pass


class CameraHub:
    """多相机封装（当前仅一台，列表化以便加第二台）。网页用主相机预览。"""

    def __init__(self, ips=None):
        self.ips = list(ips if ips is not None else CAMERA_IPS)
        self.streamers = []
        self.primary_ip = self.ips[0] if self.ips else None
        self.plc_auto = False          # 是否启用 PLC 自动触发
        self.last_auto = None          # 最近一次自动检测结果（供 UI 展示）
        self.recent_cars = []          # 最近若干台车（供网页"最近车辆"表展示）

    def connect_all(self):
        if not self.ips:
            raise RuntimeError("未配置相机 IP（CAMERA_IPS 为空）")
        self.streamers = [CameraStreamer(ip) for ip in self.ips]
        for s in self.streamers:
            s.connect()
        self.primary_ip = self.ips[0]

    def start_all(self):
        for s in self.streamers:
            s.start()

    def stop_all(self):
        for s in self.streamers:
            s.stop()

    @property
    def running(self):
        return any(s.running for s in self.streamers)

    def get_primary_jpeg(self):
        if not self.streamers:
            return None
        return self.streamers[0].get_latest_jpeg()

    def request_capture_primary(self):
        """手动按钮用：返回主相机连拍结果 dict（含 files/saved/...）。"""
        if not self.streamers:
            return {"ok": False, "error": "相机未运行"}
        return self.streamers[0].request_capture()

    def request_capture_all(self):
        """自动触发用：触发全部相机，返回主相机文件列表 + 各相机结果。"""
        by_camera = {}
        primary_files = []
        primary_result = None
        for ip, s in zip(self.ips, self.streamers):
            r = s.request_capture()
            by_camera[ip] = r
            if ip == self.primary_ip:
                primary_files = r.get("files", [])
                primary_result = r
        return {"primary_files": primary_files,
                "primary_result": primary_result,
                "by_camera": by_camera}

    def status(self):
        if not self.streamers:
            return {"running": False, "plc_auto": self.plc_auto,
                    "last_auto": self.last_auto, "recent_cars": self.recent_cars,
                    "cameras": []}
        base = self.streamers[0].status()
        base["cameras"] = [s.status() for s in self.streamers]
        base["primary_ip"] = self.primary_ip
        base["plc_auto"] = self.plc_auto
        base["last_auto"] = self.last_auto
        base["recent_cars"] = self.recent_cars
        return base


# ===================== PLC 自动触发回调（仅 --plc-auto）=====================
def handle_car_signal(ctx):
    """出车信号上升沿回调：记录全部车 → 仅(9X/8X 且 NO_Paint=0)拍照 → 落库+写追溯。

    流程（用户定义）：
      上升沿 → 先记录全部车的车型/NO_Paint/滑橇/PIN →
      若车型是 9X 或 8X 且 NO_Paint=0 → 触发相机拍照（预触发，立即）；
      否则不拍照（免检车 / 未接入车型）。
    所有车型用同一出车信号触发，避免拍照时机不一致导致照片错位。

    本迭代【不跑检测】：拍照车用占位结果记录，检测等照片确认后再接。
    全程只读 PLC；本函数不写任何 PLC 地址。
    ctx 为 plc_monitor.parse_context 的 dict，无锁存时为 None。
    """
    global hub
    if ctx is None:
        print("[自动] 上升沿但无锁存车号，跳过")
        return
    skid = ctx["skid"]
    model = ctx["model"]
    pin = ctx["pin"]
    no_paint = ctx["no_paint"]
    ts_dt = datetime.datetime.now()
    ts = ts_dt.strftime("%Y-%m-%d %H:%M:%S")

    # 车型路由：决定是否需要拍照（9X/8X 且 NO_Paint=0）
    from detection.router import route_algorithm
    key = route_algorithm(model)
    captured = (key in ("9X", "8X")) and (not no_paint)

    # 仅需要拍照的车才占相机（统一用出车信号触发）
    files = []
    if captured:
        cap = hub.request_capture_all()
        files = cap.get("primary_files", [])
        if not files:
            print(f"[自动] 拍照失败：{cap.get('primary_result', cap)}")
            captured = False   # 拍照失败则按未拍照记录

    # 本迭代不跑检测：拍照车用占位结果（后续接检测时替换此段）
    from common.interfaces import DetectionResult, Defect
    det = DetectionResult(
        car_model=model, ok=True,
        defects=[Defect(0, 0, 0, 0, "pending", 0.0,
                        meta={"reason": "拍照完成·检测待做（检测算法待接入）"})],
        confidence=0.0,
        message="拍照完成·检测待做",
    )

    # 落库（含 skid/pin/no_paint/captured）；StorageService 内部按 日期/PIN 分层、
    # 去重移动、轮转，并返回该车文件夹路径
    paths = [os.path.join(SAVE_DIR, f) for f in files]
    db_ok = False
    rec = None
    try:
        from storage.service import get_service
        rec = get_service().save(model, det, paths,
                                 skid=skid, pin=pin, no_paint=no_paint,
                                 captured=captured, event_time=ts_dt)
        db_ok = True
    except Exception as e:
        print(f"[自动] 落库失败: {e}")

    # 写追溯 sidecar：仅拍照车在 Inspection 建自包含文件夹(record.json)；
    # 非拍照车(免检/未接入)不留空文件夹——其追溯已在 SQLite，sidecar 兜底到
    # data/records/<时间戳>__skid<滑橇>.json（带时间戳+滑橇，不会互相覆盖）。
    write_sidecar(ctx, files, det, db_ok, key, captured,
                  folder=rec.folder if (rec and captured) else "")

    # 兜底清理调试目录（自动图已移入 data/inspection，这里只清手动调试残留）
    _prune_scratch()

    # 最近车辆表（网页展示，最多保留 10 台）
    hub.recent_cars.insert(0, {
        "ts": ts, "model": model, "skid": skid, "pin": pin,
        "no_paint": bool(no_paint), "captured": captured, "key": key,
    })
    hub.recent_cars = hub.recent_cars[:10]
    hub.last_auto = {"ts": ts, "skid": skid, "model": model,
                     "ok": True, "captured": captured}

    # 打印
    if captured:
        action = f"拍{len(files)}张"
    elif no_paint:
        action = "免检跳过（不拍照）"
    else:
        action = f"车型未接入({key})跳过"
    print(f"[自动] 车 滑橇={skid} 车型={model}({key}) NO_Paint={no_paint} "
          f"→ {action} 库={'已写' if db_ok else '失败'}")


def write_sidecar(ctx, files, det, db_ok, model_key, captured, folder=None):
    """为本次自动检测写追溯 JSON。

    folder 给定时写入该车自包含文件夹(record.json)，与照片同目录、开箱即得全貌；
    否则兜底写入 data/records（如未拍照的免检车，无照片文件夹）。
    """
    if folder:
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "record.json")
    else:
        os.makedirs(RECORD_DIR, exist_ok=True)
        ts_file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(RECORD_DIR, f"{ts_file}__skid{ctx['skid']}.json")
    rec = {
        "trigger_ts": datetime.datetime.now().isoformat(timespec="seconds"),
        "source": "plc_auto",
        "skid": ctx["skid"],
        "pin": ctx["pin"],
        "model": ctx["model"],
        "model_key": model_key,
        "no_paint": bool(ctx["no_paint"]),
        "captured": captured,
        "capture": {
            "saved": len(files),
            "files": files,
        },
        "detection": {
            "done": False,
            "message": det.message,
        },
        "db_recorded": db_ok,
    }
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[自动] 写 sidecar 失败 {path}: {e}")


# ===================== Flask 应用 =====================
app = Flask(__name__)
hub = CameraHub()

# 自动触发的重活（拍照+落库约 12s）派发到独立线程，PLC 监控线程立即返回继续轮询，
# 避免出车信号触发期间停止轮询导致漏检。max_workers=1 保证相机一次只拍一台、不重叠。
_capture_exec = _cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="auto-capture")


def _prune_scratch(max_age_days=7, max_files=2000):
    """清理调试用 raw_images：删除超过 N 天或总数超上限的最旧文件，防磁盘涨满。

    自动拍照的图在 StorageService.save 中已被移到 data/inspection（去重），
    故本目录只残留手动调试连拍的图，低频但需兜底清理。
    """
    try:
        files = [os.path.join(SAVE_DIR, f) for f in os.listdir(SAVE_DIR)]
        files = [f for f in files if os.path.isfile(f)]
        if len(files) <= max_files:
            cutoff = time.time() - max_age_days * 86400
            for f in files:
                if os.path.getmtime(f) < cutoff:
                    os.remove(f)
        else:
            files.sort(key=os.path.getmtime)
            for f in files[:-max_files]:
                os.remove(f)
    except Exception:
        pass


def _gen_mjpeg():
    """MJPEG 流生成器（推送节拍上限 STREAM_FPS，避免无谓的 JPEG 编码拖 CPU）。"""
    last = 0.0
    interval = 1.0 / STREAM_FPS
    while hub.running:
        jpg = hub.get_primary_jpeg()
        if jpg:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            # 限速推送：让出 CPU 给机器人站
            now = time.perf_counter()
            wait = interval - (now - last)
            if wait > 0:
                time.sleep(wait)
            last = time.perf_counter()
        else:
            time.sleep(0.05)


@app.route("/")
def index():
    st = hub.status()
    cam_lines = "".join(f"<li>{c}</li>" for c in st["config"])
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>车顶密封条相机 · 实时取景</title>
<style>
 body{{font-family:-apple-system,"Microsoft YaHei",sans-serif;margin:0;background:#111;color:#eee}}
 header{{padding:10px 16px;background:#1d1d1d;border-bottom:1px solid #333;display:flex;
        align-items:center;gap:12px;flex-wrap:wrap}}
 header h1{{font-size:16px;margin:0}}
 .badge{{font-size:12px;padding:2px 8px;border-radius:10px;background:#2d6cdf;color:#fff}}
 .wrap{{padding:16px;max-width:1100px;margin:0 auto}}
 .video{{background:#000;border:1px solid #333;border-radius:8px;overflow:hidden}}
 .video img{{width:100%;display:block}}
 .bar{{display:flex;gap:10px;align-items:center;margin:14px 0;flex-wrap:wrap}}
 button{{font-size:15px;padding:10px 20px;border:0;border-radius:6px;cursor:pointer;
        background:#2d8c4a;color:#fff;font-weight:bold}}
 button:disabled{{background:#555;cursor:not-allowed}}
 button.stop{{background:#b5482f}}
 #log{{background:#0c0c0c;border:1px solid #333;border-radius:6px;padding:10px;
      font-family:monospace;font-size:12px;white-space:pre-wrap;max-height:220px;overflow:auto}}
 .meta{{font-size:12px;color:#aaa;line-height:1.7}}
 .meta b{{color:#eee}}
 .ctab{{border-collapse:collapse;width:100%;font-size:12px;margin-top:4px}}
 .ctab th,.ctab td{{border:1px solid #333;padding:4px 6px;text-align:left}}
 .ctab th{{background:#222;color:#ccc}}
 .ctab td{{color:#ddd}}
 .ctab .yes{{color:#5ad15a;font-weight:bold}}
 .ctab .no{{color:#d18a5a}}
</style></head>
<body>
<header><h1>车顶密封条相机 · 实时取景</h1>
  <span class="badge" id="state">连接中…</span>
  <span class="badge" id="plcstate">PLC…</span></header>
<div class="wrap">
  <div class="video"><img id="feed" src="/video_feed"></div>
  <div class="bar">
    <button id="cap" onclick="capture()">📸 拍最近{DURATION_SEC}秒（{int(FPS*DURATION_SEC)} 张）</button>
    <span id="capState" class="meta"></span>
  </div>
  <div class="meta" id="meta"></div>
  <h3 style="font-size:14px;margin:18px 0 6px">最近车辆（PLC 触发）</h3>
  <table class="ctab"><thead><tr>
    <th>时间</th><th>车型</th><th>滑橇</th><th>PIN</th><th>NO_Paint</th><th>拍照</th><th>检测</th>
  </tr></thead><tbody id="carbody">
    <tr><td colspan="7" class="meta">手动模式</td></tr>
  </tbody></table>
  <h3 style="font-size:14px;margin:18px 0 6px">采集日志</h3>
  <div id="log">等待操作…</div>
</div>
<script>
const meta=document.getElementById('meta');
const log=document.getElementById('log');
const state=document.getElementById('state');
const plcstate=document.getElementById('plcstate');
const capBtn=document.getElementById('cap');
const capState=document.getElementById('capState');
const auto=document.getElementById('auto');
const carbody=document.getElementById('carbody');

function refreshStatus(){{
  fetch('/api/status').then(r=>r.json()).then(s=>{{
    if(s.running){{state.textContent='● 在线';state.style.background='#2d8c4a';}}
    else{{state.textContent='● 离线';state.style.background='#b5482f';}}
    if(s.plc_auto){{
      plcstate.textContent='● PLC自动';plcstate.style.background='#2d6cdf';
      let rows=(s.recent_cars||[]).map(c=>{{
        let ph=c.captured?'<span class="yes">是</span>':'<span class="no">否</span>';
        let dt=c.captured?'<span class="yes">是</span>':'<span class="no">否</span>';
        return `<tr><td>${{c.ts}}</td><td>${{c.model}}</td><td>${{c.skid}}</td>`
          +`<td>${{c.pin}}</td><td>${{c.no_paint?'<span class="no">是</span>':'<span class="yes">否</span>'}}</td>`
          +`<td>${{ph}}</td><td>${{dt}}</td></tr>`;
      }}).join('');
      carbody.innerHTML = rows || `<tr><td colspan="7" class="meta">等待出车信号…</td></tr>`;
    }}else{{
      plcstate.textContent='● PLC未启用';plcstate.style.background='#555';
      carbody.innerHTML='<tr><td colspan="7" class="meta">手动模式：PLC 自动触发未启用（加 --plc-auto 开启）</td></tr>';
    }}
    let h=`<b>相机</b>：${{s.camera.model}}（序列号 ${{s.camera.serial}}）<br>`;
    h+=`<b>分辨率</b>：${{s.resolution}} · ${{s.color}} · ${{s.pixel_format}}<br>`;
    h+=`<b>预触发</b>：点按钮即存最近 ${{s.params.duration_sec}} 秒 = <b>${{s.params.total}} 张</b>（运动无延迟）<br>`;
    h+=`<b>曝光</b>：${{s.params.exposure_us}} µs · <b>增益</b>：${{s.params.gain_display}} · Gamma：${{s.params.gamma}}<br>`;
    h+=`<b>照片存储</b>：${{s.save_dir}}（按 日期/PIN 分层，临时缓冲在 data/raw_images）`;
    meta.innerHTML=h;
  }}).catch(()=>{{state.textContent='● 连接失败';state.style.background='#b5482f';}});
}}

function capture(){{
  capBtn.disabled=true; capState.textContent='拍摄中…';
  fetch('/api/capture',{{method:'POST'}}).then(r=>r.json()).then(res=>{{
    if(res.ok){{
      let line=`[${{res.ts}}] 完成 拍 ${{res.saved}}/${{res.total}} 张 · 耗时 ${{res.elapsed}}s · ${{res.actual_fps}} fps\\n`+res.files.map(f=>'  '+f).join('\\n');
      log.textContent=line+'\\n\\n'+log.textContent;
    }}else{{
      log.textContent='[错误] '+(res.error||'未知错误')+'\\n'+log.textContent;
    }}
    capState.textContent='';
    capBtn.disabled=false;
  }}).catch(e=>{{capState.textContent='';capBtn.disabled=false;
    log.textContent='[网络错误] '+e+'\\n'+log.textContent;}});
}}

refreshStatus();
setInterval(refreshStatus, 5000);
</script>
</body></html>"""


@app.route("/video_feed")
def video_feed():
    if not hub.running:
        return Response("相机未运行", status=503)
    return Response(_gen_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    return jsonify(hub.status())


@app.route("/api/capture", methods=["POST"])
def api_capture():
    if not hub.running:
        return jsonify({"ok": False, "error": "相机未运行"})
    # 手动按钮：纯拍照（不检测、不入库），用于远程盯屏调试/补拍
    result = hub.request_capture_primary()
    _prune_scratch()   # 手动调试图也兜底清理，防磁盘涨满
    return jsonify(result)


def main():
    parser = argparse.ArgumentParser(description="现场相机网页版实时取景+远程触发(+PLC自动触发)")
    parser.add_argument("--host", default=WEB_HOST)
    parser.add_argument("--port", type=int, default=WEB_PORT)
    parser.add_argument("--plc-auto", action="store_true",
                        help="启用 PLC 自动触发（只读监测 DB130.DBX0.1 上升沿→记录全部车→拍照→记录）")
    parser.add_argument("--no-browser-note", action="store_true")
    args = parser.parse_args()

    if not HAS_PYPYLON or not HAS_FLASK:
        print("[错误] 缺少依赖，请在上位机执行：")
        print("  pip install pypylon opencv-python flask")
        sys.exit(1)

    hub.plc_auto = args.plc_auto

    print("=" * 55)
    print("[网页采集] 正在连接 Basler 相机 ...")
    try:
        hub.connect_all()
    except Exception as e:
        print(f"[错误] {e}")
        sys.exit(1)

    try:
        hub.start_all()
    except Exception as e:
        print(f"[错误] {e}")
        hub.stop_all()
        sys.exit(1)

    st = hub.status()
    print(f"[网页采集] 已连接：{st['camera']['model']}  IP={st['camera']['ip']}")
    for c in st["config"]:
        print(f"  - {c}")
    print(f"[网页采集] 分辨率={st['resolution']}  色彩={st['color']}  像素格式={st['pixel_format']}")
    print(f"[网页采集] 预触发缓冲：点按钮/出车信号即存最近 {st['params']['duration_sec']}秒 "
          f"= {st['params']['total']} 张（{st['params']['fps']}fps）")
    print(f"[网页采集] 照片存储：{INSPECTION_ROOT}（按 日期/PIN 分层；data/raw_images 仅临时缓冲）")

    plc = None
    if args.plc_auto:
        try:
            from plc.plc_monitor import PlcMonitor, has_snap7
            if not has_snap7():
                print("[警告] 未安装 python-snap7，自动触发不可用（pip install python-snap7）。"
                      "回退手动模式。")
            else:
                plc = PlcMonitor(ip=PLC_IP, rack=PLC_RACK, slot=PLC_SLOT,
                                 poll_ms=PLC_POLL_MS, ctx_ms=PLC_CTX_MS)
                plc.connect()
                # 重活派发到独立线程，PLC 线程立即返回继续轮询（避免阻塞漏检）
                plc.start(lambda ctx: _capture_exec.submit(handle_car_signal, ctx))
                print(f"[PLC] 已连接 {PLC_IP}（只读）· 自动触发已启用"
                      f"（记录全部车；仅 9X/8X 且 NO_Paint=0 拍照）")
        except Exception as e:
            print(f"[警告] PLC 自动触发启动失败：{e}（回退手动模式）")
            plc = None

    if not args.plc_auto:
        print("[模式] 手动模式：PLC 自动触发未启用（加 --plc-auto 开启）")
    print("=" * 55)
    print(f"[网页采集] 请用浏览器打开（上位机本机或同网段电脑）：")
    print(f"  http://localhost:{args.port}")
    print(f"  http://<上位机IP>:{args.port}   ← 远程访问用这个")
    print(f"[网页采集] 提示：上位机防火墙需放行端口 {args.port}")
    print(f"[网页采集] 按 Ctrl+C 停止服务")
    print("=" * 55)

    try:
        # use_reloader=False 避免后台线程被起两次
        app.run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        pass
    finally:
        hub.stop_all()
        if plc is not None:
            plc.disconnect()
            print("\n[PLC] 已断开（仅读，未写任何地址）")
        try:
            _capture_exec.shutdown(wait=False)
        except Exception:
            pass
        print("\n[网页采集] 已停止，相机已释放")


if __name__ == "__main__":
    main()
