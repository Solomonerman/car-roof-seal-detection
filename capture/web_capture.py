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
import concurrent.futures as _cf
import subprocess
import re

import cv2
import numpy as np
import hashlib
import collections

# 版本戳：每次修改后更新，方便现场确认是不是最新代码
VERSION = "2026-07-30-latch-trigger"  # 修复:自动触发改为"新车上下文出现即锁存触发"(不再等上升沿去读被覆盖的DB230);收到车信号写data/capture_exec.log追溯

def _find_git_repo():
    """Walk up from this file to locate the .git directory; return repo root or None."""
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.isdir(os.path.join(d, ".git")):
            return d
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    return None


def _git_push_info():
    """Return (short_commit_hash, commit_date) of the current repo, or (None, None).
    Ties the version to the exact PUSHED code. Prints a debug note if git is
    unavailable so failures are never silent."""
    repo = _find_git_repo()
    if repo is None:
        print("[web_capture.py] git: no .git found upward from script -> "
              "push id will use stamp/run fallback")
        return None, None
    candidates = ["git"]
    if os.name == "nt":
        for base in (r"C:\Program Files\Git\cmd", r"C:\Program Files (x86)\Git\cmd",
                     r"C:\Program Files\Git\bin", r"C:\Program Files (x86)\Git\bin"):
            candidates.append(os.path.join(base, "git.exe"))
    last_err = None
    for g in candidates:
        try:
            h = subprocess.run([g, "-C", repo, "rev-parse", "--short", "HEAD"],
                               capture_output=True, text=True, timeout=3)
            d = subprocess.run([g, "-C", repo, "log", "-1", "--format=%cd",
                                "--date=format:%Y-%m-%d-%H%M%S"],
                               capture_output=True, text=True, timeout=3)
            hs, ds = h.stdout.strip(), d.stdout.strip()
            if hs and ds:
                return hs, ds
        except Exception as e:
            last_err = e
    print(f"[web_capture.py] git: command unavailable ({last_err}) -> "
          "push id will use stamp/run fallback")
    return None, None


def _load_stamp():
    """Read the push id stamped at push time (capture/_push_info.py).
    Reliable fallback when git is not usable at runtime (e.g. Git-Bash-only
    install, or a deployed copy without .git)."""
    try:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_push_info.py")
        if os.path.exists(p):
            txt = open(p, encoding="utf-8").read()
            mh = re.search(r'PUSH_HASH\s*=\s*"([0-9a-fA-F]+)"', txt)
            md = re.search(r'PUSH_DATE\s*=\s*"([\d\-:]+)"', txt)
            if mh and md:
                return mh.group(1), md.group(1)
    except Exception:
        pass
    return None, None


_GIT_HASH, _GIT_DATE = _git_push_info()
_STAMP_HASH, _STAMP_DATE = _load_stamp()
RUN_TAG = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
# VERSION_TAG = PUSH identity "<hash>__<commit-date>".
# Priority: live git (if available) > pushed stamp file > run time.
if _GIT_HASH:
    VERSION_TAG = _GIT_HASH + "__" + _GIT_DATE
    _VER_SRC = "git"
elif _STAMP_HASH:
    VERSION_TAG = _STAMP_HASH + "__" + _STAMP_DATE
    _VER_SRC = "stamp"
else:
    VERSION_TAG = RUN_TAG
    _VER_SRC = "run"
print(f"[web_capture.py] VERSION={VERSION}  PUSH={VERSION_TAG}  (source={_VER_SRC}, run={RUN_TAG})")

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

# ===================== 收到车信号追溯日志（持久化到磁盘）=====================
# 用途：PLC 上下文(DB230 车型/滑橇/PIN)在相机 7 秒连拍期间会被下一台车覆盖，
# 本日志在【锁存收到车的瞬间】落盘，记录每台被程序收到的车(车型/滑橇/PIN/免喷/路由决策)，
# 即便之后被覆盖，这里也有完整追溯——直接回答"MM** 信号到底收到没有、被谁覆盖"。
CAPTURE_EXEC_LOG = os.path.join(ROOT, "data", "capture_exec.log")
_exec_log_lock = threading.Lock()

def log_capture_event(**fields):
    """追加写入一条收到车/拍照事件到持久日志（append 模式，线程安全，每行一个 JSON）。

    失败不影响主流程（仅告警），保证"收到车信号"的追溯绝不因日志异常而丢失。
    """
    try:
        evt = {"ts": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]}
        evt.update(fields)
        line = json.dumps(evt, ensure_ascii=False)
        with _exec_log_lock:
            with open(CAPTURE_EXEC_LOG, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
        return line
    except Exception as e:
        print(f"[执行日志] 写盘失败(可忽略): {e}")
        return None

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




def _pick_closest(ring, target_ts):
    """从按时间递增的 ring 列表 [(ts, frame), ...] 里找 ts 最接近 target_ts 的项。
    ring 长度 ≤ 512，线性扫描足够快(O(500) < 0.1ms)。返回 (ts, frame) 或 None。
    """
    if not ring:
        return None
    best = None
    best_diff = float('inf')
    for item in ring:
        diff = item[0] - target_ts
        abs_diff = diff if diff >= 0 else -diff
        if abs_diff < best_diff:
            best_diff = abs_diff
            best = item
        # ring 时间递增，且过 target_ts 后 abs_diff 必然单调不减(因为步长固定)；
        # 为稳妥仍扫描完，500 次循环不影响性能。
    return best



def _frame_hash(frame):
    """逐帧内容指纹(md5 前12位)，用于诊断 21 张是否完全相同。"""
    try:
        return hashlib.md5(np.ascontiguousarray(frame).tobytes()).hexdigest()[:12]
    except Exception:
        return "na"


def _grab_to_frame(grab):
    """pylon grab 结果 → 完全独立的 numpy 数组(真·深拷贝)。

    pypylon 的 grab.Array 返回相机内部缓冲区的【视图】，不同 grab 之间该缓冲区
    可能被驱动复用。必须在 grab.Release() 之前做一次深拷贝，否则环形缓冲里 21 帧
    会指向同一块被反复刷新的内存 → 全部相同(现场多次复现)。
    np.array(..., copy=True) 强制独立拷贝并锁定 dtype=uint8，对任意 pypylon 实现都安全。
    """
    return np.array(grab.Array, copy=True, dtype=np.uint8)


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

        self._capture_running = threading.Event()  # capture thread running flag
        # Pre-trigger ring buffer: (perf_counter_ts, numpy_array). Capacity covers DURATION_SEC at camera fps; ensures 连拍切片覆盖触发前 DURATION_SEC.
        self._ring = collections.deque(maxlen=512)
        self._ring_lock = threading.Lock()
        self._last_cam_ts = None        # last camera hw-ts (plan A dedup on new exposures)
        self._capture_done = threading.Event()  # 连拍完成信号
        self._last_result = None                # 最近一次连拍结果

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
        # 触发/采集模式：强制自由运行(Continuous + TriggerMode=Off)。
        # 若相机残留"外触发/软触发"配置而无触发源，会只出 1 帧后冻结 → ring 全是同一张
        # （21 张全同的又一根源）。显式关触发 + 连续采集，保证 48fps 持续出图。
        try:
            tm = nodemap.GetNode("TriggerMode")
            if tm is not None:
                tm.SetValue("Off")
                conf_log.append("触发模式=Off(自由运行)")
        except Exception as e:
            conf_log.append(f"触发模式设置异常: {e}")
        try:
            am = nodemap.GetNode("AcquisitionMode")
            if am is not None:
                am.SetValue("Continuous")
                conf_log.append("采集模式=Continuous")
        except Exception as e:
            conf_log.append(f"采集模式设置异常: {e}")
        # 采集帧率：aca1920-48gm 上单数 'AcquisitionFrameRate' 是占位节点(not available)，
        # 不控制实际输出，碰它无效还可能误导。曾经的"大坑"：在【此处】打开
        # AcquisitionFrameRateEnable 却没同时写入有效的 AcquisitionFrameRateAbs →
        # 相机冻结(21 张全同/0 张)。故本处【完全不碰帧率限制开关】，所有 Enable+Abs 的
        # 设定统一交给 _camera_rate_check，并保证"先写 Abs 安全值、再开 Enable"成对出现。
        # 时序确定性由 _do_capture 的软件 3fps 节流保证，与相机端锁不锁无关。
        conf_log.append(f"采集帧率=帧率限制交由 _camera_rate_check 安全封顶(软件节流 {FPS}fps)")
        # 像素格式放最后
        try:
            nodemap.GetNode("PixelFormat").SetValue(PIXEL_FORMAT)
            conf_log.append(f"像素格式={PIXEL_FORMAT}")
        except Exception as e:
            conf_log.append(f"像素格式设置异常(沿用当前值): {e}")
        self._camera_rate_check(nodemap, conf_log)
        self._conf_log = conf_log

    def _camera_rate_check(self, nodemap, conf_log):
        """Read rate-limiting camera params and CAP the camera at a network-safe
        production rate >= our 3fps target. Called at connect time.

        关键教训（现场复现）：本机 aca1920-48gm 上单数 'AcquisitionFrameRate' 是占位
        节点（not available），但 'AcquisitionFrameRateEnable' + 'AcquisitionFrameRateAbs'
        可用。最初版本把 Enable 关掉 -> 相机解锁成原生 48fps 自由运行，现场 1Gbps 链路
        在 48fps 的极高包速率下取图超时（RetrieveResult 失败）。所以这里**绝不解锁到
        48fps**，而是用 Enable+Abs 把相机产出封顶到一个网络可承受的安全值（SAFE_FPS），
        软件仍按 3fps 节流取图。这样既能 >=3fps，又不压垮网卡。

        诊断行**立即 print**，即使后续 RetrieveResult 失败也能看到相机真实参数。"""
        try:
            def _gv(name, attr="Value"):
                n = nodemap.GetNode(name)
                if n is None:
                    return None
                try:
                    if attr == "Max":
                        return n.GetMax()
                    if attr == "Min":
                        return n.GetMin()
                    return n.GetValue()
                except Exception:
                    return None
            fr_en = _gv("AcquisitionFrameRateEnable")
            fr_abs = _gv("AcquisitionFrameRateAbs")
            fr_max = _gv("AcquisitionFrameRateAbs", "Max")
            exp_us = _gv("ExposureTimeAbs")
            trig = _gv("TriggerMode")
            pkt = _gv("GevSCPSPacketSize")
            thr = _gv("DeviceLinkThroughputLimit")
            line = (f"cam-diag: TriggerMode={trig} FrameRateEnable={fr_en} "
                    f"FrameRateAbs={fr_abs}fps(max={fr_max}) Exposure={exp_us}us "
                    f"PacketSize={pkt} ThroughputLimit={thr}")
            conf_log.append(line)
            print("[cam-diag]", line)  # 立即打印：即使取图失败也要能看到真实参数

            # 安全封顶（【唯一权威路径】）：先写【已知安全值】AcquisitionFrameRateAbs=SAFE_FPS，
            # 再开 AcquisitionFrameRateEnable=True。二者必须成对出现——曾因"单开 Enable 而
            # Abs 未写有效值"导致相机冻结(21 张全同/0 张)。此处保证：只要 Enable=True，Abs
            # 一定是安全的 6fps（>=软件 3fps 节流，且远低于 48fps 压垮 1Gbps 链路的阈值）。
            # 注：GigE 包大小不再自动改成 8192——现场未确认开启巨帧(Jumbo Frame)，保持默认。
            SAFE_FPS = 6.0
            n_abs = nodemap.GetNode("AcquisitionFrameRateAbs")
            n_en = nodemap.GetNode("AcquisitionFrameRateEnable")
            if n_abs is None or n_en is None:
                # 占位相机：帧率节点不可用，保持相机默认模式（不强行改，软件节流兜底）
                conf_log.append("cam-diag: 帧率节点不可用(占位)，保持相机默认模式")
                print("[cam-diag] 帧率节点不可用(占位)，保持相机默认模式")
            else:
                try:
                    n_abs.SetValue(SAFE_FPS)   # ① 先写安全值
                    n_en.SetValue(True)        # ② 再开限制——成对，绝不留"Enable 无有效 Abs"
                    conf_log.append(f"cam-diag FIX: 帧率限制 Abs={SAFE_FPS}fps Enable=True(成对写入)")
                    print(f"[cam-diag] FIX: 帧率限制 Abs={SAFE_FPS}fps Enable=True")
                except Exception as e:
                    conf_log.append(f"cam-diag: 封顶失败(已忽略，软件节流兜底): {e}")
                    print("[cam-diag] 封顶失败(已忽略):", e)
        except Exception as e:
            conf_log.append(f"cam-diag error (ignored): {e}")
            print("[cam-diag] error (ignored):", e)
        # 落盘启动诊断：即使上面异常也尽量持久化已有信息，便于远程上位机(无终端)排查。
        try:
            self._persist_startup_log(conf_log)
        except Exception:
            pass

    def _persist_startup_log(self, conf_log):
        """把相机启动诊断追加落盘到 data/records/camera_startup.log。

        用途：现场上位机常无终端/SSH，且取图失败往往只在启动瞬间暴露真实参数。
        一旦"连拍照都拍不了了"，用户无需连终端，直接打开该文件即可看到
        TriggerMode / FrameRateEnable / FrameRateAbs / 是否应用 FIX，定位冻结根因。"""
        try:
            os.makedirs(RECORD_DIR, exist_ok=True)
            path = os.path.join(RECORD_DIR, "camera_startup.log")
            ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(path, "a", encoding="utf-8") as f:
                f.write(f"===== 相机启动诊断 {ts} (version p{VERSION_TAG}) =====\n")
                for ln in conf_log:
                    f.write(ln + "\n")
                f.write("\n")
                f.flush()
        except Exception as e:
            print(f"[cam-diag] 落盘失败(可忽略): {e}")

    def start(self):
        # 抓取策略：用 LatestImageOnly（与现场验证可用的 live_capture.py 一致）。
        # 关键：后台取流线程轮询仅 GRAB_FPS=6，远慢于相机产出 48fps。
        #   OneByOne 会把帧按顺序塞进有限缓冲池，慢轮询下缓冲池耗尽 → 相机触发
        #   背压、停采 → RetrieveResult 反复返回同一冻结帧 → 预触发 21 帧全相同
        #   （现场复现：同一张照片）。LatestImageOnly 始终返回最新帧、自动丢弃旧帧，
        #   慢轮询也能拿到各不相同的新鲜帧；且轮询(6fps)慢于产出(48fps)，不会重复。
        #   时序确定性仍由 _loop 的 3fps 软件节流保证，不受策略影响。
        self.cam.StartGrabbing(py.GrabStrategy_LatestImageOnly)
        # Expand internal frame buffer so 7s pre-trigger ring never drops frames.
        try:
            self.cam.MaxNumBuffer.SetValue(1000)
        except Exception:
            pass
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
        # 出图率自测：连取若干张，统计"唯一帧"与实测 fps。
        # 这是"21张全同"的终极诊断：若唯一帧<取到数，说明相机没在自由运行(外触发/冻结)。
        try:
            n_probe = 10
            probe_hashes = []
            t0p = time.perf_counter()
            for _ in range(n_probe):
                g = self.cam.RetrieveResult(600, py.TimeoutHandling_Return)
                if g and g.GrabSucceeded():
                    probe_hashes.append(_frame_hash(_grab_to_frame(g)))
                elif g:
                    g.Release()
            dtp = time.perf_counter() - t0p
            uniqp = len(set(probe_hashes))
            fpsp = len(probe_hashes) / dtp if dtp > 0 else 0
            print(f"[相机] 出图率自测: 取到{len(probe_hashes)}/{n_probe}张, "
                  f"≈{fpsp:.1f}fps, 唯一帧={uniqp}/{len(probe_hashes)}")
            if uniqp <= 1:
                print("[相机] ⚠️ 出图率自测告警: 连续取的帧内容相同 → "
                      "相机可能未自由运行(检查 TriggerMode/外触发线/曝光是否过大)")
        except Exception as e:
            print(f"[相机] 出图率自测异常(忽略): {e}")

        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        """后台取流线程：持续 RetrieveResult，每帧写入环形缓冲(带 perf_counter 时间戳)，
        同时刷新 _latest 供网页预览。出车信号 → 启动独立 capture 线程做时间戳切片。

        取流节拍匹配相机产出(48fps ≈ 21ms)，保证 DURATION_SEC 内 ring 不丢帧。
        LatestImageOnly + 大 MaxNumBuffer：保留最近帧，Python 端不取也不丢关键窗口。
        """
        GET_INTERVAL = 1.0 / 48  # ~21ms，匹配相机 48fps 产出节奏
        next_get = time.perf_counter()
        while self.running:
            try:
                wait = next_get - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                next_get = time.perf_counter() + GET_INTERVAL
                # 连拍进行中：由 _do_capture 独占相机取流，_loop 短暂让位
                if self._capture_running.is_set():
                    time.sleep(0.02)
                    continue
                try:
                    grab = self.cam.RetrieveResult(1000, py.TimeoutHandling_Return)
                except Exception:
                    time.sleep(0.05)
                    continue
                if grab and grab.GrabSucceeded():
                    frame = _grab_to_frame(grab)   # 深拷贝，杜绝 21 张全同
                    ts = time.perf_counter()
                    # 先写入 ring(给 capture 切片用)，再更新 _latest(给预览用)
                    # read camera hardware timestamp (plan A: filter dup copies produced
                    # when software reads faster than the camera outputs new frames)
                    try:
                        cam_ts = grab.GetTimeStamp()
                    except Exception:
                        cam_ts = None
                    # only let genuinely new exposure frames enter ring/preview
                    if self._last_cam_ts is None or cam_ts != self._last_cam_ts:
                        self._last_cam_ts = cam_ts
                        ts = time.perf_counter()
                        with self._ring_lock:
                            self._ring.append((ts, frame))
                        with self._lock:
                            self._latest = frame
                    grab.Release()
                    # 出车信号：启动独立 capture 线程(不阻塞 _loop)
                    if self._capture_req.is_set() and not self._capture_running.is_set():
                        self._capture_running.set()
                        threading.Thread(target=self._do_capture, daemon=True).start()
                elif grab:
                    grab.Release()
            except Exception as e:
                self._error = f"取流异常: {e}"
                print(f"[相机] 取流异常(已忽略，继续): {e}")
                time.sleep(0.1)


    def _do_capture(self):
        """直接同步连拍(治本版)：触发时立即从相机连抓 21 张、每张间隔 333ms。

        放弃环形缓冲方案——那条路在现场反复复现"21 张全同"，根因不在 Python 而在
        GrabStrategy_LatestImageOnly + 后台 _loop + ring 这一组合对真实环境的脆弱性。
        改为触发瞬间 _do_capture 独占相机同步抓 21 张：若相机在自由运行，必得 21 张
        不同帧；若相机冻结/未出图，抓取失败被直接计入诊断，日志会明确写出。
        21 帧覆盖触发时刻起共 7 秒，每张用真实拍摄时刻命名，文件名带版本号短戳。
        """
        self._capture_req.clear()
        total = int(FPS * DURATION_SEC)      # 21
        frame_interval = 1.0 / FPS           # 0.333...s
        identical = False

        t_trigger = time.perf_counter()
        t_trigger_dt = datetime.datetime.now()
        t0 = t_trigger

        os.makedirs(SAVE_DIR, exist_ok=True)
        print(f"[拍照] 手动连拍目标目录(绝对路径): {os.path.abspath(SAVE_DIR)}")
        batch = []
        saved_hashes = []
        picked_meta = []  # (idx, target_ts, actual_ts, delta_ms, frame)
        # 单张抓取超时：333ms 节拍 + 1500ms 余量（防 OS sleep 抖动/相机偶发延迟）
        timeout_ms = int(frame_interval * 1000) + 1500

        # 直接同步：第 i 张的目标时刻 = t_trigger + i*interval，第 0 张即触发瞬间
        next_at = t_trigger
        for i in range(total):
            wait = next_at - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            actual_ts = time.perf_counter()
            try:
                grab = self.cam.RetrieveResult(timeout_ms, py.TimeoutHandling_Return)
            except Exception as e:
                print(f"[相机] 第{i+1}/{total}张取图异常: {e}")
                next_at += frame_interval
                continue
            if not grab or not grab.GrabSucceeded():
                if grab:
                    grab.Release()
                print(f"[相机] 第{i+1}/{total}张取图失败(超时/相机未出图)")
                next_at += frame_interval
                continue
            frame = _grab_to_frame(grab)   # 独立深拷贝，杜绝 21 张全同
            # camera hardware timestamp: iron proof of whether this is a new exposure.
            # MUST be read before Release().
            try:
                cam_ts = grab.GetTimeStamp()
            except Exception:
                cam_ts = None
            grab.Release()

            target_ts = t_trigger + i * frame_interval
            delta_ms = (actual_ts - target_ts) * 1000.0
            picked_meta.append((i + 1, target_ts, actual_ts, delta_ms, frame, cam_ts))

            # 边抓边存：每抓一张立即写盘，避免 21 张攒一起写超时丢帧
            frame_dt = t_trigger_dt + datetime.timedelta(seconds=(actual_ts - t_trigger))
            fname = self._format_filename(idx=i + 1, dt=frame_dt)
            fpath = os.path.join(SAVE_DIR, fname)
            cv2.imwrite(fpath, frame)
            batch.append(fname)
            saved_hashes.append(_frame_hash(frame))

            next_at += frame_interval

        elapsed = time.perf_counter() - t0

        # ========== 诊断日志 ==========
        unique_hashes = len(set(saved_hashes))
        identical = (len(saved_hashes) > 1 and unique_hashes <= 1)
        # camera hardware timestamp diagnostic: iron proof that 21 frames are new exposures
        cam_ts_list = [m[5] for m in picked_meta]
        distinct_cam_ts = len(set(cam_ts_list))
        print(f"[相机] ===== 直接连拍统计 VERSION={VERSION} 启动戳=p{VERSION_TAG} =====")
        print(f"[相机] 目标={total}张 实际={len(batch)}张 唯一帧={unique_hashes}/{len(saved_hashes)} "
              f"总耗时={elapsed:.2f}s")
        if identical:
            print(f"[相机] 逐帧去重: ⚠️ 全部相同!  → 相机未持续出图 或 场景完全静止(无车经过)")
            print(f"[相机] ❌ 请查看启动日志 [相机] 出图率自测 的唯一帧/fps 判断相机端状态。")
        elif len(saved_hashes) == 0:
            print(f"[相机] 逐帧去重: ❌ 0张取到 → 相机无响应/未出图，检查连接与 TriggerMode=Off")
        elif unique_hashes == len(saved_hashes):
            print(f"[相机] 逐帧去重: ✓ 各不相同")
        else:
            print(f"[相机] 逐帧去重: 部分重复")
        # ---- camera hardware timestamp verdict (the key iron proof) ----
        if None not in cam_ts_list and distinct_cam_ts <= 1:
            print(f'[cam] HARDWARE TS ALL EQUAL ({distinct_cam_ts}) -> camera produced NO new frames (frozen / no trigger / TriggerMode not Off)')
        elif distinct_cam_ts == len(cam_ts_list):
            print(f'[cam] HARDWARE TS: OK {distinct_cam_ts}/{len(cam_ts_list)} all distinct -> 21 frames are genuine new exposures')
        else:
            print(f'[cam] HARDWARE TS: partial repeats ({distinct_cam_ts}/{len(cam_ts_list)})')
        # per-frame hw-ts
        for idx, target_ts, actual_ts, delta_ms, _f, cam_ts in picked_meta:
            offset_s = target_ts - t_trigger
            print(f'[cam] frame {idx:02d}/{total} target_off={offset_s*1000:6.1f}ms delta={delta_ms:+6.1f}ms hw_ts={cam_ts}')

            offset_s = target_ts - t_trigger
            print(f"[相机] 第{idx:02d}/{total}张 目标偏移={offset_s*1000:6.1f}ms 实际偏差={delta_ms:+6.1f}ms")
        if len(picked_meta) >= 2:
            ivs = [(picked_meta[j][2] - picked_meta[j-1][2]) * 1000.0 for j in range(1, len(picked_meta))]
            avg = sum(ivs)/len(ivs); mn = min(ivs); mx = max(ivs)
            print(f"[相机] 间隔: 平均={avg:.1f}ms 最小={mn:.1f}ms 最大={mx:.1f}ms (目标={frame_interval*1000:.1f}ms)")
        print(f"[相机] ===== 统计结束 =====")

        span = (len(batch) - 1) * frame_interval if len(batch) > 1 else 0.0
        abs_files = [os.path.abspath(os.path.join(SAVE_DIR, f)) for f in batch]
        cap_ok = (len(batch) > 0)
        self._last_result = {
            "ok": cap_ok,
            "saved": len(batch),
            "unique_frames": unique_hashes,
            "distinct_cam_ts": distinct_cam_ts,
            "identical": identical,
            "total": total,
            "elapsed": round(elapsed, 2),
            "span_sec": round(span, 2),
            "actual_fps": round(len(batch) / elapsed, 1) if elapsed > 0 else 0,
            "mode": f"直接同步连拍({DURATION_SEC}s/{total}张,不依赖ring)",
            "files": batch,
            "abs_files": abs_files,
            "save_dir": os.path.abspath(SAVE_DIR),
            "ts": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        # 自证式落盘记录：即使 0 张也写，便于现场确认"是没抓到帧而非存错地方"
        try:
            manifest = {
                "ok": cap_ok,
                "saved": len(batch),
                "total": total,
                "unique_frames": unique_hashes,
                "distinct_cam_ts": distinct_cam_ts,
                "identical": identical,
                "save_dir": os.path.abspath(SAVE_DIR),
                "abs_files": abs_files,
                "ts": self._last_result["ts"],
            }
            with open(os.path.join(SAVE_DIR, "_last_manual_capture.json"), "w", encoding="utf-8") as mf:
                json.dump(manifest, mf, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[拍照] 写 manifest 失败(可忽略): {e}")
        if cap_ok:
            print(f"[拍照] 已落盘 {len(batch)} 张 -> {os.path.abspath(SAVE_DIR)}")
            print(f"[拍照] 首张: {abs_files[0]}")
            print(f"[拍照] 末张: {abs_files[-1]}")
        else:
            print(f"[拍照] ❌ 本次连拍 0 张有效帧，未写入任何文件(相机未出图/连接已断)。")
            print(f"[拍照]   排查: 看启动日志 [相机] 出图率自测 的 fps/唯一帧；确认网口未掉流。")
        self._capture_running.clear()
        self._capture_done.set()



    def _format_filename(self, idx=None, dt=None, prefix="Image"):
        if dt is None:
            dt = datetime.datetime.now()
        ts = dt.strftime("%Y-%m-%d__%H-%M-%S-%f")[:-3]
        idx_part = f"_{idx:02d}" if idx is not None else ""
        # 文件名带版本短号 p{VERSION_TAG}，现场一眼即可确认是否跑了最新程序
        return f"{prefix}{idx_part}__p{VERSION_TAG}__{ts}{SAVE_EXT}"

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
            "version": VERSION,
            "version_tag": VERSION_TAG,
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


def run_selftest():
    """自检模式：连相机 → 预热让环形缓冲铺满 → 触发一次 21 张 → 校验是否各不相同。

    无需 PLC / 网页服务，~10 秒即可确认"每秒 3 张、21 张不同"是否已达成。
    返回 0=通过, 1=失败。
    """
    if not HAS_PYPYLON:
        print("[自检] 未安装 pypylon，请先: pip install pypylon"); return 1
    print(f"[自检] VERSION={VERSION}")
    hub = CameraHub()
    try:
        hub.connect_all()
        hub.start_all()
    except Exception as e:
        print(f"[自检] 连接/启动失败: {e}"); return 1
    print(f"[自检] 相机已启动，预热 {DURATION_SEC + 1}s 让环形缓冲铺满...")
    time.sleep(DURATION_SEC + 1)
    print(f"[自检] 触发连拍 {int(FPS * DURATION_SEC)} 张...")
    r = hub.request_capture_primary()
    try:
        hub.stop_all()
    except Exception:
        pass
    uniq = r.get("unique_frames")
    saved = r.get("saved")
    ident = r.get("identical")
    print("=" * 44)
    print(f"[自检] 结果: 保存={saved}  唯一帧={uniq}  全部相同={ident}")
    print(f"[自检] 诊断: {r.get('mode')}")
    if ident:
        print("[自检] ❌ 失败: 21 张完全相同 → 相机未持续出图 或 旧代码未更新")
        return 1
    if uniq == saved == int(FPS * DURATION_SEC):
        print("[自检] ✓ 通过: 21 张各不相同，3fps 预触发生效")
        return 0
    print(f"[自检] ⚠️ 部分通过: 唯一帧 {uniq}/{saved}")
    return 0


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
    should_capture = (key in ("9X", "8X")) and (not no_paint)

    # 持久记录"收到车信号"——PLC 上下文在 7 秒连拍期间会被下一台车覆盖，
    # 此处在该车被锁存收到的瞬间落盘，确保 MM** 等信号有完整追溯(车型/滑橇/PIN/决策)。
    log_capture_event(
        type="car_received",
        source="plc_auto",
        skid=skid, model=model, model_key=key, pin=pin,
        no_paint=bool(no_paint),
        should_capture=bool(should_capture),
        decision=("触发拍照" if should_capture else
                  ("免检跳过(不拍照)" if no_paint else f"车型未接入({key})跳过")),
        summary=(f"收到车信号 滑橇={skid} 车型={model!r}({key}) "
                 f"NO_Paint={no_paint} → {'触发拍照' if should_capture else '不拍照'}"),
    )

    # 仅需要拍照的车才占相机（统一用出车信号触发）
    files = []
    captured = False
    capture_err = None
    if should_capture:
        cap = hub.request_capture_all()
        files = cap.get("primary_files", [])
        if files:
            captured = True
        else:
            capture_err = cap.get("primary_result", cap)
            print(f"[自动] 拍照失败（相机未出图/连接异常）: {capture_err}")

    # 当前所有车型仅拍照存档，不跑检测——等算法就绪后再接入。
    # 届时 9X(MM**)→SealDetector，8X(NM41/NM42)→对应检测器，替换此段即可。
    from common.interfaces import DetectionResult, Defect
    det = DetectionResult(
        car_model=model, ok=True,
        defects=[Defect(0, 0, 0, 0, "pending", 0.0,
                        meta={"reason": "拍照存档·算法未接入"})],
        confidence=0.0,
        message="拍照存档·算法未接入",
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

    # 打印（action 必须区分三种情况，避免把"相机0帧"误显示为"车型未接入"）
    if captured:
        action = f"拍{len(files)}张"
    elif capture_err is not None:
        action = "拍照失败（相机未出图/连接异常）"
    elif no_paint:
        action = "免检跳过（不拍照）"
    else:
        action = f"车型未接入({key})跳过（非9X/8X或PLC车型码未匹配route）"
    print(f"[自动] 车 滑橇={skid} 车型={model!r}({key}) NO_Paint={no_paint} "
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
    <th>时间</th><th>车型</th><th>滑橇</th><th>PIN</th><th>NO_Paint</th><th>拍照</th><th>检测(未接入)</th>
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
        let dt='<span class="no">未接入</span>';
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
    h+=`<b>程序版本</b>：v${{s.version_tag}}（照片文件名含此短号，可确认是否最新）<br>`;
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
      let line=`[${{res.ts}}] 完成 拍 ${{res.saved}}/${{res.total}} 张 · 耗时 ${{res.elapsed}}s · ${{res.actual_fps}} fps\\n`+
               `已存到：${{res.save_dir}}\\n`+
               (res.abs_files||res.files).slice(0,3).map(f=>'  '+f).join('\\n')+
               ((res.abs_files||res.files).length>3?`\\n  ... 共 ${{(res.abs_files||res.files).length}} 张`:'');
      log.textContent=line+'\\n\\n'+log.textContent;
    }}else{{
      log.textContent='[错误] '+(res.error||'未知错误')+'（未写入任何文件）\\n'+log.textContent;
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
    if result.get("ok"):
        print(f"[网页] 手动拍照完成: {result.get('saved')} 张 -> {result.get('save_dir')}")
        for p in result.get("abs_files", [])[:3]:
            print(f"[网页]   落盘: {p}")
        if len(result.get("abs_files", [])) > 3:
            print(f"[网页]   ... 共 {len(result.get('abs_files', []))} 张")
    else:
        print(f"[网页] 手动拍照失败: {result.get('error')} (saved={result.get('saved')})")
    return jsonify(result)


def main():
    parser = argparse.ArgumentParser(description="现场相机网页版实时取景+远程触发(+PLC自动触发)")
    parser.add_argument("--host", default=WEB_HOST)
    parser.add_argument("--port", type=int, default=WEB_PORT)
    parser.add_argument("--plc-auto", action="store_true",
                        help="启用 PLC 自动触发（只读监测 DB130.DBX0.1 上升沿→记录全部车→拍照→记录）")
    parser.add_argument("--no-browser-note", action="store_true")
    parser.add_argument("--selftest", action="store_true",
                        help="自检模式: 连相机拍21张并校验是否各不相同(无需PLC/网页)")
    args = parser.parse_args()

    if args.selftest:
        rc = run_selftest()
        sys.exit(rc)


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
    print(f"[网页采集] 程序版本 VERSION={VERSION}（照片文件名含唯一到秒标记 p{VERSION_TAG}，可确认是否最新程序/实例）")
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
