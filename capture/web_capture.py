# -*- coding: utf-8 -*-
"""现场相机 · 网页版实时取景 + 远程触发采集 + PLC 自动触发检测。

为什么用这个：
  你在工位远程操作上位机，看不到现场车子。pylon Viewer 的实时窗口在
  上位机本地，你隔着远程桌面才能看到，且它独占相机、会和本脚本冲突。
  这个工具把相机的实时画面通过浏览器（局域网）推给你，你在网页上
  就能看到车有没有到位，点一下按钮就触发连拍，无需远程桌面、无需 pylon Viewer。

自动触发（--plc-auto）：
  后台启一个【只读】PLC 监控线程：持续采样 DB230 并提前锁存车身上下文；
  监测 DB130.DBX0.1（出车信号）上升沿，上升沿时用此前锁存的 DB230 去
  记录（全部车）→ 仅对 (9X/8X 且 NO_Paint=0) 的车触发拍照 → 落库 + 写追溯。
  全程仅 read_area，绝不写 PLC。

用法：
  1) 上位机安装依赖： pip install pypylon opencv-python flask python-snap7
  2) 关闭 pylon Viewer（避免相机被占用）
  3) 手动模式： python capture/web_capture.py           （不带 --plc-auto）
     自动模式： python capture/web_capture.py --plc-auto
  4) 浏览器打开 http://<上位机IP>:5000 看实时画面。
     网页顶部 [手动测试 | 自动监控] 切换：
       · 手动测试——设曝光/每秒张数/总张数，点拍即拍，与 PLC 无关、不判车型、
         照片存 data/manual_test/日期/时间/（网页直接看缩略图），纯测相机。
         （增益不可调，固定沿用相机当前值 Gain Raw 136）
       · 自动监控——PLC 出车信号触发、车型筛选、存档（原逻辑）。

取流架构（2026-08-15 重写）：
  全程序只有 CameraStreamer 后台 _loop 一个线程调用 RetrieveResult。预览与拍照
  都来自这个单循环的产出，绝不在第二个线程里并发取流——这正是上一版在工控机
  约 10 秒冻屏的根因。拍照收到请求后进入 capturing 状态，_loop 按程序固定 3fps
  节奏收集"当时最新帧"写盘，共 7s×3fps=21 张；拍照节奏由软件控制、与相机实际
  帧率解耦，相机自由运行多少 fps 都不会把拍照锁死（历史曾锁到 2 张/秒）。

连拍 / 相机参数 / PLC 参数都集中在顶部，按现场情况调整。
"""
import os
import sys
import time
import json
import datetime
import threading
import argparse
import subprocess
import re
import shutil

import cv2
import numpy as np
import hashlib
import concurrent.futures as _cf

# 版本戳：每次修改后更新，方便现场确认是不是最新代码
VERSION = "2026-08-15-single-loop"  # 重写:单循环取流根治冻屏;保留双触发/PLC锁存/存储/网页


# ------------------------- 版本戳（现场确认是否最新程序）-------------------------
def _find_git_repo():
    """从本文件向上查找 .git 目录，返回仓库根或 None。"""
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
    """返回 (短提交号, 提交日期)；git 不可用时返回 (None, None)。"""
    repo = _find_git_repo()
    if repo is None:
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
    return None, None


def _load_stamp():
    """读取推送时写死的版本戳(capture/_push_info.py)，git 不可用时兜底。"""
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
FPS = 3                # 连拍帧率（张/秒），程序侧固定节奏，与相机帧率解耦
DURATION_SEC = 7       # 连拍持续时长（秒）
TOTAL = int(FPS * DURATION_SEC)   # 总张数 = 3×7 = 21 张

# 自动监控连拍参数（PLC 触发 / 强制拍摄）：与手动测试模式解耦，单独配置。
AUTO_FPS = 2.75                # 自动连拍帧率（张/秒）= 2秒5张
AUTO_DURATION_SEC = 5         # 自动连拍持续时长（秒）
AUTO_TOTAL = int(AUTO_FPS * AUTO_DURATION_SEC)   # 2.5×5≈12.5 → 取整 12 张

# 拍摄前延迟（秒）：收到拍摄指令后，先等目标物移动到相机视野合适位置再开始连拍。
# 自动(PLC触发)与手动(点击按钮)【分开定义、可在界面分别调整】，避免把"刚到位/刚点
# 按钮"那一刻拍进去。request_capture 不再用单一全局，而由调用方显式传入对应延时。
#   - 自动(AUTO)：默认 2.0s（出车信号+同步信号满足后，等车身到视野再拍）
#   - 手动(MANUAL)：默认 5.0s（手动测试 / 自检之外的手动拍摄）
AUTO_PRE_CAPTURE_DELAY = 2.0
MANUAL_PRE_CAPTURE_DELAY = 5.0

# 自愈阈值：取流线程超过该秒数未取到任何新帧（造型相机 GigE 心跳超时/网口抖动
# 会令固件静默停出图，无异常、GrabSucceeded 恒为 False），即重启取流，不退出进程。
SELFHEAL_SEC = 5.0

# ===================== 相机连接参数 =====================
# 现场相机 IP 列表（多相机就绪：左右相机共用同一出车信号，把 IP 追加进来即可，
# 其余代码无需改动；当前仅左相机一台）。第二台接入后，自动模式会同时触发全部相机。
CAMERA_IPS = ["172.30.173.249"]   # 现场左侧 Basler aca1920-48gm
CAMERA_SERIAL = ""             # 留空则用上面列表的 IP；也可填序列号直连（单相机场景）

# ===================== 相机采集参数 =====================
EXPOSURE_TIME_US = 3000      # 曝光时间（微秒），默认 3000（2000 偏暗，现场已上调）
# 增益：设为 None = 沿用相机【当前值】、不修改。
#   你在 pylon Viewer 里已设过增益（现场暗光、曝光锁 2000µs 的可用档），
#   关掉 pylon 跑本程序时会保留该设置，拍出的图不会变暗。
#   若想强制指定固定值，填 dB 数字，例如 6.0。
#   ⚠️ 注意：相机断电/复位后会恢复默认（通常 0 dB），届时需要重设或在此填固定值。
GAIN_DB = None
GAMMA = 1.0                  # Gamma，默认 1.0（保持线性，利于检测）
PIXEL_FORMAT = "Mono8"       # 黑白相机 8bit 灰度

# ===================== 存储参数 =====================
SAVE_DIR = os.path.join(ROOT, "data", "raw_images")      # 临时缓冲(拍完即被移走)
INSPECTION_ROOT = os.path.join(ROOT, "data", "inspection")  # 最终存储根(按 日期/PIN 分层)
SAVE_EXT = ".bmp"            # BMP 无损，适合算法检测；也可改 ".jpg"
RECORD_DIR = os.path.join(ROOT, "data", "records")   # 自动触发追溯 sidecar JSON(非拍照车兜底)
# 手动测试模式照片存储根：与自动存档(data/inspection)完全隔离，不入库、不参与车型筛选。
# 每次点击拍摄建一个 日期/时间 子目录，网页缩略图直接读这里。
MANUAL_TEST_ROOT = os.path.join(ROOT, "data", "manual_test")

# ===================== 收到车信号追溯日志（持久化到磁盘）=====================
# 用途：PLC 上下文(DB230 车型/滑橇/PIN)在相机 7 秒连拍期间会被下一台车覆盖，
# 本日志在【收到出车信号的瞬间】落盘，记录每台被程序收到的车(车型/滑橇/PIN/免喷/路由决策)，
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
# 相机自由运行（Mono8 1920x1200 通常几十 fps），MJPEG 推送限速 STREAM_FPS 避免无谓编码拖 CPU。
STREAM_FPS = 12
# 预览帧"卡住"判定阈值(秒)：_latest 超过该时间未刷新，视为相机/取流异常 → 画面叠加告警。
STALE_SEC = 2.0

# ===================== PLC 自动触发参数（--plc-auto 启用）=====================
PLC_IP = "172.30.173.6"
PLC_RACK = 0
PLC_SLOT = 2
PLC_POLL_MS = 20             # 出车信号轮询间隔（毫秒）
PLC_CTX_MS = 200             # DB230 上下文采样间隔（毫秒）——提前锁存

# 缺陷标签集合（出现 → 整体 NG）；与 detection.detector.NG_LABELS 保持一致
NG_LABELS = {"missing", "break", "overspray", "width"}

# ================================================================


def _frame_hash(frame):
    """逐帧内容指纹(md5 前12位)，用于诊断 21 张是否完全相同。"""
    try:
        return hashlib.md5(np.ascontiguousarray(frame).tobytes()).hexdigest()[:12]
    except Exception:
        return "na"


def _grab_to_frame(grab):
    """pylon grab 结果 → 完全独立的 numpy 数组(真·深拷贝)。

    不用 grab.Array：本机 pypylon 版本下 GetArray() 对 Mono8 也会抛
    'Pixel format currently not supported'，且在部分流式帧上甚至返回空数组
    （导致预览 cv2.resize 报 !ssize.empty）。改为用 GetBuffer() 取原始字节，
    【按字节总数自动判断像素布局】(w*h / w*h*2 / w*h*3)，完全不依赖像素类型
    常量，对任意 8bit 单通道(Mono/Bayer)、16bit 单通道、RGB 都安全。

    仍保持真·深拷贝：GetBuffer 返回的 bytes 独立于相机内部缓冲区，np.array
    copy=True 再锁一份，确保连续帧不指向同一块被复用内存（否则 21 张全相同）。
    """
    w = grab.GetWidth()
    h = grab.GetHeight()
    # 防御：极少数 grab 返回 0 尺寸/空缓冲（GigE 抖动、缓冲回收）→ 直接判无效，
    # 绝不能造零尺寸数组（cv2.imwrite 会 !_img.empty() 崩溃，批量写盘全失败）。
    if w <= 0 or h <= 0:
        return None
    buf = grab.GetBuffer()
    n = len(buf)
    if n <= 0:
        return None
    try:
        if n == w * h * 3:
            arr = np.frombuffer(buf, np.uint8).reshape(h, w, 3)
        elif n == w * h * 2:
            arr = np.frombuffer(buf, np.uint16).reshape(h, w)
        elif n == w * h:
            arr = np.frombuffer(buf, np.uint8).reshape(h, w)
        else:
            # 尺寸对不上（极少见）→ 判无效，交由上层回退上一帧，不造黑/空图
            return None
    except Exception:
        return None
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)   # 16bit 路径转 uint8（截断高位；单色相机走不到此分支）
    return np.array(arr, copy=True)


class CameraStreamer:
    """单循环取流相机封装（单台相机）。

    架构（根治冻屏）：全程序只有本类的后台 _loop 一个线程调用 RetrieveResult。
    预览与拍照都从同一个循环取到的帧获得，绝不在第二个线程里并发取流
    （原架构"后台线程 + ring + 拍照线程并发取流"是工控机约 10 秒冻屏的根因）。

    拍照：收到 request_capture 后进入 capturing 状态，_loop 按程序固定 3fps 节奏
    收集"当时最新帧"写盘，共 21 张。拍照节奏由软件控制，与相机实际帧率解耦，
    相机自由运行多少 fps 都不会把拍照锁死（历史曾锁到 2 张/秒）。
    """

    def __init__(self, camera_ip=None):
        self.camera_ip = camera_ip
        self.cam = None
        self.running = False
        self.is_color = False
        self.width = 0
        self.height = 0
        self.actual_fmt = "未知"
        self.camera_info = {}
        # 相机参数（运行时可由网页"手动测试"面板调整，初始取自顶部常量；connect 后读回实际值）
        self._exposure_us = EXPOSURE_TIME_US
        self._exposure_user = EXPOSURE_TIME_US   # 用户"期望"曝光值（仅 set_exposure 更新；
        #   _open_camera 重连会重置相机为默认并改写 _exposure_us，但不会动本值，
        #   故自愈重连后用本值把用户设定重新下发，避免手动曝光被静默清零）
        self._exposure_node = "ExposureTime"   # 启动时确认的实际生效节点名(ExposureTime/ExposureTimeAbs)
        self._gain_db = GAIN_DB

        self._latest = None
        self._latest_ts = 0.0        # _latest 最后刷新时刻(perf_counter)
        self._frame_no = 0           # 预览帧累计计数(常驻心跳)
        self._lock = threading.Lock()

        # 拍照状态机
        self._cap_state = "idle"     # idle / capturing
        self._cap_running = False
        self._cap_idx = 0
        self._cap_start = 0.0
        self._cap_next = 0.0
        self._cap_frames = []
        self._cap_hashes = []
        self._cap_t0_dt = None
        self._capture_done = threading.Event()
        self._last_result = None

        self._thread = None
        self._error = None

    # ---------- 连接与配置 ----------
    def connect(self):
        if not HAS_PYPYLON:
            raise RuntimeError("未安装 pypylon，请先在上位机执行：pip install pypylon")
        self._open_camera()

    def _open_camera(self):
        """创建 + 打开 + 配置相机对象。供 connect() 初次连接与 _selfheal() 完整重连用。

        失败抛异常（connect 向上传播；_selfheal 捕获后下个周期重试）。调用前应确保
        旧 self.cam 句柄已 Close/DestroyDevice（_selfheal 第二层会处理）。
        """
        factory = py.TlFactory.GetInstance()
        ip = self.camera_ip
        if ip:
            info = py.DeviceInfo()
            info.SetPropertyValue("IpAddress", ip)
            try:
                dev = factory.CreateDevice(info)
            except Exception:
                raise RuntimeError(
                    f"无法连接 IP={ip} 的相机。检查：通电/网线/同网段/防火墙/"
                    f"用 pylon IP Configurator 确认 IP。若 pylon Viewer 开着请先关闭。"
                )
        elif CAMERA_SERIAL:
            info = py.DeviceInfo()
            info.SetPropertyValue("SerialNumber", CAMERA_SERIAL)
            dev = factory.CreateDevice(info)
        else:
            devices = factory.EnumerateDevices()
            if not devices:
                raise RuntimeError("未发现任何 Basler 相机。检查通电/网线/防火墙。")
            dev = devices[0]
        self.cam = py.InstantCamera(dev)
        self.cam.Open()
        self.camera_info = {
            "model": self.cam.GetDeviceInfo().GetModelName(),
            "serial": self.cam.GetDeviceInfo().GetSerialNumber(),
            "ip": ip or (self.cam.GetDeviceInfo().GetIpAddress()
                         if hasattr(self.cam.GetDeviceInfo(), "GetIpAddress") else "N/A"),
        }
        self._configure()
        # 读回相机实际生效的曝光/增益，作为运行时调整的基准（_configure 可能沿用相机当前值，
        # 也可能被钳位，以实际 GetValue 为准）。
        try:
            n = self.cam.GetNodeMap().GetNode("ExposureTime")
            if n:
                self._exposure_us = n.GetValue()
        except Exception:
            pass
        try:
            n = self.cam.GetNodeMap().GetNode("Gain")
            if n:
                self._gain_db = n.GetValue()
        except Exception:
            pass

    def _configure(self):
        nodemap = self.cam.GetNodeMap()
        conf = []
        # 增益（None = 沿用相机当前值，不修改）
        if GAIN_DB is None:
            conf.append("增益=沿用相机当前值(不修改)")
        else:
            try:
                nodemap.GetNode("Gain").SetValue(GAIN_DB)
                conf.append(f"增益={GAIN_DB}dB")
            except Exception as e:
                conf.append(f"增益设置异常: {e}")
        # Gamma
        try:
            nodemap.GetNode("Gamma").SetValue(GAMMA)
            conf.append(f"Gamma={GAMMA}")
        except Exception as e:
            conf.append(f"Gamma设置异常: {e}")
        # 曝光：自动模式必须先关掉，否则手动曝光值写不进去
        try:
            auto = nodemap.GetNode("ExposureAuto")
            if auto:
                auto.SetValue("Off")
        except Exception:
            pass
        try:
            m = nodemap.GetNode("ExposureMode")
            if m:
                m.SetValue("Timed")
        except Exception:
            pass
        # 曝光：依次尝试 ExposureTime / ExposureTimeAbs
        ok = False
        for name in ("ExposureTime", "ExposureTimeAbs"):
            try:
                n = nodemap.GetNode(name)
                if n is None:
                    continue
                n.SetValue(EXPOSURE_TIME_US)
                conf.append(f"曝光={EXPOSURE_TIME_US}µs ({name})")
                self._exposure_node = name   # 记录实际生效的曝光节点，set_exposure 复用同一节点
                ok = True
                break
            except Exception as e:
                msg = str(e)
                if "not available" in msg.lower() or "placeholder" in msg.lower():
                    continue
                conf.append(f"{name} 设置失败: {msg}")
        if not ok:
            conf.append("曝光未写入，沿用相机当前曝光值")
        # 触发/采集模式：强制自由运行(Continuous + TriggerMode=Off)。
        # 若相机残留"外触发/软触发"配置而无触发源，会只出 1 帧后冻结。
        try:
            tm = nodemap.GetNode("TriggerMode")
            if tm is not None:
                tm.SetValue("Off")
        except Exception as e:
            conf.append(f"触发模式设置异常: {e}")
        try:
            am = nodemap.GetNode("AcquisitionMode")
            if am is not None:
                am.SetValue("Continuous")
        except Exception as e:
            conf.append(f"采集模式设置异常: {e}")
        # 像素格式
        try:
            nodemap.GetNode("PixelFormat").SetValue(PIXEL_FORMAT)
            conf.append(f"像素格式={PIXEL_FORMAT}")
        except Exception as e:
            conf.append(f"像素格式设置异常(沿用当前值): {e}")
        # 末次强制 TriggerMode=Off（防相机 UserSet 重载/后续参数把触发重新打开，导致只出 1 帧冻结）。
        # 注：AcquisitionMode=Continuous 已在上方设置；TriggerMode=Off 时 TriggerSource 无意义，
        #     故此处仅保留安全关键的 TriggerMode 末次强制，去除冗余。
        try:
            tm = nodemap.GetNode("TriggerMode")
            if tm is not None:
                tm.SetValue("Off")
            conf.append("触发模式=Off(自由运行,末次强制)")
        except Exception as e:
            conf.append(f"触发模式末次强制异常: {e}")
        # 采集帧率封顶（关键：防 1Gbps 链路被 48fps 原生自由运行压垮 → 取图超时/预览冻结）。
        # 教训（现场复现）：aca1920-48gm 上单数 AcquisitionFrameRate 是占位节点不可用，
        #       可用的是 AcquisitionFrameRateEnable + AcquisitionFrameRateAbs。
        #       【必须成对写入】先写安全 Abs 值、再开 Enable；曾因"单开 Enable 而 Abs 未
        #       写有效值"导致相机冻结(21 张全同/0 张)。封顶 6fps 既 >= 软件拍照 3fps，
        #       又远低于 48fps 压垮网卡的阈值。拍照节奏由程序 3fps 节流，与相机帧率解耦，
        #       不会把拍照锁死（历史曾锁到 2 张/秒是拍照节流 bug，与此无关）。
        SAFE_FPS = 6.0
        n_abs = nodemap.GetNode("AcquisitionFrameRateAbs")
        n_en = nodemap.GetNode("AcquisitionFrameRateEnable")
        if n_abs is None or n_en is None:
            conf.append("采集帧率=帧率节点不可用(占位)，保持相机默认；软件按 3fps 节流兜底")
        else:
            try:
                n_abs.SetValue(SAFE_FPS)   # ① 先写安全值
                n_en.SetValue(True)        # ② 再开限制（成对，绝不留"Enable 无有效 Abs"）
                conf.append(f"采集帧率=封顶 {SAFE_FPS}fps(Enable+Abs 成对；拍照仍按 {FPS}fps 节流)")
            except Exception as e:
                conf.append(f"采集帧率封顶失败(已忽略，软件节流兜底): {e}")
        # GigE 心跳超时：防止网络抖动/主机繁忙时相机误判"主机掉线"而主动断开
        # （典型表现正是：运行十几秒后报 physically removed，画面定格）。之前代码
        # 误用节点名 BeatTimeOut（从未生效），正确名为 GevHeartbeatTimeout。延长到
        # 60s 给足余量（即便主机偶发繁忙也不会被相机踢掉）。上限更小则取上限。
        try:
            hb = nodemap.GetNode("GevHeartbeatTimeout")
            if hb is not None:
                try:
                    hb.SetValue(60000)
                    conf.append("心跳超时=60000ms")
                except Exception:
                    try:
                        hb.SetValue(hb.GetMax())
                        conf.append(f"心跳超时=上限({int(hb.GetMax())}ms)")
                    except Exception as e:
                        conf.append(f"心跳超时设置跳过: {e}")
        except Exception:
            pass
        self._conf_log = conf
        for c in conf:
            print(f"  - {c}")

    def start(self):
        # 抓取策略：LatestImageOnly。单循环下 loop 连续消费，始终拿最新帧、自动丢弃旧帧，
        # 既保证预览流畅，又保证拍照收集到的都是不同时间点的新鲜帧。
        self.cam.StartGrabbing(py.GrabStrategy_LatestImageOnly)
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
        img = _grab_to_frame(grab)
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
        """唯一取流线程：持续取帧更新预览；拍照进行中按 3fps 收集帧写盘。

        全程只有这里调用 RetrieveResult，彻底规避"双线程并发取流"的冻屏根因。
        另含自愈：相机 GigE 心跳超时/网口抖动会令固件静默停出图（无异常、
        GrabSucceeded 恒为 False、_loop 看似活着实则空转 → 画面定格）。故监测
        "超过阈值无新帧"即重启取流，不退出进程。
        """
        last_good = time.perf_counter()
        while self.running:
            try:
                grab = self.cam.RetrieveResult(1000, py.TimeoutHandling_Return)
                if not grab or not grab.GrabSucceeded():
                    if grab:
                        grab.Release()
                    # 自愈：超过阈值无新帧（相机静默停出图/心跳超时）→ 重启取流
                    if time.perf_counter() - last_good > SELFHEAL_SEC:
                        self._selfheal()
                        last_good = time.perf_counter()
                    continue
                frame = _grab_to_frame(grab)   # 深拷贝，杜绝连续帧指向同一缓冲
                grab.Release()
                now = time.perf_counter()
                if frame is None:
                    # 无效帧（0尺寸/空缓冲）：不更新 _latest（保留上一有效帧），
                    # 但仍按拍照节奏调用 _collect（内部回退 _latest 保住成片）。
                    # 若直接跳过 _collect，坏帧期间不收集 → 张数不足甚至连拍超时。
                    if now - last_good > SELFHEAL_SEC:
                        self._selfheal()
                        last_good = now
                    if self._cap_state == "capturing":
                        self._collect(None, now)
                    continue
                last_good = now
                with self._lock:
                    self._latest = frame          # 始终刷新 → 预览不冻结
                    self._latest_ts = now
                    self._frame_no += 1           # 每次成功取图 +1：停出图时帧号停滞
                if self._cap_state == "capturing":
                    self._collect(frame, now)
            except Exception as e:
                self._error = f"取流异常: {e}"
                print(f"[相机] 取流异常(继续): {e}")
                time.sleep(0.1)

    def _selfheal(self):
        """相机掉线（静默停出图 / physically removed / 心跳超时）→ 自动恢复，不退出进程。

        分两级：
          ① 句柄尚可用 → 仅 StopGrab+StartGrab 重启取流（应对偶发停出图）；
          ② 句柄失效(physically removed) → 销毁旧句柄，重新枚举+创建+打开+配置+取流。
        每次调用只尝试一轮，由 _loop 每 SELFHEAL_SEC 秒调用一次形成周期重试；
        这样网络/供电恢复后程序能自动重新连上，无需人工重启服务。
        """
        # 第一层：句柄尚可用时，仅重启取流
        try:
            self.cam.StopGrabbing()
            self.cam.StartGrabbing(py.GrabStrategy_LatestImageOnly)
            self._reapply_exposure()   # 重连后把用户期望曝光重新下发（防被重置为默认2000）
            g = self.cam.RetrieveResult(2000, py.TimeoutHandling_Return)
            if g and g.GrabSucceeded():
                g.Release()
                print("[相机] 自检：取流已重启")
                return
            if g:
                g.Release()
        except Exception as e:
            print(f"[相机] 自检：句柄级重启失败(准备完整重连): {e}")
        # 第二层：physically removed / 句柄失效 → 销毁旧句柄并完整重建相机
        print("[相机] 自检：尝试完整重连相机（销毁旧句柄，重新枚举并打开）...")
        try:
            try:
                self.cam.StopGrabbing()
            except Exception:
                pass
            try:
                self.cam.Close()
            except Exception:
                pass
            try:
                self.cam.DestroyDevice()
            except Exception:
                pass
        except Exception:
            pass
        try:
            self._open_camera()                     # 重建 self.cam + 打开 + 配置 + 读回参数
            self.cam.StartGrabbing(py.GrabStrategy_LatestImageOnly)
            self._reapply_exposure()   # 完整重连后把用户期望曝光重新下发（防被重置为默认2000）
            g = self.cam.RetrieveResult(3000, py.TimeoutHandling_Return)
            if g and g.GrabSucceeded():
                img = _grab_to_frame(g)
                g.Release()
                self.height, self.width = img.shape[:2]
                print("[相机] 完整重连成功，取流恢复")
            else:
                if g:
                    g.Release()
                print("[相机] 完整重连：相机已重连但暂未出图，下个周期重试")
        except Exception as e:
            print(f"[相机] 完整重连失败(下个周期重试): {e}")

    def _collect(self, frame, now):
        """拍照状态机：仅当到下一目标时间(每 1/FPS 秒)才收集一帧，节奏由程序控制。

        若相机帧率远高于 3fps，多余帧在 `now < self._cap_next` 时直接丢弃；
        若相机帧率低于 3fps，则每帧都收集（间隔变长），但仍不会因"软件节流逻辑"
        而把拍照锁死——拍照只取决于本状态机，不依赖相机帧率配置。
        """
        with self._lock:
            if self._cap_state != "capturing":
                return
            if self._cap_idx >= self._cap_total:
                return
            if now < self._cap_next:
                return
            idx = self._cap_idx + 1
            self._cap_idx = idx
            self._cap_next += 1.0 / self._cap_fps
            cap_start = self._cap_start
            cap_t0_dt = self._cap_t0_dt
            cap_save_dir = self._cap_save_dir
            cap_prefix = self._cap_prefix
            cap_latest = self._latest       # 回退源：最近一次有效帧（永不空）
            done = (self._cap_idx >= self._cap_total)
        # 选有效帧：当帧有效优先；当帧为空（偶发坏帧）则回退最近有效帧，保住成片
        cap_frame = frame if (frame is not None and getattr(frame, "size", 0) > 0) else cap_latest
        # 锁外写盘：单个写盘异常只跳过当前张，不炸整批、不炸取流
        frame_dt = cap_t0_dt + datetime.timedelta(seconds=(now - cap_start))
        fname = self._format_filename(idx=idx, dt=frame_dt, prefix=cap_prefix)
        fpath = os.path.join(cap_save_dir, fname)
        try:
            if cap_frame is None or cap_frame.size == 0:
                raise ValueError("无可写帧(当前帧与回退帧均为空)")
            cv2.imwrite(fpath, cap_frame)
            h = _frame_hash(cap_frame)
            with self._lock:
                self._cap_frames.append(fname)
                self._cap_hashes.append(h)
        except Exception as e:
            print(f"[拍照] 写盘失败(跳过该张): {fpath}: {e}")
            with self._lock:
                self._error = f"写盘失败: {e}"
        if done:
            # 注意：不能在此处持锁调用 _finish()——_finish 内部也会 with self._lock，
            # 而 threading.Lock 不可重入，会在"最后一张"时死锁 _loop 线程 → 预览永久冻结。
            # 故先释放锁读取状态，再无锁调用 _finish()（它自己会加锁）。
            with self._lock:
                still_capturing = (self._cap_state == "capturing")
            if still_capturing:
                self._finish()

    def _finish(self):
        """连拍完成：汇总结果、置 _capture_done，唤醒 request_capture 的等待。"""
        elapsed = time.perf_counter() - self._cap_start
        with self._lock:
            self._cap_state = "idle"
            self._cap_running = False
            batch = list(self._cap_frames)
            hashes = list(self._cap_hashes)
            cap_save_dir = self._cap_save_dir
            cap_fps = self._cap_fps
            cap_total = self._cap_total
        saved = len(batch)
        cam_frames = self._frame_no - self._cap_frame_no0
        unique = len(set(hashes))
        identical = (saved > 1 and unique <= 1)
        elapsed = round(elapsed, 2)
        abs_files = [os.path.abspath(os.path.join(cap_save_dir, f)) for f in batch]
        print(f"[拍照] 完成: 目标{cap_total}张 实际{saved}张 唯一帧={unique}/{saved} "
              f"耗时={elapsed}s 是否全同={identical} 相机出帧={cam_frames} "
              f"落盘={cap_save_dir}")
        if identical:
            print("[拍照] ⚠️ 全部相同 → 相机未持续出图(检查 TriggerMode=Off/网口掉流)")
        if saved:
            print(f"[拍照] 已落盘 {saved} 张 -> {os.path.abspath(cap_save_dir)}")
            print(f"[拍照] 首张: {abs_files[0]}")
            print(f"[拍照] 末张: {abs_files[-1]}")
        self._last_result = {
            "ok": saved > 0,
            "saved": saved,
            "unique_frames": unique,
            "identical": identical,
            "total": cap_total,
            "elapsed": elapsed,
            "actual_fps": round(saved / elapsed, 1) if elapsed > 0 else 0,
            "mode": f"连拍{cap_total}张(程序{cap_fps}fps,与相机帧率解耦)",
            "files": batch,
            "abs_files": abs_files,
            "save_dir": os.path.abspath(cap_save_dir),
            "ts": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._capture_done.set()

    # ---------- 运行时参数调整（手动测试面板用，自由运行下直接写相机）----------
    def set_exposure(self, us):
        """运行时设置曝光时间(µs)，返回相机真实状态用于远程诊断。安全钳位 [50, 100000]µs。

        返回字段含 auto(自动曝光当前值)/mode(曝光模式)/readbacks(各曝光节点写后读回)，
        便于判断"值写了但亮度不变"到底是：自动曝光没关掉 / 写错节点 / 还是场景已饱和。
        """
        if not self.running or self.cam is None:
            return {"ok": False, "error": "相机未运行", "exposure_us": self._exposure_us}
        us = max(50, min(100000, int(round(us))))
        try:
            nm = self.cam.GetNodeMap()
            diag = {"ok": True, "requested": us}
            # ① 关自动曝光（读回确认是否真关掉——若仍是 Continuous 则手动值会被覆盖）
            try:
                a = nm.GetNode("ExposureAuto")
                if a is not None:
                    try:
                        a.SetValue("Off")
                    except Exception:
                        pass
                    try:
                        diag["auto"] = a.GetValue()
                    except Exception:
                        diag["auto"] = "读不到"
            except Exception:
                diag["auto"] = "节点无"
            # ② 定时模式
            try:
                em = nm.GetNode("ExposureMode")
                if em is not None:
                    try:
                        em.SetValue("Timed")
                    except Exception:
                        pass
                    try:
                        diag["mode"] = em.GetValue()
                    except Exception:
                        pass
            except Exception:
                pass
            # ③ 只写启动确认的真实曝光节点（该相机仅 ExposureTimeAbs 可用，
            #    ExposureTime 节点不存在会报 ERR；不再试写不存在节点，避免噪音）
            node_name = getattr(self, "_exposure_node", "ExposureTimeAbs")
            n = nm.GetNode(node_name)
            readbacks = {}
            if n is None:
                # 兜底：启动记录的节点不可用时退回另一个节点
                alt = "ExposureTimeAbs" if node_name != "ExposureTimeAbs" else "ExposureTime"
                n = nm.GetNode(alt)
                if n is not None:
                    node_name = alt
            if n is not None:
                try:
                    n.SetValue(us)
                    readbacks[node_name] = n.GetValue()
                except Exception as e:
                    readbacks[node_name] = f"ERR:{e}"
            diag["readbacks"] = readbacks
            actual = readbacks.get(node_name)
            try:
                actual = float(actual)
            except Exception:
                actual = us
            self._exposure_us = actual
            self._exposure_user = actual   # 记录用户期望曝光，自愈重连后据此重新下发
            diag["exposure_us"] = actual
            diag["node"] = node_name
            print(f"[相机] 曝光请求 {us}µs → 实际 {actual}µs 节点 {node_name} "
                  f"auto={diag.get('auto')} mode={diag.get('mode')} readbacks={readbacks}")
            return diag
        except Exception as e:
            return {"ok": False, "error": f"曝光设置失败: {e}", "exposure_us": self._exposure_us}

    def _reapply_exposure(self):
        """自愈重连成功后，把用户期望曝光(self._exposure_user)重新下发到相机。

        完整重连会经 _open_camera 把相机曝光重置为默认 2000µs（且只读回 ExposureTime
        节点，该相机此节点不存在故不更新 _exposure_us），导致用户之前在网页设的曝光被
        静默清零。故重连后主动用 _exposure_node 把 _exposure_user 重新写回，使手动曝光
        在掉线重连后依然存活。失败仅打印告警，不抛异常（避免干扰取流恢复）。
        """
        us = getattr(self, "_exposure_user", None)
        node = getattr(self, "_exposure_node", None)
        if us is None or node is None or self.cam is None:
            return
        try:
            n = self.cam.GetNodeMap().GetNode(node)
            if n is not None:
                n.SetValue(int(round(us)))
                print(f"[相机] 重连后重发曝光={int(round(us))}µs (节点 {node})")
        except Exception as e:
            print(f"[相机] 重连后重发曝光失败(下个周期重试): {e}")

    def set_gain(self, db):
        """增益为固定值(GAIN_DB)，前端无调整入口；保留占位以兼容旧调用，实际不修改相机。"""
        return {"ok": True, "gain_db": self._gain_db, "note": "增益固定，未修改"}

    def request_capture(self, fps=None, total=None, save_dir=None, prefix="Image", delay=None):
        """外部触发连拍，阻塞直到完成，返回结果 dict。

        参数（每次可覆盖，自动模式用默认、手动测试模式由网页传入）：
          fps      : 拍照节奏(张/秒)，仅决定软件收集节拍，与相机实际帧率解耦 → 不会锁死
          total    : 总张数
          save_dir : 落盘目录（自动=SAVE_DIR 临时缓冲，手动测试=MANUAL_TEST_ROOT/日期/时间）
          prefix   : 文件名前缀（自动=Image，手动=Test）
        """
        if not self.running:
            return {"ok": False, "error": "相机未运行"}
        fps = fps or FPS
        total = total or TOTAL
        save_dir = save_dir or SAVE_DIR
        try:
            os.makedirs(save_dir, exist_ok=True)
        except Exception as e:
            return {"ok": False, "error": f"创建存储目录失败: {e}"}
        # 拍摄前延迟：收到指令(PLC出车信号/点击手动)后先等目标物移动到视野合适位置。
        # 放在加锁之前、且不持锁睡觉，既不影响取流线程，也避免此前"日志先打印、锁后拿"的
        # 竞态窗口（手动恰好卡在打印与加锁之间会误判两场同时开始）。
        # delay 由调用方显式传入（自动= AUTO_PRE_CAPTURE_DELAY / 手动= MANUAL_PRE_CAPTURE_DELAY）；
        # 未传则回退手动默认（自检等无参调用）。
        cap_delay = delay if delay is not None else MANUAL_PRE_CAPTURE_DELAY
        if cap_delay > 0:
            print(f"[拍照] 收到拍摄指令，延迟 {cap_delay}s 后开始连拍")
            time.sleep(cap_delay)
        with self._lock:
            if self._cap_running:
                return {"ok": False, "error": "拍照进行中，请稍候"}
            print(f"[拍照] 开始连拍: 目标{total}张 @{fps}fps -> {save_dir}")
            self._cap_running = True
            self._cap_state = "capturing"
            self._cap_idx = 0
            self._cap_start = time.perf_counter()
            self._cap_next = self._cap_start   # 第一张立即收集
            self._cap_frames = []
            self._cap_hashes = []
            self._cap_t0_dt = datetime.datetime.now()
            # 本次拍照参数（_collect/_finish 读取）
            self._cap_fps = fps
            self._cap_total = total
            self._cap_save_dir = save_dir
            self._cap_prefix = prefix
            self._cap_frame_no0 = self._frame_no   # 拍照起始帧号（用于诊断相机是否持续出图）
        dur = total / fps if fps > 0 else DURATION_SEC
        # 超时余量：基础 6s + 张数*0.6，给偶发 GigE 抖动/自愈留足时间，避免误超时
        timeout = dur + max(6.0, total * 0.6)
        self._capture_done.clear()
        if self._capture_done.wait(timeout=timeout):
            with self._lock:
                return self._last_result or {"ok": False, "error": "连拍无结果"}
        # 超时兜底：复位状态，避免永久卡在 capturing；并把已存张数如实回填
        with self._lock:
            self._cap_running = False
            self._cap_state = "idle"
            batch = list(self._cap_frames)
            hashes = list(self._cap_hashes)
            cap_save_dir = self._cap_save_dir
            cap_total = self._cap_total
        saved = len(batch)
        unique = len(set(hashes))          # 与 _finish 口径一致：唯一帧=已存帧去重，而非错填为 saved
        cam_frames = self._frame_no - self._cap_frame_no0
        elapsed = round(timeout, 2)        # 与等待阈值一致（原 dur+5.0 与 timeout 算法不一致）
        print(f"[拍照] 超时兜底: 目标{cap_total}张 实际仅存{saved}张 唯一帧={unique}/{saved} "
              f"（拍照期间相机出帧{cam_frames}帧）-> "
              f"{os.path.abspath(cap_save_dir) if cap_save_dir else ''}")
        self._last_result = {
            "ok": saved > 0,
            "saved": saved,
            "unique_frames": unique,
            "total": cap_total,
            "elapsed": elapsed,
            "actual_fps": round(saved / elapsed, 2) if elapsed > 0 else 0,
            "save_dir": cap_save_dir,
            "error": "连拍超时(未完成全部张数)",
        }
        return self._last_result

    def _format_filename(self, idx=None, dt=None, prefix="Image"):
        if dt is None:
            dt = datetime.datetime.now()
        ts = dt.strftime("%Y-%m-%d__%H-%M-%S-%f")[:-3]
        idx_part = f"_{idx:02d}" if idx is not None else ""
        # 文件名带版本短号 p{VERSION_TAG}，现场一眼即可确认是否跑了最新程序
        return f"{prefix}{idx_part}__p{VERSION_TAG}__{ts}{SAVE_EXT}"

    def get_latest_jpeg(self):
        """返回最新帧的 JPEG 字节，或 None。

        防御性：任何异常都返回 None（让 MJPEG 流继续存活，而不是整条流被异常打死→界面卡死）。
        若 _latest 超过 STALE_SEC 未刷新（相机冻结/取流异常），在画面上叠加红色告警，
        让操作员直接看到"预览卡住/信号丢失"，而不是静默定格还以为正常。
        """
        try:
            with self._lock:
                frame = self._latest
                age = time.perf_counter() - self._latest_ts
                fno = self._frame_no
            if frame is None or getattr(frame, "size", 0) == 0:
                return None
            scale = STREAM_WIDTH / self.width if self.width else 1.0
            disp = cv2.resize(frame, (STREAM_WIDTH, int(self.height * scale)))
            if disp.ndim == 2:  # 单色/异常 2 通道 → 转 3 通道再叠加文字（合并原两处冗余转换）
                disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)
            # 常驻心跳：帧号 + 当前时钟 + 卡顿秒数，绿色小字固定在左下角。
            # 帧号递增=实时出帧；帧号不动+下方红字=相机冻结/信号丢失。
            clk = time.strftime("%H:%M:%S")
            hb = f"LIVE #{fno}  {clk}  卡顿{age:.1f}s"
            cv2.putText(disp, hb, (8, disp.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)
            if age > STALE_SEC:
                txt = f"!! 预览卡住/信号丢失 ({age:.0f}s)"
                cv2.rectangle(disp, (0, 0), (disp.shape[1], 34), (0, 0, 180), -1)
                cv2.putText(disp, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2, cv2.LINE_AA)
            ok, buf = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            return buf.tobytes() if ok else None
        except Exception as e:
            print(f"[预览] 取最新帧异常(已忽略，流继续): {e}")
            return None

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
                "fps": AUTO_FPS, "duration_sec": AUTO_DURATION_SEC, "total": AUTO_TOTAL,
                "exposure_us": self._exposure_us,
                "gain_db": self._gain_db,
                "gain_display": ("相机当前值" if self._gain_db is None
                                 else f"{self._gain_db} dB"),
                "gamma": GAMMA,
            },
            "save_dir": INSPECTION_ROOT,
            "last_result": self._last_result,
            "preview_age_sec": round(time.perf_counter() - self._latest_ts, 1),
            "frame_no": self._frame_no,
            "capture_running": self._cap_running,
            "error": self._error,
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
    """自检模式：连相机 → 稳定 1s → 触发一次 21 张 → 校验是否各不相同。

    无需 PLC / 网页服务，~10 秒即可确认"每秒 3 张、21 张不同"是否已达成。
    返回 0=通过, 1=失败。
    """
    if not HAS_PYPYLON:
        print("[自检] 未安装 pypylon，请先: pip install pypylon")
        return 1
    print(f"[自检] VERSION={VERSION}")
    hub = CameraHub()
    try:
        hub.connect_all()
        hub.start_all()
    except Exception as e:
        print(f"[自检] 连接/启动失败: {e}")
        return 1
    print("[自检] 相机已启动，稳定 1s ...")
    time.sleep(1)
    print(f"[自检] 触发连拍 {TOTAL} 张 ...")
    r = hub.request_capture_primary(delay=2.0)
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
    if uniq == saved == TOTAL:
        print(f"[自检] ✓ 通过: {TOTAL} 张各不相同，程序 {FPS}fps 连拍生效")
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
        self._recent_lock = threading.Lock()   # 保护 recent_cars：PLC 线程写 vs Flask status 读

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

    def request_capture_primary(self, fps=None, total=None, save_dir=None, prefix="Image", delay=None):
        """手动按钮用：返回主相机连拍结果 dict（含 files/saved/...）。

        可选透传拍照参数（手动测试模式由网页传入 fps/total/save_dir/prefix/delay）；
        无参时沿用默认 21@3fps（自检/强制拍摄路径）。delay 决定拍摄前延时：
        手动路径传 MANUAL_PRE_CAPTURE_DELAY，强制拍摄传 AUTO_PRE_CAPTURE_DELAY，自检传 2.0。
        """
        if not self.streamers:
            return {"ok": False, "error": "相机未运行"}
        return self.streamers[0].request_capture(fps=fps, total=total,
                                                 save_dir=save_dir, prefix=prefix,
                                                 delay=delay)

    def request_capture_all(self):
        """自动触发用：触发全部相机，返回主相机文件列表 + 各相机结果。"""
        by_camera = {}
        primary_files = []
        primary_result = None
        for ip, s in zip(self.ips, self.streamers):
            r = s.request_capture(fps=AUTO_FPS, total=AUTO_TOTAL,
                                  delay=AUTO_PRE_CAPTURE_DELAY)
            by_camera[ip] = r
            if ip == self.primary_ip:
                primary_files = r.get("files", [])
                primary_result = r
        return {"primary_files": primary_files,
                "primary_result": primary_result,
                "by_camera": by_camera}

    def status(self):
        with self._recent_lock:
            recent = list(self.recent_cars)   # 加锁拷贝，避免与自动线程并发写竞争
        if not self.streamers:
            return {"running": False, "plc_auto": self.plc_auto,
                    "last_auto": self.last_auto, "recent_cars": recent,
                    "cameras": []}
        base = self.streamers[0].status()
        base["cameras"] = [s.status() for s in self.streamers]
        base["primary_ip"] = self.primary_ip
        base["plc_auto"] = self.plc_auto
        base["last_auto"] = self.last_auto
        base["recent_cars"] = recent
        return base


# ===================== PLC 自动触发回调（仅 --plc-auto）=====================
def _run_detection(model, key, image_paths, ts_dt):
    """拍照后跑真实检测，返回 (DetectionResult, proc_dir_rel)。

    设计约束：只“读图 + 算”，绝不碰相机/拍摄任何逻辑。
      - key 经 router 命中检测器（当前 9X）→ SealDetector 跑真实检测；
        未命中（8X/未知）→ 占位“算法未接入”，与之前行为一致（仅拍照存档）。
      - proc_dir_rel 为相对 ROOT 的目录（data/process_data/<时间>_<key>），
        内含 detector 生成的 mask/叠加图，供 ui/app.py 可视化读取。
    检测异常不致命：捕获后回退占位，保证拍照/落库链路不中断。
    """
    from detection.router import get_detector
    from common.interfaces import DetectionResult, Defect

    if not image_paths:
        return DetectionResult(car_model=model, ok=True,
                                defects=[Defect(0, 0, 0, 0, "pending", 0.0,
                                                meta={"reason": "无照片"})],
                                message="无照片·未检测"), ""

    detector = get_detector(key)
    if detector is None:
        return DetectionResult(car_model=model, ok=True,
                                defects=[Defect(0, 0, 0, 0, "pending", 0.0,
                                                meta={"reason": "算法未接入"})],
                                confidence=0.0, message="拍照存档·算法未接入"), ""

    proc_dir_rel = os.path.join("data", "process_data",
                                 f"{ts_dt.strftime('%Y%m%d_%H%M%S')}_{key}")
    proc_dir_abs = os.path.join(ROOT, proc_dir_rel)
    try:
        det = detector.detect(car_model=model, images=image_paths,
                               process_dir=proc_dir_abs)
        return det, proc_dir_rel
    except Exception as e:
        print(f"[检测] 运行异常: {e}")
        return DetectionResult(car_model=model, ok=True,
                                defects=[Defect(0, 0, 0, 0, "pending", 0.0,
                                                meta={"reason": f"检测异常:{e}"})],
                                message=f"检测异常:{e}"), proc_dir_rel


def handle_car_signal(ctx):
    """出车信号上升沿回调：记录全部车 → 仅(9X/8X 且 NO_Paint=0)拍照 → 落库+写追溯。

    流程（用户定义）：
      上升沿 → 用提前锁存的 DB230 记录全部车的车型/NO_Paint/滑橇/PIN →
      若车型是 9X 或 8X 且 NO_Paint=0 → 触发相机拍照（出车信号后立即 5s 连拍）；
      否则不拍照（免检车 / 未接入车型）。
    所有车型用同一出车信号触发，避免拍照时机不一致导致照片错位。

    本迭代已接入检测：拍照车跑真实检测（9X 命中 SealDetector；8X/未知占位），
    结果 + 过程叠加图落库，供 ui/app.py 可视化。全程只读图，不碰相机/拍摄。
    全程只读 PLC；本函数不写任何 PLC 地址。
    ctx 为 plc_monitor.parse_context 的 dict（提前锁存的 DB230 上下文）。
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

    # 持久记录"收到车信号"——PLC 上下文在 5 秒连拍期间会被下一台车覆盖，
    # 此处在该车被收到的瞬间落盘，确保 MM** 等信号有完整追溯(车型/滑橇/PIN/决策)。
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

    # —— 检测接入（核心打通）——
    # 拍照后的图跑真实检测，替换原占位结果；检测只读图、不改任何相机/拍摄逻辑。
    # 检测过程数据（mask/叠加图）落地到 data/process_data/<车>/，供可视化读取。
    paths = [os.path.join(SAVE_DIR, f) for f in files]
    det, proc_dir = _run_detection(model, key, paths, ts_dt)

    # 落库（含 skid/pin/no_paint/captured）；StorageService 内部按 日期/PIN 分层、
    # 去重移动、轮转，并返回该车文件夹路径
    db_ok = False
    rec = None
    try:
        from storage.service import get_service
        rec = get_service().save(model, det, paths,
                                 skid=skid, pin=pin, no_paint=no_paint,
                                 captured=captured, event_time=ts_dt,
                                 model_key=key, proc_dir=proc_dir)
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
    with hub._recent_lock:
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
    """MJPEG 流生成器（推送节拍上限 STREAM_FPS，避免无谓的 JPEG 编码拖 CPU）。

    防御性：单帧异常被吞掉并短暂让步，绝不让整条流被打死——否则网页实时界面会卡死。
    """
    last = 0.0
    interval = 1.0 / STREAM_FPS
    while hub.running:
        try:
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
        except Exception as e:
            print(f"[预览] MJPEG 单帧异常(已忽略，流继续): {e}")
            time.sleep(0.1)


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
  <span class="badge" id="plcstate">PLC…</span>
  <span class="badge" id="modeinfo">—</span></header>
<div class="wrap">
  <div class="video"><img id="feed" src="/video_feed"></div>

  <div style="display:flex;gap:8px;margin:14px 0">
    <button id="tabManual" onclick="switchTab('manual')" style="background:#2d6cdf">手动测试</button>
    <button id="tabAuto" onclick="switchTab('auto')" style="background:#555">自动监控</button>
  </div>

  <!-- 手动测试面板：测相机，设参数，点拍即拍，与 PLC 无关 -->
  <div id="panelManual">
    <div style="border:1px solid #333;border-radius:8px;padding:12px;margin-bottom:12px;background:#161616">
      <div style="font-size:14px;margin-bottom:8px;color:#9cf">相机参数（实时写入相机，预览立即生效）</div>
      <div style="display:flex;gap:14px;flex-wrap:wrap;align-items:flex-end">
        <label style="font-size:12px;color:#aaa">曝光(µs)<br>
          <input id="exp" type="number" min="50" max="100000" step="50" value="3000"
                 style="width:110px;padding:6px;background:#000;color:#eee;border:1px solid #444"></label>
        <label style="font-size:12px;color:#aaa">拍照延时(秒)<br>
          <input id="delay" type="number" min="0" max="30" step="0.5" value="5"
                 style="width:90px;padding:6px;background:#000;color:#eee;border:1px solid #444"></label>
        <span style="font-size:12px;color:#789">增益：固定沿用相机当前值（Gain Raw 136，不可在此修改）。帧率见下方“连拍设置”，点“应用参数”一并生效。</span>
        <button onclick="applyCam()">应用参数</button>
        <span id="camState" class="meta"></span>
      </div>
      <div style="font-size:13px;color:#aaa;margin-top:10px">连拍设置</div>
      <div style="display:flex;gap:14px;flex-wrap:wrap;align-items:flex-end;margin-top:4px">
        <label style="font-size:12px;color:#aaa">每秒张数(1-6, 允许小数如2.5=2秒5张)<br>
          <input id="fps" type="number" min="1" max="6" step="0.5" value="3"
                 style="width:80px;padding:6px;background:#000;color:#eee;border:1px solid #444"></label>
        <label style="font-size:12px;color:#aaa">总张数(1-200)<br>
          <input id="total" type="number" min="1" max="200" step="1" value="21"
                 style="width:80px;padding:6px;background:#000;color:#eee;border:1px solid #444"></label>
      </div>
      <div style="font-size:12px;color:#789;margin-top:8px">
        说明：仅测试相机拍摄功能，不判车型、不存检测库、不连 PLC；照片存 data/manual_test/日期/时间/。<br>
        相机已封顶 6fps（防 1Gbps 链路拥塞冻屏），连拍速度上限即 6，设更高无效。<br>
        调曝光后看上方状态：若显示 <b>Auto=Off</b> 且 <b>节点读回=设定值</b> 但照片仍一样，多半是场景过曝饱和——把曝光调到 100µs 看是否明显变暗即可验证（变暗=曝光生效）。</div>
    </div>
    <div class="bar">
      <button id="cap" onclick="manualCapture()">📸 开始拍摄（手动测试）</button>
      <span id="capState" class="meta"></span>
    </div>
    <div id="gal" style="display:flex;gap:6px;flex-wrap:wrap;margin:10px 0"></div>
  </div>

  <!-- 自动监控面板：PLC 触发 + 车型筛选 + 存档 -->
  <div id="panelAuto" style="display:none">
    <div class="meta" id="meta"></div>
    <div class="bar">
      <button id="capAuto" onclick="autoCapture()">📸 强制拍摄（补拍/调试）</button>
      <span id="capStateAuto" class="meta"></span>
    </div>
    <div style="border:1px solid #333;border-radius:8px;padding:12px;margin-bottom:12px;background:#161616">
      <div style="font-size:14px;margin-bottom:8px;color:#9cf">自动拍照延时（仅自动 PLC 触发 / 强制拍摄生效，手动模式不受影响）</div>
      <div style="display:flex;gap:14px;flex-wrap:wrap;align-items:flex-end">
        <label style="font-size:12px;color:#aaa">拍照延时(秒)<br>
          <input id="autoDelay" type="number" min="0" max="30" step="0.5" value="2"
                 style="width:90px;padding:6px;background:#000;color:#eee;border:1px solid #444"></label>
        <button onclick="applyAutoDelay()">应用参数</button>
        <span id="autoCamState" class="meta"></span>
      </div>
      <div style="font-size:12px;color:#789;margin-top:8px">
        说明：自动触发条件 = 出车信号(DB130.DBX0.1)上升沿 + 车身同步信号(DB890.DBX225.2 与 225.3 同时为 1)。
        三者满足后等本延时再连拍；本值仅作用自动模式，手动延时在“手动测试”面板单独设置（默认 5s）。
      </div>
    </div>
    <h3 style="font-size:14px;margin:18px 0 6px">最近车辆（PLC 触发）</h3>
    <table class="ctab"><thead><tr>
      <th>时间</th><th>车型</th><th>滑橇</th><th>PIN</th><th>NO_Paint</th><th>拍照</th><th>检测(未接入)</th>
    </tr></thead><tbody id="carbody">
      <tr><td colspan="7" class="meta">等待出车信号…</td></tr>
    </tbody></table>
  </div>

  <h3 style="font-size:14px;margin:18px 0 6px">采集日志</h3>
  <div id="log">等待操作…</div>
</div>
<script>
const state=document.getElementById('state');
const plcstate=document.getElementById('plcstate');
const modeinfo=document.getElementById('modeinfo');
const meta=document.getElementById('meta');
const log=document.getElementById('log');
const carbody=document.getElementById('carbody');
const capState=document.getElementById('capState');
const capStateAuto=document.getElementById('capStateAuto');
const camState=document.getElementById('camState');
const autoCamState=document.getElementById('autoCamState');
const gal=document.getElementById('gal');
let _inited=false;

function switchTab(name){{
  document.getElementById('panelManual').style.display=(name==='manual')?'block':'none';
  document.getElementById('panelAuto').style.display=(name==='auto')?'block':'none';
  document.getElementById('tabManual').style.background=(name==='manual')?'#2d6cdf':'#555';
  document.getElementById('tabAuto').style.background=(name==='auto')?'#2d6cdf':'#555';
}}

function applyCam(){{
  const exp=parseFloat(document.getElementById('exp').value);
  const fps=parseFloat(document.getElementById('fps').value);
  const delay=parseFloat(document.getElementById('delay').value);
  const body={{exposure_us:isNaN(exp)?null:exp,
              fps:isNaN(fps)?null:fps,
              delay:isNaN(delay)?null:delay}};
  camState.textContent='应用中…';
  fetch('/api/camera',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(body)}})
    .then(r=>r.json()).then(res=>{{
      let m='';
      if(res.exposure){{
        const e=res.exposure;
        if(e.ok){{
          m+='曝光='+e.exposure_us+'µs ';
          if(e.auto!==undefined) m+='[Auto='+e.auto+'] ';
          if(e.mode!==undefined) m+='[Mode='+e.mode+'] ';
          if(e.readbacks){{ m+='[节点读回 ';
            for(const k in e.readbacks) m+=k+'='+e.readbacks[k]+' ';
            m+='] '; }}
        }} else {{
          m+='曝光失败:'+(e.error||'')+' ';
        }}
      }}
      if(res.fps!==undefined && res.fps!==null){{ document.getElementById('fps').value=res.fps; m+='帧率='+res.fps+'fps '; }}
      if(res.delay!==undefined && res.delay!==null){{ document.getElementById('delay').value=res.delay; m+='延时='+res.delay+'s '; }}
      camState.textContent=m||'已应用';
    }}).catch(e=>{{camState.textContent='参数应用出错:'+e;}});
}}

function applyAutoDelay(){{
  const d=parseFloat(document.getElementById('autoDelay').value);
  const body={{auto_delay:isNaN(d)?null:d}};
  autoCamState.textContent='应用中…';
  fetch('/api/camera',{{method:'POST',headers:{{'Content-Type':'application/json'}},body:JSON.stringify(body)}})
    .then(r=>r.json()).then(res=>{{
      let m='';
      if(res.auto_delay!==undefined && res.auto_delay!==null){{
        document.getElementById('autoDelay').value=res.auto_delay;
        m+='自动延时='+res.auto_delay+'s ';
      }} else if(res.error){{
        m=res.error;
      }}
      autoCamState.textContent=m||'已应用';
    }}).catch(e=>{{autoCamState.textContent='参数应用出错:'+e;}});
}}

function manualCapture(){{
  let fps=parseFloat(document.getElementById('fps').value)||3;
  if(fps>6){{fps=6;}} if(fps<1){{fps=1;}}
  const total=parseInt(document.getElementById('total').value)||21;
  const btn=document.getElementById('cap');
  if(btn){{btn.disabled=true;}}
  capState.textContent='准备中…(延迟3.5秒后开始拍摄，共约10秒)'; gal.innerHTML='';
  console.log('[前端] 开始手动拍摄', {{fps:fps,total:total}});
  const payload={{test:true,fps:fps,total:total}};
  const ctrl=new AbortController();
  const t=setTimeout(()=>ctrl.abort(), 30000);
  // 单一恢复函数：无论成功/失败/超时，按钮必回到可点状态（绿色）
  let restored=false;
  const restore=()=>{{ if(restored) return; restored=true; clearTimeout(t); ctrl.abort();
    if(btn){{btn.disabled=false;}} }};   // 只恢复按钮；拍摄结果由下方 .then 写进 capState 保留显示
  const safety=setTimeout(restore, 25000);   // 终极兜底：25秒内任何情况都强制恢复按钮
  fetch('/api/capture',{{method:'POST',headers:{{'Content-Type':'application/json'}},
      body:JSON.stringify(payload), signal:ctrl.signal}})
    .then(r=>{{clearTimeout(safety); console.log('[前端] /api/capture 响应状态', r.status); return r.json();}})
    .then(res=>{{
      console.log('[前端] /api/capture 结果', res);
      if(res.ok){{
        capState.textContent='✅ 完成 '+res.saved+'/'+res.total+' 张'; capState.style.color='#6f6';
        let line=`[${{res.ts}}] 手动测试 完成 拍 ${{res.saved}}/${{res.total}} 张 · 耗时 ${{res.elapsed}}s · ${{res.actual_fps}} fps\\n`+
                 `已存到：${{res.save_dir}}\\n`+
                 `唯一帧=${{res.unique_frames}}/${{res.saved}}`+(res.identical?' ⚠️全部相同(相机未持续出图)':'')+`\\n`;
        log.textContent=line+'\\n'+log.textContent;
        (res.rel_files||[]).forEach(r=>{{
          let i=document.createElement('img');
          i.src='/api/image?rel='+encodeURIComponent(r);
          i.style.cssText='height:90px;border:1px solid #444;border-radius:4px';
          gal.appendChild(i);
        }});
        if(!(res.rel_files||[]).length) gal.innerHTML='<span class="meta">无缩略图</span>';
      }}else{{
        capState.textContent='❌ 0张 · '+(res.error||'未写入任何文件'); capState.style.color='#f66';
        log.textContent='[错误] '+(res.error||'未知错误')+'（未写入任何文件）\\n'+log.textContent;
      }}
    }})
    .catch(e=>{{ console.error('[前端] /api/capture 失败', e);
      capState.textContent='⚠️ 网络/请求错误'; capState.style.color='#fa3';
      log.textContent='[网络/请求错误] '+e+'（若已存照片请去 manual_test 目录查看）\\n'+log.textContent; }})
    .finally(restore);
}}

function autoCapture(){{
  const btn=document.getElementById('capAuto');
  btn.disabled=true; capStateAuto.textContent='拍摄中…';
  fetch('/api/capture',{{method:'POST'}}).then(r=>r.json()).then(res=>{{
    if(res.ok){{ log.textContent=`[${{res.ts}}] 强制拍摄 拍 ${{res.saved}}/${{res.total}} 张 -> ${{res.save_dir}}\\n`+log.textContent; }}
    else {{ log.textContent='[强制拍摄错误] '+(res.error||'')+'\\n'+log.textContent; }}
    capStateAuto.textContent=''; btn.disabled=false;
  }}).catch(e=>{{capStateAuto.textContent='';btn.disabled=false;
    log.textContent='[网络错误] '+e+'\\n'+log.textContent;}});
}}

function refreshStatus(){{
  fetch('/api/status').then(r=>r.json()).then(s=>{{
    if(s.running){{state.textContent='● 在线';state.style.background='#2d8c4a';}}
    else{{state.textContent='● 离线';state.style.background='#b5482f';}}
    if(s.plc_auto){{
      plcstate.textContent='● PLC自动';plcstate.style.background='#2d6cdf';
      modeinfo.textContent='自动监控：PLC触发中';
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
      modeinfo.textContent='手动测试：PLC未启用（加 --plc-auto 开启自动）';
      carbody.innerHTML='<tr><td colspan="7" class="meta">自动监控未启用（加 --plc-auto 启动）</td></tr>';
    }}
    let h=`<b>相机</b>：${{s.camera.model}}（序列号 ${{s.camera.serial}}）<br>`;
    h+=`<b>程序版本</b>：v${{s.version_tag}}（照片文件名含此短号，可确认是否最新）<br>`;
    h+=`<b>分辨率</b>：${{s.resolution}} · ${{s.color}} · ${{s.pixel_format}}<br>`;
    h+=`<b>自动连拍</b>：出车信号触发连拍 ${{s.params.duration_sec}} 秒 = <b>${{s.params.total}} 张</b>（${{s.params.fps}}fps）<br>`;
    h+=`<b>当前曝光</b>：${{s.params.exposure_us}} µs · <b>增益</b>：${{s.params.gain_display}} · Gamma：${{s.params.gamma}}<br>`;
    h+=`<b>自动照片存储</b>：${{s.save_dir}}（按 日期/PIN 分层）`;
    meta.innerHTML=h;
    if(!_inited){{
      document.getElementById('exp').value=s.params.exposure_us;
      _inited=true;
    }}
  }}).catch(()=>{{state.textContent='● 连接失败';state.style.background='#b5482f';}});
}}

switchTab('manual');
refreshStatus();
setInterval(refreshStatus, 5000);
</script>
</body></html>"""


@app.route("/video_feed")
def video_feed():
    if not hub.running:
        return Response("相机未运行", status=503)
    # no-cache 头：避免浏览器/代理把 MJPEG 流缓存住，导致网页画面"冻在首帧"
    resp = Response(_gen_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")
    resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/api/status")
def api_status():
    return jsonify(hub.status())


@app.route("/api/camera", methods=["POST"])
def api_camera():
    """运行时调整相机调试参数。手动测试面板用：曝光 / 帧率 / 手动拍照延时；
    自动监控面板用：自动拍照延时。仅在自由运行下直接写相机。

    注：增益为固定值(GAIN_DB)，前端不提供调整入口。
    - 曝光：实时写入相机（set_exposure 已关自动曝光、走 ExposureTimeAbs 真实节点）。
    - 帧率：仅改手动默认全局 FPS，不动 AUTO_FPS；同步重算 TOTAL 供自检/状态显示。
    - 手动拍照延时(delay)：改 MANUAL_PRE_CAPTURE_DELAY（仅手动测试/自检；不影响自动模式）。
    - 自动拍照延时(auto_delay)：改 AUTO_PRE_CAPTURE_DELAY（仅自动 PLC 触发/强制拍摄；不影响手动模式）。
    exposure_us / fps / delay / auto_delay 均可独立发送，按需只传需要改的项。
    """
    if not hub.running or not hub.streamers:
        return jsonify({"ok": False, "error": "相机未运行"})
    data = request.get_json(silent=True) or {}
    resp = {"ok": True}
    # 曝光（实时写相机）
    if data.get("exposure_us") is not None:
        resp["exposure"] = hub.streamers[0].set_exposure(data["exposure_us"])
    # 帧率（手动默认全局，封顶 6fps；不影响自动模式 AUTO_FPS）
    if data.get("fps") is not None:
        global FPS, TOTAL
        FPS = max(0.5, min(6.0, float(data["fps"])))
        TOTAL = int(FPS * DURATION_SEC)
        resp["fps"] = round(FPS, 2)
    # 手动拍照延时（仅手动测试/自检；不影响自动模式）
    if data.get("delay") is not None:
        global MANUAL_PRE_CAPTURE_DELAY
        MANUAL_PRE_CAPTURE_DELAY = max(0.0, min(30.0, float(data["delay"])))
        resp["delay"] = round(MANUAL_PRE_CAPTURE_DELAY, 2)
    # 自动拍照延时（仅自动 PLC 触发/强制拍摄；不影响手动模式）
    if data.get("auto_delay") is not None:
        global AUTO_PRE_CAPTURE_DELAY
        AUTO_PRE_CAPTURE_DELAY = max(0.0, min(30.0, float(data["auto_delay"])))
        resp["auto_delay"] = round(AUTO_PRE_CAPTURE_DELAY, 2)
    if set(resp.keys()) <= {"ok"}:
        return jsonify({"ok": False, "error": "未收到任何可调参数（exposure_us/fps/delay/auto_delay）"})
    return jsonify(resp)


@app.route("/api/image")
def api_image():
    """安全提供手动测试照片缩略图：rel 为相对 ROOT 的路径，越界返回 404。"""
    rel = request.args.get("rel", "")
    full = os.path.normpath(os.path.join(ROOT, rel))
    if not full.startswith(ROOT) or not os.path.isfile(full):
        return Response("not found", status=404)
    ext = os.path.splitext(full)[1].lower()
    mt = {"bmp": "image/bmp", "jpg": "image/jpeg", "jpeg": "image/jpeg",
          "png": "image/png"}.get(ext, "application/octet-stream")
    try:
        with open(full, "rb") as f:
            data = f.read()
        return Response(data, mimetype=mt)
    except Exception:
        return Response("read error", status=500)


@app.route("/api/capture", methods=["POST"])
def api_capture():
    data = request.get_json(silent=True) or {}
    print(f"[网页] 收到 /api/capture 请求: {data}")
    if not hub.running:
        print("[网页] /api/capture 拒绝: 相机未运行")
        return jsonify({"ok": False, "error": "相机未运行"})
    test = bool(data.get("test", False))
    fps = data.get("fps")
    if fps is not None:
        try:
            fps = max(1.0, min(6.0, float(fps)))   # 相机封顶 6fps；允许小数(如2.5=2秒5张)
        except (TypeError, ValueError):
            fps = None
    total = data.get("total")
    if total is not None:
        try:
            total = max(1, min(1000, int(total)))   # 后端兜底钳位：防绕过网页写海量张写满磁盘
        except (TypeError, ValueError):
            total = None
    if test:
        # 手动测试模式：与 PLC 无关、不判车型、不入库；照片落 data/manual_test/日期/时间/
        sess = datetime.datetime.now().strftime("%Y-%m-%d/%H%M%S")
        save_dir = os.path.join(MANUAL_TEST_ROOT, sess)
        result = hub.request_capture_primary(fps=fps, total=total,
                                             save_dir=save_dir, prefix="Test",
                                             delay=MANUAL_PRE_CAPTURE_DELAY)
        print(f"[网页] 手动测试拍摄: {result.get('saved')} 张 -> {result.get('save_dir')}")
    else:
        # 强制拍摄（自动模式下补拍/调试）：沿用自动参数 12@2.5fps，不入库
        result = hub.request_capture_primary(fps=AUTO_FPS, total=AUTO_TOTAL,
                                             delay=AUTO_PRE_CAPTURE_DELAY)
        _prune_scratch()
        if result.get("ok"):
            print(f"[网页] 强制拍摄完成: {result.get('saved')} 张 -> {result.get('save_dir')}")
        else:
            print(f"[网页] 强制拍摄失败: {result.get('error')}")
    # 把绝对路径转成相对 ROOT 的安全 rel，供网页 /api/image 读取缩略图
    rel_files = []
    for f in result.get("abs_files", []):
        try:
            rel = os.path.relpath(f, ROOT)
            if not rel.startswith(".."):
                rel_files.append(rel)
        except Exception:
            pass
    result["rel_files"] = rel_files
    return jsonify(result)


def main():
    parser = argparse.ArgumentParser(description="现场相机网页版实时取景+远程触发(+PLC自动触发)")
    parser.add_argument("--host", default=WEB_HOST)
    parser.add_argument("--port", type=int, default=WEB_PORT)
    parser.add_argument("--plc-auto", action="store_true",
                        help="启用 PLC 自动触发（只读监测 DB130.DBX0.1 上升沿→提前锁存DB230→记录全部车→拍照→记录）")
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
    print(f"[网页采集] 连拍：点按钮/出车信号后立即连拍 {st['params']['duration_sec']}秒 "
          f"= {st['params']['total']} 张（{st['params']['fps']}fps，与相机帧率解耦）")
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
                      f"（DB130.DBX0.1 上升沿触发；仅 9X/8X 且 NO_Paint=0 拍照）")
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
