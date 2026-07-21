# -*- coding: utf-8 -*-
"""现场相机 · 网页版实时取景 + 远程触发采集。

为什么用这个：
  你在工位远程操作上位机，看不到现场车子。pylon Viewer 的实时窗口在
  上位机本地，你隔着远程桌面才能看到，且它独占相机、会和本脚本冲突。
  这个工具把相机的实时画面通过浏览器（局域网/公网）推给你，你在网页上
  就能看到车有没有到位，点一下按钮就触发连拍，无需远程桌面、无需 pylon Viewer。

用法：
  1) 上位机安装依赖： pip install pypylon opencv-python flask
  2) 关闭 pylon Viewer（避免相机被占用）
  3) python capture/web_capture.py
  4) 在本机或任意能访问上位机 IP 的电脑浏览器打开：
       http://<上位机IP>:5000
  5) 网页里看实时画面，点【连拍一轮】即触发；拍完结果直接落入 data/raw_images/

连拍 / 相机参数都集中在顶部，按现场情况调整。

依赖：Flask（网页服务） + pypylon（相机） + opencv（图像）
"""
import os
import sys
import time
import datetime
import threading
import argparse

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

# ===================== 相机连接参数 =====================
CAMERA_IP = "172.30.173.249"   # 现场左侧 Basler aca1920-48gm
CAMERA_SERIAL = ""             # 留空则用 IP；也可填序列号直连
# 两个都空 → 自动连第一台可用 Basler 相机

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
SAVE_DIR = os.path.join(ROOT, "data", "raw_images")
SAVE_EXT = ".bmp"            # BMP 无损，适合算法检测；也可改 ".jpg"

# ===================== 预览参数 =====================
STREAM_WIDTH = 960           # 网页实时流宽度（等比缩放，不改原始采集分辨率）
JPEG_QUALITY = 80            # 实时流 JPEG 质量（1-100，越小越流畅但越糊）

# ================================================================


class CameraStreamer:
    """后台线程：持续取流 → 存最新帧；按需执行连拍。"""

    def __init__(self):
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

        self._thread = None
        self._error = None

    # ---------- 连接与配置 ----------
    def connect(self):
        if not HAS_PYPYLON:
            raise RuntimeError("未安装 pypylon，请先在上位机执行：pip install pypylon")

        factory = py.TlFactory.GetInstance()
        if CAMERA_IP:
            info = py.DeviceInfo()
            info.SetPropertyValue("IpAddress", CAMERA_IP)
            try:
                self.cam = py.InstantCamera(factory.CreateDevice(info))
            except Exception:
                raise RuntimeError(
                    f"无法连接 IP={CAMERA_IP} 的相机。检查：通电/网线/同网段/防火墙/"
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
            "ip": CAMERA_IP or (self.cam.GetDeviceInfo().GetIpAddress()
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
        # 像素格式放最后
        try:
            nodemap.GetNode("PixelFormat").SetValue(PIXEL_FORMAT)
            conf_log.append(f"像素格式={PIXEL_FORMAT}")
        except Exception as e:
            conf_log.append(f"像素格式设置异常(沿用当前值): {e}")
        self._conf_log = conf_log

    def start(self):
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
        """持续取流，维护最新帧；检测连拍事件。"""
        frame_interval = 1.0 / FPS
        while self.running:
            try:
                grab = self.cam.RetrieveResult(200, py.TimeoutHandling_Return)
            except Exception:
                time.sleep(0.05)
                continue
            if grab and grab.GrabSucceeded():
                frame = grab.Array.copy()
                grab.Release()
                with self._lock:
                    self._latest = frame
                # 处理连拍请求（在取到帧后顺手处理）
                if self._capture_req.is_set():
                    self._do_burst(frame_interval)
            elif grab:
                grab.Release()

    def _do_burst(self, frame_interval):
        """执行一轮连拍。"""
        self._capture_req.clear()
        total = int(FPS * DURATION_SEC)
        os.makedirs(SAVE_DIR, exist_ok=True)
        batch = []
        t0 = time.perf_counter()
        for i in range(total):
            t_start = time.perf_counter()
            try:
                grab = self.cam.RetrieveResult(3000, py.TimeoutHandling_ThrowException)
            except Exception:
                break
            if not grab.GrabSucceeded():
                grab.Release()
                continue
            img_save = grab.Array.copy()
            grab.Release()
            fname = self._format_filename()
            fpath = os.path.join(SAVE_DIR, fname)
            cv2.imwrite(fpath, img_save)
            batch.append(os.path.basename(fpath))
            elapsed = time.perf_counter() - t_start
            wait = max(0, int((frame_interval - elapsed) * 1000))
            if wait > 0 and i < total - 1:
                time.sleep(wait / 1000.0)
        total_elapsed = time.perf_counter() - t0
        actual_fps = len(batch) / total_elapsed if total_elapsed > 0 else 0
        self._last_result = {
            "ok": True,
            "saved": len(batch),
            "total": total,
            "elapsed": round(total_elapsed, 1),
            "actual_fps": round(actual_fps, 1),
            "files": batch,
            "ts": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._capture_done.set()

    def _format_filename(self, prefix="Image"):
        ts = datetime.datetime.now().strftime("%Y-%m-%d__%H-%M-%S-%f")[:-3]
        return f"{prefix}__{ts}{SAVE_EXT}"

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
            "save_dir": SAVE_DIR,
            "last_result": self._last_result,
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


# ===================== Flask 应用 =====================
app = Flask(__name__)
streamer = CameraStreamer()


def _gen_mjpeg():
    """MJPEG 流生成器。"""
    while streamer.running:
        jpg = streamer.get_latest_jpeg()
        if jpg:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
        else:
            time.sleep(0.05)


@app.route("/")
def index():
    st = streamer.status()
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
 .wrap{{padding:16px;max-width:1000px;margin:0 auto}}
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
</style></head>
<body>
<header><h1>车顶密封条相机 · 实时取景</h1>
  <span class="badge" id="state">连接中…</span></header>
<div class="wrap">
  <div class="video"><img id="feed" src="/video_feed"></div>
  <div class="bar">
    <button id="cap" onclick="capture()">📸 连拍一轮（{int(FPS*DURATION_SEC)} 张）</button>
    <span id="capState" class="meta"></span>
  </div>
  <div class="meta" id="meta"></div>
  <h3 style="font-size:14px;margin:18px 0 6px">采集日志</h3>
  <div id="log">等待操作…</div>
</div>
<script>
const meta=document.getElementById('meta');
const log=document.getElementById('log');
const state=document.getElementById('state');
const capBtn=document.getElementById('cap');
const capState=document.getElementById('capState');

function refreshStatus(){{
  fetch('/api/status').then(r=>r.json()).then(s=>{{
    if(s.running){{state.textContent='● 在线';state.style.background='#2d8c4a';}}
    else{{state.textContent='● 离线';state.style.background='#b5482f';}}
    let h=`<b>相机</b>：${{s.camera.model}}（序列号 ${{s.camera.serial}}）<br>`;
    h+=`<b>分辨率</b>：${{s.resolution}} · ${{s.color}} · ${{s.pixel_format}}<br>`;
    h+=`<b>连拍</b>：${{s.params.fps}} 张/秒 × ${{s.params.duration_sec}} 秒 = <b>${{s.params.total}} 张</b><br>`;
    h+=`<b>曝光</b>：${{s.params.exposure_us}} µs · <b>增益</b>：${{s.params.gain_display}} · Gamma：${{s.params.gamma}}<br>`;
    h+=`<b>保存目录</b>：${{s.save_dir}}`;
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
    if not streamer.running:
        return Response("相机未运行", status=503)
    return Response(_gen_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/status")
def api_status():
    return jsonify(streamer.status())


@app.route("/api/capture", methods=["POST"])
def api_capture():
    if not streamer.running:
        return jsonify({"ok": False, "error": "相机未运行"})
    result = streamer.request_capture()
    return jsonify(result)


def main():
    parser = argparse.ArgumentParser(description="现场相机网页版实时取景+远程触发")
    parser.add_argument("--host", default=WEB_HOST)
    parser.add_argument("--port", type=int, default=WEB_PORT)
    parser.add_argument("--no-browser-note", action="store_true")
    args = parser.parse_args()

    if not HAS_PYPYLON or not HAS_FLASK:
        print("[错误] 缺少依赖，请在上位机执行：")
        print("  pip install pypylon opencv-python flask")
        sys.exit(1)

    print("=" * 55)
    print("[网页采集] 正在连接 Basler 相机 ...")
    try:
        streamer.connect()
    except Exception as e:
        print(f"[错误] {e}")
        sys.exit(1)

    try:
        streamer.start()
    except Exception as e:
        print(f"[错误] {e}")
        streamer.stop()
        sys.exit(1)

    st = streamer.status()
    print(f"[网页采集] 已连接：{st['camera']['model']}  IP={st['camera']['ip']}")
    for c in st["config"]:
        print(f"  - {c}")
    print(f"[网页采集] 分辨率={st['resolution']}  色彩={st['color']}  像素格式={st['pixel_format']}")
    print(f"[网页采集] 连拍参数：{st['params']['fps']}张/秒 × {st['params']['duration_sec']}秒 "
          f"= {st['params']['total']} 张")
    print(f"[网页采集] 保存目录：{SAVE_DIR}")
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
        streamer.stop()
        print("\n[网页采集] 已停止，相机已释放")


if __name__ == "__main__":
    main()
