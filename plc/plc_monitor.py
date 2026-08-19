# -*- coding: utf-8 -*-
"""PLC 只读监控（生产安全）——供 web_capture.py 自动触发等复用。

设计要点：
  - 全程仅 read_area，绝不 write_area / db_write / plc_stop / download。
  - 触发策略：以【DB130.DBX0.1 出车信号上升沿(0→1)】为权威触发源（现场确认）。
  - 新增车身同步条件：出车信号仅“预约”拍照，须再等【DB890.DBX225.2 与 DBX225.3
    同时为 1】才真正触发（雪橇到达相机视野位置）。解决“车身同步性不好”问题。
    同步信号通常在出车信号后约 2s 才置 1，故用窗口等待(SYNC_WINDOW_SEC)，而非
    要求上升沿那一刻就为 1（否则会全部漏拍）。
  - 关键矛盾：PLC 收到出车信号会【立即清零 DB230】，所以上升沿那一刻去读 DB230 必然为空。
    解决：后台线程持续采样 DB230（200ms），仅在非空时更新 self.latched（提前锁存）；
    上升沿到来时，用【此前锁存的】DB230 上下文作为回调 ctx，而非当前实时去读。
  - 去重：同一车的多个抖动脉冲用 _last_car_key 去重，避免重复拍照。

ctx 结构（dict）：
  {"no_paint": int(0/1), "skid": int, "model_int": int,
   "model": str(ASCII车型如 'MM**'), "pin": str}
  若从未采样到非空上下文，回调收到的 ctx 为 None。
"""
import time
import threading

try:
    import snap7
    try:
        from snap7.types import S7AreaDB
    except Exception:
        S7AreaDB = 0x84
    HAS_SNAP7 = True
except ImportError:
    HAS_SNAP7 = False


# ===================== PLC 连接参数（按现场资料）=====================
PLC_IP = "172.30.173.6"
PLC_RACK = 0
PLC_SLOT = 2

# 只读范围（严格限定在你给的地址内）
DB130_SIG = (130, 0, 1)      # DB130 字节0（含 DBX0.1 出车信号）
DB230_A = (230, 1208, 30)    # DB230 字节1208..1237（车型/滑橇/PIN）
DB230_B = (230, 1257, 1)     # DB230 字节1257（含 DBX1257.1 NO_Paint）
DB890_SYN = (890, 225, 1)    # DB890 字节225（含 DBX225.2 / DBX225.3 车身同步信号）

POLL_MS = 10     # 信号轮询间隔（毫秒）
CTX_MS = 200     # DB230 上下文采样间隔（毫秒）
SYNC_WINDOW_SEC = 8.0   # 出车信号后，等待两个同步信号(DB890.225.2/225.3)同时为 1 的最长窗口（秒）；超时则跳过该车


def parse_context(buf_a, buf_b):
    """解析 DB230 上下文。buf_a 起点1208 长30；buf_b 为字节1257。

    返回 dict{no_paint, skid, model_int, model, pin}。
    车型 DBD1208 实为 ASCII（如 0x37503234='7P24' / 0x4D4D2A2A='MM**'）。
    """
    def i(off):
        return off - 1208
    model_int = int.from_bytes(buf_a[i(1208):i(1208) + 4], "big")     # DBD1208
    try:
        model_ascii = buf_a[i(1208):i(1208) + 4].decode("ascii").strip("\x00")
    except Exception:
        model_ascii = ""
    skid = int.from_bytes(buf_a[i(1218):i(1218) + 2], "big")          # DBW1218
    pin = buf_a[i(1224):i(1224) + 14].decode("latin-1", errors="replace")  # DBB1224..1237
    no_paint = (buf_b[0] >> 1) & 1                                   # DBX1257.1
    return {
        "no_paint": no_paint,
        "skid": skid,
        "model_int": model_int,
        "model": model_ascii,
        "pin": pin,
    }


class PlcMonitor:
    """只读 PLC 监控：锁存车号上下文 + 检测出车信号上升沿。"""

    def __init__(self, ip=PLC_IP, rack=PLC_RACK, slot=PLC_SLOT,
                 poll_ms=POLL_MS, ctx_ms=CTX_MS):
        self.ip = ip
        self.rack = rack
        self.slot = slot
        self.poll_ms = poll_ms
        self.ctx_ms = ctx_ms
        self.client = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.latched = None          # 最近一次“非空”的 DB230 上下文
        self._last_car_key = None    # 已触发过的车标识(skid,pin,model)，用于新车去重触发
        self._outcar_pending = False # 出车信号已收到、等待车身同步信号(DB890.225.2/3)期间为 True
        self._outcar_ctx = None      # 等待期间锁存的车上下文（出车信号上升沿那一刻）
        self._outcar_t = 0.0         # 出车信号上升沿时刻（perf_counter），用于窗口超时
        self._thread = None
        self.connected = False       # 健康态：供 UI/日志观测（断线重连期间为 False）

    # ---------- 连接（仅读） ----------
    def connect(self):
        if not HAS_SNAP7:
            raise RuntimeError("未安装 python-snap7，请执行：pip install python-snap7")
        self.client = snap7.client.Client()
        self.client.connect(self.ip, self.rack, self.slot)
        if not self.client.get_connected():
            raise RuntimeError(
                f"PLC 连接失败（不可达 / GET-PUT 未开 / 地址错）：{self.ip}")
        return True

    def disconnect(self):
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        try:
            if self.client:
                self.client.disconnect()
        except Exception:
            pass

    # ---------- 后台线程 ----------
    def start(self, on_rising):
        """启动监控线程。on_rising(ctx|None) 在每次上升沿被调用。"""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, args=(on_rising,), daemon=True)
        self._thread.start()

    def _maybe_latch(self, ctx):
        # 仅在非空时更新锁存（跳过 PLC 清零后的空行）
        if ctx["skid"] != 0 or ctx["pin"].strip("\x00") or ctx["model"]:
            with self._lock:
                self.latched = ctx

    def _loop(self, on_rising):
        poll_s = self.poll_ms / 1000.0
        ctx_s = self.ctx_ms / 1000.0
        prev = 0
        last_ctx_t = 0.0
        while not self._stop.is_set():
            # 读信号（失败则重连；重连期间本线程阻塞，但不影响相机/Flask）
            try:
                d = self.client.read_area(S7AreaDB, *DB130_SIG)   # 仅读 1 字节
                bit = (d[0] >> 1) & 1
                self.connected = True
            except Exception as e:
                self.connected = False
                print(f"[PLC] 读信号失败: {e}")
                if not self._reconnect():
                    break
                self._outcar_pending = False   # 重连后丢弃尚未触发的等待
                continue

            now = time.perf_counter()

            # 车身同步信号：DB890.DBX225.2 与 DBX225.3 须同时为 1（雪橇到达相机视野位置）。
            # 与出车信号同周期读取，确保上升沿后能在本轮询窗口内及时检测到。
            # 读取失败→当次视为未同步（保守跳过），不触发重连（与信号读失败区分）。
            sync_ok = False
            try:
                db890 = bytes(self.client.read_area(S7AreaDB, *DB890_SYN))
                sync_ok = ((db890[0] >> 2) & 1) == 1 and ((db890[0] >> 3) & 1) == 1
            except Exception as e:
                print(f"[PLC] 读同步信号(DB890.DBX225.2/225.3)失败: {e}")

            # 持续采样 DB230：仅在非空时更新锁存（静默，不打印）
            if now - last_ctx_t >= ctx_s:
                last_ctx_t = now
                try:
                    a = bytes(self.client.read_area(S7AreaDB, *DB230_A))
                    b = bytes(self.client.read_area(S7AreaDB, *DB230_B))
                    self._maybe_latch(parse_context(a, b))
                except Exception as e:
                    print(f"[PLC] 读上下文失败: {e}")
                    # 上下文采样失败不致命，继续轮询，下次再采

            # 触发策略（现场确认）：DB130.DBX0.1 出车信号上升沿(0→1)为权威触发源。
            # 【新增同步条件】出车信号仅“预约”一次拍照；真正拍照须等车身同步信号
            # (DB890.DBX225.2/225.3) 同时为 1（雪橇运行到相机视野位置）。这解决“车身
            # 同步性不好”问题——只有车身到位才拍，而不是一出车信号就拍。
            # 同步信号通常在出车信号后约 2s 才置 1（雪橇运行到下一传感器），故用窗口
            # 等待（SYNC_WINDOW_SEC），而非要求上升沿那一刻就为 1（那样会全部漏拍）。
            # 上升沿到来时 PLC 已清零 DB230，故用【此前锁存】的 DB230 上下文，
            # 不在此刻去读实时 DB230（会读空）。用 _last_car_key 去重防抖动脉冲。
            if bit == 1 and prev == 0:
                with self._lock:
                    lat = self.latched
                if lat is not None:
                    key = (lat["skid"], lat["pin"], lat["model"])
                    if key != self._last_car_key:
                        self._last_car_key = key
                        self._outcar_pending = True
                        self._outcar_ctx = dict(lat)   # 提前锁存，先于被下一台车覆盖
                        self._outcar_t = now
                        print(f"[PLC] 出车信号上升沿，预约拍照（等待同步信号 DB890.225.2/3）"
                              f" skid={lat['skid']} model={lat['model']!r}")

            # 已预约：等待同步信号到位（或窗口超时放弃），再触发拍照回调。
            if self._outcar_pending:
                if sync_ok:
                    self._outcar_pending = False
                    snap = self._outcar_ctx
                    self._outcar_ctx = None
                    try:
                        on_rising(snap)
                    except Exception as e:
                        print(f"[PLC] 回调异常: {e}")
                elif now - self._outcar_t > SYNC_WINDOW_SEC:
                    skid = self._outcar_ctx.get("skid") if self._outcar_ctx else "?"
                    model = self._outcar_ctx.get("model") if self._outcar_ctx else "?"
                    self._outcar_pending = False
                    self._outcar_ctx = None
                    print(f"[PLC] 出车信号后 {SYNC_WINDOW_SEC:.0f}s 内同步信号(DB890.225.2/3)"
                          f"未满足，跳过拍照 skid={skid} model={model!r}")

            prev = bit
            time.sleep(poll_s)

    def _reconnect(self):
        """PLC 断线后指数退避重连，直到连上或收到停止信号。返回是否恢复。

        仅在监控线程内调用，阻塞只影响本线程，绝不波及相机取流 / Flask。
        """
        attempt = 0
        while not self._stop.is_set():
            attempt += 1
            try:
                try:
                    self.client.disconnect()
                except Exception:
                    pass
                self.client.connect(self.ip, self.rack, self.slot)
                if self.client.get_connected():
                    print("[PLC] 已重连恢复")
                    self.connected = True
                    return True
            except Exception as e:
                if attempt % 10 == 0:
                    print(f"[PLC] 持续重连中(第{attempt}次): {e}")
            time.sleep(min(15, 1 + attempt * 0.5))   # 退避封顶 15s
        return False


def has_snap7():
    return HAS_SNAP7
