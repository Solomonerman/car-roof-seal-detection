# -*- coding: utf-8 -*-
"""PLC 只读监控（生产安全）——供 web_capture.py 自动触发等复用。

设计要点：
  - 全程仅 read_area，绝不 write_area / db_write / plc_stop / download。
  - 监测 DB130.DBX0.1（输送出车信号）上升沿；
  - 因 PLC 在出车信号后清零 DB230，故采用【先锁存、后回调】：
    后台线程持续采样 DB230，仅在非空时更新 self.latched；
    上升沿优先再读一次（数据可能尚未清零），为空则回退已锁存值，
    再把锁存的车号上下文通过 on_rising(ctx) 回调出去。

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
DB230_B = (230, 1257, 1)     # DB230 字节1257（含 DBX1257.0 NO_Paint）

POLL_MS = 10     # 信号轮询间隔（毫秒）
CTX_MS = 200     # DB230 上下文采样间隔（毫秒）


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
    no_paint = buf_b[0] & 1                                         # DBX1257.0
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
                continue

            now = time.perf_counter()

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

            # 上升沿：优先再读一次（数据可能尚未清零），为空则回退锁存值
            if bit == 1 and prev == 0:
                try:
                    a = bytes(self.client.read_area(S7AreaDB, *DB230_A))
                    b = bytes(self.client.read_area(S7AreaDB, *DB230_B))
                    self._maybe_latch(parse_context(a, b))
                except Exception:
                    pass
                with self._lock:
                    snap = dict(self.latched) if self.latched else None
                try:
                    on_rising(snap)
                except Exception as e:
                    print(f"[PLC] 回调异常: {e}")

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
