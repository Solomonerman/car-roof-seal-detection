# -*- coding: utf-8 -*-
"""PLC 检测结果回写模块（生产安全）。

写地址（电气确认 2026-11-12）：DB890.DBX231.7（单个 bit）
信号值：1=报警 / 0=放行

信号设计（用户 2026-11-12 拍板）：
  ① OK              → 放行（bit=0）
  ② NG              → 报警（bit=1）
  ③ 未检测到胶条     → 放行（按 OK；图上黄字 + loc_ok=False 留痕人工复核）
  ④ 检测异常         → 报警（安全侧兜底）

  ⚠️ 实现层坑：程序崩溃时走不到回写。
  兜底方案：① handle_car_signal 外层 try/except 里 send_alarm()；
            ② PLC 侧建议做"超时未收到信号=报警"双保险（需电气配合）。

调用时机：web_capture.py handle_car_signal 自动链路、真实检测完成后调用 send_result()。
手动拍摄 / 免检车 / 车型未接入 → 不回写（本模块不处理）。

UI 开关（2026-11-13）：data/plc_writeback_config.json {"enabled": true/false}，
8000 看板提供开关按钮，关=始终写 0（放行），开=正常写检测结果信号。
"""

import json as _json
import os as _os

# ===================== 总开关与地址 =====================
ENABLED = True            # 已启用（电气确认地址 2026-11-12）
DB_WRITE_DB = 890         # DB890
DB_WRITE_BYTE = 231       # 字节偏移 231
DB_WRITE_BIT = 7          # bit 7（DBX231.7）

# ===================== PLC 连接参数（与 plc_monitor 一致）=====================
PLC_IP = "172.30.173.6"
PLC_RACK = 0
PLC_SLOT = 2

# ===================== 信号值定义（用户指定）=====================
SIGNAL_PASS = 0           # 放行
SIGNAL_BLOCK = 1          # 报警

# ===================== UI 开关配置文件 =====================
_CONFIG_PATH = _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    "data", "plc_writeback_config.json")


def _read_toggle():
    """读取 UI 开关状态：True=开(正常发信号) / False=关(始终发0放行)。默认 True。"""
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = _json.load(f)
        return bool(cfg.get("enabled", True))
    except Exception:
        return True   # 文件不存在/读不了 → 默认开（安全侧：正常发信号）

# snap7 可用性检测（沙箱/无 snap7 环境下安全降级）
try:
    import snap7
    try:
        from snap7.types import S7AreaDB
    except Exception:
        S7AreaDB = 0x84
    HAS_SNAP7 = True
except ImportError:
    HAS_SNAP7 = False


def classify_result(det):
    """把 DetectionResult 分成 4 种情况，返回 (case_id, signal, should_write, desc)。

    分类规则：
      case 4（检测异常）→ 报警：defects 含 label="pending" 且 meta.reason 含"检测异常"
      case 2（NG）      → 报警：det.ok == False
      case 3（未检测到）→ 放行：det.ok==True 且含 label="loc_warn"
      case 1（OK）      → 放行：其余

    特殊情况（should_write=False，不回写）：
      - "无照片·未检测"：相机没出图，无可信检测结果
      - "算法未接入"：该车型没接检测器（如 9X），仅拍照存档

    参数:
        det: DetectionResult 或 None
    返回:
        (case_id, signal_value, should_write, description)
    """
    if det is None:
        return (0, SIGNAL_PASS, False, "det=None-不回写")

    defects = det.defects or []
    labels = [getattr(d, "label", "") for d in defects]

    # ---- 先排除不该写的情况（pending 类细分）----
    if "pending" in labels:
        reasons = [getattr(d, "meta", None) or {} for d in defects
                   if getattr(d, "label", "") == "pending"]
        reason = str(reasons[0].get("reason", "")) if reasons else ""
        if "检测异常" in reason:
            return (4, SIGNAL_BLOCK, True, f"检测异常-报警: {reason}")
        if "无照片" in reason:
            return (0, SIGNAL_PASS, False, "无照片-不回写")
        if "算法未接入" in reason:
            return (0, SIGNAL_PASS, False, "算法未接入-不回写")
        # 其他未知 pending → 安全侧报警
        return (4, SIGNAL_BLOCK, True, f"未知异常-报警: {reason}")

    # ---- 正常检测结果的 3 种 ----
    if not det.ok:
        return (2, SIGNAL_BLOCK, True, "NG-报警拦截")

    if "loc_warn" in labels:
        return (3, SIGNAL_PASS, True, "未检测到胶条-放行(人工复核)")

    return (1, SIGNAL_PASS, True, "OK-放行")


def _write_bit(signal):
    """把 signal(0/1) 写到 DB890.DBX231.7（读-改-写 1 字节）。

    UI 开关关时：始终写 0（放行），不发报警（覆盖 send_result 和 send_alarm）。
    返回 (success, detail)。
    """
    # UI 开关：关=始终发 0（放行）
    if not _read_toggle():
        signal = SIGNAL_PASS
    if not HAS_SNAP7:
        return (False, "snap7 未安装")
    client = None
    try:
        client = snap7.client.Client()
        client.connect(PLC_IP, PLC_RACK, PLC_SLOT)
        # 读-改-写：先读整字节 231，改 bit7，再写回
        buf = client.read_area(S7AreaDB, DB_WRITE_DB, DB_WRITE_BYTE, 1)
        val = buf[0]
        if signal:  # 报警=1：置位 bit7
            val |= (1 << DB_WRITE_BIT)
        else:       # 放行=0：清零 bit7
            val &= ~(1 << DB_WRITE_BIT) & 0xFF
        buf[0] = val
        client.write_area(S7AreaDB, DB_WRITE_DB, DB_WRITE_BYTE, buf)
        return (True, f"DB{DB_WRITE_DB}.DBX{DB_WRITE_BYTE}.{DB_WRITE_BIT}={signal}")
    except Exception as e:
        return (False, f"PLC写入失败: {e}")
    finally:
        if client is not None:
            try:
                client.disconnect()
            except Exception:
                pass


def send_result(det, skid=None, pin=""):
    """检测结果回写 PLC 入口。

    参数:
        det: DetectionResult（web_capture 的检测结果）
        skid: 滑橇号（预留）
        pin: PIN 码（预留）

    返回:
        (success: bool, detail: str)
    """
    case_id, signal, should_write, desc = classify_result(det)

    if not should_write:
        print(f"[PLC回写] 跳过: {desc}")
        return (False, f"跳过: {desc}")

    if not ENABLED:
        print(f"[PLC回写] 总开关未开启(case={case_id} {desc})，不写 PLC")
        return (False, "总开关未开启")

    ok, detail = _write_bit(signal)
    if ok:
        print(f"[PLC回写] ✅ {detail} (case={case_id} {desc} skid={skid} pin={pin})")
    else:
        print(f"[PLC回写] ❌ {detail} (case={case_id})")
    return (ok, f"case={case_id} signal={signal} {desc} | {detail}")


def send_alarm(reason="程序异常兜底"):
    """紧急报警：强制写报警信号(bit=1)。

    用于程序崩溃/异常时的兜底——在 try/except 里调用，
    确保即使主流程出错也能给 PLC 一个报警信号。
    PLC 侧建议配合"超时未收到=报警"双保险。
    """
    if not ENABLED:
        print(f"[PLC回写] send_alarm: 总开关未开启，跳过 ({reason})")
        return (False, "总开关未开启")
    ok, detail = _write_bit(SIGNAL_BLOCK)
    if ok:
        print(f"[PLC回写] 🚨 报警已发: {detail} ({reason})")
    else:
        print(f"[PLC回写] 🚨 报警发送失败: {detail} ({reason})")
    return (ok, detail)
