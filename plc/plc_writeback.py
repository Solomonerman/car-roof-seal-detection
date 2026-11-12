# -*- coding: utf-8 -*-
"""PLC 检测结果回写模块（生产安全）。

⚠️ 红线：DB 写地址【尚未向现场确认】，总开关默认关闭（ENABLED=False）。
   当前状态：接口就绪、代码可测，但不会向 PLC 写任何数据。
   启用条件：用户明确授权 + 电气确认写地址 + 非生产时段单独评审。

信号设计（用户 2026-11-12 拍板）：
  ① OK              → 放行（SIGNAL_PASS=1）
  ② NG              → 拦截报警（SIGNAL_BLOCK=0）
  ③ 未检测到胶条     → 放行（按 OK；图上黄字 + loc_ok=False 留痕人工复核）
  ④ 检测异常         → 拦截报警（安全侧兜底）

  合并后 PLC 信号为二值：1=放行 / 0=拦截报警。

  ⚠️ 实现层坑：④"检测异常"时程序可能崩溃走不到回写这一步。
  PLC 侧须定"超时未收到信号 = 默认拦截"兜底规则（需电气配合）。

调用时机：web_capture.py handle_car_signal 自动链路、真实检测完成后调用 send_result()。
手动拍摄 / 免检车 / 车型未接入 → 不回写（本模块不处理）。

待确认项（电气）：
  1. 写回用哪个 DB 块 / 字节偏移？（DB_WRITE_DB / DB_WRITE_OFFSET）
  2. 是否需要回写滑橇号/PIN 以便 PLC 对应来车？
  3. 握手方式：写完是否需要置位/清零？超时拦截的 PLC 侧逻辑？
"""

import time

# ===================== 总开关与地址（红线：未确认不写）=====================
ENABLED = False           # 总开关：用户明确授权后改 True
DB_WRITE_DB = None        # 写入的 DB 块号（如 130），待电气确认
DB_WRITE_OFFSET = None    # 写入的字节偏移（如 10），待电气确认

# ===================== PLC 连接参数（与 plc_monitor 一致）=====================
PLC_IP = "172.30.173.6"
PLC_RACK = 0
PLC_SLOT = 2

# ===================== 信号值定义 =====================
SIGNAL_PASS = 1           # 放行
SIGNAL_BLOCK = 0          # 拦截报警

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
      case 4（检测异常）→ 拦截：defects 含 label="pending" 且 meta.reason 含"检测异常"
      case 2（NG）      → 拦截：det.ok == False
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
        return (0, 0, False, "det=None-不回写")

    defects = det.defects or []
    labels = [getattr(d, "label", "") for d in defects]

    # ---- 先排除不该写的情况（pending 类细分）----
    if "pending" in labels:
        reasons = [getattr(d, "meta", None) or {} for d in defects
                   if getattr(d, "label", "") == "pending"]
        reason = str(reasons[0].get("reason", "")) if reasons else ""
        if "检测异常" in reason:
            return (4, SIGNAL_BLOCK, True, f"检测异常-拦截: {reason}")
        if "无照片" in reason:
            return (0, 0, False, "无照片-不回写")
        if "算法未接入" in reason:
            return (0, 0, False, "算法未接入-不回写")
        # 其他未知 pending → 安全侧拦截
        return (4, SIGNAL_BLOCK, True, f"未知异常-拦截: {reason}")

    # ---- 正常检测结果的 3 种 ----
    if not det.ok:
        return (2, SIGNAL_BLOCK, True, "NG-拦截报警")

    if "loc_warn" in labels:
        return (3, SIGNAL_PASS, True, "未检测到胶条-放行(人工复核)")

    return (1, SIGNAL_PASS, True, "OK-放行")


def send_result(det, skid=None, pin=""):
    """检测结果回写 PLC 入口。

    参数:
        det: DetectionResult（web_capture 的检测结果）
        skid: 滑橇号（预留，待确认是否需回写）
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

    if not HAS_SNAP7:
        print(f"[PLC回写] snap7 未安装，无法写 PLC(case={case_id})")
        return (False, "snap7 未安装")

    if DB_WRITE_DB is None or DB_WRITE_OFFSET is None:
        print(f"[PLC回写] ⚠️ 写地址未确认(DB_WRITE_DB/DB_WRITE_OFFSET=None)，拒写(case={case_id})")
        return (False, "写地址未确认")

    # ---- snap7 写入 ----
    client = None
    try:
        client = snap7.client.Client()
        client.connect(PLC_IP, PLC_RACK, PLC_SLOT)
        # 写 1 字节信号值到指定 DB 地址
        buf = bytearray([signal & 0xFF])
        client.db_write(DB_WRITE_DB, DB_WRITE_OFFSET, buf)
        print(f"[PLC回写] ✅ 已写 DB{DB_WRITE_DB}.DBB{DB_WRITE_OFFSET}={signal} "
              f"(case={case_id} {desc} skid={skid} pin={pin})")
        return (True, f"case={case_id} signal={signal} {desc}")
    except Exception as e:
        print(f"[PLC回写] ❌ PLC 写入失败(case={case_id}): {e}")
        return (False, f"PLC写入失败: {e}")
    finally:
        if client is not None:
            try:
                client.disconnect()
            except Exception:
                pass
