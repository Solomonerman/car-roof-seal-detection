# -*- coding: utf-8 -*-
"""PLC 只读连接测试（S7-300 319F）。

═══════════════════════════════════════════════════════════════════
⚠️  安全红线（生产环境务必遵守）：
    本脚本【只读取】，绝不写入、绝不强制输出、绝不停机/复位/下载。
    全程唯一的 PLC 交互是 client.read_area(...)。
    任何 write_area / db_write / plc_stop / download 都【不存在于本文件】。
    仅用于验证：能否连上 PLC，并读到已知信号。
═══════════════════════════════════════════════════════════════════

按现场资料读取（均为只读）：
  - DB130.DBX0.1      BOOL    输送出车信号（相机触发用）
  - DB230.DBX1257.1   BOOL    NO Paint（是否需要检测）
  - DB230.DBW1218     INT     滑橇号
  - DB230.DBD1208     DWORD   车型代码
  - DB230.DBB1224..1237  CHAR[14]  PIN 码
"""
import sys
import argparse

try:
    import snap7
    try:
        from snap7.types import S7AreaDB
    except Exception:
        S7AreaDB = 0x84  # S7 Area DB 标准值
    HAS_SNAP7 = True
except ImportError:
    HAS_SNAP7 = False


# ===================== PLC 连接参数（按现场资料）=====================
PLC_IP = "172.30.173.6"
PLC_RACK = 0
PLC_SLOT = 2          # S7-300 CPU 槽号


# ===================== 只读信号定义 =====================
# 只读取你明确给出的地址范围，绝不越界。
READ_DB130 = (130, 0, 1)          # DB130 字节0（含 DBX0.1）
READ_DB230_A = (230, 1208, 30)    # DB230 字节1208..1237（车型/滑橇/PIN）
READ_DB230_B = (230, 1257, 1)     # DB230 字节1257（含 DBX1257.1）


def parse_db230_a(buf):
    """buf 起始于字节1208，长度30。解析 车型/滑橇/PIN。"""
    def i(off):
        return off - 1208
    model = int.from_bytes(buf[i(1208):i(1208) + 4], "big")      # DBD1208
    skid = int.from_bytes(buf[i(1218):i(1218) + 2], "big")       # DBW1218
    pin = buf[i(1224):i(1224) + 14].decode("latin-1", errors="replace")  # DBB1224..1237
    return model, skid, pin


def main():
    if not HAS_SNAP7:
        print("[错误] 未安装 python-snap7，请在上位机执行：pip install python-snap7")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="PLC 只读连接测试（S7-300）")
    parser.add_argument("--ip", default=PLC_IP)
    parser.add_argument("--rack", type=int, default=PLC_RACK)
    parser.add_argument("--slot", type=int, default=PLC_SLOT)
    args = parser.parse_args()

    print("=" * 55)
    print("⚠️  只读测试：仅读取 PLC 信号，不进行任何写/停机操作")
    print(f"    目标 PLC : {args.ip}  rack={args.rack} slot={args.slot}")
    print("=" * 55)

    client = snap7.client.Client()
    try:
        print(f"[连接] 正在连接 {args.ip} ...")
        # 仅建立连接，不读不写任何控制
        client.connect(args.ip, args.rack, args.slot)
        if not client.get_connected():
            print("[错误] 连接失败（PLC 不可达 / GET-PUT 未开 / 地址错）")
            sys.exit(1)
        print("[连接] 成功 ✅")

        # ---------- 只读：DB130 ----------
        db, start, size = READ_DB130
        data130 = client.read_area(S7AreaDB, db, start, size)
        conveyor = (data130[0] >> 1) & 1   # DBX0.1

        # ---------- 只读：DB230（两段，均在你给的范围内）----------
        db, start, size = READ_DB230_A
        data230a = client.read_area(S7AreaDB, db, start, size)
        model, skid, pin = parse_db230_a(bytes(data230a))

        db, start, size = READ_DB230_B
        data230b = client.read_area(S7AreaDB, db, start, size)
        no_paint = (data230b[0] >> 1) & 1   # DBX1257.1

        print("-" * 55)
        print("读取结果（只读，未改动 PLC）：")
        print(f"  输送出车信号 DB130.DBX0.1      : {bool(conveyor)}")
        print(f"  NO Paint     DB230.DBX1257.1   : {bool(no_paint)}")
        print(f"  滑橇号       DB230.DBW1218     : {skid}")
        print(f"  车型代码     DB230.DBD1208     : {model} (0x{model:08X})")
        print(f"  PIN 码       DB230.DBB1224..37 : {pin!r}")
        print("-" * 55)
        print("[完成] 即将断开连接，未对 PLC 做任何写操作。")

    except Exception as e:
        print(f"[错误] {e}")
    finally:
        try:
            client.disconnect()   # 立刻断开，不长期占用连接资源
        except Exception:
            pass
        print("[断开] 连接已关闭。")


if __name__ == "__main__":
    main()
