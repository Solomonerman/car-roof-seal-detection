# -*- coding: utf-8 -*-
"""PLC 信号时序监测（只读，生产安全）。

监测 DB130.DBX0.1（输送出车信号）：
  - 高频轮询该位，检测【上升沿 / 下降沿】
  - 测每次车信号的持续时间（脉冲宽度）与相邻车间隔
  - 上升沿时读取上下文：NO Paint / 车型代码 / 滑橇号 / PIN
目的：搞清该信号的确切时序（脉冲还是电平、宽度多少），为自动触发做准备。
⚠️ 全程只读 read_area，无任何写 / 停机 / 下载操作。
"""
import sys
import time
import argparse

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
DB130 = (130, 0, 1)          # DB130 字节0（含 DBX0.1）
DB230_A = (230, 1208, 30)    # DB230 字节1208..1237（车型/滑橇/PIN）
DB230_B = (230, 1257, 1)     # DB230 字节1257（含 DBX1257.1）


def parse_context(buf_a, buf_b):
    """buf_a 起始1208 长度30；buf_b 为字节1257。
    返回 (no_paint, skid, model_int, model_ascii, pin)。
    注意：车型代码 DBD1208 是 DWORD，但内容实为 ASCII（如 0x37503234='7P24'），
          故同时返回整数与原文字符串。
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
    return no_paint, skid, model_int, model_ascii, pin


def main():
    if not HAS_SNAP7:
        print("[错误] 未安装 python-snap7，请执行：pip install python-snap7")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="PLC 信号时序监测(只读)")
    parser.add_argument("--ip", default=PLC_IP)
    parser.add_argument("--rack", type=int, default=PLC_RACK)
    parser.add_argument("--slot", type=int, default=PLC_SLOT)
    parser.add_argument("--interval", type=float, default=10.0,
                        help="轮询间隔(毫秒)，越小边沿分辨越准（默认10）")
    parser.add_argument("--ctx-interval", type=float, default=200.0,
                        help="DB230 上下文采样间隔(毫秒)，用于定位车号有效窗口（默认200）")
    args = parser.parse_args()

    print("=" * 55)
    print("⚠️  只读监测：仅读 DB130/DB230，不写 PLC。Ctrl+C 停止。")
    print(f"    目标 {args.ip}  rack={args.rack} slot={args.slot}  间隔={args.interval}ms")
    print("=" * 55)

    client = snap7.client.Client()
    try:
        client.connect(args.ip, args.rack, args.slot)
        if not client.get_connected():
            print("[错误] 连接失败（不可达 / GET-PUT 未开 / 地址错）")
            sys.exit(1)
        print("[连接] 成功 ✅  开始监测（上升沿 = 车信号到来）...")

        poll_s = args.interval / 1000.0
        ctx_s = args.ctx_interval / 1000.0
        prev = 0
        rising_t = None
        last_fall_t = None
        count = 0
        widths = []   # 脉冲宽度（秒）
        gaps = []     # 相邻车间隔（秒）
        last_ctx_key = None
        last_ctx_t = 0.0

        print("提示：车号/车型在 DB230 的有效窗口未知，下面会持续采样 DB230 变化。")
        while True:
            try:
                d = client.read_area(S7AreaDB, *DB130)   # 仅读 1 字节
                bit = (d[0] >> 1) & 1
            except Exception as e:
                print(f"[读错] {e}")
                time.sleep(poll_s)
                continue

            now = time.perf_counter()
            if bit == 1 and prev == 0:
                # 上升沿
                rising_t = now
                if last_fall_t is not None:
                    gaps.append(now - last_fall_t)
                # 读上下文（车到来时读一次）
                try:
                    a = bytes(client.read_area(S7AreaDB, *DB230_A))
                    b = bytes(client.read_area(S7AreaDB, *DB230_B))
                    no_paint, skid, model, model_ascii, pin = parse_context(a, b)
                except Exception as e:
                    no_paint = skid = model = None
                    pin = str(e)
                count += 1
                print(f"[车 {count}] 上升沿 {time.strftime('%H:%M:%S')}  "
                      f"NO_Paint={no_paint} 滑橇={skid} 车型={model_ascii}(0x{model:08X}) PIN={pin!r}")
            elif bit == 0 and prev == 1:
                # 下降沿
                if rising_t is not None:
                    w = now - rising_t
                    widths.append(w)
                    print(f"        下降沿 脉冲宽度 = {w * 1000:.0f} ms")
                    last_fall_t = now
                # 下降沿也读一次上下文，对比车离开时是否还有数据
                try:
                    a = bytes(client.read_area(S7AreaDB, *DB230_A))
                    b = bytes(client.read_area(S7AreaDB, *DB230_B))
                    no_paint, skid, model, model_ascii, pin = parse_context(a, b)
                    print(f"        下降沿上下文 NO_Paint={no_paint} 滑橇={skid} "
                          f"车型={model_ascii}(0x{model:08X}) PIN={pin!r}")
                except Exception as e:
                    print(f"        下降沿读上下文失败: {e}")

            # 持续采样 DB230，定位车号/车型的有效窗口
            if now - last_ctx_t >= ctx_s:
                last_ctx_t = now
                try:
                    a = bytes(client.read_area(S7AreaDB, *DB230_A))
                    b = bytes(client.read_area(S7AreaDB, *DB230_B))
                    no_paint, skid, model, model_ascii, pin = parse_context(a, b)
                    key = (no_paint, skid, model, pin)
                    if key != last_ctx_key:
                        nonzero = not (skid == 0 and model == 0 and pin.strip("\x00") == "")
                        print(f"[DB230] {time.strftime('%H:%M:%S')} sig={bit} "
                              f"NO_Paint={no_paint} 滑橇={skid} 车型={model_ascii}(0x{model:08X}) PIN={pin!r}"
                              f"{'  *非空*' if nonzero else ''}")
                        last_ctx_key = key
                except Exception as e:
                    print(f"[DB230读错] {e}")

            prev = bit
            time.sleep(poll_s)

    except KeyboardInterrupt:
        print("\n[停止] 汇总：")
        if widths:
            print(f"  车次数={len(widths)}  脉冲宽度(ms): "
                  f"最小={min(widths) * 1000:.0f} 最大={max(widths) * 1000:.0f} "
                  f"平均={sum(widths) / len(widths) * 1000:.0f}")
        if gaps:
            print(f"  车间隔(s): 最小={min(gaps):.1f} 最大={max(gaps):.1f} "
                  f"平均={sum(gaps) / len(gaps):.1f}")
        if not widths:
            print("  未捕获到完整脉冲（可能监测时间太短，或轮询间隔大于脉冲宽度）")
    finally:
        try:
            client.disconnect()
        except Exception:
            pass
        print("[断开] 连接已关闭。")


if __name__ == "__main__":
    main()
