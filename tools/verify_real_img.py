"""基于【真实结果图】的看板端到端验证（替代伪造占位图，避免尺寸/溢出类偏差）。

- 复制 real_1009_1 的 12 张真实 _mask/_offset.png 到临时目录
- 写入真实结构的 summary.json（frame08 设为漏水风险 NG）
- 调用真实 render_board() 生成 HTML + mosaic
- 断言：4列×3行布局 / 顺序4-9 / frame08偏移图含红色NG / 表格有数据 / 字体不溢出
"""
import os, sys, shutil, json, base64, types
import cv2, numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(REPO, "data/process_data/real_1009_1")
TMP = os.path.join(REPO, "data/process_data/_verify_real_tmp")

# fastapi 仅在沙箱缺失（且 site-packages 有空命名空间壳会抢路径）；
# 先尝试真实导入，失败才注入 sys.modules 桩，避免覆盖 IPC 上的真实 fastapi。
try:
    import fastapi  # noqa
except ImportError:
    _fa = types.ModuleType("fastapi")
    class _FastAPI:
        def __init__(self, *a, **k): pass
        def get(self, *a, **k):
            def deco(f): return f
            return deco
    _fa.FastAPI = _FastAPI
    _fa.HTTPException = type("HTTPException", (Exception,), {})
    class _Query:
        def __init__(self, *a, **k): pass
    _fa.Query = _Query
    _fr = types.ModuleType("fastapi.responses")
    _fr.HTMLResponse = type("HTMLResponse", (), {})
    _fr.FileResponse = type("FileResponse", (), {})
    sys.modules["fastapi"] = _fa
    sys.modules["fastapi.responses"] = _fr

sys.path.insert(0, REPO)
import ui.app as app

# 1) 准备临时目录 + 真实图
if os.path.isdir(TMP):
    shutil.rmtree(TMP)
os.makedirs(TMP)
for f in os.listdir(SRC):
    if f.endswith((".png",)):
        shutil.copy(os.path.join(SRC, f), os.path.join(TMP, f))

# 2) 真实结构 summary.json（name 键必须与 _photo_base 一致：完整文件名前缀）
per_photo = []
files = [f for f in os.listdir(TMP) if f.endswith("_offset.png")]
for f in files:
    base = app._photo_base(f)          # 如 08__PINC..._2026-08-26-090604
    fn = app._frame_no(base)
    rec = {
        "name": base,
        "width_mm": 21.0 + (fn - 4) * 0.5,
        "width_in_spec": True,
        "offset_mm": 33.0 if fn != 8 else 26.0,   # frame08 = 漏水风险
        "offset_level": 0 if fn != 8 else 3,
        "offset_level_name": "正常" if fn != 8 else "漏水风险",
        "deviated": False if fn != 8 else True,
        "ok": True if fn != 8 else False,
    }
    per_photo.append(rec)
json.dump({"per_photo": per_photo}, open(os.path.join(TMP, "summary.json"), "w"), ensure_ascii=False)

# 3) 构造 rec
rec = types.SimpleNamespace(
    proc_dir="data/process_data/_verify_real_tmp",
    timestamp="2026-10-12 16:00:00", car_model="8X", skid="SK001",
    pin="PIN123", no_paint=False, ok=True)

html = app.render_board(rec, img_url_fn=lambda u: u)

# 4) 断言
checks = []
# 4a 顺序
pairs = app._board_pairs(rec.proc_dir)
order = [app._frame_no(b) for b, _, _, _ in pairs]
checks.append(("照片顺序=4,5,6,7,8,9", order == [4,5,6,7,8,9], str(order)))

# 4b mosaic 尺寸 = 3行×4列 ×(1200×1920)
m = None
for mm in __import__("re").finditer(r'data:image/png;base64,([A-Za-z0-9+/=]+)', html):
    mosaic = cv2.imdecode(np.frombuffer(base64.b64decode(mm.group(1)), np.uint8), cv2.IMREAD_COLOR)
    m = mosaic
    break
exp_h, exp_w = 3*1200, 4*1920
checks.append(("mosaic=3×4×(1200×1920)", m.shape[:2] == (exp_h, exp_w), str(m.shape[:2])))

# 4c frame08 偏移图（index 9: 0-based）含红色 NG
# 顺序: (4m,4o,5m,5o,6m,6o,7m,7o,8m,8o,9m,9o); index9 = (8,o)
def tile(mosaic, idx, ncols=4, H=1200, W=1920):
    r, c = divmod(idx, ncols)
    return mosaic[r*H:(r+1)*H, c*W:(c+1)*W]
t08o = tile(m, 9)
b, g, r_ = cv2.split(t08o)
red = (r_.astype(int) - np.maximum(g.astype(int), b.astype(int))) > 80
checks.append(("frame08偏移图含红色NG", int(red.sum()) > 500, f"redpx={int(red.sum())}"))

# 4d frame08 偏移图不应是整片绿（修复前 bug）
green = (g.astype(int) - np.maximum(r_.astype(int), b.astype(int))) > 80
checks.append(("frame08偏移图非整片绿", int(green.sum()) < int(t08o.size/3), f"greenpx={int(green.sum())}"))

# 4e 表格有数据，无“无逐图数据”
checks.append(("表格含宽度数据", "21.0" in html, ""))
checks.append(("表格含漏水风险", "漏水风险" in html, ""))
checks.append(("表格含NG判定", ">NG<" in html, ""))
checks.append(("无空行(无逐图数据)", "无逐图数据" not in html, ""))

# 5) 字体防溢出：检测器最长短串在真实图上不应超出边界
import offset_check_v7 as OC
img = cv2.imread(os.path.join(TMP, [f for f in os.listdir(TMP) if f.endswith('_offset.png')][0]))
def fits(text, scale, thick=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    s = scale
    while (tw > img.shape[1]-8 or th > img.shape[0]-8) and s > 0.3:
        s *= 0.92
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, s, thick)
    return tw <= img.shape[1]-8 and th <= img.shape[0]-8, s
for label, txt, sc in [("级别", "漏水风险", OC.FONT_LEVEL), ("偏移值", "33.0mm", OC.FONT_OFFSET), ("宽度", "W=22.2mm 100% 宽异", OC.FONT_WIDTH)]:
    ok_fit, final = fits(txt, sc)
    checks.append((f"检测器字体不溢出[{label}]", ok_fit, f"base={sc}->final={final:.2f}"))

# 看板 OK/NG 大字不溢出（真实图宽1920）
(tw, th), _ = cv2.getTextSize("NG", cv2.FONT_HERSHEY_SIMPLEX, 32.0, max(4,int(32.0*0.25)))
checks.append(("看板OK/NG大字不溢出", tw <= 1920-20, f"tw={tw}"))

# 输出预览
out = os.path.join(REPO, "tools", "verify_real_out.html")
open(out, "w", encoding="utf-8").write(html)

print("="*60)
allok = True
for name, ok, info in checks:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  {info}")
    allok = allok and ok
print("="*60)
print("ALL PASS" if allok else "SOME FAILED")
print("preview:", out)

# 清理临时 tmp（保留 out 预览）
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(0 if allok else 1)
