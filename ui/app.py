# -*- coding: utf-8 -*-
"""Web 监控 UI（FastAPI）。

提供：
  GET /           检测结果可视化页（每车：缩略图 + OK/NG 徽章 + 缺陷明细 + 检测叠加图），每 8 秒自动刷新
  GET /api/records 返回最近检测记录 JSON
  GET /img?path=  安全读取图片（仅限项目 ROOT 内），用于缩略图与检测叠加图

数据从 SQLite 读取（与 web_capture.py 共用），解决两个进程内存不共享的问题。
启动：uvicorn ui.app:app --port 8000
"""
import os
import sys
import glob
import re as _re
import json as _json
import html as _html
import base64 as _b64mod
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
from storage.database import Database

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSPECTION_LOCAL = os.path.join("data", "inspection")   # 对应 ObjectStore.local_dir
app = FastAPI(title="车顶胶条检测 - 监控")
_db = Database()

# 检测帧范围（1-indexed，与 capture/web_capture.py 的 DETECT_FRAME_FROM/TO 保持一致）：
# 连拍 13 张中仅第 5~9 张为胶条位置、参与检测；UI 仅展示这些帧的原始照片与叠加图。
DET_FRAME_FROM = 5
DET_FRAME_TO = 9


# ---- 缺陷标签 → 中文 + 明细文本 ----
_LABEL_CN = {
    "seal": "胶条",
    "width": "宽度超差",
    "missing": "胶条缺失",
    "break": "断胶",
    "overspray": "过喷",
    "pending": "待检/未接入",
}

_NG_LABELS = {"missing", "break", "overspray", "width"}


def _defect_text(d) -> str:
    """把一条 Defect 渲染成可读文本。"""
    label = d.label
    meta = d.meta or {}
    cn = _LABEL_CN.get(label, label)
    if label == "width":
        w = meta.get("width_mm")
        return f"{cn} {w}mm（标准20±5）" if w is not None else cn
    if label == "missing":
        return "胶条缺失（ROI内无胶条）"
    if label == "break":
        if meta.get("type") == "纵向空洞":
            return f"断胶·纵向空洞 面积{meta.get('hole_area_px')}px"
        if "gap_mm" in meta:
            return f"断胶·段间间隙 {meta['gap_mm']}mm（常态{meta.get('typical_gap_mm')}mm）"
        return "断胶"
    if label == "overspray":
        return f"过喷 面积{meta.get('area_px')}px"
    if label == "seal":
        w = meta.get("width_mm")
        return f"胶条 OK {w}mm" if w is not None else "胶条 OK"
    # pending / 其他
    return meta.get("reason") or cn


def _img_url(ref: str) -> str:
    """把 storage 返回的 ref（seal-images/...）转成 /img 可读取的相对路径。"""
    if ref.startswith("seal-images/"):
        rel = ref[len("seal-images/"):]
    else:
        rel = ref
    return "/img?path=" + INSPECTION_LOCAL.replace("\\", "/") + "/" + rel.replace("\\", "/")


def _overlay_urls(proc_dir: str) -> list:
    """列出某车检测叠加图（data/process_data/<folder>/*_overlay.png）。"""
    if not proc_dir:
        return []
    d = os.path.join(ROOT, proc_dir)
    if not os.path.isdir(d):
        return []
    return sorted(glob.glob(os.path.join(d, "*_overlay.png")))


@app.get("/api/records")
def records():
    items = [r.__dict__ for r in _db.get_records(limit=50)]
    return {"count": _db.count(), "items": items}


def _esc(s) -> str:
    """HTML 转义，防止查询输入注入页面。"""
    return _html.escape(str(s), quote=True)


@app.get("/img")
def img(path: str = Query(..., description="相对 ROOT 的安全路径")):
    """安全读取图片：仅允许访问 ROOT 内文件，防目录穿越。"""
    full = os.path.normpath(os.path.join(ROOT, path))
    if not full.startswith(os.path.normpath(ROOT)):
        raise HTTPException(403, "禁止访问 ROOT 外路径")
    if not os.path.isfile(full):
        raise HTTPException(404, "图片不存在")
    return FileResponse(full)


@app.get("/", response_class=HTMLResponse)
def index(skid: str = Query("", description="滑橇号筛选"),
          model: str = Query("", description="车型筛选"),
          date: str = Query("", description="日期 YYYY-MM-DD"),
          only_ng: bool = Query(False, description="仅显示NG")):
    # 取较宽的历史用于查询（无筛选时再截断为最近 50 条）
    all_recs = _db.get_records(limit=500)
    q = skid.strip()
    m = model.strip()
    d = date.strip()
    if q:
        all_recs = [r for r in all_recs if q in str(r.skid)]
    if m:
        all_recs = [r for r in all_recs if m in (r.car_model or "")]
    if d:
        all_recs = [r for r in all_recs if (r.timestamp or "").startswith(d)]
    if only_ng:
        all_recs = [r for r in all_recs if not r.ok]
    recs = [r for r in all_recs if r.image_refs]   # 仅显示实际拍照(有照片)的车，过滤免检跳过等无检测记录
    total = _db.count()
    ng = sum(1 for r in recs if not r.ok)
    # 默认（无筛选）仅显示最近 50 条；有筛选时显示全部匹配（最多 200）
    if not (q or m or d or only_ng):
        recs = recs[:50]
    else:
        recs = recs[:200]

    cards = []
    for r in recs:
        badge = ('<span class="badge ok">OK</span>' if r.ok
                 else '<span class="badge ng">NG</span>')
        # 缺陷明细
        if r.defects:
            defects_html = "<ul class='defects'>"
            for d in r.defects:
                cls = "ng" if d.label in _NG_LABELS else "ok"
                defects_html += f"<li class='{cls}'>● {_defect_text(d)}</li>"
            defects_html += "</ul>"
        else:
            defects_html = "<div class='muted'>无缺陷记录</div>"

        # 仅展示胶条位置的第 5~9 张：原始照片切片 + 检测叠加图（同尺寸放大）
        det_refs = (r.image_refs or [])[DET_FRAME_FROM - 1: DET_FRAME_TO]

        # 原始照片缩略图（第5~9张，放大显示）
        raw_imgs = "".join(
            f"<img class='thumb' src='{_img_url(ref)}' title='原始照片（第{DET_FRAME_FROM}~{DET_FRAME_TO}张）'>"
            for ref in det_refs
        ) or "<div class='muted'>无照片</div>"

        # 检测叠加图（检测结果，与原始照片同尺寸，第5~9张）
        ov = _overlay_urls(r.proc_dir)
        ov_imgs = "".join(
            f"<img class='thumb ov' src='/img?path={p}' title='检测叠加图（第{DET_FRAME_FROM}~{DET_FRAME_TO}张）'>"
            for p in ov
        ) or "<div class='muted'>无叠加图（8X/未知车型或未生成）</div>"

        cards.append(f"""
        <div class="card">
          <div class="chead">
            <span class="ts">{r.timestamp}</span>
            <span class="model">{r.car_model}</span>
            {badge}
            <span class="meta">滑橇={r.skid} PIN={r.pin} NO_Paint={'是' if r.no_paint else '否'}</span>
          </div>
          <div class="sec"><b>原始照片（第{DET_FRAME_FROM}~{DET_FRAME_TO}张）：</b><div class="row">{raw_imgs}</div></div>
          <div class="sec hi"><b>检测结果（第{DET_FRAME_FROM}~{DET_FRAME_TO}张 · 胶条位置）：</b><div class="row">{ov_imgs}</div></div>
          <div class="sec"><b>缺陷判定：</b>{defects_html}</div>
        </div>
        """)

    cards_html = "\n".join(cards) if cards else "<p class='muted'>暂无记录</p>"

    skid_esc = _esc(skid)
    model_esc = _esc(model)
    date_esc = _esc(date)
    ng_checked = "checked" if only_ng else ""

    return f"""
    <html><head>
      <meta charset="utf-8">
      <meta http-equiv="refresh" content="8">
      <title>车顶胶条检测监控</title>
      <style>
        body {{ font-family: Arial, "Microsoft YaHei", sans-serif; margin: 16px; background:#f5f6f8; }}
        h2 {{ margin: 4px 0 12px; }}
        .sum {{ color:#555; margin-bottom: 12px; }}
        .qpanel {{ background:#fff; border:1px solid #cdd; border-radius:8px; padding:10px 12px; margin-bottom:12px; display:flex; flex-wrap:wrap; align-items:center; gap:10px; }}
        .qpanel span {{ font-size:13px; color:#333; }}
        .qpanel input[type=text], .qpanel input[type=date] {{ padding:3px 6px; border:1px solid #bbb; border-radius:4px; }}
        .qpanel .chk {{ display:inline-flex; align-items:center; gap:4px; }}
        .qpanel button {{ padding:4px 14px; border:0; border-radius:4px; background:#2e7fd0; color:#fff; cursor:pointer; }}
        .qpanel button:hover {{ background:#2569ad; }}
        .qpanel .reset {{ margin-left:4px; color:#666; text-decoration:none; font-size:13px; }}
        .qpanel .reset:hover {{ color:#d23b3b; }}
        .card {{ background:#fff; border:1px solid #ddd; border-radius:8px; padding:12px 14px; margin-bottom:14px; box-shadow:0 1px 3px rgba(0,0,0,.05); }}
        .chead {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; border-bottom:1px solid #eee; padding-bottom:8px; margin-bottom:8px; }}
        .ts {{ font-weight:bold; }}
        .model {{ background:#eef; padding:2px 8px; border-radius:4px; }}
        .meta {{ color:#888; font-size:12px; }}
        .badge {{ padding:2px 10px; border-radius:12px; color:#fff; font-weight:bold; }}
        .badge.ok {{ background:#2e9e4f; }}
        .badge.ng {{ background:#d23b3b; }}
        .sec {{ margin:6px 0; }}
        .sec.hi {{ background:#fff7e6; border:1px solid #e0a83a; border-radius:6px; padding:8px 10px; }}
        .banner {{ background:#eaf3ff; border:1px solid #9cc4f0; color:#23527c; border-radius:6px; padding:8px 12px; margin-bottom:12px; font-size:13px; }}
        .row {{ display:flex; flex-wrap:wrap; gap:12px; margin-top:8px; }}
        .thumb {{ width:300px; height:auto; border:1px solid #ccc; border-radius:4px; background:#000; }}
        .thumb.ov {{ width:300px; border:2px solid #e0902a; border-radius:6px; box-shadow:0 2px 6px rgba(0,0,0,.2); }}
        .defects {{ margin:4px 0; padding-left:18px; }}
        .defects li.ok {{ color:#2e9e4f; }}
        .defects li.ng {{ color:#d23b3b; font-weight:bold; }}
        .muted {{ color:#999; font-size:13px; }}
      </style>
    </head>
    <body>
      <h2>车顶胶条检测 - 实时监控</h2>
      <form method="get" class="qpanel">
        <span>滑橇号 <input type="text" name="skid" value="{skid_esc}" size="8" placeholder="如 3699"></span>
        <span>车型 <input type="text" name="model" value="{model_esc}" size="8" placeholder="如 9X"></span>
        <span>日期 <input type="date" name="date" value="{date_esc}"></span>
        <span class="chk"><label><input type="checkbox" name="only_ng" value="1" {ng_checked}> 仅 NG</label></span>
        <button type="submit">查询</button>
        <a class="reset" href="/">重置</a>
        <a class="reset" href="/board">全屏看板</a>
      </form>
      <div class="banner">本页仅显示已拍照（有检测）的车；检测仅针对连拍中第 {DET_FRAME_FROM}~{DET_FRAME_TO} 张（胶条位置），原始照片与检测叠加图同尺寸放大展示供工人目视检查。</div>
      <div class="sum">记录总数：{total}（每 8 秒自动刷新）｜ 本页显示 {len(recs)} 条｜ NG：<b style="color:#d23b3b">{ng}</b></div>
      {cards_html}
    </body></html>
    """


# ===================== 全屏看板（每台车 12 张结果图 + 底部数据表格） =====================
_BOARD_REFRESH_SEC = 5   # 看板轮询间隔(秒)：仅用于检测『是否换了新车』，不再强制整页刷新


def _frame_no(name: str) -> int:
    """从文件名/基名提取帧号（如 '04__PIN...' -> 4）。"""
    m = _re.match(r"(\d{2})", os.path.basename(str(name)))
    return int(m.group(1)) if m else 0


def _photo_base(fname: str) -> str:
    """去掉 _mask / _offset 后缀，得到照片基名（用于把同一张照片的两张结果图配对）。"""
    s = os.path.splitext(fname)[0]
    for suf in ("_mask", "_offset"):
        if s.endswith(suf):
            return s[: -len(suf)]
    return s


def _board_pairs(proc_dir_rel: str):
    """列出该车的 12 张图（6 mask + 6 offset），按帧号排序，并读取逐图汇总。

    返回 [(base, mask_url(/img?path=), offset_url(/img?path=), summary_or_None), ...]
    mask/offset 的 URL 用现有 /img 安全接口（仅 ROOT 内），线上与离线测试共用。
    """
    if not proc_dir_rel:
        return []
    d = os.path.join(ROOT, proc_dir_rel)
    if not os.path.isdir(d):
        return []
    masks = {_photo_base(f): f for f in os.listdir(d) if f.endswith("_mask.png")}
    offs = {_photo_base(f): f for f in os.listdir(d) if f.endswith("_offset.png")}
    bases = sorted(set(masks) | set(offs), key=_frame_no)
    summ_map = {}
    sj = os.path.join(d, "summary.json")
    if os.path.isfile(sj):
        try:
            for p in _json.loads(open(sj, encoding="utf-8").read()).get("per_photo", []):
                summ_map[p.get("name")] = p
        except Exception:
            pass
    pairs = []
    for b in bases:
        mfile = masks.get(b)
        ofile = offs.get(b)
        rel = proc_dir_rel.replace("\\", "/")
        murl = (f"/img?path={rel}/{mfile}") if mfile else ""
        ourl = (f"/img?path={rel}/{ofile}") if ofile else ""
        pairs.append((b, murl, ourl, summ_map.get(b)))
    return pairs


def _build_mosaic(proc_dir_rel):
    """把 12 张结果图拼成一张无缝大图（3 列 × 4 行）：上 2 行=宽度/胶条，下 2 行=偏移。

    零像素间隙；每张图：
      - 左上角缩小标题（宽度检测（第N张）/偏移检测（第N张））
      - 整图粗边框：
        宽度检测图：宽度合格=绿 / 不合格=红（只看宽度，不跟随偏移等级）
        偏移检测图：NG=红；OK=跟随偏移等级配色（正常=绿/轻微=黄/中度=橙）
      - 底部居中 OK/NG 大字（与外框同色、加粗，字号缩小20%）
    返回 PNG 字节；无图返回 None。
    """
    if not proc_dir_rel:
        return None
    d = os.path.join(ROOT, proc_dir_rel)
    if not os.path.isdir(d):
        return None
    masks = {_photo_base(f): f for f in os.listdir(d) if f.endswith("_mask.png")}
    offs = {_photo_base(f): f for f in os.listdir(d) if f.endswith("_offset.png")}
    bases = sorted(set(masks) | set(offs), key=_frame_no)
    if not bases:
        return None
    # 读 summary.json 拿逐图 OK / 偏移等级（用于边框与底部大字颜色）
    summ_map = {}
    sj = os.path.join(d, "summary.json")
    if os.path.isfile(sj):
        try:
            for p in _json.loads(open(sj, encoding="utf-8").read()).get("per_photo", []):
                summ_map[p.get("name")] = p
        except Exception:
            pass
    tiles, kinds, oks, levels, wis = [], [], [], [], []
    for kind, getter in (("mask", masks), ("offset", offs)):
        for b in bases:
            f = getter.get(b)
            if f:
                fp = os.path.join(d, f)
                arr = cv2.imdecode(np.fromfile(fp, dtype=np.uint8), cv2.IMREAD_COLOR)
            else:
                arr = None
            tiles.append(arr)
            kinds.append(kind)
            s = summ_map.get(b)
            oks.append(bool((s or {}).get("ok", True)))
            levels.append(int((s or {}).get("offset_level", 0) or 0))
            wis.append(bool((s or {}).get("width_in_spec", True)))
    valid = [t for t in tiles if t is not None and t.size]
    if not valid:
        return None
    H, W = valid[0].shape[:2]
    n = len(bases)
    out = []
    for idx, (arr, kind, ok, lv, wi) in enumerate(zip(tiles, kinds, oks, levels, wis)):
        if arr is None or arr.size == 0:
            arr = np.zeros((H, W, 3), np.uint8)
        elif arr.shape[:2] != (H, W):
            arr = cv2.resize(arr, (W, H))
        nn = (idx % n) + 1
        label = ("宽度检测（第%d张）" if kind == "mask" else "偏移检测（第%d张）") % nn
        _draw_caption(arr, label)
        # 宽度检测图用宽度合格/不合格上色；偏移检测图用偏移等级上色
        _draw_status(arr, ok, lv, mask=(kind == "mask"), width_in_spec=wi)
        out.append(arr)
    ncols = 3
    rows = [np.hstack(out[i:i + ncols]) for i in range(0, len(out), ncols)]
    mosaic = np.vstack(rows)
    ok, buf = cv2.imencode(".png", mosaic)
    return buf.tobytes() if ok else None


def _draw_caption(vis, text):
    """在图左上角画缩小后的标题（带黑底板，保证清晰）。"""
    scale = 0.48   # 0.6 * 0.8（整体缩小20%）
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x0, y0 = 8, 8
    cv2.rectangle(vis, (x0, y0), (x0 + tw + 10, y0 + th + 10), (0, 0, 0), -1)
    cv2.putText(vis, text, (x0 + 5, y0 + th + 5), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), 2)


def _draw_status(vis, ok, offset_level=0, mask=False, width_in_spec=True):
    """每张结果图：整图粗边框 + 底部居中 OK/NG 大字（加粗）。

    外框颜色：
      - 宽度检测图（mask=True）：只看宽度——
          宽度合格 → 绿框 / 底部 OK；宽度不合格 → 红框 / 底部 NG
          （不再跟随偏移等级，避免合格宽度却被轻微/中度偏移染黄/橙）
      - 偏移检测图（mask=False）：
          NG（任何 NG，含宽度超差/缺失/漏水风险）→ 红色
          OK → 跟随偏移等级配色（与照片一致）：正常=绿 / 轻微=黄 / 中度=橙
    底部 OK/NG 大字颜色与外框保持一致。
    """
    h, w = vis.shape[:2]
    if mask:
        # 宽度检测图：仅按宽度合格与否上色（绿/红）
        ng = not width_in_spec
        color = (0, 0, 255) if ng else (0, 255, 0)   # 红 / 绿
        text = "NG" if ng else "OK"
    else:
        if not ok:
            color = (0, 0, 255)                          # 红
        else:
            # 偏移等级配色（BGR，与 offset_check_v7._LEVEL_COLOR 一致）
            color = {0: (0, 255, 0),                     # 正常=绿
                     1: (0, 255, 255),                   # 轻微=黄
                     2: (0, 165, 255)}.get(offset_level, (0, 255, 0))  # 中度=橙；未知兜底绿
        text = "OK" if ok else "NG"
    bt = max(6, int(min(h, w) * 0.012))              # 粗边框线宽
    cv2.rectangle(vis, (0, 0), (w - 1, h - 1), color, bt)
    scale = 4.0                                      # 5.0 * 0.8（整体缩小20%）
    thick = 4                                        # 加粗
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cx = max(4, (w - tw) // 2)
    yb = h - th - max(16, int(h * 0.06))             # 底部，留边
    cv2.rectangle(vis, (cx - 10, yb - 10), (cx + tw + 10, yb + th + 10), (0, 0, 0), -1)
    cv2.putText(vis, text, (cx, yb + th), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thick)


def _board_cell(fn: int, label: str, url: str) -> str:
    if url:
        return (f"<div class='cell'><div class='cap'>{label} · 帧{fn:02d}</div>"
                f"<img src='{url}' alt='{label}'></div>")
    return (f"<div class='cell'><div class='cap'>{label} · 帧{fn:02d}</div>"
            f"<div class='ph'>无图</div></div>")


def _board_tablerow(fn: int, s) -> str:
    if not s:
        return (f"<tr><td>{fn:02d}</td><td colspan='5' class='muted'>"
                f"无逐图数据</td></tr>")
    w = s.get("width_mm")
    wtxt = f"{w:.1f}" if w is not None else "-"
    wok = s.get("width_in_spec", True)
    om = s.get("offset_mm")
    otxt = f"{om:.1f}" if om is not None else "-"
    lv = s.get("offset_level", -1)
    lvn = s.get("offset_level_name") or "-"
    # 偏移(mm) 与 偏移等级 两列：按照片同款配色（0绿/1黄/2橙/3红），与结果图框色一致
    lvcls = {0: "lv0", 1: "lv1", 2: "lv2", 3: "lv3"}.get(lv, "")
    dev = s.get("deviated")
    pok = s.get("ok", True)
    return (f"<tr><td>{fn:02d}</td>"
            f"<td class='{'ok' if wok else 'ng'}'>{wtxt}</td>"
            f"<td class='{'ok' if wok else 'ng'}'>{'合格' if wok else '超差'}</td>"
            f"<td class='{lvcls}'>{otxt}</td><td class='{lvcls}'>{lvn}</td>"
            f"<td>{'是' if dev else '否'}</td>"
            f"<td class='{'ok' if pok else 'ng'}'>{'OK' if pok else 'NG'}</td></tr>")


def _board_state_key(rec) -> str:
    """当前车的唯一标识（时间|滑橇|PIN|过程目录），用于判断『是否换了一台新车』。"""
    return "|".join([str(rec.timestamp or ""), str(rec.skid or ""),
                     str(rec.pin or ""), str(rec.proc_dir or "")])


def render_board(rec, img_url_fn=lambda u: u, live=False) -> str:
    """渲染全屏看板 HTML。

    img_url_fn: 把 /img?path=... 转成可嵌入的 URL/DataURI。线上传恒等函数；
    离线测试传 base64 data URI 生成器，便于脱离服务器预览。
    """
    pairs = _board_pairs(rec.proc_dir)
    # 图片区：单张「无缝拼接大图」(mosaic) —— 12 张按 3 列×4 行拼接（上2行=宽度/胶条，下2行=偏移），
    # 图片拉伸填满格子（放大显示、零黑边、零间隙）；每张带绿/红粗边框与底部居中 OK/NG 大字。
    cells_for_table = [(b, s) for (b, _, _, s) in pairs]
    tabrows = "\n".join(_board_tablerow(_frame_no(b), s) for b, s in cells_for_table)
    mosaic_bytes = _build_mosaic(rec.proc_dir)
    if mosaic_bytes:
        mosaic_src = "data:image/png;base64," + _b64mod.b64encode(mosaic_bytes).decode("ascii")
        grid_html = f"<img class='mosaic' src='{mosaic_src}' alt='检测结果拼接图'>"
    else:
        grid_html = "<div class='ph'>无检测结果图片</div>"

    # 总判定基于逐图汇总（偏移仅『漏水风险』才 NG；宽度超差 NG），与表格自洽
    if pairs:
        ok = all((s or {}).get("ok", True) for (_, _, _, s) in pairs)
    else:
        ok = bool(rec.ok)
    badge = ('<span class="badge ok">OK</span>' if ok else
             '<span class="badge ng">NG</span>')
    # 仅线上(live)注入轮询脚本：每 _BOARD_REFRESH_SEC 秒问一次『最新车』，
    # 仅当标识变化（新车检测完成）才整页重载，避免定时无脑刷新/覆盖
    live_script = ""
    if live:
        cur_key = _board_state_key(rec)
        live_script = (
            "<script>\n"
            f"  var CUR = {_json.dumps(cur_key)};\n"
            "  setInterval(function(){\n"
            "    fetch('/board_state').then(function(r){return r.json();}).then(function(d){\n"
            "      if(d.key && d.key!==CUR){ location.reload(); }\n"
            "    }).catch(function(){});\n"
            "  }, " + str(_BOARD_REFRESH_SEC * 1000) + ");\n"
            "</script>"
        )
    topbar = (f"<span class='ts'>{_esc(rec.timestamp)}</span> "
              f"<span class='model'>{_esc(rec.car_model or '-')}</span> {badge} "
              f"<span class='meta'>滑橇={_esc(rec.skid)} PIN={_esc(rec.pin)} "
              f"NO_Paint={'是' if rec.no_paint else '否'}</span>")

    return f"""
    <!doctype html><html lang="zh"><head>
      <meta charset="utf-8">
      <title>车顶胶条检测 - 全屏看板</title>
      <style>
        * {{ box-sizing: border-box; }}
        html, body {{ margin:0; height:100%; }}
        body {{ display:flex; flex-direction:column; height:100vh;
               font-family: Arial, "Microsoft YaHei", sans-serif; background:#111; }}
        .topbar {{ flex:0 0 auto; display:flex; align-items:center; gap:12px; flex-wrap:wrap;
                  padding:8px 14px; background:#1f2937; color:#fff; font-size:14px; }}
        .topbar .model {{ background:#374151; padding:2px 10px; border-radius:4px; }}
        .topbar .meta {{ color:#9ca3af; font-size:12px; }}
        .topbar .badge {{ padding:2px 12px; border-radius:12px; color:#fff; font-weight:bold; }}
        .topbar .badge.ok {{ background:#2e9e4f; }}
        .topbar .badge.ng {{ background:#d23b3b; }}
        .topbar a {{ color:#60a5fa; font-size:13px; margin-left:auto; text-decoration:none; }}
        .mosaicwrap {{ flex:1 1 auto; min-height:0; background:#000; display:flex;
                       align-items:center; justify-content:center; padding:0; }}
        .mosaicwrap img {{ width:100%; height:100%; object-fit:contain; display:block; background:#000; }}
        .mosaicwrap .ph {{ color:#888; font-size:16px; }}
        .tablewrap {{ flex:0 0 18vh; overflow:auto; background:#fff; }}
        .tablewrap h3 {{ margin:6px 12px; font-size:14px; color:#333; }}
        table {{ width:100%; border-collapse:collapse; font-size:13px; }}
        th, td {{ border:1px solid #ddd; padding:4px 8px; text-align:center; }}
        th {{ background:#2e7fd0; color:#fff; position:sticky; top:0; }}
        td.ok {{ color:#2e9e4f; font-weight:bold; }}
        td.ng {{ color:#d23b3b; font-weight:bold; }}
        td.warn {{ color:#e0902a; font-weight:bold; }}
        td.muted {{ color:#999; }}
        /* 偏移等级/偏移值 配色与结果照片同色族；白底改为『深色调+浅色底纹』保证可读
           （纯黄/纯绿在白底上几乎隐形，故加深并加底纹，仍对应 绿/黄/橙/红 四级） */
        td.lv0 {{ color:#1f9d4d; background:#e8f8ee; font-weight:bold; }}
        td.lv1 {{ color:#9a6a00; background:#fff3bf; font-weight:bold; }}
        td.lv2 {{ color:#c2410c; background:#ffe8d6; font-weight:bold; }}
        td.lv3 {{ color:#c0392b; background:#fde8e8; font-weight:bold; }}
      </style>
    </head>
    <body>
      <div class="topbar">{topbar}
        <a href="/">← 返回记录列表</a>
      </div>
      <div class="mosaicwrap">{grid_html}</div>
      <div class="tablewrap">
        <h3>检测数据明细（每台车 6 张照片）</h3>
        <table>
          <thead><tr>
            <th>帧号</th><th>宽度(mm)</th><th>宽度判定</th>
            <th>偏移(mm)</th><th>偏移等级</th><th>是否偏离</th><th>单图判定</th>
          </tr></thead>
          <tbody>{tabrows}</tbody>
        </table>
      </div>
      {live_script}
    </body></html>
    """


@app.get("/board_state")
def board_state():
    """轻量状态接口：返回当前最新车的标识，供看板前端判断是否『换新车』（仅在车完成时才覆盖）。"""
    recs = _db.get_records(limit=1)
    if not recs:
        return {"has": False, "key": ""}
    rec = recs[0]
    return {"has": True, "key": _board_state_key(rec),
            "timestamp": rec.timestamp, "model": rec.car_model,
            "skid": rec.skid, "pin": rec.pin}


@app.get("/board", response_class=HTMLResponse)
def board():
    recs = _db.get_records(limit=1)
    if not recs:
        return HTMLResponse("<html><body style='font-family:sans-serif;padding:20px'>"
                           "暂无检测记录</body></html>")
    # live=True：注入轮询脚本，仅新车完成才重新加载（避免定时刷新覆盖正在看的车）
    return HTMLResponse(render_board(recs[0], live=True))
