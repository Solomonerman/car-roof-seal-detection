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
import html as _html

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
      </form>
      <div class="banner">本页仅显示已拍照（有检测）的车；检测仅针对连拍中第 {DET_FRAME_FROM}~{DET_FRAME_TO} 张（胶条位置），原始照片与检测叠加图同尺寸放大展示供工人目视检查。</div>
      <div class="sum">记录总数：{total}（每 8 秒自动刷新）｜ 本页显示 {len(recs)} 条｜ NG：<b style="color:#d23b3b">{ng}</b></div>
      {cards_html}
    </body></html>
    """
