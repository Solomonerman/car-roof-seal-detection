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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
from storage.database import Database

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSPECTION_LOCAL = os.path.join("data", "inspection")   # 对应 ObjectStore.local_dir
app = FastAPI(title="车顶胶条检测 - 监控")
_db = Database()


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
def index():
    recs = _db.get_records(limit=50)
    total = _db.count()
    ng = sum(1 for r in recs if not r.ok)

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

        # 原始照片缩略图
        raw_imgs = "".join(
            f"<img class='thumb' src='{_img_url(ref)}' title='{ref}'>"
            for ref in (r.image_refs or [])
        ) or "<div class='muted'>无照片</div>"

        # 检测叠加图（可视化核心）
        ov = _overlay_urls(r.proc_dir)
        ov_imgs = "".join(
            f"<img class='thumb ov' src='/img?path={p}' title='检测叠加图'>"
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
          <div class="sec"><b>缺陷判定：</b>{defects_html}</div>
          <div class="sec"><b>原始照片（{len(r.image_refs or [])} 张）：</b><div class="row">{raw_imgs}</div></div>
          <div class="sec"><b>检测叠加图：</b><div class="row">{ov_imgs}</div></div>
        </div>
        """)

    cards_html = "\n".join(cards) if cards else "<p class='muted'>暂无记录</p>"

    return f"""
    <html><head>
      <meta charset="utf-8">
      <meta http-equiv="refresh" content="8">
      <title>车顶胶条检测监控</title>
      <style>
        body {{ font-family: Arial, "Microsoft YaHei", sans-serif; margin: 16px; background:#f5f6f8; }}
        h2 {{ margin: 4px 0 12px; }}
        .sum {{ color:#555; margin-bottom: 12px; }}
        .card {{ background:#fff; border:1px solid #ddd; border-radius:8px; padding:12px 14px; margin-bottom:14px; box-shadow:0 1px 3px rgba(0,0,0,.05); }}
        .chead {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; border-bottom:1px solid #eee; padding-bottom:8px; margin-bottom:8px; }}
        .ts {{ font-weight:bold; }}
        .model {{ background:#eef; padding:2px 8px; border-radius:4px; }}
        .meta {{ color:#888; font-size:12px; }}
        .badge {{ padding:2px 10px; border-radius:12px; color:#fff; font-weight:bold; }}
        .badge.ok {{ background:#2e9e4f; }}
        .badge.ng {{ background:#d23b3b; }}
        .sec {{ margin:6px 0; }}
        .row {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:4px; }}
        .thumb {{ width:150px; height:auto; border:1px solid #ccc; border-radius:4px; background:#000; }}
        .thumb.ov {{ border-color:#e0902a; }}
        .defects {{ margin:4px 0; padding-left:18px; }}
        .defects li.ok {{ color:#2e9e4f; }}
        .defects li.ng {{ color:#d23b3b; font-weight:bold; }}
        .muted {{ color:#999; font-size:13px; }}
      </style>
    </head>
    <body>
      <h2>车顶胶条检测 - 实时监控</h2>
      <div class="sum">记录总数：{total}（每 8 秒自动刷新）｜ 本页显示最近 {len(recs)} 条 ｜ NG：<b style="color:#d23b3b">{ng}</b></div>
      {cards_html}
    </body></html>
    """
