# -*- coding: utf-8 -*-
"""Web 监控 UI（FastAPI）。

提供：
  GET /           简单展示页（结果列表 + 图片数 + 缺陷数），每 5 秒自动刷新
  GET /api/records 返回最近检测记录 JSON

数据从 SQLite 读取（与 main.py 共用），解决两个进程内存不共享的问题。
启动服务用 `uvicorn ui.app:app`。
"""
import os
import sys
# uvicorn 无论从哪个目录启动 ui.app，都确保项目根在模块查找路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from storage.database import Database

app = FastAPI(title="车顶胶条检测 - 监控")
_db = Database()


@app.get("/api/records")
def records():
    items = [r.__dict__ for r in _db.get_records(limit=20)]
    return {"count": _db.count(), "items": items}


@app.get("/", response_class=HTMLResponse)
def index():
    def yn(b):
        return '<span class="yes">是</span>' if b else '<span class="no">否</span>'
    rows = "".join(
        f"<tr><td>{r.timestamp}</td><td>{r.car_model}</td><td>{r.skid}</td>"
        f"<td>{r.pin}</td><td>{yn(r.no_paint)}</td>"
        f"<td>{'OK' if r.ok else 'NG'}</td>"
        f"<td>{yn(r.captured)}</td><td>{len(r.image_refs)}</td></tr>"
        for r in _db.get_records(limit=20)
    )
    return f"""
    <html><head>
      <meta charset="utf-8">
      <meta http-equiv="refresh" content="5">
      <title>车顶胶条检测监控</title>
      <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; max-width: 1000px; }}
        th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
        th {{ background: #f2f2f2; }}
        .ok {{ color: green; font-weight: bold; }}
        .ng {{ color: red; font-weight: bold; }}
        .yes {{ color: green; }}
        .no {{ color: #c60; }}
      </style>
    </head>
    <body>
      <h2>车顶胶条检测 - 实时监控</h2>
      <p>记录数：{_db.count()}（每 5 秒自动刷新）</p>
      <table>
        <tr><th>时间</th><th>车型</th><th>滑橇</th><th>PIN</th><th>NO_Paint</th>
            <th>结果</th><th>检测</th><th>图片数</th></tr>
        {rows}
      </table>
    </body></html>
    """
