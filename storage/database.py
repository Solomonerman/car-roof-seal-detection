# -*- coding: utf-8 -*-
"""SQLite 本地存储（沙盒/笔记本阶段）。

把检测记录落到本地 SQLite 文件，解决 main.py 和 uvicorn 两个进程
数据不同步的问题。后续现场联调可替换为 MySQL / SQLAlchemy，
接口（save_record / get_records）保持不变。
"""
import json
import os
import sqlite3
from common.interfaces import InspectionRecord, Defect

# 数据库默认路径：项目根 / data / inspection.db
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "data", "inspection.db")


class Database:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    car_model TEXT,
                    ok INTEGER,
                    image_refs TEXT,
                    defects TEXT,
                    timestamp TEXT
                )
            """)
            conn.commit()

    def save_record(self, rec: InspectionRecord) -> InspectionRecord:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO records (car_model, ok, image_refs, defects, timestamp) VALUES (?, ?, ?, ?, ?)",
                (rec.car_model,
                 int(rec.ok),
                 json.dumps(rec.image_refs),
                 json.dumps([d.__dict__ for d in rec.defects]),
                 rec.timestamp)
            )
            conn.commit()
        print(f"[存储] 落库: 车型={rec.car_model} 结果={'OK' if rec.ok else 'NG'} "
              f"图片={len(rec.image_refs)}张 缺陷={len(rec.defects)}处")
        return rec

    def get_records(self, limit: int = 50) -> list:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM records ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        out = []
        for row in rows:
            out.append(InspectionRecord(
                car_model=row[1],
                ok=bool(row[2]),
                image_refs=json.loads(row[3]) if row[3] else [],
                defects=[Defect(**d) for d in json.loads(row[4])] if row[4] else [],
                timestamp=row[5]
            ))
        return out

    def count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM records").fetchone()
            return row[0] if row else 0
