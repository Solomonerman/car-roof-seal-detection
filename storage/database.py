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
    # records 表列顺序（用于 SELECT * 结果按名取值，兼容旧库缺列）
    _COLUMNS = ["id", "car_model", "ok", "image_refs", "defects", "timestamp",
                "skid", "pin", "no_paint", "captured", "proc_dir"]

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
            # 向后兼容：旧库可能没有追溯字段，按需加列（重复列忽略）
            for col, ctype in (("skid", "INTEGER"), ("pin", "TEXT"),
                               ("no_paint", "INTEGER"), ("captured", "INTEGER"),
                               ("proc_dir", "TEXT")):
                try:
                    conn.execute(f"ALTER TABLE records ADD COLUMN {col} {ctype}")
                except Exception:
                    pass
            conn.commit()

    def save_record(self, rec: InspectionRecord) -> InspectionRecord:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO records "
                "(car_model, ok, image_refs, defects, timestamp, skid, pin, no_paint, captured, proc_dir) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (rec.car_model,
                 int(rec.ok),
                 json.dumps(rec.image_refs),
                 json.dumps([d.__dict__ for d in rec.defects]),
                 rec.timestamp,
                 rec.skid,
                 rec.pin,
                 int(rec.no_paint),
                 int(rec.captured),
                 rec.proc_dir or "")
            )
            conn.commit()
        # 不在此 print：逐条打印既与 sidecar/UI 重复，又会被日志系统回灌成上下文噪声。
        # 需要审计时查 data/records 下的 sidecar JSON 或监控 UI 即可。
        return rec

    def get_records(self, limit: int = 50) -> list:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM records ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        out = []
        for row in rows:
            # 兼容旧库：新列（skid/pin/no_paint/captured）可能为 NULL 或不存在
            def col(name, default):
                try:
                    idx = Database._COLUMNS.index(name)
                    if idx < len(row):
                        v = row[idx]
                        return v if v is not None else default
                except ValueError:
                    pass
                return default
            out.append(InspectionRecord(
                car_model=row[1],
                ok=bool(row[2]),
                image_refs=json.loads(row[3]) if row[3] else [],
                defects=[Defect(**d) for d in json.loads(row[4])] if row[4] else [],
                timestamp=row[5],
                skid=col("skid", None),
                pin=col("pin", ""),
                no_paint=bool(col("no_paint", 0)),
                captured=bool(col("captured", 0)),
                proc_dir=col("proc_dir", ""),
            ))
        return out

    def count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM records").fetchone()
            return row[0] if row else 0
