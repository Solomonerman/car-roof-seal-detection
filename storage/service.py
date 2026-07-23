# -*- coding: utf-8 -*-
"""存储服务：聚合数据库 + 对象存储，对总控暴露统一的 save(model, det, images)。

总控只调用这一个方法，不必关心图片存哪、记录怎么落库。
"""
import datetime
import json
import os
import sqlite3
from common.interfaces import InspectionRecord
from storage.database import Database
from storage.object_store import ObjectStore


# 保留最近 N 条记录，超出则清理旧记录+对应图片（0=不清理，谨慎可设大值）。
# 这是防止现场磁盘无限增长、最终写满崩溃的最后一道闸。
KEEP_RECORDS = 3000
_PRUNE_EVERY = 20      # 每 N 次 save 执行一次清理（降频，避免每次都扫库）

_svc = None
def get_service():
    """进程内单例：复用同一个 DB 连接，避免每车 new 一次（省连接开销）。"""
    global _svc
    if _svc is None:
        _svc = StorageService()
    return _svc


class StorageService:
    def __init__(self):
        self.db = Database()
        self.store = ObjectStore()
        self._save_count = 0

    def save(self, model: str, det, images: list,
             skid=None, pin="", no_paint=False, captured=False) -> InspectionRecord:
        refs = [self.store.upload(p) for p in images]   # upload 内部已去重(移动)
        rec = InspectionRecord(
            car_model=model,
            ok=det.ok,
            image_refs=refs,
            defects=det.defects,
            timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
            skid=skid,
            pin=pin,
            no_paint=no_paint,
            captured=captured,
        )
        self.db.save_record(rec)
        self._maybe_prune()
        return rec

    def _maybe_prune(self):
        self._save_count += 1
        if self._save_count % _PRUNE_EVERY == 0:
            self.prune()

    def prune(self, keep=KEEP_RECORDS):
        """清理最旧记录及其图片，防磁盘无限增长。keep<=0 表示不清理。"""
        if not keep or keep <= 0:
            return
        try:
            n = self.db.count()
            if n <= keep:
                return
            excess = n - keep
            with sqlite3.connect(self.db.db_path) as conn:
                rows = conn.execute(
                    "SELECT id, image_refs FROM records ORDER BY id ASC LIMIT ?",
                    (excess,),
                ).fetchall()
            for rid, refs_json in rows:
                try:
                    refs = json.loads(refs_json) if refs_json else []
                except Exception:
                    refs = []
                for ref in refs:
                    name = ref.split("/")[-1]
                    p = os.path.join(self.store.local_dir, name)
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                    except Exception:
                        pass
                with sqlite3.connect(self.db.db_path) as conn:
                    conn.execute("DELETE FROM records WHERE id=?", (rid,))
        except Exception as e:
            print(f"[存储] 清理失败: {e}")
