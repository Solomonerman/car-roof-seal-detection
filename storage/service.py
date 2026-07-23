# -*- coding: utf-8 -*-
"""存储服务：聚合数据库 + 对象存储，对总控暴露统一的 save(model, det, images)。

总控只调用这一个方法，不必关心图片存哪、记录怎么落库。
"""
import datetime
import json
import os
import re
import sqlite3
from common.interfaces import InspectionRecord
from storage.database import Database
from storage.object_store import ObjectStore


# ---- 文件名/目录安全化与分层命名 ----
_SAFE_RE = re.compile(r"[^A-Za-z0-9_-]")
def _safe(s):
    """把任意字符串变成文件名安全片段（PIN 可能含特殊字符）。"""
    return _SAFE_RE.sub("_", str(s)).strip("_")[:40] or "x"

def _car_key(pin, skid):
    """车的目录名：优先 PIN，退化为 SKID，再退化为 UNK。"""
    pin = (pin or "").strip()
    if pin:
        return "PIN" + _safe(pin)
    if skid:
        return "SKID" + _safe(skid)
    return "UNK"

_STAMP_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})__(\d{2})-(\d{2})-(\d{2})-(\d{3})")
def _stamp_from_name(fname):
    """从原文件名(含帧时间戳)提炼 YYYYMMDD_HHMMSS；失败则用当前时刻兜底。"""
    m = _STAMP_RE.search(os.path.basename(fname))
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}_{m.group(4)}{m.group(5)}{m.group(6)}"
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


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
             skid=None, pin="", no_paint=False, captured=False,
             event_time=None, cam_idx=0) -> InspectionRecord:
        # 一台车一个自包含文件夹：data/inspection/<日期>/<车>/
        ev = event_time or datetime.datetime.now()
        date = ev.strftime("%Y-%m-%d")
        car_key = _car_key(pin, skid)
        subdir = os.path.join(date, car_key)
        folder = os.path.join(self.store.local_dir, subdir)
        refs = []
        for i, p in enumerate(images, 1):
            ext = os.path.splitext(p)[1] or ".bmp"
            stamp = _stamp_from_name(p)
            new_name = f"{i:02d}__{car_key}__cam{cam_idx}__{stamp}{ext}"
            refs.append(self.store.upload(p, subdir=subdir, new_name=new_name))
        rec = InspectionRecord(
            car_model=model,
            ok=det.ok,
            image_refs=refs,
            defects=det.defects,
            timestamp=ev.isoformat(timespec="seconds"),
            skid=skid,
            pin=pin,
            no_paint=no_paint,
            captured=captured,
            folder=folder,
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
                car_dir = None
                for ref in refs:
                    rel = ref.split("/")[1:]          # 去掉 bucket 段
                    p = os.path.join(self.store.local_dir, *rel)
                    car_dir = os.path.dirname(p)
                    try:
                        if os.path.exists(p):
                            os.remove(p)
                    except Exception:
                        pass
                # 清理已空的 车→天 文件夹，避免留下一堆空目录
                if car_dir:
                    for d in (car_dir, os.path.dirname(car_dir)):
                        try:
                            os.rmdir(d)
                        except OSError:
                            pass
                with sqlite3.connect(self.db.db_path) as conn:
                    conn.execute("DELETE FROM records WHERE id=?", (rid,))
        except Exception as e:
            print(f"[存储] 清理失败: {e}")
