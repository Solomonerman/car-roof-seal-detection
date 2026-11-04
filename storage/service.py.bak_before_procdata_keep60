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
_VER_RE = re.compile(r"p([0-9a-fA-F]+__\d{4}-\d{2}-\d{2}-\d{6})")   # 原文件名里的版本运行时戳

def _version_tag_from_name(fname):
    m = _VER_RE.search(os.path.basename(fname))
    return m.group(1) if m else ""

def _stamp_from_name(fname):
    """从原文件名(含帧时间戳)提炼 YYYYMMDD_HHMMSS；失败则用当前时刻兜底。"""
    m = _STAMP_RE.search(os.path.basename(fname))
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}_{m.group(4)}{m.group(5)}{m.group(6)}"
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# 保留策略（防现场磁盘无限增长、最终写满崩溃的最后一道闸）：
#   - 时间维度：保留最近 KEEP_DAYS 天（默认60天≈2个月）的记录与照片；
#   - 数量维度：记录总数超过 KEEP_RECORDS 仍按最旧清理，作为节拍突增/时钟异常的兜底。
# 两者同时生效：先按时间删过期记录，若仍超硬上限再按 id 最旧删。
# KEEP_DAYS<=0 关闭时间清理；KEEP_RECORDS<=0 关闭数量兜底（谨慎，可能写满磁盘）。
KEEP_DAYS = 60
KEEP_RECORDS = 60000
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
             event_time=None, cam_idx=0, model_key=None,
             proc_dir="") -> InspectionRecord:
        # 一台车一个自包含文件夹：data/inspection/<日期>/<车>/
        ev = event_time or datetime.datetime.now()
        date = ev.strftime("%Y-%m-%d")
        car_key = _car_key(pin, skid)
        # 车型代码后缀（如 9X / 8X）：仅追加到【文件夹名】，文件名保持原 car_key 不变。
        # model_key 由自动回调(handle_car_signal)传入；不传则不加后缀，
        # 兼容旧调用方(state_machine)与无车型信息的场景——行为完全不变。
        car_dir_name = car_key + ("_" + _safe(model_key) if model_key else "")
        subdir = os.path.join(date, car_dir_name)
        folder = os.path.join(self.store.local_dir, subdir)
        refs = []
        for i, p in enumerate(images, 1):
            ext = os.path.splitext(p)[1] or ".bmp"
            stamp = _stamp_from_name(p)
            ver = _version_tag_from_name(p)
            ver_part = f"__p{ver}" if ver else ""
            new_name = f"{i:02d}__{car_key}__cam{cam_idx}__{stamp}{ver_part}{ext}"
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
            proc_dir=proc_dir or "",
        )
        self.db.save_record(rec)
        self._maybe_prune()
        return rec

    def _maybe_prune(self):
        self._save_count += 1
        if self._save_count % _PRUNE_EVERY == 0:
            self.prune()

    def prune(self, keep_days=KEEP_DAYS, hard_cap=KEEP_RECORDS):
        """清理过期/超额记录及其图片，防磁盘无限增长。

        - 时间维度：删除 timestamp 早于 (now - keep_days) 的记录与照片；
        - 数量维度：若仍超过 hard_cap，再按 id 最旧删（节拍突增/时钟异常兜底）。
        keep_days<=0 关闭时间清理；hard_cap<=0 关闭数量兜底。
        """
        try:
            if keep_days and keep_days > 0:
                cutoff = (datetime.datetime.now()
                          - datetime.timedelta(days=keep_days))
                cutoff_iso = cutoff.isoformat(timespec="seconds")
                with sqlite3.connect(self.db.db_path) as conn:
                    rows = conn.execute(
                        "SELECT id, image_refs FROM records "
                        "WHERE timestamp IS NOT NULL AND timestamp < ?",
                        (cutoff_iso,),
                    ).fetchall()
                if rows:
                    print(f"[存储][{datetime.datetime.now():%m-%d %H:%M}] 清理过期记录 {len(rows)} 条（保留最近{keep_days}天）")
                for rid, refs_json in rows:
                    self._delete_record_files(rid, refs_json)

            if hard_cap and hard_cap > 0:
                n = self.db.count()
                if n > hard_cap:
                    excess = n - hard_cap
                    with sqlite3.connect(self.db.db_path) as conn:
                        rows = conn.execute(
                            "SELECT id, image_refs FROM records "
                            "ORDER BY id ASC LIMIT ?",
                            (excess,),
                        ).fetchall()
                    if rows:
                        print(f"[存储][{datetime.datetime.now():%m-%d %H:%M}] 超上限{hard_cap}条，清理最旧 {len(rows)} 条")
                    for rid, refs_json in rows:
                        self._delete_record_files(rid, refs_json)
        except Exception as e:
            print(f"[存储] 清理失败: {e}")

    def _delete_record_files(self, rid, refs_json):
        """删除单条记录对应的照片文件+空文件夹，并从 DB 删除该记录。"""
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
