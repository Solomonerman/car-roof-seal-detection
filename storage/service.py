# -*- coding: utf-8 -*-
"""存储服务：聚合数据库 + 对象存储，对总控暴露统一的 save(model, det, images)。

总控只调用这一个方法，不必关心图片存哪、记录怎么落库。
"""
import datetime
import json
import os
import re
import shutil
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
#   磁盘水位式滚动清理（产量不稳场景，按剩余空间反推删旧）：
#   - 剩余空间 < DISK_FREE_LOW_GB(20G) 触发清理，删到 >= DISK_FREE_HIGH_GB(40G) 停手；
#   - 顺序：先删 process_data 过程图（最旧先删），不足再删 inspection 照片（最旧先删）；
#   - 保底：今天的数据永不删除；空间够就一直留，产量低的时期自动攒更长保留期。
# DISK_FREE_LOW_GB<=0 关闭清理。
DISK_FREE_LOW_GB = 20.0
DISK_FREE_HIGH_GB = 40.0
_PRUNE_EVERY = 20      # 每 N 次 save 检查一次磁盘水位（磁盘检查开销极小）

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
            self.prune_by_space()

    def _disk_free_gb(self):
        """返回数据所在磁盘的剩余空间(GB)。取 data/inspection 所在卷。"""
        p = os.path.abspath(self.store.local_dir)
        while p and not os.path.isdir(p):
            parent = os.path.dirname(p)
            if parent == p:
                break
            p = parent
        return shutil.disk_usage(p).free / (1024.0 ** 3)

    def prune_by_space(self, low_gb=DISK_FREE_LOW_GB, high_gb=DISK_FREE_HIGH_GB):
        """磁盘水位式滚动清理（产量不稳场景，按剩余空间反推删旧）。

        - 剩余空间 < low_gb 触发清理，删到 >= high_gb 停手；
        - 顺序：先删 process_data 过程图（最旧先删），不足再删 inspection 照片（最旧先删）；
        - 保底：今天的数据永不删除。low_gb<=0 关闭清理。
        """
        if not low_gb or low_gb <= 0:
            return
        try:
            free = self._disk_free_gb()
            if free >= low_gb:
                return
            now = datetime.datetime.now()
            today = now.date()
            print(f"[存储][{now:%m-%d %H:%M}] 磁盘剩余 {free:.1f}G < {low_gb}G，开始滚动清理（目标 {high_gb}G，先过程图后照片）")
            removed_proc = self._prune_process_data_until(high_gb, today)
            removed_rec = 0
            if self._disk_free_gb() < high_gb:
                removed_rec = self._prune_inspection_until(high_gb, today)
            print(f"[存储][{now:%m-%d %H:%M}] 清理结束：过程图 {removed_proc} 个、照片记录 {removed_rec} 条，剩余 {self._disk_free_gb():.1f}G")
        except Exception as e:
            print(f"[存储] 清理失败: {e}")

    @staticmethod
    def _folder_dt(name, mtime):
        """process_data 文件夹名前缀 YYYYMMDD_HHMMSS → 判定时刻；失败退回 mtime。"""
        m = re.match(r"(\d{8})_(\d{6})", name)
        if m:
            try:
                return datetime.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
            except ValueError:
                pass
        return datetime.datetime.fromtimestamp(mtime)

    def _prune_process_data_until(self, target_gb, today):
        """从最旧开始删 process_data 文件夹，删到磁盘剩余 >= target_gb。今天不删。"""
        proc_root = os.path.join("data", "process_data")
        if not os.path.isdir(proc_root):
            return 0
        entries = []
        for name in os.listdir(proc_root):
            full = os.path.join(proc_root, name)
            if not os.path.isdir(full):
                continue
            try:
                dt = self._folder_dt(name, os.path.getmtime(full))
            except Exception:
                continue
            if dt.date() == today:
                continue
            entries.append((dt, full))
        entries.sort(key=lambda x: x[0])
        removed = 0
        for dt, full in entries:
            if self._disk_free_gb() >= target_gb:
                break
            try:
                shutil.rmtree(full, ignore_errors=True)
                removed += 1
            except Exception:
                pass
        if removed:
            print(f"[存储][{datetime.datetime.now():%m-%d %H:%M}] 滚动清理 process_data 最旧 {removed} 个文件夹（保今天，删到剩余 {target_gb}G）")
        return removed

    def _prune_inspection_until(self, target_gb, today):
        """从最旧开始删 inspection 记录+照片，删到磁盘剩余 >= target_gb。今天不删。"""
        removed = 0
        today_s = today.isoformat()
        while self._disk_free_gb() < target_gb:
            with sqlite3.connect(self.db.db_path) as conn:
                rows = conn.execute(
                    "SELECT id, image_refs FROM records "
                    "WHERE timestamp IS NOT NULL AND substr(timestamp,1,10) != ? "
                    "ORDER BY id ASC LIMIT 20",
                    (today_s,),
                ).fetchall()
            if not rows:
                break
            for rid, refs_json in rows:
                self._delete_record_files(rid, refs_json)
                removed += 1
                if self._disk_free_gb() >= target_gb:
                    break
        if removed:
            print(f"[存储][{datetime.datetime.now():%m-%d %H:%M}] 滚动清理 inspection 最旧 {removed} 条记录+照片（保今天，删到剩余 {target_gb}G）")
        return removed

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
