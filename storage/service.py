# -*- coding: utf-8 -*-
"""存储服务：聚合数据库 + 对象存储，对总控暴露统一的 save(model, det, images)。

总控只调用这一个方法，不必关心图片存哪、记录怎么落库。
"""
import datetime
from common.interfaces import InspectionRecord
from storage.database import Database
from storage.object_store import ObjectStore


class StorageService:
    def __init__(self):
        self.db = Database()
        self.store = ObjectStore()

    def save(self, model: str, det, images: list) -> InspectionRecord:
        refs = [self.store.upload(p) for p in images]
        rec = InspectionRecord(
            car_model=model,
            ok=det.ok,
            image_refs=refs,
            defects=det.defects,
            timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
        )
        self.db.save_record(rec)
        return rec
