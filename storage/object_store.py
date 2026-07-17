# -*- coding: utf-8 -*-
"""MinIO 对象存储（mock 实现）。

沙盒阶段把图像复制到本地 data/stored_images 目录，返回对象名；
现场联调时把 upload 内部替换为 minio SDK 的 put_object，对外接口不变。
"""
import os
import shutil


class ObjectStore:
    def __init__(self, bucket: str = "seal-images", local_dir: str = "data/stored_images"):
        self.bucket = bucket
        self.local_dir = local_dir
        os.makedirs(local_dir, exist_ok=True)

    def upload(self, local_path: str) -> str:
        name = os.path.basename(local_path)
        dst = os.path.join(self.local_dir, name)
        try:
            shutil.copy(local_path, dst)
        except Exception as e:
            print(f"[存储] 复制图片失败 {local_path}: {e}")
        return f"{self.bucket}/{name}"
