# -*- coding: utf-8 -*-
"""MinIO 对象存储（mock 实现）。

沙盒阶段把图像移动到本地 data/inspection/<日期>/<车>/ 目录，返回对象名；
现场联调时把 upload 内部替换为 minio SDK 的 put_object，对外接口不变。

目录结构（便于运维按车检索，且一台车一个自包含文件夹）：
  data/inspection/<YYYY-MM-DD>/<PINxxx|SKIDxxx|UNK>/<序号>__<车>__cam<N>__<时间>.bmp
"""
import os
import shutil


class ObjectStore:
    def __init__(self, bucket: str = "seal-images", local_dir: str = "data/inspection"):
        self.bucket = bucket
        self.local_dir = local_dir
        os.makedirs(local_dir, exist_ok=True)

    def upload(self, local_path: str, subdir: str = "", new_name: str = None) -> str:
        """移动（rename）单张图到 local_dir/subdir/ 下，重命名为 new_name。

        同文件系统为 rename（零额外 IO）；跨文件系统退化为 复制+删除。
        返回 ref = bucket/subdir/name，供 DB/sidecar 记录与后续读取。
        """
        name = new_name or os.path.basename(local_path)
        dst_dir = os.path.join(self.local_dir, subdir) if subdir else self.local_dir
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, name)
        try:
            if os.path.abspath(local_path) != os.path.abspath(dst):
                shutil.move(local_path, dst)
        except Exception as e:
            print(f"[存储] 移动图片失败 {local_path}: {e}")
        return f"{self.bucket}/{subdir}/{name}" if subdir else f"{self.bucket}/{name}"
