# -*- coding: utf-8 -*-
"""相机采集（mock 实现）。

数据来源：优先读取现场真实相片目录 data/raw_images；
若该目录为空（沙盒阶段），回退到 mock/sample_images 的占位图。

模拟"按时间节点采集 N 张"：从可用图像中按时间间隔挑选 count 张返回路径列表。
现场联调时，把 _list_images / capture 内部替换为相机 SDK 取帧逻辑即可，
对外 start / stop / capture 接口保持不变，总控无需改动。
"""
import os
import time
import glob
from config import constants as C
from common.interfaces import CaptureRequest, CaptureResult


class CameraCapture:
    def __init__(self, lighting=None):
        self.lighting = lighting
        self.streaming = False

    def start(self):
        # mock: 启动取流（真实场景打开相机）；同时开照明
        self.streaming = True
        if self.lighting:
            self.lighting.on()
        print("[采集] 相机取流已启动")

    def stop(self):
        self.streaming = False
        if self.lighting:
            self.lighting.off()
        print("[采集] 相机取流已停止")

    def _list_images(self) -> list:
        raw = sorted(glob.glob(os.path.join(C.RAW_IMAGE_DIR, "*")))
        if raw:
            return raw
        return sorted(glob.glob(os.path.join(C.MOCK_IMAGE_DIR, "*.png")))

    def capture(self, req: CaptureRequest) -> CaptureResult:
        imgs = self._list_images()
        if not imgs:
            return CaptureResult(ok=False, message="未找到任何图像")
        # 模拟按时间节点采集：从可用图像中按间隔挑选 count 张
        step = max(1, len(imgs) // req.count)
        selected = imgs[::step][:req.count]
        while len(selected) < req.count and imgs:
            selected.append(imgs[len(selected) % len(imgs)])
        # 模拟采集时间间隔（沙盒里压缩等待，避免拖慢 demo）
        for _ in selected:
            time.sleep(min(req.interval_sec, 0.01))
        return CaptureResult(ok=True, images=selected, message=f"采集 {len(selected)} 张")
