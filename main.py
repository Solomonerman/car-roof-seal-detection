# -*- coding: utf-8 -*-
"""车顶胶条检测系统 - 总控入口。

装配各模块并以 mock 模式跑通端到端流程：
  PLC(模拟信号) -> 车型校验 -> 采集(现场相片) -> 检测(占位算法)
  -> 存储(SQLite) -> 回写PLC(打印)

检测记录会自动保存到 SQLite（data/inspection.db），uvicorn 启动的
网页监控页直接读取该数据库，解决两个进程内存不共享的问题。

运行：
  python main.py            # 跑若干轮 mock 演示（默认 6 轮）
  python main.py 10         # 指定轮数

查看网页监控：
  uvicorn ui.app:app --reload --port 8000
  浏览器打开 http://localhost:8000
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from plc.plc_controller import PLCController
from plc.state_machine import InspectionStateMachine
from capture.camera import CameraCapture
from capture.lighting import Lighting
from detection.detector import SealDetector
from storage.service import StorageService
from storage.database import Database


def build():
    plc = PLCController()
    plc.connect()
    capture = CameraCapture(lighting=Lighting())
    detector = SealDetector()
    storage = StorageService()
    return InspectionStateMachine(plc, capture, detector, storage)


def main():
    sm = build()
    db = Database()
    print("=== 车顶胶条检测系统（沙盒 mock 模式）启动 ===")
    rounds = 6
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        rounds = int(sys.argv[1])
    for i in range(rounds):
        rec = sm.run_once()
        if rec:
            print(f"[结果] 车型={rec.car_model} {'OK' if rec.ok else 'NG'} "
                  f"图片={len(rec.image_refs)} 缺陷={len(rec.defects)}")
        time.sleep(0.2)
    print(f"=== 演示结束，累计记录 {db.count()} 条 ===")


if __name__ == "__main__":
    main()
