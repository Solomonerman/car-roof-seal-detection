# -*- coding: utf-8 -*-
"""PLC 控制器（mock 实现）。

沙盒阶段用模拟信号驱动流程；现场联调时只需把内部实现替换为
pymodbus / python-snap7，对外的 connect / read_car_model /
is_vehicle_moving / write_result 接口保持不变，总控无需改动。
"""
import random
from config import constants as C


class PLCController:
    def __init__(self, host: str = C.PLC_HOST, port: int = C.PLC_PORT):
        self.host, self.port = host, port
        self.connected = False
        self._model = C.DEFAULT_MODEL
        self._moving = False

    def connect(self) -> bool:
        # mock: 直接置为已连接
        self.connected = True
        return True

    def read_car_model(self) -> str:
        # mock: 随机返回一个支持的车型（真实场景读车型寄存器）
        return random.choice(C.SUPPORTED_MODELS)

    def is_vehicle_moving(self) -> bool:
        # mock: 随机模拟车辆移动状态（真实场景读移动状态寄存器）
        self._moving = random.random() > 0.5
        return self._moving

    def write_result(self, ok: bool) -> bool:
        # mock: 打印回写内容（真实场景写结果寄存器）
        print(f"[PLC] 回写检测结果 -> {'OK' if ok else 'NG'}")
        return True

    def close(self):
        self.connected = False
