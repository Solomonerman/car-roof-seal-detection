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

    # ⚠️ TODO（项目待办，切勿擅自动手）：
    #   结果回写（OK/NG）的 DB 写地址【尚未向现场确认】，用户明确要求"先记下来、现在不写"。
    #   真实实现替换本 mock 时，write_result 会调用 snap7 的【写】操作——属于高风险动作：
    #   必须在用户明确授权、且最好非生产时段，单独评审后方可启用。当前任何代码都不得写 PLC。
    #   待确认项：写回用哪个 DB 块/偏移？是否需要回写滑橇号/PIN 以便与来车对应？

    def close(self):
        self.connected = False
