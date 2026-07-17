# -*- coding: utf-8 -*-
"""总控状态机：把流程图转成可执行的流程编排。

状态流转：
  INIT -> MONITOR_PLC -> CHECK_MODEL
       -> (车型非法) 回到 MONITOR_PLC
       -> (车型合法) CAPTURE -> DETECT -> STORE -> WRITE_BACK -> 循环

总控只负责"编排与决策"，不实现任何硬件 / 算法细节，
所有具体动作都委托给注入的 plc / capture / detector / storage 对象。
"""
from config import constants as C
from common.interfaces import CaptureRequest, InspectionRecord


class InspectionStateMachine:
    def __init__(self, plc, capture, detector, storage):
        self.plc = plc
        self.capture = capture
        self.detector = detector
        self.storage = storage
        self.state = "INIT"

    def run_once(self):
        """执行一轮完整检测流程，返回 InspectionRecord；未触发检测则返回 None。"""
        # MONITOR_PLC：车辆还在移动则等待
        if self.plc.is_vehicle_moving():
            return None

        # CHECK_MODEL：车型非法则忽略
        model = self.plc.read_car_model()
        if model not in C.SUPPORTED_MODELS:
            print(f"[总控] 车型 {model} 非法，忽略本次")
            return None

        # CAPTURE：启动取流 + 照明，按时间节点采集 8 张
        self.capture.start()
        cap = self.capture.capture(CaptureRequest(
            car_model=model, count=C.CAPTURE_COUNT, interval_sec=C.CAPTURE_INTERVAL_SEC))
        self.capture.stop()
        if not cap.ok:
            print(f"[总控] 采集失败: {cap.message}")
            return None

        # DETECT：按车型检测
        det = self.detector.detect(model, cap.images)

        # STORE：落库 + 存图
        rec = self.storage.save(model, det, cap.images)

        # WRITE_BACK：结果回写 PLC
        self.plc.write_result(det.ok)
        return rec
