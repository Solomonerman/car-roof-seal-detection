# -*- coding: utf-8 -*-
"""车型 → 检测算法 路由。

现场车型代码（DB230.DBD1208，ASCII）：
  - 以 "MM" 开头（如 "MM**"）→ 9X 车型 → 用现有 SealDetector
  - 其他（如 "7P24"）→ 未来待接入车型 → 暂不处理（留扩展位）

扩展新车型：在 route_algorithm 里加一个分支返回新 key，
并在 get_detector 里为该 key 返回对应检测器即可，总控流程无需改动。
"""
from common.interfaces import DetectionResult, Defect


# 车型前缀 → 算法 key
ROUTING = {
    "MM": "9X",      # 9X 车型：现有胶条检测算法
    # "7P": "FUTURE_B",   # 未来车型：待接入算法后取消注释并补充 get_detector
}


def route_algorithm(model_ascii: str) -> str:
    """返回算法 key：'9X' / 'future'（未匹配任何已知规则）。"""
    m = (model_ascii or "").upper()
    for prefix, key in ROUTING.items():
        if m.startswith(prefix):
            return key
    return "future"


def get_detector(key: str):
    """按算法 key 返回检测器实例；'future' 或未知 → 返回 None（暂不处理）。"""
    if key == "9X":
        from detection.detector import SealDetector
        return SealDetector()
    # 未来车型：算法未实现，返回 None，由调用方跳过
    return None


def make_pending_result(model_ascii: str, message: str) -> DetectionResult:
    """未来车型占位结果（不判 OK/NG，仅记录待复检）。"""
    return DetectionResult(
        car_model=model_ascii,
        ok=True,                       # 不误判 NG；message 明确提示待复检
        defects=[Defect(0, 0, 0, 0, "pending", 0.0,
                        meta={"reason": "算法未实现，待接入后复检"})],
        confidence=0.0,
        message=message,
    )
