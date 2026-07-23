# -*- coding: utf-8 -*-
"""车型 → 检测算法 路由。

现场车型代码（DB230.DBD1208，ASCII）：
  - 以 "MM" 开头（如 "MM**"）→ 9X 车型 → 用现有 SealDetector
  - "NM41" / "NM42" → 8X 车型（同一套检测程序）→ 算法尚未实现，预留
  - 其他（如 "7P24"）→ 未知/未接入车型 → 暂不处理

扩展新车型：在 ROUTING 里加一条 代码前缀→算法 key，并在 get_detector
里为该 key 返回对应检测器即可，总控流程无需改动。
"""
from common.interfaces import DetectionResult, Defect


# 车型代码（前缀）→ 算法 key
ROUTING = {
    "MM": "9X",       # 9X 车型：现有胶条检测算法
    "NM41": "8X",     # 8X 车型：算法未实现，预留（与 NM42 同一套程序）
    "NM42": "8X",     # 8X 车型：同上
}


def route_algorithm(model_ascii: str) -> str:
    """返回算法 key：'9X' / '8X' / 'future'（未匹配任何已知规则）。"""
    m = (model_ascii or "").upper()
    for prefix, key in ROUTING.items():
        if m.startswith(prefix):
            return key
    return "future"


def get_detector(key: str):
    """按算法 key 返回检测器实例；'8X' 及未知 → 返回 None（预留/暂不处理）。"""
    if key == "9X":
        from detection.detector import SealDetector
        return SealDetector()
    # 8X 及其他：算法未实现，返回 None，由调用方跳过实际检测（仅拍照存档）
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
