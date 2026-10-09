# -*- coding: utf-8 -*-
"""车型 → 检测算法 路由。

现场车型代码（DB230.DBD1208，ASCII）：
  - 以 "NM" 开头（如 "NM41" / "NM42" / "NM**"）→ 8X 车型 → 本版胶条检测算法（偏移 + 宽度）
  - 以 "MM" 开头（如 "MM**"）→ 9X 车型 → 本版算法未覆盖，暂不检测（仅拍照存档）
  - 其他（如 "7P24"）→ 未知/未接入车型 → 暂不处理

扩展新车型：在 ROUTING 里加一条 代码前缀→算法 key，并在 get_detector
里为该 key 返回对应检测器即可，总控流程无需改动。
"""
from common.interfaces import DetectionResult, Defect


# 车型代码（前缀）→ 算法 key
ROUTING = {
    "NM": "8X",       # 8X 车型：NM41/NM42/NM** 等所有 NM 开头 → 本版胶条检测算法
    "MM": "9X",       # 9X 车型：本版算法未覆盖，暂不检测（占位）
}


def route_algorithm(model_ascii: str) -> str:
    """返回算法 key：'8X' / '9X' / 'future'（未匹配任何已知规则）。

    先 strip 前后空白（PLC ASCII 字段常为定长、可能带前导/尾随空格，
    如 ' NM4'），否则 startswith 前缀匹配失败、把本应识别的车型误判为未接入。
    """
    m = (model_ascii or "").strip().upper()
    for prefix, key in ROUTING.items():
        if m.startswith(prefix):
            return key
    return "future"


def get_detector(key: str):
    """按算法 key 返回检测器实例；'9X' 及未知 → 返回 None（暂不检测）。"""
    if key == "8X":
        # 现场在线版：偏移 + 宽度检测，与 0921 离线验证同一套算法
        # （roi_locator YOLO/红框回退 + seal_segment + seal_metrics + offset_check_v7）
        from detection.seal_offset_detector import SealOffsetDetector
        return SealOffsetDetector()
    # 9X 及其他：本版算法未覆盖，返回 None，由调用方跳过实际检测（仅拍照存档）
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
