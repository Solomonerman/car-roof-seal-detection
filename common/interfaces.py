# -*- coding: utf-8 -*-
"""模块间的数据结构与接口契约。

设计目的：各模块（采集 / 算法 / 存储 / PLC）只通过这些结构通信，
互不依赖彼此的实现细节。改算法、换相机、换数据库都不影响总控流程。
"""
from dataclasses import dataclass, field
from typing import List
from enum import Enum


class CarModel(str, Enum):
    TIGUAN = "Tiguan"
    A5 = "A5"
    UNKNOWN = "Unknown"


@dataclass
class CaptureRequest:
    """总控 -> 采集模块：触发采集的请求。"""
    car_model: str
    count: int = 8
    interval_sec: float = 0.5


@dataclass
class CaptureResult:
    """采集模块 -> 总控：采集结果。"""
    ok: bool
    images: List[str] = field(default_factory=list)   # 采集到的图像路径列表
    message: str = ""


@dataclass
class Defect:
    """单处缺陷/胶条位置的标注。

    label 取值：
      seal      : 正常检出的胶条段（信息，不计入 NG）
      width     : 宽度超出 20±5mm
      missing   : ROI 内整段无胶条
      break     : 断胶（纵向空洞 / 段间间隙离群）
      overspray : 过喷（主胶条附近离散暗斑）
    meta: 附加过程信息（如 width_mm），落库时一并保存。
    """
    x: int
    y: int
    w: int
    h: int
    label: str = "seal"
    confidence: float = 0.0
    meta: dict = None


@dataclass
class DetectionResult:
    """算法模块 -> 总控：检测结果。"""
    car_model: str
    ok: bool                              # 整体判定 OK / NG
    defects: List[Defect] = field(default_factory=list)
    confidence: float = 0.0
    message: str = ""


@dataclass
class InspectionRecord:
    """一条完整的检测记录，用于落库与 UI 展示。"""
    car_model: str
    ok: bool                              # 整体判定 OK / NG
    image_refs: List[str] = field(default_factory=list)
    defects: List[Defect] = field(default_factory=list)
    timestamp: str = ""
    # —— 追溯与筛选字段（PLC 上下文）——
    skid: int = None                      # 滑橇号
    pin: str = ""                         # PIN
    no_paint: bool = False                # NO_Paint=1 表示免检（不拍照）
    captured: bool = False                # 是否拍照纳入检测流程（UI"是否检测"列依据）
