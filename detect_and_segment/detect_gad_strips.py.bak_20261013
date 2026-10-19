"""用训练好的检测器替代人工红框，产出可以喂给 `refine_strip_mask` 的 ROI。

    from detect_gad_strips import detect_strips, make_detector

    detector = make_detector("weights/gad_yolo11n.pt")
    for x, y, w, h in detect_strips(gray, detector):
        roi = gray[y:y + h, x:x + w]
        mask, notes = refine_strip_mask(roi, coarse, (x, y), gray.shape)

命令行自检（只读，产物写 Temp）：

    python detect_gad_strips.py --weights weights/gad_yolo11n.pt
    python detect_gad_strips.py --weights <某折的 best.pt> --fold 0

## 为什么要做「贴边吸附」和「外扩留白」

精修算法里 `EDGE_CAP_BORDER_TOL = 3` 用「ROI 边是不是画面边」来决定该侧要不要做端头收缩。
今天 ROI 就是人手画的红框，人画的时候是**离散**的：贴边就贴到 0~3px，不贴边就离得很远
（275 个框里 68.4% 贴边，`tol∈(3,10]` 之间零样本）。所以那个判据今天"要么一直对、要么一直错"。

检测器给的是**连续**坐标。若直接把 `x=3.4` 和 `x=2.9` 分别传进去，同一张图会走上完全不同的
分支，而且不可预测。所以这里做两件事，**都只往外扩**：

- `snap`：某侧距画面边 ≤ `DET_SNAP_TOL` 就推到画面边，把"贴边"重新变成离散的；
- `pad`：其余侧外扩 `DET_ROI_PAD_X`（左右）/ `DET_ROI_PAD_Y`（上下）。ROI 必须**包住**胶条——包不住是灾难性的
  （胶条被裁掉一段），包多了只是给精修多一点余量。
- `wide_fill`：宽框再水平铺满（`DET_WIDE_FILL`）。`snap` 的容差只有 20px，而视角 `/07` 的
  长胶条上模型框右端距画面边 29~456px，`snap` 吸不到、`pad` 也补不齐（那 5 条要 121~456px，
  其余 49 条最多只要 18px——一个全局 pad 不可能两头都对）。**但 2026-09-22 验收否决了它：
  它会把左右两侧误差放大 5~7 倍**，原因见 `DET_WIDE_FILL` 处的注释。当前定版 **关闭**。

三个数都是先验，最终要靠 `eval_detector_end2end.py` 的三组对照来定。
**改了 `DET_MAX_ASPECT` / `DET_WIDE_FILL` 之后必须重跑 `--groups A,B,C --folds all`**，
只看包住率会漏掉四侧精度有没有被带坏——宽框规则就是这么被否掉的。

## imgsz 为什么是 896（它是被上下限夹出来的区间，不是越大越好）

检测头能表示的框宽是**结构决定**的，与数据量、与视角都无关：

    reg_max=16  ->  DFL 距离分箱 arange(0,16)，单边最多 15 格
    最大 stride 32（P5）  ->  单边最远 15 x 32 = 480 网络像素
    ->  框宽上限 960 网络像素，**这个数与 imgsz 无关**

换到原图坐标要过两道：letterbox 按长边缩放（`gain = imgsz / 长边`，本数据集是 1920），
网络像素除以 `gain`；再被 `ultralytics.utils.ops.scale_boxes` 结尾的
`clip_boxes(boxes, img0_shape)` 裁到画面内。所以

    原图上的宽度上限 = min( 960 x 长边 / imgsz , 画面宽 )

`imgsz` 越大，这个上限**越小**。所以它给的是**上限约束**，与"高度分辨率"给的
下限约束方向正好相反，896 就是同时满足两者的那个区间：

    imgsz   letterbox 后   胶条像素高   原图宽度上限
    1280    1280x800       80px         1440px   <- 咬掉 22 个标签，/07 的塌缩就是这么来的
    896     896x560        56px         1920px   <- min(2057, 1920) = 画面宽，够不着
    640     640x400        40px         1920px   <- 同样够不着，但高度开始吃紧

**闭式判据**：对 W >= H 的横幅帧，`上限 >= 画面宽` 等价于

    imgsz <= 2 x (reg_max - 1) x max_stride        # 16/32 -> 960

也就是说**只要 imgsz <= 960 就没有这个 bug**，896 只是留了 64 的余量（960 是零余量的临界点）。

2026-09-22 实测：全量 158 张 / 283 个预测框，最大宽 1439.5px，**无一个超过 1440**；
而 275 个人工标签里有 22 个比 1440 宽（`/07` 16 个、`/06` 5 个、`/08` 1 个，
分到 5 折是 5/3/5/4/5 个——每折都能看见），**在 1280 下无论怎么训练都学不出来**。
同时 5 折 val mAP50 全是 0.995——mAP@0.5 是瞎的：1439 宽的框对 1912 的目标 IoU 仍有 0.75，
照样算 TP。这是"指标全绿但东西是坏的"的标准案例。

上限由 `Detector.width_cap()` 从模型**实读** `reg_max` / `max_stride`，不写死 16/32。
若以后改走「把 `reg_max` 改成 32」那条路（上限 1984 网络像素，`imgsz <= 1984` 都不约束），
这些函数不用动。

⚠️ 因此**旧文档里"若劣化就放大 imgsz"是错误建议**——放大 imgsz 只会让上限更小。
另外 `DET_WIDE_FILL` / `DET_MAX_ASPECT` 那两段论证**都是在 1280 下测出来的**，
896 重训后必须重扫，不能沿用（见各自常量的注释）。
"""

from __future__ import annotations

import argparse
import json
import re
import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
SCRATCH = Path(r"C:\Users\lx\AppData\Local\Temp\gadbase")
DEFAULT_WEIGHTS = ROOT / "weights" / "gad_yolo11n.pt"

# 一轮训练会写两个检查点，**该用哪个**是有结论的，不是随便挑：
#   best.pt —— 按 `fitness = 0.1*mAP50 + 0.9*mAP50-95` 挑。**这个指标对框宽是瞎的**：
#              `/07` 的框差 100+px 时 mAP50 照样 0.995（IoU 还有 0.75，仍是 TP），
#              所以 `close_mosaic` 之后那 40 轮真尺度把右端推远的改善，它一次都没选上。
#   last.pt —— 最后一轮（第 150 轮，真尺度尾段的终点）。
# 实测（imgsz 896，同一轮训练，C 组验收）：**last 包住 72/74，best 只有 70/74**，
# 而 last 多包住的那两条误差**全在右边界上**（差 18px、差 2px）——正是尾段该修的那一类。
# 所以定版用 last.pt。想回到按 mAP 挑（比如以后换了别的指标），把这里改成 "best.pt" 即可。
DET_WEIGHTS_NAME = "last.pt"

# 推理参数
DET_CONF = 0.25                # 置信度门限。胶条是唯一目标且对比强，不需要压得很低
DET_IOU = 0.60                 # NMS 的 IoU。一条胶条可能被拆成两段（中间被机械臂挡住），
                               # 所以不要把 NMS 调太狠，宁可留两个框
DET_MAX_DET = 10

# 与训练时一致。**这个数有下限也有上限，不是越大越好**——理由见 docstring「imgsz 为什么是 896」：
# 高度分辨率给的是下限（640 时胶条只剩 40px 高），DFL 头的框宽上限给的是上限
# （原图上限 = min(960 x 长边/imgsz, 画面宽)，1280 时只剩 1440px，咬掉 22 个标签）。
# 闭式判据：横幅帧下 imgsz <= 2*(reg_max-1)*max_stride (=960) 即可，896 留了 64 的余量。
DET_IMGSZ = 896

# 读不到模型属性时的兜底值。**兜底会让自检报出一个"看起来很精确"的假数**，
# 所以 read_head_capacity() 会把 source 一起返回，调用方必须把它打出来。
DET_REG_MAX = 16               # ultralytics `tasks.py` 的默认 reg_max
DFL_STRIDE_FALLBACK = 32       # ultralytics 自己也兜底 32

# ROI 外扩（像素）。见模块 docstring 与 eval_detector_end2end.py 的四组对照
# 距画面边 ≤ 这么多像素 -> 吸附到画面边。
# ⚠️ 判的是**原始预测框**，不是外扩后的框（snap_and_pad 是 snap -> pad，见该函数 docstring），
# 所以「有效容差」不是 snap+pad：一条原始框距边 37px 的胶条，snap=20 不吸附，
# pad 只能补回 X 侧 20 / Y 侧 12，最终仍差 17px。
# 20 -> 40 是 imgsz 896 下重扫的结果（此前 20 是在 **1280** 下定的，而 1280 那轮 `/07`
# 的框离画面边永远差 100+px，"贴不贴边"根本没被测过，这个常数等于没验过）：
DET_SNAP_TOL = 40              #   20 -> 72/74，30 -> 72/74（够不着），**40 -> 73/74 且四侧均值一个数没动**，
                               #   50 -> 73/74 但左 3.21->3.89、右 2.84->4.34（不该贴边的框被拖到边上了）。
                               #   所以 40 是拐点，50 越界。改动前重跑：
                               #   eval_detector_end2end.py --folds all --groups C --snap <值>

# ROI 外扩，**横纵分开**（2026-09-24 从单个 DET_ROI_PAD 拆出来的）。人工框上的扫描显示
# 15~25 是拐点——那是**横向**的结论，纵向当时没单独扫过。
#
# **2026-09-29 重扫，两轴一起收到 14/10。** 起因是"框还是偏大，压住了精修"。
# `--pad` 是现成参数，每档跑一次全量 A/B/C（`det_eval/20260929_1433*`~`1438*`）：
#     pad     包住   C上    C下    C左    C右
#     20,12   99%   1.88   1.83   3.19   2.84      <- 旧定版
#     20,10   99%   1.88   1.54   3.44   2.80
#     16,10   99%   1.78   1.46   2.42   2.14
#     14,10   99%   1.78   1.46   2.35   2.02      <- 新定版
#     12,10   97%   1.75   1.42   2.58   2.14      <- 出局
#     10,10   95%   1.72   1.42   2.81   2.10
# 结论：**收小反而全面变好**（C左 3.19 -> 2.35、C右 2.84 -> 2.02），而且
# 收 pad_x 的贡献比 pad_y 大得多。12 以下翻车——不只是包住率掉，**C左还反弹回升**
# （2.35 -> 2.58 -> 2.81），因为 ROI 开始啃进胶条、端头逻辑反而乱。14 是拐点。
#
# ⚠️ **上面那个 99% 是把风险抹平了的**，它用的是各折权重。换成预览批次的
# **alldata 权重**逐个人工框数，旧定版 20,12 其实只有 269/275 = **98%**（已漏 6 个）；
#     20,12 269/275   |  16,10 268/275（新增 `09866_8X/08#0`，左超出 16px）
#     14,10 267/275（再增 `09894_8X/08#0`，上 7px）| 12,10 266/275（再增 `08869_8X/07#0`）
# 即 **pad_y 12→10 本身要付一个框的代价**，pad_x 20→14 是白送的。逐侧量过超出量：
# 全是 7~16px，不是几百——人工框是胶条的松代理，这个量级大多是"框比胶条松"而非真裁掉胶条。
# 但这几台车身没有真值多边形，**残留风险无法量化**（查 `b3_overhang.py` / `b3_truth_outside.py`）。
# 改动前重跑：
#   eval_detector_end2end.py --folds all --groups A,B,C --pad <X>,<Y>
DET_ROI_PAD_X = 14             # 左右。旧值 20 是"人工框上 15~25"那条先验，2026-09-29 被实测推翻。
DET_ROI_PAD_Y = 10             # 上下。旧值 12（再往前 20）。8 仍**出局**：包住率掉到 96%（3/74 漏），
                               #   那是胶条被裁掉一段、不可逆，换不来什么。10 是本次实测的落点。
                               #   ⚠️ pad 只是把端头方块的**天花板**压低，**清不掉它**——
                               #   真正清方块的是 `EDGE_OUTER_PEN`（见 crop_and_segment.py:184）。

# 宽框规则：预测框宽 >= 这个比例 × 图宽 时，认为胶条横跨画面，ROI 水平方向铺到两个画面边。
# 0 = 关闭，**这就是当前定版值**。
#
# ⚠️ **下面这一整段论证都是 imgsz=1280 下测出来的，896 重训后必须重扫。**
# 「模型框宽塌缩在 1423~1439px」正是 1440 那个上限的症状，**不是检测器的性质**；
# 896 之后那 5 条 /07 长胶条应该能预测到约 1900px，`snap`（20px 容差）可能直接吸得住，
# 那么这条规则的**原始动机可能整体蒸发**。禁用它的理由（端头收缩被误关）与上限无关，
# 即使重扫确认动机消失，也不要顺手把它打开——要打开得先解决那个机制问题。
#
# 起因：视角 /07 的 5 条长胶条，模型框宽塌缩在 1423~1439px（74~75% 图宽，5 张独立图、
# 3 个折），而人工框右端全部贴画面右边，缺口 29~456px——`snap`(20) 吸不到、`pad` 也补不齐。
# 全量 5 折审计：触发 42 次，42 次全部落在有人工红框的位置上（0 次纯背景外扩）。
#
# **2026-09-22 验收否决**（`det_eval/20260922_1410` 对 `20260922_1420`，differ 只有这一个开关）：
#     包住率 68/74 -> 73/74（换回 5 条，就是那五条 /07）
#     但 左|均| 5.67 -> 32.04px、右|均| 3.03 -> 21.86px（5.6x / 7.2x）,
#        左>5px 13.32% -> 18.92%、右>5px 14.24% -> 17.33%
#     上下两侧几乎不动（2.27->2.26、4.16->4.07）——铺满只改 x 不改 y。
# 机制：`EDGE_CAP_BORDER_TOL = 3` 用「ROI 边是不是画面边」决定该侧要不要做端头收缩。
# 把 ROI 边推到画面边 = 告诉精修"这一侧是画面边" = **关掉该侧的端头收缩**，而 /07 那几条
# 左侧是凌空收尾的，于是边界外冲约 30px。**这是只看包住率看不出来的那类劣化。**
# 要再启用它，必须先解决端头收缩被误关的问题，不能只调阈值。
DET_WIDE_FILL = 0.0

# 形状过滤：胶条是"很扁的横条"（人工框里 宽/画幅 中位 0.67，高/画幅 中位 0.10）
# h/w 超过这个值就不像胶条。
# ⚠️ **这条也是 imgsz=1280 下测出来的，896 重训后必须重扫**：被拒的那 32 个框里
# 有多少是 DFL 截断的产物（框宽被压到 1440 以内 → h/w 被动变大），1280 下看不出来。
#
# **2026-09-22 全量 5 折审计后由 0.5 抬到 1.00**：
# 被它拒掉的 32 个框 100% 贴画面边（它实际是个纯粹的"画面边短桩"闸门，不是形状闸门），
# 其中 25 个与人工标注紧配（IoU 中位 0.93~0.98）= 误杀。抬到 1.00 拿回 18 个、混进 0 个
# 谁都盖不住的框；再往上（1.20）就开始收 IoU=0.00 的杂物了，1.00 是这笔账的拐点。
DET_MAX_ASPECT = 1.00
DET_MIN_AREA_PCT = 0.005       # 面积占比下限，挡掉小目标误检

# 碎片覆盖比：predict_boxes 去重时，若一个框「交叠面积 / 自身面积」>= 此值，
# 认为它是被另一框包住的「小碎片」，丢小的留大的。专门吃掉「贴在真框上的细条」类假阳性
# （如 07 左胶条 S0：与真框 IoU 仅 0.32，但 74% 落在真框内，用覆盖比才抓得到）。
# 0.7 是保守值：真正相邻 / 被机械臂挡开的两段，交叠 / 小框面积比远低于此，不会误并。
# 改动后重跑 eval_detector_end2end.py --folds all --groups A,B,C 确认四侧精度与包住率没被带坏。
DET_SUBBOX_COVER = 0.70


def dfl_width_limit(reg_max: int, max_stride: int) -> float:
    """DFL 头能表示的最大框宽（**网络像素**）。左右各最多 `reg_max - 1` 格。"""
    return 2.0 * (reg_max - 1) * max_stride


def frames_needed_imgsz(reg_max: int, max_stride: int) -> float:
    """横幅帧下"宽度上限不再咬人"的 imgsz 门槛。

    把 `width_cap_px` 的 `limit / gain >= 画面宽` 解出来：W >= H 时 `gain = imgsz / W`，
    画面宽与长边都是 W，约掉，剩下这个与帧尺寸无关的常数。16/32 -> 960。
    用来把提示写成一句可执行的话。
    """
    return dfl_width_limit(reg_max, max_stride)


def width_cap_px(shape: tuple[int, int], imgsz: int, reg_max: int, max_stride: int) -> float:
    """一帧在该 imgsz 下**最大可表示**的框宽（原图像素）。推导见模块 docstring。

    两道换算：letterbox 的 `gain = imgsz / 长边`（长边取 max(H, W)，与 ultralytics 的方阵
    letterbox 一致），以及 `ultralytics.utils.ops.scale_boxes` 结尾把框裁进画面。
    **比这个宽的胶条不是"没训好"，是物理上表达不出来。**
    """
    height, width = shape[:2]
    gain = imgsz / max(height, width)
    return min(dfl_width_limit(reg_max, max_stride) / gain, float(width))


def read_head_capacity(model: object) -> tuple[int, int, str]:
    """从**实际模型**上读 `(reg_max, max_stride, source)`，不写死 16/32。

    `source` 是 `"model"`（两个数都读到了）或 `"fallback"`（至少一个用了先验值）。
    **调用方必须把 source 打出来**——静默兜底会让自检报出一个"看起来很精确"的假数，
    而那正是这次要根治的那类病症（指标全绿、东西是坏的）。
    """
    reg_max = stride_max = 0
    try:
        net = model.model                      # YOLO(...).model -> DetectionModel
        head = net.model[-1]                   # Detect
        # `Detect.reg_max` 是显式属性（head.py: self.reg_max = reg_max）；
        # `dfl.c1` 作退路，但 reg_max=1 时 dfl 会变成 nn.Identity（没有 c1），所以先读属性。
        reg_max = int(getattr(head, "reg_max", 0)
                      or getattr(getattr(head, "dfl", None), "c1", 0))
        stride = getattr(net, "stride", None)
        if stride is not None:
            stride_max = int(max(float(s) for s in stride))   # 不假设 P3/P4/P5 的顺序
    except Exception:
        pass
    source = "model"
    if reg_max <= 1:                           # 未 build 或读不到 -> 退回先验
        reg_max, source = DET_REG_MAX, "fallback"
    if stride_max <= 0:
        stride_max, source = DFL_STRIDE_FALLBACK, "fallback"
    return reg_max, stride_max, source


@dataclass
class Detector:
    """模型 + 每次调用都要带的参数 + 从模型实读的框宽容量。

    **不要**把这些参数塞进 `model.overrides`：`ultralytics.engine.model.Model.predict` 里是
    `args = {**self.overrides, **custom, **kwargs}`，而 `custom` 里硬编码了 `conf: 0.25`，
    合并顺序上在 `self.overrides` 之后，会把 override 里的 conf 打回 0.25。
    只有**调用时传的关键字**优先级最高。所以这里存下来，在 `detect_strips` 里显式传。

    `reg_max` / `max_stride` 由 `make_detector` 从模型实读（不是常量），所以"把 reg_max
    改成 32"那条路不需要改这里。
    """
    model: object
    weights: Path
    conf: float = DET_CONF
    iou: float = DET_IOU
    imgsz: int = DET_IMGSZ
    reg_max: int = DET_REG_MAX
    max_stride: int = DFL_STRIDE_FALLBACK
    capacity_source: str = "fallback"

    def width_cap(self, shape: tuple[int, int]) -> float:
        """这个 imgsz 下本帧能表示的框宽上限（原图像素）。比它宽的胶条必然被截断。"""
        return width_cap_px(shape, self.imgsz, self.reg_max, self.max_stride)

    def imgsz_without_cap(self) -> float:
        """宽度上限不再咬人的 imgsz 门槛（16/32 -> 960）。见 `frames_needed_imgsz`。"""
        return frames_needed_imgsz(self.reg_max, self.max_stride)


def print_capacity(detector: Detector) -> None:
    """把模型的框宽容量打出来。

    **这一行以前在日志里根本不存在，所以那个 bug 藏了很久**：mAP 全绿、包住率只差 5 条，
    没有任何一个数字在说"模型表达不了这个宽度"。现在每次加载模型它都会出现。
    """
    limit = dfl_width_limit(detector.reg_max, detector.max_stride)
    need = frames_needed_imgsz(detector.reg_max, detector.max_stride)
    if detector.capacity_source == "fallback":
        print(f"[容量] reg_max={detector.reg_max} max_stride={detector.max_stride}"
              f" <- **兜底值，没从模型上读到** -> 网络像素上限 {limit:.0f}"
              f"；imgsz 需 <= {need:.0f}（这个数不可信，先查模型为什么读不到）")
        return
    state = "OK" if detector.imgsz <= need else "!! 上限会咬人：调小 imgsz"
    print(f"[容量] reg_max={detector.reg_max} max_stride={detector.max_stride}"
          f" -> 网络像素上限 {limit:.0f}；imgsz 需 <= {need:.0f}，当前 {detector.imgsz}  [{state}]")
    # run 目录名里编码了训练时的 imgsz（`..._img896_e150`）。与本次推理的 imgsz 不一致时
    # **必须喊出来**：`weights_for()` 会按轮数挑出 896 的权重，而手传的 `--imgsz 1280`
    # 会把上限打回 1440，那个 bug 原样复现，且没有任何一个指标会变难看。
    trained = re.search(r"img(\d+)", detector.weights.parent.parent.name)
    if trained and int(trained.group(1)) != detector.imgsz:
        print(f"[!] 权重是按 imgsz={trained.group(1)} 训的，这次却用 imgsz={detector.imgsz} 推理。"
              f"两者不一致会让宽度上限对不上（推理端更小 = 反而更安全，更大 = 会截断）。"
              f"除非你在刻意扫 imgsz，否则把两者对齐。")


def make_detector(weights: Path | str = DEFAULT_WEIGHTS, conf: float = DET_CONF,
                  iou: float = DET_IOU, imgsz: int = DET_IMGSZ) -> Detector:
    """加载模型。返回可复用的 `Detector`——每张图都重新 load 会很慢。

    顺便把模型的框宽容量打出来（`print_capacity`）：这个数以前在日志里不存在，
    所以"模型表达不了 /07 的宽度"这件事一直没人看见。
    """
    from ultralytics import YOLO

    weights = Path(weights)
    if not weights.exists():
        raise FileNotFoundError(f"找不到权重 {weights}，先跑 train_gad_detector.py")
    yolo = YOLO(str(weights))
    reg_max, max_stride, source = read_head_capacity(yolo)
    detector = Detector(yolo, weights, conf=conf, iou=iou, imgsz=imgsz,
                        reg_max=reg_max, max_stride=max_stride, capacity_source=source)
    print_capacity(detector)
    return detector


def snap_and_pad(box: tuple[float, float, float, float], shape: tuple[int, int],
                 pad_x: int = DET_ROI_PAD_X, pad_y: int = DET_ROI_PAD_Y,
                 snap: int = DET_SNAP_TOL,
                 wide_fill: float = DET_WIDE_FILL) -> tuple[int, int, int, int]:
    """把预测框变成 ROI：贴边的侧吸附到画面边，其余侧外扩，宽框再水平铺满，最后裁进画面。

    `box` 是 **xyxy**，返回 **xywh**（下游 `red_boxes` / `contains` / 精修都用 xywh）。
    `pad_x` / `pad_y` 分开给，因为实测**纵向能收到 12 而横向不能**（见常量块的扫描数据）。
    想复现拆分前"一个 pad 管四边"的行为，传 `pad_x=pad_y=<同一个数>`。
    `wide_fill > 0` 时，若外扩后的宽度已 >= `wide_fill × 画面宽`，左右两端都推到画面边。
    三条规则**都只会让 ROI 变大**（唯一"不对外扩"的动作是裁进画面），不会裁掉胶条；
    顺序是 snap -> pad -> wide_fill，`wide_fill` 判的是**外扩之后**的宽度。
    """
    x0, y0, x1, y1 = box
    height, width = shape[:2]
    if x0 <= snap:
        x0 = 0.0
    if y0 <= snap:
        y0 = 0.0
    if width - x1 <= snap:
        x1 = float(width)
    if height - y1 <= snap:
        y1 = float(height)
    left = max(0, int(round(x0 - pad_x)))
    top = max(0, int(round(y0 - pad_y)))
    right = min(width, int(round(x1 + pad_x)))
    bottom = min(height, int(round(y1 + pad_y)))
    if wide_fill > 0 and right - left >= wide_fill * width:
        left, right = 0, width
    return left, top, right - left, bottom - top


def keep_box(w: int, h: int, shape: tuple[int, int]) -> bool:
    """形状过滤。只挡明显不像胶条的，挡多了会把漏检伪装成"过滤掉了"。"""
    if w <= 0 or h <= 0:
        return False
    if h / w > DET_MAX_ASPECT:
        return False
    return (w * h) / (shape[0] * shape[1]) >= DET_MIN_AREA_PCT


def predict_boxes(image: np.ndarray, detector: Detector) -> list[tuple[float, float, float, float]]:
    """原始预测框（xyxy 浮点，已过形状过滤、已去重、已按置信度截断），不做 snap/pad。

    拆出这一步是为了让 `eval_detector_end2end.py` 扫 pad/snap 时不必重复推理
    （CPU 上一次 1280 的前向要几十毫秒）。
    """
    # conf/iou/imgsz 必须在这里传，不能靠 detector.model.overrides（见 Detector 的注释）
    result = detector.model(image, conf=detector.conf, iou=detector.iou,
                            imgsz=detector.imgsz, verbose=False, device="cpu")[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []
    found: list[tuple[tuple[float, float, float, float], float]] = []
    for xyxy, conf in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
        x0, y0, x1, y1 = (float(v) for v in xyxy)
        if not keep_box(int(round(x1 - x0)), int(round(y1 - y0)), image.shape):
            continue
        found.append(((x0, y0, x1, y1), float(conf)))
    # 同一位置的重复框（NMS 之后仍可能留下 IoU 略低于门限的一对）按 IoU 去重，
    # 保留置信度高的那个：否则同一条胶条会当成两条送去精修。
    # 另加「碎片覆盖」判定：若候选框 / 已留框基本被对方包住（覆盖比 >= DET_SUBBOX_COVER），
    # 丢小的留大的——专门吃掉「贴在真框上的细条」类假阳性（见 is_subbox 注释）。
    # 两个方向都查，无论高置信框先到还是后到，最终都只留大的那一个。
    found.sort(key=lambda item: -item[1])
    kept: list[tuple[float, float, float, float]] = []
    for box, _ in found:
        drop_box = False
        survivors = []
        for other in kept:
            if iou_xyxy(box, other) >= 0.5:          # 标准重复：丢候选、留已存（更高 conf）
                drop_box = True
                survivors.append(other)
                continue
            if is_subbox(box, other):                 # 候选是小碎片、被已存框包住 -> 丢候选
                drop_box = True
                survivors.append(other)
                continue
            if is_subbox(other, box):                 # 已存框是小碎片、被候选框包住 -> 丢已存、留候选
                continue
            survivors.append(other)
        if drop_box:
            kept = survivors
            continue
        kept = survivors
        kept.append(box)
    return kept[:DET_MAX_DET]


def snap_and_pad_boxes(boxes: list[tuple[float, float, float, float]],
                       shape: tuple[int, int],
                       pad_x: int = DET_ROI_PAD_X, pad_y: int = DET_ROI_PAD_Y,
                       snap: int = DET_SNAP_TOL,
                       wide_fill: float = DET_WIDE_FILL) -> list[tuple[int, int, int, int]]:
    """裸框 -> ROI（snap -> pad -> wide_fill），滤掉零尺寸，按 (top, left) 排。

    **这是"取窗"的唯一实现**，`detect_strips` 只是它套上 `predict_boxes` 的一层壳。
    单独拆出来是给 `run_batch_detect.py` 用的：那个脚本既要 ROI **又要**裸框
    （红框图和报告里的 `det_boxes_raw` 都要它），若它自己再写一遍 snap/pad，
    就等于把这条路径抄成两份——抄漏过一次：批处理脚本里把 `pad_x/pad_y` 写死成
    `DET_ROI_PAD_X/Y`，于是 `--pad-x/--pad-y` 只改了打印、没改行为，两批图逐位相同
    （2026-09-29 踩过）。所以让它调这里，而不是让它"照抄一遍但记得传参"。

    顺序按 (top, left) 排，与 `crop_and_segment.map_boxes` 的
    `sorted(found, key=lambda b: (b[1], b[0]))` 一致——下游的「第 N 个框」才指同一个东西。
    """
    rois = [snap_and_pad(box, shape, pad_x=pad_x, pad_y=pad_y, snap=snap,
                         wide_fill=wide_fill)
            for box in boxes]
    return sorted((r for r in rois if r[2] > 0 and r[3] > 0), key=lambda b: (b[1], b[0]))


def detect_strips(image: np.ndarray, detector: Detector,
                  pad_x: int = DET_ROI_PAD_X, pad_y: int = DET_ROI_PAD_Y,
                  snap: int = DET_SNAP_TOL,
                  wide_fill: float = DET_WIDE_FILL) -> list[tuple[int, int, int, int]]:
    """一张灰度/彩色图 -> 原生坐标的 ROI 列表，已做贴边吸附、外扩与宽框铺满。

    顺序按 (top, left) 排，与 `crop_and_segment.map_boxes` 的
    `sorted(found, key=lambda b: (b[1], b[0]))` 一致——下游的「第 N 个框」才指同一个东西。
    """
    _warn_if_capped(image, detector)
    return snap_and_pad_boxes(predict_boxes(image, detector), image.shape,
                              pad_x=pad_x, pad_y=pad_y, snap=snap, wide_fill=wide_fill)


_cap_warned = False


def _warn_if_capped(image: np.ndarray, detector: Detector) -> None:
    """画面比模型能表示的框还宽时喊一声，**整个进程只喊一次**。

    这里刻意**不抛异常**：一个已经训好、正在服务的模型不该因为配置不理想而拒绝推理，
    而且"够不够"取决于每次调用都能改的 `imgsz`，那是策略不是前置条件。
    真正的症状检测是"预测框贴着上限"（饱和），那属于统计，见 `label_width_report` 与
    `train_gad_detector.py` 的训练后自检。这一行只是让配置错误当场可见。
    """
    global _cap_warned
    if _cap_warned:
        return
    _cap_warned = True
    cap = detector.width_cap(image.shape)
    width = image.shape[1]
    if cap < width:
        print(f"[!] imgsz={detector.imgsz} 下模型能表示的最大框宽只有 {cap:.0f}px，"
              f"而画面宽 {width}px：横向超过 {cap:.0f}px 的胶条本图**必然被截断**。"
              f"修法：把 imgsz 调到 <= {detector.imgsz_without_cap():.0f}"
              f"（或把 reg_max 提到 32 重训）。")


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    """`(x0, y0, x1, y1)` 的 IoU。整数框（ROI）和浮点框（原始预测）都用它。"""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    iy = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = ix * iy
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / union if union > 0 else 0.0


def _box_area(b: Sequence[float]) -> float:
    """xyxy 框面积（非负）。"""
    w = b[2] - b[0]
    h = b[3] - b[1]
    return w * h if w > 0 and h > 0 else 0.0


def _box_inter_area(a: Sequence[float], b: Sequence[float]) -> float:
    """xyxy 两框交叠面积（非负）。"""
    ix = min(a[2], b[2]) - max(a[0], b[0])
    iy = min(a[3], b[3]) - max(a[1], b[1])
    return max(0.0, ix) * max(0.0, iy)


def _x_overlap_ratio(a: Sequence[float], b: Sequence[float]) -> float:
    """两 xyxy 框 X 跨度的交叠比（交叠 / 较短 X 跨度），用于判断「是不是同一条横胶条」。"""
    inter = min(a[2], b[2]) - max(a[0], b[0])
    if inter <= 0:
        return 0.0
    min_span = min(a[2] - a[0], b[2] - b[0])
    return inter / min_span if min_span > 0 else 0.0


def is_subbox(small: Sequence[float], big: Sequence[float]) -> bool:
    """`small` 是否应被视作 `big` 的重复碎片而合并掉（保留 `big`）。

    两类情况（任一满足即判冗余）：
    (1) 覆盖比高：`small` 大部分落在 `big` 内（交叠 / small 面积 >= DET_SUBBOX_COVER）；
    (2) 同 X 跨度：两者 X 跨度几乎重合（同一条横胶条），`small` 明显更小且 Y 有交叠——
        专门抓 07 左胶条 S0：宽同真框、仅半高，覆盖比仅 0.67（X 顶部探出 22px），
        但 X 重合 0.999、面积比 0.42，靠这条才抓得到。
    两类都不动真正相邻 / 被机械臂挡开的两段（它们 X 不重叠、覆盖比低）。
    判的是「候选框相对已留框的谁包谁」，两个方向都查，处理顺序无关。
    """
    a_s, a_b = _box_area(small), _box_area(big)
    if a_s <= 0 or a_b <= 0:
        return False
    inter = _box_inter_area(small, big)
    if inter <= 0:
        return False
    if inter / a_s >= DET_SUBBOX_COVER:                      # 情形(1)
        return True
    if _x_overlap_ratio(small, big) >= 0.9 and a_s / a_b <= 0.6:   # 情形(2)
        return True
    return False


# --------------------------------------------------------------------------- 命令行自检


def red_boxes(png: Path, image_shape: tuple[int, int]) -> list[tuple[int, int, int, int]]:
    """人工红框，口径与生产一致（`extract_boxes` + `map_boxes`）。只在自检里用，不参与推理。"""
    from crop_and_segment import extract_boxes, map_boxes

    annotated = cv2.imdecode(np.fromfile(png, np.uint8), cv2.IMREAD_COLOR)
    _, red = extract_boxes(png)
    return sorted(map_boxes(red, annotated.shape[:2], image_shape), key=lambda b: (b[1], b[0]))


def bmp_size(path: Path) -> tuple[int, int]:
    """只读 BMP 头拿 `(宽, 高)`。**不解图**——全量 347MB，自检不该为此付代价。

    BMP 头 18/22 字节处是两个 int32。编码器可能写成负数表示自上而下存，
    所以取绝对值。`folds.json` 的 `image_sizes` 是同一件事的聚合版，可用来交叉校验。
    """
    with open(path, "rb") as fh:
        head = fh.read(26)
    width, height = struct.unpack_from("<ii", head, 18)
    return abs(width), abs(height)


def label_width_report(imgsz: int, reg_max: int = DET_REG_MAX,
                       max_stride: int = DFL_STRIDE_FALLBACK,
                       dataset: Path | None = None) -> dict:
    """训练标签里的框宽 vs 模型在该 imgsz 下的宽度上限。

    这里问的是"**训练集要求模型拟合什么**"，所以用喂给模型的那一份
    （`det_dataset/labels/*.txt`），不是标注 PNG。**不要**在这里引入第三份"红框在哪"的
    实现——仓库已经因为两套提取口径差 5 倍吃过一次亏，而这个数不需要图。

    不传 detector 而是收 `imgsz/reg_max/max_stride`，因为**训练前**就该能跑（那时还没有
    模型，用 16/32 的先验），那才是 0 GPU 时间发现问题的地方。
    """
    dataset = dataset or (ROOT / "det_dataset")
    per_image: list[tuple[str, float, float]] = []       # (图名, 最大标签宽px, 该图上限)
    n_labels = n_over = n_near = 0
    for txt in sorted((dataset / "labels").glob("*.txt")):
        bmp = dataset / "images" / f"{txt.stem}.bmp"
        if not bmp.exists():
            continue
        width, height = bmp_size(bmp)
        cap = width_cap_px((height, width), imgsz, reg_max, max_stride)
        widest = 0.0
        for line in txt.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            n_labels += 1
            w_px = float(parts[3]) * width
            widest = max(widest, w_px)
            if w_px > cap + 0.5:                         # +0.5 免浮点等号
                n_over += 1
            elif w_px > 0.98 * cap:                      # 能学，但会系统性偏窄
                n_near += 1
        if widest > 0:
            # 全名是 `<批次>__<车身>__<图名>`，图名本身能有 80 字符，打全了没法看。
            # 缩成 `车身尾号/视角`，与本脚本 CLI 里 `f"{body[-6:]}/{stem[:2]}"` 一个口径。
            # **必须 maxsplit=2**：图名内部还有好几段 `__`，不限制就会取到 `p7b3be43` 那种串。
            parts = txt.stem.split("__", 2)
            tag = f"{parts[1][-6:]}/{parts[2][:2]}" if len(parts) >= 3 else txt.stem
            per_image.append((tag, widest, cap))
    worst = max(per_image, key=lambda r: r[1] - r[2], default=None)
    return {
        "imgsz": imgsz, "reg_max": reg_max, "max_stride": max_stride,
        "n_labels": n_labels, "n_over": n_over, "n_near": n_near,
        "worst_over": per_image and max(0.0, worst[1] - worst[2]),
        "worst_image": worst[0] if worst else None,
        "worst_width": worst[1] if worst else 0.0,
        "worst_cap": worst[2] if worst else 0.0,
        "n_affected": sum(1 for _n, wid, cap in per_image if wid > cap + 0.5),
    }


def print_label_width_report(rep: dict) -> None:
    """把 `label_width_report` 的结果打成几行。`n_over > 0` 是**配置问题**，不是训练问题。"""
    print(f"[标签宽度] imgsz={rep['imgsz']} reg_max={rep['reg_max']} max_stride={rep['max_stride']}"
          f" -> 上限随帧尺寸浮动，最紧的一帧 {rep['worst_cap']:.0f}px")
    print(f"   标签 {rep['n_labels']} 个：超过上限 {rep['n_over']} 个"
          f"（期望 0）、贴着上限(>=98%) {rep['n_near']} 个")
    if rep["n_over"]:
        print(f"   [!] 有 {rep['n_over']} 个标签超过上限、分布在 {rep['n_affected']} 张图上，"
              f"**这些标签在当前配置下无论怎么训练都学不出来**（不是数据量问题）。")
        print(f"       最宽的是 {rep['worst_image']}：{rep['worst_width']:.0f}px vs 上限 "
              f"{rep['worst_cap']:.0f}px，缺口 {rep['worst_over']:.0f}px"
              f"——那个缺口就是 ROI 会把胶条切掉的长度。")
        print(f"       修法：imgsz <= {frames_needed_imgsz(rep['reg_max'], rep['max_stride']):.0f}"
              f"，或把 reg_max 提到 32 重训。")


def main() -> None:
    parser = argparse.ArgumentParser(description="检测器推理自检：预测框 vs 人工红框")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--fold", type=int, default=None, help="只看某一折的验证集车身")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=DET_IMGSZ)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, str(ROOT))
    from crop_and_segment import LABEL_DIR_NAME

    dataset = ROOT / "det_dataset"
    meta = json.loads((dataset / "folds.json").read_text(encoding="utf-8"))
    wanted = set(meta["folds"][str(args.fold)]) if args.fold is not None else None

    detector = make_detector(args.weights, imgsz=args.imgsz)
    # 上限和"有多少标签超出上限"永远打：这个数是"配置问题还是训练问题"的分界线
    print_label_width_report(label_width_report(args.imgsz, detector.reg_max,
                                                detector.max_stride, dataset))
    SCRATCH.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")

    # 覆盖度检查：预测 ROI 有没有把人工框包住。包不住是灾难性的，单独统计。
    total = covered = 0
    rows = []
    for image_path in sorted((dataset / "images").glob("*.bmp")):
        batch, body, stem = image_path.stem.split("__", 2)
        if wanted is not None and f"{batch}__{body}" not in wanted:
            continue
        gray = cv2.imdecode(np.fromfile(image_path, np.uint8), cv2.IMREAD_GRAYSCALE)
        pred = detect_strips(gray, detector)
        truth = red_boxes(ROOT / "8X" / batch / body / LABEL_DIR_NAME / f"{stem}_已标注.png",
                          gray.shape)
        for box in truth:
            total += 1
            if any(contains(p, box) for p in pred):
                covered += 1
        rows.append((f"{body[-6:]}/{stem[:2]}", truth, pred, gray))
        if len(rows) >= args.limit:
            break

    print(f"权重 {args.weights}")
    print(f"抽查 {len(rows)} 张：人工框 {total} 个，其中被某个预测 ROI 完全包住 {covered} 个"
          f"（{covered / max(total, 1):.0%}）")
    for tag, truth, pred, gray in rows:
        print(f"  {tag:<12} 人工 {len(truth)} 个 {[b[2] for b in truth]}px 宽"
              f"   预测 {len(pred)} 个 {[b[2] for b in pred]}px 宽")
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        for x, y, w, h in truth:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 3)       # 红 = 人工
        for x, y, w, h in pred:
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 1)     # 黄 = 预测
        out = SCRATCH / f"_detpred_{tag.replace('/', '_')}_{stamp}.png"
        cv2.imencode(".png", vis)[1].tofile(out)
    print(f"覆盖图写到 {SCRATCH}\\_detpred_*_{stamp}.png（红=人工红框，黄=预测 ROI）")


def contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    """outer 有没有把 inner 完全包进去。"""
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    return ox <= ix and oy <= iy and ox + ow >= ix + iw and oy + oh >= iy + ih


if __name__ == "__main__":
    main()
