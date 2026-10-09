"""按绿框裁剪、红框内深色 GAD 胶条分割。

针对 8X 工位的单通道灰度 BMP 图像。**曝光组不再写进文件名**：20260914 起现场换过
照明，那之后的数据不是 4000us 曝光组了，而本算法对两组都适用（参数在 4000us 那批上
标定，换组后是否要重标见 `GAD胶条边界精修算法_说明文档.md`）。流程：
1. 从标注 PNG 中提取绿色裁剪框与全部红色 GAD 区域框；
2. 按尺寸比例将框坐标映射到原始 BMP；
3. 按绿框裁剪原图（未标绿框时按全部红框包围盒外扩裁剪）；
4. 在裁剪图内每个红框区域中，用固定阈值分割深色胶条，
   再用逐列垂直梯度精修上下边界，并保留具有横向几何特征的连通域作为胶条主体；
5. 输出裁剪图、分割可视化图与 JSON 报告。
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

LABEL_DIR_NAME = "标注出GAD胶条区域的图片"

# 红框/绿框提取与胶条几何约束参数
MIN_BOX_AREA = 2_000
MIN_COMPONENT_AREA = 300
MIN_COMPONENT_ASPECT_RATIO = 1.5
CLOSE_KERNEL = (5, 25)
OPEN_KERNEL = (3, 3)

# 无绿框标注时，以全部红框包围盒外扩该边距作为裁剪区域
CROP_MARGIN = 60

# 深色胶条灰度阈值，用 05 号标准样本标定：
# 胶条像素 p95=132，非胶条像素 min=134，取 130 作为分界。
DARK_THRESHOLD = 130

# 边界梯度精修参数（用 07 号蓝色线标注验证：偏差中位数 -2.9px）
EDGE_SMOOTH_SIGMA = 7      # 列方向高斯平滑核
EDGE_SEARCH_ABOVE = 50     # 粗边界上方搜索窗口
EDGE_SEARCH_BELOW = 40     # 粗边界下方搜索窗口
EDGE_GRAD_MIN = 10.0       # 最小梯度强度
EDGE_OUTLIER_WIN = 31      # 离群剔除滑动窗口
EDGE_OUTLIER_THR = 8.0     # 离群偏差阈值(px)
EDGE_SMOOTH_WIN = 8.0      # 边界曲线高斯平滑 sigma

# 边界候选的"暗区闸门"：胶条上方背景中的白色密封胶/反光斑块会产生
# 更强的灰阶跃变，误导梯度峰搜索（实测上边界偏高达 28px）。
# 真正的胶条边界必须通向暗区：候选点朝向胶条一侧的相邻像素
# 中位灰度须低于该阈值，否则视为背景纹理梯度予以剔除。
EDGE_DARK_GATE = 130.0     # 暗区灰度闸门（与粗分割阈值一致）
EDGE_DARK_SPAN = 8         # 暗区检查的相邻像素数

# 自适应暗核边界参数：更新照明后胶条上方白色密封胶/反光斑块、胶条
# 边缘 10~20px 的平缓过渡带都会使梯度峰搜索失效（06 号实测上边界
# 双峰偏差 -23/+15px）。改为逐列检测"平坦暗核"：以该列粗分割区域
# 中位灰度 + EDGE_CORE_MARGIN 为阈值，沿列扫描找首个/最后一个持续
# 暗区作为上/下边界（06 号实测偏差中位约 +5/-0.5px 且分布集中）。
EDGE_CORE_MARGIN = 15.0    # 暗核灰度余量
EDGE_CORE_RUN = 5          # 持续暗区检查窗口
EDGE_CORE_MIN_DARK = 4     # 窗口内判定为暗的最小像素数
EDGE_CORE_SCAN = 60        # 从粗边界向内扫描的最大距离

# 外轮廓精修参数（生产算法，2026-09-15 起替代上面的暗核法）。
#
# 人工真值画的是胶条的"外轮廓"——从胶条外侧由外向内第一次进入胶条的那条边。
# 暗核法找的是"平坦暗核的边缘"，会被胶条顶面的镜面高光带挡住：高光带本身
# 暗、且其下沿是真边（06 实测上边界 39.7% 的列偏 >5px）。
# 本方法分三步：
#   1) 逐列候选：横向长核平滑后，从粗边界外向内找首个同时满足
#      "纵向梯度 >= EDGE_OUTER_GRAD" 且"内侧 EDGE_OUTER_DARK_WIN 窗内
#      最小值 <= tau" 的行；tau = 芯色 + EDGE_OUTER_CORE_MARGIN。
#      上边界取最外（最上）的一个，下边界取最外（最下）的一个。
#   2) 越阈兜底：胶条顶边若是缓坡（5-1600/08 整段梯度仅 5~8 灰阶）或
#      ROI 顶边本身就是胶条，"强边候选"会整列找不到而塌到胶条内部的
#      纹理边上。此时以越阈点 L = core + alpha*(bg-core) 兜底，并约束
#      强边候选不得越过越阈点 slack 像素（slack<0 关闭该约束）。
#      芯色与背景色都按列鲁棒估计（见 _column_levels 的说明）。
#   3) 动态规划：L1 数据项（截断 EDGE_OUTER_DEV_CLIP）+ L1 平滑项，
#      把逐列候选连成横向连续的光滑曲线，并让缺失列由邻居自然补全。
#
# 真值验证（05/06 手工绿线，|误差|>5px 的列占比）：
#   上边界 23.8%~39.7% -> 2.9%~9.2%；下边界 16.1%~27.1% -> 4.6%~8.5%。
EDGE_OUTER_WIN_X = 21          # 横向长核平滑宽度（列）
EDGE_OUTER_GRAD = 8.0          # 强边纵向梯度阈值（灰阶）
EDGE_OUTER_GRAD_WEAK = 5.0     # 强边缺失时的弱边兜底阈值
EDGE_OUTER_DARK_WIN = 5        # 暗核闸门窗（列向最小值）
EDGE_OUTER_CORE_PCT = 30.0     # 胶条芯色百分位
EDGE_OUTER_CORE_WIN = 61       # 芯色的列向滚动中位数窗
# 30 → 35。2026-09-20 用 8X 下 49 条人工真值重新标定（旧值 30 从未标定过）。
#
# 闸门有两个互相拉扯的职责：**漏检**（真值那条边过不了闸，候选只能往内滑）与
# **误抢**（真值上方的背景边过闸，而且它是"最外侧合格行"，直接把候选抢走）。
# 逐列实测 38941 列（49 条）的 recall（真值行 inner_min ≤ tau 的占比）与
# steal（真值上方 40 行内存在过闸强边的占比）：
#
#   margin   30     35     40     45     50     60
#   recall  76.7%  81.6%  85.9%  88.9%  91.3%  94.4%
#   steal   18.8%  21.6%  24.6%  28.1%  32.9%  49.1%
#   recall-steal 57.9 / 59.9 / 61.2 / 60.8 / 58.4 / 45.3
#
# 曲线在 35~45 是一段平台，40 是数学上的峰值，但端到端回归把它否掉了：
# 40 会把 `0914-05 条2` 上边界从 +0.06 打到 −2.39（>5px 2.6%→19.6%），
# `0914-07 条1` 从 −0.88 打到 −1.67。35 是**不破坏任何一条已记录验收值**
# （偏移全部 ≤1px，实为 ≤0.84px）的最大取值；36 就跳到 1.54px 并破一项。
# 同时它也是旧真值集 mIoU 最高的取值（0.9522→0.9550），新真值集 0.9467→0.9520。
#
# 注意它**不能**改成按对比度缩放的形式（core + k*(bg-core)）：k 从 0.15 扫到 0.50,
# recall-steal 全部低于绝对余量版（最高 56.8 < 57.9），端到端也单调变差。
EDGE_OUTER_CORE_MARGIN = 35.0  # tau = 芯色 + 该余量
EDGE_OUTER_BG_PCT = 90.0       # 外侧参照色百分位
EDGE_OUTER_BG_OFF = 10         # 参照取样起点距粗边界的距离
EDGE_OUTER_BG_SPAN = 25        # 参照取样行数
EDGE_OUTER_BG_MIN = 1          # 外侧至少要有这么多行可作参照
EDGE_OUTER_BG_CONTRAST = 20.0  # 参照色至少高出芯色这么多才采信
# 越阈水平 L = core + alpha*(bg-core)。α 越小，L 越贴近芯色，越阈点外移得越远，
# 这是不动暗核闸门的前提下唯一能改善边界的旋钮。
# 0.20 → 0.35 是为修 0914-05 条1 等两条（§3.3.1，代价是 0914-05 条2 / 0914-07 条1 被改坏）。
# 2026-09-20 改为 **0.25**：依据是 49 条真值上实测 β 中位恰好 0.25（§3.3.4 末），
# 且用户看过 13 个车身文件夹的实际出图后确认采用。
# **已接受的代价**：3888 那 10 条旧真值 mIoU 0.9551→0.9483、上边界 |均| 1.99→2.47、
# >5px 6.3%→8.5%。**收益**：新 39 条上边界 |均| 1.93→0.98、mIoU 0.9467→0.9643。
# 出图批次见 §5「_8X_20260920_1801」（该批即 α=0.25）。
# ⚠️ 0.45 就是悬崖：同一批数据 0914 上边界均值从 +0.33 跳到 -4.24、>5px 9.6%→20.8%
# （越阈点冲出胶条、被上方背景勾走）。0.25 离悬崖更远，但往上调之前仍要先重跑回归。
EDGE_OUTER_LEVEL_ALPHA = 0.25  # 越阈水平 L = core + alpha*(bg-core)
# 3 → 1。连续 3 行的判据会让穿越点系统性再内退 1~1.5px，而这 1~1.5px 正是上边界
# 的常驻偏内量。端头方向一直是 1，现在两边一致。下边界的 level2 共用本参数。
EDGE_OUTER_LEVEL_RUN = 1       # 越阈判据要求连续这么多行低于 L
EDGE_OUTER_LEVEL_SLACK = 12.0  # 强边候选允许越过越阈点的松弛量(px)，<0 关闭
# 越阈点离强边候选多远就不再采信（px，<0 关闭）。越阈点的职责是修正"候选落
# 在高光带内沿"这一种错误，那种偏差只有几到二十几像素；它一旦离候选几十像素，
# 说明越阈点自己跑偏了——因为 bg 取自"粗边界外侧 10~35px 的 90 百分位"，粗边界
# 偏外时会采到远处车身的强折光线，把 L 抬得比车身还高，于是胶条**下方**的车身
# 也低于 L，越阈点一路滑到搜索带外沿。0914-08 右下端就是这样：强边候选在真值
# +2px（正确），越阈点却在 +44px，把正确的候选换掉了，多标 44px。
# 单看越阈点自己分辨不出这两种情况（"往内几行见没见暗核"对 05 条1 上边界这类
# 正确越阈点同样不成立，实测会把上边界均值从 +3.4 打到 +8.8），距离才是判据。
EDGE_OUTER_LEVEL_MAX_PULL = 25.0  # 越阈点最多把候选往外拉这么多 px，<0 关闭
# ---- 高光兜底：外侧上限夹逼（2026-09-20 新增，**默认关闭**；见文档 §3.5.4）----
# 用户反馈："剩下一些胶条上有比较强的反光区域的位置没有包含在标注区域内"。
# 成因是暗核闸门要求"紧贴候选的内芯必须暗（<= tau）"：胶条顶面被强反光洗白的
# 那些列过不了闸，候选被顶到亮区**下面**的内部纹理边上，边界整体偏内
# （0914-3888-05-01 上边界 |均| 7.7px、40% 的列偏内 >5px；最重的三条是
# 0914-3888-05-01、0916-7060-09-01、0916-8203-08-01）。
#
# 本段实现"离开背景"上限夹逼：逐列取最外侧的、外侧那半窗仍在背景水平
# （>= bg-glare_drop）而内侧这半窗已低于它的行 d；d 比现有候选更靠外、且差距在
# [glare_min_gap, glare_pull] 内时把候选拉过去。
#
# **四种形态全部实测回退，因此整段默认关闭（FALLBACK = 0）。49 条真值回归：**
#   形态                              上边界 |均| / >5px    四侧 |均| / mIoU
#   关闭（现行，α=0.25 基线）            1.32 /  8.3%        1.35 / 0.9590
#   放宽闸门并列竞争                     8.20 / 44.4%        3.40 / 0.8684
#   "薄壳盖暗核"并列（平台+壳厚+见暗核）  1.32 /  8.3%        1.35 / 0.9590  ← 一次都没触发
#   夹逼 pull=30                        5.09 / 34.5%        2.29 / 0.9046
#   夹逼 min_gap=15                     4.15 / 23.4%        2.18 / 0.9181
#   夹逼 min_gap=25（最严）              2.38 / 12.9%        1.71 / 0.9438
#
# 两条结论，别再重走：
#   * "薄壳"模型不成立。真实坏列（0914-3888-05-01 列 493~497）的顶面是**从背景
#     连续下降 20~30 行**的斜坡，既没有平顶，很多列整列都到不了 tau（横向最小值
#     全程 > 150）。所以"平台/壳厚/必须见暗核"这类判据在 93.5 万条带内强边行上
#     的交集只剩 166 行。
#   * "离开背景"那一行本身是个**偏外 17.7px** 的坏估计（38967 列实测 |均| 17.7px、
#     80% 超 5px），比它要修的偏差（最坏列 33px、正常列 ~1px）没有余量。当上限用
#     时它只在候选偏内的列生效，所以整体是净改善（离线逐列 1.54->1.48px）；但一进
#     真实流水线，拉外候选会顺带让"越阈点"因 |y_lvl - cand| > MAX_PULL 而失效，
#     失掉那一道修正，回归因此变差（5.09）。min_gap 只挡掉小拉，挡不住这条。
#   它也不等于当年被否掉的"无差别放宽 EDGE_OUTER_DARK_WIN"（那条路让背景边也
#   过闸，steal）：这里要求外侧仍在背景水平，背景内部的行两头都过不了 bg-drop。
#
# 保留代码是因为扫描工具一条命令就能重跑（见 §5 那份 Temp/gadbase/sweep.py）：
# 哪天要重开这条路，必须同时处理"越阈修正被顶掉"这一条，否则仍会回退。
EDGE_OUTER_GLARE_LO = 150.0      # 内侧亮于此才算亮区（更暗的列本来就有候选）
EDGE_OUTER_GLARE_DROP = 30.0     # 比该列背景低这么多就算"离开背景"
EDGE_OUTER_GLARE_FALLBACK = 0    # 1 = 启用；0 = 关闭（默认，退回 α=0.25 基线）
EDGE_OUTER_GLARE_PULL = 30.0     # 最多把候选往外拉这么多 px
EDGE_OUTER_GLARE_MIN_GAP = 0.0   # 候选至少比它深这么多才拉
EDGE_OUTER_DEV_W = 1.0         # DP 数据项权重
EDGE_OUTER_DEV_CLIP = 15.0     # DP 数据项截断(px)
# DP 平滑代价（每 px 跳变）。**12 -> 24，2026-09-29 实测后落地。**
#
# 机理：数据项被 `EDGE_OUTER_DEV_CLIP` 封顶、平滑项按每像素收费，于是
# **约 `dev_clip / pen` 列"候选在别处"的证据就能压过挪动 1px**（12/15 时是 1.25 列）。
# 胶条结束后 ROI 里还剩几十列（pad_x 外扩 + 预测框偏宽），那里的软暗带在 ROI 底部
# 给出候选，DP 被拖下去；`jump` 与平滑项又把这次下沉**反传**到胶条真正的最后几列
# —— 这就是图上那根端头方块。pen 把"挪 1px 的票价"抬上去，下沉就付不起了。
#
# 证据（`b2_knob_test.py`，68 条 ROI / 100 个端头，进程内 monkeypatch）：
#     pen=12 dev=15(旧)   |超|中位 3.0   |超|均值 8.7   >20px 13   >40px 3
#     pen=24 dev=15(新)   |超|中位 3.0   |超|均值 5.2   >20px  2   >40px 0
# **全体 |超真值| 中位稳在 3.0px 不动**（没伤到已收敛的那 87 个），只削右尾。
# `pen=40 dev=2` 已证明会过头（中位被带坏到 5.8），所以只动这一个旋钮。
#
# 等价备选是 `EDGE_OUTER_DEV_CLIP` 15 -> 8（两者是同一杠杆的两端，实测几乎一样），
# 但 dev_clip 那一侧在 dev<8 时因 `min(dev, clip)` 对近处正常列零影响、更有针对性——
# 2026-09-29 选了 pen，理由是 pen 对着"跳变"本身，语义更直接。**改任何一个都别两个一起改。**
#
# ⚠️ 这两个常量**只被 `_long_profile()` 用**（长边方向，逐列上下边界）。
# 端头轴走 `_cap_profile()`，吃的是 `EDGE_CAP_PEN`，**不受影响**。
# ⚠️ 改它会让 A 组与已存的 `真值比对/真值比对报告.json` 不再逐位一致，
# `eval_detector_end2end.py` 的 `check_group_a` 会 `sys.exit(1)`——那不是回归失败，
# 是基线过期。做法是 `--no-check-a` 停用自检，改为**直接比 A 组四侧的老/新值**。
EDGE_OUTER_PEN = 24.0
EDGE_OUTER_JUMP = 3            # DP 相邻列最大跳变(px)
EDGE_OUTER_UP = 40             # 候选搜索带：粗边界上方
EDGE_OUTER_DOWN = 70           # 候选搜索带：粗边界下方

# 2026-09-21 新增。**粗掩膜在长边方向也可能是"整片"的**，不只是端头方向的问题。
#
# 红框框住的那片区域如果本身比 DARK_THRESHOLD 还暗（车身深色漆面、大片阴影），
# segment_dark_strip 的 `coverage >= 0.50` 整体接受分支会直接返回"所有低于阈值的
# 像素"，粗掩膜于是变成一整片顶到 ROI 边的东西——实测这是**常态**，不是例外：
# 四个批次 273 张图里 coverage 中位 0.81，49/50、76/77、61/61、87/87 全部走这一支。
#
# 后果在 `_search_bands`：粗上边界塌到 0（或粗下边界顶到 height-1）的那些列，
# 搜索带内侧只剩 `rough + EDGE_OUTER_DOWN`，本来该在那之后才出现的真边直接被带子
# 切掉，`_outer_candidates` 找不到合格梯度，DP 于是把路径**钉在带边上**。
# 实测 11507-06：列 0..425 的上边界全部正好落在 `rough + 70`（426 列同一个值），
# 而真值在再往下 13px 处；列 ≥430 粗上边界跳到 67，路径随之跳到 78，接缝处
# 出现一道台阶。这条把"退化的粗边界"当成"没有粗支持"处理——跟端头方向对
# `anchor_pct is not None` 的 `open_ended` 是同一件事：
#   * side='lo'（上边界）：rough <= tol 视为无支撑 -> hi 放开到 height-1
#   * side='hi'（下边界）：rough >= height-1-tol 同理 -> lo 放开到 0
# lo 侧仍由 `outer - EDGE_OUTER_UP` 兜着，所以不是无界搜索。
#
# 扫 tol ∈ {0,2,5,8}（* 见说明文档 §5 的 Temp/gadbase/sweepband.py）：
#   tol   上 |均| / >5px     左 |均| / >5px    右 |均| / >5px    四侧 |均| / >5px
#   关    1.84 /  8.31%      1.70 /  4.85%     1.95 /  7.72%    1.63 /  5.44%
#   5     1.77 /  7.46%      1.72 /  4.96%     1.95 /  7.69%    1.62 /  5.22%
# tol=0 与 5 只差个别列（0 会误伤粗边界本来就贴边的正常条），取 5。
EDGE_OUTER_OPEN_TOL = 5

# 端头（左右边界）精修参数。2026-09-18 起 refine_strip_mask 在长边之外再跑一遍。
#
# 做法是把 ROI 转置后**复用长边那套机制**：转置图里"逐列上下边界"就是原图里
# "逐行左右边界"，候选/越阈/DP 全是对称的纯数组代码。端头的斜率也说明可以复用——
# 胶条倾角约 1:13，端头在转置帧里的 |dx/dy| 与长边在原帧里的斜率同量级，
# 所以 EDGE_OUTER_WIN_X / DARK_WIN / LEVEL_RUN 保留物理含义。
#
# 但**搜索带必须换一种开法**，这是端头方向唯一的结构性差异：
#   长边是围绕逐列粗边界开带（粗边界本身可靠）；端头方向若照搬，会让整段端头
#   落在带外。原因是某行的最外侧粗像素常常落在**斜长边**上，此时该行的粗左边界
#   已经深入胶条几十到上百像素，用它当外锚，端头就被排除在搜索带外，DP 只能
#   贴着斜长边跑——实测过切 +36px（左）/ +14px（右），个别条 +60~+78px。
#   改成"外侧锚定到粗边界的低百分位（全行共用一个值）+ 内侧逐行收紧"即可。
#
# 选行规则同样不能用"粗极值附近的行"：粗 mask 被红框裁过，极值常出现在斜长边
# 行上，实测与真值端头带只对上 39/60（左）、9/90（右），5 条完全没有交集。
EDGE_CAP_WIN_X = 21            # 转置后的横向平滑窗 = 沿端头切线的方向
EDGE_CAP_GRAD = 8.0            # 强边梯度阈值，先与长边相同，实测再调
EDGE_CAP_GRAD_WEAK = 5.0
EDGE_CAP_DARK_WIN = 5
EDGE_CAP_CORE_PCT = 30.0
EDGE_CAP_CORE_WIN = 31         # 不是长边的 61：滚动中位数窗按量程定，端头量程只有胶条厚度
EDGE_CAP_CORE_MARGIN = 30.0    # tau = 芯色 + 该余量。橡胶切面若比它亮，端头会欠切，见说明文档
EDGE_CAP_BG_PCT = 90.0
EDGE_CAP_BG_OFF = 10
EDGE_CAP_BG_SPAN = 25
EDGE_CAP_BG_MIN = 1
EDGE_CAP_BG_CONTRAST = 20.0
# 端头的 level_alpha 远高于长边的 0.35，这是端头方向与长边方向最重要的差别。
# 端头的灰度过渡是**斜坡**（橡胶端面倒圆 + 镜头模糊，约 8px 宽），而"最外侧梯度
# 候选"只保证落在坡内某处，具体在哪一侧由坡的形状决定，端头因此有 ±数像素的
# 系统性偏差。越阈水平 L = core + alpha*(bg-core) 把边界钉在斜坡的固定比例位置上，
# 这才是端头方向的定标手段 —— α 就在 0.50~0.80 之间扫，各端头均值摆动 ±3px 量级，
# 网格合计从 113.9 涨到 227.2（见文档 §3.3.3）。
#
# alpha 取多少，直接量：对端头带内每一行的真值边界处取样，算
# beta = (真值处灰度 - core) / (bg - core)，即"人工真值落在斜坡的哪个比例上"。
# 24 个端头、1739 行实测：beta 中位 +0.57、均 +0.57、10% +0.29、90% +0.88
# （真值处灰度中位 138~220，core 30~67，bg 235~255）。所以 alpha 应当在 0.57 附近。
# |beta - alpha| 中位：alpha=0.55/0.60 → 0.151；alpha=0.80 → 0.230。
#
# 端到端网格（8 个汇总 + 4 个点名端头，Σ>5px 与 Σ|均| 两个口径一起看，单看一个必然
# 过拟合）在 alpha=0.80 处是最差点：0.50 → 104.6/17.66，0.55 → 108.1/16.79，
# 0.60 → 113.9/15.89，0.80 → 227.2/21.45。此前取 0.80 的理由是"人工真值画在斜坡
# 起跳点、约灰度 200"——这条前提被上述实测推翻（真值处灰度中位远低于 200），
# 故改取 0.60。0.50/0.55 的 Σ>5px 略优但 Σ|均| 更差，0.60 是两个口径的折中，
# 也最贴近实测的 beta 中位。
#
# 下限由 0.35（长边值）到 0.50 之间单调改善，上限 0.95/1.00 时水平贴在 bg 上、
# 背景噪点会提前触发，均值翻负且 >5px 变大。
EDGE_CAP_LEVEL_ALPHA = 0.60
# 长边是 3。端头是 1：连续 3 像素的判据会让穿越点系统性再往里退 1~1.5px，而端头
# 本来就在补一个偏内偏差。单像素触发看着更脆，但实测把 level_slack 收到 4px 仍然
# 逐位不变 —— 即穿越点从来不会离梯度候选超过 4px，噪声甩不出去。
EDGE_CAP_LEVEL_RUN = 1
EDGE_CAP_LEVEL_SLACK = 8.0     # 长边是 12。端头方向它有两个作用：强边候选可以越过
                               # 越阈点的松弛量，以及越阈点允许把候选往**外**拉的上限
                               # （见 _refine_axis 的 prefer_outer_level）。
EDGE_CAP_DEV_W = 1.0
EDGE_CAP_DEV_CLIP = 15.0
EDGE_CAP_PEN = 6.0      # 长边是 12。端头的弧长只有一个坡宽，12 会把圆端头压平
                        # （实测端头真值是 945→935→941 的圆弧，压平后差 6px）
EDGE_CAP_JUMP = 5              # 长边是 3：圆端头在靠近两端处比长边陡
EDGE_CAP_ANCHOR_PCT = 10.0     # 外侧锚点取粗边界的低百分位（左端取 10，右端取 90）
EDGE_CAP_ANCHOR_SLACK = 0      # 锚点再往外放这么多像素
EDGE_CAP_INNER = 40            # 内限相对逐行粗边界内缩的上限，兜住 runaway
EDGE_CAP_MIN_CAND = 0.25       # 候选覆盖率低于该比例就放弃该侧收缩（保持宽松）
EDGE_CAP_BORDER_TOL = 3        # ROI 边距画面边不超过这么多像素就算"画面边界"
EDGE_CAP_AREA_FUSE = 0.5       # 收缩后面积掉到逐列结果的这个比例以下就整体放弃
EDGE_CAP_EDGE_SKIP = 2         # 搜索带从量程两端各收掉这么多列。粗掩膜溢到 ROI 边时，
                               # 最外那两列会造出一条假强梯度，把端头往里带偏
EDGE_CAP_XBAND = 40            # 每条胶条最外这么多列做端头回退（见 _restore_cap_columns）
EDGE_CAP_XTHIN = 0.85          # 只在纵轴这一列的高度塌到中位高的这个比例以下时才回退
EDGE_CAP_XCLIP = 0             # 回退时最多比纵轴边界外扩这么多像素；0 = 不限（整列改用粗掩膜）
EDGE_CAP_XFRAME = 1            # 1 = 画面切断的那一侧不做端头回退（那里没有端头转角，
                               # 粗掩膜又正好是 ROI 边的闭运算伪影，回退会焊出竖条）
EDGE_CAP_XFIT = 0              # 回退时粗掩膜那一段的高度不得超过中位列高的这么多倍。
                               # 0 = 不设这道闸门。**实测是个坏闸门**：1.25 让右边界
                               # |均| 1.95->2.18、>5px 7.72%->10.18%，1.5 也还差
                               # （2.02/8.60%）—— 说明真值上"粗行程比中位列高还高"的
                               # 回退大量是合理的，撑满 ROI 高那种极端情况交给 XFULL
EDGE_CAP_XNEAR = 0             # 1 = 粗掩膜在这一列没有与纵轴区间重叠的行程时，退回
                               # 最近的一段。会产生与纵轴完全不相交的孤块，默认关
EDGE_CAP_XFULL = 1             # 1 = 粗掩膜那一段上下都顶到 ROI 边时跳过这一列。粗掩膜
                               # 把胶条和周围暗区连成一整片、撑满 ROI 高时就是这个样子，
                               # 回退会在端头竖起一根贯穿画面的竖线

# 2026-09-21 新增：把端头回退从"整列高矮"推广到"逐侧"。上面那道 XTHIN 是个**整列**
# 判据（纵轴这一列的高度塌了才回退），只有纵轴在两个方向同时塌掉的列才会被救到。
# 实测 3911408-09 右端：纵轴塌掉的是**上边界**（比真值深 ~20px），但整列高度没塌到
# 0.85×中位，于是 283..293 这些列一次都不回退，端头右边整段上边界偏深；294..303
# 塌够了才回退，一进一出，边界呈锯齿。
#
# 逐侧判据：粗行程的**上界**不是 ROI 边的伪影（离边超过 BORDER_TOL），且比纵轴上边界
# 更外超过 EDGE_CAP_XGAP 像素，就把那一侧补到粗行程上。XTHIN 那支保留原样（整列改用粗）。
#   * **只做 lo 侧（上边界）。** 对称地把 hi 侧（下边界）也放开，实测 上/右 的收益
#     一分不多，而下边界 |均| 从 1.14 涨到 1.20、>5px 1.47%->1.76% —— 胶条下方那片
#     暗区（阴影/保险杠）跟胶条连成一段，粗下界本来就偏深，纵轴在那儿是对的。
#   * 有 XGAP 闸门，才能跟"粗边界本来就比纵轴外一点"的正常列区分开——实测 gap=0
#     会让端头回退在所有列上都生效，右边界反而变差。
#   * XGAP_MAX 是单侧外扩上限，沿用 XCLIP 的思路。实测 gap=10 时，外扩落在
#     20~24px 区间的那几列会把左边界的 |均| 从 1.70 顶到 2.34；封到 20 就能拿到
#     全部收益而不付这份代价。
# 扫过的组合见说明文档 §5 的 Temp/gadbase/sweepgap.py：
#   gap/clip       上 |均| / >5px    左 |均| / >5px    右 |均| / >5px    四侧 |均| / >5px
#   关（只开 OPEN_TOL） 1.77 / 7.46%   1.72 / 4.96%   1.95 / 7.69%     1.62 / 5.22%
#   10 / 无上限      1.65 / 6.90%   2.34 / 4.80%   1.65 / 2.93%     1.70 / 4.16%
#   10 / 20          1.66 / 6.90%   1.69 / 4.74%   1.66 / 3.01%     1.54 / 4.17%
#   10 / 20 仅 lo 侧 1.66 / 6.90%   1.70 / 4.73%   1.66 / 3.01%     1.52 / 4.08%
EDGE_CAP_XGAP = 10.0           # 逐侧回退：粗上界比纵轴上边界更外超过这么多像素才补；0 = 关闭
EDGE_CAP_XGAP_MAX = 20         # 逐侧回退的单侧外扩上限(px)；0 = 不限


@dataclass
class SegmentRecord:
    file: str
    green_box: tuple[int, int, int, int]  # 原图坐标系 (x, y, w, h)
    red_boxes: list[tuple[int, int, int, int]]
    crop_size: tuple[int, int]
    strips: list[dict] = field(default_factory=list)
    error: str = ""


def label_stem(path: Path) -> str:
    """将标注文件名还原为对应原始 BMP 的文件名前缀。"""
    return (
        path.stem.replace("_已标注", "")
        .replace("_自动标注", "")
        .replace("_图片已标注", "")
        .replace("_区域已标出", "")
    )


def extract_boxes(label_path: Path) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int, int]]]:
    """提取标注图中的绿框（裁剪区域）与全部红框（GAD 区域）。"""
    annotated = cv2.imdecode(np.fromfile(label_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if annotated is None:
        raise ValueError(f"无法读取标注图: {label_path}")
    hsv = cv2.cvtColor(annotated, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255)) | cv2.inRange(hsv, (170, 100, 100), (180, 255, 255))
    green_mask = cv2.inRange(hsv, (35, 80, 80), (85, 255, 255))

    def boxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        found = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w * h >= MIN_BOX_AREA:
                found.append((int(x), int(y), int(w), int(h)))
        return sorted(found, key=lambda b: (b[1], b[0]))

    return boxes(green_mask), boxes(red_mask)


def map_boxes(
    boxes: list[tuple[int, int, int, int]],
    label_shape: tuple[int, int],
    image_shape: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    """按标注图与原图尺寸比例，将框坐标映射到原图坐标系。"""
    raw_h, raw_w = image_shape
    scale_x = raw_w / label_shape[1]
    scale_y = raw_h / label_shape[0]
    mapped = []
    for x, y, w, h in boxes:
        left = max(0, round(x * scale_x))
        top = max(0, round(y * scale_y))
        right = min(raw_w, round((x + w) * scale_x))
        bottom = min(raw_h, round((y + h) * scale_y))
        mapped.append((left, top, right - left, bottom - top))
    return mapped


def segment_dark_strip(roi: np.ndarray) -> np.ndarray | None:
    """在红框 ROI 内分割深色横向胶条，返回二值 mask，失败返回 None。"""
    blurred = cv2.medianBlur(roi, 5)
    if blurred.std() < 1.0:
        return None
    # 固定阈值反向二值化：灰度低于 DARK_THRESHOLD 的深色胶条为 255。
    # 阈值由 05 号标准样本标定，避免 Otsu 在周围颜色较深时阈值过低。
    _, candidate = cv2.threshold(blurred, DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones(OPEN_KERNEL, np.uint8))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones(CLOSE_KERNEL, np.uint8))
    coverage = float(candidate.mean()) / 255.0
    # 深色区域几乎占满整个红框时，红框整体即胶条（如胶条端部截面），
    # 直接整体接受，不再做几何约束过滤。归一化为 0/1 二值。
    if coverage >= 0.50:
        return (candidate > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    selected = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < MIN_COMPONENT_AREA or h == 0:
            continue
        if w / h < MIN_COMPONENT_ASPECT_RATIO:
            continue
        selected.append(label)
    if not selected:
        return None
    return np.isin(labels, selected).astype(np.uint8)


def strip_geometry(mask: np.ndarray) -> dict:
    """汇总胶条 mask 的几何统计信息。"""
    cols = mask.sum(axis=0)
    rows = mask.sum(axis=1)
    present_cols = cols > 0
    core = present_cols.copy()
    core[:5] = False
    core[-5:] = False
    median_width = float(np.median(cols[core])) if core.any() else 0.0
    ys, xs = np.nonzero(mask)
    return {
        "area": int(np.count_nonzero(mask)),
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
        "median_thickness_px": round(median_width, 2),
        "thickness_std_px": round(float(cols[core].std()), 2) if core.any() else 0.0,
    }


def _edge_scan(roi: np.ndarray, grad: np.ndarray, rough: np.ndarray, side: str) -> np.ndarray:
    """逐列在粗边界附近搜索灰度跃升点，返回每列精修边界 y 值（NaN 表示未找到）。

    side='top' 时找暗→亮跃升上方界（负梯度峰值），side='bottom' 时找下方界（正梯度峰值）。
    候选峰必须满足"暗区闸门"：朝向胶条一侧的相邻像素中位灰度低于 EDGE_DARK_GATE，
    以排除胶条上方白色密封胶/反光斑块等背景强纹理；在有效候选中取离粗边界最近者。
    """
    width = roi.shape[1]
    height = roi.shape[0]
    refined = np.full(width, np.nan)
    for x in range(width):
        if np.isnan(rough[x]):
            continue
        rough_y = int(rough[x])
        y0 = max(2, rough_y - EDGE_SEARCH_ABOVE)
        y1 = min(height - 2, rough_y + EDGE_SEARCH_BELOW)
        if y1 <= y0:
            continue
        best_y = None
        best_distance = float("inf")
        for y in range(y0, y1):
            value = grad[y, x]
            strength = value if side == "bottom" else -value
            if strength < EDGE_GRAD_MIN:
                continue
            if side == "top":
                dark_side = roi[y + 1 : min(y + 1 + EDGE_DARK_SPAN, height), x]
            else:
                dark_side = roi[max(0, y - EDGE_DARK_SPAN) : y, x]
            if len(dark_side) and float(np.median(dark_side)) >= EDGE_DARK_GATE:
                continue
            distance = abs(y - rough_y)
            if distance < best_distance:
                best_distance = distance
                best_y = y
        if best_y is not None:
            refined[x] = best_y
    return refined


def _smooth_boundary(values: np.ndarray) -> np.ndarray:
    """离群剔除、线性插值填补与高斯平滑，返回逐列连续边界。"""
    width = len(values)
    cleaned = values.copy()
    med = np.full(width, np.nan)
    half = EDGE_OUTLIER_WIN // 2
    for i in range(width):
        window = values[max(0, i - half) : min(width, i + half + 1)]
        window = window[~np.isnan(window)]
        if len(window):
            med[i] = np.median(window)
    bad = np.abs(values - med) > EDGE_OUTLIER_THR
    cleaned[bad] = np.nan
    ok = ~np.isnan(cleaned)
    if np.count_nonzero(ok) < 3:
        return values
    idx = np.arange(width)
    filled = np.interp(idx, idx[ok], cleaned[ok])
    k = np.exp(-np.arange(-15, 16) ** 2 / (2 * EDGE_SMOOTH_WIN ** 2))
    k /= k.sum()
    return np.convolve(filled, k, mode="same")


def refine_strip_edges(roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """用逐列垂直梯度精修胶条 mask 的上下边界，返回精修后的二值 mask。

    粗阈值分割的下边界常受胶条下方深色阴影影响而偏下（07 实测偏 32px），
    灰度跃升点（最大正梯度）才是真实边界（07 实测偏差约 -3px）。
    边界候选须通过"暗区闸门"（通向胶条暗区），避免被胶条上方的白色
    密封胶/反光斑块强梯度误导（改进后 06 号图上边界偏差从 -28px 降至约 ±2px）。
    """
    mask = (mask > 0).astype(np.uint8)  # 归一化为 0/1
    m = mask.astype(bool)
    width = roi.shape[1]
    rough_top = np.full(width, np.nan)
    rough_bottom = np.full(width, np.nan)
    for x in range(width):
        col = np.nonzero(m[:, x])[0]
        if len(col):
            rough_top[x] = col.min()
            rough_bottom[x] = col.max()
    smooth = cv2.GaussianBlur(roi, (1, EDGE_SMOOTH_SIGMA), 0)
    grad = np.diff(smooth.astype(np.float32), axis=0)
    top = _smooth_boundary(_edge_scan(roi, grad, rough_top, "top"))
    bottom = _smooth_boundary(_edge_scan(roi, grad, rough_bottom, "bottom"))
    refined = np.zeros_like(mask)
    for x in range(width):
        if np.isnan(top[x]) or np.isnan(bottom[x]):
            refined[:, x] = mask[:, x]
            continue
        y0, y1 = int(round(top[x])), int(round(bottom[x]))
        y0 = max(0, y0)
        y1 = min(roi.shape[0], y1 + 1)
        if y1 > y0:
            refined[y0:y1, x] = 1
    return refined


def refine_strip_edges_core(roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """用逐列自适应暗核检测精修胶条 mask 的上下边界，返回精修后的二值 mask。

    更新照明后胶条上方的白色密封胶/反光斑块会产生比胶条边界更强的灰阶
    跃变，且胶条边缘常为 10~20px 的平缓过渡带，梯度峰搜索会失效
    （06 号实测上边界偏差中位 -28px、双峰分布 -23/+15px）。本方法把
    人工真值（绿框）对应到"平坦暗核"的边缘：以该列粗分割区域中位
    灰度 + EDGE_CORE_MARGIN 为阈值，沿列扫描找首个/最后一个持续暗区
    （EDGE_CORE_RUN 窗内至少 EDGE_CORE_MIN_DARK 个像素低于阈值）作为
    上/下边界，再经离群剔除与平滑成连续曲线。
    """
    mask = (mask > 0).astype(np.uint8)  # 归一化为 0/1
    m = mask.astype(bool)
    width = roi.shape[1]
    height = roi.shape[0]
    rough_top = np.full(width, np.nan)
    rough_bottom = np.full(width, np.nan)
    for x in range(width):
        col = np.nonzero(m[:, x])[0]
        if len(col):
            rough_top[x] = col.min()
            rough_bottom[x] = col.max()
    top = np.full(width, np.nan)
    bottom = np.full(width, np.nan)
    for x in range(width):
        if np.isnan(rough_top[x]):
            continue
        core = float(np.median(roi[m[:, x], x]))
        threshold = core + EDGE_CORE_MARGIN
        # 上边界：从粗上边界上方 5px 向下，找首个持续暗区的起始行
        scan_start = max(0, int(rough_top[x]) - 5)
        scan_end = min(height - EDGE_CORE_RUN, int(rough_top[x]) + EDGE_CORE_SCAN)
        for y in range(scan_start, scan_end):
            window = roi[y : y + EDGE_CORE_RUN, x]
            if np.count_nonzero(window <= threshold) >= EDGE_CORE_MIN_DARK:
                top[x] = y
                break
        # 下边界：从粗下边界下方 5px 向上，找最后一个持续暗区的结束行
        scan_start = min(height - 1, int(rough_bottom[x]) + 5)
        scan_end = max(EDGE_CORE_RUN - 1, int(rough_top[x]) - 1)
        for y in range(scan_start, scan_end, -1):
            window = roi[y - EDGE_CORE_RUN + 1 : y + 1, x]
            if np.count_nonzero(window <= threshold) >= EDGE_CORE_MIN_DARK:
                bottom[x] = y
                break
    top = _smooth_boundary(top)
    bottom = _smooth_boundary(bottom)
    refined = np.zeros_like(mask)
    for x in range(width):
        if np.isnan(top[x]) or np.isnan(bottom[x]):
            refined[:, x] = mask[:, x]
            continue
        y0, y1 = int(round(top[x])), int(round(bottom[x]))
        y0 = max(0, y0)
        y1 = min(height, y1 + 1)
        if y1 > y0:
            refined[y0:y1, x] = 1
    return refined


def _fill_nan(values: np.ndarray) -> np.ndarray:
    """线性插值填补 NaN；全为 NaN 时返回原值。"""
    ok = ~np.isnan(values)
    if ok.all() or not ok.any():
        return values.copy()
    idx = np.arange(len(values))
    return np.interp(idx, idx[ok], values[ok])


def _rolling_median(values: np.ndarray, win: int) -> np.ndarray:
    """滑窗中位数，用于对逐列估计做局部共识平滑。"""
    n = len(values)
    half = win // 2
    out = np.empty(n)
    for i in range(n):
        out[i] = np.median(values[max(0, i - half) : min(n, i + half + 1)])
    return out


@dataclass(frozen=True)
class EdgeProfile:
    """一个方向上的边界精修调参。

    同一套机制要跑两个方向：长边（逐列上下边界）和端头（逐行左右边界）。
    两者的物理尺度差得远——长边一两千像素，端头只有一个胶条厚度（约 100
    像素）——所以按量程定的窗口常量必须分开，端头方向的取值见 EDGE_CAP_*。
    """

    win_x: int                  # 沿边界切线的平滑窗宽
    grad: float                 # 强边梯度阈值
    grad_weak: float            # 强边缺失时的弱边兜底阈值
    dark_win: int               # 暗核闸门窗（沿切线的最小值）
    core_pct: float             # 芯色百分位
    core_win: int               # 芯色的切线向滚动中位数窗
    core_margin: float          # tau = 芯色 + 该余量
    bg_pct: float               # 外侧参照色百分位
    bg_off: int                 # 参照取样起点距粗边界的距离
    bg_span: int                # 参照取样宽度
    bg_min: int                 # 外侧至少要有这么多像素可作参照
    bg_contrast: float          # 参照色至少高出芯色这么多才采信
    level_alpha: float          # 越阈水平 L = core + alpha*(bg-core)
    level_run: int              # 越阈判据要求连续这么多像素低于 L
    level_slack: float          # 强边候选允许越过越阈点的松弛量(px)，<0 关闭
    dev_w: float                # DP 数据项权重
    dev_clip: float             # DP 数据项截断(px)
    pen: float                  # DP 平滑代价（每 px 跳变）
    jump: int                   # DP 相邻列最大跳变(px)
    out_off: int                # 搜索带向**外**侧延伸的像素数
    in_off: int                 # 搜索带向**内**侧延伸的像素数
    anchor_pct: float | None = None   # 非 None 时外侧参考改取粗边界的低百分位（端头方向）
    min_cand: float = 0.0             # 候选覆盖率下限，低于它放弃该侧；0 = 不启用
    edge_skip: int = 0                # 搜索带从量程两端各收掉这么多列；0 = 不启用
    prefer_outer_level: bool = False  # 允许亮度穿越点把候选往**外**侧拉（端头方向）
    max_pull: float = -1.0            # 越阈点最多把候选往外拉多少 px（见 EDGE_OUTER_LEVEL_MAX_PULL）
    glare_lo: float = 0.0             # 高光兜底：内侧亮于此才算亮区
    glare_drop: float = 0.0           # 高光兜底：比该列背景低这么多就算离开背景
    glare_fallback: int = 0           # 高光兜底开关，0 = 关闭（逐位退回旧行为）
    glare_pull: float = 0.0           # 高光兜底最多把候选往外拉这么多 px
    glare_min_gap: float = 0.0        # 高光兜底只在候选比它深这么多时生效


@dataclass
class AxisEdges:
    """一条轴向上的精修结果。``lo`` = 坐标小的一侧（上边界 / 左端头）。"""

    lo_edge: np.ndarray         # 精修后的边界曲线，NaN = 该列(行)无解
    hi_edge: np.ndarray
    rough_lo: np.ndarray        # 粗 mask 的对应边界，NaN = 该列(行)无粗支持
    rough_hi: np.ndarray
    cand_lo: np.ndarray         # DP 之前的逐列(行)原始候选，用来算覆盖率
    cand_hi: np.ndarray


def _long_profile() -> EdgeProfile:
    """长边方向（逐列上下边界）的调参，即历史行为。

    每次调用重新读模块常量，而不是在导入时烘成一个实例：精修常量是要现场调参的，
    烘死的实例会让"临时改一个 EDGE_OUTER_* 再调精修函数做对照"这种用法静默失效。
    """
    return EdgeProfile(
        win_x=EDGE_OUTER_WIN_X,
        grad=EDGE_OUTER_GRAD,
        grad_weak=EDGE_OUTER_GRAD_WEAK,
        dark_win=EDGE_OUTER_DARK_WIN,
        core_pct=EDGE_OUTER_CORE_PCT,
        core_win=EDGE_OUTER_CORE_WIN,
        core_margin=EDGE_OUTER_CORE_MARGIN,
        bg_pct=EDGE_OUTER_BG_PCT,
        bg_off=EDGE_OUTER_BG_OFF,
        bg_span=EDGE_OUTER_BG_SPAN,
        bg_min=EDGE_OUTER_BG_MIN,
        bg_contrast=EDGE_OUTER_BG_CONTRAST,
        level_alpha=EDGE_OUTER_LEVEL_ALPHA,
        level_run=EDGE_OUTER_LEVEL_RUN,
        level_slack=EDGE_OUTER_LEVEL_SLACK,
        dev_w=EDGE_OUTER_DEV_W,
        dev_clip=EDGE_OUTER_DEV_CLIP,
        pen=EDGE_OUTER_PEN,
        jump=EDGE_OUTER_JUMP,
        out_off=EDGE_OUTER_UP,
        in_off=EDGE_OUTER_DOWN,
        max_pull=EDGE_OUTER_LEVEL_MAX_PULL,
        glare_lo=EDGE_OUTER_GLARE_LO,
        glare_drop=EDGE_OUTER_GLARE_DROP,
        glare_fallback=EDGE_OUTER_GLARE_FALLBACK,
        glare_pull=EDGE_OUTER_GLARE_PULL,
        glare_min_gap=EDGE_OUTER_GLARE_MIN_GAP,
    )


def _cap_profile() -> EdgeProfile:
    """端头方向（逐行左右边界）的调参。理由见 EDGE_CAP_* 块。"""
    return EdgeProfile(
        win_x=EDGE_CAP_WIN_X,
        grad=EDGE_CAP_GRAD,
        grad_weak=EDGE_CAP_GRAD_WEAK,
        dark_win=EDGE_CAP_DARK_WIN,
        core_pct=EDGE_CAP_CORE_PCT,
        core_win=EDGE_CAP_CORE_WIN,
        core_margin=EDGE_CAP_CORE_MARGIN,
        bg_pct=EDGE_CAP_BG_PCT,
        bg_off=EDGE_CAP_BG_OFF,
        bg_span=EDGE_CAP_BG_SPAN,
        bg_min=EDGE_CAP_BG_MIN,
        bg_contrast=EDGE_CAP_BG_CONTRAST,
        level_alpha=EDGE_CAP_LEVEL_ALPHA,
        level_run=EDGE_CAP_LEVEL_RUN,
        level_slack=EDGE_CAP_LEVEL_SLACK,
        dev_w=EDGE_CAP_DEV_W,
        dev_clip=EDGE_CAP_DEV_CLIP,
        pen=EDGE_CAP_PEN,
        jump=EDGE_CAP_JUMP,
        out_off=EDGE_CAP_ANCHOR_SLACK,
        in_off=EDGE_CAP_INNER,
        anchor_pct=EDGE_CAP_ANCHOR_PCT,
        min_cand=EDGE_CAP_MIN_CAND,
        edge_skip=EDGE_CAP_EDGE_SKIP,
        prefer_outer_level=True,
        max_pull=EDGE_OUTER_LEVEL_MAX_PULL,
    )


def _column_levels(
    roi: np.ndarray, rough_top: np.ndarray, rough_bottom: np.ndarray, profile: "EdgeProfile"
):
    """逐列估计胶条芯色 core、外侧参照色 bg/bg2，以及各自是否可信。

    core 取粗上下边界之间该列的 30 百分位，再做列向滚动中位数平滑。不能改用
    "粗 mask 中间 50% 行"：粗 mask 在某些列会混进大片亮背景，该列芯色被高估到
    255，tau 与越阈水平随之失效（05 号上边界因此整体偏上约 30px）。
    bg 是粗上边界上方的参照色、bg2 是粗下边界下方的参照色。ROI 顶边本身就是
    胶条时上方并无背景，但顶边那一行仍是有效的亮度参照，故只要求"至少有
    profile.bg_min 行、且比芯色亮出 profile.bg_contrast"。

    端头方向复用本函数时，"上/下"是转置后的左/右，参数由 profile 给定——
    端头的量程只有一个胶条厚度，core_win 必须跟着缩，否则滚动中位数窗会盖住
    半个端头、把逐行芯色抹平。
    """
    height, width = roi.shape
    top = _fill_nan(rough_top)
    bottom = _fill_nan(rough_bottom)
    core = np.empty(width)
    bg = np.empty(width)
    bg2 = np.empty(width)
    ok = np.zeros(width, bool)
    ok2 = np.zeros(width, bool)
    for x in range(width):
        y0 = int(np.clip(top[x], 0, height - 1))
        y1 = int(np.clip(bottom[x], y0 + 1, height))
        core[x] = float(np.percentile(roi[y0:y1, x], profile.core_pct)) if y1 > y0 else 128.0
    core = _rolling_median(core, profile.core_win)
    for x in range(width):
        y1 = max(1, int(top[x]) - profile.bg_off)
        y0 = max(0, y1 - profile.bg_span)
        bg[x] = float(np.percentile(roi[y0:y1, x], profile.bg_pct)) if y1 > y0 else 255.0
        ok[x] = (y1 - y0 >= profile.bg_min) and (bg[x] - core[x] >= profile.bg_contrast)
        y0 = min(height - 1, int(bottom[x]) + profile.bg_off)
        y1 = min(height, y0 + profile.bg_span)
        bg2[x] = float(np.percentile(roi[y0:y1, x], profile.bg_pct)) if y1 > y0 else 255.0
        ok2[x] = (y1 - y0 >= profile.bg_min) and (bg2[x] - core[x] >= profile.bg_contrast)
    return core, bg, bg2, ok, ok2


def _search_bands(
    rough: np.ndarray, profile: "EdgeProfile", side: str, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """生成这一侧的搜索带 [band_lo, band_hi]（含端点，逐列）。

    长边方向围绕逐列粗边界开带（profile.anchor_pct 为 None）；端头方向的外侧
    改用一个**全行共用**的锚点——粗边界的低百分位。这个区别是端头方向的关键，
    理由见 EDGE_CAP_* 块：斜长边行的粗边界已经深入胶条，拿它当外锚会把整段
    端头排除在搜索带外。
    """
    finite = np.isfinite(rough)
    filled = _fill_nan(rough)
    if profile.anchor_pct is None:
        outer = filled
    else:
        # 外侧锚点：左端取低百分位、右端取高百分位，都是"最靠外"的那一端。
        pct = profile.anchor_pct if side == "lo" else 100.0 - profile.anchor_pct
        anchor = float(np.percentile(rough[finite], pct)) if finite.any() else 0.0
        outer = np.full_like(filled, anchor)
    open_ended = profile.anchor_pct is not None
    if side == "lo":
        # lo = 取小坐标的那一侧（上边界 / 左端头）
        lo = outer - profile.out_off
        hi = filled + profile.in_off
        if open_ended:
            # 粗边界缺失的行不该被搜索带挡住，内侧放开到量程另一端
            hi = np.where(finite, hi, height - 1)
        elif EDGE_OUTER_OPEN_TOL > 0:
            # 粗掩膜"整片"盖到 ROI 顶时，粗上边界=0 并不代表真边就在附近，
            # 内侧不放开就会把真边切在带外，见 EDGE_OUTER_OPEN_TOL 块。
            deg = finite & (rough <= EDGE_OUTER_OPEN_TOL)
            hi = np.where(deg, height - 1, hi)
    else:
        lo = filled - profile.in_off
        hi = outer + profile.out_off
        if open_ended:
            lo = np.where(finite, lo, 0)
        elif EDGE_OUTER_OPEN_TOL > 0:
            deg = finite & (rough >= height - 1 - EDGE_OUTER_OPEN_TOL)
            lo = np.where(deg, 0, lo)
    return np.clip(lo, 0, height - 1).astype(int), np.clip(hi, 0, height - 1).astype(int)


def _outermost(sel: np.ndarray, side: str) -> np.ndarray:
    """在布尔图 sel 上逐列取**最外侧**的合格行号（无合格行给 NaN）。

    "外侧"= 坐标小的一侧（上边界 / 左端头），另一侧取反。
    """
    height = sel.shape[0]
    if side == "lo":
        has = sel.any(axis=0)
        idx = np.argmax(sel, axis=0)
    else:
        has = sel[::-1].any(axis=0)
        idx = height - 1 - np.argmax(sel[::-1], axis=0)
    return np.where(has, idx, np.nan).astype(float)


def _outer_candidates(
    edge: np.ndarray, inner_min: np.ndarray, tau: np.ndarray,
    band_lo: np.ndarray, band_hi: np.ndarray, side: str, profile: "EdgeProfile",
    outer_min: np.ndarray | None = None, bg: np.ndarray | None = None,
) -> np.ndarray:
    """逐列取搜索带内最外侧的强边候选 y（NaN 表示该列无候选）。

    坐标小的一侧（上边界 / 左端头）取自小而大的第一个合格行，另一侧取反，
    即"外轮廓"。

    ``outer_min``/``bg`` 只供**高光兜底**用（见 EDGE_OUTER_GLARE_*）：任一为 None
    或 ``profile.glare_fallback <= 0`` 时该兜底关闭，行为与它引入前逐位一致。
    """
    height = edge.shape[0]
    ys = np.arange(height)[:, None]
    for thr in (profile.grad, profile.grad_weak):
        sel = (
            (edge >= thr)
            & (inner_min <= tau[None, :])
            & (ys >= band_lo[None, :])
            & (ys <= band_hi[None, :])
        )
        found = _outermost(sel, side)
        if thr == profile.grad:
            strong = found
        else:
            # 强边不足时用弱边兜底：胶条顶边若是缓坡，整列找不到强边，
            # 只靠邻居插值会一路塌到胶条内部的纹理边上。
            cand = np.where(np.isnan(strong), found, strong)
            break

    if outer_min is not None and bg is not None and profile.glare_fallback > 0:
        # 高光兜底：判据是"离开背景"——外侧那半窗仍在背景水平、内侧这半窗已经低于
        # 它；取最外侧的这样一行 d，即背景→亮顶面的过渡起点。
        # 它只做**外侧上限夹逼**：d 比现有候选更靠外、且距离 ≤ glare_pull 时才改。
        # 单独用它代替候选是坏估计（实测这批数据 |均| 17.7px、80% 超 5px），但当上限
        # 用是净改善（+0.06px），因为它只在候选明显偏内的那些列生效。
        lim = bg[None, :] - profile.glare_drop
        sel = (
            (outer_min >= lim)
            & (inner_min <= lim)
            & (inner_min >= profile.glare_lo)
            & (edge >= profile.grad_weak)     # 必须踩在真实过渡上，不是平坦区
            & (ys >= band_lo[None, :])
            & (ys <= band_hi[None, :])
        )
        d = _outermost(sel, side)
        edge_inf = np.inf if side == "lo" else -np.inf
        cand_f = np.nan_to_num(cand, nan=edge_inf)
        gap = (cand_f - d) if side == "lo" else (d - cand_f)
        near = (gap >= profile.glare_min_gap) & (np.isnan(cand) | (gap <= profile.glare_pull))
        cand = np.where(~np.isnan(d) & near, d, cand)
    return cand


def _level_crossing(
    img: np.ndarray, level: np.ndarray,
    band_lo: np.ndarray, band_hi: np.ndarray, side: str, profile: "EdgeProfile",
) -> np.ndarray:
    """逐列最外侧的"连续 profile.level_run 行低于 level"起点/终点。"""
    height, width = img.shape
    run = profile.level_run
    wmin = cv2.dilate(img, np.ones((run, 1), np.uint8))  # 纵向极大 = 该窗口最大值
    shift = run // 2
    cond = np.zeros((height, width), bool)
    n = height - run + 1
    if n > 0:
        cond[shift : shift + n] = wmin[shift : shift + n] <= level[None, :]
    ys = np.arange(height)[:, None]
    sel = cond & (ys >= band_lo[None, :]) & (ys <= band_hi[None, :])
    if side == "lo":
        has = sel.any(axis=0)
        idx = np.argmax(sel, axis=0)
    else:
        has = sel[::-1].any(axis=0)
        idx = height - 1 - np.argmax(sel[::-1], axis=0)
    return np.where(has, idx, np.nan).astype(float)


def _dp_path(
    cost: np.ndarray, band_lo: np.ndarray, band_hi: np.ndarray, profile: "EdgeProfile"
) -> np.ndarray:
    """在逐列允许的 [band_lo, band_hi] 内求最小代价路径，返回每列 y（NaN=不可行）。"""
    height, width = cost.shape
    ys = np.arange(height)[:, None]
    valid = (ys >= band_lo[None, :]) & (ys <= band_hi[None, :])
    c = np.where(valid, cost, 1e9)
    dp = c[:, 0].copy()
    back = np.zeros((height, width), np.int32)
    for x in range(1, width):
        best = np.full(height, np.inf)
        arg = np.zeros(height, np.int32)
        for d in range(-profile.jump, profile.jump + 1):
            if d >= height or -d >= height:
                continue  # 跳变超过图高，整列无可行起点
            shifted = np.full(height, np.inf)
            if d >= 0:
                shifted[d:] = dp[: height - d]
            else:
                shifted[: height + d] = dp[-d:]
            cand = shifted + profile.pen * abs(d)
            upd = cand < best
            best[upd] = cand[upd]
            arg[upd] = ys.ravel()[upd] - d
        dp = best + c[:, x]
        back[:, x] = arg
    dp = np.where(valid[:, -1], dp, np.inf)
    y = int(np.argmin(dp))
    if not np.isfinite(dp[y]):
        return np.full(width, np.nan)
    path = np.zeros(width, np.int32)
    path[-1] = y
    for x in range(width - 1, 0, -1):
        path[x - 1] = back[path[x], x]
    return path.astype(float)


def _refine_axis(roi: np.ndarray, mask: np.ndarray, profile: EdgeProfile) -> AxisEdges:
    """沿 roi 的列方向精修每列两侧的边界，返回两条曲线与中间量。

    传入**转置过的** roi 和 mask，这里算出来的就是原图里逐行的左/右边界——
    这是端头方向复用的全部机制，本函数对方向无感。
    """
    mask = (mask > 0).astype(np.uint8)
    height, width = roi.shape
    rough_lo = np.full(width, np.nan)
    rough_hi = np.full(width, np.nan)
    for x in range(width):
        col = np.nonzero(mask[:, x])[0]
        if len(col):
            rough_lo[x] = col.min()
            rough_hi[x] = col.max()

    img = cv2.blur(roi.astype(np.float32), (profile.win_x, 1))
    grad = np.zeros_like(img)
    grad[1:] = img[:-1] - img[1:]
    grad[0] = grad[1]
    half = profile.dark_win // 2
    mn = cv2.erode(img, np.ones((1, profile.dark_win), np.uint8))
    core, bg, bg2, bg_ok, bg2_ok = _column_levels(roi, rough_lo, rough_hi, profile)
    tau = core + profile.core_margin
    level = core + profile.level_alpha * (bg - core)
    level2 = core + profile.level_alpha * (bg2 - core)

    curves = {}
    cands = {}
    for side, edge, rough, lv, outer in (
        ("lo", np.maximum(grad, 0), rough_lo, level, bg_ok),
        ("hi", np.maximum(-grad, 0), rough_hi, level2, bg2_ok),
    ):
        band_lo, band_hi = _search_bands(rough, profile, side, height)
        if profile.edge_skip > 0:
            # 粗掩膜溢到 ROI 边时，最外侧那几列会造出一条假强梯度，把端头往里
            # 带偏。把搜索带从量程两端各收掉这么多——候选、越阈、DP 一致生效。
            band_lo = np.maximum(band_lo, profile.edge_skip)
            band_hi = np.minimum(band_hi, height - 1 - profile.edge_skip)
        if side == "lo":
            inner_min = np.vstack([mn[half:], np.repeat(mn[-1:], half, axis=0)])
            outer_min = np.vstack([np.repeat(mn[:1], half, axis=0), mn[:-half]])
        else:
            inner_min = np.vstack([np.repeat(mn[:1], half, axis=0), mn[:-half]])
            outer_min = np.vstack([mn[half:], np.repeat(mn[-1:], half, axis=0)])
        cand = _outer_candidates(
            edge, inner_min, tau, band_lo, band_hi, side, profile,
            outer_min=outer_min, bg=(bg if side == "lo" else bg2))
        if profile.level_slack >= 0:
            y_lvl = _level_crossing(img, lv, band_lo, band_hi, side, profile)
            y_lvl = np.where(outer, y_lvl, np.nan)  # 该侧外侧无参照时不采信
            if profile.max_pull >= 0:
                # 越阈点离候选太远就整列作废：那种情况是越阈点自己被抬高的 L 勾走，
                # 不是候选落在高光带内沿。理由与实测见 EDGE_OUTER_LEVEL_MAX_PULL。
                y_lvl = np.where(np.abs(y_lvl - cand) > profile.max_pull, np.nan, y_lvl)
            if side == "lo":
                bad = ~np.isnan(cand) & ~np.isnan(y_lvl) & (cand > y_lvl + profile.level_slack)
            else:
                bad = ~np.isnan(cand) & ~np.isnan(y_lvl) & (cand < y_lvl - profile.level_slack)
            cand = np.where(bad | np.isnan(cand), y_lvl, cand)
            if profile.prefer_outer_level:
                # 端头专属。真值画在模糊斜坡的**起跳点**上（灰度刚离开背景那一两
                # 像素），而梯度候选落在斜坡最陡处，两者差半个斜坡宽 —— 端头因此
                # 系统性偏内 5~7px。亮度穿越点正好是"离开背景"的那一步，比梯度更
                # 接近起跳点，所以允许它把候选往**外**拉；上限仍是 level_slack，
                # 免得背景里的噪点把边界甩出去。长边方向不开这个开关：那边实测
                # 起跳点与最陡点几乎重合，开了只会放大噪声。
                if side == "lo":
                    pull = (~np.isnan(y_lvl) & (y_lvl < cand)
                            & (y_lvl >= cand - profile.level_slack))
                else:
                    pull = (~np.isnan(y_lvl) & (y_lvl > cand)
                            & (y_lvl <= cand + profile.level_slack))
                cand = np.where(pull, y_lvl, cand)
        cands[side] = cand
        if not np.isnan(cand).all():
            dev = np.abs(np.arange(height)[:, None] - _fill_nan(cand)[None, :])
            cost = profile.dev_w * np.minimum(dev, profile.dev_clip)
            # 没有候选的列不给数据项。_fill_nan 是 np.interp，它只能"连直线"，
            # 而候选缺口常常长达上百列：0914-07 条1 左端在 lx=0 有一个孤立候选
            # （y=0，ROI 顶边）、下一个候选在 156 列之后（y=66），中间整段没有
            # 任何证据，直线插值却把它写成一条 0→66 的斜坡，DP 照抄，左端因此
            # 比真值多标 91px。缺口段交给平滑项即可：两端有数据时它同样连成
            # 直线，而孤立的假候选不再能把它整段拉歪。
            cost[:, np.isnan(cand)] = 0.0
        else:
            cost = np.zeros((height, width), np.float32)
        curves[side] = _dp_path(cost.astype(np.float32), band_lo, band_hi, profile)

    return AxisEdges(
        lo_edge=curves["lo"], hi_edge=curves["hi"],
        rough_lo=rough_lo, rough_hi=rough_hi,
        cand_lo=cands["lo"], cand_hi=cands["hi"],
    )


def _fill_columns(mask: np.ndarray, edges: AxisEdges) -> np.ndarray:
    """把两条边界曲线填成 mask。粗 mask 无支撑的列原样退回。"""
    mask = (mask > 0).astype(np.uint8)
    height, width = mask.shape
    refined = np.zeros_like(mask)
    for x in range(width):
        # 粗 mask 完全没有支撑的列必须原样退回，不能填 DP 路径。
        # _fill_nan 用的是 np.interp，它**会外推**：区间外的列照样拿到有效的
        # 粗边界，于是搜索带成立、_dp_path 返回路径、这里就把整列填成胶条——
        # 精修 mask 于是从粗掩膜的实际范围一路撑到 ROI 边（实测 05 号从
        # x=662 撑到 649）。只 gate 输出，_fill_nan 本身不动，DP 内部的平滑
        # 仍要靠它跨越缺失列。
        if (np.isnan(edges.lo_edge[x]) or np.isnan(edges.hi_edge[x])
                or np.isnan(edges.rough_lo[x]) or np.isnan(edges.rough_hi[x])):
            refined[:, x] = mask[:, x]
            continue
        y0 = max(0, int(round(edges.lo_edge[x])))
        y1 = min(height, int(round(edges.hi_edge[x])) + 1)
        if y1 > y0:
            refined[y0:y1, x] = 1
    return refined


def _column_span(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """逐列取 mask 的上下极值，无支撑的列给 NaN。"""
    height, width = mask.shape
    lo = np.full(width, np.nan)
    hi = np.full(width, np.nan)
    for x in range(width):
        col = np.nonzero(mask[:, x])[0]
        if len(col):
            lo[x] = col.min()
            hi[x] = col.max()
    return lo, hi


def _coarse_run(coarse: np.ndarray, x: int, lo: float, hi: float) -> tuple[int, int] | None:
    """取第 x 列粗掩膜里**与 [lo, hi] 重叠**的那一段连续行程，返回 (上, 下)。

    不能直接用这一列的极值：粗掩膜在同一列里可能是几段互不相连的暗区（胶条上方
    还有别的深色结构、或闭运算在 ROI 边复制出来的伪影），取极值会跨过中间的空隙
    把整列焊成一块。

    ⚠️ **默认不"退回最近的一段"**（`EDGE_CAP_XNEAR=0`）。退回最近的一段会把与纵轴区间
    **完全不相交**的暗块并进来：实测 `11583 09` 右端，纵轴 `73..105`、粗掩膜是胶条下方
    一道 12px 的暗影 `113..124`，既不重叠又只差 8 行，并进来就在右下角留下一个 12x12 的
    孤立方块（用户报的"右下角有个方框"）。
    """
    rows = np.flatnonzero(coarse[:, x])
    if not rows.size or not (np.isfinite(lo) and np.isfinite(hi)):
        return None
    breaks = np.flatnonzero(np.diff(rows) > 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, rows.size - 1]
    best = None
    for s, e in zip(starts, ends):
        a, b = int(rows[s]), int(rows[e])
        if b >= lo and a <= hi:
            gain = min(b, hi) - max(a, lo)
            if best is None or gain > best[0]:
                best = (gain, a, b)
    if best is not None:
        return best[1], best[2]
    if not EDGE_CAP_XNEAR:
        return None
    d = [min(abs(int(rows[s]) - hi), abs(int(rows[e]) - lo))
         for s, e in zip(starts, ends)]
    i = int(np.argmin(d))
    return int(rows[starts[i]]), int(rows[ends[i]])


def _restore_cap_columns(
    coarse: np.ndarray,
    column: np.ndarray,
    roi_origin: tuple[int, int] = (0, 0),
    image_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """把每条胶条最外侧若干列的上下界退回粗掩膜，修端头转角处的纵轴内切。

    纵轴 DP 在端头转角处**失去证据**（那几列的候选是缺口，代价项被置 0），路径
    于是自由漂移到胶条里面。实测 ROI 第 384 列：粗掩膜下界 135、纵轴 113、真值
    138 —— 纵轴自己短了 25px，而端头轴那条曲线是准的（24 = 真值），是纵轴把它
    卡在里面。粗掩膜在端头那几列的**横向位置**本来就在真值 2~5px 以内，所以把
    最外 EDGE_CAP_XBAND 列的上下界改回粗掩膜的逐列上下界即可，body 的长边不动。

    只能回退**纵轴自己塌掉**的列：先量每一列纵轴结果的高度，超过中位高的
    EDGE_CAP_XTHIN 倍就说明纵轴在这一列还站得住，不动它。没有这道闸门时，回退会
    把粗掩膜整列多出来的阴影（上下各 20 多像素）带到长边上 —— 实测下长边 >5px
    从 1.5% 涨到 9.7%，而端头并没有更好。

    2026-09-21 起，XTHIN 不达标的列再走一遍**上边界逐侧**回退（EDGE_CAP_XGAP）：纵轴
    只塌了上边界、整列高度没塌的列，原来一次也救不到（实测 3911408-09 右端 283..293
    列的上边界整段偏深 ~20px，而 294..303 列塌够了才回退，一进一出成锯齿）。

    ⚠️ **画面切断的那一侧不回退。** 那里根本没有端头转角，而粗掩膜在 ROI 边上
    正好是闭运算复制边界造成的伪影——整列撑满 ROI 高。`CLOSE_KERNEL=(5,25)` 的
    25 宽横向闭运算在 ROI 边界按 `BORDER_REPLICATE` 复制像素、腐蚀再也剥不回来
    （见说明文档 §3.5.2）。实测 `11142 08`：ROI 起点就在画面左边（x=0），粗掩膜
    在第 0 列是 0..128 撑满，纵轴是 65..115，回退把整根 129px 高的竖条焊在了
    51px 厚的胶条上（用户报的"左侧端头完全改坏"）。
    """
    height, width = coarse.shape
    ox = roi_origin[0]
    frame_left = EDGE_CAP_XFRAME and ox <= EDGE_CAP_BORDER_TOL
    frame_right = (EDGE_CAP_XFRAME and image_shape is not None
                   and ox + width >= image_shape[1] - EDGE_CAP_BORDER_TOL)
    llo, lhi = _column_span(column)
    height_col = lhi - llo + 1
    finite = np.isfinite(height_col)
    med = float(np.nanmedian(height_col[finite])) if finite.any() else np.nan
    out = column.copy()
    supported = np.flatnonzero(np.isfinite(llo))
    if not supported.size:
        return out
    spans = [(supported.min(), supported.min() + EDGE_CAP_XBAND, frame_left),
             (supported.max() - EDGE_CAP_XBAND, supported.max(), frame_right)]
    for x0, x1, skip in spans:
        if skip:
            continue
        for x in range(max(0, int(x0)), min(width - 1, int(x1)) + 1):
            if not np.isfinite(height_col[x]) or not np.isfinite(med):
                continue
            thin = height_col[x] <= EDGE_CAP_XTHIN * med
            run = _coarse_run(coarse, x, llo[x], lhi[x])
            if run is None:
                continue
            y0, y1 = run
            if EDGE_CAP_XFULL and y0 <= 0 and y1 >= height - 1:
                # 粗掩膜把胶条和周围的暗区连成一整片、撑满整个 ROI 高。这种"行程"
                # 跟胶条没关系，并进来会在端头竖起一根贯穿画面的竖线（实测 09849 09
                # 右端第 307 列：纵轴 36..95，粗行程 0..122）。
                continue
            if EDGE_CAP_XFIT > 0 and y1 - y0 + 1 > EDGE_CAP_XFIT * med:
                # 粗掩膜在这一列把胶条和旁边的暗区连成了一整段（能撑到 ROI 顶/底），
                # 回退会把这一段整个并进来，在端头竖起一根比胶条高得多的柱子。
                continue
            if thin:
                if EDGE_CAP_XCLIP > 0:
                    # 粗掩膜整列含阴影，上下各多 20 多像素。允许外扩但设上限，免得
                    # 端头转角（真值本来就收窄）那几列把长边一起拖出去。
                    y0 = max(y0, int(llo[x]) - EDGE_CAP_XCLIP)
                    y1 = min(y1, int(lhi[x]) + EDGE_CAP_XCLIP)
                if y1 >= y0:
                    out[y0:y1 + 1, x] = 1
                continue
            if EDGE_CAP_XGAP <= 0:
                continue          # 纵轴在这一列还站得住 -> 不覆盖整列
            # 逐侧回退（见 EDGE_CAP_XGAP 块）：只补粗行程更外、且不在 ROI 边上的上边界
            if y0 > EDGE_CAP_BORDER_TOL and llo[x] - y0 > EDGE_CAP_XGAP:
                top = y0 if EDGE_CAP_XGAP_MAX <= 0 else max(y0, int(llo[x]) - EDGE_CAP_XGAP_MAX)
                out[int(top):int(llo[x]), x] = 1
    return out


def refine_strip_edges_outer(roi: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """用逐列"外轮廓强边候选 + 越阈兜底 + DP 平滑"精修胶条 mask 的上下边界。

    人工真值画的是胶条外轮廓（由外向内第一次进入胶条的那条边）。逐列找
    最强梯度会失准：胶条顶面常有镜面高光带，其下沿的梯度比真顶边还强。
    因此候选取"从粗边界外向内、首个梯度达标且内侧确实是暗区"的行，
    再用越阈点约束候选不得深入胶条内部，最后用 DP 连成横向连续的曲线。
    参数含义与实测效果见上方 EDGE_OUTER_* 常量注释。

    **只做上下边界。** 需要连左右端头一起精修的走 refine_strip_mask。
    """
    return _fill_columns(mask, _refine_axis(roi, mask, _long_profile()))


def refine_strip_ends(
    roi: np.ndarray,
    mask: np.ndarray,
    roi_origin: tuple[int, int] = (0, 0),
    image_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """逐行精修左右端头，返回 (左边界, 右边界, 说明)，ROI 坐标系，NaN = 该行无解。

    端头方向复用了长边那套机制（转置）与同一组 EDGE_CAP_* 参数，但搜索带的开法
    和画面边规则是端头独有的，理由见 EDGE_CAP_* 块。

    ``roi_origin`` 是 ROI 左上角在原图里的坐标，``image_shape`` 是原图尺寸；
    给全了才知道哪一侧的 ROI 边其实就是画面边界。
    """
    profile = _cap_profile()
    edges = _refine_axis(
        np.ascontiguousarray(roi.T),
        np.ascontiguousarray((mask > 0).astype(np.uint8).T),
        profile,
    )
    # 转置后 roi.T 的形状是 (W, H)，所以 _refine_axis 的"列数"= 原图行数 H，
    # 而每条曲线的**取值**范围是原图的列坐标 [0, W-1]。两者别混。
    rows = roi.shape[0]   # 曲线长度 = 原图行数
    width = roi.shape[1]  # 边界坐标量程 = 原图列数
    left, right = edges.lo_edge, edges.hi_edge
    notes = []

    # 候选覆盖率闸门：覆盖不足时 _fill_nan 的平外推会把个别 runaway 候选
    # 放大成整条过切（实测 +31.5 / +41.2px）。这种条多半是被画面切断或红框
    # 贴边的，保持宽松才对，所以整侧放弃收缩。
    for label, cand in (("左", edges.cand_lo), ("右", edges.cand_hi)):
        hit = int(np.isfinite(cand).sum())
        if hit >= profile.min_cand * rows:
            continue
        notes.append(f"{label}端头候选覆盖不足（{hit}/{rows} 行），该侧不收缩")
        blank = np.full(rows, np.nan)
        if label == "左":
            left = blank
        else:
            right = blank

    # 画面边界规则：ROI 的某一侧本来就是画面边界时，胶条是被画面切断的，
    # 端头不在画面内，该侧必须保留 ROI 边、不能收缩。必须有**两个**条件——
    # 光看 ROI 位置不够：红框恰好画在离画面 2px 处、而胶条其实没到画面时，
    # 会误伤一条本来有端头的胶条。所以还要求粗 mask 真的顶到了那条 ROI 边。
    ox = roi_origin[0]
    left_at_frame = ox <= EDGE_CAP_BORDER_TOL
    right_at_frame = (
        image_shape is not None and ox + width >= image_shape[1] - EDGE_CAP_BORDER_TOL
    )
    tol = EDGE_CAP_BORDER_TOL
    if left_at_frame:
        keep = np.isfinite(edges.rough_lo) & (edges.rough_lo <= tol)
        left = np.where(keep, 0.0, left)
    if right_at_frame:
        keep = np.isfinite(edges.rough_hi) & (edges.rough_hi >= width - 1 - tol)
        right = np.where(keep, float(width - 1), right)
    return left, right, notes


def refine_strip_mask(
    roi: np.ndarray,
    mask: np.ndarray,
    roi_origin: tuple[int, int] = (0, 0),
    image_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """生产入口：逐列上下边界 + 逐行左右端头，两条轴取交集。返回 (mask, 说明)。

    两条轴都只可能把 mask 收窄，不会撑大。角上两个方向同时收会在对角上多切
    几像素——这也是上下边界那边不能反过来再叠一个"偏内"偏置的原因。

    例外是端头那几列：纵轴在端头转角处失去证据、会往里切 20 多像素，那几列先用
    粗掩膜补回来（见 _restore_cap_columns），再和端头曲线取交集。补回来的列仍受
    端头曲线约束，所以横向不会被撑宽。
    """
    binary = (mask > 0).astype(np.uint8)
    column = _fill_columns(binary, _refine_axis(roi, binary, _long_profile()))
    if not column.any():
        return column, []
    column = _restore_cap_columns(binary, column, roi_origin, image_shape)
    left, right, notes = refine_strip_ends(roi, binary, roi_origin, image_shape)

    # 边界曲线为 NaN 的行**当宽松处理**，不退回粗边界：粗边界最不可靠的行恰恰
    # 就是这些行，退回去等于在最不该切的地方切一刀。
    width = roi.shape[1]
    columns = np.arange(width)[None, :]
    lo_open = np.where(np.isnan(left), -1.0, left)[:, None]
    hi_open = np.where(np.isnan(right), float(width), right)[:, None]
    trimmed = column & (columns >= lo_open) & (columns <= hi_open)

    kept = int(trimmed.sum())
    if kept == 0 or kept < EDGE_CAP_AREA_FUSE * int(column.sum()):
        notes.append(
            f"左右收缩被放弃：面积从 {int(column.sum())}px 掉到 {kept}px，"
            f"该处退回只做上下精修"
        )
        return column, notes
    return trimmed.astype(np.uint8), notes


def process_image(image_path: Path, label_path: Path, out_crops: Path, out_vis: Path) -> SegmentRecord:
    record = SegmentRecord(file=image_path.name, green_box=(0, 0, 0, 0), red_boxes=[], crop_size=(0, 0))
    try:
        gray = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            record.error = "无法读取原图"
            return record
        green_boxes, red_boxes = extract_boxes(label_path)
        if not red_boxes:
            record.error = "未提取到红色 GAD 框"
            return record
        annotated_label = cv2.imdecode(np.fromfile(label_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        red_mapped = map_boxes(red_boxes, annotated_label.shape[:2], gray.shape)
        if green_boxes:
            green_box = map_boxes(green_boxes, annotated_label.shape[:2], gray.shape)[0]
        else:
            # 未标绿框：以全部红框包围盒外扩边距作为裁剪区域
            min_x = min(b[0] for b in red_mapped)
            min_y = min(b[1] for b in red_mapped)
            max_x = max(b[0] + b[2] for b in red_mapped)
            max_y = max(b[1] + b[3] for b in red_mapped)
            left = max(0, min_x - CROP_MARGIN)
            top = max(0, min_y - CROP_MARGIN)
            right = min(gray.shape[1], max_x + CROP_MARGIN)
            bottom = min(gray.shape[0], max_y + CROP_MARGIN)
            green_box = (left, top, right - left, bottom - top)
        record.green_box = green_box
        record.red_boxes = red_mapped

        gx, gy, gw, gh = green_box
        crop = gray[gy : gy + gh, gx : gx + gw]
        record.crop_size = crop.shape

        vis = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        for rx, ry, rw, rh in red_mapped:
            cx, cy = rx - gx, ry - gy
            cv2.rectangle(vis, (cx, cy), (cx + rw - 1, cy + rh - 1), (0, 0, 255), 2)
            roi = crop[cy : cy + rh, cx : cx + rw]
            mask = segment_dark_strip(roi)
            if mask is None:
                record.strips.append({"red_box": [rx, ry, rw, rh], "status": "未分割到深色胶条"})
                continue
            # ROI 原点要按**原图**坐标给：roi 取自 crop，crop 的原点是 (gx, gy)，
            # 而 cx = rx - gx，所以 roi 在原图里的原点就是 (rx, ry)。画面边规则
            # 拿它跟 gray.shape 比，判断这一侧是不是被画面切掉、该保留成画面边界。
            mask, notes = refine_strip_mask(roi, mask, (rx, ry), gray.shape)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                contour[:, 0, 0] += cx
                contour[:, 0, 1] += cy
                cv2.drawContours(vis, [contour], -1, (0, 255, 255), 1)
            stats = strip_geometry(mask)
            entry = {"red_box": [rx, ry, rw, rh], "status": "已分割", **stats}
            if notes:
                entry["refine_notes"] = notes
            record.strips.append(entry)

        out_crops.mkdir(parents=True, exist_ok=True)
        cv2.imencode(".png", crop)[1].tofile(out_crops / f"{image_path.stem}_裁剪.png")
        out_vis.mkdir(parents=True, exist_ok=True)
        cv2.imencode(".png", vis)[1].tofile(out_vis / f"{image_path.stem}_分割.png")
    except Exception as exc:  # 单张失败不影响批处理
        record.error = f"{type(exc).__name__}: {exc}"
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="按绿框裁剪并在红框内分割深色 GAD 胶条")
    parser.add_argument("input", type=Path, help="含原图 BMP 与标注图子目录的车身文件夹")
    parser.add_argument("--output", type=Path, default=None, help="输出根目录，默认为输入目录下 output_segmentation")
    args = parser.parse_args()

    input_dir = args.input
    label_dir = input_dir / LABEL_DIR_NAME
    if not label_dir.exists():
        raise SystemExit(f"未找到标注目录: {label_dir}")
    out_root = args.output or input_dir / "output_segmentation"
    out_crops = out_root / "裁剪图"
    out_vis = out_root / "分割结果"

    labels = {label_stem(path): path for path in label_dir.glob("*.png")}
    results = []
    for stem, label_path in sorted(labels.items()):
        bmp = input_dir / f"{stem}.bmp"
        if not bmp.exists():
            continue
        record = process_image(bmp, label_path, out_crops, out_vis)
        results.append(record)
        status = f"红框{len(record.red_boxes)}个, 分割{sum(1 for s in record.strips if s['status'] == '已分割')}处" if not record.error else record.error
        print(f"{record.file}  {status}")

    report_path = out_root / "segmentation_report.json"
    report_path.write_text(json.dumps([asdict(r) for r in results], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"报告: {report_path}")


if __name__ == "__main__":
    main()
