# -*- coding: utf-8 -*-
"""照明控制（mock 实现）。

真实场景替换为 GPIO / 串口 / 相机触发线控制。对外只暴露 on / off 两个方法，
采集模块在 start / stop 时调用，与相机取流同步开关。
"""


class Lighting:
    def on(self):
        print("[照明] 已开启")

    def off(self):
        print("[照明] 已关闭")
