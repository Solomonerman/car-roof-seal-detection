# 快速上手（笔记本 / PyCharm）

面向已安装 **Python 3.10 + PyCharm** 的开发机，用模拟图约 10 分钟跑通端到端 demo。
（云端沙盒用 3.11 搭建，本代码未用 3.11 专属语法，3.10 完全兼容。）

## 1. 获取代码
解压 `car_roof_seal_detection.zip` 到任意目录，例如 `D:\dev\car_roof_seal_detection`。

## 2. 用 PyCharm 打开
`File → Open` → 选择项目目录。PyCharm 会识别为 Python 项目。

## 3. 配置解释器（建议虚拟环境）
- PyCharm 弹出提示时确认创建 venv（基于 Python 3.10）；
- 或 `Settings → Python Interpreter → Add → New Virtual Environment`。

## 4. 安装依赖
```bash
pip install numpy opencv-python-headless Pillow fastapi
```
- 启动 Web 监控才需要 `uvicorn`（可选）：`pip install uvicorn`
- 笔记本有桌面显示、想弹窗看图的，可把 `opencv-python-headless` 换成 `opencv-python`

## 5. 运行 demo
右键 `main.py` → `Run 'main'`；或终端：
```bash
cd car_roof_seal_detection
python main.py          # 默认 6 轮
python main.py 10       # 指定 10 轮
```

## 6. 预期输出
```
=== 车顶胶条检测系统（沙盒 mock 模式）启动 ===
[照明] 已开启
[采集] 相机取流已启动
[照明] 已关闭
[采集] 相机取流已停止
[存储] 落库: 车型=Tiguan 结果=OK 图片=8张 缺陷=8处
[PLC] 回写检测结果 -> OK
[结果] 车型=Tiguan OK 图片=8 缺陷=8
```
> 部分轮次因模拟"车辆移动"而跳过（返回 None），属正常现象。

## 7. 换成你的现场真图
把现场拍摄的胶条相片（任意文件名，建议 8 张以上）放进：
```
car_roof_seal_detection/data/raw_images/
```
再次 `python main.py`，采集模块会**自动优先使用真图**。

## 8. 启动 Web 监控（可选）
```bash
uvicorn ui.app:app --reload --port 8000
```
浏览器打开 http://localhost:8000

## 常见问题
- **cv2 导入报错**：确认装的是 `opencv-python` 或 `opencv-python-headless`，没有单独的 `cv2` 包。
- **找不到图片**：必须在 `main.py` 所在项目根目录运行，相对路径 `data/raw_images` 才正确。
- **fastapi 缺失**：`main.py` 会 import `ui.app`，需安装 `fastapi`。

## 下一步
跑通后，把真图放进 `data/raw_images` 看检测效果，再带着真图表现回来，我们针对性增强
`detection/detector.py` 的胶条定位与 OK/NG 判定逻辑（可平滑替换为深度学习模型）。
