# 车顶胶条检测系统

基于计算机视觉的车顶胶条在线检测系统。覆盖从 PLC 信号、工业相机采集、
视觉算法检测、数据存储到结果回写与 UI 展示的完整工业流程。

> 当前仓库为 **沙盒 mock 框架**：所有硬件（PLC / 相机 / 照明 / 数据库）均用
> 占位实现驱动，整套软件流程已可端到端跑通。现场联调时只需替换各模块的
> mock 实现，对外接口保持不变，总控无需改动。

---

## 一、目录结构

```
car_roof_seal_detection/
├── main.py                 # 总控入口：装配模块、跑流程
├── requirements.txt        # 依赖清单（沙盒只需前 4 项）
├── config/
│   └── constants.py        # 全局常量（车型、采集张数、PLC地址、存储配置）
├── common/
│   └── interfaces.py       # 模块间数据结构与接口契约（dataclass）
├── plc/
│   ├── plc_controller.py   # PLC 通信（mock）
│   └── state_machine.py    # 总控状态机（流程编排，对应流程图）
├── capture/
│   ├── camera.py           # 相机采集（读现场相片 / mock 图）
│   └── lighting.py         # 照明控制（mock）
├── detection/
│   └── detector.py         # 视觉检测（占位算法，可换深度学习）
├── storage/
│   ├── database.py         # MySQL 存储（mock）
│   ├── object_store.py     # MinIO 对象存储（mock）
│   └── service.py          # 存储服务（聚合落库+存图）
├── ui/
│   └── app.py              # FastAPI 监控页面（/ 与 /api/records）
├── mock/
│   └── sample_images/      # 模拟胶条图（占位，无真图时回退）
└── data/
    ├── raw_images/         # ★ 把现场真实胶条相片拷到这里
    └── stored_images/      # 检测图片落盘（mock 阶段）
```

---

## 二、模块职责与流程对应

| 模块 | 流程图环节 | 职责 |
|------|-----------|------|
| `plc/state_machine` | 初始化→PLC通信→车型校验→触发 | 状态机编排，只决策不实现 |
| `plc/plc_controller` | 车辆移动 / 车型 / 结果回写 | PLC 信号读写（mock） |
| `capture/camera` + `lighting` | 启动取流+照明、采集8张 | 图像采集（读现场相片） |
| `detection/detector` | 触发检测线程、分车型检测 | 胶条定位与 OK/NG（占位） |
| `storage/*` | 写入 MySQL + MinIO | 落库 + 存图 |
| `ui/app` | UI 定时刷新展示 | 结果/图片展示 |

**接口契约**（见 `common/interfaces.py`）：模块间只通过这些数据结构通信——
`CaptureRequest / CaptureResult / Defect / DetectionResult / InspectionRecord`，
保证改算法、换相机、换数据库互不牵连。

---

## 三、本地运行（沙盒 demo）

```bash
cd car_roof_seal_detection
pip install -r requirements.txt        # 沙盒阶段装前 4 项即可
python main.py                         # 跑 6 轮 mock 演示
python main.py 10                      # 指定轮数

# 启动 Web 监控（可选，需取消 requirements 中 fastapi/uvicorn 注释并安装）
uvicorn ui.app:app --reload --port 8000
# 浏览器打开 http://localhost:8000
```

把现场真实胶条相片（任意文件名）放进 `data/raw_images/`，采集模块会自动优先使用。

---

## 四、现场联调：替换 mock 实现

各模块对外接口已固定，联调时只改内部实现：

| 模块 | mock → 真实 | 改动点 |
|------|------------|--------|
| PLC | `plc_controller` | 内部改为 `pymodbus` / `python-snap7` 读写真实寄存器，保留 `connect/read_car_model/is_vehicle_moving/write_result` |
| 相机 | `capture/camera` | `_list_images` / `capture` 改为相机 SDK 取帧，保留 `start/stop/capture` |
| 照明 | `capture/lighting` | `on/off` 改为 GPIO / 串口控制 |
| 算法 | `detection/detector` | `detect()` 内换为 YOLO / 分割模型，保留返回 `DetectionResult` |
| 存储 | `storage/database` + `object_store` | 内部改为 SQLAlchemy / MinIO SDK，保留 `save_record` / `upload` |

---

## 五、建议：拆分为 4 个 WorkBuddy 项目（对抗长上下文降智）

本项目横跨 PLC / 视觉 / 后端 / 前端多个领域。若长期在一个上下文里开发，
AI 注意力会被稀释、错误累积。建议按"一层一个项目"拆分，各自隔离上下文：

1. **车顶胶条检测 - 总控与 PLC**：状态机 + PLC 通信
2. **车顶胶条检测 - 图像采集**：相机 + 照明
3. **车顶胶条检测 - 视觉算法**：分车型检测（最核心、最常迭代）
4. **车顶胶条检测 - 数据与 UI**：MySQL / MinIO / FastAPI

各项目通过 `common/interfaces.py` 的契约与共享代码仓库协作；总控项目负责编排，
其余项目只实现自己的层。本仓库可作为"参考单体骨架"，拆分时按模块复制即可。

---

## 六、开发顺序建议

1. 总控状态机（已完成骨架，跑通 mock 流程）
2. 图像采集接入真实现场相片（替换 `_list_images`）
3. 视觉算法增强（先用真图调优占位逻辑，再换深度学习）
4. 数据与 UI 接真实 MySQL / MinIO / 前端
5. 现场联调：PLC + 相机 + 照明真实接入
