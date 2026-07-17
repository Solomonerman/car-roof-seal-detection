# -*- coding: utf-8 -*-
"""系统常量配置：所有模块共享的配置集中在此，便于现场调整，不散落各文件。"""
import os

# 项目根目录（本文件位于 config/ 下，回退两级到项目根）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 支持的车型（流程图中的分车型检测）
SUPPORTED_MODELS = ["Tiguan", "A5"]
DEFAULT_MODEL = "Tiguan"

# 图像采集参数（对应流程图：按时间节点采集 8 张图片）
CAPTURE_COUNT = 8                 # 每次采集图片数量
CAPTURE_INTERVAL_SEC = 0.5       # 相邻两张的时间间隔（秒）
IMAGE_WIDTH, IMAGE_HEIGHT = 640, 480

# 图像来源：优先用现场真实相片，缺失时回退到模拟图
# 使用绝对路径，保证无论从哪个工作目录启动都能找到
RAW_IMAGE_DIR = os.path.join(BASE_DIR, "data", "raw_images")
MOCK_IMAGE_DIR = os.path.join(BASE_DIR, "mock", "sample_images")

# PLC 通信配置（mock 阶段仅占位，现场联调时填真实值）
PLC_HOST = "192.168.0.10"
PLC_PORT = 502
PLC_MODEL_REGISTER = 0     # 车型寄存器地址
PLC_MOVE_REGISTER = 1      # 车辆移动状态寄存器
PLC_RESULT_REGISTER = 2    # 检测结果回写寄存器

# 存储配置（现场联调时启用）
MYSQL_URL = "mysql+pymysql://user:pass@localhost:3306/seal_db"
MINIO_ENDPOINT = "localhost:9000"
MINIO_BUCKET = "seal-images"
