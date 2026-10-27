# -*- coding: utf-8 -*-
"""
工控机一键启停台 - 集中配置
只描述"拉起/停止哪两个程序、用什么解释器、端口",不含 GUI 逻辑。
启动项路径一律相对仓库根, 跨机(开发机/工控机)可跑, 不写死绝对路径。
"""
import os
import json
import subprocess

# 仓库根 = launcher/ 的上一级
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 本地配置(每台工控机不同, 不提交 git): 存放解释器路径等
LOCAL_CFG = os.path.join(REPO_ROOT, "launcher", "launch_config.json")

# 两个要拉起的程序(路径相对仓库根)
# 注意: --plc-auto 强制带, 否则现场只剩手动模式不会自动拍照
LAUNCH_ITEMS = [
    {
        "name": "相机采集 web_capture",
        "cmd": ["capture/web_capture.py", "--plc-auto"],
        "port": 5000,
        "log": "logs/工控机_web_capture.log",
        "open_url": "http://localhost:5000",   # 启动后自动开相机实时取景页
    },
    {
        "name": "监控看板 ui.app",
        "cmd": ["-m", "uvicorn", "ui.app:app", "--port", "8000"],
        "port": 8000,
        "log": "logs/工控机_ui.log",
        "open_url": "http://localhost:8000/board",   # 启动后自动开全屏看板(/board, 显示OK/NG)
    },
]


def find_python():
    """解释器探测优先级: 本地配置 > 仓库 .venv > 返回 None(交给 GUI 让用户选)"""
    if os.path.exists(LOCAL_CFG):
        try:
            with open(LOCAL_CFG, encoding="utf-8") as f:
                cfg = json.load(f)
            p = cfg.get("python")
            if p and os.path.exists(p):
                return p
        except Exception:
            pass
    venv = os.path.join(REPO_ROOT, ".venv", "Scripts", "python.exe")
    if os.path.exists(venv):
        return venv
    return None


def save_python(py):
    """把用户选定的解释器存到本地配置(不提交 git, 每台机各自一份)"""
    os.makedirs(os.path.dirname(LOCAL_CFG), exist_ok=True)
    with open(LOCAL_CFG, "w", encoding="utf-8") as f:
        json.dump({"python": py}, f, ensure_ascii=False, indent=2)


def python_works(py):
    """验证该解释器能 import 两个程序所需依赖, 避免拉起来直接崩(sys.exit)"""
    if not py or not os.path.exists(py):
        return False
    try:
        r = subprocess.run(
            [py, "-c", "import pypylon, cv2, flask, fastapi, uvicorn"],
            capture_output=True, text=True, timeout=60,
        )
        return r.returncode == 0
    except Exception:
        return False
