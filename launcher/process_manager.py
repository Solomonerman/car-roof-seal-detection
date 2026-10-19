# -*- coding: utf-8 -*-
"""
工控机启停台 - 进程管理核心(不依赖 GUI, 可在无显示器环境单测)
负责: 拉起子进程、停止(发 Ctrl+C 优雅退出 + taskkill 兜底)、PID 追踪。

关闭机制说明(不改主程序 web_capture.py):
  web_capture.py 现有退出逻辑 = except KeyboardInterrupt -> finally: 释放相机/PLC。
  Windows 下 subprocess 子进程默认收不到 Ctrl+C, 故用 AttachConsole + GenerateConsoleCtrlEvent
  向子进程控制台发 Ctrl+C, 触发其 finally 干净释放。Linux 直接 os.kill(pid, SIGINT)。
  等 ~8s 仍存活再 taskkill /T /F 兜底(极端才用; 进程退出后 pylon 由内核回收相机句柄)。
"""
import os
import sys
import time
import signal
import subprocess

try:
    import ctypes
    _HAS_WIN = sys.platform == "win32"
except Exception:
    _HAS_WIN = False


def start_item(py, item, repo_root):
    """拉起一个启动项, 返回 (Popen, 日志路径)"""
    cmd = [py] + list(item["cmd"])
    log_path = os.path.join(repo_root, item["log"])
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logf = open(log_path, "a", encoding="utf-8")
    # CREATE_NEW_PROCESS_GROUP: 子进程独立进程组, 便于单独发 Ctrl+C
    p = subprocess.Popen(
        cmd,
        cwd=repo_root,
        stdout=logf,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if _HAS_WIN else 0,
    )
    return p, log_path


def _send_ctrl_c(pid):
    """向子进程发 Ctrl+C(优雅退出)。Windows 用 AttachConsole 技巧; 其他平台 os.kill。"""
    if _HAS_WIN:
        try:
            k = ctypes.windll.kernel32
            k.AttachConsole(pid)
            k.GenerateConsoleCtrlEvent(signal.CTRL_C_EVENT, pid)
            k.FreeConsole()
            return
        except Exception:
            pass
    try:
        os.kill(pid, signal.SIGINT)
    except Exception:
        pass


def stop_item(p, timeout=8):
    """优雅停止: 先发 Ctrl+C, 等 timeout 秒, 仍存活则 taskkill /T /F 兜底"""
    if p is None:
        return True
    if p.poll() is not None:
        return True
    _send_ctrl_c(p.pid)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if p.poll() is not None:
            return True
        time.sleep(0.1)
    # 兜底强杀(极端才用)
    try:
        subprocess.run(
            ["taskkill", "/PID", str(p.pid), "/T", "/F"],
            capture_output=True,
        )
    except Exception:
        pass
    return p.poll() is not None
