# -*- coding: utf-8 -*-
"""
工控机启停台 - 进程管理核心(不依赖 GUI, 可在无显示器环境单测)
负责: 拉起子进程、停止(发 Ctrl+C 优雅退出 + taskkill 兜底)、PID 追踪。

关闭机制说明(不改主程序 web_capture.py):
  web_capture.py 现有退出逻辑 = except KeyboardInterrupt -> finally: 释放相机/PLC。
  Windows 下 subprocess 子进程默认收不到 Ctrl+C, 故用 AttachConsole + GenerateConsoleCtrlEvent
  向子进程控制台发 Ctrl+Break(比 Ctrl+C 跨进程组送达更可靠; Python 的 console 控制处理会把
  BREAK 也转成 KeyboardInterrupt, 同样触发 finally 干净释放)。发送前用 SetConsoleCtrlHandler
  关掉 launcher 自身处理, 避免 BREAK 误伤 launcher。Linux 直接 os.kill(pid, SIGINT)。
  等 ~8s 仍存活再 taskkill /T /F 兜底(极端才用; 进程退出后 pylon 由内核回收相机句柄)。
"""
import os
import sys
import time
import signal
import csv
import io
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
    logf = open(log_path, "a", encoding="utf-8-sig")
    kwargs = dict(cwd=repo_root, stdout=logf, stderr=subprocess.STDOUT)
    if _HAS_WIN:
        # 完全隐藏子进程控制台窗口(SW_HIDE): 不进任务栏、桌面不可见, 不会被误关;
        # 但控制台对象仍存在 -> [停止] 仍可 AttachConsole 发 Ctrl+Break 优雅退出释放相机。
        si = subprocess.STARTUPINFO()
        si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = subprocess.SW_HIDE
        kwargs["startupinfo"] = si
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    p = subprocess.Popen(cmd, **kwargs)
    return p, log_path


def _send_ctrl_c(pid):
    """向子进程发 Ctrl+Break(优雅退出)。Windows 用 AttachConsole + GenerateConsoleCtrlEvent
    发 CTRL_BREAK_EVENT(比 CTRL_C 跨进程组送达更可靠), Python 会将其转成 KeyboardInterrupt
    从而触发 web_capture 的 finally 干净释放。发送前关掉 launcher 自身 Ctrl 处理以免误伤。
    其他平台直接 os.kill(pid, SIGINT)。"""
    if _HAS_WIN:
        try:
            k = ctypes.windll.kernel32
            if k.AttachConsole(pid):
                # 关掉本进程(launcher)自己的 Ctrl 处理, 避免 BREAK 事件误伤 launcher
                k.SetConsoleCtrlHandler(None, 1)
                k.GenerateConsoleCtrlEvent(signal.CTRL_BREAK_EVENT, pid)
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


def _list_python_cmds():
    """返回 [(pid, cmdline)] for all running python[w].exe 进程。
    用于跨实例检测'相机是否已被别的 web_capture 占用'(改主程序之外做防呆)。
    Windows 走 wmic; 其他平台走 ps。失败返回空列表(不影响启动, 仅失去防呆)。"""
    out = []
    if _HAS_WIN:
        try:
            proc = subprocess.run(
                ["wmic", "path", "win32_process", "where",
                 "name='python.exe' OR name='pythonw.exe'",
                 "get", "ProcessId,CommandLine", "/FORMAT:CSV"],
                capture_output=True, timeout=15,
            )
            text = proc.stdout.decode("utf-8", "replace")
            rows = list(csv.reader(io.StringIO(text)))
            if not rows:
                return []
            header = [h.strip() for h in rows[0]]
            try:
                i_pid = header.index("ProcessId")
                i_cmd = header.index("CommandLine")
            except ValueError:
                i_pid, i_cmd = 0, 1
            for r in rows[1:]:
                if len(r) <= max(i_pid, i_cmd):
                    continue
                try:
                    pid = int(r[i_pid])
                except ValueError:
                    continue
                out.append((pid, r[i_cmd] or ""))
        except Exception:
            pass
    else:
        try:
            proc = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True,
                                  text=True, timeout=15)
            for line in proc.stdout.splitlines()[1:]:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                try:
                    pid = int(parts[0])
                except ValueError:
                    continue
                out.append((pid, parts[1]))
        except Exception:
            pass
    return out


def find_stray_web_capture(exclude_pids):
    """扫描系统里所有 python 进程, 返回命令含 'web_capture.py' 且不在 exclude_pids 的 PID 列表。
    防呆目的: 防止'另一个启停台 / 手动 cmd 已拉起 web_capture 占着相机'导致双开撞锁
    (报错 The device is controlled by another application)。"""
    exclude = set(exclude_pids)
    strays = []
    for pid, cmd in _list_python_cmds():
        if pid in exclude:
            continue
        if "web_capture.py" in cmd:
            strays.append(pid)
    return strays


def _pid_alive(pid):
    if _HAS_WIN:
        try:
            proc = subprocess.run(["tasklist", "/fi", f"PID eq {pid}"],
                                  capture_output=True, text=True, timeout=5)
            return str(pid) in proc.stdout
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def stop_pids(pids, timeout=8):
    """停止一组外部 PID(非本 launcher 子进程): 发 Ctrl+Break 优雅退出触发相机释放,
    等退出, 仍存活则 taskkill /T /F 兜底。用于清理占用相机的旧 web_capture。"""
    pids = [p for p in pids if p and _pid_alive(p)]
    if not pids:
        return True
    for pid in pids:
        _send_ctrl_c(pid)
    deadline = time.time() + timeout
    remaining = set(pids)
    while time.time() < deadline and remaining:
        for pid in list(remaining):
            if not _pid_alive(pid):
                remaining.discard(pid)
        if remaining:
            time.sleep(0.2)
    for pid in list(remaining):
        try:
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                           capture_output=True)
        except Exception:
            pass
    return not remaining
