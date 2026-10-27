# -*- coding: utf-8 -*-
"""
工控机一键启停台（GUI, tkinter, Python 自带零额外依赖）  —— 推导版

现场人员：双击桌面"启动启停台"快捷方式 -> 点 [▶ 启动] / [■ 停止]。

设计要点（针对过去反馈逐个根治）：
1) 窗口一定弹到最前面：root.lift() + 短暂 -topmost，杜绝"进程在、窗口看不见"。
2) 解释器单一来源、不弹窗：bat 已用仓库 .venv 的解释器拉起本 GUI；本 GUI 派生子进程时，
   直接用"与 GUI 同目录的 python.exe"，保证子进程与 GUI 同一环境（不再出现
   bat 用系统 pyw、子进程用 venv 的不一致）。不再弹"选择解释器"文件框。
3) 实时日志面板：过去"看不到 log"——现在 GUI 直接尾巴读取 web_capture / 看板的日志文件，
   实时显示在面板里，不必开 Terminal；日志文件同时落盘，可事后查看。
4) 无黑框：GUI 用 _hide_own_console() 隐藏自身控制台窗口；子进程控制台也设为 SW_HIDE。
5) 优雅停止：发 Ctrl+Break 触发 web_capture 的 finally 释放相机/PLC，超时再 taskkill 兜底。
6) 防双开：启动前扫描并清理占用相机的旧 web_capture，避免"The device is controlled by
   another application"。
7) 仅管进程启停，绝不读写 PLC（web_capture 本身只读）。主程序 web_capture.py 一行不改。
"""
import os
import sys
import ctypes
import webbrowser
import tkinter as tk
from tkinter import messagebox, scrolledtext

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import launch_config as C
import process_manager as PM

REPO_ROOT = C.REPO_ROOT


class LauncherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("车顶胶条检测 - 工控机启停台")
        self.root.geometry("640x560")
        # 强制窗口弹到最前面（根治"进程在、窗口看不见"）
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(1200, lambda: self.root.attributes("-topmost", False))

        self.items = {}  # name -> {"proc": Popen, "log": path, "pos": int, "lamp": Label, "port": int}

        # ---- 顶部状态 ----
        top = tk.Frame(root)
        top.pack(pady=8, fill=tk.X, padx=14)
        self.lamp = tk.Label(top, text="●", font=("Arial", 24), fg="gray")
        self.lamp.pack(side=tk.LEFT, padx=6)
        self.status_var = tk.StringVar(value="未启动")
        tk.Label(top, textvariable=self.status_var, font=("Arial", 15)).pack(side=tk.LEFT)

        # ---- 组件运行状态面板 ----
        comp_frm = tk.LabelFrame(root, text="组件运行状态", padx=10, pady=6)
        comp_frm.pack(fill=tk.X, padx=14, pady=4)
        for item in C.LAUNCH_ITEMS:
            row = tk.Frame(comp_frm)
            row.pack(fill=tk.X, pady=2)
            lamp = tk.Label(row, text="○", fg="gray", font=("Arial", 13))
            lamp.pack(side=tk.LEFT, padx=4)
            name = item["name"]
            tk.Label(row, text=name, width=30, anchor="w").pack(side=tk.LEFT)
            port = item.get("port")
            tk.Label(row, text=("端口 %d" % port) if port else "", width=10).pack(side=tk.LEFT)
            tk.Button(row, text="日志", width=6,
                      command=lambda p=item["log"]: self.open_log(p)).pack(side=tk.RIGHT, padx=2)
            self.items[name] = {"proc": None, "log": os.path.join(REPO_ROOT, item["log"]),
                                "pos": 0, "lamp": lamp, "port": port}

        # ---- 大按钮 ----
        btn_frm = tk.Frame(root)
        btn_frm.pack(pady=10)
        self.btn_start = tk.Button(btn_frm, text="▶ 启动", width=14, height=2,
                                   bg="#2e7d32", fg="white", command=self.on_start)
        self.btn_start.pack(side=tk.LEFT, padx=12)
        self.btn_stop = tk.Button(btn_frm, text="■ 停止", width=14, height=2,
                                  bg="#c62828", fg="white", command=self.on_stop,
                                  state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=12)
        tk.Button(btn_frm, text="打开看板", width=10, height=2,
                  command=lambda: webbrowser.open(
                      next((it["open_url"] for it in C.LAUNCH_ITEMS
                            if it.get("port") == 8000 and it.get("open_url")),
                           "http://localhost:8000/board"))).pack(side=tk.LEFT, padx=6)

        # ---- 日志面板 ----
        tk.Label(root, text="实时日志（启停台 + 采集程序 + 看板）:").pack(anchor="w", padx=14)
        self.txt = scrolledtext.ScrolledText(root, height=16, state=tk.DISABLED, font=("Consolas", 9))
        self.txt.pack(fill=tk.BOTH, expand=True, padx=14, pady=4)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        # 启动日志尾巴轮询
        self.root.after(500, self._tail_logs)

    # ---------- 日志 ----------
    def _append(self, text):
        self.txt.configure(state=tk.NORMAL)
        self.txt.insert(tk.END, text)
        self.txt.see(tk.END)
        self.txt.configure(state=tk.DISABLED)

    def log(self, msg):
        self._append("[启停台] " + msg + "\n")

    def _tail_logs(self):
        """实时尾巴读取各子进程日志文件并显示（不必开 Terminal 也能看运行过程）。"""
        for name, info in self.items.items():
            p = info["proc"]
            if p is None or p.poll() is not None:
                continue
            try:
                with open(info["log"], "r", encoding="utf-8", errors="replace") as f:
                    f.seek(info["pos"])
                    chunk = f.read()
                    if chunk:
                        info["pos"] = f.tell()
                        self._append("── %s ──\n%s" % (name, chunk))
            except Exception:
                pass
        self.root.after(500, self._tail_logs)

    # ---------- 解释器（单一来源，不弹窗） ----------
    def _resolve_python(self):
        """派生子进程用的解释器：优先与 GUI 同目录的 python.exe（保证同一环境），
        其次仓库 venv，最后 sys.executable。全程不弹窗、不打断现场。"""
        exe = sys.executable
        d = os.path.dirname(exe)
        cand = os.path.join(d, "python.exe")
        if os.path.exists(cand):
            return cand
        return C.find_python() or exe

    # ---------- 启停 ----------
    def on_start(self):
        py = self._resolve_python()
        self.log("[信息] 使用解释器: %s" % py)
        # 防呆1：残留子进程先停
        if any(info["proc"] and info["proc"].poll() is None for info in self.items.values()):
            self.log("[防呆] 本启停台仍有未停止的子进程, 先停止...")
            self.on_stop()
        # 防呆2：清理占用相机的旧 web_capture
        self._cleanup_strays()

        started = 0
        for item in C.LAUNCH_ITEMS:
            try:
                p, log_path = PM.start_item(py, item, REPO_ROOT)
                name = item["name"]
                self.items[name]["proc"] = p
                self.items[name]["log"] = log_path
                self.items[name]["pos"] = 0
                self._set_comp_running(name, True)
                self.log("[启动] %s (PID=%d)  日志: %s" % (name, p.pid, log_path))
                started += 1
                if item.get("open_url"):
                    webbrowser.open(item["open_url"])
            except Exception as e:
                self.log("[错误] 启动 %s 失败: %s" % (item["name"], e))
        if started:
            self._set_running()

    def on_stop(self):
        for name, info in self.items.items():
            p = info["proc"]
            if p is None:
                continue
            ok = PM.stop_item(p)
            self.log("[%s] %s (PID=%d)" % ("已停止" if ok else "强杀兜底", name, p.pid))
            info["proc"] = None
            self._set_comp_running(name, False)
        self._set_stopped()

    def _cleanup_strays(self):
        my_pid = os.getpid()
        exclude = {my_pid}
        for info in self.items.values():
            p = info["proc"]
            if p and p.pid:
                exclude.add(p.pid)
        strays = PM.find_stray_web_capture(exclude)
        if not strays:
            return
        self.log("[防呆] 检测到相机已被其他进程占用: PID=%s, 先优雅停止释放相机..." % strays)
        ok = PM.stop_pids(strays)
        self.log("[防呆] 旧进程已释放相机" if ok else
                 "[防呆] 旧进程未能完全停止, 若仍报'被另一程序控制'请到任务管理器结束 python.exe")

    def _set_comp_running(self, name, running):
        info = self.items.get(name)
        if not info:
            return
        info["lamp"].configure(fg="#2e7d32" if running else "gray",
                               text="●" if running else "○")

    def _set_running(self):
        self.status_var.set("运行中")
        self.lamp.configure(fg="#2e7d32")
        self.btn_start.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)

    def _set_stopped(self):
        self.status_var.set("已停止")
        self.lamp.configure(fg="gray")
        self.btn_start.configure(state=tk.NORMAL)
        self.btn_stop.configure(state=tk.DISABLED)

    def open_log(self, rel):
        path = os.path.join(REPO_ROOT, rel)
        if not os.path.exists(path):
            messagebox.showinfo("提示", "日志文件尚不存在:\n%s" % path)
            return
        try:
            os.startfile(path)
        except Exception as e:
            messagebox.showerror("打开失败", str(e))

    def on_close(self):
        running = [n for n, i in self.items.items() if i["proc"] and i["proc"].poll() is None]
        if running:
            if not messagebox.askyesno("确认", "以下程序仍在运行: %s\n关闭窗口会先停止它们吗?" % "、".join(running)):
                return
            self.on_stop()
        self.root.destroy()


def _hide_own_console():
    """若本 GUI 是被控制台解释器(python.exe/py.exe)拉起的, 启动时隐藏自身控制台窗口,
    保证无论怎么启动 GUI 都无黑框。用 pythonw 启动时无控制台, GetConsoleWindow 返回 0 自动跳过。"""
    if sys.platform != "win32":
        return
    try:
        k = ctypes.windll.kernel32
        hwnd = k.GetConsoleWindow()
        if hwnd:
            k.ShowWindow(hwnd, 0)  # SW_HIDE = 0
    except Exception:
        pass


def main():
    _hide_own_console()
    root = tk.Tk()
    LauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
